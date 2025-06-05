from collections import defaultdict
import argparse
import numpy as np
import statistics
import torch
from typing import Literal
from cs336_alignment.sft import get_response_log_probs, setup_wandb, load_model_and_tokenizer, load_training_data, \
                                load_and_format_prompts
from vllm import SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    metadata = defaultdict(list)
    raw_rewards = []
    advantages = []

    rollout_batch_size = len(rollout_responses)
    for i in range(0, rollout_batch_size, group_size):
        group_responses = rollout_responses[i : i + group_size]
        group_ground_truths = repeated_ground_truths[i : i + group_size]

        group_rewards = []
        for response, ground_truth in zip(group_responses, group_ground_truths):
            reward = reward_fn(response, ground_truth)
            group_rewards.append(reward["reward"])
            raw_rewards.append(reward["reward"])

        avg_reward = sum(group_rewards) / group_size
        std_reward = statistics.stdev(group_rewards)
        metadata["avg_reward"].append(avg_reward)
        metadata["std_reward"].append(std_reward)

        raw_adv_rewards = []
        for reward in group_rewards:
            advantage = reward - avg_reward
            raw_adv_rewards.append(advantage)
            if normalize_by_std:
                advantage /= (std_reward + advantage_eps)

            advantages.append(advantage)

    return advantages, raw_rewards, metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    metadata = defaultdict(list)

    importance_ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_importance_ratio = torch.clamp(importance_ratio, 1 - cliprange, 1 + cliprange)
    raw_policy_weight = advantages * importance_ratio
    clipped_policy_weight = advantages * clipped_importance_ratio

    metadata["clipped"].append(clipped_policy_weight < raw_policy_weight)

    return -torch.min(raw_policy_weight, clipped_policy_weight), metadata

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        print(raw_rewards.shape, policy_log_probs.shape)
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    else:
        assert loss_type == "grpo_clip"
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None= None,
) -> torch.Tensor:
    return torch.sum(torch.where(mask, tensor, torch.zeros_like(tensor)), dim=dim) / mask.sum(dim=dim)

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    policy_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange
    )

    # mean over all dimensions wherever we have a response token
    response_policy_loss = masked_mean(policy_loss, response_mask)
    scaled_response_policy_loss = response_policy_loss / gradient_accumulation_steps
    scaled_response_policy_loss.backward()

    return scaled_response_policy_loss, metadata

# TODO: implement this
def generate_rollout_responses(
    policy,
    rollout_prompts,
    sampling_temperature,
    sampling_min_tokens,
    sampling_max_tokens,
):
    rollout_responses = []
    repeated_ground_truths = []
    pass
    return rollout_responses, repeated_ground_truths

def get_rollout_prompts(n_prompts_per_batch: int):
    pass

def grpo_train_loop(n_grpo_steps, learning_rate, advantage_eps, rollout_batch_size,
                    group_size, sampling_temperature, sampling_min_tokens, sampling_max_tokens,
                    epochs_per_rollout_batch, train_batch_size, gradient_accumulation_steps,
                    gpu_memory_utilization, loss_type, use_std_normalization, n_prompts_per_batch,
                    reward_fn, model, tokenizer, vllm_model, rollout_sampling_params, train_prompts,
                    train_answers, train_ground_truths, optimizer, output_dir):
    for i in range(n_grpo_steps):
        # TODO: get a batch of questions from dataset
        rollout_prompts = get_rollout_prompts(n_prompts_per_batch)
        old_policy = policy
        # TODO: generate G outputs per question
        rollout_responses, repeated_ground_truths = generate_rollout_responses(policy, rollout_prompts, sampling_temperature, sampling_min_tokens, sampling_max_tokens)
        advantages, raw_rewards, metadata = compute_group_normalized_rewards(
            reward_fn, rollout_responses, repeated_ground_truths, group_size, advantage_eps, use_std_normalization
        )

        policy_log_probs = get_response_log_probs(policy, rollout_responses)

        grpo_microbatch_train_step(
            policy_log_probs, response_mask, gradient_accumulation_steps, loss_type, raw_rewards, advantages, old_log_probs, cliprange
        )

        if i % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

if __name__ == "__main__":
    # add relevant args
    parser = argparse.ArgumentParser()
    # hyperparameters
    parser.add_argument("--n_grpo_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument("--rollout_batch_size", type=int, default=256)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--sampling_min_tokens", type=int, default=4)
    parser.add_argument("--sampling_max_tokens", type=int, default=1024)
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--loss_type", type=str, default="reinforce_with_baseline")
    parser.add_argument("--use_std_normalization", type=bool, default=True)
    parser.add_argument("--n_prompts_per_batch", type=int, default=32)

    # experiment args
    parser.add_argument("--experiment_name", type=str, default="grpo")
    parser.add_argument("--model_path", type=str, default="/data/a5-alignment/models/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data_path", type=str, default="/data/a5-alignment/MATH/train.jsonl")
    parser.add_argument("--data_amount", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default="/data/c-aalag/rl/grpo")
    args = parser.parse_args()

    setup_wandb(args.experiment_name)

    # load model and tokenizer
    train_device = "cuda:0"
    eval_device = "cuda:1"
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    model.to(train_device)

    # optimize model with training data
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    train_prompts, train_answers, train_ground_truths = load_training_data(args.train_data_path, args.data_amount)

    eval_sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )

    # eval vllm
    prompt_path = "prompts/r1_zero.prompt"
    eval_prompts, eval_answers = load_and_format_prompts(args.train_data_path, prompt_path)

    # TODO: load the policy/model
    policy = None
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=0.0, betas=(0.9, 0.95))

    grpo_train_loop(args.n_grpo_steps, args.learning_rate, args.advantage_eps,
                    args.rollout_batch_size, args.group_size, args.sampling_temperature,
                    args.sampling_min_tokens, args.sampling_max_tokens, args.epochs_per_rollout_batch,
                    args.train_batch_size, args.gradient_accumulation_steps, args.gpu_memory_utilization,
                    args.loss_type, args.use_std_normalization, args.n_prompts_per_batch, r1_zero_reward_fn,
                    model, tokenizer, vllm_model, rollout_sampling_params, train_prompts, train_answers, train_ground_truths,
                    optimizer, args.output_dir)