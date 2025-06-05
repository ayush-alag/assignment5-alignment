from collections import defaultdict
import numpy as np
import statistics
import torch
from typing import Literal
from cs336_alignment.sft import get_response_log_probs
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

def grpo_train_loop():
    # define hyperparameters
    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4 # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = 1024
    epochs_per_rollout_batch: int = 1 # On-policy
    train_batch_size: int = 256 # On-policy
    gradient_accumulation_steps: int = 128 # microbatch size is 2, will fit on H100
    gpu_memory_utilization: float = 0.85
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
    ] = "reinforce_with_baseline"
    use_std_normalization: bool = True

    n_prompts_per_batch: int = 32 # 256 / 8 = 32
    reward_fn = r1_zero_reward_fn

    # TODO: load the policy/model
    policy = None
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95))

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