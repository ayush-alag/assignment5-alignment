import argparse
import random
import torch

from vllm import SamplingParams
from cs336_alignment.sft import setup_wandb, load_model_and_tokenizer, load_training_data, \
                                init_vllm, load_policy_into_vllm_instance, sft_training_loop, \
                                load_and_format_prompts
from cs336_alignment.baseline import run_vllm

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def training_loop(n_ei_steps, model, tokenizer, vllm_model, sampling_params, batch_size_per_ei_step,
                  train_prompts, train_answers, train_ground_truths, reward_fn,
                  optimizer, gradient_accumulation_steps, microbatch_size, device,
                  eval_prompts, eval_answers, eval_steps, eval_sampling_params, output_dir):

    for i in range(n_ei_steps):
        print(f"Expert iteration {i}")
        batch_indices = random.sample(range(len(train_prompts)), batch_size_per_ei_step)
        batch_prompts = [train_prompts[i] for i in batch_indices]
        batch_answers = [train_answers[i] for i in batch_indices]
        batch_ground_truths = [train_ground_truths[i] for i in batch_indices]

        # TODO: do we need to load from one model to the vllm model?
        # I think we do this internally in sft_training_loop
        responses = run_vllm(vllm_model, batch_prompts, sampling_params)

        sft_prompts = []
        sft_answers = []
        sft_ground_truths = []
        for response, ground_truth in zip(responses, batch_ground_truths):
            reward = reward_fn(response, ground_truth)
            if reward["answer_reward"] == 1:
                sft_prompts.append(batch_prompts[i])
                sft_answers.append(batch_answers[i])
                sft_ground_truths.append(ground_truth)

        print(f"Expert iteration {i} has {len(sft_prompts)} filtered samples")

        sft_training_loop(model, tokenizer, vllm_model, sft_prompts, sft_answers, sft_ground_truths, optimizer,
                          gradient_accumulation_steps, microbatch_size, device, eval_prompts,
                          eval_answers, eval_steps, eval_sampling_params, output_dir, epochs=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_ei_steps", type=int, default=5)
    parser.add_argument("--G", type=int, default=4)
    parser.add_argument("--sampling_min_tokens", type=int, default=4)
    parser.add_argument("--sampling_max_tokens", type=int, default=1024)
    parser.add_argument("--batch_size_per_ei_step", type=int, default=128)
    parser.add_argument("--experiment_name", type=str, default="expert_iteration")
    parser.add_argument("--model_path", type=str, default="/data/a5-alignment/models/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data_path", type=str, default="/data/a5-alignment/MATH/sft.jsonl")
    parser.add_argument("--eval_data_path", type=str, default="/data/a5-alignment/MATH/validation.jsonl")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--microbatch_size", type=int, default=2)
    parser.add_argument("--eval_steps", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="/data/c-aalag/rl/sft_expert")
    args = parser.parse_args()

    setup_wandb(args.experiment_name)

    # load model and tokenizer
    train_device = "cuda:0"
    eval_device = "cuda:1"
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    model.to(train_device)

    # optimize model with training data
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    train_prompts, train_answers, train_ground_truths = load_training_data(args.train_data_path, data_amount=-1)

    rollout_sampling_params = SamplingParams(
        temperature=0.75,
        top_p=0.9,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens,
        n=args.G,
        seed=42,
    )

    eval_sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )

    prompt_path = "prompts/r1_zero.prompt"
    eval_prompts, eval_answers = load_and_format_prompts(args.eval_data_path, prompt_path)

    # vllm for generating the rollouts
    llm = init_vllm(args.model_path, device=eval_device, seed=42)
    load_policy_into_vllm_instance(model, llm)

    training_loop(args.n_ei_steps, model, tokenizer, llm, rollout_sampling_params, args.batch_size_per_ei_step,
                  train_prompts, train_answers, train_ground_truths, r1_zero_reward_fn, optimizer,
                  args.gradient_accumulation_steps, args.microbatch_size, train_device, eval_prompts,
                  eval_answers, args.eval_steps, eval_sampling_params, args.output_dir)