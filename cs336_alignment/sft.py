from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from typing import List
from transformers import PreTrainedTokenizer, PreTrainedModel
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import LLM
import wandb
import argparse
import json
from unittest.mock import patch
from cs336_alignment.baseline import evaluate_vllm, r1_zero_reward_fn, load_and_format_prompts

QWEN_BASE_PATH = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def setup_wandb():
    # Setup wandb metrics
    wandb.init(project="cs336-alignment", name="sft")
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    # the x‑axis for evaluation
    # everything that starts with train/ is tied to train_step
    wandb.define_metric("train/*", step_metric="train_step")
    # everything that starts with eval/ is tied to eval_step
    wandb.define_metric("eval/*", step_metric="eval_step")

def load_model_and_tokenizer(model_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer

def save_model_tokenizer(model, tokenizer, output_dir: str):
    model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)

def tokenize_prompt_and_output(prompt_strs: List[str], output_strs: List[str], tokenizer: PreTrainedTokenizer):
    tokenized_prompts = tokenizer(prompt_strs, padding=False, add_special_tokens=False)["input_ids"]
    tokenized_outputs = tokenizer(output_strs, padding=False, add_special_tokens=False)["input_ids"]

    concat_input_ids = []
    # range of prompt to output (for the labels, so we subtract 1)
    response_starts = []
    response_ends = []
    for tokenized_prompt, tokenized_output in zip(tokenized_prompts, tokenized_outputs):
        concat_input_ids.append(tokenized_prompt + tokenized_output)
        response_start = len(tokenized_prompt) - 1
        response_starts.append(response_start)
        response_ends.append(response_start + len(tokenized_output) - 1)

    max_len = max(len(input_ids) for input_ids in concat_input_ids)
    for i in range(len(concat_input_ids)):
        concat_input_ids[i] = concat_input_ids[i] + [tokenizer.pad_token_id] * (max_len - len(concat_input_ids[i]))

    concat_input_ids = torch.tensor(concat_input_ids)
    input_ids = concat_input_ids[:, :-1]
    labels = concat_input_ids[:, 1:]

    response_starts = torch.tensor(response_starts).unsqueeze(1)
    response_ends = torch.tensor(response_ends).unsqueeze(1)
    positions = torch.arange(max_len - 1).unsqueeze(0)
    response_mask = (positions >= response_starts) & (positions <= response_ends)

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}

# compute the per-token entropy of the logits
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_z = torch.logsumexp(logits, dim=-1)
    probs = torch.exp(logits - log_z.unsqueeze(-1))
    entropy = log_z - torch.sum(probs * logits, dim=-1)
    return entropy

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    output_logits = model(input_ids).logits
    log_probs = F.log_softmax(output_logits, dim=-1)
    gathered_log_probs = torch.gather(log_probs, index=labels.unsqueeze(-1), dim=-1).squeeze(-1)

    result_dict = {"log_probs": gathered_log_probs}

    # compute the log_probs
    if return_token_entropy:
        result_dict["token_entropy"] = compute_entropy(output_logits)

    return result_dict

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None= None,
) -> torch.Tensor:
    return torch.sum(torch.where(mask, tensor, torch.zeros_like(tensor)), dim=dim) / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    normalized_loss = masked_normalize(-policy_log_probs, response_mask, normalize_constant, dim=-1)
    adjusted_normalized_loss = normalized_loss.mean() / gradient_accumulation_steps
    adjusted_normalized_loss.backward()

    return adjusted_normalized_loss, {"normalized_loss": normalized_loss}

def eval_validation_set(
    model: PreTrainedModel,
    llm: LLM,
    tokenizer: PreTrainedTokenizer,
    eval_prompts: List[str],
    eval_answers: List[str],
) -> None:
    load_policy_into_vllm_instance(model, llm)

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )

    return evaluate_vllm(llm, r1_zero_reward_fn, eval_prompts, eval_answers, sampling_params)

def log_generations(generated_info_dicts: List[dict[str, float]]):
    entropies = []
    lengths = []
    num_correct = 0
    total_reward = 0

    # randomly choose 10 generations
    for info_dict in random.sample(generated_info_dicts, 10):
        if info_dict["correct"]:
            num_correct += 1
        total_reward += info_dict["reward"]
        entropies.append(info_dict["token_entropy"])
        lengths.append(len(info_dict["response"]))

    wandb.log({"eval/correct": num_correct / len(generated_info_dicts)})
    wandb.log({"eval/entropy": sum(entropies) / len(entropies)})
    wandb.log({"eval/length": sum(lengths) / len(lengths)})
    wandb.log({"eval/reward": total_reward / len(generated_info_dicts)})

# before this: set up wandb, model, tokenizer, slice data
# TODO: periodically evaluate on the validation set using parallelized VLLM
def training_loop(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    llm: LLM,
    prompts: List[str],
    answers: List[str],
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    microbatch_size: int,
    device: str,
    eval_device: str,
    eval_prompts: List[str],
    eval_answers: List[str],
    eval_steps: int,
    max_grad_norm: float | None = 1.0
) -> None:
    for i in range(0, len(prompts), microbatch_size):
        microbatch_prompts = prompts[i:i+microbatch_size]
        microbatch_answers = answers[i:i+microbatch_size]

        # tokenize the data
        tokenize_result = tokenize_prompt_and_output(microbatch_prompts, microbatch_answers, tokenizer)
        input_ids = tokenize_result["input_ids"].to(device)
        labels = tokenize_result["labels"].to(device)
        response_mask = tokenize_result["response_mask"].to(device)

        # model response
        model_output_dict = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
        policy_log_probs = model_output_dict["log_probs"].to(device)
        token_entropy = model_output_dict["token_entropy"].to(device)

        # loss and train step
        loss, loss_dict = sft_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps)
        if (i + 1) % gradient_accumulation_steps == 0:
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        if i % eval_steps == 0:
            eval_results = eval_validation_set(model, llm, tokenizer, eval_prompts, eval_answers)
            log_generations(eval_results)
            wandb.log({"eval/loss": eval_results["loss"]})

        # log the loss
        wandb.log({"train/loss": loss_dict["normalized_loss"].item()})

    # eval vllm
    eval_results = eval_validation_set(model, llm, tokenizer, eval_prompts, eval_answers)
    wandb.log({"eval/loss": eval_results["loss"]})

# TODO: log generations
def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    answers: List[str],
) -> None:

    pass

def load_training_data(data_path: str, data_amount: int) -> tuple[List[str], List[str]]:
    prompts = []
    answers = []
    with open(data_path, "r") as data_file:
        for line in data_file:
            data = json.loads(line)
            prompts.append(data["prompt"])
            answers.append(data["ground_truth"])

    return prompts[:data_amount] if data_amount else prompts, answers[:data_amount] if data_amount else answers

# TODO: model saving inside training loop?
def main(train_data_path: str, eval_data_path: str, model_path: str, output_dir: str, microbatch_size: int, gradient_accumulation_steps: int, data_amount: int, eval_steps: int, max_grad_norm: float = 1.0):
    setup_wandb()

    # load model and tokenizer
    train_device = "cuda:0"
    eval_device = "cuda:1"
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.to(train_device)

    # optimize model with training data
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    prompts, answers = load_training_data(train_data_path, data_amount)

    # eval vllm
    prompt_path = "prompts/r1_zero.prompt"
    eval_prompts, eval_answers = load_and_format_prompts(eval_data_path, prompt_path)
    llm = init_vllm(model_path, device=eval_device, seed=42)
    load_policy_into_vllm_instance(model, llm)
    training_loop(model, tokenizer, llm, prompts, answers, optimizer, gradient_accumulation_steps,
                  microbatch_size, train_device, eval_device, eval_prompts, eval_answers,
                  eval_steps, max_grad_norm)

    # save model and tokenizer
    save_model_tokenizer(model, tokenizer, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--microbatch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--data_amount", type=int, default=128, choices=[128, 256, 512, 1024, None])
    parser.add_argument("--output_dir", type=str, default="/data/c-aalag/rl/sft")
    parser.add_argument("--eval_steps", type=int, default=32)
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for gradient clipping")
    args = parser.parse_args()

    main(
        train_data_path="/data/a5-alignment/MATH/sft.jsonl",
        eval_data_path="/data/a5-alignment/MATH/validation.jsonl",
        model_path=QWEN_BASE_PATH,
        output_dir=args.output_dir,
        microbatch_size=args.microbatch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        data_amount=args.data_amount,
        eval_steps=args.eval_steps,
        max_grad_norm=args.max_grad_norm
    )