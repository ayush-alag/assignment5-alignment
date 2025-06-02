from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from typing import List
from transformers import PreTrainedTokenizer, PreTrainedModel

QWEN_BASE_PATH = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"

def load_model_and_tokenizer(model_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer

def forward_pass(model, tokenizer, prompt: str):
    input_ids = train_batch["input_ids"].to(device)
    labels = train_batch["labels"].to(device)
    logits = model(input_ids).logits
    loss = F.cross_entropy(logits, labels)

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

def main():
    model, tokenizer = load_model_and_tokenizer(QWEN_BASE_PATH)
    print(model)
    print(tokenizer)

if __name__ == "__main__":
    main()