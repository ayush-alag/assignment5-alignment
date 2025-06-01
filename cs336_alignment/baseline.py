from vllm import LLM, SamplingParams
from typing import Callable, List, Tuple
from drgrpo_grader import r1_zero_reward_fn
from prompts import r1_zero_prompt
import json
import os

QWEN_BASE_PATH = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
# LLAMA_8B_PATH = "/data/a5-alignment/models/Llama-3.1-8B"
# LLAMA_70B_PATH = "/data/a5-alignment/models/Llama-3.3-70B-Instruct"

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    reward_dicts = [reward_fn(output.outputs[0].text, output.prompt) for output in outputs]
    return reward_dicts

def load_and_format_prompts(data_path: str) -> List[str]:
    with open(data_path, "r") as f:
        data = json.load(f)
    return [r1_zero_prompt.format(question=item["question"]) for item in data]

def build_llm_and_params(model_path: str) -> Tuple[LLM, SamplingParams]:
    llm = LLM(model=model_path)
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )

    return llm, sampling_params

def serialize_results(reward_dicts: List[dict[str, float]], output_path: str):
    with open(output_path, "w") as f:
        json.dump(reward_dicts, f)

def main(data_path: str, model_path: str, output_path: str):
    prompts = load_and_format_prompts(data_path)
    llm, sampling_params = build_llm_and_params(model_path)
    reward_dicts = evaluate_vllm(llm, r1_zero_reward_fn, prompts, sampling_params)
    serialize_results(reward_dicts, output_path)

if __name__ == "__main__":
    data_path = "/data/a5-alignment/MATH/validation.jsonl"
    model_path = QWEN_BASE_PATH

    output_dir = "/data/c-aalag/rl/baseline"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_path.split('/')[-1]}_r1_zero.jsonl")
    main(data_path, model_path, output_path)