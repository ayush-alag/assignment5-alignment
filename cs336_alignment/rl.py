from collections import defaultdict
import numpy as np
import statistics
import torch
from typing import Literal

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