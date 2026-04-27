from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import Dataset


class AgentRiskToyDataset(Dataset):
    """Toy action-selection dataset with explicit utility and risk factors."""

    def __init__(self, size: int, num_actions: int = 4, feature_dim: int = 8, seed: int = 0) -> None:
        if size <= 0 or num_actions < 2 or feature_dim < 4:
            raise ValueError("invalid agent-risk dataset configuration")
        self.size = size
        self.num_actions = num_actions
        self.feature_dim = feature_dim
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        utility = torch.rand(self.num_actions, generator=generator)
        cost = torch.rand(self.num_actions, generator=generator)
        risk = torch.rand(self.num_actions, generator=generator)
        uncertainty = torch.rand(self.num_actions, generator=generator)
        hidden_success = (utility + 0.2 * torch.randn(self.num_actions, generator=generator) > 0.55).float()
        hidden_unsafe = (risk + uncertainty + 0.1 * torch.randn(self.num_actions, generator=generator) > 1.15).float()
        reward = utility * hidden_success - 0.45 * cost - 1.25 * hidden_unsafe - 0.25 * risk
        target_action = reward.argmax().long()
        approval_target = (risk[target_action] + uncertainty[target_action] > 1.0).float()
        features = torch.stack([utility, cost, risk, uncertainty], dim=-1)
        if self.feature_dim > 4:
            noise = torch.randn(self.num_actions, self.feature_dim - 4, generator=generator) * 0.1
            features = torch.cat([features, noise], dim=-1)
        return {
            "features": features.float(),
            "target_action": target_action,
            "utility": utility.float(),
            "cost": cost.float(),
            "risk": risk.float(),
            "uncertainty": uncertainty.float(),
            "unsafe": hidden_unsafe.float(),
            "success": hidden_success.float(),
            "reward": reward.float(),
            "approval_target": approval_target,
            "risk_target": risk.float(),
        }


def agent_risk_collate(batch: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


__all__ = ["AgentRiskToyDataset", "agent_risk_collate"]
