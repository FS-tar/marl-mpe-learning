# -*- coding: utf-8 -*-
"""Individual Q network for simplified QMIX."""

from __future__ import annotations

import torch
from torch import nn


class AgentQNetwork(nn.Module):
    """MLP mapping one agent observation to discrete action Q values."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
