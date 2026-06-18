# -*- coding: utf-8 -*-
"""Recurrent individual Q network for QMIX."""

from __future__ import annotations

import torch
from torch import nn


class RNNAgent(nn.Module):
    """Observation encoder + GRUCell + action-value head."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        input_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_dim = int(input_dim) if input_dim is not None else self.obs_dim

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.fc_q = nn.Linear(self.hidden_dim, self.action_dim)
        self.activation = nn.ReLU()

    def init_hidden(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        """Return zero hidden state with shape [batch_size, hidden_dim]."""

        return torch.zeros(int(batch_size), self.hidden_dim, device=device)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return q_values [batch, action_dim] and next_hidden [batch, hidden_dim]."""

        if obs.dim() != 2:
            raise RuntimeError(f"RNNAgent expects obs [batch, obs_dim], got {tuple(obs.shape)}")
        if hidden.dim() != 2:
            raise RuntimeError(
                f"RNNAgent expects hidden [batch, hidden_dim], got {tuple(hidden.shape)}"
            )
        if obs.shape[0] != hidden.shape[0]:
            raise RuntimeError(
                f"obs batch {obs.shape[0]} must match hidden batch {hidden.shape[0]}"
            )
        if obs.shape[1] != self.input_dim:
            raise RuntimeError(f"input_dim mismatch: expected {self.input_dim}, got {obs.shape[1]}")
        if hidden.shape[1] != self.hidden_dim:
            raise RuntimeError(
                f"hidden_dim mismatch: expected {self.hidden_dim}, got {hidden.shape[1]}"
            )

        x = self.activation(self.fc1(obs))
        next_hidden = self.rnn(x, hidden)
        q_values = self.fc_q(next_hidden)
        return q_values, next_hidden
