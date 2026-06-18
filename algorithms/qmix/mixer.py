# -*- coding: utf-8 -*-
"""QMIX monotonic mixing network."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class QMixer(nn.Module):
    """Mix per-agent Q values into Q_tot with state-conditioned weights."""

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        hidden_dim: int = 32,
        hypernet_hidden_dim: int = 64,
        use_layer_norm: bool = True,
        mixer_weight_clip: float = 1.0,
        mixer_bias_clip: float = 5.0,
        mixer_weight_activation: str = "softplus",
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.use_layer_norm = bool(use_layer_norm)
        self.mixer_weight_clip = float(mixer_weight_clip)
        self.mixer_bias_clip = float(mixer_bias_clip)
        self.mixer_weight_activation = mixer_weight_activation

        if mixer_weight_activation not in ("softplus", "abs"):
            raise ValueError(
                "mixer_weight_activation must be 'softplus' or 'abs', "
                f"got {mixer_weight_activation}"
            )

        self.state_norm = nn.LayerNorm(state_dim) if self.use_layer_norm else nn.Identity()
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, n_agents * hidden_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_w_final = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, hidden_dim),
        )
        self.v = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _positive_weight(self, raw_weight: torch.Tensor) -> torch.Tensor:
        if self.mixer_weight_activation == "softplus":
            weight = F.softplus(raw_weight)
        else:
            weight = torch.abs(raw_weight)
        return weight.clamp(min=0.0, max=self.mixer_weight_clip)

    def _bounded_bias(self, raw_bias: torch.Tensor) -> torch.Tensor:
        return raw_bias.clamp(min=-self.mixer_bias_clip, max=self.mixer_bias_clip)

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Return q_tot with shape [batch, 1]."""

        batch_size = agent_qs.shape[0]
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)
        states = self.state_norm(states)

        w1 = self._positive_weight(self.hyper_w1(states)).view(
            batch_size,
            self.n_agents,
            self.hidden_dim,
        )
        b1 = self._bounded_bias(self.hyper_b1(states)).view(batch_size, 1, self.hidden_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        w_final = self._positive_weight(self.hyper_w_final(states)).view(
            batch_size,
            self.hidden_dim,
            1,
        )
        v = self._bounded_bias(self.v(states)).view(batch_size, 1, 1)
        q_tot = torch.bmm(hidden, w_final) + v
        return q_tot.view(batch_size, 1)


class VDNMixer(nn.Module):
    """Value decomposition mixer: Q_tot is the sum of per-agent Q values."""

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor | None = None) -> torch.Tensor:
        del states
        if agent_qs.dim() != 2:
            raise RuntimeError(f"VDNMixer expects agent_qs [batch, n_agents], got {tuple(agent_qs.shape)}")
        return agent_qs.sum(dim=1, keepdim=True)
