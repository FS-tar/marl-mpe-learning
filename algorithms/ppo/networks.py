# -*- coding: utf-8 -*-
"""PPO 使用的共享 Actor-Critic 网络。

本项目先做教学版 PPO：三个 agent 共用同一个网络，只是每个 agent
在每一步输入自己的 18 维 observation。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """离散动作 PPO 的 Actor-Critic。

    actor 输出 action logits，critic 输出当前 observation 的 value。
    simple_spread_v3 默认 observation 是 18 维，动作空间是 Discrete(5)。
    """

    def __init__(self, obs_dim: int = 18, action_dim: int = 5, hidden_dim: int = 128):
        super().__init__()

        # 共享特征层让 actor 和 critic 都先提取 observation 表示。
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # actor head 输出每个离散动作的 logits，后续交给 Categorical 分布。
        self.actor = nn.Linear(hidden_dim, action_dim)

        # critic head 输出一个标量 value，表示当前 observation 的状态价值估计。
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        logits = self.actor(features)
        values = self.critic(features).squeeze(-1)
        return logits, values

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样动作，或重新计算给定动作的 log_prob/value。

        PPO 更新时需要用新策略重新计算旧动作的 log_prob，所以这里同时
        支持 action=None 的采样模式和 action!=None 的评估模式。
        """

        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, values
