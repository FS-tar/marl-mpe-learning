# -*- coding: utf-8 -*-
"""单个 MADDPG agent 的网络与动作选择封装。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from algorithms.maddpg.networks import (
    CentralizedCritic,
    DiscreteActor,
    hard_update,
    one_hot_from_logits,
)


@dataclass
class MADDPGAgentConfig:
    name: str
    obs_dim: int
    action_dim: int
    global_obs_dim: int
    global_action_dim: int
    hidden_dim: int
    actor_lr: float
    critic_lr: float
    device: torch.device
    ensemble_size: int = 1


class MADDPGAgent:
    """每个环境 agent 对应一个 actor、critic 以及各自的 target network。"""

    def __init__(self, config: MADDPGAgentConfig):
        self.name = config.name
        self.obs_dim = config.obs_dim
        self.action_dim = config.action_dim
        self.device = config.device

        self.ensemble_size = max(1, int(config.ensemble_size))

        self.actors = nn.ModuleList(
            [
                DiscreteActor(
                    obs_dim=config.obs_dim,
                    action_dim=config.action_dim,
                    hidden_dim=config.hidden_dim,
                )
                for _ in range(self.ensemble_size)
            ]
        ).to(config.device)
        self.target_actors = nn.ModuleList(
            [
                DiscreteActor(
                    obs_dim=config.obs_dim,
                    action_dim=config.action_dim,
                    hidden_dim=config.hidden_dim,
                )
                for _ in range(self.ensemble_size)
            ]
        ).to(config.device)

        self.critics = nn.ModuleList(
            [
                CentralizedCritic(
                    global_obs_dim=config.global_obs_dim,
                    global_action_dim=config.global_action_dim,
                    hidden_dim=config.hidden_dim,
                )
                for _ in range(self.ensemble_size)
            ]
        ).to(config.device)
        self.target_critics = nn.ModuleList(
            [
                CentralizedCritic(
                    global_obs_dim=config.global_obs_dim,
                    global_action_dim=config.global_action_dim,
                    hidden_dim=config.hidden_dim,
                )
                for _ in range(self.ensemble_size)
            ]
        ).to(config.device)

        for policy_id in range(self.ensemble_size):
            hard_update(self.actors[policy_id], self.target_actors[policy_id])
            hard_update(self.critics[policy_id], self.target_critics[policy_id])

        self.actor_optimizers = [
            Adam(actor.parameters(), lr=config.actor_lr)
            for actor in self.actors
        ]
        self.critic_optimizers = [
            Adam(critic.parameters(), lr=config.critic_lr)
            for critic in self.critics
        ]

        # 兼容旧 joint MADDPG 代码路径：K=1 时这些属性仍指向第 0 个子策略。
        self.actor = self.actors[0]
        self.target_actor = self.target_actors[0]
        self.critic = self.critics[0]
        self.target_critic = self.target_critics[0]
        self.actor_optimizer = self.actor_optimizers[0]
        self.critic_optimizer = self.critic_optimizers[0]

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        epsilon: float = 0.0,
        explore: bool = True,
        policy_id: int = 0,
    ) -> tuple[int, np.ndarray]:
        """选择离散动作，并返回对应 one-hot。

        训练时使用 epsilon-greedy：epsilon 概率随机探索，否则按 actor 的 categorical
        分布采样；评估时 explore=False，直接使用 argmax。
        """

        policy_id = self._normalize_policy_id(policy_id)
        if explore and np.random.random() < epsilon:
            action = int(np.random.randint(self.action_dim))
            one_hot = np.eye(self.action_dim, dtype=np.float32)[action]
            return action, one_hot

        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        logits = self.actors[policy_id](obs_tensor)

        if explore:
            probs = F.softmax(logits, dim=-1)
            action_tensor = torch.distributions.Categorical(probs=probs).sample()
            action = int(action_tensor.item())
            one_hot = F.one_hot(action_tensor, num_classes=self.action_dim).float()
        else:
            one_hot = one_hot_from_logits(logits)
            action = int(torch.argmax(one_hot, dim=-1).item())

        return action, one_hot.squeeze(0).cpu().numpy()

    def save(self, path: str) -> None:
        torch.save(
            {
                "ensemble_size": self.ensemble_size,
                "actors": [actor.state_dict() for actor in self.actors],
                "critics": [critic.state_dict() for critic in self.critics],
                "target_actors": [
                    target_actor.state_dict()
                    for target_actor in self.target_actors
                ],
                "target_critics": [
                    target_critic.state_dict()
                    for target_critic in self.target_critics
                ],
                "actor_optimizers": [
                    optimizer.state_dict()
                    for optimizer in self.actor_optimizers
                ],
                "critic_optimizers": [
                    optimizer.state_dict()
                    for optimizer in self.critic_optimizers
                ],
            },
            path,
        )

    def _normalize_policy_id(self, policy_id: int) -> int:
        if policy_id < 0 or policy_id >= self.ensemble_size:
            raise ValueError(
                f"{self.name} policy_id={policy_id} 超出范围 [0, {self.ensemble_size - 1}]"
            )
        return int(policy_id)
