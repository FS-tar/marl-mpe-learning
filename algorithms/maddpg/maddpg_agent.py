# -*- coding: utf-8 -*-
"""单个 MADDPG agent 的网络与动作选择封装。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
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


class MADDPGAgent:
    """每个环境 agent 对应一个 actor、critic 以及各自的 target network。"""

    def __init__(self, config: MADDPGAgentConfig):
        self.name = config.name
        self.obs_dim = config.obs_dim
        self.action_dim = config.action_dim
        self.device = config.device

        self.actor = DiscreteActor(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        ).to(config.device)
        self.target_actor = DiscreteActor(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        ).to(config.device)

        self.critic = CentralizedCritic(
            global_obs_dim=config.global_obs_dim,
            global_action_dim=config.global_action_dim,
            hidden_dim=config.hidden_dim,
        ).to(config.device)
        self.target_critic = CentralizedCritic(
            global_obs_dim=config.global_obs_dim,
            global_action_dim=config.global_action_dim,
            hidden_dim=config.hidden_dim,
        ).to(config.device)

        hard_update(self.actor, self.target_actor)
        hard_update(self.critic, self.target_critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.critic_lr)

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        epsilon: float = 0.0,
        explore: bool = True,
    ) -> tuple[int, np.ndarray]:
        """选择离散动作，并返回对应 one-hot。

        训练时使用 epsilon-greedy：epsilon 概率随机探索，否则按 actor 的 categorical
        分布采样；评估时 explore=False，直接使用 argmax。
        """

        if explore and np.random.random() < epsilon:
            action = int(np.random.randint(self.action_dim))
            one_hot = np.eye(self.action_dim, dtype=np.float32)[action]
            return action, one_hot

        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        logits = self.actor(obs_tensor)

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
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )
