# -*- coding: utf-8 -*-
"""MADDPG 训练器。

本文件负责把 replay buffer 中的 per-agent 数据拼成 centralized critic 需要的
global_obs 和 global_actions，并按 agent 逐个更新 critic 和 actor。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.maddpg.maddpg_agent import MADDPGAgent, MADDPGAgentConfig
from algorithms.maddpg.networks import (
    gumbel_softmax_action,
    one_hot_from_logits,
    soft_update,
)
from algorithms.maddpg.replay_buffer import MADDPGBatch, MultiAgentReplayBuffer


@dataclass
class MADDPGUpdateInfo:
    mean_critic_loss: float
    mean_actor_loss: float
    critic_losses: dict[str, float]
    actor_losses: dict[str, float]


class MADDPGTrainer:
    """管理 4 个 MADDPG agent 的联合训练。"""

    def __init__(
        self,
        agent_names: list[str],
        obs_dims: dict[str, int],
        action_dim: int,
        hidden_dim: int,
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        tau: float,
        buffer_size: int,
        device: str | None = None,
    ):
        self.agent_names = list(agent_names)
        self.obs_dims = dict(obs_dims)
        self.action_dim = action_dim
        self.global_obs_dim = sum(obs_dims[agent] for agent in agent_names)
        self.global_action_dim = action_dim * len(agent_names)
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.agents = {
            name: MADDPGAgent(
                MADDPGAgentConfig(
                    name=name,
                    obs_dim=obs_dims[name],
                    action_dim=action_dim,
                    global_obs_dim=self.global_obs_dim,
                    global_action_dim=self.global_action_dim,
                    hidden_dim=hidden_dim,
                    actor_lr=actor_lr,
                    critic_lr=critic_lr,
                    device=self.device,
                )
            )
            for name in agent_names
        }
        self.replay_buffer = MultiAgentReplayBuffer(
            capacity=buffer_size,
            agent_names=agent_names,
            obs_dims=obs_dims,
            action_dim=action_dim,
            device=self.device,
        )

    def act(
        self,
        observations: dict[str, np.ndarray],
        epsilon: float,
        explore: bool,
    ) -> tuple[dict[str, int], dict[str, np.ndarray]]:
        actions = {}
        one_hot_actions = {}
        for name in self.agent_names:
            action, one_hot = self.agents[name].act(
                observations[name],
                epsilon=epsilon,
                explore=explore,
            )
            actions[name] = action
            one_hot_actions[name] = one_hot
        return actions, one_hot_actions

    def add_transition(
        self,
        obs: dict[str, np.ndarray],
        actions: dict[str, int],
        one_hot_actions: dict[str, np.ndarray],
        rewards: dict[str, float],
        next_obs: dict[str, np.ndarray],
        dones: dict[str, bool],
    ) -> None:
        self.replay_buffer.add(
            obs=obs,
            actions=actions,
            one_hot_actions=one_hot_actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
        )

    def update(self, batch_size: int) -> MADDPGUpdateInfo:
        batch = self.replay_buffer.sample(batch_size)
        global_obs = self._concat_obs(batch.obs)
        global_actions = self._concat_actions(batch.one_hot_actions)
        next_global_obs = self._concat_obs(batch.next_obs)

        critic_losses = {}
        actor_losses = {}

        for name in self.agent_names:
            agent = self.agents[name]

            # critic 更新：用 target actor 生成 next_global_actions，估计 TD target。
            with torch.no_grad():
                next_actions = {}
                for other_name in self.agent_names:
                    logits = self.agents[other_name].target_actor(
                        batch.next_obs[other_name]
                    )
                    next_actions[other_name] = one_hot_from_logits(logits)
                next_global_actions = self._concat_actions(next_actions)
                target_q = agent.target_critic(next_global_obs, next_global_actions)
                y = batch.rewards[name] + self.gamma * target_q * (1.0 - batch.dones[name])

            current_q = agent.critic(global_obs, global_actions)
            critic_loss = F.mse_loss(current_q, y)

            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), max_norm=1.0)
            agent.critic_optimizer.step()

            # actor 更新：其他 agent 的 action 固定为 replay 中动作；当前 agent 的 action
            # 换成当前 actor 输出的可微 hard one-hot，以最大化自己的 centralized Q。
            for param in agent.critic.parameters():
                param.requires_grad_(False)
            mixed_actions = {
                other_name: batch.one_hot_actions[other_name].detach()
                for other_name in self.agent_names
            }
            logits = agent.actor(batch.obs[name])
            mixed_actions[name] = gumbel_softmax_action(logits)
            mixed_global_actions = self._concat_actions(mixed_actions)
            actor_loss = -agent.critic(global_obs, mixed_global_actions).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=1.0)
            agent.actor_optimizer.step()
            for param in agent.critic.parameters():
                param.requires_grad_(True)

            soft_update(agent.actor, agent.target_actor, self.tau)
            soft_update(agent.critic, agent.target_critic, self.tau)

            critic_losses[name] = float(critic_loss.item())
            actor_losses[name] = float(actor_loss.item())

        return MADDPGUpdateInfo(
            mean_critic_loss=float(np.mean(list(critic_losses.values()))),
            mean_actor_loss=float(np.mean(list(actor_losses.values()))),
            critic_losses=critic_losses,
            actor_losses=actor_losses,
        )

    def save_checkpoints(self, checkpoint_root) -> None:
        for name, agent in self.agents.items():
            agent_dir = checkpoint_root / name
            agent_dir.mkdir(parents=True, exist_ok=True)
            agent.save(str(agent_dir / "latest.pt"))

    def _concat_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([obs[name] for name in self.agent_names], dim=-1)

    def _concat_actions(self, actions: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([actions[name] for name in self.agent_names], dim=-1)
