# -*- coding: utf-8 -*-
"""教学版 shared PPO agent。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from algorithms.ppo.buffer import PPOBatch
from algorithms.ppo.networks import ActorCritic


@dataclass
class PPOUpdateInfo:
    policy_loss: float
    value_loss: float
    entropy: float


class PPOAgent:
    """三个 MPE agent 共享的一套 PPO 策略。"""

    def __init__(
        self,
        obs_dim: int = 18,
        action_dim: int = 5,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """给多个 agent 的 observation 同时采样动作。"""

        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        actions, log_probs, _, values = self.model.get_action_and_value(obs_tensor)

        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
        )

    @torch.no_grad()
    def value(self, obs_batch: np.ndarray) -> np.ndarray:
        """只计算 value，用于 rollout 末尾 bootstrap。"""

        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        _, values = self.model(obs_tensor)
        return values.cpu().numpy()

    def update(
        self,
        batch: PPOBatch,
        ppo_epochs: int,
        minibatch_size: int,
    ) -> PPOUpdateInfo:
        """执行 PPO clipped objective 更新。"""

        advantages = batch.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = batch.obs.shape[0]
        policy_losses = []
        value_losses = []
        entropies = []

        for _ in range(ppo_epochs):
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]

                _, new_log_probs, entropy, new_values = self.model.get_action_and_value(
                    batch.obs[mb_idx],
                    batch.actions[mb_idx],
                )

                # ratio = pi_new(a|s) / pi_old(a|s)，old_log_prob 必须来自采样当时的策略。
                log_ratio = new_log_probs - batch.old_log_probs[mb_idx]
                ratio = log_ratio.exp()

                mb_advantages = advantages[mb_idx]
                unclipped = ratio * mb_advantages
                clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                clipped = clipped * mb_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                # critic 学习拟合 GAE 得到的 return。
                value_loss = nn.functional.mse_loss(new_values, batch.returns[mb_idx])

                entropy_loss = entropy.mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_loss.item())

        return PPOUpdateInfo(
            policy_loss=float(np.mean(policy_losses)),
            value_loss=float(np.mean(value_losses)),
            entropy=float(np.mean(entropies)),
        )

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
