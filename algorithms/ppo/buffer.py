# -*- coding: utf-8 -*-
"""教学版 PPO rollout buffer。

这里按 [rollout_steps, num_agents, ...] 保存数据。虽然 PPO 更新时会把
这些样本展平成一个 batch，但先按时间和 agent 保存更容易理解 GAE。
"""

#收集数据，计算advantage和return


from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class PPOBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_values: torch.Tensor
#训练数据包

class RolloutBuffer:
    """保存一次 rollout，并计算 GAE advantage。"""

    def __init__(self, rollout_steps: int, num_agents: int, obs_dim: int, device: str):
        self.rollout_steps = rollout_steps
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.obs = np.zeros(
            (self.rollout_steps, self.num_agents, self.obs_dim),
            dtype=np.float32,
        )
        self.actions = np.zeros((self.rollout_steps, self.num_agents), dtype=np.int64)
        self.log_probs = np.zeros((self.rollout_steps, self.num_agents), dtype=np.float32)
        self.values = np.zeros((self.rollout_steps, self.num_agents), dtype=np.float32)
        self.rewards = np.zeros((self.rollout_steps, self.num_agents), dtype=np.float32)
        self.dones = np.zeros((self.rollout_steps, self.num_agents), dtype=np.float32)
    #创建空表

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """保存一个环境 step 中所有 agent 的数据。"""

        if self.step >= self.rollout_steps:
            raise RuntimeError("RolloutBuffer 已满，请先调用 reset()。")

        self.obs[self.step] = obs
        self.actions[self.step] = actions
        self.log_probs[self.step] = log_probs
        self.values[self.step] = values
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.step += 1

    #很重要！advantage：这个动作比 critic 原来预期的好多少 return：critic 后面要学习的目标值
    def compute_gae(
        self,
        last_values: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> PPOBatch:
        """用 GAE(lambda) 计算 advantage 和 return。

        done=1 表示该 agent 的轨迹在这一步结束，下一步 value 不再 bootstrap。
        """

        advantages = np.zeros_like(self.rewards, dtype=np.float32)
        last_gae = np.zeros(self.num_agents, dtype=np.float32)

        #从最后一步往前推
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values

        # PPO 更新时不再区分时间和 agent，把所有样本展平成 batch。
        flat_obs = self.obs.reshape(-1, self.obs_dim)
        flat_actions = self.actions.reshape(-1)
        flat_log_probs = self.log_probs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_values = self.values.reshape(-1)

        return PPOBatch(
            obs=torch.as_tensor(flat_obs, dtype=torch.float32, device=self.device),
            actions=torch.as_tensor(flat_actions, dtype=torch.long, device=self.device),
            old_log_probs=torch.as_tensor(flat_log_probs, dtype=torch.float32, device=self.device),
            advantages=torch.as_tensor(flat_advantages, dtype=torch.float32, device=self.device),
            returns=torch.as_tensor(flat_returns, dtype=torch.float32, device=self.device),
            old_values=torch.as_tensor(flat_values, dtype=torch.float32, device=self.device),
        )
