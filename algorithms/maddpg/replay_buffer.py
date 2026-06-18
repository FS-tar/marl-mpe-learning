# -*- coding: utf-8 -*-
"""MADDPG 回放池。

每条 transition 都保存所有 agent 在同一个环境 step 的 obs、action、one-hot action、
reward、next_obs、done。采样时仍按 agent 名称保留字段，便于 centralized critic
拼接全局输入。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class MADDPGBatch:
    obs: dict[str, torch.Tensor]
    actions: dict[str, torch.Tensor]
    one_hot_actions: dict[str, torch.Tensor]
    rewards: dict[str, torch.Tensor]
    next_obs: dict[str, torch.Tensor]
    dones: dict[str, torch.Tensor]
    adv_policy_ids: torch.Tensor
    prey_policy_ids: torch.Tensor


class MultiAgentReplayBuffer:
    """固定容量环形 replay buffer。"""

    def __init__(
        self,
        capacity: int,
        agent_names: list[str],
        obs_dims: dict[str, int],
        action_dim: int,
        device: torch.device | str,
    ):
        self.capacity = int(capacity)
        self.agent_names = list(agent_names)
        self.obs_dims = dict(obs_dims)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.position = 0
        self.size = 0

        self.obs = {
            agent: np.zeros((capacity, obs_dims[agent]), dtype=np.float32)
            for agent in agent_names
        }
        self.next_obs = {
            agent: np.zeros((capacity, obs_dims[agent]), dtype=np.float32)
            for agent in agent_names
        }
        self.actions = {
            agent: np.zeros((capacity,), dtype=np.int64)
            for agent in agent_names
        }
        self.one_hot_actions = {
            agent: np.zeros((capacity, action_dim), dtype=np.float32)
            for agent in agent_names
        }
        self.rewards = {
            agent: np.zeros((capacity,), dtype=np.float32)
            for agent in agent_names
        }
        self.dones = {
            agent: np.zeros((capacity,), dtype=np.float32)
            for agent in agent_names
        }
        self.adv_policy_ids = np.zeros((capacity,), dtype=np.int64)
        self.prey_policy_ids = np.zeros((capacity,), dtype=np.int64)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: dict[str, np.ndarray],
        actions: dict[str, int],
        one_hot_actions: dict[str, np.ndarray],
        rewards: dict[str, float],
        next_obs: dict[str, np.ndarray],
        dones: dict[str, bool],
        adv_policy_id: int = 0,
        prey_policy_id: int = 0,
    ) -> None:
        """写入一个完整的 multi-agent transition。"""

        index = self.position
        for agent in self.agent_names:
            self.obs[agent][index] = np.asarray(obs[agent], dtype=np.float32)
            self.actions[agent][index] = int(actions[agent])
            self.one_hot_actions[agent][index] = np.asarray(
                one_hot_actions[agent],
                dtype=np.float32,
            )
            self.rewards[agent][index] = float(rewards.get(agent, 0.0))
            self.next_obs[agent][index] = np.asarray(next_obs[agent], dtype=np.float32)
            self.dones[agent][index] = float(dones.get(agent, False))

        self.adv_policy_ids[index] = int(adv_policy_id)
        self.prey_policy_ids[index] = int(prey_policy_id)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        adv_policy_id: int | None = None,
        prey_policy_id: int | None = None,
    ) -> MADDPGBatch:
        if self.size < batch_size:
            raise ValueError(
                f"replay buffer 样本不足：当前 {self.size}，需要 {batch_size}"
            )

        valid_indices = np.arange(self.size)
        if adv_policy_id is not None:
            valid_indices = valid_indices[
                self.adv_policy_ids[valid_indices] == int(adv_policy_id)
            ]
        if prey_policy_id is not None:
            valid_indices = valid_indices[
                self.prey_policy_ids[valid_indices] == int(prey_policy_id)
            ]
        if len(valid_indices) < batch_size:
            raise ValueError(
                "replay buffer 中匹配 policy_id 的样本不足："
                f"当前 {len(valid_indices)}，需要 {batch_size}"
            )

        indices = np.random.choice(valid_indices, size=batch_size, replace=True)

        def tensor_dict(source: dict[str, np.ndarray], dtype: torch.dtype) -> dict[str, torch.Tensor]:
            return {
                agent: torch.as_tensor(
                    source[agent][indices],
                    dtype=dtype,
                    device=self.device,
                )
                for agent in self.agent_names
            }

        return MADDPGBatch(
            obs=tensor_dict(self.obs, torch.float32),
            actions=tensor_dict(self.actions, torch.long),
            one_hot_actions=tensor_dict(self.one_hot_actions, torch.float32),
            rewards=tensor_dict(self.rewards, torch.float32),
            next_obs=tensor_dict(self.next_obs, torch.float32),
            dones=tensor_dict(self.dones, torch.float32),
            adv_policy_ids=torch.as_tensor(
                self.adv_policy_ids[indices],
                dtype=torch.long,
                device=self.device,
            ),
            prey_policy_ids=torch.as_tensor(
                self.prey_policy_ids[indices],
                dtype=torch.long,
                device=self.device,
            ),
        )

    def count_policy_samples(
        self,
        adv_policy_id: int | None = None,
        prey_policy_id: int | None = None,
    ) -> int:
        valid_indices = np.arange(self.size)
        if adv_policy_id is not None:
            valid_indices = valid_indices[
                self.adv_policy_ids[valid_indices] == int(adv_policy_id)
            ]
        if prey_policy_id is not None:
            valid_indices = valid_indices[
                self.prey_policy_ids[valid_indices] == int(prey_policy_id)
            ]
        return int(len(valid_indices))
