# -*- coding: utf-8 -*-
"""Transition replay buffer for simplified QMIX."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Circular buffer storing one parallel-env transition per step."""

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
    ) -> None:
        self.capacity = int(capacity)
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)

        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        index = self.position
        self.obs[index] = np.asarray(obs, dtype=np.float32)
        self.state[index] = np.asarray(state, dtype=np.float32)
        self.actions[index] = np.asarray(actions, dtype=np.int64)
        self.rewards[index] = float(reward)
        self.next_obs[index] = np.asarray(next_obs, dtype=np.float32)
        self.next_state[index] = np.asarray(next_state, dtype=np.float32)
        self.dones[index] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | str) -> dict[str, torch.Tensor]:
        if self.size < batch_size:
            raise ValueError(
                f"Cannot sample batch_size={batch_size}; buffer only has {self.size} items."
            )

        indices = np.random.randint(0, self.size, size=batch_size)
        obs = torch.as_tensor(self.obs[indices], dtype=torch.float32, device=device)
        next_obs = torch.as_tensor(self.next_obs[indices], dtype=torch.float32, device=device)
        actions = torch.as_tensor(self.actions[indices], dtype=torch.long, device=device)
        rewards = torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device)
        dones = torch.as_tensor(self.dones[indices], dtype=torch.float32, device=device)

        if obs.shape != (batch_size, self.n_agents, self.obs_dim):
            raise RuntimeError(f"obs batch shape mismatch: {tuple(obs.shape)}")
        if next_obs.shape != (batch_size, self.n_agents, self.obs_dim):
            raise RuntimeError(f"next_obs batch shape mismatch: {tuple(next_obs.shape)}")
        if actions.shape != (batch_size, self.n_agents):
            raise RuntimeError(f"actions batch shape mismatch: {tuple(actions.shape)}")
        if rewards.shape != (batch_size,):
            raise RuntimeError(f"reward batch shape mismatch: {tuple(rewards.shape)}")
        if dones.shape != (batch_size,):
            raise RuntimeError(f"done batch shape mismatch: {tuple(dones.shape)}")

        return {
            "obs": obs,
            "state": torch.as_tensor(self.state[indices], dtype=torch.float32, device=device),
            "actions": actions,
            "reward": rewards,
            "next_obs": next_obs,
            "next_state": torch.as_tensor(self.next_state[indices], dtype=torch.float32, device=device),
            "done": dones,
        }
