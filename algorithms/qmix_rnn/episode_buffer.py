# -*- coding: utf-8 -*-
"""Episode replay buffer for recurrent QMIX."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch


class EpisodeReplayBuffer:
    """Circular replay buffer storing complete episodes with variable lengths."""

    def __init__(self, capacity: int, n_agents: int, obs_dim: int) -> None:
        self.capacity = int(capacity)
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.episodes: deque[dict[str, np.ndarray]] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.episodes)

    def add_episode(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32).reshape(-1, 1)

        if obs.ndim != 3 or obs.shape[1:] != (self.n_agents, self.obs_dim):
            raise ValueError(
                f"obs must be [T, {self.n_agents}, {self.obs_dim}], got {obs.shape}"
            )
        if next_obs.shape != obs.shape:
            raise ValueError(f"next_obs shape must match obs, got {next_obs.shape} vs {obs.shape}")
        if actions.shape != (obs.shape[0], self.n_agents):
            raise ValueError(
                f"actions must be [T, {self.n_agents}], got {actions.shape}"
            )
        if rewards.shape != (obs.shape[0], 1):
            raise ValueError(f"rewards must be [T, 1], got {rewards.shape}")
        if dones.shape != (obs.shape[0], 1):
            raise ValueError(f"dones must be [T, 1], got {dones.shape}")

        filled = np.ones((obs.shape[0], 1), dtype=np.float32)
        self.episodes.append(
            {
                "obs": obs,
                "actions": actions,
                "rewards": rewards,
                "next_obs": next_obs,
                "dones": dones,
                "filled": filled,
            }
        )

    def sample(self, batch_size: int, device: torch.device | str) -> dict[str, torch.Tensor]:
        if len(self.episodes) < batch_size:
            raise ValueError(
                f"Cannot sample batch_size={batch_size}; buffer has {len(self.episodes)} episodes."
            )

        indices = np.random.randint(0, len(self.episodes), size=int(batch_size))
        episodes = [self.episodes[index] for index in indices]
        max_seq_len = max(int(episode["obs"].shape[0]) for episode in episodes)

        obs = np.zeros(
            (batch_size, max_seq_len, self.n_agents, self.obs_dim),
            dtype=np.float32,
        )
        actions = np.zeros((batch_size, max_seq_len, self.n_agents), dtype=np.int64)
        rewards = np.zeros((batch_size, max_seq_len, 1), dtype=np.float32)
        next_obs = np.zeros_like(obs)
        dones = np.ones((batch_size, max_seq_len, 1), dtype=np.float32)
        filled = np.zeros((batch_size, max_seq_len, 1), dtype=np.float32)

        for batch_index, episode in enumerate(episodes):
            length = int(episode["obs"].shape[0])
            obs[batch_index, :length] = episode["obs"]
            actions[batch_index, :length] = episode["actions"]
            rewards[batch_index, :length] = episode["rewards"]
            next_obs[batch_index, :length] = episode["next_obs"]
            dones[batch_index, :length] = episode["dones"]
            filled[batch_index, :length] = episode["filled"]

        return {
            "obs": torch.as_tensor(obs, dtype=torch.float32, device=device),
            "actions": torch.as_tensor(actions, dtype=torch.long, device=device),
            "rewards": torch.as_tensor(rewards, dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(next_obs, dtype=torch.float32, device=device),
            "dones": torch.as_tensor(dones, dtype=torch.float32, device=device),
            "filled": torch.as_tensor(filled, dtype=torch.float32, device=device),
        }
