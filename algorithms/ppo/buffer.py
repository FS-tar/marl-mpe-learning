# -*- coding: utf-8 -*-
"""教学版 PPO rollout buffer。

这里按 [rollout_steps, num_agents, ...] 保存数据。虽然 PPO 更新时会把
这些样本展平成一个 batch，但先按时间和 agent 保存更容易理解 GAE。
"""

# 本文件在 PPO 训练流程中的位置：
# 1. train_ppo_simple_spread.py 的 rollout 循环每走一步环境，就调用 buffer.add()。
# 2. buffer 保存 obs/actions/log_probs/values/rewards/dones。
# 3. rollout 收集结束后，训练脚本调用 buffer.compute_gae()。
# 4. compute_gae() 计算 advantages 和 returns，并把多 agent 数据展平成 PPO batch。
#收集数据，计算advantage和return


from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class PPOBatch:
    # PPOAgent.update() 只接收展平后的 PPOBatch。
    # rollout 阶段的数据形状通常是 [rollout_steps, num_agents, ...]；
    # 到这里会变成 [rollout_steps * num_agents, ...]。
    obs: torch.Tensor
    # obs：展平后的 observation，例如 [256 * 3, 18]。
    actions: torch.Tensor
    # actions：展平后的离散动作编号，例如 [256 * 3]。
    old_log_probs: torch.Tensor
    # old_log_probs：采样动作时旧策略给出的 log_prob，用于计算 PPO ratio。
    advantages: torch.Tensor
    # advantages：GAE 计算出的优势，表示动作比 critic 原先预期好多少。
    returns: torch.Tensor
    # returns：critic 的学习目标，通常等于 advantages + old_values。
    old_values: torch.Tensor
    # old_values：rollout 时 critic 的 value 估计，当前教学版保存但未做 value clipping。
#训练数据包

class RolloutBuffer:
    """保存一次 rollout，并计算 GAE advantage。"""

    def __init__(self, rollout_steps: int, num_agents: int, obs_dim: int, device: str):
        # 输入：
        # - rollout_steps：一次更新前先收集多少个环境 step。
        # - num_agents：simple_spread_v3 默认是 3。
        # - obs_dim：每个 agent 的 observation 维度，默认 18。
        # - device：后续转换成 torch.Tensor 时放到 CPU 或 CUDA。
        # 输出：
        # - 初始化一个空 rollout buffer。
        # 调用位置：
        # - train_ppo_simple_spread.py / main() 创建 buffer。
        self.rollout_steps = rollout_steps
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        # 每个 update 开始前清空旧数据。
        # self.step 表示当前已经写入了多少个环境 step。
        self.step = 0
        self.obs = np.zeros(
            (self.rollout_steps, self.num_agents, self.obs_dim),
            dtype=np.float32,
        )
        # obs：保存 observation，形状 [rollout_steps, num_agents, obs_dim]。
        self.actions = np.zeros((self.rollout_steps, self.num_agents), dtype=np.int64)
        # actions：保存每个 agent 执行的离散动作编号。
        self.log_probs = np.zeros((self.rollout_steps, self.num_agents), dtype=np.float32)
        # log_probs：保存采样时旧策略的 log_prob，后续作为 old_log_probs。
        self.values = np.zeros((self.rollout_steps, self.num_agents), dtype=np.float32)
        # values：保存 rollout 当时 critic 的 value 估计，用于 GAE。
        self.rewards = np.zeros((self.rollout_steps, self.num_agents), dtype=np.float32)
        # rewards：保存 env.step 后每个 agent 得到的 reward。
        self.dones = np.zeros((self.rollout_steps, self.num_agents), dtype=np.float32)
        # dones：保存每个 agent 是否结束，1.0 表示 terminated 或 truncated。
    #创建空表

    # 关键函数：RolloutBuffer.add()
    # 输入：一个环境 step 中所有 agent 的 obs/actions/log_probs/values/rewards/dones。
    # 输出：无返回值；写入 buffer 当前 step。
    # 调用位置：train_ppo_simple_spread.py 的 rollout 采样循环。
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

        # 输入：
        # - obs：当前 step 的 observation 数组，例如 [3, 18]。
        # - actions：当前 step 的动作数组，例如 [3]。
        # - log_probs：采样这些动作时的 log_prob，例如 [3]。
        # - values：critic 对当前 obs 的 value 估计，例如 [3]。
        # - rewards：env.step 后的 reward，例如 [3]。
        # - dones：env.step 后每个 agent 是否结束，例如 [3]。
        # 输出：
        # - 无返回值；把数据写入 self.step 对应的位置。
        # 调用位置：
        # - train_ppo_simple_spread.py 的 rollout 采样循环。
        if self.step >= self.rollout_steps:
            raise RuntimeError("RolloutBuffer 已满，请先调用 reset()。")

        self.obs[self.step] = obs
        # obs 是 PPO 的状态输入，后续会展平成 batch.obs。
        self.actions[self.step] = actions
        # actions 是 actor 实际执行的动作，PPO 更新时会重新计算这些动作的概率。
        self.log_probs[self.step] = log_probs
        # log_probs 是旧策略概率，更新时变成 old_log_probs。
        self.values[self.step] = values
        # values 是旧 critic 估计，compute_gae() 用它计算 delta。
        self.rewards[self.step] = rewards
        # rewards 是环境反馈，决定 returns 和 advantages 的方向。
        self.dones[self.step] = dones
        # dones 用来判断是否还能 bootstrap 下一步 value。
        self.step += 1

    # 关键函数：RolloutBuffer.compute_gae()
    # 输入：last_values、gamma、gae_lambda。
    # 输出：包含 advantages 和 returns 的 PPOBatch。
    # 调用位置：每个 update 收集完 rollout 后、PPOAgent.update() 前。
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

        # 输入：
        # - last_values：rollout 最后一个 next_obs 的 value，用于最后一步 bootstrap。
        # - gamma：未来 reward 折扣系数。
        # - gae_lambda：GAE 平滑系数，在偏差和方差之间折中。
        # 输出：
        # - PPOBatch：展平后的 obs/actions/old_log_probs/advantages/returns/old_values。
        # 调用位置：
        # - train_ppo_simple_spread.py 在每个 update 的 rollout 收集完成后调用。
        #
        # GAE 的直觉：
        # - critic 先预测当前 obs 的 value。
        # - 环境给出 reward 后，我们检查“实际结果”是否比 critic 预期更好。
        # - advantage > 0：这个动作比预期好，PPO 会倾向于提高它的概率。
        # - advantage < 0：这个动作比预期差，PPO 会倾向于降低它的概率。
        advantages = np.zeros_like(self.rewards, dtype=np.float32)
        # advantages 的形状和 rewards 一样：[rollout_steps, num_agents]。
        last_gae = np.zeros(self.num_agents, dtype=np.float32)
        # last_gae 保存“从未来传回来的 advantage”，每个 agent 各有一个值。

        #从最后一步往前推
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_values = last_values
                # 最后一个 rollout step 没有 self.values[t + 1]，
                # 所以使用外部传入的 last_values 作为下一步 value。
            else:
                next_values = self.values[t + 1]
                # 非最后一步可以直接使用 buffer 中保存的下一步 value。

            next_non_terminal = 1.0 - self.dones[t]
            # next_non_terminal=1 表示 episode 还没结束，可以使用 next_values。
            # next_non_terminal=0 表示 episode 已结束，不能再 bootstrap。
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            # delta 是一步 TD 误差：
            # 实际 reward + 折扣后的下一步 value - 当前 value 预测。
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            # GAE 把当前 delta 和未来 advantage 平滑地合在一起。
            advantages[t] = last_gae
            # 保存当前时间步、所有 agent 的 advantage。

        returns = advantages + self.values
        # return 是 critic 要学习的目标：
        # advantage = return - value，所以 return = advantage + value。

        # PPO 更新时不再区分时间和 agent，把所有样本展平成 batch。
        flat_obs = self.obs.reshape(-1, self.obs_dim)
        # [rollout_steps, num_agents, obs_dim] -> [rollout_steps * num_agents, obs_dim]
        flat_actions = self.actions.reshape(-1)
        # [rollout_steps, num_agents] -> [rollout_steps * num_agents]
        flat_log_probs = self.log_probs.reshape(-1)
        # 这些 log_probs 会作为 old_log_probs，计算 PPO ratio。
        flat_advantages = advantages.reshape(-1)
        # 展平后的 advantages 与 flat_obs 一一对应。
        flat_returns = returns.reshape(-1)
        # 展平后的 returns 用于 value_loss。
        flat_values = self.values.reshape(-1)
        # 展平后的 old_values 当前主要用于保留调试信息。

        return PPOBatch(
            obs=torch.as_tensor(flat_obs, dtype=torch.float32, device=self.device),
            actions=torch.as_tensor(flat_actions, dtype=torch.long, device=self.device),
            old_log_probs=torch.as_tensor(flat_log_probs, dtype=torch.float32, device=self.device),
            advantages=torch.as_tensor(flat_advantages, dtype=torch.float32, device=self.device),
            returns=torch.as_tensor(flat_returns, dtype=torch.float32, device=self.device),
            old_values=torch.as_tensor(flat_values, dtype=torch.float32, device=self.device),
        )
