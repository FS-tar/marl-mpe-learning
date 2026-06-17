# -*- coding: utf-8 -*-
"""教学版 shared PPO agent。"""

# 本文件在 PPO 训练流程中的位置：
# 1. PPOAgent.act() 在 rollout 阶段被训练脚本调用，用当前策略采样动作。
# 2. PPOAgent.value() 在 rollout 结束后被调用，用于 GAE 的最后一步 bootstrap。
# 3. PPOAgent.update() 在 compute_gae() 之后被调用，执行 PPO clipped objective 更新。
# 4. 这里的 PPOAgent 是算法封装，不是 MPE 环境中的某一个 agent；
#    simple_spread_v3 的 3 个 agent 共享同一个 PPOAgent.model。
#数据更新网络

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
    # PPOAgent.update() 返回给训练脚本的日志信息。
    policy_loss: float
    # policy_loss：actor 的 PPO clipped loss。
    value_loss: float
    # value_loss：critic 拟合 returns 的 MSE loss。
    entropy: float
    # entropy：动作分布的平均熵，用于观察探索程度。


class PPOAgent:
    """三个 MPE agent 共享的一套 PPO 策略。"""

    def __init__(
        self,
        obs_dim: int = 18,
        action_dim: int = 5,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str | None = None,
    ):
        # 输入：
        # - obs_dim/action_dim：网络输入输出维度，simple_spread_v3 中是 18 和 5。
        # - hidden_dim：ActorCritic 隐藏层宽度，训练脚本可用 --hidden-dim 调整。
        # - lr：Adam 学习率。
        # - clip_eps：PPO ratio 裁剪范围。
        # - entropy_coef：entropy bonus 权重。
        # - value_coef：value loss 权重。
        # - max_grad_norm：梯度裁剪阈值。
        # 输出：
        # - 初始化 PPOAgent，包括共享 ActorCritic 和 optimizer。
        # 调用位置：
        # - train_ppo_simple_spread.py / main()。
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.clip_eps = clip_eps
        # clip_eps 控制 ratio 允许偏离 1 的范围，例如 0.2 表示 [0.8, 1.2]。
        self.entropy_coef = entropy_coef
        # entropy_coef 越大，越鼓励策略保持随机性和探索。
        self.value_coef = value_coef
        # value_coef 控制 critic loss 在 total loss 中的权重。
        self.max_grad_norm = max_grad_norm
        # max_grad_norm 用于防止梯度过大导致训练不稳定。

        self.model = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        # self.model 是三个 MPE agent 共享的 ActorCritic 网络。
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        # optimizer 负责根据 total loss 更新 ActorCritic 参数。

    @torch.no_grad()
    # 关键函数：PPOAgent.act()
    # 输入：obs_batch，例如 [3, 18]。
    # 输出：actions、log_probs、values，分别对应每个 agent。
    # 调用位置：train_ppo_simple_spread.py 的 rollout 采样循环。
    def act(self, obs_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """给多个 agent 的 observation 同时采样动作。"""

        # 输入：
        # - obs_batch：多个 agent 的 observation，例如 [3, 18]。
        # 输出：
        # - actions：每个 agent 采样到的动作，例如 [3]。
        # - log_probs：旧策略对这些动作的 log_prob，例如 [3]。
        # - values：critic 对这些 observation 的 value 估计，例如 [3]。
        # 调用位置：
        # - train_ppo_simple_spread.py 的 rollout 采样循环。
        #
        # @torch.no_grad() 表示这里只采样数据，不计算梯度、不更新网络。
        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        # obs 从 numpy 转成 torch.Tensor，才能输入神经网络。
        actions, log_probs, _, values = self.model.get_action_and_value(obs_tensor)
        # get_action_and_value() 内部会创建 Categorical 分布并采样 action。
        # 这里的 log_probs 会被 buffer 保存成 old_log_probs。

        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
        )

    @torch.no_grad()
    def act_deterministic(self, obs_batch: np.ndarray) -> np.ndarray:
        """给评估阶段使用的确定性动作。

        输入：
        - obs_batch：多个 agent 的 observation，例如 [3, 18]。
        输出：
        - actions：actor logits 最大的动作编号，例如 [3]。
        调用位置：
        - train_ppo_simple_spread.py 的 evaluation 函数。

        和 act() 的区别：
        - act() 从 Categorical 分布里随机采样，用于训练探索。
        - act_deterministic() 直接取 logits 最大的动作，用于观察当前策略本身的表现。
        """

        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        logits, _ = self.model(obs_tensor)
        actions = torch.argmax(logits, dim=-1)
        return actions.cpu().numpy()

    @torch.no_grad()
    def value(self, obs_batch: np.ndarray) -> np.ndarray:
        """只计算 value，用于 rollout 末尾 bootstrap。"""

        # 输入：
        # - obs_batch：rollout 结束时最后一个 next_observation，例如 [3, 18]。
        # 输出：
        # - values：最后一个 next_observation 的 value，例如 [3]。
        # 调用位置：
        # - train_ppo_simple_spread.py 在 buffer.compute_gae() 之前。
        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        _, values = self.model(obs_tensor)
        # 这里只需要 critic value，不需要 actor logits 或 action。
        return values.cpu().numpy()

    # 关键函数：PPOAgent.update()
    # 输入：compute_gae() 返回的 PPOBatch、ppo_epochs、minibatch_size。
    # 输出：PPOUpdateInfo，用于日志显示 policy_loss/value_loss/entropy。
    # 调用位置：train_ppo_simple_spread.py 每个 update 的 rollout 结束后。
    def update(
        self,
        batch: PPOBatch,
        ppo_epochs: int,
        minibatch_size: int,
    ) -> PPOUpdateInfo:
        """执行 PPO clipped objective 更新。"""

        # 输入：
        # - batch：RolloutBuffer.compute_gae() 返回的展平样本。
        # - ppo_epochs：同一批 rollout 数据重复训练几轮。
        # - minibatch_size：每次更新使用多少条 transition。
        # 输出：
        # - PPOUpdateInfo：返回平均 policy_loss/value_loss/entropy，供训练日志打印。
        # 调用位置：
        # - train_ppo_simple_spread.py 的每个 update 末尾。
        #
        # update() 是 PPO 的核心：
        # old_log_probs 来自 rollout 时的旧策略；
        # new_log_probs 是当前正在更新的新策略重新算出来的；
        # ratio 比较新旧策略对同一个动作的概率变化。
        advantages = batch.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # advantages 标准化后均值接近 0、标准差接近 1，通常能让 policy update 更稳定。

        batch_size = batch.obs.shape[0]
        # batch_size = rollout_steps * num_agents，例如 256 * 3 = 768。
        policy_losses = []
        # 保存每个 minibatch 的 policy_loss，最后取平均用于日志。
        value_losses = []
        # 保存每个 minibatch 的 value_loss。
        entropies = []
        # 保存每个 minibatch 的 entropy。

        for _ in range(ppo_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            #把样本顺序打乱

            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                #切开成小块

                _, new_log_probs, entropy, new_values = self.model.get_action_and_value(
                    batch.obs[mb_idx],
                    batch.actions[mb_idx],
                )
                #entropy：当前策略随机程度 new_values：当前 critic 对这些 obs 的新 value 估计
                # 注意：这里传入 batch.actions[mb_idx]，表示“评估旧动作”，不是重新采样动作。
                # new_log_probs 是新策略对旧动作的 log_prob。

                # ratio = pi_new(a|s) / pi_old(a|s)，old_log_prob 必须来自采样当时的策略。
                #比较新旧策略对同一个动作的概率变化
                log_ratio = new_log_probs - batch.old_log_probs[mb_idx]
                # log_ratio = log(pi_new) - log(pi_old) = log(pi_new / pi_old)。
                ratio = log_ratio.exp()
                # ratio > 1：新策略比旧策略更倾向这个动作。
                # ratio < 1：新策略比旧策略更不倾向这个动作。

                mb_advantages = advantages[mb_idx]
                # mb_advantages 表示这些动作比 critic 预期好还是差。
                unclipped = ratio * mb_advantages
                #动作好坏决定新策略概率
                clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                #限制策略变化不要太大
                clipped = clipped * mb_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                # policy_loss 前面取负号，是因为 PyTorch optimizer 默认最小化 loss；
                # PPO 原始目标是最大化 clipped objective。

                # critic 学习拟合 GAE 得到的 return。
                value_loss = nn.functional.mse_loss(new_values, batch.returns[mb_idx])
                #让 new_values 尽量接近 returns
                # value_loss 越小，说明 critic 对 return 的估计越接近。

                entropy_loss = entropy.mean()
                #鼓励探索
                # entropy 越大，动作分布越分散；过低可能说明策略过早变得确定。
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                #总 loss = actor loss + critic loss - 探索奖励。
                # 这里的 loss 就是 total_loss：
                # actor 希望 policy_loss 小，critic 希望 value_loss 小，
                # entropy 项用减号表示“鼓励 entropy 变大”。
                self.optimizer.zero_grad()
                # 清空上一轮 minibatch 的梯度。
                loss.backward()
                # 反向传播，计算 ActorCritic 所有参数的梯度。
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                # 梯度裁剪，避免一次更新步子太大。
                self.optimizer.step()
                # 根据梯度更新共享 ActorCritic 参数。

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
