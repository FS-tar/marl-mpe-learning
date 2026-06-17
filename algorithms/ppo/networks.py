# -*- coding: utf-8 -*-
"""PPO 使用的共享 Actor-Critic 网络。

本项目先做教学版 PPO：三个 agent 共用同一个网络，只是每个 agent
在每一步输入自己的 18 维 observation。
"""

# 本文件在 PPO 训练流程中的位置：
# 1. 训练脚本收集到 observation 后，会调用 PPOAgent.act()。
# 2. PPOAgent.act() 内部会调用这里的 ActorCritic.get_action_and_value()。
# 3. ActorCritic 同时输出 actor 需要的动作分布，以及 critic 需要的 value。
# 4. PPOAgent.update() 更新时，也会再次调用 get_action_and_value()，
#    用“新策略”重新计算“旧动作”的 log_prob，从而计算 PPO ratio。
#把observation转成动作概率actor和状态价值critic
#logits+value

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """离散动作 PPO 的 Actor-Critic。

    actor 输出 action logits，critic 输出当前 observation 的 value。
    simple_spread_v3 默认 observation 是 18 维，动作空间是 Discrete(5)。
    """

    def __init__(self, obs_dim: int = 18, action_dim: int = 5, hidden_dim: int = 128):
        # 输入：
        # - obs_dim：单个 agent 的 observation 维度，simple_spread_v3 中是 18。
        # - action_dim：离散动作数量，simple_spread_v3 中是 Discrete(5)。
        # - hidden_dim：教学版 MLP 隐藏层宽度。
        # 输出：
        # - 初始化一个共享 ActorCritic 网络对象。
        # 调用位置：
        # - PPOAgent.__init__() 中创建 self.model。
        super().__init__()

        # 共享特征层让 actor 和 critic 都先提取 observation 表示。
        # obs 的形状可以是 [batch_size, obs_dim]。
        # 在 simple_spread 的一次环境 step 中，batch_size 通常是 3，
        # 分别对应 agent_0、agent_1、agent_2。
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # actor head 输出每个离散动作的 logits，后续交给 Categorical 分布。
        # logits 不是概率，而是“动作偏好分数”；Categorical 会把它转成概率分布。
        self.actor = nn.Linear(hidden_dim, action_dim)

        # critic head 输出一个标量 value，表示当前 observation 的状态价值估计。
        # value 用于后续计算 GAE advantage 和 value_loss。
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 输入：
        # - obs：一批 observation，例如 [3, 18] 或 [minibatch_size, 18]。
        # 输出：
        # - logits：actor 输出的动作 logits，例如 [3, 5]。
        # - values：critic 输出的 value，例如 [3]。
        # 调用位置：
        # - get_action_and_value() 中用于采样动作和计算 log_prob。
        # - PPOAgent.value() 中用于 rollout 末尾 bootstrap。
        features = self.backbone(obs)
        # features 是共享特征，actor 和 critic 都从这里继续往下算。
        logits = self.actor(features)
        # logits 的每一行对应一个 agent/样本的 5 个动作分数。
        values = self.critic(features).squeeze(-1)
        # squeeze(-1) 把 [batch_size, 1] 变成 [batch_size]，方便和 rewards/returns 对齐。
        return logits, values
    #提取特征放入网络

    # 关键函数：ActorCritic.get_action_and_value()
    # 输入：obs，以及可选的 action。
    # 输出：action、log_prob、entropy、values。
    # 调用位置：PPOAgent.act() 采样动作；PPOAgent.update() 重新计算旧动作概率。
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样动作，或重新计算给定动作的 log_prob/value。

        PPO 更新时需要用新策略重新计算旧动作的 log_prob，所以这里同时
        支持 action=None 的采样模式和 action!=None 的评估模式。
        """

        # 输入：
        # - obs：一批 observation。
        # - action：可选。如果为 None，表示 rollout 阶段要从策略中采样新动作；
        #   如果不是 None，表示 PPO 更新阶段要评估“旧动作”的新 log_prob。
        # 输出：
        # - action：采样得到或传入的动作。
        # - log_prob：当前策略下 action 的 log probability。
        # - entropy：动作分布的熵，用于 entropy bonus，鼓励探索。
        # - values：critic 对 obs 的价值估计。
        # 调用位置：
        # - PPOAgent.act()：rollout 采样时 action=None。
        # - PPOAgent.update()：更新时传入 batch.actions，重新计算 new_log_probs。
        logits, values = self.forward(obs)
        # 用 actor logits 创建离散动作分布；适用于 Discrete(5)。
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()
            #按概率随机选择一个动作

        log_prob = dist.log_prob(action)
        #储存在buffer中
        # log_prob 会保存为 old_log_probs。PPO 更新时用它计算：
        # ratio = exp(new_log_prob - old_log_prob)。
        entropy = dist.entropy()
        # entropy 越大表示动作分布越随机；训练中会用它防止策略过早变得太确定。
        return action, log_prob, entropy, values
