# -*- coding: utf-8 -*-
"""MADDPG 使用的 Actor 和 centralized Critic 网络。

数据流说明：
- actor 是 decentralized 的：每个 agent 只输入自己的 observation，输出 Discrete(5)
  动作的 logits。
- critic 是 centralized 的：每个 agent 各有一个 critic，但训练时输入所有 agent
  的 observation 拼接和所有 agent 的 one-hot action 拼接，输出该 agent 的 Q value。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DiscreteActor(nn.Module):
    """离散动作 actor：obs -> action logits。

    simple_tag_v3 的动作空间是 Discrete(5)，所以 actor 不直接输出连续动作，而是输出
    每个离散动作的打分 logits。与环境交互时可 argmax 或采样；训练 actor 时需要可微的
    one-hot 动作，因此会在 trainer 中使用 straight-through Gumbel-Softmax。
    """

    def __init__(self, obs_dim: int, action_dim: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CentralizedCritic(nn.Module):
    """centralized critic：global_obs + global_actions -> Q_i。

    对 simple_tag_v3：
    - global_obs_dim = 16 + 16 + 16 + 14 = 62
    - global_action_dim = 5 * 4 = 20
    - critic_input_dim = 82
    """

    def __init__(
        self,
        global_obs_dim: int,
        global_action_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim + global_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        global_obs: torch.Tensor,
        global_actions: torch.Tensor,
    ) -> torch.Tensor:
        critic_input = torch.cat([global_obs, global_actions], dim=-1)
        return self.net(critic_input).squeeze(-1)


def one_hot_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """把 logits 转成 hard one-hot 动作，常用于 target action。"""

    action_indices = torch.argmax(logits, dim=-1)
    return F.one_hot(action_indices, num_classes=logits.shape[-1]).float()


def gumbel_softmax_action(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """返回可反传的 hard one-hot 动作。

    离散动作的 argmax/采样本身不可导；MADDPG 的 actor loss 需要通过 critic 的 Q
    反传到当前 actor，所以这里用 straight-through Gumbel-Softmax：前向看起来是
    one-hot 离散动作，反向使用 soft sample 的梯度。
    """

    return F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)


@torch.no_grad()
def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """target_param = tau * param + (1 - tau) * target_param。"""

    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.mul_(1.0 - tau)
        target_param.data.add_(tau * source_param.data)


@torch.no_grad()
def hard_update(source: nn.Module, target: nn.Module) -> None:
    """初始化 target network 时直接复制参数。"""

    target.load_state_dict(source.state_dict())
