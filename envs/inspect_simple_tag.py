# -*- coding: utf-8 -*-
"""检查 MPE simple_tag_v3 环境接口。

这个脚本只用于学习 simple_tag_v3 的多智能体环境结构：
- 不训练算法
- 不实现 PPO/QMIX/MADDPG
- 不修改已有 PPO 训练代码
"""

from __future__ import annotations

import importlib

import numpy as np


def load_simple_tag():
    """优先导入 mpe2.simple_tag_v3，失败后回退到 pettingzoo.mpe。"""

    try:
        return importlib.import_module("mpe2.simple_tag_v3"), "mpe2"
    except ImportError as mpe2_error:
        try:
            return (
                importlib.import_module("pettingzoo.mpe.simple_tag_v3"),
                "pettingzoo.mpe",
            )
        except ImportError as pettingzoo_error:
            raise ImportError(
                "无法导入 simple_tag_v3，请确认已安装 mpe2 或 pettingzoo.mpe。"
            ) from pettingzoo_error


def is_adversary(agent_name: str) -> bool:
    """根据 agent 名称粗略判断是否为 adversary。"""

    return "adversary" in agent_name.lower()


def observation_shape(observation) -> tuple:
    """把 observation 转成 numpy 后返回 shape。"""

    return np.asarray(observation).shape


def all_agents_done(terminations: dict, truncations: dict) -> bool:
    """所有当前返回的 agent 都结束时，episode 才算结束。"""

    agents = set(terminations) | set(truncations)
    if not agents:
        return False

    return all(
        terminations.get(agent, False) or truncations.get(agent, False)
        for agent in agents
    )


def print_agent_spaces(env, observations: dict) -> None:
    """打印每个 agent 的空间、obs shape 和角色信息。"""

    print("\n每个 agent 的空间与 observation 信息：")
    for agent in env.possible_agents:
        obs = observations.get(agent)
        obs_shape = observation_shape(obs) if obs is not None else None

        print(f"- {agent}")
        print(f"  observation_space: {env.observation_space(agent)}")
        print(f"  action_space:      {env.action_space(agent)}")
        print(f"  obs shape:         {obs_shape}")
        print(f"  名称是否包含 adversary: {is_adversary(agent)}")


def summarize_simple_tag(env, observations: dict, first_rewards: dict | None) -> None:
    """根据实际打印到的信息，总结 simple_tag 和 simple_spread 的差异。"""

    possible_agents = list(env.possible_agents)
    obs_shapes = {
        agent: observation_shape(observation)
        for agent, observation in observations.items()
    }
    obs_dims = {
        agent: shape[0] if len(shape) == 1 else shape
        for agent, shape in obs_shapes.items()
    }
    unique_obs_dims = set(obs_dims.values())
    has_adversary = any(is_adversary(agent) for agent in possible_agents)

    print("\n和 simple_spread_v3 的关键差异总结：")
    print(f"- agent 数量：simple_tag_v3 当前 possible_agents 数量是 {len(possible_agents)}。")
    print(
        "- 角色是否同质："
        + ("不是同质角色；名称中包含 adversary，说明有追捕方/逃跑方。"
           if has_adversary else
           "从 agent 名称看没有 adversary，但仍建议结合 reward 和源码确认角色。")
    )
    print(
        "- obs_dim 是否一致："
        + ("一致。" if len(unique_obs_dims) == 1 else "不一致。")
        + f" 当前 observation 维度记录为 {obs_dims}。"
    )

    if first_rewards:
        reward_values = [float(value) for value in first_rewards.values()]
        reward_same_sign = all(value >= 0 for value in reward_values) or all(
            value <= 0 for value in reward_values
        )
        print(
            "- reward 是否同向："
            + ("这一步 reward 符号看起来同向，仍需多步观察。"
               if reward_same_sign else
               "这一步 reward 出现不同方向，说明不同角色目标可能相反。")
        )
        print(f"  第一次 step 的 rewards: {first_rewards}")
    else:
        print("- reward 是否同向：本次没有拿到 step reward，无法判断。")

    print(
        "- 是否适合直接 shared PPO：不建议直接照搬 simple_spread 的 shared PPO。"
        " simple_spread 通常是 3 个同质协作 agent；simple_tag 通常包含 adversary 和 good agents，"
        "角色、目标和 observation 维度可能不同。更稳妥的做法是先区分角色，再决定是否共享网络。"
    )


def main() -> None:
    simple_tag_v3, source = load_simple_tag()
    env = simple_tag_v3.parallel_env(max_cycles=100)

    observations, infos = env.reset()
    del infos

    print("========== simple_tag_v3 环境检查 ==========")
    print(f"环境来源: {source}")
    print(f"possible_agents: {env.possible_agents}")
    print(f"当前 observations keys: {list(observations.keys())}")

    print_agent_spaces(env, observations)

    print("\n随机运行 5 步：")
    first_rewards = None
    for step in range(1, 6):
        current_agents = list(observations.keys())
        if not current_agents:
            print("当前 observations 为空，episode 已结束。")
            break

        actions = {
            agent: env.action_space(agent).sample()
            for agent in current_agents
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        del infos

        if first_rewards is None:
            first_rewards = rewards

        print(f"\nStep {step}")
        print(f"actions:      {actions}")
        print(f"rewards:      {rewards}")
        print(f"terminations: {terminations}")
        print(f"truncations:  {truncations}")

        if all_agents_done(terminations, truncations):
            print("所有 agent 都已结束，本次随机检查提前停止。")
            break

    summarize_simple_tag(env, observations, first_rewards)
    env.close()


if __name__ == "__main__":
    main()
