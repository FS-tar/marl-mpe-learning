# -*- coding: utf-8 -*-
"""用 5 步 rollout 打印 shared PPO 的数据流。

这个脚本只用于学习和调试数据形状：
- 不训练很多轮
- 不保存模型
- 不画图
- 不修改已有 PPO 训练逻辑
"""

from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algorithms.ppo.buffer import RolloutBuffer
from algorithms.ppo.networks import ActorCritic
from algorithms.ppo.ppo_agent import PPOAgent


OBS_DIM = 18
ACTION_DIM = 5
AGENT_NUM = 3
ROLLOUT_STEPS = 5
MAX_CYCLES = 100
GAMMA = 0.99
GAE_LAMBDA = 0.95
SEED = 1


def load_simple_spread():
    """优先导入 mpe2，失败后回退到 pettingzoo.mpe。"""

    try:
        return importlib.import_module("mpe2.simple_spread_v3"), "mpe2"
    except ImportError as mpe2_error:
        try:
            return (
                importlib.import_module("pettingzoo.mpe.simple_spread_v3"),
                "pettingzoo.mpe",
            )
        except ImportError as pettingzoo_error:
            raise ImportError(
                "无法导入 simple_spread_v3，请确认已安装 mpe2 或 pettingzoo。"
            ) from pettingzoo_error


def reset_env(env, seed: int | None = None):
    """兼容不同版本 reset(seed=...) 的支持情况。"""

    if seed is None:
        return env.reset()

    try:
        return env.reset(seed=seed)
    except TypeError:
        return env.reset()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def obs_to_array(observations: dict, agents: list[str]) -> np.ndarray:
    """把 MPE 的 {agent: obs} 转成 [agent_num, obs_dim] 数组。"""

    return np.stack(
        [np.asarray(observations[agent], dtype=np.float32) for agent in agents],
        axis=0,
    )


def rewards_to_array(rewards: dict, agents: list[str]) -> np.ndarray:
    return np.asarray([float(rewards.get(agent, 0.0)) for agent in agents], dtype=np.float32)


def dones_to_array(terminations: dict, truncations: dict, agents: list[str]) -> np.ndarray:
    return np.asarray(
        [
            float(terminations.get(agent, False) or truncations.get(agent, False))
            for agent in agents
        ],
        dtype=np.float32,
    )


def combined_dones_dict(terminations: dict, truncations: dict, agents: list[str]) -> dict:
    return {
        agent: bool(terminations.get(agent, False) or truncations.get(agent, False))
        for agent in agents
    }


def short_array(values, max_items: int = 5) -> str:
    """把数组前几项格式化成适合教学打印的短字符串。"""

    array = np.asarray(values).reshape(-1)[:max_items]
    return np.array2string(array, precision=4, separator=", ")


def print_obs_details(observations: dict, agents: list[str]) -> None:
    print(f"observations 的 keys: {list(observations.keys())}")
    for agent in agents:
        obs = np.asarray(observations[agent], dtype=np.float32)
        print(f"  {agent} 的 obs shape: {obs.shape}, 前 5 个值: {short_array(obs)}")


def print_actor_outputs(
    agents: list[str],
    actions: np.ndarray,
    log_probs: np.ndarray,
    values: np.ndarray,
) -> None:
    print("ActorCritic 输出：")
    for index, agent in enumerate(agents):
        print(
            f"  {agent}: action={int(actions[index])}, "
            f"log_prob={float(log_probs[index]):.6f}, "
            f"value={float(values[index]):.6f}"
        )


def main() -> None:
    set_seed(SEED)

    simple_spread_v3, source = load_simple_spread()
    env = simple_spread_v3.parallel_env(render_mode=None, max_cycles=MAX_CYCLES)
    observations, infos = reset_env(env, seed=SEED)
    del infos

    agents = list(env.possible_agents)
    print("========== PPO 数据流调试脚本 ==========")
    print(f"环境来源: {source}")
    print(f"环境名称: simple_spread_v3")
    print(f"agent_num: {len(agents)}，期望值: {AGENT_NUM}")
    print(f"obs_dim: {OBS_DIM}, action_dim: {ACTION_DIM}")
    print(f"rollout_steps: {ROLLOUT_STEPS}")
    print("")

    ppo_agent = PPOAgent(obs_dim=OBS_DIM, action_dim=ACTION_DIM, device="cpu")
    actor_critic: ActorCritic = ppo_agent.model
    buffer = RolloutBuffer(
        rollout_steps=ROLLOUT_STEPS,
        num_agents=len(agents),
        obs_dim=OBS_DIM,
        device=str(ppo_agent.device),
    )

    print("已创建 PPOAgent、共享 ActorCritic 和 RolloutBuffer。")
    print(f"ActorCritic 类型: {actor_critic.__class__.__name__}")
    print(f"RolloutBuffer 初始 step: {buffer.step}")
    print("")

    for step in range(ROLLOUT_STEPS):
        print(f"========== 第 {step + 1} / {ROLLOUT_STEPS} 步 ==========")
        print_obs_details(observations, agents)

        obs_array = obs_to_array(observations, agents)
        print(f"组装后的 obs_array shape: {obs_array.shape}")

        actions, log_probs, values = ppo_agent.act(obs_array)
        print_actor_outputs(agents, actions, log_probs, values)

        action_dict = {
            agent: int(action)
            for agent, action in zip(agents, actions)
            if agent in observations
        }
        print(f"传给 env.step 的 actions 字典: {action_dict}")

        next_observations, rewards, terminations, truncations, infos = env.step(action_dict)
        del infos

        done_dict = combined_dones_dict(terminations, truncations, agents)
        reward_array = rewards_to_array(rewards, agents)
        done_array = dones_to_array(terminations, truncations, agents)

        print(f"env.step 后 rewards 字典: {rewards}")
        print(f"env.step 后 dones 字典: {done_dict}")
        print(f"转成数组后的 reward_array: {short_array(reward_array, max_items=AGENT_NUM)}")
        print(f"转成数组后的 done_array: {short_array(done_array, max_items=AGENT_NUM)}")

        buffer.add(
            obs=obs_array,
            actions=actions,
            log_probs=log_probs,
            values=values,
            rewards=reward_array,
            dones=done_array,
        )
        print(f"buffer 当前已保存到第 {buffer.step} 步 / 共 {ROLLOUT_STEPS} 步")
        print("")

        observations = next_observations
        if not observations:
            print("当前 episode 已结束，调试脚本 reset 环境以继续补满 5 步 rollout。")
            observations, infos = reset_env(env)
            del infos

    print("========== rollout 结束，开始计算 GAE ==========")
    last_obs_array = obs_to_array(observations, agents)
    last_values = ppo_agent.value(last_obs_array)
    print(f"最后一个 next obs 的 last_values shape: {last_values.shape}")
    print(f"last_values 前几项: {short_array(last_values, max_items=AGENT_NUM)}")

    batch = buffer.compute_gae(
        last_values=last_values,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
    )

    print("")
    print("========== buffer 内部保存的数据形状 ==========")
    print(f"buffer.obs shape: {buffer.obs.shape}")
    print(f"buffer.actions shape: {buffer.actions.shape}")
    print(f"buffer.rewards shape: {buffer.rewards.shape}")
    print(f"buffer.values shape: {buffer.values.shape}")

    print("")
    print("========== compute_gae 后展平 batch 的数据 ==========")
    print(f"advantages shape: {tuple(batch.advantages.shape)}")
    print(f"advantages 前几项: {short_array(batch.advantages.cpu().numpy(), max_items=10)}")
    print(f"returns shape: {tuple(batch.returns.shape)}")
    print(f"returns 前几项: {short_array(batch.returns.cpu().numpy(), max_items=10)}")
    print(f"展平后 batch.obs shape: {tuple(batch.obs.shape)}")
    print(f"batch.actions shape: {tuple(batch.actions.shape)}")
    print(f"batch.advantages shape: {tuple(batch.advantages.shape)}")
    print(f"batch.returns shape: {tuple(batch.returns.shape)}")

    print("")
    print("========== 数据流总结 ==========")
    print(
        "5 步 rollout * 3 个 agent = 15 条 transition；"
        "因此展平后的 batch.obs 形状应该接近 [15, 18]。"
    )
    print("这个脚本没有执行 PPO update，没有保存模型，也没有生成图片。")

    env.close()


if __name__ == "__main__":
    main()
