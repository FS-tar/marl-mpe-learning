# -*- coding: utf-8 -*-
"""在 MPE simple_spread_v3 上训练教学版 shared PPO baseline。"""

from __future__ import annotations

import argparse
import csv
import importlib
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algorithms.ppo.buffer import RolloutBuffer
from algorithms.ppo.ppo_agent import PPOAgent


OBS_DIM = 18
ACTION_DIM = 5
OUTPUT_DIR = ROOT_DIR / "outputs" / "ppo_mpe"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CSV_PATH = OUTPUT_DIR / "train_log.csv"
PNG_PATH = OUTPUT_DIR / "reward_curve.png"


def load_simple_spread():
    """优先使用 mpe2，失败后回退到 PettingZoo 的旧导入路径。"""

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
                "无法导入 simple_spread_v3，请确认已安装 mpe2 或 pettingzoo[mpe]。"
            ) from pettingzoo_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train shared PPO on MPE simple_spread_v3.")
    parser.add_argument("--total-updates", type=int, default=50)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--max-cycles", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--render", action="store_true", default=False)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_env(env, seed: int | None = None):
    """兼容不同版本的 reset(seed=...) 支持情况。"""

    if seed is None:
        return env.reset()

    try:
        return env.reset(seed=seed)
    except TypeError:
        return env.reset()


def all_agents_done(terminations: dict, truncations: dict, agents: list[str]) -> bool:
    return all(
        terminations.get(agent, False) or truncations.get(agent, False)
        for agent in agents
    )


def obs_to_array(observations: dict, agents: list[str]) -> np.ndarray:
    """把 {agent: obs} 转成固定 agent 顺序的二维数组。"""

    return np.stack(
        [np.asarray(observations[agent], dtype=np.float32) for agent in agents],
        axis=0,
    )


def dict_values_to_array(values: dict, agents: list[str], default: float = 0.0) -> np.ndarray:
    return np.asarray([float(values.get(agent, default)) for agent in agents], dtype=np.float32)


def save_log(rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "update",
                "mean_episode_return",
                "policy_loss",
                "value_loss",
                "entropy",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def save_reward_curve(rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    updates = [row["update"] for row in rows]
    returns = [row["mean_episode_return"] for row in rows]

    plt.figure(figsize=(8, 4.5))
    plt.plot(updates, returns, marker="o")
    plt.xlabel("Update")
    plt.ylabel("Mean episode return")
    plt.title("Shared PPO on MPE simple_spread_v3")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    simple_spread_v3, source = load_simple_spread()
    render_mode = "human" if args.render else None
    env = simple_spread_v3.parallel_env(render_mode=render_mode, max_cycles=args.max_cycles)

    observations, infos = reset_env(env, seed=args.seed)
    del infos

    agents = list(env.possible_agents)
    if len(agents) != 3:
        print(f"警告：当前环境 agent 数量是 {len(agents)}，本脚本按共享 PPO 继续训练。")

    agent = PPOAgent(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        lr=args.lr,
        clip_eps=args.clip_eps,
    )
    buffer = RolloutBuffer(
        rollout_steps=args.rollout_steps,
        num_agents=len(agents),
        obs_dim=OBS_DIM,
        device=str(agent.device),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    log_rows = []
    recent_episode_returns = []
    current_episode_return = 0.0

    print(f"Loaded simple_spread_v3 from: {source}")

    for update in range(1, args.total_updates + 1):
        buffer.reset()

        for _ in range(args.rollout_steps):
            obs_array = obs_to_array(observations, agents)
            actions, log_probs, values = agent.act(obs_array)

            # PettingZoo Parallel API 需要 {agent: action} 字典。
            action_dict = {
                agent_name: int(action)
                for agent_name, action in zip(agents, actions)
                if agent_name in observations
            }

            next_observations, rewards, terminations, truncations, infos = env.step(action_dict)
            del infos

            reward_array = dict_values_to_array(rewards, agents)
            done_array = np.asarray(
                [
                    float(terminations.get(agent_name, False) or truncations.get(agent_name, False))
                    for agent_name in agents
                ],
                dtype=np.float32,
            )

            buffer.add(
                obs=obs_array,
                actions=actions,
                log_probs=log_probs,
                values=values,
                rewards=reward_array,
                dones=done_array,
            )

            # simple_spread 通常是团队共享 reward，这里用所有 agent reward 的均值记 episode return。
            current_episode_return += float(np.mean(reward_array))

            if all_agents_done(terminations, truncations, agents) or not next_observations:
                recent_episode_returns.append(current_episode_return)
                current_episode_return = 0.0
                next_observations, infos = reset_env(env)
                del infos

            observations = next_observations

        last_obs_array = obs_to_array(observations, agents)
        last_values = agent.value(last_obs_array)
        batch = buffer.compute_gae(
            last_values=last_values,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        info = agent.update(
            batch=batch,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
        )

        if recent_episode_returns:
            mean_episode_return = float(np.mean(recent_episode_returns[-10:]))
        else:
            mean_episode_return = current_episode_return

        row = {
            "update": update,
            "mean_episode_return": mean_episode_return,
            "policy_loss": info.policy_loss,
            "value_loss": info.value_loss,
            "entropy": info.entropy,
        }
        log_rows.append(row)

        print(
            f"update={update:03d} "
            f"mean_episode_return={mean_episode_return:.3f} "
            f"policy_loss={info.policy_loss:.4f} "
            f"value_loss={info.value_loss:.4f} "
            f"entropy={info.entropy:.4f}"
        )

        save_log(log_rows)
        save_reward_curve(log_rows)

        if update % 10 == 0:
            checkpoint_path = CHECKPOINT_DIR / f"ppo_update_{update:03d}.pt"
            agent.save(str(checkpoint_path))

    env.close()
    print(f"CSV saved to: {CSV_PATH}")
    print(f"Reward curve saved to: {PNG_PATH}")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
