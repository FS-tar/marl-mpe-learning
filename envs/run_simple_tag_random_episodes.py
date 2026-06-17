# -*- coding: utf-8 -*-
"""统计 simple_tag_v3 随机策略表现。

simple_tag 是追捕-逃逸环境：
- adversary 是追捕者，通常有多个。
- agent_0 是逃避者，也可以理解为 prey。

这个脚本只建立 random baseline，不训练 PPO/QMIX/MADDPG。
"""

from __future__ import annotations

import argparse
import csv
import importlib
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("outputs") / "mpe"
CSV_PATH = OUTPUT_DIR / "random_simple_tag.csv"
PNG_PATH = OUTPUT_DIR / "random_simple_tag.png"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run random baseline on simple_tag_v3.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-cycles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def reset_env(env, seed: int | None = None):
    """兼容不同版本 reset(seed=...) 的支持情况。"""

    if seed is None:
        return env.reset()

    try:
        return env.reset(seed=seed)
    except TypeError:
        return env.reset()


def all_agents_done(terminations: dict, truncations: dict) -> bool:
    """如果所有 agent 都 terminated 或 truncated，就提前结束 episode。"""

    agents = set(terminations) | set(truncations)
    if not agents:
        return False

    return all(
        terminations.get(agent, False) or truncations.get(agent, False)
        for agent in agents
    )


def is_adversary(agent_name: str) -> bool:
    """simple_tag 中名称包含 adversary 的 agent 视为追捕者。"""

    return "adversary" in agent_name.lower()


def adversary_agents(possible_agents: list[str]) -> list[str]:
    return [agent for agent in possible_agents if is_adversary(agent)]


def mean_adversary_reward(rewards: dict, adversaries: list[str]) -> float:
    """计算当前 step 的追捕者团队平均 reward。"""

    values = [float(rewards.get(agent, 0.0)) for agent in adversaries]
    if not values:
        return 0.0
    return sum(values) / len(values)


def prey_reward(rewards: dict, prey_agent: str = "agent_0") -> float:
    """agent_0 是逃避者，单独统计它的累计回报。"""

    return float(rewards.get(prey_agent, 0.0))


def next_available_path(path: Path) -> Path:
    """如果目标文件已存在，返回一个不会覆盖旧结果的新路径。"""

    if not path.exists():
        return path

    for index in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"无法找到可用输出文件名: {path}")


def write_csv(rows: list[dict]) -> Path:
    """保存 episode 统计结果，不覆盖已有 random baseline。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = next_available_path(CSV_PATH)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "episode",
                "steps",
                "adversary_team_return",
                "prey_return",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


def write_plot(adversary_returns: list[float], prey_returns: list[float]) -> Path:
    """把追捕者和逃避者的随机策略回报画在同一张图里。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = next_available_path(PNG_PATH)

    episodes = list(range(1, len(adversary_returns) + 1))
    plt.figure(figsize=(8, 4.5))
    plt.plot(episodes, adversary_returns, marker="o", label="adversary_team_return")
    plt.plot(episodes, prey_returns, marker="s", label="prey_return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Random policy on MPE simple_tag_v3")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def print_summary(name: str, returns: list[float]) -> None:
    average_return = sum(returns) / len(returns)
    best_return = max(returns)
    worst_return = min(returns)

    print(f"{name} average return: {average_return:.3f}")
    print(f"{name} best return:    {best_return:.3f}")
    print(f"{name} worst return:   {worst_return:.3f}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    simple_tag_v3, source = load_simple_tag()
    env = simple_tag_v3.parallel_env(max_cycles=args.max_cycles)
    adversaries = adversary_agents(list(env.possible_agents))

    print(f"Loaded simple_tag_v3 from: {source}")
    print(f"possible_agents: {env.possible_agents}")
    print(f"adversary agents: {adversaries}")
    print("prey agent: agent_0")

    episode_rows = []
    adversary_returns = []
    prey_returns = []

    for episode in range(1, args.episodes + 1):
        observations, infos = reset_env(env, seed=args.seed + episode - 1)
        del infos

        adversary_team_return = 0.0
        prey_episode_return = 0.0
        step_count = 0

        for _ in range(args.max_cycles):
            if not observations:
                break

            # 每个当前活跃 agent 都使用随机动作。
            actions = {
                agent: env.action_space(agent).sample()
                for agent in observations.keys()
            }

            observations, rewards, terminations, truncations, infos = env.step(actions)
            del infos

            # adversary 是追捕者，统计三个 adversary reward 的平均累计回报。
            adversary_team_return += mean_adversary_reward(rewards, adversaries)

            # agent_0 是逃避者，单独统计它的累计回报。
            prey_episode_return += prey_reward(rewards, prey_agent="agent_0")

            step_count += 1

            if all_agents_done(terminations, truncations):
                break

        episode_rows.append(
            {
                "episode": episode,
                "steps": step_count,
                "adversary_team_return": adversary_team_return,
                "prey_return": prey_episode_return,
            }
        )
        adversary_returns.append(adversary_team_return)
        prey_returns.append(prey_episode_return)

        print(
            f"episode={episode}, steps={step_count}, "
            f"adversary_team_return={adversary_team_return:.3f}, "
            f"prey_return={prey_episode_return:.3f}"
        )

    env.close()

    print_summary("adversary", adversary_returns)
    print_summary("prey", prey_returns)

    csv_path = write_csv(episode_rows)
    png_path = write_plot(adversary_returns, prey_returns)

    print(f"CSV saved to: {csv_path}")
    print(f"Plot saved to: {png_path}")


if __name__ == "__main__":
    main()
