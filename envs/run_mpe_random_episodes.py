# -*- coding: utf-8 -*-
"""统计 simple_spread_v3 随机策略表现。

这个脚本不打开图形界面，也不训练任何模型；它只是建立一个随机 baseline。
"""

from __future__ import annotations

import csv
import importlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("outputs") / "mpe"
CSV_PATH = OUTPUT_DIR / "random_simple_spread.csv"
PNG_PATH = OUTPUT_DIR / "random_simple_spread.png"


def load_simple_spread():
    """优先使用 mpe2.simple_spread_v3，失败后回退到 pettingzoo.mpe。"""

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
                "无法从 mpe2 或 pettingzoo.mpe 导入 simple_spread_v3: "
                f"{mpe2_error}; {pettingzoo_error}"
            ) from pettingzoo_error


def all_agents_done(terminations: dict, truncations: dict) -> bool:
    """如果所有 agent 都 terminated 或 truncated，就提前结束 episode。"""

    agents = set(terminations) | set(truncations)
    if not agents:
        return False

    return all(
        terminations.get(agent, False) or truncations.get(agent, False)
        for agent in agents
    )


def mean_reward(rewards: dict) -> float:
    """simple_spread 通常是团队共享奖励，这里用均值作为 team reward。"""

    if not rewards:
        return 0.0
    return sum(float(reward) for reward in rewards.values()) / len(rewards)


def write_csv(rows: list[dict]) -> Path:
    """保存 episode 统计结果。

    为了不覆盖已有实验结果，如果默认文件已存在，就保存到带编号的新文件。
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = next_available_path(CSV_PATH)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("episode", "steps", "team_return"),
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


def write_plot(returns: list[float]) -> Path:
    """把每个 episode 的 return 画成曲线并保存。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = next_available_path(PNG_PATH)

    episodes = list(range(1, len(returns) + 1))
    plt.figure(figsize=(8, 4.5))
    plt.plot(episodes, returns, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Team return")
    plt.title("Random policy on MPE simple_spread_v3")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def next_available_path(path: Path) -> Path:
    """如果目标文件已存在，返回一个不会覆盖旧结果的新路径。"""

    if not path.exists():
        return path

    for index in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"无法找到可用输出文件名: {path}")


def main() -> None:
    simple_spread_v3, source = load_simple_spread()
    env = simple_spread_v3.parallel_env(render_mode=None, max_cycles=100)

    print(f"Loaded simple_spread_v3 from: {source}")

    episode_rows = []
    episode_returns = []

    for episode in range(1, 21):
        observations, infos = env.reset()
        del infos

        episode_return = 0.0
        step_count = 0

        for _ in range(100):
            if not observations:
                break

            # observations 是 dict，所以这里给每个 observations.keys() 里的 agent 随机动作。
            actions = {
                agent: env.action_space(agent).sample()
                for agent in observations.keys()
            }

            observations, rewards, terminations, truncations, infos = env.step(actions)
            del infos

            # simple_spread 多个 agent 的 reward 通常相同，用 mean 作为 team reward。
            episode_return += mean_reward(rewards)
            step_count += 1

            if all_agents_done(terminations, truncations):
                break

        episode_rows.append(
            {
                "episode": episode,
                "steps": step_count,
                "team_return": episode_return,
            }
        )
        episode_returns.append(episode_return)

        print(
            f"episode={episode}, steps={step_count}, "
            f"episode_team_return={episode_return:.3f}"
        )

    env.close()

    average_return = sum(episode_returns) / len(episode_returns)
    best_return = max(episode_returns)
    worst_return = min(episode_returns)

    print(f"average return: {average_return:.3f}")
    print(f"best return:    {best_return:.3f}")
    print(f"worst return:   {worst_return:.3f}")

    csv_path = write_csv(episode_rows)
    png_path = write_plot(episode_returns)

    print(f"CSV saved to: {csv_path}")
    print(f"Plot saved to: {png_path}")


if __name__ == "__main__":
    main()
