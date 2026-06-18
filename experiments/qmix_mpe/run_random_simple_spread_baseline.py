# -*- coding: utf-8 -*-
"""Run a random-action baseline for MPE simple_spread_v3."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from envs.mpe_env_factory import get_mpe_env_source, make_mpe_env


OUTPUT_ROOT = ROOT_DIR / "outputs" / "qmix_mpe" / "simple_spread"
OUTPUT_DIR = OUTPUT_ROOT / "random_baselines"
LATEST_JSON_PATH = OUTPUT_ROOT / "random_baseline_latest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random baseline on MPE simple_spread_v3.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-cycles", type=int, default=50)
    parser.add_argument("--team-reward-mode", choices=("mean", "sum", "first"), default="mean")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def reset_env(env, seed: int | None = None):
    try:
        if seed is None:
            return env.reset()
        return env.reset(seed=seed)
    except TypeError:
        return env.reset()


def team_reward(rewards: dict, mode: str = "mean") -> float:
    if not rewards:
        return 0.0
    values = [float(value) for value in rewards.values()]
    if mode == "mean":
        return float(np.mean(values))
    if mode == "sum":
        return float(np.sum(values))
    if mode == "first":
        return float(values[0])
    raise ValueError(f"Unknown team_reward_mode: {mode}")


def done_flag(next_observations: dict, terminations: dict, truncations: dict) -> bool:
    if not next_observations:
        return True
    agents = set(terminations) | set(truncations)
    return bool(agents) and all(
        bool(terminations.get(agent, False) or truncations.get(agent, False))
        for agent in agents
    )


def next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find available output path for {path}")


def baseline_stem(max_cycles: int, team_reward_mode: str) -> str:
    return f"random_baseline_cycles{max_cycles}_mode_{team_reward_mode}"


def main() -> None:
    args = parse_args()
    env = make_mpe_env("simple_spread_v3", max_cycles=args.max_cycles)
    source = get_mpe_env_source(env)

    rows = []
    returns = []
    mean_agent_returns = []
    episode_lengths = []
    for episode in range(1, args.episodes + 1):
        observations, infos = reset_env(env, seed=args.seed + episode)
        del infos
        episode_return = 0.0
        agent_returns = {agent: 0.0 for agent in env.possible_agents}
        steps = 0

        for _ in range(args.max_cycles):
            if not observations:
                break
            actions = {
                agent: env.action_space(agent).sample()
                for agent in observations.keys()
            }
            observations, rewards, terminations, truncations, infos = env.step(actions)
            del infos
            episode_return += team_reward(rewards, args.team_reward_mode)
            for agent in agent_returns:
                agent_returns[agent] += float(rewards.get(agent, 0.0))
            steps += 1
            if done_flag(observations, terminations, truncations):
                break

        rows.append(
            {
                "episode": episode,
                "steps": steps,
                "team_return": episode_return,
            }
        )
        returns.append(episode_return)
        mean_agent_returns.append(float(np.mean(list(agent_returns.values()))))
        episode_lengths.append(steps)
        print(f"episode={episode:04d} steps={steps} team_return={episode_return:.3f}")

    env.close()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = baseline_stem(args.max_cycles, args.team_reward_mode)
    csv_path = OUTPUT_DIR / f"{stem}.csv"
    tmp_path = csv_path.with_name(f".{csv_path.name}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["episode", "steps", "team_return"])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, csv_path)

    baseline = {
        "random_team_return": float(np.mean(returns)),
        "random_mean_agent_return": float(np.mean(mean_agent_returns)),
        "random_episode_length": float(np.mean(episode_lengths)),
        "episodes": int(args.episodes),
        "max_cycles": int(args.max_cycles),
        "team_reward_mode": args.team_reward_mode,
    }
    json_path = OUTPUT_DIR / f"{stem}.json"
    json_tmp_path = json_path.with_name(f".{json_path.name}.tmp")
    with json_tmp_path.open("w", encoding="utf-8") as file:
        json.dump(baseline, file, ensure_ascii=False, indent=2)
    os.replace(json_tmp_path, json_path)

    latest_tmp_path = LATEST_JSON_PATH.with_name(f".{LATEST_JSON_PATH.name}.tmp")
    with latest_tmp_path.open("w", encoding="utf-8") as file:
        json.dump({**baseline, "source": str(json_path)}, file, ensure_ascii=False, indent=2)
    os.replace(latest_tmp_path, LATEST_JSON_PATH)

    print(f"Loaded simple_spread_v3 from: {source}")
    print(f"Average team return: {baseline['random_team_return']:.3f}")
    print(f"CSV saved to: {csv_path}")
    print(f"JSON saved to: {json_path}")
    print(f"Latest JSON saved to: {LATEST_JSON_PATH}")


if __name__ == "__main__":
    main()
