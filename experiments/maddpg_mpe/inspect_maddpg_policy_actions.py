# -*- coding: utf-8 -*-
"""检查 MADDPG simple_tag checkpoint 的 deterministic 动作分布。

用法：
python experiments/maddpg_mpe/inspect_maddpg_policy_actions.py --run-dir outputs/maddpg_mpe/simple_tag/run006
"""

from __future__ import annotations

import argparse
import importlib
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algorithms.maddpg.networks import DiscreteActor


ADVERSARY_AGENTS = ["adversary_0", "adversary_1", "adversary_2"]
PREY_AGENT = "agent_0"
AGENT_NAMES = [*ADVERSARY_AGENTS, PREY_AGENT]
OBS_DIMS = {
    "adversary_0": 16,
    "adversary_1": 16,
    "adversary_2": 16,
    "agent_0": 14,
}
ACTION_DIM = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect deterministic action distributions of MADDPG simple_tag actors."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-cycles", type=int, default=100)
    parser.add_argument("--policy-id", type=int, default=0)
    parser.add_argument("--all-policy-ids", action="store_true")
    return parser.parse_args()


def load_simple_tag():
    try:
        return importlib.import_module("mpe2.simple_tag_v3"), "mpe2"
    except ImportError:
        return importlib.import_module("pettingzoo.mpe.simple_tag_v3"), "pettingzoo.mpe"


def reset_env(env, seed: int | None = None):
    try:
        if seed is None:
            return env.reset()
        return env.reset(seed=seed)
    except TypeError:
        return env.reset()


def latest_checkpoint(agent_dir: Path) -> Path:
    checkpoints = sorted(agent_dir.glob("*.pt"), key=checkpoint_sort_key)
    if not checkpoints:
        raise FileNotFoundError(f"找不到 checkpoint: {agent_dir}")
    return checkpoints[-1]


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"episode_(\d+)\.pt$", path.name)
    if match:
        return int(match.group(1)), path.name
    if path.name == "latest.pt":
        return 10**12, path.name
    return -1, path.name


def infer_hidden_dim(actor_state: dict[str, torch.Tensor]) -> int:
    first_weight = actor_state.get("net.0.weight")
    if first_weight is None:
        raise KeyError("checkpoint actor state 中找不到 net.0.weight，无法推断 hidden_dim")
    return int(first_weight.shape[0])


def load_actor_for_agent(
    checkpoint_root: Path,
    agent_name: str,
    device: torch.device,
    policy_id: int,
) -> tuple[DiscreteActor, Path, int, int]:
    checkpoint_path = latest_checkpoint(checkpoint_root / agent_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "actors" in checkpoint:
        actors_state = checkpoint["actors"]
        ensemble_size = int(checkpoint.get("ensemble_size", len(actors_state)))
        if policy_id < 0 or policy_id >= len(actors_state):
            raise ValueError(
                f"{agent_name} policy_id={policy_id} 超出范围 [0, {len(actors_state) - 1}]"
            )
        actor_state = actors_state[policy_id]
    elif "actor" in checkpoint:
        if policy_id != 0:
            raise ValueError(f"{agent_name} 是单 actor checkpoint，只支持 --policy-id 0")
        actor_state = checkpoint["actor"]
        ensemble_size = 1
    else:
        raise KeyError(f"{checkpoint_path} 中找不到 actor 或 actors 字段")

    hidden_dim = infer_hidden_dim(actor_state)
    actor = DiscreteActor(
        obs_dim=OBS_DIMS[agent_name],
        action_dim=ACTION_DIM,
        hidden_dim=hidden_dim,
    ).to(device)
    actor.load_state_dict(actor_state)
    actor.eval()
    return actor, checkpoint_path, ensemble_size, policy_id


def detect_ensemble_size(checkpoint_root: Path) -> int:
    ensemble_sizes = []
    for agent in AGENT_NAMES:
        checkpoint_path = latest_checkpoint(checkpoint_root / agent)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "actors" in checkpoint:
            ensemble_sizes.append(int(checkpoint.get("ensemble_size", len(checkpoint["actors"]))))
        elif "actor" in checkpoint:
            ensemble_sizes.append(1)
        else:
            raise KeyError(f"{checkpoint_path} 中找不到 actor 或 actors 字段")
    unique_sizes = sorted(set(ensemble_sizes))
    if len(unique_sizes) != 1:
        raise RuntimeError(f"各 agent 的 ensemble_size 不一致: {ensemble_sizes}")
    return unique_sizes[0]


def observations_to_arrays(observations: dict) -> dict[str, np.ndarray]:
    return {
        agent: np.asarray(observations[agent], dtype=np.float32)
        for agent in AGENT_NAMES
    }


@torch.no_grad()
def deterministic_actions(
    actors: dict[str, DiscreteActor],
    observations: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, int]:
    actions = {}
    for agent in AGENT_NAMES:
        obs_tensor = torch.as_tensor(
            observations[agent],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        logits = actors[agent](obs_tensor)
        actions[agent] = int(torch.argmax(logits, dim=-1).item())
    return actions


def done_all(next_observations: dict, terminations: dict, truncations: dict) -> bool:
    if not next_observations:
        return True
    return all(
        bool(terminations.get(agent, False) or truncations.get(agent, False))
        for agent in AGENT_NAMES
    )


def entropy_from_counts(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total == 0:
        return 0.0
    probs = counts.astype(np.float64) / total
    return float(-sum(p * math.log(p + 1e-12) for p in probs if p > 0.0))


def print_action_distribution(action_counts: dict[str, np.ndarray]) -> None:
    print("\n=== Deterministic action distribution ===")
    for agent in AGENT_NAMES:
        counts = action_counts[agent]
        total = int(counts.sum())
        entropy = entropy_from_counts(counts)
        print(f"\n{agent}: total_actions={total}, entropy={entropy:.6f}")
        for action in range(ACTION_DIM):
            percent = 0.0 if total == 0 else 100.0 * counts[action] / total
            print(f"  action {action}: count={int(counts[action])}, percent={percent:.2f}%")

        if total > 0:
            max_percent = 100.0 * counts.max() / total
            if max_percent >= 90.0:
                print(
                    "  warning: policy may collapse to a nearly constant action."
                )


def load_actors_for_policy_id(
    checkpoint_root: Path,
    device: torch.device,
    policy_id: int,
) -> dict[str, DiscreteActor]:
    actors = {}
    print("=== Loading actors ===")
    for agent in AGENT_NAMES:
        actor, checkpoint_path, ensemble_size, policy_id = load_actor_for_agent(
            checkpoint_root,
            agent,
            device,
            policy_id=policy_id,
        )
        actors[agent] = actor
        print(
            f"{agent}: {checkpoint_path} "
            f"(ensemble_size={ensemble_size}, inspected_policy_id={policy_id})"
        )
        if ensemble_size > 1:
            print(
                f"  note: ensemble checkpoint detected; inspecting actor[{policy_id}]."
            )
    return actors


def inspect_policy_id(
    args: argparse.Namespace,
    checkpoint_root: Path,
    device: torch.device,
    policy_id: int,
) -> None:
    print(f"\n================ policy_id={policy_id} ================")
    actors = load_actors_for_policy_id(
        checkpoint_root=checkpoint_root,
        device=device,
        policy_id=policy_id,
    )

    simple_tag_v3, source = load_simple_tag()
    env = simple_tag_v3.parallel_env(max_cycles=args.max_cycles)
    print(f"\nLoaded simple_tag_v3 from: {source}")

    action_counts = {
        agent: np.zeros(ACTION_DIM, dtype=np.int64)
        for agent in AGENT_NAMES
    }

    for episode in range(1, args.episodes + 1):
        observations, infos = reset_env(env, seed=episode)
        del infos
        episode_returns = {agent: 0.0 for agent in AGENT_NAMES}

        for _ in range(args.max_cycles):
            if not observations:
                break
            obs_arrays = observations_to_arrays(observations)
            actions = deterministic_actions(actors, obs_arrays, device)
            for agent, action in actions.items():
                action_counts[agent][action] += 1

            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            del infos
            for agent in AGENT_NAMES:
                episode_returns[agent] += float(rewards.get(agent, 0.0))

            if done_all(next_observations, terminations, truncations):
                break
            observations = next_observations

        adversary_team_return = sum(
            episode_returns[agent]
            for agent in ADVERSARY_AGENTS
        )
        prey_return = episode_returns[PREY_AGENT]
        print(
            f"episode={episode:03d} "
            f"adversary_team_return={adversary_team_return:.3f} "
            f"prey_return={prey_return:.3f}"
        )

    env.close()
    print_action_distribution(action_counts)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_root = args.run_dir / "checkpoints"
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"找不到 checkpoints 目录: {checkpoint_root}")

    ensemble_size = detect_ensemble_size(checkpoint_root)
    if args.all_policy_ids:
        policy_ids = list(range(ensemble_size))
    else:
        policy_ids = [args.policy_id]

    for policy_id in policy_ids:
        inspect_policy_id(
            args=args,
            checkpoint_root=checkpoint_root,
            device=device,
            policy_id=policy_id,
        )


if __name__ == "__main__":
    main()
