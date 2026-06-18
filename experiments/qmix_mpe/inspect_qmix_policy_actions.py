# -*- coding: utf-8 -*-
"""Inspect deterministic action distributions for a simplified QMIX checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluate_qmix_simple_spread import load_q_networks, resolve_checkpoint
from envs.mpe_env_factory import get_mpe_env_source, make_mpe_env
from train_qmix_simple_spread import (
    inspect_action_distribution,
    policy_collapse_risk,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect simplified QMIX policy actions.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=456)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    q_networks, metadata = load_q_networks(checkpoint_path, device)
    env = make_mpe_env("simple_spread_v3", max_cycles=1)
    source = get_mpe_env_source(env)
    env.close()

    max_cycles = args.max_cycles
    if max_cycles is None:
        max_cycles = int(metadata.get("max_cycles", 100))

    counts = inspect_action_distribution(
        q_networks=q_networks,
        agents=list(metadata["agent_names"]),
        obs_dim=int(metadata["obs_dim"]),
        action_dim=int(metadata["action_dim"]),
        max_cycles=max_cycles,
        episodes=args.episodes,
        seed=args.seed,
        device=device,
    )

    print(f"Loaded simple_spread_v3 from: {source}")
    print(f"Checkpoint: {checkpoint_path}")
    for agent, agent_counts in counts.items():
        total = sum(agent_counts)
        if total == 0:
            percentages = ["0.0%" for _ in agent_counts]
        else:
            percentages = [f"{count / total * 100:.1f}%" for count in agent_counts]
        print(f"{agent}: counts={agent_counts}, distribution={percentages}")
    print(f"Policy collapse risk: {policy_collapse_risk(counts)}")


if __name__ == "__main__":
    main()
