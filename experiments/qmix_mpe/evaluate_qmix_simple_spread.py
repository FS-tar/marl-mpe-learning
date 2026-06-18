# -*- coding: utf-8 -*-
"""Evaluate a simplified QMIX checkpoint on MPE simple_spread_v3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algorithms.qmix import AgentQNetwork
from envs.mpe_env_factory import get_mpe_env_source, make_mpe_env
from train_qmix_simple_spread import evaluate_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate simplified QMIX on simple_spread_v3.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def resolve_checkpoint(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_file():
        return path

    candidates = []
    if path.is_dir():
        candidates.append(path / "checkpoints" / "qmix_final.pt")
        candidates.extend(sorted((path / "checkpoints").glob("*.pt")))
        candidates.extend(sorted(path.glob("*.pt")))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find QMIX checkpoint under: {path}")


def load_q_networks(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = checkpoint["metadata"]
    q_networks = torch.nn.ModuleList(
        [
            AgentQNetwork(
                obs_dim=int(metadata["obs_dim"]),
                action_dim=int(metadata["action_dim"]),
                hidden_dim=int(metadata.get("hidden_dim", 128)),
            )
            for _ in metadata["agent_names"]
        ]
    ).to(device)

    for network, state_dict in zip(q_networks, checkpoint["q_networks"]):
        network.load_state_dict(state_dict)
    q_networks.eval()
    return q_networks, metadata


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

    avg_team_return, avg_mean_agent_return = evaluate_policy(
        q_networks=q_networks,
        agents=list(metadata["agent_names"]),
        obs_dim=int(metadata["obs_dim"]),
        action_dim=int(metadata["action_dim"]),
        max_cycles=max_cycles,
        eval_episodes=args.episodes,
        seed=args.seed,
        device=device,
    )

    print(f"Loaded simple_spread_v3 from: {source}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Average team return: {avg_team_return:.3f}")
    print(f"Average mean-agent return: {avg_mean_agent_return:.3f}")


if __name__ == "__main__":
    main()
