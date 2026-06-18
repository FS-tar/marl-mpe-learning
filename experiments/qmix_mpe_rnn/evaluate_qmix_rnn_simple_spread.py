# -*- coding: utf-8 -*-
"""Evaluate a recurrent QMIX checkpoint on MPE simple_spread_v3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algorithms.qmix_rnn import RNNAgent
from envs.mpe_env_factory import get_mpe_env_source, make_mpe_env
from experiments.qmix_mpe_rnn.train_qmix_rnn_simple_spread import evaluate_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RNN QMIX on simple_spread_v3.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--checkpoint-type", choices=("best", "final"), default="best")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def resolve_checkpoint(path_text: str, checkpoint_type: str = "best") -> tuple[Path, str]:
    path = Path(path_text)
    if path.is_file():
        return path, "explicit"
    if path.is_dir():
        checkpoint_dir = path / "checkpoints"
        best_path = checkpoint_dir / "qmix_rnn_best.pt"
        final_path = checkpoint_dir / "qmix_rnn_final.pt"
        if checkpoint_type == "best":
            if best_path.is_file():
                return best_path, "best"
            if final_path.is_file():
                print(
                    f"warning: best checkpoint not found at {best_path}; "
                    f"falling back to final checkpoint."
                )
                return final_path, "final"
            raise FileNotFoundError(
                f"Cannot find best or final RNN QMIX checkpoint under: {checkpoint_dir}"
            )
        if final_path.is_file():
            return final_path, "final"
        raise FileNotFoundError(f"Cannot find final RNN QMIX checkpoint: {final_path}")
    raise FileNotFoundError(f"Cannot find RNN QMIX checkpoint under: {path}")


def load_agent(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = checkpoint["metadata"]
    agent = RNNAgent(
        obs_dim=int(metadata["obs_dim"]),
        action_dim=int(metadata["action_dim"]),
        hidden_dim=int(metadata.get("rnn_hidden_dim", 64)),
        input_dim=int(metadata.get("rnn_input_dim", metadata["obs_dim"])),
    ).to(device)
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()
    return agent, metadata


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path, loaded_checkpoint_type = resolve_checkpoint(
        args.checkpoint,
        args.checkpoint_type,
    )
    agent, metadata = load_agent(checkpoint_path, device)
    env = make_mpe_env("simple_spread_v3", max_cycles=1)
    source = get_mpe_env_source(env)
    env.close()

    max_cycles = args.max_cycles
    if max_cycles is None:
        max_cycles = int(metadata.get("max_cycles", 100))
    team_reward_mode = str(metadata.get("team_reward_mode", "mean"))
    include_last_action = bool(metadata.get("include_last_action", False))
    include_agent_id = bool(metadata.get("include_agent_id", False))

    avg_team_return, avg_mean_agent_return = evaluate_policy(
        agent=agent,
        agents=list(metadata["agent_names"]),
        obs_dim=int(metadata["obs_dim"]),
        action_dim=int(metadata["action_dim"]),
        max_cycles=max_cycles,
        eval_episodes=args.episodes,
        seed=args.seed,
        device=device,
        epsilon_eval=0.0,
        team_reward_mode=team_reward_mode,
        include_last_action=include_last_action,
        include_agent_id=include_agent_id,
    )

    print(f"Loaded simple_spread_v3 from: {source}")
    print(f"Loaded checkpoint type: {loaded_checkpoint_type}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Average team return: {avg_team_return:.3f}")
    print(f"Average mean-agent return: {avg_mean_agent_return:.3f}")


if __name__ == "__main__":
    main()
