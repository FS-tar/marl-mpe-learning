# -*- coding: utf-8 -*-
"""Diagnostic checks for a recurrent QMIX simple_spread run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algorithms.qmix_rnn import RNNAgent
from envs.mpe_env_factory import get_mpe_env_source, make_mpe_env
from experiments.qmix_mpe_rnn.train_qmix_rnn_simple_spread import (
    action_coverage_stats,
    build_step_inputs,
    build_mixer,
    done_flag,
    evaluate_policy,
    make_state,
    obs_to_array,
    one_hot_np,
    reset_env,
    select_actions_from_q,
)

RANDOM_BASELINE_DIR = ROOT_DIR / "outputs" / "qmix_mpe" / "simple_spread" / "random_baselines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug a RNN QMIX simple_spread run.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--checkpoint-type", choices=("best", "final"), default="best")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--epsilon-eval", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


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


def finite_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def baseline_path(max_cycles: int, team_reward_mode: str) -> Path:
    return RANDOM_BASELINE_DIR / f"random_baseline_cycles{max_cycles}_mode_{team_reward_mode}.json"


def print_matching_baseline(config: dict) -> None:
    max_cycles = config.get("max_cycles")
    team_reward_mode = str(config.get("team_reward_mode", "mean"))
    print("\nRandom baseline:")
    if max_cycles is None:
        print("baseline source: not found")
        print("warning: max_cycles missing in config.json")
        return
    path = baseline_path(int(max_cycles), team_reward_mode)
    if not path.is_file():
        print("baseline source: not found")
        print(f"expected: {path}")
        return
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if data.get("max_cycles") != int(max_cycles) or data.get("team_reward_mode") != team_reward_mode:
        print("baseline source: not found")
        print(f"warning: metadata mismatch in {path}")
        return
    print(f"baseline source: {path}")
    print(f"random_team_return: {data.get('random_team_return')}")
    print(f"baseline max_cycles: {data.get('max_cycles')}")
    print(f"baseline team_reward_mode: {data.get('team_reward_mode')}")


def load_qmix_rnn_checkpoint(checkpoint_path: Path, device: torch.device, config: dict):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = checkpoint["metadata"]
    agents = list(metadata["agent_names"])
    obs_dim = int(metadata["obs_dim"])
    action_dim = int(metadata["action_dim"])
    state_dim = int(metadata.get("state_dim", len(agents) * obs_dim))
    rnn_hidden_dim = int(metadata.get("rnn_hidden_dim", config.get("rnn_hidden_dim", 64)))
    rnn_input_dim = int(metadata.get("rnn_input_dim", config.get("rnn_input_dim", obs_dim)))
    mixer_hidden_dim = int(metadata.get("hidden_dim", config.get("hidden_dim", 32)))
    mixer_type = str(config.get("mixer_type", metadata.get("mixer_type", "qmix")))

    agent = RNNAgent(obs_dim, action_dim, rnn_hidden_dim, input_dim=rnn_input_dim).to(device)
    mixer = build_mixer(
        mixer_type=mixer_type,
        n_agents=len(agents),
        state_dim=state_dim,
        hidden_dim=mixer_hidden_dim,
        use_layer_norm=bool(config.get("use_layer_norm", metadata.get("use_layer_norm", True))),
        mixer_weight_clip=float(config.get("mixer_weight_clip", metadata.get("mixer_weight_clip", 1.0))),
        mixer_bias_clip=float(config.get("mixer_bias_clip", metadata.get("mixer_bias_clip", 5.0))),
        mixer_weight_activation=str(
            config.get(
                "mixer_weight_activation",
                metadata.get("mixer_weight_activation", "softplus"),
            )
        ),
    ).to(device)
    agent.load_state_dict(checkpoint["agent"])
    mixer.load_state_dict(checkpoint["mixer"], strict=False)
    agent.eval()
    mixer.eval()
    return agent, mixer, metadata


def entropy_from_counts(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability + 1e-12)
    return float(entropy)


def collect_policy_stats(
    agent: RNNAgent,
    mixer: torch.nn.Module,
    agents: list[str],
    obs_dim: int,
    action_dim: int,
    episodes: int,
    max_cycles: int,
    seed: int,
    device: torch.device,
    epsilon_eval: float = 0.0,
    include_last_action: bool = False,
    include_agent_id: bool = False,
):
    env = make_mpe_env("simple_spread_v3", max_cycles=max_cycles)
    action_counts = {name: [0 for _ in range(action_dim)] for name in agents}
    q_values_by_agent = {name: [] for name in agents}
    q_tot_values = []

    for episode in range(episodes):
        observations, infos = reset_env(env, seed=seed + episode)
        del infos
        hidden = agent.init_hidden(len(agents), device)
        last_actions = np.zeros((len(agents), action_dim), dtype=np.float32)
        for _ in range(max_cycles):
            if not observations:
                break
            obs_array = obs_to_array(observations, agents, obs_dim)
            obs_tensor = build_step_inputs(
                obs_array,
                last_actions,
                include_last_action,
                include_agent_id,
                device,
            )
            with torch.no_grad():
                q_values, hidden = agent(obs_tensor, hidden)
                actions = select_actions_from_q(q_values, epsilon_eval, action_dim)
                last_actions = one_hot_np(actions, action_dim)
                chosen_qs = q_values.gather(
                    1,
                    torch.as_tensor(actions, dtype=torch.long, device=device).view(-1, 1),
                ).view(1, len(agents))
                state_tensor = torch.as_tensor(
                    make_state(obs_array),
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                q_tot = mixer(chosen_qs, state_tensor)
                q_tot_values.append(float(q_tot.item()))

            for index, name in enumerate(agents):
                q_values_by_agent[name].extend(q_values[index].detach().cpu().numpy().tolist())
            for name, action in zip(agents, actions):
                action_counts[name][int(action)] += 1

            action_dict = {
                name: int(action)
                for name, action in zip(agents, actions)
                if name in observations
            }
            observations, rewards, terminations, truncations, infos = env.step(action_dict)
            del rewards, infos
            if done_flag(observations, terminations, truncations, agents):
                break

    env.close()
    return action_counts, q_values_by_agent, q_tot_values


def print_last_log_stats(rows: list[dict]) -> None:
    eval_rows = [row for row in rows if finite_float(row.get("eval_team_return")) is not None]
    print("\nLast eval_team_return values:")
    for row in eval_rows[-5:]:
        print(f"episode={row.get('episode')} eval_team_return={row.get('eval_team_return')}")
    if rows:
        last = rows[-1]
        print("\nLast train_log values:")
        for key in ("td_loss", "epsilon", "q_tot_mean", "target_q_tot_mean", "grad_norm"):
            print(f"{key}: {last.get(key)}")


def log_eval_summary(rows: list[dict]) -> dict:
    eval_points = []
    for row in rows:
        value = finite_float(row.get("eval_team_return"))
        if value is None:
            continue
        episode_value = finite_float(row.get("episode"))
        episode = None if episode_value is None else int(episode_value)
        eval_points.append((episode, value))
    if not eval_points:
        return {
            "best_eval_episode": None,
            "best_eval_team_return": None,
            "final_eval_team_return": None,
        }
    best_episode, best_value = max(eval_points, key=lambda item: item[1])
    final_episode, final_value = eval_points[-1]
    del final_episode
    return {
        "best_eval_episode": best_episode,
        "best_eval_team_return": best_value,
        "final_eval_team_return": final_value,
    }


def print_action_stats(action_counts: dict[str, list[int]]) -> str:
    coverage_stats = action_coverage_stats(action_counts)
    print("\nAction distribution:")
    for agent_name, stats in coverage_stats["per_agent"].items():
        counts = stats["counts"]
        total = sum(counts)
        ratios = [0.0 if total == 0 else count / total for count in counts]
        ratio_text = [f"{ratio:.3f}" for ratio in ratios]
        print(
            f"{agent_name}: counts={counts}, ratios={ratio_text}, "
            f"max_action_ratio={stats['max_action_ratio']:.3f}, "
            f"top2_action_ratio={stats['top2_action_ratio']:.3f}, "
            f"used_action_count={stats['used_action_count']}, "
            f"action_entropy={stats['action_entropy']:.4f}, "
            f"collapse_reason={stats['collapse_reason']}"
        )
    collapse = str(coverage_stats["policy_collapse_risk"])
    print(f"collapse_reason: {coverage_stats['collapse_reason']}")
    print(f"policy collapse risk: {collapse}")
    print(f"obvious collapse: {'yes' if collapse != 'no obvious collapse' else 'no'}")
    return collapse


def print_value_stats(q_values_by_agent: dict[str, list[float]], q_tot_values: list[float]) -> None:
    print("\nAgent Q statistics:")
    for agent_name, values in q_values_by_agent.items():
        array = np.asarray(values, dtype=np.float32)
        if array.size == 0:
            print(f"{agent_name}: no Q samples")
            continue
        print(
            f"{agent_name}: q_mean={array.mean():.4f}, q_min={array.min():.4f}, "
            f"q_max={array.max():.4f}, q_std={array.std():.4f}"
        )

    q_tot = np.asarray(q_tot_values, dtype=np.float32)
    print("\nq_tot statistics:")
    if q_tot.size == 0:
        print("no q_tot samples")
    else:
        print(
            f"q_tot_mean={q_tot.mean():.4f}, q_tot_min={q_tot.min():.4f}, "
            f"q_tot_max={q_tot.max():.4f}, q_tot_std={q_tot.std():.4f}"
        )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.txt"
    train_log_path = run_dir / "train_log.csv"
    config = {}
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    rows = read_csv_rows(train_log_path)
    summary_text = read_text(summary_path)

    print(f"Run directory: {run_dir}")
    print(f"Config path: {config_path} exists={config_path.is_file()}")
    print(f"Summary path: {summary_path} exists={summary_path.is_file()}")
    print(f"Train log path: {train_log_path} exists={train_log_path.is_file()}")
    if summary_text:
        print("\nSummary:")
        print(summary_text.strip())

    print_last_log_stats(rows)
    print_matching_baseline(config)

    eval_summary = log_eval_summary(rows)
    checkpoint_path, loaded_checkpoint_type = resolve_checkpoint(
        str(run_dir),
        args.checkpoint_type,
    )
    agent, mixer, metadata = load_qmix_rnn_checkpoint(checkpoint_path, device, config)
    agents = list(metadata["agent_names"])
    obs_dim = int(metadata["obs_dim"])
    action_dim = int(metadata["action_dim"])
    max_cycles = args.max_cycles
    if max_cycles is None:
        max_cycles = int(config.get("max_cycles", metadata.get("max_cycles", 100)))
    team_reward_mode = str(config.get("team_reward_mode", "mean"))
    mixer_type = str(config.get("mixer_type", metadata.get("mixer_type", "qmix")))
    double_q = bool(config.get("double_q", metadata.get("double_q", False)))
    include_last_action = bool(config.get("include_last_action", metadata.get("include_last_action", False)))
    include_agent_id = bool(config.get("include_agent_id", metadata.get("include_agent_id", False)))
    rnn_input_dim = int(config.get("rnn_input_dim", metadata.get("rnn_input_dim", obs_dim)))

    print("\nMixer config:")
    print(f"loaded checkpoint type: {loaded_checkpoint_type}")
    print(f"loaded checkpoint path: {checkpoint_path}")
    print(f"best eval episode: {eval_summary['best_eval_episode']}")
    print(f"best eval_team_return: {eval_summary['best_eval_team_return']}")
    print(f"final eval_team_return: {eval_summary['final_eval_team_return']}")
    print(f"mixer_type: {mixer_type}")
    print(f"double_q: {double_q}")
    print(f"include_last_action: {include_last_action}")
    print(f"include_agent_id: {include_agent_id}")
    print(f"rnn_input_dim: {rnn_input_dim}")
    if mixer_type == "qmix":
        print(f"use_layer_norm: {config.get('use_layer_norm', metadata.get('use_layer_norm', True))}")
        print(f"mixer_weight_clip: {config.get('mixer_weight_clip', metadata.get('mixer_weight_clip', 1.0))}")
        print(f"mixer_bias_clip: {config.get('mixer_bias_clip', metadata.get('mixer_bias_clip', 5.0))}")
        print(
            "mixer_weight_activation: "
            f"{config.get('mixer_weight_activation', metadata.get('mixer_weight_activation', 'softplus'))}"
        )

    eval_team_return, eval_mean_agent_return = evaluate_policy(
        agent=agent,
        agents=agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_cycles=max_cycles,
        eval_episodes=args.episodes,
        seed=args.seed + 1000,
        device=device,
        epsilon_eval=args.epsilon_eval,
        team_reward_mode=team_reward_mode,
        include_last_action=include_last_action,
        include_agent_id=include_agent_id,
    )
    label = "Deterministic eval" if args.epsilon_eval == 0.0 else f"Epsilon eval (epsilon={args.epsilon_eval})"
    print(f"\n{label}:")
    print(f"eval_team_return={eval_team_return:.3f}")
    print(f"eval_mean_agent_return={eval_mean_agent_return:.3f}")

    action_counts, q_values_by_agent, q_tot_values = collect_policy_stats(
        agent=agent,
        mixer=mixer,
        agents=agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        episodes=args.episodes,
        max_cycles=max_cycles,
        seed=args.seed + 2000,
        device=device,
        epsilon_eval=args.epsilon_eval,
        include_last_action=include_last_action,
        include_agent_id=include_agent_id,
    )
    collapse = print_action_stats(action_counts)
    print_value_stats(q_values_by_agent, q_tot_values)

    env = make_mpe_env("simple_spread_v3", max_cycles=1)
    print(f"\nEnvironment source: {get_mpe_env_source(env)}")
    env.close()
    print(f"Loaded checkpoint type: {loaded_checkpoint_type}")
    print(f"Loaded checkpoint path: {checkpoint_path}")
    print(f"Obvious collapse: {'yes' if collapse != 'no obvious collapse' else 'no'}")


if __name__ == "__main__":
    main()
