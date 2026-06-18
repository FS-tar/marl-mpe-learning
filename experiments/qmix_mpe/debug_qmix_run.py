# -*- coding: utf-8 -*-
"""Diagnostic checks for a simplified QMIX simple_spread run."""

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

from algorithms.qmix import AgentQNetwork
from algorithms.qmix.replay_buffer import ReplayBuffer
from envs.mpe_env_factory import get_mpe_env_source, make_mpe_env
from experiments.qmix_mpe.train_qmix_simple_spread import (
    done_flag,
    evaluate_policy,
    build_mixer,
    make_state,
    obs_to_array,
    reset_env,
    select_actions,
    team_reward,
)

OUTPUT_BASE_DIR = ROOT_DIR / "outputs" / "qmix_mpe" / "simple_spread"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug a QMIX simple_spread run.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-cycles", type=int, default=100)
    parser.add_argument("--debug-shapes", action="store_true")
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
    return (
        OUTPUT_BASE_DIR
        / "random_baselines"
        / f"random_baseline_cycles{max_cycles}_mode_{team_reward_mode}.json"
    )


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


def load_qmix_checkpoint(checkpoint_path: Path, device: torch.device, config: dict | None = None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = config or {}
    metadata = checkpoint["metadata"]
    agents = list(metadata["agent_names"])
    obs_dim = int(metadata["obs_dim"])
    action_dim = int(metadata["action_dim"])
    hidden_dim = int(metadata.get("hidden_dim", 128))
    state_dim = int(metadata.get("state_dim", len(agents) * obs_dim))
    mixer_type = str(config.get("mixer_type", metadata.get("mixer_type", "qmix")))

    q_networks = torch.nn.ModuleList(
        [AgentQNetwork(obs_dim, action_dim, hidden_dim) for _ in agents]
    ).to(device)
    mixer = build_mixer(
        mixer_type=mixer_type,
        n_agents=len(agents),
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        use_layer_norm=bool(config.get("use_layer_norm", metadata.get("use_layer_norm", True))),
        mixer_weight_clip=float(config.get("mixer_weight_clip", metadata.get("mixer_weight_clip", 10.0))),
        mixer_bias_clip=float(config.get("mixer_bias_clip", metadata.get("mixer_bias_clip", 5.0))),
        mixer_weight_activation=str(
            config.get(
                "mixer_weight_activation",
                metadata.get("mixer_weight_activation", "softplus"),
            )
        ),
    ).to(device)

    for network, state_dict in zip(q_networks, checkpoint["q_networks"]):
        network.load_state_dict(state_dict)
    mixer.load_state_dict(checkpoint["mixer"], strict=False)
    q_networks.eval()
    mixer.eval()
    return q_networks, mixer, metadata


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
    q_networks: torch.nn.ModuleList,
    mixer: torch.nn.Module,
    agents: list[str],
    obs_dim: int,
    action_dim: int,
    episodes: int,
    max_cycles: int,
    seed: int,
    device: torch.device,
):
    env = make_mpe_env("simple_spread_v3", max_cycles=max_cycles)
    action_counts = {agent: [0 for _ in range(action_dim)] for agent in agents}
    q_values_by_agent = {agent: [] for agent in agents}
    q_tot_values = []

    for episode in range(episodes):
        observations, infos = reset_env(env, seed=seed + episode)
        del infos
        for _ in range(max_cycles):
            if not observations:
                break
            obs_array = obs_to_array(observations, agents, obs_dim)
            actions = select_actions(q_networks, obs_array, 0.0, action_dim, device)
            action_dict = {
                agent: int(action)
                for agent, action in zip(agents, actions)
                if agent in observations
            }

            chosen_qs = []
            with torch.no_grad():
                for agent_index, q_net in enumerate(q_networks):
                    obs_tensor = torch.as_tensor(
                        obs_array[agent_index],
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)
                    q_values = q_net(obs_tensor).squeeze(0)
                    q_values_by_agent[agents[agent_index]].extend(
                        q_values.detach().cpu().numpy().tolist()
                    )
                    chosen_qs.append(q_values[int(actions[agent_index])])

                state_tensor = torch.as_tensor(
                    make_state(obs_array),
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                agent_qs = torch.stack(chosen_qs).view(1, len(agents))
                q_tot = mixer(agent_qs, state_tensor)
                q_tot_values.append(float(q_tot.item()))

            for agent, action in zip(agents, actions):
                action_counts[agent][int(action)] += 1

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


def print_action_stats(action_counts: dict[str, list[int]]) -> str:
    collapse = "no obvious collapse"
    print("\nAction distribution:")
    for agent, counts in action_counts.items():
        total = sum(counts)
        ratios = [0.0 if total == 0 else count / total for count in counts]
        max_ratio = max(ratios) if ratios else 0.0
        entropy = entropy_from_counts(counts)
        if max_ratio > 0.95:
            collapse = "high collapse risk"
        ratio_text = [f"{ratio:.3f}" for ratio in ratios]
        print(
            f"{agent}: counts={counts}, ratios={ratio_text}, "
            f"action_entropy={entropy:.4f}, max_action_ratio={max_ratio:.3f}"
        )
    print(collapse)
    return collapse


def print_value_stats(q_values_by_agent: dict[str, list[float]], q_tot_values: list[float]) -> None:
    print("\nAgent Q statistics:")
    for agent, values in q_values_by_agent.items():
        array = np.asarray(values, dtype=np.float32)
        if array.size == 0:
            print(f"{agent}: no Q samples")
            continue
        print(
            f"{agent}: q_mean={array.mean():.4f}, q_min={array.min():.4f}, "
            f"q_max={array.max():.4f}, q_std={array.std():.4f}"
        )

    q_tot = np.asarray(q_tot_values, dtype=np.float32)
    print("\nMixer q_tot statistics:")
    if q_tot.size == 0:
        print("no q_tot samples")
    else:
        print(
            f"q_tot_mean={q_tot.mean():.4f}, q_tot_min={q_tot.min():.4f}, "
            f"q_tot_max={q_tot.max():.4f}, q_tot_std={q_tot.std():.4f}"
        )
        all_q_values = np.concatenate(
            [
                np.asarray(values, dtype=np.float32)
                for values in q_values_by_agent.values()
                if values
            ]
        )
        return all_q_values, q_tot
    return np.asarray([], dtype=np.float32), q_tot


def collect_random_transition(env, agents: list[str], obs_dim: int, team_reward_mode: str, reward_scale: float):
    observations, infos = reset_env(env)
    del infos
    obs_array = obs_to_array(observations, agents, obs_dim)
    state = make_state(obs_array)
    actions = np.asarray(
        [env.action_space(agent).sample() for agent in agents],
        dtype=np.int64,
    )
    action_dict = {
        agent: int(action)
        for agent, action in zip(agents, actions)
        if agent in observations
    }
    next_observations, rewards, terminations, truncations, infos = env.step(action_dict)
    del infos
    reward = team_reward(rewards, team_reward_mode) * reward_scale
    done = done_flag(next_observations, terminations, truncations, agents)
    if all(agent in next_observations for agent in agents):
        next_obs_array = obs_to_array(next_observations, agents, obs_dim)
    else:
        next_obs_array = obs_array.copy()
    return obs_array, state, actions, reward, next_obs_array, make_state(next_obs_array), done


def print_debug_shapes(
    q_networks: torch.nn.ModuleList,
    mixer: torch.nn.Module,
    agents: list[str],
    obs_dim: int,
    action_dim: int,
    max_cycles: int,
    team_reward_mode: str,
    reward_scale: float,
    device: torch.device,
) -> None:
    del action_dim
    batch_size = 4
    state_dim = len(agents) * obs_dim
    buffer = ReplayBuffer(batch_size, len(agents), obs_dim, state_dim)
    env = make_mpe_env("simple_spread_v3", max_cycles=max_cycles)
    for _ in range(batch_size):
        obs, state, actions, reward, next_obs, next_state, done = collect_random_transition(
            env,
            agents,
            obs_dim,
            team_reward_mode,
            reward_scale,
        )
        buffer.add(obs, state, actions, reward, next_obs, next_state, done)
    env.close()

    batch = buffer.sample(batch_size, device)
    obs = batch["obs"]
    actions = batch["actions"].long()
    rewards = batch["reward"].view(batch_size, 1)
    dones = batch["done"].view(batch_size, 1)
    states = batch["state"]
    next_obs = batch["next_obs"]
    next_states = batch["next_state"]

    gathered_qs = []
    next_qs = []
    with torch.no_grad():
        agent_q_tensors = []
        for agent_index, q_net in enumerate(q_networks):
            q_values = q_net(obs[:, agent_index, :])
            agent_q_tensors.append(q_values)
            gathered_qs.append(q_values.gather(1, actions[:, agent_index].unsqueeze(1)).squeeze(1))
            next_qs.append(q_net(next_obs[:, agent_index, :]).max(dim=1).values)
        gathered_qs_tensor = torch.stack(gathered_qs, dim=1)
        current_q_tot = mixer(gathered_qs_tensor, states)
        target_q_tot = mixer(torch.stack(next_qs, dim=1), next_states)
        td_target = rewards + 0.99 * (1.0 - dones) * target_q_tot

    print("\nDebug shapes from simulated batch:")
    for name, tensor in (
        ("obs", obs),
        ("actions", actions),
        ("rewards", rewards),
        ("dones", dones),
        ("agent_qs", torch.stack(agent_q_tensors, dim=1)),
        ("gathered_qs", gathered_qs_tensor),
        ("current_q_tot", current_q_tot),
        ("target_q_tot", target_q_tot),
        ("td_target", td_target),
    ):
        print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")


def print_reward_and_done_check(max_cycles: int, seed: int, team_reward_mode: str, reward_scale: float) -> None:
    env = make_mpe_env("simple_spread_v3", max_cycles=max_cycles)
    observations, infos = reset_env(env, seed=seed)
    del infos
    actions = {
        agent: env.action_space(agent).sample()
        for agent in observations.keys()
    }
    next_observations, rewards, terminations, truncations, infos = env.step(actions)
    del infos
    raw_team_reward = team_reward(rewards, team_reward_mode)
    print("\nReward handling check:")
    print(f"raw rewards dict: {rewards}")
    print(f"team_reward_mode: {team_reward_mode}")
    print("team_reward code: mean / sum / first selected by --team-reward-mode; default is mean")
    print(f"team_reward: {raw_team_reward}")
    print(f"reward_scale: {reward_scale}")
    print(f"scaled_reward: {raw_team_reward * reward_scale}")

    agents = list(env.possible_agents)
    done = done_flag(next_observations, terminations, truncations, agents)
    print("\nDone mask check:")
    print("done logic: done=True when next_observations is empty or all agents are terminated/truncated")
    print(f"terminations: {terminations}")
    print(f"truncations: {truncations}")
    print(f"computed done: {done}")
    print("TD target code uses: reward + gamma * (1 - done) * target_q_tot")
    env.close()


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
    summary_text = read_text(summary_path)
    rows = read_csv_rows(train_log_path)

    print(f"Run directory: {run_dir}")
    print(f"Config path: {config_path} exists={config_path.is_file()}")
    print(f"Summary path: {summary_path} exists={summary_path.is_file()}")
    print(f"Train log path: {train_log_path} exists={train_log_path.is_file()}")
    if summary_text:
        print("\nSummary:")
        print(summary_text.strip())

    print_last_log_stats(rows)
    print_matching_baseline(config)

    checkpoint_path = resolve_checkpoint(str(run_dir))
    q_networks, mixer, metadata = load_qmix_checkpoint(checkpoint_path, device, config=config)
    agents = list(metadata["agent_names"])
    obs_dim = int(metadata["obs_dim"])
    action_dim = int(metadata["action_dim"])
    team_reward_mode = str(config.get("team_reward_mode", "mean"))
    reward_scale = float(config.get("reward_scale", 1.0))
    mixer_type = str(config.get("mixer_type", metadata.get("mixer_type", "qmix")))
    double_q = bool(config.get("double_q", metadata.get("double_q", False)))
    qtot_l2_coef = float(config.get("qtot_l2_coef", 0.0))

    print("\nMixer config:")
    print(f"mixer_type: {mixer_type}")
    print(f"double_q: {double_q}")
    print(f"qtot_l2_coef: {qtot_l2_coef}")
    if mixer_type == "qmix":
        print(f"use_layer_norm: {config.get('use_layer_norm', metadata.get('use_layer_norm', True))}")
        print(f"mixer_weight_clip: {config.get('mixer_weight_clip', metadata.get('mixer_weight_clip', 10.0))}")
        print(f"mixer_bias_clip: {config.get('mixer_bias_clip', metadata.get('mixer_bias_clip', 5.0))}")
        print(
            "mixer_weight_activation: "
            f"{config.get('mixer_weight_activation', metadata.get('mixer_weight_activation', 'softplus'))}"
        )

    eval_team_return, eval_mean_agent_return = evaluate_policy(
        q_networks=q_networks,
        agents=agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_cycles=args.max_cycles,
        eval_episodes=args.episodes,
        seed=args.seed + 1000,
        device=device,
        epsilon_eval=0.0,
        team_reward_mode=team_reward_mode,
    )
    print("\nDeterministic eval:")
    print(f"eval_team_return={eval_team_return:.3f}")
    print(f"eval_mean_agent_return={eval_mean_agent_return:.3f}")

    action_counts, q_values_by_agent, q_tot_values = collect_policy_stats(
        q_networks=q_networks,
        mixer=mixer,
        agents=agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        episodes=args.episodes,
        max_cycles=args.max_cycles,
        seed=args.seed + 2000,
        device=device,
    )
    print_action_stats(action_counts)
    all_q_values, q_tot_values_array = print_value_stats(q_values_by_agent, q_tot_values)
    if mixer_type == "qmix":
        print("\nScale interpretation:")
        print(
            "current_q_tot / target_q_tot should usually stay on the same rough "
            "scale as the discounted scaled team reward and mixed individual Q values."
        )
        if (
            all_q_values.size > 0
            and q_tot_values_array.size > 0
            and np.max(np.abs(all_q_values)) <= 3.0
            and np.max(np.abs(q_tot_values_array)) > 100.0
        ):
            print("warning: mixer output scale is much larger than individual Q scale")
    if args.debug_shapes:
        print_debug_shapes(
            q_networks=q_networks,
            mixer=mixer,
            agents=agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_cycles=args.max_cycles,
            team_reward_mode=team_reward_mode,
            reward_scale=reward_scale,
            device=device,
        )
    print_reward_and_done_check(
        max_cycles=args.max_cycles,
        seed=args.seed + 3000,
        team_reward_mode=team_reward_mode,
        reward_scale=reward_scale,
    )
    env = make_mpe_env("simple_spread_v3", max_cycles=1)
    print(f"\nEnvironment source: {get_mpe_env_source(env)}")
    env.close()
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
