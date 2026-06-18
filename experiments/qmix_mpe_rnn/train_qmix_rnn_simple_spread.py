# -*- coding: utf-8 -*-
"""Train recurrent QMIX/VDN on MPE simple_spread_v3."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algorithms.qmix_rnn import EpisodeReplayBuffer, QMixer, RNNAgent, VDNMixer
from envs.mpe_env_factory import get_mpe_env_source, make_mpe_env


ENV_NAME = "simple_spread_v3"
OUTPUT_BASE_DIR = ROOT_DIR / "outputs" / "qmix_mpe_rnn" / "simple_spread"
RANDOM_BASELINE_DIR = ROOT_DIR / "outputs" / "qmix_mpe" / "simple_spread" / "random_baselines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RNN QMIX on MPE simple_spread_v3.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-cycles", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--buffer-size", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--rnn-hidden-dim", type=int, default=64)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--min-epsilon", type=float, default=None)
    parser.add_argument("--epsilon-decay-steps", type=int, default=10000)
    parser.add_argument("--epsilon-eval", type=float, default=0.0)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--loss-type", choices=("mse", "huber"), default="huber")
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--team-reward-mode", choices=("mean", "sum", "first"), default="mean")
    parser.add_argument("--mixer-type", choices=("qmix", "vdn"), default="qmix")
    parser.add_argument("--double-q", action="store_true", default=False)
    parser.add_argument("--mixer-weight-clip", type=float, default=1.0)
    parser.add_argument("--mixer-bias-clip", type=float, default=5.0)
    parser.add_argument(
        "--mixer-weight-activation",
        choices=("softplus", "abs"),
        default="softplus",
    )
    parser.add_argument("--use-layer-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-last-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-agent-id", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-shapes", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.min_epsilon is None:
        args.min_epsilon = args.epsilon_end
    return args


def reset_env(env, seed: int | None = None):
    try:
        if seed is None:
            return env.reset()
        return env.reset(seed=seed)
    except TypeError:
        return env.reset()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_next_run_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, 10000):
        run_dir = base_dir / f"run{index:03d}"
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Cannot create a new runXXX directory under {base_dir}")


def obs_to_array(observations: dict, agents: list[str], obs_dim: int) -> np.ndarray:
    rows = []
    for agent in agents:
        if agent not in observations:
            raise KeyError(f"Missing observation for {agent}")
        obs = np.asarray(observations[agent], dtype=np.float32)
        if obs.shape[-1] != obs_dim:
            raise ValueError(f"{agent} obs_dim should be {obs_dim}, got {obs.shape[-1]}")
        rows.append(obs)
    return np.stack(rows, axis=0)


def next_obs_to_array(
    next_observations: dict,
    agents: list[str],
    obs_dim: int,
    fallback_obs: np.ndarray,
) -> np.ndarray:
    if all(agent in next_observations for agent in agents):
        return obs_to_array(next_observations, agents, obs_dim)
    return fallback_obs.copy()


def make_state(obs_array: np.ndarray) -> np.ndarray:
    return obs_array.reshape(-1).astype(np.float32)


def batch_states(obs: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, n_agents, obs_dim = obs.shape
    return obs.reshape(batch_size, seq_len, n_agents * obs_dim)


def rnn_input_dim(
    obs_dim: int,
    action_dim: int,
    n_agents: int,
    include_last_action: bool,
    include_agent_id: bool,
) -> int:
    return (
        int(obs_dim)
        + (int(action_dim) if include_last_action else 0)
        + (int(n_agents) if include_agent_id else 0)
    )


def agent_id_tensor(
    batch_size: int,
    seq_len: int,
    n_agents: int,
    device: torch.device | str,
) -> torch.Tensor:
    eye = torch.eye(n_agents, dtype=torch.float32, device=device)
    return eye.view(1, 1, n_agents, n_agents).expand(batch_size, seq_len, n_agents, n_agents)


def shifted_last_actions(actions: torch.Tensor, action_dim: int) -> torch.Tensor:
    batch_size, seq_len, n_agents = actions.shape
    last_actions = torch.zeros(
        batch_size,
        seq_len,
        n_agents,
        action_dim,
        dtype=torch.float32,
        device=actions.device,
    )
    if seq_len > 1:
        last_actions[:, 1:] = F.one_hot(actions[:, :-1].long(), num_classes=action_dim).float()
    return last_actions


def build_sequence_inputs(
    obs: torch.Tensor,
    actions: torch.Tensor,
    action_dim: int,
    include_last_action: bool,
    include_agent_id: bool,
    next_obs_inputs: bool = False,
) -> torch.Tensor:
    batch_size, seq_len, n_agents, _ = obs.shape
    parts = [obs]
    if include_last_action:
        if next_obs_inputs:
            last_actions = F.one_hot(actions.long(), num_classes=action_dim).float()
        else:
            last_actions = shifted_last_actions(actions, action_dim)
        parts.append(last_actions)
    if include_agent_id:
        parts.append(agent_id_tensor(batch_size, seq_len, n_agents, obs.device))
    return torch.cat(parts, dim=-1)


def build_step_inputs(
    obs_array: np.ndarray,
    last_actions: np.ndarray,
    include_last_action: bool,
    include_agent_id: bool,
    device: torch.device,
) -> torch.Tensor:
    obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=device)
    parts = [obs_tensor]
    if include_last_action:
        parts.append(torch.as_tensor(last_actions, dtype=torch.float32, device=device))
    if include_agent_id:
        n_agents = int(obs_array.shape[0])
        parts.append(torch.eye(n_agents, dtype=torch.float32, device=device))
    return torch.cat(parts, dim=-1)


def one_hot_np(indices: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes, dtype=np.float32)[np.asarray(indices, dtype=np.int64)]


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


def done_flag(next_observations: dict, terminations: dict, truncations: dict, agents: list[str]) -> bool:
    if not next_observations:
        return True
    return all(
        bool(terminations.get(agent, False) or truncations.get(agent, False))
        for agent in agents
    )


def epsilon_by_step(args: argparse.Namespace, total_steps: int) -> float:
    if args.epsilon_decay_steps <= 0:
        return float(args.min_epsilon)
    fraction = min(1.0, total_steps / float(args.epsilon_decay_steps))
    decayed = float(args.epsilon_start + fraction * (args.epsilon_end - args.epsilon_start))
    return float(max(args.min_epsilon, decayed))


def select_actions_from_q(q_values: torch.Tensor, epsilon: float, action_dim: int) -> np.ndarray:
    actions = []
    for row in q_values.detach().cpu():
        if random.random() < epsilon:
            actions.append(random.randrange(action_dim))
        else:
            actions.append(int(torch.argmax(row).item()))
    return np.asarray(actions, dtype=np.int64)


def build_mixer(
    mixer_type: str,
    n_agents: int,
    state_dim: int,
    hidden_dim: int,
    use_layer_norm: bool,
    mixer_weight_clip: float,
    mixer_bias_clip: float,
    mixer_weight_activation: str,
) -> torch.nn.Module:
    if mixer_type == "qmix":
        return QMixer(
            n_agents=n_agents,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
            mixer_weight_clip=mixer_weight_clip,
            mixer_bias_clip=mixer_bias_clip,
            mixer_weight_activation=mixer_weight_activation,
        )
    if mixer_type == "vdn":
        return VDNMixer()
    raise ValueError(f"Unknown mixer_type: {mixer_type}")


def hard_update(source: torch.nn.Module, target: torch.nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * source_param.data)


def ensure_shape(tensor: torch.Tensor, expected_shape: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape) != expected_shape:
        raise RuntimeError(
            f"{name} shape mismatch: expected {expected_shape}, got {tuple(tensor.shape)}"
        )


def debug_tensor(name: str, tensor: torch.Tensor) -> None:
    print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")


def unroll_agent(agent: RNNAgent, inputs: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, n_agents, input_dim = inputs.shape
    hidden = agent.init_hidden(batch_size * n_agents, inputs.device)
    q_values = []
    for step in range(seq_len):
        q_step, hidden = agent(inputs[:, step].reshape(batch_size * n_agents, input_dim), hidden)
        q_values.append(q_step.view(batch_size, n_agents, agent.action_dim))
    return torch.stack(q_values, dim=1)


def train_update(
    agent: RNNAgent,
    target_agent: RNNAgent,
    mixer: torch.nn.Module,
    target_mixer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay_buffer: EpisodeReplayBuffer,
    batch_size: int,
    gamma: float,
    tau: float,
    grad_clip: float,
    loss_type: str,
    double_q: bool,
    include_last_action: bool,
    include_agent_id: bool,
    device: torch.device,
    update_index: int = 0,
    debug_shapes: bool = False,
) -> dict[str, float]:
    batch = replay_buffer.sample(batch_size, device)
    obs = batch["obs"]
    actions = batch["actions"].long()
    rewards = batch["rewards"]
    next_obs = batch["next_obs"]
    dones = batch["dones"]
    filled = batch["filled"]

    batch_size, seq_len, n_agents, obs_dim = obs.shape
    ensure_shape(actions, (batch_size, seq_len, n_agents), "actions")
    ensure_shape(rewards, (batch_size, seq_len, 1), "rewards")
    ensure_shape(next_obs, (batch_size, seq_len, n_agents, obs_dim), "next_obs")
    ensure_shape(dones, (batch_size, seq_len, 1), "dones")
    ensure_shape(filled, (batch_size, seq_len, 1), "filled")

    states = batch_states(obs)
    next_states = batch_states(next_obs)
    agent_inputs = build_sequence_inputs(
        obs=obs,
        actions=actions,
        action_dim=agent.action_dim,
        include_last_action=include_last_action,
        include_agent_id=include_agent_id,
        next_obs_inputs=False,
    )
    next_agent_inputs = build_sequence_inputs(
        obs=next_obs,
        actions=actions,
        action_dim=agent.action_dim,
        include_last_action=include_last_action,
        include_agent_id=include_agent_id,
        next_obs_inputs=True,
    )
    q_values = unroll_agent(agent, agent_inputs)
    chosen_qs = q_values.gather(3, actions.unsqueeze(-1)).squeeze(-1)
    q_tot = mixer(chosen_qs, states)

    with torch.no_grad():
        target_next_q_values = unroll_agent(target_agent, next_agent_inputs)
        if double_q:
            online_next_q_values = unroll_agent(agent, next_agent_inputs)
            next_actions = online_next_q_values.argmax(dim=3, keepdim=True)
            target_agent_qs = target_next_q_values.gather(3, next_actions).squeeze(-1)
        else:
            target_agent_qs = target_next_q_values.max(dim=3).values
        target_q_tot = target_mixer(target_agent_qs, next_states)
        td_target = rewards + gamma * (1.0 - dones) * target_q_tot

    ensure_shape(chosen_qs, (batch_size, seq_len, n_agents), "chosen_qs")
    ensure_shape(q_tot, (batch_size, seq_len, 1), "q_tot")
    ensure_shape(target_q_tot, (batch_size, seq_len, 1), "target_q_tot")
    ensure_shape(td_target, (batch_size, seq_len, 1), "td_target")

    if debug_shapes and update_index < 3:
        print(f"\n[debug-shapes] update={update_index + 1}")
        for name, tensor in (
            ("obs", obs),
            ("actions", actions),
            ("rewards", rewards),
            ("dones", dones),
            ("filled", filled),
            ("chosen_qs", chosen_qs),
            ("q_tot", q_tot),
            ("target_q_tot", target_q_tot),
            ("td_target", td_target),
        ):
            debug_tensor(name, tensor)

    if loss_type == "mse":
        element_loss = (q_tot - td_target).pow(2)
    elif loss_type == "huber":
        element_loss = F.smooth_l1_loss(q_tot, td_target, reduction="none")
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    td_loss = (element_loss * filled).sum() / filled.sum().clamp_min(1.0)

    optimizer.zero_grad()
    td_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        list(agent.parameters()) + list(mixer.parameters()),
        max_norm=grad_clip,
    )
    optimizer.step()
    soft_update(agent, target_agent, tau)
    soft_update(mixer, target_mixer, tau)

    return {
        "td_loss": float(td_loss.item()),
        "total_loss": float(td_loss.item()),
        "grad_norm": float(grad_norm.item()),
        "q_tot_mean": float((q_tot * filled).sum().detach().item() / filled.sum().clamp_min(1.0).item()),
        "q_tot_std": float(q_tot.detach()[filled.expand_as(q_tot).bool()].std(unbiased=False).item()),
        "q_tot_max": float(q_tot.detach().abs().max().item()),
        "target_q_tot_mean": float(
            (target_q_tot * filled).sum().detach().item() / filled.sum().clamp_min(1.0).item()
        ),
        "target_q_tot_std": float(
            target_q_tot.detach()[filled.expand_as(target_q_tot).bool()].std(unbiased=False).item()
        ),
        "target_q_tot_max": float(target_q_tot.detach().abs().max().item()),
    }


def evaluate_policy(
    agent: RNNAgent,
    agents: list[str],
    obs_dim: int,
    action_dim: int,
    max_cycles: int,
    eval_episodes: int,
    seed: int,
    device: torch.device,
    epsilon_eval: float = 0.0,
    team_reward_mode: str = "mean",
    include_last_action: bool = True,
    include_agent_id: bool = True,
) -> tuple[float, float]:
    eval_env = make_mpe_env(ENV_NAME, max_cycles=max_cycles)
    team_returns = []
    mean_agent_returns = []
    agent.eval()

    for episode in range(eval_episodes):
        observations, infos = reset_env(eval_env, seed=seed + episode)
        del infos
        hidden = agent.init_hidden(len(agents), device)
        last_actions = np.zeros((len(agents), action_dim), dtype=np.float32)
        agent_returns = {name: 0.0 for name in agents}
        episode_team_return = 0.0

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
            action_dict = {
                name: int(action)
                for name, action in zip(agents, actions)
                if name in observations
            }
            observations, rewards, terminations, truncations, infos = eval_env.step(action_dict)
            del infos
            episode_team_return += team_reward(rewards, team_reward_mode)
            for name in agents:
                agent_returns[name] += float(rewards.get(name, 0.0))
            if done_flag(observations, terminations, truncations, agents):
                break

        team_returns.append(episode_team_return)
        mean_agent_returns.append(float(np.mean(list(agent_returns.values()))))

    eval_env.close()
    return float(np.mean(team_returns)), float(np.mean(mean_agent_returns))


def inspect_action_distribution(
    agent: RNNAgent,
    agents: list[str],
    obs_dim: int,
    action_dim: int,
    max_cycles: int,
    episodes: int,
    seed: int,
    device: torch.device,
    epsilon_eval: float = 0.0,
    include_last_action: bool = True,
    include_agent_id: bool = True,
) -> dict[str, list[int]]:
    counts = {name: [0 for _ in range(action_dim)] for name in agents}
    env = make_mpe_env(ENV_NAME, max_cycles=max_cycles)
    agent.eval()
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
            for name, action in zip(agents, actions):
                counts[name][int(action)] += 1
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
    return counts


def action_entropy(counts: list[int]) -> float:
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


def action_coverage_stats(action_counts: dict[str, list[int]]) -> dict:
    per_agent = {}
    collapse_reason = "none"
    collapse_priority = 0
    for agent_name, counts in action_counts.items():
        total = sum(counts)
        sorted_counts = sorted(counts, reverse=True)
        max_action_ratio = 0.0 if total == 0 else sorted_counts[0] / total
        top2_action_ratio = 0.0 if total == 0 else sum(sorted_counts[:2]) / total
        used_action_count = sum(1 for count in counts if count > 0)
        entropy = action_entropy(counts)
        reason = "none"
        priority = 0
        if max_action_ratio > 0.95:
            reason = "high collapse risk"
            priority = 3
        elif used_action_count <= 2 and top2_action_ratio > 0.95:
            reason = "action coverage collapse"
            priority = 2
        elif entropy < 0.7:
            reason = "low entropy warning"
            priority = 1
        if priority > collapse_priority:
            collapse_priority = priority
            collapse_reason = reason
        per_agent[agent_name] = {
            "counts": counts,
            "max_action_ratio": float(max_action_ratio),
            "top2_action_ratio": float(top2_action_ratio),
            "used_action_count": int(used_action_count),
            "action_entropy": float(entropy),
            "collapse_reason": reason,
        }

    if not per_agent:
        return {
            "policy_collapse_risk": "no obvious collapse",
            "collapse_reason": "none",
            "action_entropy_mean": 0.0,
            "used_action_count_min": 0,
            "max_action_ratio_max": 0.0,
            "top2_action_ratio_max": 0.0,
            "per_agent": per_agent,
        }

    collapse = "no obvious collapse" if collapse_reason == "none" else collapse_reason
    return {
        "policy_collapse_risk": collapse,
        "collapse_reason": collapse_reason,
        "action_entropy_mean": float(
            np.mean([stats["action_entropy"] for stats in per_agent.values()])
        ),
        "used_action_count_min": int(
            min(stats["used_action_count"] for stats in per_agent.values())
        ),
        "max_action_ratio_max": float(
            max(stats["max_action_ratio"] for stats in per_agent.values())
        ),
        "top2_action_ratio_max": float(
            max(stats["top2_action_ratio"] for stats in per_agent.values())
        ),
        "per_agent": per_agent,
    }


def policy_collapse_risk(action_counts: dict[str, list[int]]) -> str:
    return str(action_coverage_stats(action_counts)["policy_collapse_risk"])


def training_health(
    last_td_loss: float | None,
    collapse: str,
    final_eval_team_return: float | None,
    random_team_baseline: float | None,
    last_q_tot_max: float | None,
) -> str:
    if last_td_loss is None:
        return "warning: no updates were run"
    if not math.isfinite(last_td_loss) or last_td_loss > 100:
        return "warning"
    if collapse != "no obvious collapse":
        return "warning"
    if last_q_tot_max is not None and math.isfinite(last_q_tot_max) and abs(last_q_tot_max) > 500:
        return "warning"
    if final_eval_team_return is not None and random_team_baseline is not None:
        if random_team_baseline < 0 and final_eval_team_return < random_team_baseline * 1.5:
            return "warning"
        if random_team_baseline >= 0 and final_eval_team_return < random_team_baseline * 0.5:
            return "warning"
    return "OK"


def baseline_path(max_cycles: int, team_reward_mode: str) -> Path:
    return RANDOM_BASELINE_DIR / f"random_baseline_cycles{max_cycles}_mode_{team_reward_mode}.json"


def load_random_baseline(max_cycles: int, team_reward_mode: str) -> dict | None:
    path = baseline_path(max_cycles, team_reward_mode)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if data.get("max_cycles") != max_cycles or data.get("team_reward_mode") != team_reward_mode:
        return None
    if data.get("random_team_return") is None:
        return None
    data["source"] = str(path)
    return data


def algorithm_label(mixer_type: str) -> str:
    if mixer_type == "vdn":
        return "RNN-VDN"
    return "RNN-QMIX"


def better_than_random_status(
    eval_team_return: float | None,
    random_team_baseline: float | None,
) -> str:
    if eval_team_return is None or random_team_baseline is None:
        return "not checked"
    return "better" if eval_team_return > random_team_baseline else "worse"


def learning_status_from_better(better_status: str) -> str:
    if better_status == "better":
        return "better than random"
    if better_status == "worse":
        return "worse than random"
    return "not checked"


def last_numeric_log_value(rows: list[dict], key: str) -> float | None:
    for row in reversed(rows):
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def save_config(args: argparse.Namespace, run_dir: Path, source: str, metadata: dict) -> None:
    config = vars(args).copy()
    config.update(metadata)
    config["env_source"] = source
    tmp_path = run_dir / "config.tmp.json"
    final_path = run_dir / "config.json"
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    os.replace(tmp_path, final_path)


def save_log(rows: list[dict], csv_path: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "update",
        "episode",
        "total_steps",
        "train_team_return",
        "td_loss",
        "total_loss",
        "grad_norm",
        "q_tot_mean",
        "q_tot_std",
        "q_tot_max",
        "target_q_tot_mean",
        "target_q_tot_std",
        "target_q_tot_max",
        "epsilon",
        "action_entropy_mean",
        "used_action_count_min",
        "max_action_ratio_max",
        "top2_action_ratio_max",
        "eval_team_return",
        "eval_mean_agent_return",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_name(f".{csv_path.stem}.{uuid.uuid4().hex}.tmp{csv_path.suffix}")
    with tmp_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, csv_path)


def save_checkpoint(
    checkpoint_path: Path,
    agent: RNNAgent,
    target_agent: RNNAgent,
    mixer: torch.nn.Module,
    target_mixer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: dict,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(".tmp.pt")
    payload = {
        "agent": agent.state_dict(),
        "target_agent": target_agent.state_dict(),
        "mixer": mixer.state_dict(),
        "target_mixer": target_mixer.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, checkpoint_path)


def write_summary(
    run_dir: Path,
    total_episodes: int,
    total_steps: int,
    final_eval_team_return: float | None,
    final_td_loss: float | None,
    final_epsilon: float,
    best_eval_team_return: float | None,
    best_eval_episode: int | None,
    best_checkpoint_path: Path,
    random_team_baseline: float | None,
    final_better_than_random: str,
    best_better_than_random: str,
    random_baseline_source: str,
    random_baseline_warning: str | None,
    final_learning_status: str,
    best_learning_status: str,
    collapse: str,
    coverage_stats: dict,
    health: str,
    args: argparse.Namespace,
) -> None:
    lines = [
        f"run directory: {run_dir}",
        f"total episodes: {total_episodes}",
        f"total steps: {total_steps}",
        f"final eval_team_return: {final_eval_team_return}",
        f"final td_loss: {final_td_loss}",
        f"final epsilon: {final_epsilon}",
        f"best eval_team_return: {best_eval_team_return}",
        f"best eval episode: {best_eval_episode}",
        f"best checkpoint path: {best_checkpoint_path}",
        f"random team baseline: {'not found' if random_team_baseline is None else random_team_baseline}",
        f"algorithm label: {algorithm_label(args.mixer_type)}",
        f"final better than random: {final_better_than_random}",
        f"best better than random: {best_better_than_random}",
        f"learning status final: {final_learning_status}",
        f"learning status best: {best_learning_status}",
        f"final learning status: {final_learning_status}",
        f"best learning status: {best_learning_status}",
        f"random baseline source: {random_baseline_source}",
        f"policy collapse risk: {collapse}",
        f"collapse_reason: {coverage_stats.get('collapse_reason', 'none')}",
        f"action_entropy_mean: {coverage_stats.get('action_entropy_mean')}",
        f"used_action_count_min: {coverage_stats.get('used_action_count_min')}",
        f"max_action_ratio_max: {coverage_stats.get('max_action_ratio_max')}",
        f"top2_action_ratio_max: {coverage_stats.get('top2_action_ratio_max')}",
        f"training health: {health}",
        f"mixer_type: {args.mixer_type}",
        f"double_q: {args.double_q}",
        f"reward_scale: {args.reward_scale}",
        f"team_reward_mode: {args.team_reward_mode}",
        f"rnn_hidden_dim: {args.rnn_hidden_dim}",
        f"include_last_action: {args.include_last_action}",
        f"include_agent_id: {args.include_agent_id}",
        f"rnn_input_dim: {getattr(args, 'rnn_input_dim', None)}",
        f"hidden_dim: {args.hidden_dim}",
        f"use_layer_norm: {args.use_layer_norm}",
        f"mixer_weight_clip: {args.mixer_weight_clip}",
        f"mixer_bias_clip: {args.mixer_bias_clip}",
        f"mixer_weight_activation: {args.mixer_weight_activation}",
    ]
    if random_baseline_warning:
        lines.append(f"random baseline warning: {random_baseline_warning}")
    for agent_name, stats in coverage_stats.get("per_agent", {}).items():
        lines.append(
            "action stats "
            f"{agent_name}: max_action_ratio={stats['max_action_ratio']}, "
            f"top2_action_ratio={stats['top2_action_ratio']}, "
            f"used_action_count={stats['used_action_count']}, "
            f"action_entropy={stats['action_entropy']}, "
            f"collapse_reason={stats['collapse_reason']}"
        )
    tmp_path = run_dir / "summary.tmp.txt"
    summary_path = run_dir / "summary.txt"
    with tmp_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    os.replace(tmp_path, summary_path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = get_next_run_dir(OUTPUT_BASE_DIR)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "qmix_rnn_best.pt"
    final_checkpoint_path = checkpoint_dir / "qmix_rnn_final.pt"

    env = make_mpe_env(ENV_NAME, max_cycles=args.max_cycles)
    source = get_mpe_env_source(env)
    observations, infos = reset_env(env, seed=args.seed)
    del observations, infos
    agents = list(env.possible_agents)
    if not agents:
        raise RuntimeError("simple_spread_v3 returned no possible_agents.")

    obs_dim = int(env.observation_space(agents[0]).shape[0])
    action_dim = int(env.action_space(agents[0]).n)
    n_agents = len(agents)
    state_dim = n_agents * obs_dim
    args.rnn_input_dim = rnn_input_dim(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_agents=n_agents,
        include_last_action=args.include_last_action,
        include_agent_id=args.include_agent_id,
    )
    for name in agents:
        if int(env.observation_space(name).shape[0]) != obs_dim:
            raise RuntimeError("RNN QMIX expects equal obs_dim for all agents.")
        if int(env.action_space(name).n) != action_dim:
            raise RuntimeError("RNN QMIX expects equal action_dim for all agents.")

    agent = RNNAgent(obs_dim, action_dim, args.rnn_hidden_dim, input_dim=args.rnn_input_dim).to(device)
    target_agent = RNNAgent(
        obs_dim,
        action_dim,
        args.rnn_hidden_dim,
        input_dim=args.rnn_input_dim,
    ).to(device)
    mixer = build_mixer(
        mixer_type=args.mixer_type,
        n_agents=n_agents,
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        use_layer_norm=args.use_layer_norm,
        mixer_weight_clip=args.mixer_weight_clip,
        mixer_bias_clip=args.mixer_bias_clip,
        mixer_weight_activation=args.mixer_weight_activation,
    ).to(device)
    target_mixer = build_mixer(
        mixer_type=args.mixer_type,
        n_agents=n_agents,
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        use_layer_norm=args.use_layer_norm,
        mixer_weight_clip=args.mixer_weight_clip,
        mixer_bias_clip=args.mixer_bias_clip,
        mixer_weight_activation=args.mixer_weight_activation,
    ).to(device)
    hard_update(agent, target_agent)
    hard_update(mixer, target_mixer)

    optimizer = torch.optim.Adam(
        list(agent.parameters()) + list(mixer.parameters()),
        lr=args.learning_rate,
    )
    replay_buffer = EpisodeReplayBuffer(args.buffer_size, n_agents, obs_dim)

    metadata = {
        "algorithm": "rnn_qmix",
        "env": ENV_NAME,
        "agent_names": agents,
        "n_agents": n_agents,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "hidden_dim": int(args.hidden_dim),
        "rnn_hidden_dim": int(args.rnn_hidden_dim),
        "rnn_input_dim": int(args.rnn_input_dim),
        "include_last_action": bool(args.include_last_action),
        "include_agent_id": bool(args.include_agent_id),
        "max_cycles": int(args.max_cycles),
        "mixer_type": args.mixer_type,
        "double_q": bool(args.double_q),
        "use_layer_norm": bool(args.use_layer_norm),
        "mixer_weight_clip": float(args.mixer_weight_clip),
        "mixer_bias_clip": float(args.mixer_bias_clip),
        "mixer_weight_activation": args.mixer_weight_activation,
        "reward_scale": float(args.reward_scale),
        "team_reward_mode": args.team_reward_mode,
        "device": str(device),
    }
    save_config(args, run_dir, source, metadata)

    print(f"Loaded {ENV_NAME} from: {source}")
    print(f"Run directory: {run_dir}")
    print("QMIX RNN: shared RNNAgent + episode replay buffer")
    print(
        f"agents={agents}, obs_dim={obs_dim}, action_dim={action_dim}, "
        f"state_dim={state_dim}, rnn_input_dim={args.rnn_input_dim}"
    )

    log_rows: list[dict] = []
    total_steps = 0
    update_count = 0
    final_eval_team_return = None
    final_eval_mean_agent_return = None
    best_eval_team_return = None
    best_eval_episode = None
    last_td_loss = None

    for episode in range(1, args.episodes + 1):
        observations, infos = reset_env(env, seed=args.seed + episode)
        del infos
        hidden = agent.init_hidden(n_agents, device)
        last_actions = np.zeros((n_agents, action_dim), dtype=np.float32)
        episode_team_return = 0.0
        episode_update_losses = []
        episode_obs = []
        episode_actions = []
        episode_rewards = []
        episode_next_obs = []
        episode_dones = []
        agent.train()

        for _ in range(args.max_cycles):
            if not observations:
                break
            obs_array = obs_to_array(observations, agents, obs_dim)
            epsilon = epsilon_by_step(args, total_steps)
            obs_tensor = build_step_inputs(
                obs_array,
                last_actions,
                args.include_last_action,
                args.include_agent_id,
                device,
            )
            with torch.no_grad():
                q_values, next_hidden = agent(obs_tensor, hidden)
            actions = select_actions_from_q(q_values, epsilon, action_dim)
            last_actions = one_hot_np(actions, action_dim)
            action_dict = {
                name: int(action)
                for name, action in zip(agents, actions)
                if name in observations
            }
            next_observations, rewards, terminations, truncations, infos = env.step(action_dict)
            del infos

            reward = team_reward(rewards, args.team_reward_mode)
            episode_team_return += reward
            done = done_flag(next_observations, terminations, truncations, agents)
            next_obs_array = next_obs_to_array(next_observations, agents, obs_dim, obs_array)

            episode_obs.append(obs_array)
            episode_actions.append(actions)
            episode_rewards.append([reward * args.reward_scale])
            episode_next_obs.append(next_obs_array)
            episode_dones.append([float(done)])
            total_steps += 1
            hidden = next_hidden.detach()

            if done:
                break
            observations = next_observations

        if episode_obs:
            replay_buffer.add_episode(
                obs=np.asarray(episode_obs, dtype=np.float32),
                actions=np.asarray(episode_actions, dtype=np.int64),
                rewards=np.asarray(episode_rewards, dtype=np.float32),
                next_obs=np.asarray(episode_next_obs, dtype=np.float32),
                dones=np.asarray(episode_dones, dtype=np.float32),
            )

        if len(replay_buffer) >= args.batch_size:
            update_info = train_update(
                agent=agent,
                target_agent=target_agent,
                mixer=mixer,
                target_mixer=target_mixer,
                optimizer=optimizer,
                replay_buffer=replay_buffer,
                batch_size=args.batch_size,
                gamma=args.gamma,
                tau=args.tau,
                grad_clip=args.grad_clip,
                loss_type=args.loss_type,
                double_q=args.double_q,
                include_last_action=args.include_last_action,
                include_agent_id=args.include_agent_id,
                device=device,
                update_index=update_count,
                debug_shapes=args.debug_shapes,
            )
            update_count += 1
            last_td_loss = update_info["td_loss"]
            episode_update_losses.append(last_td_loss)
            log_rows.append(
                {
                    "update": update_count,
                    "episode": episode,
                    "total_steps": total_steps,
                    "train_team_return": episode_team_return,
                    "td_loss": update_info["td_loss"],
                    "total_loss": update_info["total_loss"],
                    "grad_norm": update_info["grad_norm"],
                    "q_tot_mean": update_info["q_tot_mean"],
                    "q_tot_std": update_info["q_tot_std"],
                    "q_tot_max": update_info["q_tot_max"],
                    "target_q_tot_mean": update_info["target_q_tot_mean"],
                    "target_q_tot_std": update_info["target_q_tot_std"],
                    "target_q_tot_max": update_info["target_q_tot_max"],
                    "epsilon": epsilon_by_step(args, total_steps),
                    "action_entropy_mean": None,
                    "used_action_count_min": None,
                    "max_action_ratio_max": None,
                    "top2_action_ratio_max": None,
                    "eval_team_return": None,
                    "eval_mean_agent_return": None,
                }
            )

        eval_team_return = None
        eval_mean_agent_return = None
        if args.eval_interval > 0 and episode % args.eval_interval == 0:
            eval_team_return, eval_mean_agent_return = evaluate_policy(
                agent=agent,
                agents=agents,
                obs_dim=obs_dim,
                action_dim=action_dim,
                max_cycles=args.max_cycles,
                eval_episodes=args.eval_episodes,
                seed=args.seed + episode * 1000,
                device=device,
                epsilon_eval=args.epsilon_eval,
                team_reward_mode=args.team_reward_mode,
                include_last_action=args.include_last_action,
                include_agent_id=args.include_agent_id,
            )
            eval_action_counts = inspect_action_distribution(
                agent=agent,
                agents=agents,
                obs_dim=obs_dim,
                action_dim=action_dim,
                max_cycles=args.max_cycles,
                episodes=args.eval_episodes,
                seed=args.seed + episode * 1000 + 100,
                device=device,
                epsilon_eval=args.epsilon_eval,
                include_last_action=args.include_last_action,
                include_agent_id=args.include_agent_id,
            )
            eval_coverage_stats = action_coverage_stats(eval_action_counts)
            final_eval_team_return = eval_team_return
            final_eval_mean_agent_return = eval_mean_agent_return
            if best_eval_team_return is None or eval_team_return > best_eval_team_return:
                best_eval_team_return = eval_team_return
                best_eval_episode = episode
                best_metadata = metadata.copy()
                best_metadata.update(
                    {
                        "checkpoint_type": "best",
                        "best_eval_team_return": float(best_eval_team_return),
                        "best_eval_episode": int(best_eval_episode),
                    }
                )
                save_checkpoint(
                    best_checkpoint_path,
                    agent,
                    target_agent,
                    mixer,
                    target_mixer,
                    optimizer,
                    best_metadata,
                )
            if log_rows:
                log_rows[-1]["eval_team_return"] = eval_team_return
                log_rows[-1]["eval_mean_agent_return"] = eval_mean_agent_return
                log_rows[-1]["action_entropy_mean"] = eval_coverage_stats["action_entropy_mean"]
                log_rows[-1]["used_action_count_min"] = eval_coverage_stats["used_action_count_min"]
                log_rows[-1]["max_action_ratio_max"] = eval_coverage_stats["max_action_ratio_max"]
                log_rows[-1]["top2_action_ratio_max"] = eval_coverage_stats["top2_action_ratio_max"]
            else:
                log_rows.append(
                    {
                        "update": update_count,
                        "episode": episode,
                        "total_steps": total_steps,
                        "train_team_return": episode_team_return,
                        "td_loss": None,
                        "total_loss": None,
                        "grad_norm": None,
                        "q_tot_mean": None,
                        "q_tot_std": None,
                        "q_tot_max": None,
                        "target_q_tot_mean": None,
                        "target_q_tot_std": None,
                        "target_q_tot_max": None,
                        "epsilon": epsilon_by_step(args, total_steps),
                        "action_entropy_mean": eval_coverage_stats["action_entropy_mean"],
                        "used_action_count_min": eval_coverage_stats["used_action_count_min"],
                        "max_action_ratio_max": eval_coverage_stats["max_action_ratio_max"],
                        "top2_action_ratio_max": eval_coverage_stats["top2_action_ratio_max"],
                        "eval_team_return": eval_team_return,
                        "eval_mean_agent_return": eval_mean_agent_return,
                    }
                )
            save_checkpoint(
                checkpoint_dir / f"qmix_rnn_episode_{episode:04d}.pt",
                agent,
                target_agent,
                mixer,
                target_mixer,
                optimizer,
                metadata,
            )
            save_log(log_rows, run_dir / "train_log.csv")

        loss_text = "None" if not episode_update_losses else f"{float(np.mean(episode_update_losses)):.4f}"
        eval_text = "None" if eval_team_return is None else f"{eval_team_return:.3f}"
        print(
            f"episode={episode:04d} steps={total_steps} "
            f"train_team_return={episode_team_return:.3f} "
            f"mean_td_loss={loss_text} "
            f"epsilon={epsilon_by_step(args, total_steps):.3f} "
            f"eval_team_return={eval_text}"
        )

    env.close()

    if final_eval_team_return is None:
        final_eval_team_return, final_eval_mean_agent_return = evaluate_policy(
            agent=agent,
            agents=agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_cycles=args.max_cycles,
            eval_episodes=args.eval_episodes,
            seed=args.seed + args.episodes * 1000 + 500,
            device=device,
            epsilon_eval=args.epsilon_eval,
            team_reward_mode=args.team_reward_mode,
            include_last_action=args.include_last_action,
            include_agent_id=args.include_agent_id,
        )
        if log_rows:
            log_rows[-1]["eval_team_return"] = final_eval_team_return
            log_rows[-1]["eval_mean_agent_return"] = final_eval_mean_agent_return
        else:
            log_rows.append(
                {
                    "update": update_count,
                    "episode": args.episodes,
                    "total_steps": total_steps,
                    "train_team_return": None,
                    "td_loss": None,
                    "total_loss": None,
                    "grad_norm": None,
                    "q_tot_mean": None,
                    "q_tot_std": None,
                    "q_tot_max": None,
                    "target_q_tot_mean": None,
                    "target_q_tot_std": None,
                    "target_q_tot_max": None,
                    "epsilon": epsilon_by_step(args, total_steps),
                    "action_entropy_mean": None,
                    "used_action_count_min": None,
                    "max_action_ratio_max": None,
                    "top2_action_ratio_max": None,
                    "eval_team_return": final_eval_team_return,
                    "eval_mean_agent_return": final_eval_mean_agent_return,
                }
            )
        if best_eval_team_return is None or final_eval_team_return > best_eval_team_return:
            best_eval_team_return = final_eval_team_return
            best_eval_episode = args.episodes
            best_metadata = metadata.copy()
            best_metadata.update(
                {
                    "checkpoint_type": "best",
                    "best_eval_team_return": float(best_eval_team_return),
                    "best_eval_episode": int(best_eval_episode),
                }
            )
            save_checkpoint(
                best_checkpoint_path,
                agent,
                target_agent,
                mixer,
                target_mixer,
                optimizer,
                best_metadata,
            )

    save_checkpoint(
        final_checkpoint_path,
        agent,
        target_agent,
        mixer,
        target_mixer,
        optimizer,
        metadata,
    )
    save_log(log_rows, run_dir / "train_log.csv")

    action_counts = inspect_action_distribution(
        agent=agent,
        agents=agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_cycles=args.max_cycles,
        episodes=min(5, max(1, args.eval_episodes)),
        seed=args.seed + 900000,
        device=device,
        epsilon_eval=0.0,
        include_last_action=args.include_last_action,
        include_agent_id=args.include_agent_id,
    )
    coverage_stats = action_coverage_stats(action_counts)
    collapse = str(coverage_stats["policy_collapse_risk"])
    if log_rows:
        log_rows[-1]["action_entropy_mean"] = coverage_stats["action_entropy_mean"]
        log_rows[-1]["used_action_count_min"] = coverage_stats["used_action_count_min"]
        log_rows[-1]["max_action_ratio_max"] = coverage_stats["max_action_ratio_max"]
        log_rows[-1]["top2_action_ratio_max"] = coverage_stats["top2_action_ratio_max"]
        save_log(log_rows, run_dir / "train_log.csv")
    baseline_info = load_random_baseline(args.max_cycles, args.team_reward_mode)
    random_baseline_warning = None
    if baseline_info is None:
        random_team_baseline = None
        random_baseline_source = "not found"
        final_better_than_random = "not checked"
        best_better_than_random = "not checked"
        final_learning_status = "not checked"
        best_learning_status = "not checked"
        random_baseline_warning = (
            f"no matching baseline for max_cycles={args.max_cycles}, "
            f"team_reward_mode={args.team_reward_mode}"
        )
    else:
        random_team_baseline = float(baseline_info["random_team_return"])
        random_baseline_source = baseline_info["source"]
        final_better_than_random = better_than_random_status(
            final_eval_team_return,
            random_team_baseline,
        )
        best_better_than_random = better_than_random_status(
            best_eval_team_return,
            random_team_baseline,
        )
        final_learning_status = learning_status_from_better(final_better_than_random)
        best_learning_status = learning_status_from_better(best_better_than_random)
    health = training_health(
        last_td_loss=last_td_loss,
        collapse=collapse,
        final_eval_team_return=final_eval_team_return,
        random_team_baseline=random_team_baseline,
        last_q_tot_max=last_numeric_log_value(log_rows, "q_tot_max"),
    )
    final_epsilon = epsilon_by_step(args, total_steps)
    write_summary(
        run_dir=run_dir,
        total_episodes=args.episodes,
        total_steps=total_steps,
        final_eval_team_return=final_eval_team_return,
        final_td_loss=last_td_loss,
        final_epsilon=final_epsilon,
        best_eval_team_return=best_eval_team_return,
        best_eval_episode=best_eval_episode,
        best_checkpoint_path=best_checkpoint_path,
        random_team_baseline=random_team_baseline,
        final_better_than_random=final_better_than_random,
        best_better_than_random=best_better_than_random,
        random_baseline_source=random_baseline_source,
        random_baseline_warning=random_baseline_warning,
        final_learning_status=final_learning_status,
        best_learning_status=best_learning_status,
        collapse=collapse,
        coverage_stats=coverage_stats,
        health=health,
        args=args,
    )

    try:
        from experiments.qmix_mpe_rnn.plot_qmix_rnn_run import save_all_plots

        for plot_path in save_all_plots(run_dir):
            print(f"Plot saved to: {plot_path}")
    except Exception as error:
        print(f"warning: could not save plots for {run_dir}: {error}")

    print("\n=== Run summary ===")
    print(f"Run directory: {run_dir}")
    print(f"Final eval_team_return: {final_eval_team_return:.3f}")
    print(f"Final eval_mean_agent_return: {final_eval_mean_agent_return:.3f}")
    print(f"Final td_loss: {last_td_loss}")
    print(f"Final epsilon: {final_epsilon:.3f}")
    print(f"Best eval_team_return: {best_eval_team_return}")
    print(f"Best eval episode: {best_eval_episode}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Random team baseline: {random_team_baseline}")
    print(f"Algorithm label: {algorithm_label(args.mixer_type)}")
    print(f"Final better than random: {final_better_than_random}")
    print(f"Best better than random: {best_better_than_random}")
    print(f"Learning status final: {final_learning_status}")
    print(f"Learning status best: {best_learning_status}")
    print(f"Policy collapse risk: {collapse}")
    print(f"Collapse reason: {coverage_stats['collapse_reason']}")
    print(f"Action entropy mean: {coverage_stats['action_entropy_mean']:.4f}")
    print(f"Used action count min: {coverage_stats['used_action_count_min']}")
    print(f"Max action ratio max: {coverage_stats['max_action_ratio_max']:.4f}")
    print(f"Top2 action ratio max: {coverage_stats['top2_action_ratio_max']:.4f}")
    print(f"Training health: {health}")
    print(f"CSV saved to: {run_dir / 'train_log.csv'}")
    print(f"Config saved to: {run_dir / 'config.json'}")
    print(f"Summary saved to: {run_dir / 'summary.txt'}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
