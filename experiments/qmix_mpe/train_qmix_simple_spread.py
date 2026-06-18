# -*- coding: utf-8 -*-
"""Train a simplified MLP QMIX agent on MPE simple_spread_v3."""

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

from algorithms.qmix import AgentQNetwork, QMixer, ReplayBuffer, VDNMixer
from envs.mpe_env_factory import get_mpe_env_source, make_mpe_env


OUTPUT_BASE_DIR = ROOT_DIR / "outputs" / "qmix_mpe" / "simple_spread"
ENV_NAME = "simple_spread_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train simplified QMIX on MPE simple_spread_v3.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-cycles", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--start-steps", type=int, default=200)
    parser.add_argument("--update-after", type=int, default=200)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=10000)
    parser.add_argument("--min-epsilon", type=float, default=None)
    parser.add_argument("--epsilon-eval", type=float, default=0.0)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--loss-type", choices=("mse", "huber"), default="huber")
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--team-reward-mode", choices=("mean", "sum", "first"), default="mean")
    parser.add_argument("--mixer-type", choices=("qmix", "vdn"), default="qmix")
    parser.add_argument("--double-q", action="store_true", default=False)
    parser.add_argument("--debug-shapes", action="store_true")
    parser.add_argument("--use-layer-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixer-weight-clip", type=float, default=1.0)
    parser.add_argument("--mixer-bias-clip", type=float, default=5.0)
    parser.add_argument(
        "--mixer-weight-activation",
        choices=("softplus", "abs"),
        default="softplus",
    )
    parser.add_argument("--qtot-l2-coef", type=float, default=0.0)
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


def select_actions(
    q_networks: torch.nn.ModuleList,
    obs_array: np.ndarray,
    epsilon: float,
    action_dim: int,
    device: torch.device,
) -> np.ndarray:
    actions = []
    for agent_index, q_net in enumerate(q_networks):
        if random.random() < epsilon:
            actions.append(random.randrange(action_dim))
            continue
        obs_tensor = torch.as_tensor(obs_array[agent_index], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = q_net(obs_tensor)
        actions.append(int(torch.argmax(q_values, dim=1).item()))
    return np.asarray(actions, dtype=np.int64)


def soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * source_param.data)


def hard_update(source: torch.nn.Module, target: torch.nn.Module) -> None:
    target.load_state_dict(source.state_dict())


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


def ensure_shape(tensor: torch.Tensor, expected_shape: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape) != expected_shape:
        raise RuntimeError(
            f"{name} shape mismatch: expected {expected_shape}, got {tuple(tensor.shape)}"
        )


def debug_tensor(name: str, tensor: torch.Tensor) -> None:
    print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")


def train_update(
    q_networks: torch.nn.ModuleList,
    target_q_networks: torch.nn.ModuleList,
    mixer: torch.nn.Module,
    target_mixer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    tau: float,
    grad_clip: float,
    loss_type: str,
    qtot_l2_coef: float,
    double_q: bool,
    device: torch.device,
    update_index: int = 0,
    debug_shapes: bool = False,
) -> dict[str, float]:
    batch = replay_buffer.sample(batch_size, device)
    obs = batch["obs"]
    actions = batch["actions"].long()
    states = batch["state"]
    rewards = batch["reward"].view(batch_size, 1)
    next_obs = batch["next_obs"]
    next_states = batch["next_state"]
    dones = batch["done"].view(batch_size, 1)

    ensure_shape(obs, (batch_size, replay_buffer.n_agents, replay_buffer.obs_dim), "obs")
    ensure_shape(next_obs, (batch_size, replay_buffer.n_agents, replay_buffer.obs_dim), "next_obs")
    ensure_shape(actions, (batch_size, replay_buffer.n_agents), "actions")
    ensure_shape(states, (batch_size, replay_buffer.state_dim), "states")
    ensure_shape(next_states, (batch_size, replay_buffer.state_dim), "next_states")
    ensure_shape(rewards, (batch_size, 1), "rewards")
    ensure_shape(dones, (batch_size, 1), "dones")

    agent_q_values = []
    online_next_actions = []
    target_next_q_values = []
    all_agent_qs = []
    for agent_index, q_net in enumerate(q_networks):
        q_values = q_net(obs[:, agent_index, :])
        all_agent_qs.append(q_values)
        chosen_q = q_values.gather(1, actions[:, agent_index].unsqueeze(1)).squeeze(1)
        agent_q_values.append(chosen_q)

        with torch.no_grad():
            if double_q:
                online_next_q_values = q_net(next_obs[:, agent_index, :])
                online_next_actions.append(online_next_q_values.argmax(dim=1))
            target_q_values = target_q_networks[agent_index](next_obs[:, agent_index, :])
            if double_q:
                target_next_q = target_q_values.gather(
                    1,
                    online_next_actions[-1].unsqueeze(1),
                ).squeeze(1)
            else:
                target_next_q = target_q_values.max(dim=1).values
            target_next_q_values.append(target_next_q)

    agent_qs = torch.stack(agent_q_values, dim=1)
    q_tot = mixer(agent_qs, states)
    ensure_shape(agent_qs, (batch_size, replay_buffer.n_agents), "gathered_qs")
    ensure_shape(q_tot, (batch_size, 1), "current_q_tot")

    with torch.no_grad():
        target_agent_qs = torch.stack(target_next_q_values, dim=1)
        target_q_tot = target_mixer(target_agent_qs, next_states)
        ensure_shape(target_agent_qs, (batch_size, replay_buffer.n_agents), "target_agent_qs")
        ensure_shape(target_q_tot, (batch_size, 1), "target_q_tot")
        td_target = rewards + gamma * (1.0 - dones) * target_q_tot
        ensure_shape(td_target, q_tot.shape, "td_target")

    if debug_shapes and update_index < 3:
        print(f"\n[debug-shapes] update={update_index + 1}")
        debug_tensor("obs", obs)
        debug_tensor("actions", actions)
        debug_tensor("rewards", rewards)
        debug_tensor("dones", dones)
        debug_tensor("agent_qs", torch.stack(all_agent_qs, dim=1))
        debug_tensor("gathered_qs", agent_qs)
        debug_tensor("current_q_tot", q_tot)
        debug_tensor("target_q_tot", target_q_tot)
        debug_tensor("td_target", td_target)

    ensure_shape(q_tot, td_target.shape, "current_q_tot before loss")

    if loss_type == "mse":
        td_loss = F.mse_loss(q_tot, td_target)
    elif loss_type == "huber":
        td_loss = F.smooth_l1_loss(q_tot, td_target)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    raw_qtot_l2_loss = (q_tot ** 2).mean()
    if qtot_l2_coef > 0:
        qtot_l2_loss = raw_qtot_l2_loss
    else:
        qtot_l2_loss = torch.zeros((), dtype=q_tot.dtype, device=q_tot.device)
    loss = td_loss + float(qtot_l2_coef) * raw_qtot_l2_loss

    optimizer.zero_grad()
    loss.backward()
    trainable_params = list(q_networks.parameters()) + list(mixer.parameters())
    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable_params,
        max_norm=grad_clip,
    )
    optimizer.step()

    for q_net, target_q_net in zip(q_networks, target_q_networks):
        soft_update(q_net, target_q_net, tau)
    soft_update(mixer, target_mixer, tau)

    return {
        "td_loss": float(td_loss.item()),
        "total_loss": float(loss.item()),
        "qtot_l2_loss": float(qtot_l2_loss.detach().item()),
        "q_tot_mean": float(q_tot.detach().mean().item()),
        "q_tot_std": float(q_tot.detach().std(unbiased=False).item()),
        "q_tot_max": float(q_tot.detach().abs().max().item()),
        "target_q_tot_mean": float(td_target.detach().mean().item()),
        "target_q_tot_std": float(td_target.detach().std(unbiased=False).item()),
        "target_q_tot_max": float(td_target.detach().abs().max().item()),
        "grad_norm": float(grad_norm.item()),
    }


def evaluate_policy(
    q_networks: torch.nn.ModuleList,
    agents: list[str],
    obs_dim: int,
    action_dim: int,
    max_cycles: int,
    eval_episodes: int,
    seed: int,
    device: torch.device,
    epsilon_eval: float = 0.0,
    team_reward_mode: str = "mean",
) -> tuple[float, float]:
    eval_env = make_mpe_env(ENV_NAME, max_cycles=max_cycles)
    team_returns = []
    mean_agent_returns = []

    for episode in range(eval_episodes):
        observations, infos = reset_env(eval_env, seed=seed + episode)
        del infos
        agent_returns = {agent: 0.0 for agent in agents}
        episode_team_return = 0.0

        for _ in range(max_cycles):
            if not observations:
                break
            obs_array = obs_to_array(observations, agents, obs_dim)
            actions = select_actions(q_networks, obs_array, epsilon_eval, action_dim, device)
            action_dict = {
                agent: int(action)
                for agent, action in zip(agents, actions)
                if agent in observations
            }
            observations, rewards, terminations, truncations, infos = eval_env.step(action_dict)
            del infos

            episode_team_return += team_reward(rewards, team_reward_mode)
            for agent in agents:
                agent_returns[agent] += float(rewards.get(agent, 0.0))

            if done_flag(observations, terminations, truncations, agents):
                break

        team_returns.append(episode_team_return)
        mean_agent_returns.append(float(np.mean(list(agent_returns.values()))))

    eval_env.close()
    return float(np.mean(team_returns)), float(np.mean(mean_agent_returns))


def inspect_action_distribution(
    q_networks: torch.nn.ModuleList,
    agents: list[str],
    obs_dim: int,
    action_dim: int,
    max_cycles: int,
    episodes: int,
    seed: int,
    device: torch.device,
) -> dict[str, list[int]]:
    counts = {agent: [0 for _ in range(action_dim)] for agent in agents}
    env = make_mpe_env(ENV_NAME, max_cycles=max_cycles)
    for episode in range(episodes):
        observations, infos = reset_env(env, seed=seed + episode)
        del infos
        for _ in range(max_cycles):
            if not observations:
                break
            obs_array = obs_to_array(observations, agents, obs_dim)
            actions = select_actions(q_networks, obs_array, 0.0, action_dim, device)
            for agent, action in zip(agents, actions):
                counts[agent][int(action)] += 1
            action_dict = {
                agent: int(action)
                for agent, action in zip(agents, actions)
                if agent in observations
            }
            observations, rewards, terminations, truncations, infos = env.step(action_dict)
            del rewards, infos
            if done_flag(observations, terminations, truncations, agents):
                break
    env.close()
    return counts


def policy_collapse_risk(action_counts: dict[str, list[int]]) -> str:
    for counts in action_counts.values():
        total = sum(counts)
        if total > 0 and max(counts) / total > 0.95:
            return "high collapse risk"
    return "no obvious collapse"


def much_worse_than_random(
    final_eval_team_return: float | None,
    random_team_baseline: float | None,
) -> bool:
    if final_eval_team_return is None or random_team_baseline is None:
        return False
    if random_team_baseline < 0:
        return final_eval_team_return < random_team_baseline * 1.5
    return final_eval_team_return < random_team_baseline * 0.5


def training_health(
    last_td_loss: float | None,
    collapse: str = "no obvious collapse",
    final_eval_team_return: float | None = None,
    random_team_baseline: float | None = None,
    last_q_tot_mean: float | None = None,
    last_target_q_tot_mean: float | None = None,
    last_q_tot_max: float | None = None,
) -> str:
    if last_td_loss is None:
        return "warning: no updates were run"
    if not math.isfinite(last_td_loss):
        return "warning"
    if last_td_loss > 100:
        return "warning"
    if collapse == "high collapse risk":
        return "warning"
    if much_worse_than_random(final_eval_team_return, random_team_baseline):
        return "warning"
    for value in (last_q_tot_mean, last_target_q_tot_mean):
        if value is not None and math.isfinite(value) and abs(value) > 100:
            return "warning"
    if last_q_tot_max is not None and math.isfinite(last_q_tot_max) and abs(last_q_tot_max) > 500:
        return "warning"
    return "OK"


def baseline_path(max_cycles: int, team_reward_mode: str) -> Path:
    return (
        OUTPUT_BASE_DIR
        / "random_baselines"
        / f"random_baseline_cycles{max_cycles}_mode_{team_reward_mode}.json"
    )


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
        "qtot_l2_loss",
        "grad_norm",
        "q_tot_mean",
        "q_tot_std",
        "q_tot_max",
        "target_q_tot_mean",
        "target_q_tot_std",
        "target_q_tot_max",
        "epsilon",
        "eval_team_return",
        "eval_mean_agent_return",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_name(f".{csv_path.stem}.{uuid.uuid4().hex}.tmp{csv_path.suffix}")
    try:
        with tmp_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, csv_path)
    except PermissionError as error:
        print(
            f"warning: could not save train log because the file is in use: "
            f"{csv_path} ({error}); keeping temp file: {tmp_path}"
        )
    except OSError as error:
        print(
            f"warning: could not save train log: {csv_path} ({error}); "
            f"keeping temp file: {tmp_path}"
        )


def save_checkpoint(
    checkpoint_path: Path,
    q_networks: torch.nn.ModuleList,
    target_q_networks: torch.nn.ModuleList,
    mixer: torch.nn.Module,
    target_mixer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: dict,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(".tmp.pt")
    payload = {
        "q_networks": [network.state_dict() for network in q_networks],
        "target_q_networks": [network.state_dict() for network in target_q_networks],
        "mixer": mixer.state_dict(),
        "target_mixer": target_mixer.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metadata": metadata,
    }
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, checkpoint_path)
    except (OSError, RuntimeError) as error:
        print(
            f"warning: could not save checkpoint: {checkpoint_path} ({error}); "
            f"temp file may remain: {tmp_path}"
        )


def write_summary(
    run_dir: Path,
    total_episodes: int,
    total_steps: int,
    final_eval_team_return: float | None,
    final_td_loss: float | None,
    final_epsilon: float,
    collapse: str,
    health: str,
    random_team_baseline: float | None,
    qmix_better_than_random: str,
    random_baseline_source: str,
    random_baseline_max_cycles: int | str,
    random_baseline_team_reward_mode: str,
    random_baseline_warning: str | None,
    grad_clip: float,
    loss_type: str,
    reward_scale: float,
    team_reward_mode: str,
    epsilon_end: float,
    min_epsilon: float,
    epsilon_eval: float,
    use_layer_norm: bool,
    mixer_weight_clip: float,
    mixer_bias_clip: float,
    mixer_weight_activation: str,
    qtot_l2_coef: float,
    mixer_type: str,
    double_q: bool,
    learning_status: str,
) -> None:
    lines = [
        f"run directory: {run_dir}",
        f"total episodes: {total_episodes}",
        f"total steps: {total_steps}",
        f"final eval_team_return: {final_eval_team_return}",
        f"final td_loss: {final_td_loss}",
        f"final epsilon: {final_epsilon}",
        f"random team baseline: {'not found' if random_team_baseline is None else random_team_baseline}",
        f"qmix better than random: {qmix_better_than_random}",
        f"learning status: {learning_status}",
        f"random baseline source: {random_baseline_source}",
        f"random baseline max_cycles: {random_baseline_max_cycles}",
        f"random baseline team_reward_mode: {random_baseline_team_reward_mode}",
        f"grad_clip: {grad_clip}",
        f"loss_type: {loss_type}",
        f"reward_scale: {reward_scale}",
        f"team_reward_mode: {team_reward_mode}",
        f"epsilon_end: {epsilon_end}",
        f"min_epsilon: {min_epsilon}",
        f"epsilon_eval: {epsilon_eval}",
        f"use_layer_norm: {use_layer_norm}",
        f"mixer_weight_clip: {mixer_weight_clip}",
        f"mixer_bias_clip: {mixer_bias_clip}",
        f"mixer_weight_activation: {mixer_weight_activation}",
        f"qtot_l2_coef: {qtot_l2_coef}",
        f"mixer_type: {mixer_type}",
        f"double_q: {double_q}",
        f"policy collapse risk: {collapse}",
        f"training health: {health}",
    ]
    if random_baseline_warning:
        lines.append(f"random baseline warning: {random_baseline_warning}")
    tmp_path = run_dir / "summary.tmp.txt"
    summary_path = run_dir / "summary.txt"
    try:
        with tmp_path.open("w", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")
        os.replace(tmp_path, summary_path)
    except OSError as error:
        print(
            f"warning: could not save summary: {summary_path} ({error}); "
            f"temp file may remain: {tmp_path}"
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = get_next_run_dir(OUTPUT_BASE_DIR)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = make_mpe_env(ENV_NAME, max_cycles=args.max_cycles)
    source = get_mpe_env_source(env)
    observations, infos = reset_env(env, seed=args.seed)
    del infos
    agents = list(env.possible_agents)
    if not agents:
        raise RuntimeError("simple_spread_v3 returned no possible_agents.")

    obs_dim = int(env.observation_space(agents[0]).shape[0])
    action_dim = int(env.action_space(agents[0]).n)
    n_agents = len(agents)
    state_dim = n_agents * obs_dim
    for agent in agents:
        if int(env.observation_space(agent).shape[0]) != obs_dim:
            raise RuntimeError("This simplified QMIX script expects equal obs_dim for all agents.")
        if int(env.action_space(agent).n) != action_dim:
            raise RuntimeError("This simplified QMIX script expects equal action_dim for all agents.")

    q_networks = torch.nn.ModuleList(
        [AgentQNetwork(obs_dim, action_dim, args.hidden_dim) for _ in agents]
    ).to(device)
    target_q_networks = torch.nn.ModuleList(
        [AgentQNetwork(obs_dim, action_dim, args.hidden_dim) for _ in agents]
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
    hard_update(q_networks, target_q_networks)
    hard_update(mixer, target_mixer)

    optimizer = torch.optim.Adam(
        list(q_networks.parameters()) + list(mixer.parameters()),
        lr=args.learning_rate,
    )
    replay_buffer = ReplayBuffer(args.buffer_size, n_agents, obs_dim, state_dim)

    metadata = {
        "algorithm": "simplified_mlp_qmix",
        "env": ENV_NAME,
        "agent_names": agents,
        "n_agents": n_agents,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "hidden_dim": int(args.hidden_dim),
        "max_cycles": int(args.max_cycles),
        "mixer_type": args.mixer_type,
        "double_q": bool(args.double_q),
        "use_layer_norm": bool(args.use_layer_norm),
        "mixer_weight_clip": float(args.mixer_weight_clip),
        "mixer_bias_clip": float(args.mixer_bias_clip),
        "mixer_weight_activation": args.mixer_weight_activation,
        "device": str(device),
    }
    save_config(args, run_dir, source, metadata)

    print(f"Loaded {ENV_NAME} from: {source}")
    print(f"Run directory: {run_dir}")
    print("QMIX: individual MLP Q networks + monotonic mixer")
    print(f"agents={agents}, obs_dim={obs_dim}, action_dim={action_dim}, state_dim={state_dim}")

    log_rows: list[dict] = []
    total_steps = 0
    update_count = 0
    final_eval_team_return = None
    final_eval_mean_agent_return = None
    last_td_loss = None

    for episode in range(1, args.episodes + 1):
        observations, infos = reset_env(env, seed=args.seed + episode)
        del infos
        episode_team_return = 0.0
        episode_update_losses = []

        for _ in range(args.max_cycles):
            if not observations:
                break
            obs_array = obs_to_array(observations, agents, obs_dim)
            state = make_state(obs_array)
            epsilon = epsilon_by_step(args, total_steps)

            if total_steps < args.start_steps:
                actions = np.asarray([env.action_space(agent).sample() for agent in agents], dtype=np.int64)
            else:
                actions = select_actions(q_networks, obs_array, epsilon, action_dim, device)

            action_dict = {
                agent: int(action)
                for agent, action in zip(agents, actions)
                if agent in observations
            }
            next_observations, rewards, terminations, truncations, infos = env.step(action_dict)
            del infos

            reward = team_reward(rewards, args.team_reward_mode)
            episode_team_return += reward
            done = done_flag(next_observations, terminations, truncations, agents)
            next_obs_array = next_obs_to_array(next_observations, agents, obs_dim, obs_array)
            next_state = make_state(next_obs_array)

            replay_buffer.add(
                obs=obs_array,
                state=state,
                actions=actions,
                reward=reward * args.reward_scale,
                next_obs=next_obs_array,
                next_state=next_state,
                done=done,
            )
            total_steps += 1

            if (
                total_steps >= args.update_after
                and total_steps % max(1, args.update_every) == 0
                and len(replay_buffer) >= args.batch_size
            ):
                update_info = train_update(
                    q_networks=q_networks,
                    target_q_networks=target_q_networks,
                    mixer=mixer,
                    target_mixer=target_mixer,
                    optimizer=optimizer,
                    replay_buffer=replay_buffer,
                    batch_size=args.batch_size,
                    gamma=args.gamma,
                    tau=args.tau,
                    grad_clip=args.grad_clip,
                    loss_type=args.loss_type,
                    qtot_l2_coef=args.qtot_l2_coef,
                    double_q=args.double_q,
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
                        "qtot_l2_loss": update_info["qtot_l2_loss"],
                        "grad_norm": update_info["grad_norm"],
                        "q_tot_mean": update_info["q_tot_mean"],
                        "q_tot_std": update_info["q_tot_std"],
                        "q_tot_max": update_info["q_tot_max"],
                        "target_q_tot_mean": update_info["target_q_tot_mean"],
                        "target_q_tot_std": update_info["target_q_tot_std"],
                        "target_q_tot_max": update_info["target_q_tot_max"],
                        "epsilon": epsilon_by_step(args, total_steps),
                        "eval_team_return": None,
                        "eval_mean_agent_return": None,
                    }
                )

            if done:
                break
            observations = next_observations

        eval_team_return = None
        eval_mean_agent_return = None
        if args.eval_interval > 0 and episode % args.eval_interval == 0:
            eval_team_return, eval_mean_agent_return = evaluate_policy(
                q_networks=q_networks,
                agents=agents,
                obs_dim=obs_dim,
                action_dim=action_dim,
                max_cycles=args.max_cycles,
                eval_episodes=args.eval_episodes,
                seed=args.seed + episode * 1000,
                device=device,
                epsilon_eval=args.epsilon_eval,
                team_reward_mode=args.team_reward_mode,
            )
            final_eval_team_return = eval_team_return
            final_eval_mean_agent_return = eval_mean_agent_return
            if log_rows:
                log_rows[-1]["eval_team_return"] = eval_team_return
                log_rows[-1]["eval_mean_agent_return"] = eval_mean_agent_return
            else:
                log_rows.append(
                    {
                        "update": update_count,
                        "episode": episode,
                        "total_steps": total_steps,
                        "train_team_return": episode_team_return,
                        "td_loss": None,
                        "total_loss": None,
                        "qtot_l2_loss": None,
                        "grad_norm": None,
                        "q_tot_mean": None,
                        "q_tot_std": None,
                        "q_tot_max": None,
                        "target_q_tot_mean": None,
                        "target_q_tot_std": None,
                        "target_q_tot_max": None,
                        "epsilon": epsilon_by_step(args, total_steps),
                        "eval_team_return": eval_team_return,
                        "eval_mean_agent_return": eval_mean_agent_return,
                    }
                )
            save_checkpoint(
                checkpoint_dir / f"qmix_episode_{episode:04d}.pt",
                q_networks,
                target_q_networks,
                mixer,
                target_mixer,
                optimizer,
                metadata,
            )

        if eval_team_return is not None:
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
            q_networks=q_networks,
            agents=agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_cycles=args.max_cycles,
            eval_episodes=args.eval_episodes,
            seed=args.seed + args.episodes * 1000 + 500,
            device=device,
            epsilon_eval=args.epsilon_eval,
            team_reward_mode=args.team_reward_mode,
        )
        if log_rows:
            log_rows[-1]["eval_team_return"] = final_eval_team_return
            log_rows[-1]["eval_mean_agent_return"] = final_eval_mean_agent_return
            save_log(log_rows, run_dir / "train_log.csv")

    save_checkpoint(
        checkpoint_dir / "qmix_final.pt",
        q_networks,
        target_q_networks,
        mixer,
        target_mixer,
        optimizer,
        metadata,
    )
    save_log(log_rows, run_dir / "train_log.csv")

    action_counts = inspect_action_distribution(
        q_networks=q_networks,
        agents=agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_cycles=args.max_cycles,
        episodes=min(5, max(1, args.eval_episodes)),
        seed=args.seed + 900000,
        device=device,
    )
    collapse = policy_collapse_risk(action_counts)
    final_epsilon = epsilon_by_step(args, total_steps)
    baseline_info = load_random_baseline(args.max_cycles, args.team_reward_mode)
    random_baseline_warning = None
    if baseline_info is None:
        random_team_baseline = None
        random_baseline_source = "not found"
        random_baseline_max_cycles = "not found"
        random_baseline_team_reward_mode = "not found"
        qmix_better_than_random = "not checked"
        learning_status = "not checked"
        random_baseline_warning = (
            f"no matching baseline for max_cycles={args.max_cycles}, "
            f"team_reward_mode={args.team_reward_mode}"
        )
        print("warning: no matching random baseline found.")
        print("Please run:")
        print(
            "python experiments/qmix_mpe/run_random_simple_spread_baseline.py "
            f"--episodes 100 --max-cycles {args.max_cycles} "
            f"--team-reward-mode {args.team_reward_mode}"
        )
    else:
        random_team_baseline = float(baseline_info["random_team_return"])
        random_baseline_source = baseline_info["source"]
        random_baseline_max_cycles = int(baseline_info["max_cycles"])
        random_baseline_team_reward_mode = str(baseline_info["team_reward_mode"])
        qmix_better_than_random = (
            "better"
            if final_eval_team_return is not None
            and final_eval_team_return > random_team_baseline
            else "worse"
        )
        learning_status = (
            "better than random"
            if qmix_better_than_random == "better"
            else "worse than random"
        )
    health = training_health(
        last_td_loss=last_td_loss,
        collapse=collapse,
        final_eval_team_return=final_eval_team_return,
        random_team_baseline=random_team_baseline,
        last_q_tot_mean=last_numeric_log_value(log_rows, "q_tot_mean"),
        last_target_q_tot_mean=last_numeric_log_value(log_rows, "target_q_tot_mean"),
        last_q_tot_max=last_numeric_log_value(log_rows, "q_tot_max"),
    )
    if qmix_better_than_random == "worse":
        health = "warning"
    write_summary(
        run_dir=run_dir,
        total_episodes=args.episodes,
        total_steps=total_steps,
        final_eval_team_return=final_eval_team_return,
        final_td_loss=last_td_loss,
        final_epsilon=final_epsilon,
        collapse=collapse,
        health=health,
        random_team_baseline=random_team_baseline,
        qmix_better_than_random=qmix_better_than_random,
        learning_status=learning_status,
        random_baseline_source=random_baseline_source,
        random_baseline_max_cycles=random_baseline_max_cycles,
        random_baseline_team_reward_mode=random_baseline_team_reward_mode,
        random_baseline_warning=random_baseline_warning,
        grad_clip=args.grad_clip,
        loss_type=args.loss_type,
        reward_scale=args.reward_scale,
        team_reward_mode=args.team_reward_mode,
        epsilon_end=args.epsilon_end,
        min_epsilon=args.min_epsilon,
        epsilon_eval=args.epsilon_eval,
        use_layer_norm=args.use_layer_norm,
        mixer_weight_clip=args.mixer_weight_clip,
        mixer_bias_clip=args.mixer_bias_clip,
        mixer_weight_activation=args.mixer_weight_activation,
        qtot_l2_coef=args.qtot_l2_coef,
        mixer_type=args.mixer_type,
        double_q=args.double_q,
    )

    try:
        from experiments.qmix_mpe.plot_qmix_simple_spread_run import save_all_plots

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
    print(f"Random team baseline: {random_team_baseline}")
    print(f"QMIX better than random: {qmix_better_than_random}")
    print(f"Learning status: {learning_status}")
    print(f"Policy collapse risk: {collapse}")
    print(f"Training health: {health}")
    print(f"CSV saved to: {run_dir / 'train_log.csv'}")
    print(f"Config saved to: {run_dir / 'config.json'}")
    print(f"Summary saved to: {run_dir / 'summary.txt'}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
