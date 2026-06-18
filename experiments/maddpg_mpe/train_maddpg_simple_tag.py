# -*- coding: utf-8 -*-
"""在 MPE simple_tag_v3 上训练离散动作 MADDPG。

simple_tag_v3 是追捕-逃逸对抗任务：
- adversary_0、adversary_1、adversary_2 是追捕者，obs_dim=16；
- agent_0 是逃避者，obs_dim=14；
- 4 个 agent 的动作空间都是 Discrete(5)。

MADDPG 数据流：
- decentralized actor：每个 agent 的 actor 只看自己的 obs，输出 5 维动作 logits；
- centralized critic：每个 agent 的 critic 训练时看所有 obs 和所有 one-hot action，
  即 critic_input_dim = 62 + 20 = 82，输出该 agent 的 Q value。
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algorithms.maddpg.maddpg_trainer import MADDPGTrainer
from experiments.maddpg_mpe.plot_maddpg_simple_tag_run import (
    save_all_outputs_from_rows,
    save_all_plots_from_rows,
)


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
GLOBAL_OBS_DIM = 62
GLOBAL_ACTION_DIM = 20
CRITIC_INPUT_DIM = 82

OUTPUT_BASE_DIR = ROOT_DIR / "outputs" / "maddpg_mpe" / "simple_tag"

DEFAULT_TRAINING_ARGS = {
    "batch_size": 1024,
    "buffer_size": 200000,
    "learning_rate_actor": 1e-3,
    "learning_rate_critic": 1e-3,
    "gamma": 0.95,
    "tau": 0.01,
    "hidden_dim": 128,
    "update_every": 50,
    "eval_episodes": 5,
}
PAPER_PRESET_ARGS = {
    "batch_size": 1024,
    "buffer_size": 1000000,
    "learning_rate_actor": 0.01,
    "learning_rate_critic": 0.01,
    "gamma": 0.95,
    "tau": 0.01,
    "hidden_dim": 128,
    "update_every": 100,
    "eval_episodes": 50,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MADDPG on MPE simple_tag_v3."
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-cycles", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--learning-rate-actor", type=float, default=None)
    parser.add_argument("--learning-rate-critic", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--start-steps", type=int, default=1000)
    parser.add_argument("--update-after", type=int, default=1000)
    parser.add_argument("--update-every", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-updates", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=20)
    parser.add_argument("--use-policy-ensemble", action="store_true")
    parser.add_argument("--ensemble-size", type=int, default=2)
    parser.add_argument("--paper-preset", action="store_true")
    parser.add_argument("--save-debug-plots", action="store_true")
    parser.add_argument("--actor-entropy-coef", type=float, default=0.0)
    parser.add_argument(
        "--actor-action-mode",
        choices=("gumbel_hard", "gumbel_soft", "softmax"),
        default="gumbel_hard",
    )
    parser.add_argument("--gumbel-tau", type=float, default=1.0)
    parser.add_argument(
        "--adversary-reward-scale",
        type=float,
        default=1.0,
        help="Scale adversary rewards only for replay buffer / critic training.",
    )
    parser.add_argument(
        "--prey-reward-scale",
        type=float,
        default=1.0,
        help="Scale prey rewards only for replay buffer / critic training.",
    )
    parser.add_argument(
        "--plot-interval",
        type=int,
        default=0,
        help="Save plots every N episodes during training; 0 means only save final plots.",
    )
    args = parser.parse_args()
    apply_training_defaults(args)
    return args


def apply_training_defaults(args: argparse.Namespace) -> None:
    """paper preset 只填充用户未显式传入的参数。"""

    defaults = PAPER_PRESET_ARGS if args.paper_preset else DEFAULT_TRAINING_ARGS
    for name, default_value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, default_value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_simple_tag():
    """优先使用 mpe2，失败时回退到 pettingzoo.mpe。"""

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


def get_next_run_dir(base_dir: Path) -> Path:
    """创建下一个递增编号 run 目录，避免覆盖历史实验输出。"""

    base_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, 10000):
        run_dir = base_dir / f"run{index:03d}"
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"无法在 {base_dir} 下创建新的 runXXX 目录。")


def one_hot(action: int, action_dim: int = ACTION_DIM) -> np.ndarray:
    value = np.zeros(action_dim, dtype=np.float32)
    value[int(action)] = 1.0
    return value


def observations_to_arrays(observations: dict) -> dict[str, np.ndarray]:
    obs_arrays = {}
    for agent in AGENT_NAMES:
        if agent not in observations:
            raise KeyError(f"环境 observation 中缺少 {agent}")
        obs = np.asarray(observations[agent], dtype=np.float32)
        expected_dim = OBS_DIMS[agent]
        if obs.shape[-1] != expected_dim:
            raise ValueError(
                f"{agent} obs_dim 应为 {expected_dim}，实际为 {obs.shape[-1]}"
            )
        obs_arrays[agent] = obs
    return obs_arrays


def next_observations_to_arrays(
    next_observations: dict,
    fallback_obs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """episode 结束时 parallel_env 可能返回空 obs，此时用当前 obs 占位。

    因为 done=1，critic target 中的 bootstrap 项会被 (1 - done) 清零，占位 next_obs
    不会影响 TD target，只是为了保持 batch 形状完整。
    """

    if all(agent in next_observations for agent in AGENT_NAMES):
        return observations_to_arrays(next_observations)
    return {agent: fallback_obs[agent].copy() for agent in AGENT_NAMES}


def done_dict(terminations: dict, truncations: dict) -> dict[str, bool]:
    return {
        agent: bool(terminations.get(agent, False) or truncations.get(agent, False))
        for agent in AGENT_NAMES
    }


def is_episode_done(
    next_observations: dict,
    terminations: dict,
    truncations: dict,
) -> bool:
    if not next_observations:
        return True
    dones = done_dict(terminations, truncations)
    return all(dones.values())


def random_actions() -> tuple[dict[str, int], dict[str, np.ndarray]]:
    actions = {
        agent: int(np.random.randint(ACTION_DIM))
        for agent in AGENT_NAMES
    }
    one_hot_actions = {
        agent: one_hot(action)
        for agent, action in actions.items()
    }
    return actions, one_hot_actions


def adversary_team_return(agent_returns: dict[str, float]) -> float:
    return float(sum(agent_returns[agent] for agent in ADVERSARY_AGENTS))


def scale_rewards_for_training(
    raw_rewards: dict,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Scale rewards only for replay buffer / critic target training.

    Raw environment rewards are still used for terminal printing, train_log.csv,
    evaluation, plots, and summary.txt. This keeps the human-facing metrics
    comparable with the random baselines while making critic targets numerically
    easier to learn.
    """

    scaled_rewards = {}
    for agent in AGENT_NAMES:
        reward = float(raw_rewards.get(agent, 0.0))
        if agent in ADVERSARY_AGENTS:
            reward *= float(args.adversary_reward_scale)
        else:
            reward *= float(args.prey_reward_scale)
        scaled_rewards[agent] = reward
    return scaled_rewards


def approximate_touch_count(adversary_return: float | None) -> float | None:
    """基于 simple_tag adversary reward 的近似碰撞统计。

    simple_tag 中 adversary 团队回报大致和抓到 prey 的碰撞次数相关；这里先用
    adversary_team_return / 10.0 作为 touch_count 近似，便于跟踪追捕能力。
    """

    if adversary_return is None:
        return None
    return float(adversary_return) / 10.0


def epsilon_by_update(args: argparse.Namespace, update_count: int) -> float:
    if args.epsilon_decay_updates <= 0:
        return float(args.epsilon_end)
    fraction = min(1.0, update_count / float(args.epsilon_decay_updates))
    return float(args.epsilon_start + fraction * (args.epsilon_end - args.epsilon_start))


def save_config(args: argparse.Namespace, run_dir: Path, source: str) -> None:
    config_path = run_dir / "config.json"
    tmp_path = run_dir / "config.tmp.json"
    config = vars(args).copy()
    config.update(
        {
            "env": "simple_tag_v3",
            "env_source": source,
            "agent_names": AGENT_NAMES,
            "adversary_agents": ADVERSARY_AGENTS,
            "prey_agent": PREY_AGENT,
            "obs_dims": OBS_DIMS,
            "action_dim": ACTION_DIM,
            "global_obs_dim": GLOBAL_OBS_DIM,
            "global_action_dim": GLOBAL_ACTION_DIM,
            "critic_input_dim": CRITIC_INPUT_DIM,
            "algorithm": "MADDPG with discrete straight-through Gumbel-Softmax",
            "use_policy_ensemble": bool(args.use_policy_ensemble),
            "ensemble_size": int(args.ensemble_size),
            "paper_preset": bool(args.paper_preset),
            "actor_entropy_coef": float(args.actor_entropy_coef),
            "actor_action_mode": args.actor_action_mode,
            "gumbel_tau": float(args.gumbel_tau),
            "adversary_reward_scale": float(args.adversary_reward_scale),
            "prey_reward_scale": float(args.prey_reward_scale),
            "plot_interval": int(args.plot_interval),
        }
    )

    try:
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)
        os.replace(tmp_path, config_path)
    except OSError as error:
        print(f"warning: 保存 config.json 失败: {config_path} ({error})")
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def save_log(rows: list[dict], csv_path: Path) -> None:
    tmp_path = csv_path.with_name("train_log.tmp.csv")
    fieldnames = [
        "episode",
        "total_steps",
        "adversary_team_return",
        "prey_return",
        "adversary_0_return",
        "adversary_1_return",
        "adversary_2_return",
        "agent_0_return",
        "train_touch_count",
        "mean_critic_loss",
        "mean_actor_loss",
        "mean_actor_entropy",
        "epsilon",
        "eval_self_play_adversary_return",
        "eval_self_play_prey_return",
        "eval_adv_vs_random_prey_adversary_return",
        "eval_adv_vs_random_prey_prey_return",
        "eval_random_adv_vs_prey_adversary_return",
        "eval_random_adv_vs_prey_prey_return",
        "eval_self_play_touch_count",
        "eval_adv_vs_random_prey_touch_count",
        "eval_random_adv_vs_prey_touch_count",
    ]
    try:
        with tmp_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, csv_path)
    except OSError as error:
        print(f"warning: 保存 train_log.csv 失败: {csv_path} ({error})")
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def curve_values(rows: list[dict], key: str) -> list[float]:
    """把未评估 episode 的 None 转成 NaN，方便 matplotlib 跳过。"""

    return [
        np.nan if row[key] is None else row[key]
        for row in rows
    ]


def format_optional(value: float | None) -> str:
    return "None" if value is None else f"{value:.3f}"


def finite_float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def update_actor_entropy_value(update_info) -> float | None:
    """从一次 update 的返回值中取 actor entropy，确保写入 CSV 的是有效数字。"""

    entropy = finite_float_or_none(getattr(update_info, "mean_actor_entropy", None))
    if entropy is not None:
        return entropy

    actor_entropies = getattr(update_info, "actor_entropies", None)
    if actor_entropies:
        values = [
            finite_float_or_none(value)
            for value in actor_entropies.values()
        ]
        values = [value for value in values if value is not None]
        if values:
            return float(np.mean(values))
    return None


def save_reward_curve(
    rows: list[dict],
    png_path: Path,
    save_debug_plots: bool = False,
) -> list[Path]:
    """保存完整 MADDPG simple_tag 图表。

    参数名保留为 png_path，是为了少改主训练流程；实际会在同一个 run 目录下生成
    默认生成核心图；开启 save_debug_plots 时额外生成详细诊断图。
    """

    return save_all_plots_from_rows(
        rows,
        png_path.parent,
        save_debug_plots_enabled=save_debug_plots,
    )


def save_checkpoints(trainer: MADDPGTrainer, checkpoint_root: Path, episode: int) -> None:
    for agent_name, maddpg_agent in trainer.agents.items():
        agent_dir = checkpoint_root / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = agent_dir / f"episode_{episode:04d}.pt"
        tmp_path = agent_dir / f"episode_{episode:04d}.tmp.pt"
        try:
            maddpg_agent.save(str(tmp_path))
            os.replace(tmp_path, checkpoint_path)
        except (OSError, RuntimeError) as error:
            print(f"warning: 保存 checkpoint 失败: {checkpoint_path} ({error})")
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def select_eval_actions(
    eval_env,
    trainer: MADDPGTrainer,
    observations: dict,
    mode: str,
) -> dict[str, int]:
    """按 evaluation mode 选择动作。

    eval 时不使用 epsilon 探索：trained policy 使用 argmax action；random policy
    直接从对应 action_space.sample() 采样，作为随机基线。
    """

    obs_arrays = observations_to_arrays(observations)
    trained_actions, _ = trainer.act(obs_arrays, epsilon=0.0, explore=False)

    if mode == "self_play_eval":
        return trained_actions

    if mode == "adv_vs_random_prey_eval":
        actions = trained_actions.copy()
        actions[PREY_AGENT] = int(eval_env.action_space(PREY_AGENT).sample())
        return actions

    if mode == "random_adv_vs_prey_eval":
        actions = trained_actions.copy()
        for agent in ADVERSARY_AGENTS:
            actions[agent] = int(eval_env.action_space(agent).sample())
        return actions

    raise ValueError(f"未知 evaluation mode: {mode}")


def evaluate_policy(
    simple_tag_v3,
    trainer: MADDPGTrainer,
    max_cycles: int,
    eval_episodes: int,
    seed: int,
    mode: str,
) -> tuple[float, float]:
    """评估阶段不加探索噪声，所有 actor 使用 argmax action。"""

    eval_env = simple_tag_v3.parallel_env(max_cycles=max_cycles)
    adversary_returns = []
    prey_returns = []

    for index, agent in enumerate(AGENT_NAMES):
        try:
            eval_env.action_space(agent).seed(seed + 100 + index)
        except AttributeError:
            pass

    for episode in range(eval_episodes):
        observations, infos = reset_env(eval_env, seed=seed + episode)
        del infos
        episode_returns = {agent: 0.0 for agent in AGENT_NAMES}

        for _ in range(max_cycles):
            if not observations:
                break
            actions = select_eval_actions(
                eval_env=eval_env,
                trainer=trainer,
                observations=observations,
                mode=mode,
            )
            next_observations, rewards, terminations, truncations, infos = eval_env.step(actions)
            del infos

            for agent in AGENT_NAMES:
                episode_returns[agent] += float(rewards.get(agent, 0.0))

            if is_episode_done(next_observations, terminations, truncations):
                break
            observations = next_observations

        adversary_returns.append(adversary_team_return(episode_returns))
        prey_returns.append(float(episode_returns[PREY_AGENT]))

    eval_env.close()
    return float(np.mean(adversary_returns)), float(np.mean(prey_returns))


def validate_env(env) -> None:
    missing_agents = [agent for agent in AGENT_NAMES if agent not in env.possible_agents]
    if missing_agents:
        raise RuntimeError(f"simple_tag_v3 缺少这些 agent: {missing_agents}")

    for agent in AGENT_NAMES:
        action_space = env.action_space(agent)
        if not hasattr(action_space, "n") or int(action_space.n) != ACTION_DIM:
            raise RuntimeError(f"{agent} 动作空间应为 Discrete({ACTION_DIM})")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    simple_tag_v3, source = load_simple_tag()
    run_dir = get_next_run_dir(OUTPUT_BASE_DIR)
    checkpoint_root = run_dir / "checkpoints"
    for agent in AGENT_NAMES:
        (checkpoint_root / agent).mkdir(parents=True, exist_ok=True)
    save_config(args, run_dir, source)

    env = simple_tag_v3.parallel_env(max_cycles=args.max_cycles)
    validate_env(env)
    for index, agent in enumerate(AGENT_NAMES):
        try:
            env.action_space(agent).seed(args.seed + index)
        except AttributeError:
            pass

    trainer = MADDPGTrainer(
        agent_names=AGENT_NAMES,
        obs_dims=OBS_DIMS,
        action_dim=ACTION_DIM,
        hidden_dim=args.hidden_dim,
        actor_lr=args.learning_rate_actor,
        critic_lr=args.learning_rate_critic,
        gamma=args.gamma,
        tau=args.tau,
        buffer_size=args.buffer_size,
        ensemble_size=args.ensemble_size if args.use_policy_ensemble else 1,
        actor_entropy_coef=args.actor_entropy_coef,
        actor_action_mode=args.actor_action_mode,
        gumbel_tau=args.gumbel_tau,
    )

    print(f"Loaded simple_tag_v3 from: {source}")
    print(f"Run directory: {run_dir}")
    print("MADDPG: decentralized actors + centralized critics")
    print(f"critic input dim: {CRITIC_INPUT_DIM} = obs {GLOBAL_OBS_DIM} + actions {GLOBAL_ACTION_DIM}")
    print(
        "reward scales for training only: "
        f"adversary={args.adversary_reward_scale}, prey={args.prey_reward_scale}"
    )

    total_steps = 0
    update_count = 0
    log_rows = []

    for episode in range(1, args.episodes + 1):
        observations, infos = reset_env(env, seed=args.seed + episode)
        del infos
        if args.use_policy_ensemble:
            adv_policy_id = int(np.random.randint(args.ensemble_size))
            prey_policy_id = int(np.random.randint(args.ensemble_size))
        else:
            adv_policy_id = 0
            prey_policy_id = 0
        episode_returns = {agent: 0.0 for agent in AGENT_NAMES}
        episode_critic_losses = []
        episode_actor_losses = []
        actor_entropy_values = []

        for _ in range(args.max_cycles):
            obs_arrays = observations_to_arrays(observations)
            epsilon = epsilon_by_update(args, update_count)

            if total_steps < args.start_steps:
                actions, one_hot_actions = random_actions()
            else:
                actions, one_hot_actions = trainer.act(
                    obs_arrays,
                    epsilon=epsilon,
                    explore=True,
                    adv_policy_id=adv_policy_id,
                    prey_policy_id=prey_policy_id,
                )

            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            del infos
            total_steps += 1

            dones = done_dict(terminations, truncations)
            next_obs_arrays = next_observations_to_arrays(next_observations, obs_arrays)
            scaled_rewards = scale_rewards_for_training(rewards, args)
            trainer.add_transition(
                obs=obs_arrays,
                actions=actions,
                one_hot_actions=one_hot_actions,
                rewards=scaled_rewards,
                next_obs=next_obs_arrays,
                dones=dones,
                adv_policy_id=adv_policy_id,
                prey_policy_id=prey_policy_id,
            )

            for agent in AGENT_NAMES:
                episode_returns[agent] += float(rewards.get(agent, 0.0))

            if (
                total_steps >= args.update_after
                and total_steps % max(1, args.update_every) == 0
                and len(trainer.replay_buffer) >= args.batch_size
                and (
                    not args.use_policy_ensemble
                    or trainer.count_policy_samples(adv_policy_id, prey_policy_id)
                    >= args.batch_size
                )
            ):
                update_info = trainer.update(
                    args.batch_size,
                    adv_policy_id=adv_policy_id,
                    prey_policy_id=prey_policy_id,
                    filter_policy_ids=args.use_policy_ensemble,
                )
                update_count += 1
                episode_critic_losses.append(update_info.mean_critic_loss)
                episode_actor_losses.append(update_info.mean_actor_loss)
                actor_entropy = update_actor_entropy_value(update_info)
                if actor_entropy is not None:
                    actor_entropy_values.append(actor_entropy)

            if is_episode_done(next_observations, terminations, truncations):
                break
            observations = next_observations

        mean_critic_loss = (
            float(np.mean(episode_critic_losses))
            if episode_critic_losses
            else None
        )
        mean_actor_loss = (
            float(np.mean(episode_actor_losses))
            if episode_actor_losses
            else None
        )
        mean_actor_entropy = (
            float(np.mean(actor_entropy_values))
            if actor_entropy_values
            else None
        )
        epsilon = epsilon_by_update(args, update_count)

        eval_self_adv_return = None
        eval_self_prey_return = None
        eval_adv_random_adv_return = None
        eval_adv_random_prey_return = None
        eval_random_adv_return = None
        eval_random_prey_return = None
        if args.eval_interval > 0 and episode % args.eval_interval == 0:
            eval_self_adv_return, eval_self_prey_return = evaluate_policy(
                simple_tag_v3=simple_tag_v3,
                trainer=trainer,
                max_cycles=args.max_cycles,
                eval_episodes=args.eval_episodes,
                seed=args.seed + episode * 1000,
                mode="self_play_eval",
            )
            eval_adv_random_adv_return, eval_adv_random_prey_return = evaluate_policy(
                simple_tag_v3=simple_tag_v3,
                trainer=trainer,
                max_cycles=args.max_cycles,
                eval_episodes=args.eval_episodes,
                seed=args.seed + episode * 1000 + 100,
                mode="adv_vs_random_prey_eval",
            )
            eval_random_adv_return, eval_random_prey_return = evaluate_policy(
                simple_tag_v3=simple_tag_v3,
                trainer=trainer,
                max_cycles=args.max_cycles,
                eval_episodes=args.eval_episodes,
                seed=args.seed + episode * 1000 + 200,
                mode="random_adv_vs_prey_eval",
            )

        train_adversary_return = adversary_team_return(episode_returns)
        row = {
            "episode": episode,
            "total_steps": total_steps,
            "adversary_team_return": train_adversary_return,
            "prey_return": float(episode_returns[PREY_AGENT]),
            "adversary_0_return": float(episode_returns["adversary_0"]),
            "adversary_1_return": float(episode_returns["adversary_1"]),
            "adversary_2_return": float(episode_returns["adversary_2"]),
            "agent_0_return": float(episode_returns["agent_0"]),
            "train_touch_count": approximate_touch_count(train_adversary_return),
            "mean_critic_loss": mean_critic_loss,
            "mean_actor_loss": mean_actor_loss,
            "mean_actor_entropy": mean_actor_entropy,
            "epsilon": epsilon,
            "eval_self_play_adversary_return": eval_self_adv_return,
            "eval_self_play_prey_return": eval_self_prey_return,
            "eval_adv_vs_random_prey_adversary_return": eval_adv_random_adv_return,
            "eval_adv_vs_random_prey_prey_return": eval_adv_random_prey_return,
            "eval_random_adv_vs_prey_adversary_return": eval_random_adv_return,
            "eval_random_adv_vs_prey_prey_return": eval_random_prey_return,
            "eval_self_play_touch_count": approximate_touch_count(eval_self_adv_return),
            "eval_adv_vs_random_prey_touch_count": approximate_touch_count(
                eval_adv_random_adv_return
            ),
            "eval_random_adv_vs_prey_touch_count": approximate_touch_count(
                eval_random_adv_return
            ),
        }
        log_rows.append(row)
        save_log(log_rows, run_dir / "train_log.csv")
        if args.plot_interval > 0 and episode % args.plot_interval == 0:
            save_reward_curve(
                log_rows,
                run_dir / "reward_curve.png",
                save_debug_plots=args.save_debug_plots,
            )

        if args.save_interval > 0 and episode % args.save_interval == 0:
            save_checkpoints(trainer, checkpoint_root, episode)

        critic_text = "None" if mean_critic_loss is None else f"{mean_critic_loss:.4f}"
        actor_text = "None" if mean_actor_loss is None else f"{mean_actor_loss:.4f}"
        actor_entropy_text = format_optional(mean_actor_entropy)
        self_adv_text = format_optional(eval_self_adv_return)
        self_prey_text = format_optional(eval_self_prey_return)
        adv_rand_text = format_optional(eval_adv_random_adv_return)
        prey_rand_text = format_optional(eval_adv_random_prey_return)
        rand_adv_text = format_optional(eval_random_adv_return)
        rand_prey_text = format_optional(eval_random_prey_return)
        print(
            f"episode={episode:04d} "
            f"steps={total_steps} "
            f"adv_team={row['adversary_team_return']:.3f} "
            f"prey={row['prey_return']:.3f} "
            f"critic_loss={critic_text} "
            f"actor_loss={actor_text} "
            f"actor_entropy={actor_entropy_text} "
            f"epsilon={epsilon:.3f} "
            f"self_adv={self_adv_text} "
            f"self_prey={self_prey_text} "
            f"adv_rand={adv_rand_text} "
            f"prey_rand={prey_rand_text} "
            f"rand_adv={rand_adv_text} "
            f"rand_prey={rand_prey_text}"
        )

    env.close()
    if args.save_interval <= 0 or args.episodes % args.save_interval != 0:
        save_checkpoints(trainer, checkpoint_root, args.episodes)
    save_log(log_rows, run_dir / "train_log.csv")
    generated_plots, summary = save_all_outputs_from_rows(
        log_rows,
        run_dir,
        save_debug_plots_enabled=args.save_debug_plots,
    )
    core_plot_names = {
        "learning_status.png",
        "training_health.png",
        "touch_counts.png",
    }
    core_plots = [
        plot_path
        for plot_path in generated_plots
        if plot_path.name in core_plot_names
    ]

    print("\n=== Run summary ===")
    print(f"Run directory: {run_dir}")
    print(
        "Final adv_vs_random_prey return: "
        f"{summary['adv_return']} / baseline 17.0 -> {summary['adv_status']}"
    )
    print(
        "Final random_adv_vs_prey prey return: "
        f"{summary['prey_return']} / baseline -181.9 -> {summary['prey_status']}"
    )
    print(
        "Final touch count: "
        f"{summary['touch_count']} / baseline 1.7 -> {summary['touch_status']}"
    )
    print(f"Training health: {summary['health']}")
    print(f"Policy collapse: {summary['collapse']}")
    print("Core plots saved:")
    for plot_path in core_plots:
        print(f"- {plot_path.name}")
    print(
        "Reward scales used for training only: "
        f"adversary={args.adversary_reward_scale}, prey={args.prey_reward_scale}"
    )
    print(f"CSV saved to: {run_dir / 'train_log.csv'}")
    print(f"Config saved to: {run_dir / 'config.json'}")
    print(f"Summary saved to: {run_dir / 'summary.txt'}")
    print(f"Checkpoints saved to: {checkpoint_root}")


if __name__ == "__main__":
    main()
