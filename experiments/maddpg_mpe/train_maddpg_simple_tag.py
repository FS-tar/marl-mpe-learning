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
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algorithms.maddpg.maddpg_trainer import MADDPGTrainer


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MADDPG on MPE simple_tag_v3."
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-cycles", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--buffer-size", type=int, default=200000)
    parser.add_argument("--learning-rate-actor", type=float, default=1e-3)
    parser.add_argument("--learning-rate-critic", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--start-steps", type=int, default=1000)
    parser.add_argument("--update-after", type=int, default=1000)
    parser.add_argument("--update-every", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-updates", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=20)
    return parser.parse_args()


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
        "mean_critic_loss",
        "mean_actor_loss",
        "epsilon",
        "eval_adversary_team_return",
        "eval_prey_return",
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


def save_reward_curve(rows: list[dict], png_path: Path) -> None:
    tmp_path = png_path.with_name("reward_curve.tmp.png")
    try:
        episodes = [row["episode"] for row in rows]
        adversary_returns = [row["adversary_team_return"] for row in rows]
        prey_returns = [row["prey_return"] for row in rows]
        eval_adversary = [
            np.nan if row["eval_adversary_team_return"] is None
            else row["eval_adversary_team_return"]
            for row in rows
        ]
        eval_prey = [
            np.nan if row["eval_prey_return"] is None
            else row["eval_prey_return"]
            for row in rows
        ]

        plt.figure(figsize=(9, 5))
        plt.plot(episodes, adversary_returns, label="train adversary team")
        plt.plot(episodes, prey_returns, label="train prey")
        plt.plot(episodes, eval_adversary, marker="o", linestyle="--", label="eval adversary team")
        plt.plot(episodes, eval_prey, marker="o", linestyle="--", label="eval prey")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("MADDPG on MPE simple_tag_v3")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(tmp_path, dpi=150)
        os.replace(tmp_path, png_path)
    except OSError as error:
        print(f"warning: 保存 reward_curve.png 失败: {png_path} ({error})")
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
    finally:
        plt.close()


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


def evaluate_policy(
    simple_tag_v3,
    trainer: MADDPGTrainer,
    max_cycles: int,
    eval_episodes: int,
    seed: int,
) -> tuple[float, float]:
    """评估阶段不加探索噪声，所有 actor 使用 argmax action。"""

    eval_env = simple_tag_v3.parallel_env(max_cycles=max_cycles)
    adversary_returns = []
    prey_returns = []

    for episode in range(eval_episodes):
        observations, infos = reset_env(eval_env, seed=seed + episode)
        del infos
        episode_returns = {agent: 0.0 for agent in AGENT_NAMES}

        for _ in range(max_cycles):
            if not observations:
                break
            obs_arrays = observations_to_arrays(observations)
            actions, _ = trainer.act(obs_arrays, epsilon=0.0, explore=False)
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
    )

    print(f"Loaded simple_tag_v3 from: {source}")
    print(f"Run directory: {run_dir}")
    print("MADDPG: decentralized actors + centralized critics")
    print(f"critic input dim: {CRITIC_INPUT_DIM} = obs {GLOBAL_OBS_DIM} + actions {GLOBAL_ACTION_DIM}")

    total_steps = 0
    update_count = 0
    log_rows = []

    for episode in range(1, args.episodes + 1):
        observations, infos = reset_env(env, seed=args.seed + episode)
        del infos
        episode_returns = {agent: 0.0 for agent in AGENT_NAMES}
        episode_critic_losses = []
        episode_actor_losses = []

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
                )

            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            del infos
            total_steps += 1

            dones = done_dict(terminations, truncations)
            next_obs_arrays = next_observations_to_arrays(next_observations, obs_arrays)
            trainer.add_transition(
                obs=obs_arrays,
                actions=actions,
                one_hot_actions=one_hot_actions,
                rewards={agent: float(rewards.get(agent, 0.0)) for agent in AGENT_NAMES},
                next_obs=next_obs_arrays,
                dones=dones,
            )

            for agent in AGENT_NAMES:
                episode_returns[agent] += float(rewards.get(agent, 0.0))

            if (
                total_steps >= args.update_after
                and total_steps % max(1, args.update_every) == 0
                and len(trainer.replay_buffer) >= args.batch_size
            ):
                update_info = trainer.update(args.batch_size)
                update_count += 1
                episode_critic_losses.append(update_info.mean_critic_loss)
                episode_actor_losses.append(update_info.mean_actor_loss)

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
        epsilon = epsilon_by_update(args, update_count)

        eval_adv_return = None
        eval_prey_return = None
        if args.eval_interval > 0 and episode % args.eval_interval == 0:
            eval_adv_return, eval_prey_return = evaluate_policy(
                simple_tag_v3=simple_tag_v3,
                trainer=trainer,
                max_cycles=args.max_cycles,
                eval_episodes=args.eval_episodes,
                seed=args.seed + episode * 1000,
            )

        row = {
            "episode": episode,
            "total_steps": total_steps,
            "adversary_team_return": adversary_team_return(episode_returns),
            "prey_return": float(episode_returns[PREY_AGENT]),
            "adversary_0_return": float(episode_returns["adversary_0"]),
            "adversary_1_return": float(episode_returns["adversary_1"]),
            "adversary_2_return": float(episode_returns["adversary_2"]),
            "agent_0_return": float(episode_returns["agent_0"]),
            "mean_critic_loss": mean_critic_loss,
            "mean_actor_loss": mean_actor_loss,
            "epsilon": epsilon,
            "eval_adversary_team_return": eval_adv_return,
            "eval_prey_return": eval_prey_return,
        }
        log_rows.append(row)
        save_log(log_rows, run_dir / "train_log.csv")
        save_reward_curve(log_rows, run_dir / "reward_curve.png")

        if args.save_interval > 0 and episode % args.save_interval == 0:
            save_checkpoints(trainer, checkpoint_root, episode)

        critic_text = "None" if mean_critic_loss is None else f"{mean_critic_loss:.4f}"
        actor_text = "None" if mean_actor_loss is None else f"{mean_actor_loss:.4f}"
        eval_adv_text = "None" if eval_adv_return is None else f"{eval_adv_return:.3f}"
        eval_prey_text = "None" if eval_prey_return is None else f"{eval_prey_return:.3f}"
        print(
            f"episode={episode:04d} "
            f"steps={total_steps} "
            f"adv_team={row['adversary_team_return']:.3f} "
            f"prey={row['prey_return']:.3f} "
            f"critic_loss={critic_text} "
            f"actor_loss={actor_text} "
            f"epsilon={epsilon:.3f} "
            f"eval_adv={eval_adv_text} "
            f"eval_prey={eval_prey_text}"
        )

    env.close()
    if args.save_interval <= 0 or args.episodes % args.save_interval != 0:
        save_checkpoints(trainer, checkpoint_root, args.episodes)
    save_log(log_rows, run_dir / "train_log.csv")
    save_reward_curve(log_rows, run_dir / "reward_curve.png")
    print(f"CSV saved to: {run_dir / 'train_log.csv'}")
    print(f"Reward curve saved to: {run_dir / 'reward_curve.png'}")
    print(f"Config saved to: {run_dir / 'config.json'}")
    print(f"Checkpoints saved to: {checkpoint_root}")


if __name__ == "__main__":
    main()
