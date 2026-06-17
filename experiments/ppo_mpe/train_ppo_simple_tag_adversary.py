# -*- coding: utf-8 -*-
"""在 simple_tag_v3 中只训练 adversary 追捕者的 shared PPO。

simple_tag 是追捕-逃逸环境：
- adversary_0、adversary_1、adversary_2 是追捕者，本脚本只训练它们。
- agent_0 是逃避者，也叫 prey，本脚本让它始终使用 random policy。

这个脚本复用现有 PPOAgent、RolloutBuffer、ActorCritic，不实现 QMIX/MADDPG。
"""

from __future__ import annotations

import argparse
import csv
import importlib
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

from algorithms.ppo.buffer import RolloutBuffer
from algorithms.ppo.ppo_agent import PPOAgent


ADVERSARY_AGENTS = ["adversary_0", "adversary_1", "adversary_2"]
PREY_AGENT = "agent_0"
OBS_DIM = 16
ACTION_DIM = 5

OUTPUT_DIR = ROOT_DIR / "outputs" / "ppo_mpe"
CSV_PATH = OUTPUT_DIR / "simple_tag_adversary_train_log.csv"
PNG_PATH = OUTPUT_DIR / "simple_tag_adversary_reward_curve.png"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints_simple_tag_adversary"


def load_simple_tag():
    """优先导入 mpe2.simple_tag_v3，失败后回退到 pettingzoo.mpe。"""

    try:
        return importlib.import_module("mpe2.simple_tag_v3"), "mpe2"
    except ImportError as mpe2_error:
        try:
            return (
                importlib.import_module("pettingzoo.mpe.simple_tag_v3"),
                "pettingzoo.mpe",
            )
        except ImportError as pettingzoo_error:
            raise ImportError(
                "无法导入 simple_tag_v3，请确认已安装 mpe2 或 pettingzoo.mpe。"
            ) from pettingzoo_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train shared PPO for simple_tag_v3 adversaries only."
    )
    parser.add_argument("--total-updates", type=int, default=50)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--max-cycles", type=int, default=100)
    parser.add_argument("--reward-scale", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=3)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_env(env, seed: int | None = None):
    """兼容不同版本 reset(seed=...) 的支持情况。"""

    if seed is None:
        return env.reset()

    try:
        return env.reset(seed=seed)
    except TypeError:
        return env.reset()


def all_agents_done(terminations: dict, truncations: dict, agents: list[str]) -> bool:
    return all(
        terminations.get(agent, False) or truncations.get(agent, False)
        for agent in agents
    )


def adversary_obs_to_array(observations: dict) -> np.ndarray:
    """只取三个 adversary 的 observation，prey 不进入 PPO buffer。"""

    return np.stack(
        [
            np.asarray(observations[agent], dtype=np.float32)
            for agent in ADVERSARY_AGENTS
        ],
        axis=0,
    )


def dict_values_to_array(values: dict, agents: list[str], default: float = 0.0) -> np.ndarray:
    return np.asarray([float(values.get(agent, default)) for agent in agents], dtype=np.float32)


def adversary_done_array(terminations: dict, truncations: dict) -> np.ndarray:
    return np.asarray(
        [
            float(terminations.get(agent, False) or truncations.get(agent, False))
            for agent in ADVERSARY_AGENTS
        ],
        dtype=np.float32,
    )


def mean_adversary_reward(rewards: dict) -> float:
    values = [float(rewards.get(agent, 0.0)) for agent in ADVERSARY_AGENTS]
    return sum(values) / len(values)


def build_action_dict(env, observations: dict, adversary_actions: np.ndarray) -> dict:
    """组装 Parallel API 需要的动作字典。

    adversary 使用 PPO 动作；agent_0 是逃避者，始终使用随机动作。
    """

    action_dict = {}
    for agent, action in zip(ADVERSARY_AGENTS, adversary_actions):
        if agent in observations:
            action_dict[agent] = int(action)

    for agent in observations.keys():
        if agent not in action_dict:
            action_dict[agent] = env.action_space(agent).sample()

    return action_dict


def save_log(rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "update",
                "train_mean_adversary_return",
                "eval_deterministic_mean_adversary_return",
                "eval_stochastic_mean_adversary_return",
                "policy_loss",
                "value_loss",
                "entropy",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def save_reward_curve(rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    updates = [row["update"] for row in rows]
    train_returns = [row["train_mean_adversary_return"] for row in rows]
    eval_deterministic_returns = [
        np.nan
        if row["eval_deterministic_mean_adversary_return"] is None
        else row["eval_deterministic_mean_adversary_return"]
        for row in rows
    ]
    eval_stochastic_returns = [
        np.nan
        if row["eval_stochastic_mean_adversary_return"] is None
        else row["eval_stochastic_mean_adversary_return"]
        for row in rows
    ]

    plt.figure(figsize=(8, 4.5))
    plt.plot(updates, train_returns, marker="o", label="train_mean_adversary_return")
    plt.plot(
        updates,
        eval_deterministic_returns,
        marker="s",
        label="eval_deterministic_mean_adversary_return",
    )
    plt.plot(
        updates,
        eval_stochastic_returns,
        marker="^",
        label="eval_stochastic_mean_adversary_return",
    )
    plt.xlabel("Update")
    plt.ylabel("Adversary mean episode return")
    plt.title("Shared PPO adversaries on MPE simple_tag_v3")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150)
    plt.close()


def evaluate_policy(
    simple_tag_v3,
    agent: PPOAgent,
    max_cycles: int,
    eval_episodes: int,
    seed: int,
    deterministic: bool,
) -> float:
    """评估三个 adversary，prey agent_0 仍然随机动作。"""

    eval_env = simple_tag_v3.parallel_env(max_cycles=max_cycles)
    episode_returns = []

    for episode in range(eval_episodes):
        observations, infos = reset_env(eval_env, seed=seed + episode)
        del infos

        episode_return = 0.0

        for _ in range(max_cycles):
            if not observations:
                break

            if not all(agent in observations for agent in ADVERSARY_AGENTS):
                observations, infos = reset_env(eval_env)
                del infos

            obs_array = adversary_obs_to_array(observations)
            if deterministic:
                adversary_actions = agent.act_deterministic(obs_array)
            else:
                adversary_actions, _, _ = agent.act(obs_array)

            action_dict = build_action_dict(eval_env, observations, adversary_actions)
            observations, rewards, terminations, truncations, infos = eval_env.step(action_dict)
            del infos

            episode_return += mean_adversary_reward(rewards)

            if all_agents_done(terminations, truncations, list(terminations.keys())):
                break

        episode_returns.append(episode_return)

    eval_env.close()

    if not episode_returns:
        return 0.0
    return float(np.mean(episode_returns))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    simple_tag_v3, source = load_simple_tag()
    env = simple_tag_v3.parallel_env(max_cycles=args.max_cycles)
    observations, infos = reset_env(env, seed=args.seed)
    del infos

    missing_adversaries = [agent for agent in ADVERSARY_AGENTS if agent not in env.possible_agents]
    if missing_adversaries:
        raise RuntimeError(f"环境中找不到这些 adversary: {missing_adversaries}")

    agent = PPOAgent(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=args.hidden_dim,
        lr=args.learning_rate,
        clip_eps=args.clip_coef,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
    )
    buffer = RolloutBuffer(
        rollout_steps=args.rollout_steps,
        num_agents=len(ADVERSARY_AGENTS),
        obs_dim=OBS_DIM,
        device=str(agent.device),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loaded simple_tag_v3 from: {source}")
    print("simple_tag 是追捕-逃逸环境：adversary 是追捕者，agent_0 是逃避者。")
    print(f"训练对象: {ADVERSARY_AGENTS}")
    print(f"prey random policy: {PREY_AGENT}")
    print(f"CSV 保存路径: {CSV_PATH}")
    print(f"曲线保存路径: {PNG_PATH}")
    print(f"checkpoint 保存路径: {CHECKPOINT_DIR}")

    log_rows = []
    recent_adversary_returns = []
    current_adversary_return = 0.0

    for update in range(1, args.total_updates + 1):
        buffer.reset()

        for _ in range(args.rollout_steps):
            if not all(agent in observations for agent in ADVERSARY_AGENTS):
                observations, infos = reset_env(env)
                del infos

            obs_array = adversary_obs_to_array(observations)
            adversary_actions, log_probs, values = agent.act(obs_array)
            action_dict = build_action_dict(env, observations, adversary_actions)

            next_observations, rewards, terminations, truncations, infos = env.step(action_dict)
            del infos

            reward_array = dict_values_to_array(rewards, ADVERSARY_AGENTS)
            done_array = adversary_done_array(terminations, truncations)

            buffer.add(
                obs=obs_array,
                actions=adversary_actions,
                log_probs=log_probs,
                values=values,
                rewards=reward_array * args.reward_scale,
                dones=done_array,
            )

            current_adversary_return += mean_adversary_reward(rewards)

            if all_agents_done(terminations, truncations, list(terminations.keys())) or not next_observations:
                recent_adversary_returns.append(current_adversary_return)
                current_adversary_return = 0.0
                next_observations, infos = reset_env(env)
                del infos

            observations = next_observations

        if not all(agent in observations for agent in ADVERSARY_AGENTS):
            observations, infos = reset_env(env)
            del infos

        last_obs_array = adversary_obs_to_array(observations)
        last_values = agent.value(last_obs_array)
        batch = buffer.compute_gae(
            last_values=last_values,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        minibatch_size = max(1, batch.obs.shape[0] // args.num_minibatches)
        info = agent.update(
            batch=batch,
            ppo_epochs=args.update_epochs,
            minibatch_size=minibatch_size,
        )

        if recent_adversary_returns:
            train_mean_adversary_return = float(np.mean(recent_adversary_returns[-10:]))
        else:
            train_mean_adversary_return = current_adversary_return

        eval_deterministic = None
        eval_stochastic = None
        if args.eval_interval > 0 and update % args.eval_interval == 0:
            eval_deterministic = evaluate_policy(
                simple_tag_v3=simple_tag_v3,
                agent=agent,
                max_cycles=args.max_cycles,
                eval_episodes=args.eval_episodes,
                seed=args.seed + update * 1000,
                deterministic=True,
            )
            eval_stochastic = evaluate_policy(
                simple_tag_v3=simple_tag_v3,
                agent=agent,
                max_cycles=args.max_cycles,
                eval_episodes=args.eval_episodes,
                seed=args.seed + update * 1000 + 100,
                deterministic=False,
            )

        row = {
            "update": update,
            "train_mean_adversary_return": train_mean_adversary_return,
            "eval_deterministic_mean_adversary_return": eval_deterministic,
            "eval_stochastic_mean_adversary_return": eval_stochastic,
            "policy_loss": info.policy_loss,
            "value_loss": info.value_loss,
            "entropy": info.entropy,
        }
        log_rows.append(row)

        eval_det_text = f"{eval_deterministic:.3f}" if eval_deterministic is not None else "None"
        eval_sto_text = f"{eval_stochastic:.3f}" if eval_stochastic is not None else "None"
        print(
            f"update={update:03d} "
            f"train_mean_adversary_return={train_mean_adversary_return:.3f} "
            f"eval_deterministic_mean_adversary_return={eval_det_text} "
            f"eval_stochastic_mean_adversary_return={eval_sto_text} "
            f"policy_loss={info.policy_loss:.4f} "
            f"value_loss={info.value_loss:.4f} "
            f"entropy={info.entropy:.4f}"
        )

        save_log(log_rows)
        save_reward_curve(log_rows)

        if args.checkpoint_interval > 0 and update % args.checkpoint_interval == 0:
            checkpoint_path = CHECKPOINT_DIR / f"ppo_simple_tag_adversary_update_{update:03d}.pt"
            agent.save(str(checkpoint_path))

    env.close()
    print(f"CSV saved to: {CSV_PATH}")
    print(f"Reward curve saved to: {PNG_PATH}")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
