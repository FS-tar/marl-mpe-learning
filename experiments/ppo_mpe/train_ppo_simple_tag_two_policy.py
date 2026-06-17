# -*- coding: utf-8 -*-
"""simple_tag_v3 two-policy PPO 训练脚本。

simple_tag 是追捕-逃逸对抗环境：
- adversary_0、adversary_1、adversary_2 是追捕者，共享 adversary PPO。
- agent_0 是逃避者，单独使用 prey PPO。

本脚本只复用现有 PPOAgent、RolloutBuffer、ActorCritic，不实现 QMIX/MADDPG，
也不修改 simple_spread 的 PPO 训练代码。
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

from algorithms.ppo.buffer import RolloutBuffer
from algorithms.ppo.ppo_agent import PPOAgent


ADVERSARY_AGENTS = ["adversary_0", "adversary_1", "adversary_2"]
PREY_AGENT = "agent_0"
ADVERSARY_OBS_DIM = 16
PREY_OBS_DIM = 14
ACTION_DIM = 5

OUTPUT_BASE_DIR = ROOT_DIR / "outputs" / "ppo_mpe" / "simple_tag_two_policy"


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
        description="Train two-policy PPO on simple_tag_v3."
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
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    return parser.parse_args()


def get_next_run_dir(base_dir: Path) -> Path:
    """返回下一个不存在的 runXXX 目录，并立即创建。

    例：如果 run001 已存在，就尝试 run002；如果 run002 已存在，就继续尝试 run003。
    """

    base_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, 10000):
        run_dir = base_dir / f"run{index:03d}"
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue

    raise RuntimeError(f"无法在 {base_dir} 下找到可用 run 目录。")


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
    if not agents:
        return False

    return all(
        terminations.get(agent, False) or truncations.get(agent, False)
        for agent in agents
    )


def required_agents_present(observations: dict) -> bool:
    return PREY_AGENT in observations and all(
        agent in observations for agent in ADVERSARY_AGENTS
    )


def adversary_obs_to_array(observations: dict) -> np.ndarray:
    """取三个追捕者 observation，形状是 [3, 16]。"""

    return np.stack(
        [
            np.asarray(observations[agent], dtype=np.float32)
            for agent in ADVERSARY_AGENTS
        ],
        axis=0,
    )


def prey_obs_to_array(observations: dict) -> np.ndarray:
    """取逃避者 agent_0 的 observation，形状是 [1, 14]。"""

    return np.asarray(observations[PREY_AGENT], dtype=np.float32).reshape(1, -1)


def dict_values_to_array(values: dict, agents: list[str], default: float = 0.0) -> np.ndarray:
    return np.asarray([float(values.get(agent, default)) for agent in agents], dtype=np.float32)


def prey_value_to_array(values: dict, default: float = 0.0) -> np.ndarray:
    return np.asarray([float(values.get(PREY_AGENT, default))], dtype=np.float32)


def done_array(terminations: dict, truncations: dict, agents: list[str]) -> np.ndarray:
    return np.asarray(
        [
            float(terminations.get(agent, False) or truncations.get(agent, False))
            for agent in agents
        ],
        dtype=np.float32,
    )


def prey_done_array(terminations: dict, truncations: dict) -> np.ndarray:
    return np.asarray(
        [
            float(
                terminations.get(PREY_AGENT, False)
                or truncations.get(PREY_AGENT, False)
            )
        ],
        dtype=np.float32,
    )


def mean_adversary_reward(rewards: dict) -> float:
    values = [float(rewards.get(agent, 0.0)) for agent in ADVERSARY_AGENTS]
    return sum(values) / len(values)


def prey_reward(rewards: dict) -> float:
    return float(rewards.get(PREY_AGENT, 0.0))


def build_action_dict(adversary_actions: np.ndarray, prey_actions: np.ndarray) -> dict:
    """把两个 PPO 网络输出的动作合并成 Parallel API 需要的 action dict。"""

    action_dict = {
        agent: int(action)
        for agent, action in zip(ADVERSARY_AGENTS, adversary_actions)
    }
    action_dict[PREY_AGENT] = int(prey_actions[0])
    return action_dict


def minibatch_size_from_num_minibatches(batch_size: int, num_minibatches: int) -> int:
    return max(1, batch_size // max(1, num_minibatches))


def save_config(args: argparse.Namespace, run_dir: Path) -> None:
    """保存本次训练参数，方便之后对比实验。"""

    config_path = run_dir / "config.json"
    tmp_path = run_dir / "config.tmp.json"
    config = vars(args).copy()
    config["adversary_agents"] = ADVERSARY_AGENTS
    config["prey_agent"] = PREY_AGENT
    config["adversary_obs_dim"] = ADVERSARY_OBS_DIM
    config["prey_obs_dim"] = PREY_OBS_DIM
    config["action_dim"] = ACTION_DIM

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
    """保存 train_log.csv。先写 tmp，再 os.replace，避免 Windows 文件占用导致崩溃。"""

    tmp_path = csv_path.with_name("train_log.tmp.csv")
    try:
        with tmp_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=(
                    "update",
                    "train_mean_adversary_return",
                    "train_mean_prey_return",
                    "eval_deterministic_adversary_return",
                    "eval_deterministic_prey_return",
                    "eval_stochastic_adversary_return",
                    "eval_stochastic_prey_return",
                    "adversary_policy_loss",
                    "adversary_value_loss",
                    "adversary_entropy",
                    "prey_policy_loss",
                    "prey_value_loss",
                    "prey_entropy",
                ),
            )
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
    """保存 reward_curve.png。先写 tmp，再 os.replace，失败只 warning。"""

    tmp_path = png_path.with_name("reward_curve.tmp.png")
    updates = [row["update"] for row in rows]
    train_adv = [row["train_mean_adversary_return"] for row in rows]
    train_prey = [row["train_mean_prey_return"] for row in rows]
    eval_det_adv = [
        np.nan if row["eval_deterministic_adversary_return"] is None
        else row["eval_deterministic_adversary_return"]
        for row in rows
    ]
    eval_det_prey = [
        np.nan if row["eval_deterministic_prey_return"] is None
        else row["eval_deterministic_prey_return"]
        for row in rows
    ]
    eval_sto_adv = [
        np.nan if row["eval_stochastic_adversary_return"] is None
        else row["eval_stochastic_adversary_return"]
        for row in rows
    ]
    eval_sto_prey = [
        np.nan if row["eval_stochastic_prey_return"] is None
        else row["eval_stochastic_prey_return"]
        for row in rows
    ]

    try:
        plt.figure(figsize=(9, 5))
        plt.plot(updates, train_adv, marker="o", label="train adversary")
        plt.plot(updates, train_prey, marker="o", label="train prey")
        plt.plot(updates, eval_det_adv, marker="s", label="eval det adversary")
        plt.plot(updates, eval_det_prey, marker="s", label="eval det prey")
        plt.plot(updates, eval_sto_adv, marker="^", label="eval stochastic adversary")
        plt.plot(updates, eval_sto_prey, marker="^", label="eval stochastic prey")
        plt.xlabel("Update")
        plt.ylabel("Episode return")
        plt.title("Two-policy PPO on MPE simple_tag_v3")
        plt.legend()
        plt.grid(True, alpha=0.3)
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


def save_checkpoint(agent: PPOAgent, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(str(path))
    except (OSError, RuntimeError) as error:
        print(f"warning: 保存 checkpoint 失败: {path} ({error})")


def choose_actions(
    adversary_agent: PPOAgent,
    prey_agent: PPOAgent,
    observations: dict,
    deterministic: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """根据 deterministic 开关，用两个策略同时产生动作。"""

    adversary_obs = adversary_obs_to_array(observations)
    prey_obs = prey_obs_to_array(observations)

    if deterministic:
        adversary_actions = adversary_agent.act_deterministic(adversary_obs)
        prey_actions = prey_agent.act_deterministic(prey_obs)
    else:
        adversary_actions, _, _ = adversary_agent.act(adversary_obs)
        prey_actions, _, _ = prey_agent.act(prey_obs)

    return adversary_actions, prey_actions


def evaluate_policy(
    simple_tag_v3,
    adversary_agent: PPOAgent,
    prey_agent: PPOAgent,
    max_cycles: int,
    eval_episodes: int,
    seed: int,
    deterministic: bool,
) -> tuple[float, float]:
    """评估 two-policy PPO，不写 buffer、不更新网络。"""

    eval_env = simple_tag_v3.parallel_env(max_cycles=max_cycles)
    adversary_returns = []
    prey_returns = []

    for episode in range(eval_episodes):
        observations, infos = reset_env(eval_env, seed=seed + episode)
        del infos

        adv_return = 0.0
        prey_return_value = 0.0

        for _ in range(max_cycles):
            if not observations:
                break
            if not required_agents_present(observations):
                observations, infos = reset_env(eval_env)
                del infos

            adversary_actions, prey_actions = choose_actions(
                adversary_agent,
                prey_agent,
                observations,
                deterministic=deterministic,
            )
            action_dict = build_action_dict(adversary_actions, prey_actions)
            observations, rewards, terminations, truncations, infos = eval_env.step(action_dict)
            del infos

            adv_return += mean_adversary_reward(rewards)
            prey_return_value += prey_reward(rewards)

            if all_agents_done(terminations, truncations, list(terminations.keys())):
                break

        adversary_returns.append(adv_return)
        prey_returns.append(prey_return_value)

    eval_env.close()
    return float(np.mean(adversary_returns)), float(np.mean(prey_returns))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_dir = get_next_run_dir(OUTPUT_BASE_DIR)
    checkpoints_adversary_dir = run_dir / "checkpoints" / "adversary"
    checkpoints_prey_dir = run_dir / "checkpoints" / "prey"
    checkpoints_adversary_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_prey_dir.mkdir(parents=True, exist_ok=True)
    save_config(args, run_dir)

    print("Current run directory:")
    print(f"{run_dir}/")

    simple_tag_v3, source = load_simple_tag()
    env = simple_tag_v3.parallel_env(max_cycles=args.max_cycles)
    observations, infos = reset_env(env, seed=args.seed)
    del infos

    missing_agents = [
        agent
        for agent in [*ADVERSARY_AGENTS, PREY_AGENT]
        if agent not in env.possible_agents
    ]
    if missing_agents:
        raise RuntimeError(f"simple_tag_v3 中缺少这些 agent: {missing_agents}")

    adversary_agent = PPOAgent(
        obs_dim=ADVERSARY_OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=args.hidden_dim,
        lr=args.learning_rate,
        clip_eps=args.clip_coef,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
    )
    prey_agent = PPOAgent(
        obs_dim=PREY_OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=args.hidden_dim,
        lr=args.learning_rate,
        clip_eps=args.clip_coef,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
    )
    adversary_buffer = RolloutBuffer(
        rollout_steps=args.rollout_steps,
        num_agents=len(ADVERSARY_AGENTS),
        obs_dim=ADVERSARY_OBS_DIM,
        device=str(adversary_agent.device),
    )
    prey_buffer = RolloutBuffer(
        rollout_steps=args.rollout_steps,
        num_agents=1,
        obs_dim=PREY_OBS_DIM,
        device=str(prey_agent.device),
    )

    print(f"Loaded simple_tag_v3 from: {source}")
    print("simple_tag 是追捕-逃逸环境：adversary 是追捕者，agent_0 是逃避者。")
    print(f"adversary policy controls: {ADVERSARY_AGENTS}")
    print(f"prey policy controls: {PREY_AGENT}")

    log_rows = []
    recent_adversary_returns = []
    recent_prey_returns = []
    current_adversary_return = 0.0
    current_prey_return = 0.0

    for update in range(1, args.total_updates + 1):
        adversary_buffer.reset()
        prey_buffer.reset()

        for _ in range(args.rollout_steps):
            if not required_agents_present(observations):
                observations, infos = reset_env(env)
                del infos

            adversary_obs = adversary_obs_to_array(observations)
            prey_obs = prey_obs_to_array(observations)

            adversary_actions, adversary_log_probs, adversary_values = adversary_agent.act(
                adversary_obs
            )
            prey_actions, prey_log_probs, prey_values = prey_agent.act(prey_obs)

            action_dict = build_action_dict(adversary_actions, prey_actions)
            next_observations, rewards, terminations, truncations, infos = env.step(action_dict)
            del infos

            adversary_rewards = dict_values_to_array(rewards, ADVERSARY_AGENTS)
            prey_rewards = prey_value_to_array(rewards)
            adversary_dones = done_array(terminations, truncations, ADVERSARY_AGENTS)
            prey_dones = prey_done_array(terminations, truncations)

            # two-policy 数据流：
            # adversary 的 transition 只进 adversary_buffer；
            # prey agent_0 的 transition 只进 prey_buffer。
            adversary_buffer.add(
                obs=adversary_obs,
                actions=adversary_actions,
                log_probs=adversary_log_probs,
                values=adversary_values,
                rewards=adversary_rewards * args.reward_scale,
                dones=adversary_dones,
            )
            prey_buffer.add(
                obs=prey_obs,
                actions=prey_actions,
                log_probs=prey_log_probs,
                values=prey_values,
                rewards=prey_rewards * args.reward_scale,
                dones=prey_dones,
            )

            current_adversary_return += mean_adversary_reward(rewards)
            current_prey_return += prey_reward(rewards)

            if all_agents_done(terminations, truncations, list(terminations.keys())) or not next_observations:
                recent_adversary_returns.append(current_adversary_return)
                recent_prey_returns.append(current_prey_return)
                current_adversary_return = 0.0
                current_prey_return = 0.0
                next_observations, infos = reset_env(env)
                del infos

            observations = next_observations

        if not required_agents_present(observations):
            observations, infos = reset_env(env)
            del infos

        adversary_last_values = adversary_agent.value(adversary_obs_to_array(observations))
        prey_last_values = prey_agent.value(prey_obs_to_array(observations))

        adversary_batch = adversary_buffer.compute_gae(
            last_values=adversary_last_values,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        prey_batch = prey_buffer.compute_gae(
            last_values=prey_last_values,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )

        adversary_minibatch_size = minibatch_size_from_num_minibatches(
            adversary_batch.obs.shape[0],
            args.num_minibatches,
        )
        prey_minibatch_size = minibatch_size_from_num_minibatches(
            prey_batch.obs.shape[0],
            args.num_minibatches,
        )

        adversary_info = adversary_agent.update(
            batch=adversary_batch,
            ppo_epochs=args.update_epochs,
            minibatch_size=adversary_minibatch_size,
        )
        prey_info = prey_agent.update(
            batch=prey_batch,
            ppo_epochs=args.update_epochs,
            minibatch_size=prey_minibatch_size,
        )

        if recent_adversary_returns:
            train_mean_adversary_return = float(np.mean(recent_adversary_returns[-10:]))
            train_mean_prey_return = float(np.mean(recent_prey_returns[-10:]))
        else:
            train_mean_adversary_return = current_adversary_return
            train_mean_prey_return = current_prey_return

        eval_det_adv = None
        eval_det_prey = None
        eval_sto_adv = None
        eval_sto_prey = None
        if args.eval_interval > 0 and update % args.eval_interval == 0:
            eval_det_adv, eval_det_prey = evaluate_policy(
                simple_tag_v3=simple_tag_v3,
                adversary_agent=adversary_agent,
                prey_agent=prey_agent,
                max_cycles=args.max_cycles,
                eval_episodes=args.eval_episodes,
                seed=args.seed + update * 1000,
                deterministic=True,
            )
            eval_sto_adv, eval_sto_prey = evaluate_policy(
                simple_tag_v3=simple_tag_v3,
                adversary_agent=adversary_agent,
                prey_agent=prey_agent,
                max_cycles=args.max_cycles,
                eval_episodes=args.eval_episodes,
                seed=args.seed + update * 1000 + 100,
                deterministic=False,
            )

        row = {
            "update": update,
            "train_mean_adversary_return": train_mean_adversary_return,
            "train_mean_prey_return": train_mean_prey_return,
            "eval_deterministic_adversary_return": eval_det_adv,
            "eval_deterministic_prey_return": eval_det_prey,
            "eval_stochastic_adversary_return": eval_sto_adv,
            "eval_stochastic_prey_return": eval_sto_prey,
            "adversary_policy_loss": adversary_info.policy_loss,
            "adversary_value_loss": adversary_info.value_loss,
            "adversary_entropy": adversary_info.entropy,
            "prey_policy_loss": prey_info.policy_loss,
            "prey_value_loss": prey_info.value_loss,
            "prey_entropy": prey_info.entropy,
        }
        log_rows.append(row)

        if args.log_interval > 0 and update % args.log_interval == 0:
            save_log(log_rows, run_dir / "train_log.csv")
            save_reward_curve(log_rows, run_dir / "reward_curve.png")

        if args.checkpoint_interval > 0 and update % args.checkpoint_interval == 0:
            save_checkpoint(
                adversary_agent,
                checkpoints_adversary_dir / f"adversary_update_{update:03d}.pt",
            )
            save_checkpoint(
                prey_agent,
                checkpoints_prey_dir / f"prey_update_{update:03d}.pt",
            )

        det_adv_text = f"{eval_det_adv:.3f}" if eval_det_adv is not None else "None"
        det_prey_text = f"{eval_det_prey:.3f}" if eval_det_prey is not None else "None"
        sto_adv_text = f"{eval_sto_adv:.3f}" if eval_sto_adv is not None else "None"
        sto_prey_text = f"{eval_sto_prey:.3f}" if eval_sto_prey is not None else "None"
        print(
            f"update={update:03d} "
            f"train_adv={train_mean_adversary_return:.3f} "
            f"train_prey={train_mean_prey_return:.3f} "
            f"eval_det_adv={det_adv_text} "
            f"eval_det_prey={det_prey_text} "
            f"eval_sto_adv={sto_adv_text} "
            f"eval_sto_prey={sto_prey_text} "
            f"adv_loss={adversary_info.policy_loss:.4f}/"
            f"{adversary_info.value_loss:.4f}/"
            f"{adversary_info.entropy:.4f} "
            f"prey_loss={prey_info.policy_loss:.4f}/"
            f"{prey_info.value_loss:.4f}/"
            f"{prey_info.entropy:.4f}"
        )

    env.close()
    save_log(log_rows, run_dir / "train_log.csv")
    save_reward_curve(log_rows, run_dir / "reward_curve.png")
    print(f"CSV saved to: {run_dir / 'train_log.csv'}")
    print(f"Reward curve saved to: {run_dir / 'reward_curve.png'}")
    print(f"Config saved to: {run_dir / 'config.json'}")
    print(f"Checkpoints saved to: {run_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
