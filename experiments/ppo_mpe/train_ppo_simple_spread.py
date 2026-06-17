# -*- coding: utf-8 -*-
"""在 MPE simple_spread_v3 上训练教学版 shared PPO baseline。"""

# 本文件在 PPO 训练流程中的作用：
# 1. 创建 simple_spread_v3 的 parallel_env。
# 2. 把 MPE 返回的 observations 字典转成 PPO 网络需要的数组。
# 3. 调用 PPOAgent.act() 采样 actions/log_probs/values。
# 4. 调用 env.step(action_dict) 和环境交互。
# 5. 调用 RolloutBuffer.add() 保存 rollout 数据。
# 6. rollout 结束后调用 compute_gae() 和 PPOAgent.update()。
# 7. 打印训练日志，保存 CSV、reward 曲线和 checkpoint。
# 8. 定期用确定性和随机采样两种 evaluation，区分 argmax 策略表现和采样策略表现。

from __future__ import annotations

import argparse
import csv
import importlib
import os
import random
import re
import shutil
import sys
from datetime import datetime
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


OBS_DIM = 18
# simple_spread_v3 中单个 agent 的 observation 维度。
ACTION_DIM = 5
# simple_spread_v3 中单个 agent 的离散动作数，动作空间是 Discrete(5)。
OUTPUT_DIR = ROOT_DIR / "outputs" / "ppo_mpe"
RUNS_DIR = OUTPUT_DIR / "runs"
LATEST_CSV_PATH = OUTPUT_DIR / "latest_train_log.csv"
LATEST_PNG_PATH = OUTPUT_DIR / "latest_reward_curve.png"
LATEST_CHECKPOINT_DIR = OUTPUT_DIR / "latest_checkpoints"


def load_simple_spread():
    """优先使用 mpe2，失败后回退到 PettingZoo 的旧导入路径。"""

    # 输入：无。
    # 输出：
    # - simple_spread_v3 模块对象。
    # - source 字符串，说明环境来自 mpe2 还是 pettingzoo.mpe。
    # 调用位置：
    # - main() 一开始创建环境前调用。
    try:
        return importlib.import_module("mpe2.simple_spread_v3"), "mpe2"
    except ImportError as mpe2_error:
        try:
            return (
                importlib.import_module("pettingzoo.mpe.simple_spread_v3"),
                "pettingzoo.mpe",
            )
        except ImportError as pettingzoo_error:
            raise ImportError(
                "无法导入 simple_spread_v3，请确认已安装 mpe2 或 pettingzoo[mpe]。"
            ) from pettingzoo_error


def parse_args() -> argparse.Namespace:
    # 输入：命令行参数。
    # 输出：argparse.Namespace，里面保存 PPO 训练超参数。
    # 调用位置：
    # - main() 开头。
    parser = argparse.ArgumentParser(description="Train shared PPO on MPE simple_spread_v3.")
    parser.add_argument("--total-updates", type=int, default=50)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--max-cycles", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--reward-scale", type=float, default=0.1)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-stochastic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-latest", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    # 输入：随机种子。
    # 输出：无。
    # 作用：尽量让 numpy、random、torch 的随机性可复现。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_env(env, seed: int | None = None):
    """兼容不同版本的 reset(seed=...) 支持情况。"""

    # 输入：
    # - env：PettingZoo/MPE parallel_env。
    # - seed：可选随机种子。
    # 输出：
    # - observations：{agent_name: obs} 字典。
    # - infos：环境额外信息。
    # 调用位置：
    # - main() 初始化环境时。
    # - episode 结束后重新 reset 时。
    if seed is None:
        return env.reset()

    try:
        return env.reset(seed=seed)
    except TypeError:
        return env.reset()


def all_agents_done(terminations: dict, truncations: dict, agents: list[str]) -> bool:
    # 输入：
    # - terminations：环境自然结束标记。
    # - truncations：达到时间上限等截断标记。
    # - agents：固定 agent 顺序。
    # 输出：
    # - 如果所有 agent 都 terminated 或 truncated，返回 True。
    return all(
        terminations.get(agent, False) or truncations.get(agent, False)
        for agent in agents
    )


def obs_to_array(observations: dict, agents: list[str]) -> np.ndarray:
    """把 {agent: obs} 转成固定 agent 顺序的二维数组。"""

    # 输入：
    # - observations：MPE Parallel API 返回的字典，例如 {"agent_0": obs0, ...}。
    # - agents：固定顺序，例如 ["agent_0", "agent_1", "agent_2"]。
    # 输出：
    # - obs_array：形状通常是 [3, 18]。
    # 调用位置：
    # - rollout 采样循环中，作为 PPOAgent.act() 的输入。
    # - rollout 结束后，作为 PPOAgent.value() 的输入。
    return np.stack(
        [np.asarray(observations[agent], dtype=np.float32) for agent in agents],
        axis=0,
    )


def dict_values_to_array(values: dict, agents: list[str], default: float = 0.0) -> np.ndarray:
    # 输入：
    # - values：按 agent 名称组织的字典，例如 rewards。
    # - agents：固定 agent 顺序。
    # - default：缺失 agent 时的默认值。
    # 输出：
    # - 按固定 agent 顺序排列的一维数组，例如 [reward0, reward1, reward2]。
    return np.asarray([float(values.get(agent, default)) for agent in agents], dtype=np.float32)


def format_float_for_run_name(value: float) -> str:
    """把浮点参数变成短字符串，用在自动 run_name 里。"""

    return f"{value:g}"


def sanitize_run_name(run_name: str) -> str:
    """移除 Windows 路径中不适合使用的字符。"""

    return re.sub(r"[^0-9A-Za-z._-]+", "_", run_name).strip("._") or "ppo_run"


def build_run_name(args: argparse.Namespace) -> str:
    """根据时间戳和主要参数生成本次实验名。"""

    if args.run_name:
        return sanitize_run_name(args.run_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reward_scale = format_float_for_run_name(args.reward_scale)
    entropy_coef = format_float_for_run_name(args.entropy_coef)
    return sanitize_run_name(
        f"{timestamp}_spread_u{args.total_updates}_r{args.rollout_steps}"
        f"_rs{reward_scale}_ec{entropy_coef}"
    )


def save_log(rows: list[dict], csv_path: Path) -> None:
    # 输入：
    # - rows：每个 update 的日志字典。
    # - csv_path：目标 CSV 路径。
    # 输出：
    # - 写入指定路径。
    # 注意：这是训练日志保存，不影响 PPO 算法更新逻辑。
    tmp_path = csv_path.with_name(f".{csv_path.name}.tmp")

    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=(
                    "update",
                    "train_mean_episode_return",
                    "eval_deterministic_mean_episode_return",
                    "eval_stochastic_mean_episode_return",
                    "policy_loss",
                    "value_loss",
                    "entropy",
                ),
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, csv_path)
    except OSError as error:
        print(f"warning: 保存 CSV 失败: {csv_path} ({error})")
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def save_reward_curve(rows: list[dict], png_path: Path) -> None:
    # 输入：
    # - rows：每个 update 的日志字典。
    # - png_path：目标图片路径。
    # 输出：
    # - 写入指定路径。
    tmp_path = png_path.with_name(f".{png_path.stem}.tmp{png_path.suffix}")

    updates = [row["update"] for row in rows]
    train_returns = [row["train_mean_episode_return"] for row in rows]
    eval_deterministic_returns = [
        np.nan
        if row["eval_deterministic_mean_episode_return"] is None
        else row["eval_deterministic_mean_episode_return"]
        for row in rows
    ]
    eval_stochastic_returns = [
        np.nan
        if row["eval_stochastic_mean_episode_return"] is None
        else row["eval_stochastic_mean_episode_return"]
        for row in rows
    ]

    try:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4.5))
        plt.plot(updates, train_returns, marker="o", label="train_mean_episode_return")
        plt.plot(
            updates,
            eval_deterministic_returns,
            marker="s",
            label="eval_deterministic_mean_episode_return",
        )
        plt.plot(
            updates,
            eval_stochastic_returns,
            marker="^",
            label="eval_stochastic_mean_episode_return",
        )
        plt.xlabel("Update")
        plt.ylabel("Mean episode return")
        plt.title("Shared PPO on MPE simple_spread_v3")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(tmp_path, dpi=150)
        os.replace(tmp_path, png_path)
    except OSError as error:
        print(f"warning: 保存 reward 曲线失败: {png_path} ({error})")
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
    finally:
        plt.close()


def copy_checkpoint_to_latest(checkpoint_path: Path, latest_checkpoint_dir: Path) -> None:
    """把当前 run 的 checkpoint 额外复制到 latest_checkpoints。"""

    try:
        latest_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        latest_checkpoint_path = latest_checkpoint_dir / checkpoint_path.name
        tmp_path = latest_checkpoint_path.with_name(f".{latest_checkpoint_path.name}.tmp")
        shutil.copy2(checkpoint_path, tmp_path)
        os.replace(tmp_path, latest_checkpoint_path)
    except OSError as error:
        print(f"warning: 保存 latest checkpoint 失败: {latest_checkpoint_dir} ({error})")


def save_checkpoint(agent: PPOAgent, checkpoint_path: Path) -> bool:
    """保存当前 run 的 checkpoint，失败时只打印 warning。"""

    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(str(checkpoint_path))
        return True
    except (OSError, RuntimeError) as error:
        print(f"warning: 保存 checkpoint 失败: {checkpoint_path} ({error})")
        return False


def evaluate_policy(
    simple_spread_v3,
    agent: PPOAgent,
    max_cycles: int,
    eval_episodes: int,
    seed: int,
    deterministic: bool,
) -> float:
    """评估当前策略，不写 buffer，也不更新网络。"""

    # 输入：
    # - simple_spread_v3：已导入的环境模块。
    # - agent：当前正在训练的 PPOAgent。
    # - max_cycles：每个 episode 的环境步数上限。
    # - eval_episodes：评估多少个 episode。
    # - seed：评估环境 reset 的基础随机种子。
    # - deterministic：True 使用 argmax 动作；False 使用 Categorical 随机采样动作。
    # 输出：
    # - 使用原始 reward 统计的平均 episode return。
    eval_env = simple_spread_v3.parallel_env(render_mode=None, max_cycles=max_cycles)
    eval_agents = list(eval_env.possible_agents)
    episode_returns = []

    for episode in range(eval_episodes):
        observations, infos = reset_env(eval_env, seed=seed + episode)
        del infos

        episode_return = 0.0

        for _ in range(max_cycles):
            if not observations:
                break

            obs_array = obs_to_array(observations, eval_agents)
            if deterministic:
                actions = agent.act_deterministic(obs_array)
            else:
                actions, _, _ = agent.act(obs_array)
            action_dict = {
                agent_name: int(action)
                for agent_name, action in zip(eval_agents, actions)
                if agent_name in observations
            }

            observations, rewards, terminations, truncations, infos = eval_env.step(action_dict)
            del infos

            reward_array = dict_values_to_array(rewards, eval_agents)
            episode_return += float(np.mean(reward_array))

            if all_agents_done(terminations, truncations, eval_agents):
                break

        episode_returns.append(episode_return)

    eval_env.close()

    if not episode_returns:
        return 0.0
    return float(np.mean(episode_returns))


# 关键函数：main()
# 输入：命令行参数。
# 输出：无直接返回；负责完整训练流程、日志、曲线和 checkpoint。
# 调用位置：脚本入口 if __name__ == "__main__"。
def main() -> None:
    # main() 是训练主入口：
    # - 初始化环境、PPOAgent、RolloutBuffer。
    # - 外层 update 循环负责“收集一批数据 + 更新一次 PPO”。
    # - 内层 rollout 循环负责和 MPE 环境交互。
    args = parse_args()
    set_seed(args.seed)

    run_name = build_run_name(args)
    run_dir = RUNS_DIR / run_name
    run_csv_path = run_dir / "train_log.csv"
    run_png_path = run_dir / "reward_curve.png"
    run_checkpoint_dir = run_dir / "checkpoints"

    simple_spread_v3, source = load_simple_spread()
    render_mode = "human" if args.render else None
    env = simple_spread_v3.parallel_env(render_mode=render_mode, max_cycles=args.max_cycles)
    # parallel_env 表示多个 agent 在同一个 step 中并行动作。

    observations, infos = reset_env(env, seed=args.seed)
    # observations 是 dict：
    # {
    #     "agent_0": 18 维 obs,
    #     "agent_1": 18 维 obs,
    #     "agent_2": 18 维 obs,
    # }
    del infos

    agents = list(env.possible_agents)
    # agents 固定了字典转数组时的顺序，后续 action/reward/done 都按这个顺序排列。
    if len(agents) != 3:
        print(f"警告：当前环境 agent 数量是 {len(agents)}，本脚本按共享 PPO 继续训练。")

    agent = PPOAgent(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
    )
    # PPOAgent 内部包含共享 ActorCritic。三个 MPE agent 共用这一套网络参数。
    buffer = RolloutBuffer(
        rollout_steps=args.rollout_steps,
        num_agents=len(agents),
        obs_dim=OBS_DIM,
        device=str(agent.device),
    )
    # buffer 按 [rollout_steps, num_agents, ...] 保存 rollout 数据。

    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.save_latest:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        LATEST_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    log_rows = []
    recent_episode_returns = []
    current_episode_return = 0.0

    print(f"Loaded simple_spread_v3 from: {source}")
    print(f"当前 run_name: {run_name}")
    print(f"当前 run_dir: {run_dir}")
    print(f"CSV 保存路径: {run_csv_path}")
    print(f"图表保存路径: {run_png_path}")
    print(f"checkpoint 保存路径: {run_checkpoint_dir}")
    if args.save_latest:
        print(f"latest CSV 保存路径: {LATEST_CSV_PATH}")
        print(f"latest 图表保存路径: {LATEST_PNG_PATH}")
        print(f"latest checkpoint 保存路径: {LATEST_CHECKPOINT_DIR}")

    for update in range(1, args.total_updates + 1):
        # 一个 update 包含：
        # 1. 收集 rollout_steps 步数据。
        # 2. 用 GAE 计算 advantages/returns。
        # 3. 用这批数据执行 PPO 更新。
        buffer.reset()

        for _ in range(args.rollout_steps):
            # ===== rollout 采样循环开始 =====
            # 这一段是最重要的数据流：
            # observations(dict) -> obs_array -> PPOAgent.act()
            # -> action_dict -> env.step() -> reward/done -> buffer.add()
            obs_array = obs_to_array(observations, agents)
            # obs_array：形状通常是 [3, 18]。
            # 3 表示三个 agent，18 表示每个 agent 的 observation 维度。
            actions, log_probs, values = agent.act(obs_array)
            # actions：共享 actor 采样出的动作，形状 [3]。
            # log_probs：旧策略对这些动作的 log_prob，形状 [3]，后续保存为 old_log_probs。
            # values：critic 对当前 obs 的 value 估计，形状 [3]，后续用于 GAE。

            # PettingZoo Parallel API 需要 {agent: action} 字典。
            action_dict = {
                agent_name: int(action)
                for agent_name, action in zip(agents, actions)
                if agent_name in observations
            }
            # action_dict 示例：
            # {"agent_0": 1, "agent_1": 4, "agent_2": 0}

            next_observations, rewards, terminations, truncations, infos = env.step(action_dict)
            # env.step() 返回的 rewards/terminations/truncations 都是 dict，
            # key 是 agent 名称，value 是该 agent 的 reward 或结束标记。
            del infos

            reward_array = dict_values_to_array(rewards, agents)
            # rewards：环境原始奖励字典。
            # reward_array：按 agents 顺序排列的一维数组，形状 [3]。
            done_array = np.asarray(
                [
                    float(terminations.get(agent_name, False) or truncations.get(agent_name, False))
                    for agent_name in agents
                ],
                dtype=np.float32,
            )
            # dones：这里由 termination 或 truncation 合成。
            # done=1 表示该 agent 的当前轨迹结束，GAE 不再继续 bootstrap 下一步 value。

            buffer.add(
                obs=obs_array,
                actions=actions,
                log_probs=log_probs,
                values=values,
                rewards=reward_array * args.reward_scale,
                dones=done_array,
            )
            # buffer.add() 保存当前 step 的 3 条 agent transition：
            # obs/action/log_prob/value/reward/done。
            # 注意：PPO 训练用 reward 做了 reward_scale；
            # 但下面的 episode return 日志继续使用原始 reward_array。

            # simple_spread 通常是团队共享 reward，这里用所有 agent reward 的均值记 episode return。
            current_episode_return += float(np.mean(reward_array))
            # mean reward 用于日志统计，不直接改变 PPO loss。

            if all_agents_done(terminations, truncations, agents) or not next_observations:
                recent_episode_returns.append(current_episode_return)
                current_episode_return = 0.0
                next_observations, infos = reset_env(env)
                del infos
                # episode 结束后 reset，继续收集直到凑满 rollout_steps。

            observations = next_observations
            # 下一轮 rollout 使用新的 observations。
            # ===== rollout 采样循环结束 =====

        last_obs_array = obs_to_array(observations, agents)
        # rollout 结束后，还需要最后一个 next_obs 的 value，给 GAE 的最后一步使用。
        last_values = agent.value(last_obs_array)
        # last_values：形状 [3]，用于 compute_gae() 的 bootstrap。
        batch = buffer.compute_gae(
            last_values=last_values,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        # batch 中包含展平后的 obs/actions/old_log_probs/advantages/returns。
        info = agent.update(
            batch=batch,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
        )
        # agent.update() 内部计算 ratio、policy_loss、value_loss、entropy，并更新网络。

        if recent_episode_returns:
            train_mean_episode_return = float(np.mean(recent_episode_returns[-10:]))
        else:
            train_mean_episode_return = current_episode_return
        # 如果已经有完整 episode，就统计最近 10 个 episode 的平均 return；
        # 如果还没有完整 episode，就暂时显示当前 episode 已累计的 return。

        eval_deterministic_mean_episode_return = None
        eval_stochastic_mean_episode_return = None
        if args.eval_interval > 0 and update % args.eval_interval == 0:
            eval_deterministic_mean_episode_return = evaluate_policy(
                simple_spread_v3=simple_spread_v3,
                agent=agent,
                max_cycles=args.max_cycles,
                eval_episodes=args.eval_episodes,
                seed=args.seed + update * 1000,
                deterministic=True,
            )
            if args.eval_stochastic:
                eval_stochastic_mean_episode_return = evaluate_policy(
                    simple_spread_v3=simple_spread_v3,
                    agent=agent,
                    max_cycles=args.max_cycles,
                    eval_episodes=args.eval_episodes,
                    seed=args.seed + update * 1000 + 100,
                    deterministic=False,
                )

        row = {
            "update": update,
            "train_mean_episode_return": train_mean_episode_return,
            "eval_deterministic_mean_episode_return": eval_deterministic_mean_episode_return,
            "eval_stochastic_mean_episode_return": eval_stochastic_mean_episode_return,
            "policy_loss": info.policy_loss,
            "value_loss": info.value_loss,
            "entropy": info.entropy,
        }
        # policy_loss/value_loss/entropy 来自 PPOAgent.update() 的平均值。
        log_rows.append(row)

        eval_deterministic_text = (
            f"{eval_deterministic_mean_episode_return:.3f}"
            if eval_deterministic_mean_episode_return is not None
            else "None"
        )
        eval_stochastic_text = (
            f"{eval_stochastic_mean_episode_return:.3f}"
            if eval_stochastic_mean_episode_return is not None
            else "None"
        )
        print(
            f"update={update:03d} "
            f"train_mean_episode_return={train_mean_episode_return:.3f} "
            f"eval_deterministic_mean_episode_return={eval_deterministic_text} "
            f"eval_stochastic_mean_episode_return={eval_stochastic_text} "
            f"policy_loss={info.policy_loss:.4f} "
            f"value_loss={info.value_loss:.4f} "
            f"entropy={info.entropy:.4f}"
        )

        save_log(log_rows, run_csv_path)
        save_reward_curve(log_rows, run_png_path)

        if args.save_latest:
            save_log(log_rows, LATEST_CSV_PATH)
            save_reward_curve(log_rows, LATEST_PNG_PATH)

        if args.checkpoint_interval > 0 and update % args.checkpoint_interval == 0:
            checkpoint_path = run_checkpoint_dir / f"ppo_update_{update:03d}.pt"
            checkpoint_saved = save_checkpoint(agent, checkpoint_path)
            if checkpoint_saved and args.save_latest:
                copy_checkpoint_to_latest(checkpoint_path, LATEST_CHECKPOINT_DIR)

    env.close()
    print(f"CSV saved to: {run_csv_path}")
    print(f"Reward curve saved to: {run_png_path}")
    print(f"Checkpoints saved to: {run_checkpoint_dir}")


if __name__ == "__main__":
    main()


"""1. env.reset()
   得到 observations

2. obs_to_array()
   dict → (3, 18)

3. agent.act()
   obs → action/log_prob/value

4. env.step(action_dict)
   action → reward/next_obs/done

5. buffer.add()
   保存 obs/action/log_prob/value/reward/done

6. 重复 rollout_steps 次

7. agent.value()
   算 last_values

8. buffer.compute_gae()
   算 advantages 和 returns

9. agent.update()
   用 PPO loss 更新网络

10. save_log / save_reward_curve / checkpoint"""
