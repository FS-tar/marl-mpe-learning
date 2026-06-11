# -*- coding: utf-8 -*-
"""图形化观察不同 MPE 环境。

这个脚本只用于看环境、看 agent 如何运动，不实现 PPO/QMIX/MADDPG。
"""

from __future__ import annotations

import argparse
import importlib
import time


ENV_MODULES = {
    "simple": "simple_v3",
    "simple_spread": "simple_spread_v3",
    "simple_adversary": "simple_adversary_v3",
    "simple_tag": "simple_tag_v3",
    "simple_push": "simple_push_v3",
}

ACTION_MODES = ("random", "stay", "right", "left", "down", "up", "cycle")

FIXED_ACTIONS = {
    "stay": 0,
    "left": 1,
    "right": 2,
    "down": 3,
    "up": 4,
}


def load_mpe_env(env_key: str):
    """优先从 mpe2 子模块导入环境，失败后再尝试 pettingzoo.mpe 子模块。"""

    module_name = ENV_MODULES[env_key]

    try:
        return importlib.import_module(f"mpe2.{module_name}"), "mpe2", module_name
    except ImportError as mpe2_error:
        try:
            return (
                importlib.import_module(f"pettingzoo.mpe.{module_name}"),
                "pettingzoo.mpe",
                module_name,
            )
        except ImportError as pettingzoo_error:
            raise ImportError(
                f"无法从 mpe2 或 pettingzoo.mpe 导入 {module_name}: "
                f"{mpe2_error}; {pettingzoo_error}"
            ) from pettingzoo_error


def all_agents_done(terminations: dict, truncations: dict) -> bool:
    """如果所有 agent 都 terminated 或 truncated，就认为 episode 结束。"""

    agents = set(terminations) | set(truncations)
    if not agents:
        return False

    return all(
        terminations.get(agent, False) or truncations.get(agent, False)
        for agent in agents
    )


def action_for_mode(mode: str, step: int) -> int | None:
    """根据 --mode 返回固定动作；random 返回 None 表示随机采样。

    MPE simple 系列常见离散动作约定：
    0=不动，1=左，2=右，3=下，4=上。
    cycle 模式每 20 step 在 1、2、3、4 之间切换，方便观察运动方向。
    """

    if mode == "random":
        return None

    if mode == "cycle":
        cycle_actions = (1, 2, 3, 4)
        index = ((step - 1) // 20) % len(cycle_actions)
        return cycle_actions[index]

    return FIXED_ACTIONS[mode]


def build_actions(env, observations: dict, mode: str, step: int) -> dict:
    """根据当前 observations.keys() 为每个 agent 构造 actions 字典。

    observations 是 dict：agent 名称 -> 当前 observation。
    actions 也是 dict：agent 名称 -> 本 step 的动作。
    random 模式保持原逻辑，从 env.action_space(agent).sample() 随机采样。
    其他模式给所有当前 agent 同一个固定动作，便于观察图形界面里的运动。
    """

    fixed_action = action_for_mode(mode, step)
    if fixed_action is None:
        return {
            agent: env.action_space(agent).sample()
            for agent in observations.keys()
        }

    return {
        agent: fixed_action
        for agent in observations.keys()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="图形化观察 MPE 环境。")
    parser.add_argument(
        "--env",
        choices=sorted(ENV_MODULES),
        default="simple_spread",
        help="要观察的 MPE 环境。",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="最多运行多少个 step。",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="每一步之间暂停多少秒，方便看清图形界面。",
    )
    parser.add_argument(
        "--mode",
        choices=ACTION_MODES,
        default="random",
        help="动作观察模式：random/stay/right/left/down/up/cycle。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_module, source, module_name = load_mpe_env(args.env)

    # render_mode="human" 表示打开图形化窗口，适合肉眼观察环境。
    # max_cycles=args.steps 表示环境内部最多允许运行多少个 cycle。
    # parallel_env 是 Parallel API：多个 agent 在同一个 step 同时行动。
    env = env_module.parallel_env(render_mode="human", max_cycles=args.steps)

    # observations 是 dict：agent 名称 -> 当前 observation。
    # infos 也是 dict：agent 名称 -> 额外调试信息。
    observations, infos = env.reset()

    print(f"env: {args.env} ({module_name})")
    print(f"source: {source}")
    print(f"mode: {args.mode}")
    print(f"possible_agents: {env.possible_agents}")
    print(f"reset infos: {infos}")

    print("\nSpaces:")
    for agent in env.possible_agents:
        print(f"- {agent}")
        print(f"  observation_space: {env.observation_space(agent)}")
        print(f"  action_space:      {env.action_space(agent)}")

    for step in range(1, args.steps + 1):
        if not observations:
            print("No observations returned; episode ended.")
            break

        # 构造 actions 时仍然使用 observations.keys()。
        # 这些 key 表示当前 step 需要行动的 agent。
        actions = build_actions(env, observations, args.mode, step)

        # rewards 是 dict：agent 名称 -> 本 step 得到的 reward。
        observations, rewards, terminations, truncations, infos = env.step(actions)
        del infos

        # 每 10 step 打印一次，避免终端刷屏太快；第 1 step 也打印一次方便确认。
        if step == 1 or step % 10 == 0:
            print(
                f"step={step}, mode={args.mode}, "
                f"actions={actions}, rewards={rewards}"
            )

        if all_agents_done(terminations, truncations):
            print("All agents terminated or truncated; episode ended early.")
            break

        # 暂停一下，避免 human render 窗口刷新太快。
        time.sleep(args.sleep)

    env.close()


if __name__ == "__main__":
    main()
