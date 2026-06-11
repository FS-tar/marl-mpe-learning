# -*- coding: utf-8 -*-
"""图形化观察 MPE simple_spread_v3。

这个脚本只用于观察环境画面和随机动作效果，不实现 PPO、QMIX、MADDPG。
"""

from __future__ import annotations

import time


def load_simple_spread():
    """优先从 mpe2 加载 simple_spread_v3，失败后回退到 PettingZoo。"""

    try:
        from mpe2 import simple_spread_v3

        return simple_spread_v3, "mpe2"
    except ImportError:
        from pettingzoo.mpe import simple_spread_v3

        return simple_spread_v3, "pettingzoo.mpe"


def all_agents_done(terminations: dict, truncations: dict) -> bool:
    """判断所有 agent 是否都已经 terminated 或 truncated。"""

    agents = set(terminations) | set(truncations)
    if not agents:
        return False

    return all(
        terminations.get(agent, False) or truncations.get(agent, False)
        for agent in agents
    )


def main() -> None:
    simple_spread_v3, source = load_simple_spread()

    # render_mode="human" 会打开图形窗口。
    # max_cycles=200 表示一个 episode 最多运行 200 个环境 step。
    env = simple_spread_v3.parallel_env(render_mode="human", max_cycles=200)
    print(f"Loaded simple_spread_v3 from: {source}")

    # reset 开始一个新 episode。
    # observations 是字典：agent 名称 -> 当前 observation。
    observations, infos = env.reset()
    print(f"reset infos: {infos}")

    for step in range(1, 201):
        current_agents = list(observations.keys())
        if not current_agents:
            print("No active observations; episode ended.")
            break

        # 每一步给每个当前 agent 一个随机动作。
        # 后续真正训练 PPO/QMIX/MADDPG 时，这里会替换为神经网络输出动作。
        actions = {
            agent: env.action_space(agent).sample()
            for agent in current_agents
        }

        observations, rewards, terminations, truncations, infos = env.step(actions)
        del infos

        print(f"step={step}, actions={actions}, rewards={rewards}")

        # human render 可能已经由环境自动刷新；sleep 让画面不要闪得太快。
        time.sleep(0.05)

        if all_agents_done(terminations, truncations):
            print("All agents terminated or truncated; episode ended early.")
            break

    env.close()


if __name__ == "__main__":
    main()
