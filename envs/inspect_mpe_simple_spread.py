# -*- coding: utf-8 -*-
"""详细检查 MPE simple_spread 环境。

这个脚本只用于学习 MPE 环境接口，不实现 PPO、QMIX、MADDPG 等算法。
重点是把每一段 print 输出和对应代码逻辑连接起来，方便逐行理解。
"""

from __future__ import annotations

import numpy as np


def load_simple_spread():
    """优先使用 mpe2；如果本机没有 mpe2，再回退到 PettingZoo 的 MPE。"""

    try:
        from mpe2 import simple_spread_v3

        return simple_spread_v3, "mpe2"
    except ImportError:
        from pettingzoo.mpe import simple_spread_v3

        return simple_spread_v3, "pettingzoo.mpe"


def preview_observation(observation, limit: int = 5) -> str:
    """把 observation 展平成一维，只显示前 5 个数，方便阅读。"""

    values = np.asarray(observation).reshape(-1)
    return np.array2string(values[:limit], precision=3, separator=", ")


def all_agents_done(terminations: dict, truncations: dict) -> bool:
    """判断当前 step 返回的 agent 是否都已经结束。

    terminated 表示任务自然结束；truncated 表示因为步数上限等外部条件结束。
    只要一个 agent 的 terminated 或 truncated 为 True，就说明它不需要继续行动。
    """

    agents = set(terminations) | set(truncations)
    if not agents:
        return False

    return all(
        terminations.get(agent, False) or truncations.get(agent, False)
        for agent in agents
    )


def print_observations(title: str, observations: dict) -> None:
    """打印每个 agent 的 observation shape 和前 5 个数值。

    这个函数对应输出中的 Initial observations 或 step 后的 observations。
    它展示的是字典 observations 里的每一个 value。
    """

    print(title)
    for agent, observation in observations.items():
        array = np.asarray(observation)
        print(
            f"- {agent}: shape={array.shape}, "
            f"first_5_values={preview_observation(array)}"
        )


def main() -> None:
    # load_simple_spread() 返回两个值：
    # 1. simple_spread_v3 模块本身。
    # 2. source 字符串，记录这次到底是从 mpe2 还是 pettingzoo.mpe 加载。
    simple_spread_v3, source = load_simple_spread()

    # parallel_env 是 Parallel API：每个 step 同时给多个 agent 传入动作。
    # 注意：mpe2 某些版本返回的是 aec_to_parallel_wrapper。
    # reset 之前不要读取 env.agents；reset 后以 observations.keys() 为准。
    env = simple_spread_v3.parallel_env(render_mode=None)

    # 下面这行 print 对应输出：
    # Loaded simple_spread_v3 from: mpe2
    # 其中 mpe2 来自上面的 source 变量。
    print(f"Loaded simple_spread_v3 from: {source}")

    # possible_agents 来自 env.possible_agents。
    # 它表示环境中“可能出现”的全部 agent 名称。
    # reset 之前只打印 possible_agents，不访问当前活跃 agents。
    print(f"possible_agents: {env.possible_agents}")

    # Spaces 标题下面的内容来自：
    # env.observation_space(agent) 和 env.action_space(agent)。
    # 这一步是在查看每个 agent 的输入空间和输出动作空间。
    print("\nSpaces from env.observation_space(agent) and env.action_space(agent):")
    for agent in env.possible_agents:
        print(f"- {agent}")

        # observation_space：该 agent 观测向量的形状、范围和数据类型。
        # 后续 PPO/QMIX/MADDPG 的网络输入维度会参考它。
        print(f"  observation_space: {env.observation_space(agent)}")

        # action_space：该 agent 可以采取的动作集合。
        # simple_spread 常见是 Discrete(5)，也就是 0~4 五个离散动作。
        # 后续算法会用神经网络输出动作，而不是手写固定动作。
        print(f"  action_space:      {env.action_space(agent)}")

    # reset 开始一个新的 episode。
    # observations：字典，agent 名称 -> 该 agent 的初始观测。
    # infos：字典，agent 名称 -> 环境提供的额外信息，常用于调试。
    observations, infos = env.reset()

    # reset 后，用 observations.keys() 表示当前需要行动的 agents。
    # 下面两个 print 对应输出中的：
    # agents after reset
    # observation keys
    current_agents = list(observations.keys())
    print(f"\nagents after reset from observations.keys(): {current_agents}")
    print(f"observation keys from observations.keys(): {current_agents}")
    print(f"reset infos: {infos}")

    # Initial observations 来自 reset 后返回的 observations 字典。
    # 每个 value 是一个 agent 的初始 observation。
    print_observations("\nInitial observations from reset observations:", observations)

    print("\nRandom steps:")
    for step_index in range(1, 4):
        # 每个 step 开始时，都重新从 observations.keys() 取当前 agents。
        # 这样不依赖 env.agents，可以兼容 mpe2 的 aec_to_parallel_wrapper。
        current_agents = list(observations.keys())
        if not current_agents:
            print("No observations returned; episode ended.")
            break

        # actions 来自 env.action_space(agent).sample()。
        # sample() 表示从合法动作空间中随机抽一个动作。
        # 后续实现 PPO/QMIX/MADDPG 时，这一行会被替换：
        # 随机 sample 动作 -> 神经网络根据 observation 输出动作。
        actions = {
            agent: env.action_space(agent).sample()
            for agent in current_agents
        }

        # env.step(actions) 执行一步环境交互，并返回五个字典：
        # observations：执行动作后的新 observation。
        # rewards：每个 agent 在这一步得到的奖励。
        # terminations：任务自然结束标记。
        # truncations：因为步数上限等外部条件导致的截断标记。
        # infos：额外调试信息或环境细节。
        observations, rewards, terminations, truncations, infos = env.step(actions)

        print(f"\nStep {step_index}")

        # 下面几段 print 分别对应 env.step(actions) 的输入和返回值。
        # actions 是刚刚随机 sample 出来的动作字典。
        print(f"actions from env.action_space(agent).sample(): {actions}")

        # rewards、terminations、truncations、infos 都来自 env.step(actions)。
        print(f"rewards from env.step(actions):      {rewards}")
        print(f"terminations from env.step(actions): {terminations}")
        print(f"truncations from env.step(actions):  {truncations}")
        print(f"infos from env.step(actions):        {infos}")

        # step 后的 observations 是执行 actions 之后的新 observation。
        # 它会作为下一轮构造 actions 的输入。
        print_observations(
            "observations after env.step(actions):",
            observations,
        )

        # 只随机执行最多 3 step；如果环境提前结束，也立即停止。
        if not observations or all_agents_done(terminations, truncations):
            print("Episode ended after this step.")
            break

    env.close()


if __name__ == "__main__":
    main()
