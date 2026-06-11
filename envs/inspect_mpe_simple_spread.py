"""详细检查 MPE simple_spread 环境。

这个脚本只用于学习 MPE 环境接口，不实现 PPO、QMIX、MADDPG 等算法。
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
    """把 observation 展平成一维，只显示前几个数，方便阅读。"""

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
    """打印每个 agent 的 observation shape 和前 5 个数值。"""

    print(title)
    for agent, observation in observations.items():
        array = np.asarray(observation)
        print(
            f"- {agent}: shape={array.shape}, "
            f"first_5_values={preview_observation(array)}"
        )


def main() -> None:
    simple_spread_v3, source = load_simple_spread()

    # parallel_env 是 Parallel API：每个 step 同时给多个 agent 传入动作。
    # 注意：mpe2 某些版本返回的是 aec_to_parallel_wrapper。
    # reset 之前不要读取当前 agents；reset 后以 observations.keys() 为准。
    env = simple_spread_v3.parallel_env(render_mode=None)

    print(f"Loaded simple_spread_v3 from: {source}")

    # possible_agents 是环境中“可能出现”的全部 agent 名称。
    # reset 之前只打印 possible_agents，不访问当前活跃 agents。
    print(f"possible_agents: {env.possible_agents}")

    print("\nSpaces:")
    for agent in env.possible_agents:
        # observation_space：该 agent 观测向量的形状、范围和数据类型。
        print(f"- {agent}")
        print(f"  observation_space: {env.observation_space(agent)}")

        # action_space：该 agent 可以采取的动作集合。
        # simple_spread 常见是离散动作，例如不动、向左、向右、向上、向下。
        print(f"  action_space:      {env.action_space(agent)}")

    # reset 开始一个新的 episode。
    # observations：字典，agent 名称 -> 该 agent 的初始观测。
    # infos：字典，agent 名称 -> 环境提供的额外信息，常用于调试。
    observations, infos = env.reset()

    # reset 后，用 observations.keys() 表示当前需要行动的 agents。
    current_agents = list(observations.keys())
    print(f"\nagents after reset: {current_agents}")
    print(f"observation keys: {current_agents}")
    print(f"reset infos: {infos}")
    print_observations("\nInitial observations:", observations)

    print("\nRandom steps:")
    for step_index in range(1, 4):
        current_agents = list(observations.keys())
        if not current_agents:
            print("No observations returned; episode ended.")
            break

        # actions：字典，agent 名称 -> 本 step 给它的动作。
        # 这里从每个 agent 的 action_space 随机采样，只用于测试环境交互。
        actions = {
            agent: env.action_space(agent).sample()
            for agent in current_agents
        }

        # step 执行一步环境交互，并返回五个字典：
        # observations：下一步仍需要行动的 agent 的观测。
        # rewards：每个 agent 在这一步得到的奖励。
        # terminations：任务自然结束标记。
        # truncations：因为步数上限等外部条件导致的截断标记。
        # infos：额外调试信息或环境细节。
        observations, rewards, terminations, truncations, infos = env.step(actions)

        print(f"\nStep {step_index}")
        print(f"actions:      {actions}")
        print(f"rewards:      {rewards}")
        print(f"terminations: {terminations}")
        print(f"truncations:  {truncations}")
        print(f"infos:        {infos}")
        print_observations("observations:", observations)

        # 只随机执行最多 3 step；如果环境提前结束，也立即停止。
        if not observations or all_agents_done(terminations, truncations):
            print("Episode ended after this step.")
            break

    env.close()


if __name__ == "__main__":
    main()
