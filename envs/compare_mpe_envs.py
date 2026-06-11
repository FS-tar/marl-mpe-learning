"""对比多个 MPE 环境的接口信息。

这个脚本只查看环境本身，不实现 PPO/QMIX/MADDPG。
"""

from __future__ import annotations

import importlib

ENV_NAMES = (
    "simple_v3",
    "simple_spread_v3",
    "simple_adversary_v3",
    "simple_tag_v3",
    "simple_push_v3",
    "simple_reference_v3",
    "simple_speaker_listener_v4",
    "simple_world_comm_v3",
)


def load_env_module(env_name: str):
    """优先从 mpe2 导入环境，失败后再尝试 pettingzoo.mpe。"""

    try:
        return importlib.import_module(f"mpe2.{env_name}"), "mpe2"
    except ImportError as mpe2_error:
        try:
            return importlib.import_module(f"pettingzoo.mpe.{env_name}"), "pettingzoo.mpe"
        except ImportError as pettingzoo_error:
            return None, f"{mpe2_error}; {pettingzoo_error}"


def inspect_env(env_name: str) -> None:
    """打印单个环境的 agent 列表、agent 数量和空间信息。"""

    env_module, source = load_env_module(env_name)

    if env_module is None:
        print(f"\n{env_name}: import failed")
        print(f"  reason: {source}")
        return

    try:
        env = env_module.parallel_env(render_mode=None)
        env.reset()
    except Exception as error:
        print(f"\n{env_name}: create/reset failed")
        print(f"  reason: {error}")
        return

    print(f"\n{env_name}")
    print(f"  source: {source}")
    print(f"  possible_agents: {env.possible_agents}")
    print(f"  agent_count: {len(env.possible_agents)}")

    for agent in env.possible_agents:
        print(f"  - {agent}")
        print(f"    observation_space: {env.observation_space(agent)}")
        print(f"    action_space:      {env.action_space(agent)}")

    env.close()


def main() -> None:
    print("MPE environment comparison")
    for env_name in ENV_NAMES:
        inspect_env(env_name)


if __name__ == "__main__":
    main()
