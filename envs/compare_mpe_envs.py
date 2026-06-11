"""Compare several MPE environments at the interface level.

This script only imports environments and prints their basic spaces. It is a
small learning aid before implementing any MARL algorithm.
"""

from __future__ import annotations


ENV_NAMES = (
    "simple_spread_v3",
    "simple_tag_v3",
    "simple_adversary_v3",
)


def load_env_module(env_name: str):
    """Try to load one MPE environment from mpe2, then PettingZoo."""

    try:
        mpe2 = __import__("mpe2", fromlist=[env_name])
        return getattr(mpe2, env_name), "mpe2"
    except (ImportError, AttributeError) as mpe2_error:
        try:
            pettingzoo_mpe = __import__("pettingzoo.mpe", fromlist=[env_name])
            return getattr(pettingzoo_mpe, env_name), "pettingzoo.mpe"
        except (ImportError, AttributeError) as pettingzoo_error:
            return None, f"{mpe2_error}; {pettingzoo_error}"


def inspect_env(env_name: str) -> None:
    env_module, source = load_env_module(env_name)

    if env_module is None:
        print(f"\n{env_name}: import failed")
        print(f"  reason: {source}")
        return

    env = env_module.parallel_env(render_mode=None)
    env.reset()

    print(f"\n{env_name}")
    print(f"  source: {source}")
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
