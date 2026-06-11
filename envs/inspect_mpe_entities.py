# -*- coding: utf-8 -*-
"""检查 MPE 图形界面中的实体对象。

这个脚本只用于理解环境里的 agent、adversary 和 landmark，不实现 PPO/QMIX/MADDPG。
"""

from __future__ import annotations

import argparse
import importlib
from collections import deque

import numpy as np


ENV_MODULES = {
    "simple": "simple_v3",
    "simple_spread": "simple_spread_v3",
    "simple_tag": "simple_tag_v3",
    "simple_adversary": "simple_adversary_v3",
    "simple_push": "simple_push_v3",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 MPE 环境中的实体。")
    parser.add_argument(
        "--env",
        choices=sorted(ENV_MODULES),
        default="simple_spread",
        help="要检查的 MPE 环境。",
    )
    return parser.parse_args()


def format_value(value) -> str:
    """把颜色、位置等值格式化成易读字符串。"""

    if value is None:
        return "None"
    return np.array2string(np.asarray(value), precision=4, separator=", ")


def entity_position(entity):
    """从 entity.state.p_pos 读取位置；如果没有就返回 None。"""

    return getattr(getattr(entity, "state", None), "p_pos", None)


def find_world(env):
    """尽量从 env 和 wrapper 链中找到底层 world 对象。"""

    queue = deque(
        [
            ("env", env),
            ("env.unwrapped", getattr(env, "unwrapped", None)),
            ("env.aec_env", getattr(env, "aec_env", None)),
            (
                "env.aec_env.unwrapped",
                getattr(getattr(env, "aec_env", None), "unwrapped", None),
            ),
        ]
    )
    seen_ids = set()

    while queue:
        label, obj = queue.popleft()
        if obj is None or id(obj) in seen_ids:
            continue
        seen_ids.add(id(obj))

        if hasattr(obj, "world"):
            return getattr(obj, "world"), label

        # 常见 wrapper 会把下一层环境放在这些属性里。
        for attr in ("env", "aec_env", "unwrapped", "par_env"):
            try:
                child = getattr(obj, attr, None)
            except Exception:
                child = None
            if child is not None and id(child) not in seen_ids:
                queue.append((f"{label}.{attr}", child))

    return None, None


def safe_attrs(obj) -> list[str]:
    """安全列出对象属性，访问失败时也不中断。"""

    try:
        return sorted(name for name in dir(obj) if not name.startswith("__"))
    except Exception as error:
        return [f"<dir failed: {error}>"]


def print_env_debug_info(env) -> None:
    """访问不到 world 时，打印当前 env 类型和属性，方便继续排查。"""

    print("world is not accessible.")
    print(f"env type: {type(env)}")
    print("env attrs:")
    print(", ".join(safe_attrs(env)))


def print_agent(agent) -> None:
    """打印一个 agent 的关键属性。"""

    print(
        f"  name={getattr(agent, 'name', '<unnamed>')}, "
        f"adversary={getattr(agent, 'adversary', None)}, "
        f"movable={getattr(agent, 'movable', None)}, "
        f"collide={getattr(agent, 'collide', None)}, "
        f"color={format_value(getattr(agent, 'color', None))}, "
        f"position={format_value(entity_position(agent))}"
    )


def print_landmark(landmark) -> None:
    """打印一个 landmark 的关键属性。"""

    print(
        f"  name={getattr(landmark, 'name', '<unnamed>')}, "
        f"movable={getattr(landmark, 'movable', None)}, "
        f"collide={getattr(landmark, 'collide', None)}, "
        f"color={format_value(getattr(landmark, 'color', None))}, "
        f"position={format_value(entity_position(landmark))}"
    )


def print_world_entities(world, world_path: str) -> None:
    """打印 world 中的 agents 和 landmarks。"""

    print(f"world found at: {world_path}")

    agents = list(getattr(world, "agents", []))
    print(f"\nagents ({len(agents)}):")
    for agent in agents:
        print_agent(agent)

    landmarks = list(getattr(world, "landmarks", []))
    print(f"\nlandmarks ({len(landmarks)}):")
    for landmark in landmarks:
        print_landmark(landmark)


def main() -> None:
    args = parse_args()
    env_module, source, module_name = load_mpe_env(args.env)

    # render_mode="human" 打开图形化窗口，方便把打印信息和画面对应起来。
    env = env_module.parallel_env(render_mode="human")
    observations, infos = env.reset()
    del observations

    print(f"env: {args.env} ({module_name})")
    print(f"source: {source}")
    print(f"possible_agents: {env.possible_agents}")
    print(f"reset infos: {infos}")

    world, world_path = find_world(env)
    if world is None:
        print_env_debug_info(env)
    else:
        print_world_entities(world, world_path)

    env.close()


if __name__ == "__main__":
    main()
