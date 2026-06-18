# -*- coding: utf-8 -*-
"""Inspect MPE simple_spread_v3 through the shared env factory."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from envs.mpe_env_factory import get_mpe_env_source, make_mpe_env


def reset_env(env):
    return env.reset()


def main() -> None:
    env_name = "simple_spread_v3"
    env = make_mpe_env(env_name, max_cycles=25)

    print(f"environment name: {env_name}")
    print(f"environment source: {get_mpe_env_source(env)}")
    print(f"possible_agents: {env.possible_agents}")

    print("\nSpaces:")
    for agent in env.possible_agents:
        print(f"- {agent}")
        print(f"  observation_space: {env.observation_space(agent)}")
        print(f"  action_space:      {env.action_space(agent)}")

    observations, infos = reset_env(env)
    del infos
    print("\nReset observation shapes:")
    for agent, observation in observations.items():
        print(f"- {agent}: {np.asarray(observation).shape}")

    actions = {
        agent: env.action_space(agent).sample()
        for agent in observations.keys()
    }
    next_observations, rewards, terminations, truncations, infos = env.step(actions)
    del next_observations, infos

    print("\nOne random step:")
    print(f"action keys: {list(actions.keys())}")
    print(f"reward keys: {list(rewards.keys())}")
    print(f"termination keys: {list(terminations.keys())}")
    print(f"truncation keys: {list(truncations.keys())}")
    print(f"rewards: {rewards}")
    print(f"terminations: {terminations}")
    print(f"truncations: {truncations}")

    env.close()


if __name__ == "__main__":
    main()
