"""Run a small random-policy test on the MPE simple_spread environment.

This script is intentionally simple. It is meant to help beginners check that
the MPE environment can be imported, reset, stepped, and rendered before any
reinforcement learning algorithm is added.
"""

from __future__ import annotations

import argparse
import importlib
from typing import Any


def load_simple_spread_module() -> tuple[Any, str]:
    """Import simple_spread from mpe2 first, then fall back to PettingZoo.

    mpe2 is the newer package for MPE environments. Some tutorials and older
    projects still use pettingzoo.mpe.simple_spread_v3, so we keep that fallback
    to make the learning project easier to run on different machines.
    """

    candidates = (
        ("mpe2.simple_spread_v3", "mpe2"),
        ("pettingzoo.mpe.simple_spread_v3", "pettingzoo.mpe"),
    )

    last_error: Exception | None = None
    for module_name, provider_name in candidates:
        try:
            return importlib.import_module(module_name), provider_name
        except ImportError as exc:
            last_error = exc

    raise ImportError(
        "Could not import simple_spread_v3 from mpe2 or pettingzoo.mpe. "
        "Please install the project dependencies with `pip install -r requirements.txt`."
    ) from last_error


def make_env(render_mode: str | None):
    """Create a parallel simple_spread environment.

    The parallel API returns observations, rewards, termination flags, and
    truncation flags for all active agents at the same time. This makes it a
    friendly first interface for MARL experiments.
    """

    simple_spread_v3, provider_name = load_simple_spread_module()
    env = simple_spread_v3.parallel_env(render_mode=render_mode)
    return env, provider_name


def print_spaces(env) -> None:
    """Print each agent's observation and action spaces."""

    print("Agent spaces:")
    for agent in env.possible_agents:
        print(f"- {agent}")
        print(f"  observation_space: {env.observation_space(agent)}")
        print(f"  action_space:      {env.action_space(agent)}")


def run_random_episode(env, episode_index: int) -> None:
    """Run one episode with a random action for every active agent."""

    observations, infos = env.reset()
    del observations, infos

    episode_rewards = {agent: 0.0 for agent in env.possible_agents}
    step_count = 0

    while env.agents:
        # Only active agents should act. Agents may disappear after they are
        # done, so we build the action dictionary from env.agents each step.
        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }

        observations, rewards, terminations, truncations, infos = env.step(actions)
        del observations, terminations, truncations, infos

        for agent, reward in rewards.items():
            episode_rewards[agent] += float(reward)

        step_count += 1

    total_reward = sum(episode_rewards.values())
    per_agent = ", ".join(
        f"{agent}: {reward:.3f}"
        for agent, reward in episode_rewards.items()
    )
    print(
        f"Episode {episode_index}: total_reward={total_reward:.3f}, "
        f"steps={step_count}, per_agent={{ {per_agent} }}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test MPE simple_spread with a random policy."
    )
    parser.add_argument(
        "--render-mode",
        default=None,
        choices=(None, "human"),
        help='Use "human" to open the environment renderer. Default: None.',
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of random-policy episodes to run. Default: 5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env, provider_name = make_env(render_mode=args.render_mode)

    print(f"Using simple_spread_v3 from: {provider_name}")
    print_spaces(env)

    for episode_index in range(1, args.episodes + 1):
        run_random_episode(env, episode_index)

    env.close()


if __name__ == "__main__":
    main()
