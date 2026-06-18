# -*- coding: utf-8 -*-
"""Overfit a fixed episode batch for recurrent QMIX diagnostics."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algorithms.qmix_rnn import EpisodeReplayBuffer, RNNAgent
from envs.mpe_env_factory import make_mpe_env
from experiments.qmix_mpe_rnn.train_qmix_rnn_simple_spread import (
    build_mixer,
    build_sequence_inputs,
    build_step_inputs,
    batch_states,
    done_flag,
    one_hot_np,
    reset_env,
    rnn_input_dim,
    select_actions_from_q,
    set_seed,
    team_reward,
    train_update,
    unroll_agent,
    obs_to_array,
    next_obs_to_array,
)


class FixedBatchBuffer:
    """Replay-like wrapper returning the same pre-sampled episode batch."""

    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        self.batch = batch

    def sample(self, batch_size: int, device: torch.device | str) -> dict[str, torch.Tensor]:
        del batch_size
        return {key: value.to(device) for key, value in self.batch.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit a fixed RNN QMIX episode batch.")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--rnn-hidden-dim", type=int, default=64)
    parser.add_argument("--loss-type", choices=("mse", "huber"), default="huber")
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--team-reward-mode", choices=("mean", "sum", "first"), default="mean")
    parser.add_argument("--mixer-type", choices=("qmix", "vdn"), default="vdn")
    parser.add_argument("--double-q", action="store_true", default=False)
    parser.add_argument("--mixer-weight-clip", type=float, default=1.0)
    parser.add_argument("--mixer-bias-clip", type=float, default=5.0)
    parser.add_argument("--mixer-weight-activation", choices=("softplus", "abs"), default="softplus")
    parser.add_argument("--use-layer-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-last-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-agent-id", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-shapes", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def collect_episode(
    env,
    agents: list[str],
    obs_dim: int,
    action_dim: int,
    max_cycles: int,
    reward_scale: float,
    team_reward_mode: str,
    seed: int,
) -> dict[str, np.ndarray]:
    observations, infos = reset_env(env, seed=seed)
    del infos
    episode_obs = []
    episode_actions = []
    episode_rewards = []
    episode_next_obs = []
    episode_dones = []

    for _ in range(max_cycles):
        if not observations:
            break
        obs_array = obs_to_array(observations, agents, obs_dim)
        actions = np.asarray([env.action_space(agent).sample() for agent in agents], dtype=np.int64)
        action_dict = {
            agent: int(action)
            for agent, action in zip(agents, actions)
            if agent in observations
        }
        next_observations, rewards, terminations, truncations, infos = env.step(action_dict)
        del infos
        reward = team_reward(rewards, team_reward_mode)
        done = done_flag(next_observations, terminations, truncations, agents)
        next_obs_array = next_obs_to_array(next_observations, agents, obs_dim, obs_array)

        episode_obs.append(obs_array)
        episode_actions.append(actions)
        episode_rewards.append([reward * reward_scale])
        episode_next_obs.append(next_obs_array)
        episode_dones.append([float(done)])

        if done:
            break
        observations = next_observations

    return {
        "obs": np.asarray(episode_obs, dtype=np.float32),
        "actions": np.asarray(episode_actions, dtype=np.int64),
        "rewards": np.asarray(episode_rewards, dtype=np.float32),
        "next_obs": np.asarray(episode_next_obs, dtype=np.float32),
        "dones": np.asarray(episode_dones, dtype=np.float32),
    }


def print_overfit_shapes(
    batch: dict[str, torch.Tensor],
    agent: RNNAgent,
    target_agent: RNNAgent,
    mixer: torch.nn.Module,
    target_mixer: torch.nn.Module,
    gamma: float,
    include_last_action: bool,
    include_agent_id: bool,
) -> None:
    obs = batch["obs"]
    actions = batch["actions"].long()
    rewards = batch["rewards"]
    next_obs = batch["next_obs"]
    dones = batch["dones"]
    filled = batch["filled"]

    with torch.no_grad():
        states = batch_states(obs)
        next_states = batch_states(next_obs)
        agent_inputs = build_sequence_inputs(
            obs,
            actions,
            agent.action_dim,
            include_last_action,
            include_agent_id,
            next_obs_inputs=False,
        )
        next_agent_inputs = build_sequence_inputs(
            next_obs,
            actions,
            agent.action_dim,
            include_last_action,
            include_agent_id,
            next_obs_inputs=True,
        )
        q_values = unroll_agent(agent, agent_inputs)
        chosen_qs = q_values.gather(3, actions.unsqueeze(-1)).squeeze(-1)
        q_tot = mixer(chosen_qs, states)
        target_next_q_values = unroll_agent(target_agent, next_agent_inputs)
        target_agent_qs = target_next_q_values.max(dim=3).values
        target_q_tot = target_mixer(target_agent_qs, next_states)
        td_target = rewards + gamma * (1.0 - dones) * target_q_tot

    print("\nFixed batch shapes:")
    for name, tensor in (
        ("obs", obs),
        ("actions", actions),
        ("rewards", rewards),
        ("dones", dones),
        ("filled", filled),
        ("chosen_qs", chosen_qs),
        ("q_tot", q_tot),
        ("target_q_tot", target_q_tot),
        ("td_target", td_target),
    ):
        print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_mpe_env("simple_spread_v3", max_cycles=args.max_cycles)
    observations, infos = reset_env(env, seed=args.seed)
    del observations, infos
    agents = list(env.possible_agents)
    obs_dim = int(env.observation_space(agents[0]).shape[0])
    action_dim = int(env.action_space(agents[0]).n)
    n_agents = len(agents)
    state_dim = n_agents * obs_dim
    input_dim = rnn_input_dim(
        obs_dim,
        action_dim,
        n_agents,
        args.include_last_action,
        args.include_agent_id,
    )

    replay_buffer = EpisodeReplayBuffer(args.episodes, n_agents, obs_dim)
    for index in range(args.episodes):
        episode = collect_episode(
            env=env,
            agents=agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_cycles=args.max_cycles,
            reward_scale=args.reward_scale,
            team_reward_mode=args.team_reward_mode,
            seed=args.seed + index + 1,
        )
        replay_buffer.add_episode(**episode)
    env.close()

    batch = replay_buffer.sample(min(args.batch_size, len(replay_buffer)), device)
    fixed_buffer = FixedBatchBuffer(batch)
    agent = RNNAgent(obs_dim, action_dim, args.rnn_hidden_dim, input_dim=input_dim).to(device)
    target_agent = RNNAgent(obs_dim, action_dim, args.rnn_hidden_dim, input_dim=input_dim).to(device)
    target_agent.load_state_dict(agent.state_dict())
    mixer = build_mixer(
        mixer_type=args.mixer_type,
        n_agents=n_agents,
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        use_layer_norm=args.use_layer_norm,
        mixer_weight_clip=args.mixer_weight_clip,
        mixer_bias_clip=args.mixer_bias_clip,
        mixer_weight_activation=args.mixer_weight_activation,
    ).to(device)
    target_mixer = build_mixer(
        mixer_type=args.mixer_type,
        n_agents=n_agents,
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        use_layer_norm=args.use_layer_norm,
        mixer_weight_clip=args.mixer_weight_clip,
        mixer_bias_clip=args.mixer_bias_clip,
        mixer_weight_activation=args.mixer_weight_activation,
    ).to(device)
    target_mixer.load_state_dict(mixer.state_dict())
    optimizer = torch.optim.Adam(
        list(agent.parameters()) + list(mixer.parameters()),
        lr=args.learning_rate,
    )

    print(f"agents={agents}, obs_dim={obs_dim}, action_dim={action_dim}, rnn_input_dim={input_dim}")
    print_overfit_shapes(
        batch=batch,
        agent=agent,
        target_agent=target_agent,
        mixer=mixer,
        target_mixer=target_mixer,
        gamma=args.gamma,
        include_last_action=args.include_last_action,
        include_agent_id=args.include_agent_id,
    )

    loss_first = None
    info = None
    for update_index in range(args.updates):
        info = train_update(
            agent=agent,
            target_agent=target_agent,
            mixer=mixer,
            target_mixer=target_mixer,
            optimizer=optimizer,
            replay_buffer=fixed_buffer,
            batch_size=batch["obs"].shape[0],
            gamma=args.gamma,
            tau=args.tau,
            grad_clip=args.grad_clip,
            loss_type=args.loss_type,
            double_q=args.double_q,
            include_last_action=args.include_last_action,
            include_agent_id=args.include_agent_id,
            device=device,
            update_index=update_index,
            debug_shapes=args.debug_shapes,
        )
        if loss_first is None:
            loss_first = info["td_loss"]
        if (update_index + 1) in (1, args.updates):
            print(f"update={update_index + 1} td_loss={info['td_loss']:.6f}")

    if info is None or loss_first is None:
        raise RuntimeError("No overfit updates were run.")

    loss_final = info["td_loss"]
    loss_ratio = loss_final / max(loss_first, 1e-12)
    print("\nOverfit episode result:")
    print(f"loss_first: {loss_first:.8f}")
    print(f"loss_final: {loss_final:.8f}")
    print(f"loss_ratio: {loss_ratio:.8f}")
    print(f"q_tot_mean: {info['q_tot_mean']:.6f}")
    print(f"q_tot_max: {info['q_tot_max']:.6f}")
    print(f"target_q_tot_mean: {info['target_q_tot_mean']:.6f}")


if __name__ == "__main__":
    main()
