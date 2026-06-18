# -*- coding: utf-8 -*-
"""Overfit a fixed random batch to smoke-test QMIX update mechanics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algorithms.qmix import AgentQNetwork
from algorithms.qmix.replay_buffer import ReplayBuffer
from envs.mpe_env_factory import make_mpe_env
from experiments.qmix_mpe.train_qmix_simple_spread import (
    done_flag,
    hard_update,
    build_mixer,
    make_state,
    obs_to_array,
    reset_env,
    team_reward,
    train_update,
)


class FixedBatchReplay:
    def __init__(self, batch: dict[str, torch.Tensor], n_agents: int, obs_dim: int, state_dim: int) -> None:
        self.batch = batch
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim

    def sample(self, batch_size: int, device: torch.device | str) -> dict[str, torch.Tensor]:
        del batch_size
        return {key: value.to(device) for key, value in self.batch.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit a fixed QMIX batch.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--max-cycles", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--team-reward-mode", choices=("mean", "sum", "first"), default="mean")
    parser.add_argument("--loss-type", choices=("mse", "huber"), default="huber")
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--mixer-type", choices=("qmix", "vdn"), default="qmix")
    parser.add_argument("--double-q", action="store_true", default=False)
    parser.add_argument("--use-layer-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixer-weight-clip", type=float, default=1.0)
    parser.add_argument("--mixer-bias-clip", type=float, default=5.0)
    parser.add_argument("--mixer-weight-activation", choices=("softplus", "abs"), default="softplus")
    parser.add_argument("--qtot-l2-coef", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def collect_random_batch(args: argparse.Namespace, env, agents: list[str], obs_dim: int, state_dim: int) -> ReplayBuffer:
    buffer = ReplayBuffer(args.batch_size, len(agents), obs_dim, state_dim)
    while len(buffer) < args.batch_size:
        observations, infos = reset_env(env, seed=args.seed + len(buffer))
        del infos
        for _ in range(args.max_cycles):
            if not observations or len(buffer) >= args.batch_size:
                break
            obs_array = obs_to_array(observations, agents, obs_dim)
            actions = np.asarray(
                [env.action_space(agent).sample() for agent in agents],
                dtype=np.int64,
            )
            action_dict = {
                agent: int(action)
                for agent, action in zip(agents, actions)
                if agent in observations
            }
            next_observations, rewards, terminations, truncations, infos = env.step(action_dict)
            del infos
            reward = team_reward(rewards, args.team_reward_mode) * args.reward_scale
            done = done_flag(next_observations, terminations, truncations, agents)
            if all(agent in next_observations for agent in agents):
                next_obs_array = obs_to_array(next_observations, agents, obs_dim)
            else:
                next_obs_array = obs_array.copy()
            buffer.add(
                obs=obs_array,
                state=make_state(obs_array),
                actions=actions,
                reward=reward,
                next_obs=next_obs_array,
                next_state=make_state(next_obs_array),
                done=done,
            )
            if done:
                break
            observations = next_observations
    return buffer


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_mpe_env("simple_spread_v3", max_cycles=args.max_cycles)
    observations, infos = reset_env(env, seed=args.seed)
    del infos
    agents = list(env.possible_agents)
    obs_dim = int(env.observation_space(agents[0]).shape[0])
    action_dim = int(env.action_space(agents[0]).n)
    state_dim = len(agents) * obs_dim

    source_buffer = collect_random_batch(args, env, agents, obs_dim, state_dim)
    fixed_batch = source_buffer.sample(args.batch_size, device)
    env.close()
    fixed_replay = FixedBatchReplay(fixed_batch, len(agents), obs_dim, state_dim)

    q_networks = torch.nn.ModuleList(
        [AgentQNetwork(obs_dim, action_dim, args.hidden_dim) for _ in agents]
    ).to(device)
    target_q_networks = torch.nn.ModuleList(
        [AgentQNetwork(obs_dim, action_dim, args.hidden_dim) for _ in agents]
    ).to(device)
    mixer = build_mixer(
        mixer_type=args.mixer_type,
        n_agents=len(agents),
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        use_layer_norm=args.use_layer_norm,
        mixer_weight_clip=args.mixer_weight_clip,
        mixer_bias_clip=args.mixer_bias_clip,
        mixer_weight_activation=args.mixer_weight_activation,
    ).to(device)
    target_mixer = build_mixer(
        mixer_type=args.mixer_type,
        n_agents=len(agents),
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        use_layer_norm=args.use_layer_norm,
        mixer_weight_clip=args.mixer_weight_clip,
        mixer_bias_clip=args.mixer_bias_clip,
        mixer_weight_activation=args.mixer_weight_activation,
    ).to(device)
    hard_update(q_networks, target_q_networks)
    hard_update(mixer, target_mixer)
    optimizer = torch.optim.Adam(
        list(q_networks.parameters()) + list(mixer.parameters()),
        lr=args.learning_rate,
    )

    losses = []
    q_tot_max_values = []
    for update in range(args.updates):
        info = train_update(
            q_networks=q_networks,
            target_q_networks=target_q_networks,
            mixer=mixer,
            target_mixer=target_mixer,
            optimizer=optimizer,
            replay_buffer=fixed_replay,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            grad_clip=args.grad_clip,
            loss_type=args.loss_type,
            qtot_l2_coef=args.qtot_l2_coef,
            double_q=args.double_q,
            device=device,
            update_index=update,
            debug_shapes=(update == 0),
        )
        losses.append(info["td_loss"])
        q_tot_max_values.append(info["q_tot_max"])
        if update in (0, 1, 2, 9, 49, 99, args.updates - 1):
            print(
                f"update={update + 1:03d} loss={info['td_loss']:.6f} "
                f"q_tot_mean={info['q_tot_mean']:.4f} q_tot_max={info['q_tot_max']:.4f} "
                f"target_q_tot_mean={info['target_q_tot_mean']:.4f}"
            )

    first_loss = losses[0]
    final_loss = losses[-1]
    print("\nOverfit batch summary:")
    print(f"loss_first={first_loss:.6f}")
    print(f"loss_final={final_loss:.6f}")
    print(f"loss_ratio={final_loss / max(first_loss, 1e-12):.6f}")
    if final_loss > first_loss * 0.8:
        print("warning: loss did not clearly decrease; update/gather/mixer/target may have a bug")
    if max(q_tot_max_values) > 100.0:
        print("warning: q_tot quickly reached a large scale; mixer scale may still be too high")


if __name__ == "__main__":
    main()
