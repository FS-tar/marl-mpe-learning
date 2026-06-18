# -*- coding: utf-8 -*-
"""诊断离散动作 MADDPG 的 actor 梯度是否能正确回传。

这个脚本不加载环境、不跑长训练，只构造一个假的 simple_tag batch：
- adversary_0/1/2 obs_dim=16
- agent_0 obs_dim=14
- action_dim=5
- batch_size=8

重点检查：
1. 当前 agent 的 actor action 是否来自可导的 Gumbel-Softmax。
2. actor_loss backward 后，当前 actor 是否有非零梯度。
3. 其他 agent actor 是否没有梯度，验证 detach/未参与图计算的逻辑没有串梯度。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algorithms.maddpg.maddpg_trainer import MADDPGTrainer
from algorithms.maddpg.networks import gumbel_softmax_action


AGENT_NAMES = ["adversary_0", "adversary_1", "adversary_2", "agent_0"]
OBS_DIMS = {
    "adversary_0": 16,
    "adversary_1": 16,
    "adversary_2": 16,
    "agent_0": 14,
}
ACTION_DIM = 5
BATCH_SIZE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug MADDPG actor gradients for discrete action modes."
    )
    parser.add_argument(
        "--actor-action-mode",
        choices=("gumbel_hard", "gumbel_soft", "softmax"),
        default="gumbel_hard",
    )
    parser.add_argument("--gumbel-tau", type=float, default=1.0)
    return parser.parse_args()


def actor_action_from_logits(
    logits: torch.Tensor,
    actor_action_mode: str,
    gumbel_tau: float,
) -> torch.Tensor:
    if actor_action_mode == "gumbel_hard":
        return gumbel_softmax_action(logits, temperature=gumbel_tau)
    if actor_action_mode == "gumbel_soft":
        return F.gumbel_softmax(logits, tau=gumbel_tau, hard=False, dim=-1)
    if actor_action_mode == "softmax":
        return torch.softmax(logits, dim=-1)
    raise ValueError(f"未知 actor_action_mode: {actor_action_mode}")


def build_fake_tensors(trainer: MADDPGTrainer) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    obs = {
        agent: torch.randn(
            BATCH_SIZE,
            OBS_DIMS[agent],
            dtype=torch.float32,
            device=trainer.device,
        )
        for agent in AGENT_NAMES
    }
    replay_actions = {}
    for agent in AGENT_NAMES:
        action_indices = torch.randint(
            low=0,
            high=ACTION_DIM,
            size=(BATCH_SIZE,),
            device=trainer.device,
        )
        replay_actions[agent] = torch.nn.functional.one_hot(
            action_indices,
            num_classes=ACTION_DIM,
        ).float()
    return obs, replay_actions


def zero_all_actor_grads(trainer: MADDPGTrainer) -> None:
    for maddpg_agent in trainer.agents.values():
        for actor in maddpg_agent.actors:
            actor.zero_grad(set_to_none=True)


def actor_grad_norms(actor: torch.nn.Module) -> dict[str, float | None]:
    norms = {}
    for name, param in actor.named_parameters():
        norms[name] = None if param.grad is None else float(param.grad.norm().item())
    return norms


def has_nonzero_grad(actor: torch.nn.Module) -> bool:
    for param in actor.parameters():
        if param.grad is not None and float(param.grad.abs().sum().item()) > 0.0:
            return True
    return False


def print_actor_grad_report(trainer: MADDPGTrainer, current_agent: str) -> None:
    print(f"\n=== Actor gradient report for {current_agent} ===")
    for agent_name in AGENT_NAMES:
        actor = trainer.agents[agent_name].actors[0]
        norms = actor_grad_norms(actor)
        status = "CURRENT" if agent_name == current_agent else "OTHER"
        print(f"[{status}] {agent_name}")
        for param_name, norm in norms.items():
            text = "None" if norm is None else f"{norm:.8f}"
            print(f"  {param_name}: grad_norm={text}")
        print(f"  nonzero_grad={has_nonzero_grad(actor)}")


def check_static_discrete_action_code() -> None:
    """打印源码层面的离散动作处理检查。

    训练 actor 时应使用可导的 Gumbel-Softmax；环境交互/eval 可以 sample 或 argmax。
    这里只做字符串级检查，真正的梯度通路由后面的 backward 诊断验证。
    """

    trainer_path = ROOT_DIR / "algorithms" / "maddpg" / "maddpg_trainer.py"
    agent_path = ROOT_DIR / "algorithms" / "maddpg" / "maddpg_agent.py"
    trainer_source = trainer_path.read_text(encoding="utf-8")
    agent_source = agent_path.read_text(encoding="utf-8")

    print("=== Static discrete-action checks ===")
    print(
        "actor update has action mode helper:",
        "_actor_action_from_logits(logits)" in trainer_source,
    )
    print(
        "actor update supports gumbel_hard/gumbel_soft/softmax:",
        all(mode in trainer_source for mode in ("gumbel_hard", "gumbel_soft", "softmax")),
    )
    print(
        "other replay actions are detached:",
        ".detach()" in trainer_source and "batch.one_hot_actions" in trainer_source,
    )
    print(
        "environment/eval action path can use argmax:",
        "one_hot_from_logits(logits)" in agent_source,
    )
    print(
        "environment training action path can sample categorical:",
        "Categorical" in agent_source and "sample()" in agent_source,
    )


def run_single_actor_backward(
    trainer: MADDPGTrainer,
    current_agent: str,
    actor_action_mode: str,
    gumbel_tau: float,
) -> None:
    obs, replay_actions = build_fake_tensors(trainer)
    global_obs = torch.cat([obs[agent] for agent in AGENT_NAMES], dim=-1)

    zero_all_actor_grads(trainer)
    maddpg_agent = trainer.agents[current_agent]
    actor = maddpg_agent.actors[0]
    critic = maddpg_agent.critics[0]

    logits = actor(obs[current_agent])
    current_action = actor_action_from_logits(
        logits,
        actor_action_mode=actor_action_mode,
        gumbel_tau=gumbel_tau,
    )
    mixed_actions = {
        agent: replay_actions[agent].detach()
        for agent in AGENT_NAMES
    }
    mixed_actions[current_agent] = current_action
    mixed_global_actions = torch.cat(
        [mixed_actions[agent] for agent in AGENT_NAMES],
        dim=-1,
    )

    print(f"\n=== Action graph checks for {current_agent} ===")
    print(f"actor_action_mode={actor_action_mode}, gumbel_tau={gumbel_tau}")
    print(f"current_action.requires_grad={current_action.requires_grad}")
    print(
        "mixed_actions[current_agent].requires_grad="
        f"{mixed_actions[current_agent].requires_grad}"
    )
    for agent in AGENT_NAMES:
        if agent == current_agent:
            continue
        print(
            f"mixed_actions[{agent}].requires_grad="
            f"{mixed_actions[agent].requires_grad}"
        )

    # 和训练代码一致：actor loss 通过 centralized critic 反传到当前 actor；
    # critic 参数临时冻结，避免这里统计 actor 梯度时混入 critic 参数更新。
    for param in critic.parameters():
        param.requires_grad_(False)
    actor_loss = -critic(global_obs, mixed_global_actions).mean()
    actor_loss.backward()
    for param in critic.parameters():
        param.requires_grad_(True)

    print(f"actor_loss={float(actor_loss.item()):.8f}")
    print_actor_grad_report(trainer, current_agent)

    current_has_grad = has_nonzero_grad(actor)
    other_has_grad = {
        agent: has_nonzero_grad(trainer.agents[agent].actors[0])
        for agent in AGENT_NAMES
        if agent != current_agent
    }
    print(f"CHECK current actor has nonzero grad: {current_has_grad}")
    print(f"CHECK other actors have no grad: {not any(other_has_grad.values())}")
    if not current_has_grad:
        print("WARNING: 当前 actor 梯度为 0，actor update 可能断了。")
    if any(other_has_grad.values()):
        print(f"WARNING: 其他 actor 出现梯度，detach/图隔离可能有问题: {other_has_grad}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(1)
    check_static_discrete_action_code()

    trainer = MADDPGTrainer(
        agent_names=AGENT_NAMES,
        obs_dims=OBS_DIMS,
        action_dim=ACTION_DIM,
        hidden_dim=128,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.95,
        tau=0.01,
        buffer_size=128,
        ensemble_size=1,
        actor_action_mode=args.actor_action_mode,
        gumbel_tau=args.gumbel_tau,
        device="cpu",
    )

    print("\n=== Fake batch setup ===")
    print(f"agents={AGENT_NAMES}")
    print(f"obs_dims={OBS_DIMS}")
    print(f"action_dim={ACTION_DIM}")
    print(f"batch_size={BATCH_SIZE}")
    print(f"actor_action_mode={args.actor_action_mode}")
    print(f"gumbel_tau={args.gumbel_tau}")

    for agent in AGENT_NAMES:
        run_single_actor_backward(
            trainer,
            agent,
            actor_action_mode=args.actor_action_mode,
            gumbel_tau=args.gumbel_tau,
        )


if __name__ == "__main__":
    main()
