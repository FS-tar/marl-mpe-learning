# -*- coding: utf-8 -*-
"""Factory helpers for MPE PettingZoo parallel environments."""

from __future__ import annotations

import importlib
from dataclasses import dataclass


SUPPORTED_MPE_ENVS = {
    "simple_spread_v3",
    "simple_tag_v3",
}


@dataclass(frozen=True)
class MPEEnvSpec:
    env_name: str
    source: str


def load_mpe_env_module(env_name: str):
    """Load an MPE env module, preferring mpe2 used by this project."""

    if env_name not in SUPPORTED_MPE_ENVS:
        supported = ", ".join(sorted(SUPPORTED_MPE_ENVS))
        raise ValueError(f"Unsupported MPE env '{env_name}'. Supported: {supported}")

    try:
        return importlib.import_module(f"mpe2.{env_name}"), "mpe2"
    except ImportError as mpe2_error:
        try:
            return importlib.import_module(f"pettingzoo.mpe.{env_name}"), "pettingzoo.mpe"
        except ImportError as pettingzoo_error:
            raise ImportError(
                f"Cannot import {env_name}. Please check mpe2 or pettingzoo[mpe]."
            ) from pettingzoo_error


def make_mpe_env(
    env_name: str,
    max_cycles: int,
    continuous_actions: bool = False,
    render_mode: str | None = None,
):
    """Create a parallel MPE env for training or inspection scripts."""

    module, source = load_mpe_env_module(env_name)
    kwargs = {
        "max_cycles": max_cycles,
        "continuous_actions": continuous_actions,
        "render_mode": render_mode,
    }

    try:
        env = module.parallel_env(**kwargs)
    except TypeError:
        kwargs.pop("continuous_actions")
        env = module.parallel_env(**kwargs)

    try:
        env.mpe_env_spec = MPEEnvSpec(env_name=env_name, source=source)
    except AttributeError:
        pass
    return env


def get_mpe_env_source(env) -> str:
    spec = getattr(env, "mpe_env_spec", None)
    if spec is None:
        return "unknown"
    return spec.source
