# -*- coding: utf-8 -*-
"""Plot core training curves for a simplified QMIX run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

OUTPUT_BASE_DIR = ROOT_DIR / "outputs" / "qmix_mpe" / "simple_spread"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot simplified QMIX simple_spread run.")
    parser.add_argument("--run-dir", type=str, required=True)
    return parser.parse_args()


def to_float(value) -> float:
    if value in (None, ""):
        return np.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not math.isfinite(number):
        return np.nan
    return number


def read_rows(run_dir: Path) -> list[dict]:
    csv_path = run_dir / "train_log.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Cannot find train_log.csv: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        print(f"warning: config.json not found: {config_path}")
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def baseline_path(max_cycles: int, team_reward_mode: str) -> Path:
    return (
        OUTPUT_BASE_DIR
        / "random_baselines"
        / f"random_baseline_cycles{max_cycles}_mode_{team_reward_mode}.json"
    )


def read_random_baseline(run_dir: Path) -> float | None:
    config = read_config(run_dir)
    max_cycles = config.get("max_cycles")
    team_reward_mode = config.get("team_reward_mode", "mean")
    if max_cycles is None:
        print("warning: max_cycles missing in config.json; random baseline not plotted")
        return None
    path = baseline_path(int(max_cycles), str(team_reward_mode))
    if not path.is_file():
        print(
            "warning: matching random baseline not found; "
            f"expected {path}"
        )
        return None
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if data.get("max_cycles") != int(max_cycles) or data.get("team_reward_mode") != str(team_reward_mode):
        print(f"warning: random baseline metadata mismatch in {path}")
        return None
    value = data.get("random_team_return")
    if value is None:
        return None
    print(f"Using random baseline: {path}")
    return float(value)


def save_learning_status(
    rows: list[dict],
    run_dir: Path,
    random_team_return: float | None = None,
    mixer_type: str = "qmix",
) -> Path:
    points = [
        (
            to_float(row.get("episode")),
            to_float(row.get("eval_team_return")),
        )
        for row in rows
    ]
    points = [(episode, value) for episode, value in points if not np.isnan(value)]
    output_path = run_dir / "learning_status.png"

    plt.figure(figsize=(8, 4.5))
    if points:
        episodes, values = zip(*points)
        plt.plot(episodes, values, marker="o", label="QMIX eval return")
    if random_team_return is not None:
        plt.axhline(
            random_team_return,
            color="tab:orange",
            linestyle="--",
            label="random baseline",
        )
    plt.xlabel("Episode")
    plt.ylabel("Eval team return")
    plt.title(f"QMIX simple_spread learning status ({mixer_type})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def save_training_health(rows: list[dict], run_dir: Path, mixer_type: str = "qmix") -> Path:
    x_values = [to_float(row.get("episode")) for row in rows]
    series = {
        "mean_td_loss": [to_float(row.get("td_loss")) for row in rows],
        "epsilon": [to_float(row.get("epsilon")) for row in rows],
        "q_tot_mean": [to_float(row.get("q_tot_mean")) for row in rows],
        "target_q_tot_mean": [to_float(row.get("target_q_tot_mean")) for row in rows],
        "grad_norm": [to_float(row.get("grad_norm")) for row in rows],
    }
    output_path = run_dir / "training_health.png"

    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    axes_flat = axes.reshape(-1)
    for axis, (name, values) in zip(axes_flat, series.items()):
        axis.plot(x_values, values, linewidth=1.2)
        axis.set_title(name)
        axis.grid(True, alpha=0.3)
    for axis in axes_flat[len(series):]:
        axis.axis("off")
    axes[-1, 0].set_xlabel("Episode")
    axes[-1, 1].set_xlabel("Episode")
    fig.suptitle(f"QMIX training health ({mixer_type})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_all_plots(run_dir: Path) -> list[Path]:
    rows = read_rows(run_dir)
    config = read_config(run_dir)
    mixer_type = str(config.get("mixer_type", "qmix"))
    random_team_return = read_random_baseline(run_dir)
    return [
        save_learning_status(rows, run_dir, random_team_return=random_team_return, mixer_type=mixer_type),
        save_training_health(rows, run_dir, mixer_type=mixer_type),
    ]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    for path in save_all_plots(run_dir):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
