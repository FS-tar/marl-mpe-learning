# -*- coding: utf-8 -*-
"""Plot and summarize one MADDPG simple_tag_v3 run.

Default output is intentionally small:
- learning_status.png
- training_health.png
- touch_counts.png, only when touch-count data exists
- summary.txt

Use --save-debug-plots to also generate detailed diagnostic plots.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Iterable

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    from PIL import Image, ImageDraw, ImageFont

    plt = None
    HAS_MATPLOTLIB = False


RANDOM_ADVERSARY_BASELINE = 17.0
RANDOM_PREY_BASELINE = -181.9
RANDOM_TOUCH_BASELINE = 1.7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot MADDPG simple_tag_v3 charts for one run directory."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--save-debug-plots", action="store_true")
    return parser.parse_args()


def parse_optional_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in {"none", "nan"}:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def is_nan_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        value = value.strip().lower()
        return value == "nan"
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def read_train_log(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            add_missing_touch_counts(row)
            rows.append(row)
    return rows


def add_missing_touch_counts(row: dict) -> None:
    touch_sources = {
        "train_touch_count": "adversary_team_return",
        "eval_self_play_touch_count": "eval_self_play_adversary_return",
        "eval_adv_vs_random_prey_touch_count": (
            "eval_adv_vs_random_prey_adversary_return"
        ),
        "eval_random_adv_vs_prey_touch_count": (
            "eval_random_adv_vs_prey_adversary_return"
        ),
    }
    for touch_key, return_key in touch_sources.items():
        if row.get(touch_key) not in (None, ""):
            continue
        adversary_return = parse_optional_float(row.get(return_key))
        row[touch_key] = (
            "" if adversary_return is None else str(adversary_return / 10.0)
        )


def series_points(rows: Iterable[dict], key: str) -> tuple[list[float], list[float]]:
    xs = []
    ys = []
    for row in rows:
        episode = parse_optional_float(row.get("episode"))
        value = parse_optional_float(row.get(key))
        if episode is None or value is None:
            continue
        xs.append(episode)
        ys.append(value)
    return xs, ys


def has_valid_series(rows: list[dict], key: str) -> bool:
    xs, ys = series_points(rows, key)
    return bool(xs and ys)


def latest_valid(rows: list[dict], key: str) -> float | None:
    for row in reversed(rows):
        value = parse_optional_float(row.get(key))
        if value is not None:
            return value
    return None


def better_text(value: float | None, baseline: float, higher_is_better: bool = True) -> str:
    if value is None:
        return "not available"
    is_better = value > baseline if higher_is_better else value < baseline
    return "better" if is_better else "worse"


def training_health(rows: list[dict]) -> str:
    if any(
        is_nan_value(row.get("mean_critic_loss")) or is_nan_value(row.get("mean_actor_loss"))
        for row in rows
    ):
        return "NaN detected"
    critic_loss = latest_valid(rows, "mean_critic_loss")
    if critic_loss is not None and critic_loss > 100.0:
        return "critic may be unstable"
    return "OK"


def policy_collapse_status(rows: list[dict]) -> str:
    entropy = latest_valid(rows, "mean_actor_entropy")
    if entropy is None:
        return "not checked"
    if entropy < 0.1:
        return "high collapse risk"
    return "no obvious collapse from entropy"


def actor_entropy_note(rows: list[dict]) -> str | None:
    if latest_valid(rows, "mean_actor_entropy") is None:
        return "mean_actor_entropy not available because no actor update happened."
    return None


def safe_replace_png(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")


def plot_with_matplotlib(
    rows: list[dict],
    output_path: Path,
    *,
    title: str,
    ylabel: str,
    series: list[tuple[str, str]],
    baselines: list[tuple[float, str]] | None = None,
) -> bool:
    baselines = baselines or []
    valid_series = [(key, label) for key, label in series if has_valid_series(rows, key)]
    if not valid_series and not baselines:
        return False

    tmp_path = safe_replace_png(output_path)
    try:
        plt.figure(figsize=(9, 5))
        for key, label in valid_series:
            xs, ys = series_points(rows, key)
            plt.plot(xs, ys, marker="o", label=label)
        for value, label in baselines:
            plt.axhline(value, color="gray", linestyle="--", label=label)
        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(tmp_path, dpi=150)
        os.replace(tmp_path, output_path)
        return True
    except Exception as error:
        print(f"warning: failed to save plot: {output_path} ({error})")
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    finally:
        plt.close()


def load_pillow_font(size: int):
    font_paths = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
    ]
    for font_path in font_paths:
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def plot_with_pillow(
    rows: list[dict],
    output_path: Path,
    *,
    title: str,
    ylabel: str,
    series: list[tuple[str, str]],
    baselines: list[tuple[float, str]] | None = None,
) -> bool:
    baselines = baselines or []
    valid_series = [(key, label) for key, label in series if has_valid_series(rows, key)]
    if not valid_series and not baselines:
        return False

    tmp_path = safe_replace_png(output_path)
    width, height = 1200, 720
    left, right, top, bottom = 95, 45, 75, 120
    plot_width = width - left - right
    plot_height = height - top - bottom
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

    try:
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        small_font = load_pillow_font(15)
        title_font = load_pillow_font(26)

        prepared = []
        all_x = []
        all_y = []
        for key, label in valid_series:
            xs, ys = series_points(rows, key)
            prepared.append((xs, ys, label))
            all_x.extend(xs)
            all_y.extend(ys)
        for value, _ in baselines:
            all_y.append(value)

        x_min = min(all_x) if all_x else 0.0
        x_max = max(all_x) if all_x else 1.0
        if x_min == x_max:
            x_max = x_min + 1.0

        y_min = min(all_y) if all_y else 0.0
        y_max = max(all_y) if all_y else 1.0
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0
        padding = (y_max - y_min) * 0.08
        y_min -= padding
        y_max += padding

        def to_xy(x_value: float, y_value: float) -> tuple[int, int]:
            x = left + int((x_value - x_min) / (x_max - x_min) * plot_width)
            y = top + plot_height - int((y_value - y_min) / (y_max - y_min) * plot_height)
            return x, y

        draw.text((left, 24), title, fill="black", font=title_font)
        draw.line((left, top, left, top + plot_height), fill="black", width=2)
        draw.line((left, top + plot_height, left + plot_width, top + plot_height), fill="black", width=2)
        draw.text((12, top + plot_height // 2), ylabel, fill="black", font=small_font)
        draw.text((left + plot_width // 2, height - 55), "Episode", fill="black", font=small_font)

        for tick in range(6):
            x = left + int(plot_width * tick / 5)
            y = top + int(plot_height * tick / 5)
            draw.line((x, top, x, top + plot_height), fill="#eeeeee")
            draw.line((left, y, left + plot_width, y), fill="#eeeeee")
            x_value = x_min + (x_max - x_min) * tick / 5
            y_value = y_max - (y_max - y_min) * tick / 5
            draw.text((x - 16, top + plot_height + 8), f"{x_value:.0f}", fill="#333333", font=small_font)
            draw.text((8, y - 5), f"{y_value:.1f}", fill="#333333", font=small_font)

        legend_x = left + 12
        legend_y = top + 10
        for index, (value, label) in enumerate(baselines):
            color = "#777777"
            _, y = to_xy(x_min, value)
            draw.line((left, y, left + plot_width, y), fill=color, width=2)
            draw.text((legend_x, legend_y + index * 22), label, fill=color, font=small_font)

        legend_offset = len(baselines)
        for index, (xs, ys, label) in enumerate(prepared):
            color = colors[index % len(colors)]
            points = [to_xy(x, y) for x, y in zip(xs, ys)]
            if len(points) == 1:
                x, y = points[0]
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            else:
                draw.line(points, fill=color, width=3)
                for x, y in points:
                    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            draw.text(
                (legend_x, legend_y + (legend_offset + index) * 22),
                label,
                fill=color,
                font=small_font,
            )

        image.save(tmp_path, "PNG")
        os.replace(tmp_path, output_path)
        return True
    except Exception as error:
        print(f"warning: failed to save plot: {output_path} ({error})")
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def save_plot(
    rows: list[dict],
    run_dir: Path,
    filename: str,
    *,
    title: str,
    ylabel: str,
    series: list[tuple[str, str]],
    baselines: list[tuple[float, str]] | None = None,
) -> Path | None:
    output_path = run_dir / filename
    saved = (
        plot_with_matplotlib(
            rows,
            output_path,
            title=title,
            ylabel=ylabel,
            series=series,
            baselines=baselines,
        )
        if HAS_MATPLOTLIB
        else plot_with_pillow(
            rows,
            output_path,
            title=title,
            ylabel=ylabel,
            series=series,
            baselines=baselines,
        )
    )
    return output_path if saved else None


def save_core_plots(rows: list[dict], run_dir: Path) -> list[Path]:
    generated = []
    specs = [
        (
            "learning_status.png",
            "Learning Status vs Random Baselines",
            "Return",
            [
                ("eval_adv_vs_random_prey_adversary_return", "Adversary vs random prey"),
                ("eval_random_adv_vs_prey_prey_return", "Prey vs random adversaries"),
            ],
            [
                (RANDOM_ADVERSARY_BASELINE, "random adversary baseline = 17.0"),
                (RANDOM_PREY_BASELINE, "random prey baseline = -181.9"),
            ],
        ),
        (
            "training_health.png",
            "Training Health",
            "Value",
            [
                ("mean_critic_loss", "critic loss"),
                ("mean_actor_loss", "actor loss"),
                ("mean_actor_entropy", "actor entropy"),
                ("epsilon", "epsilon"),
            ],
            [],
        ),
    ]
    for filename, title, ylabel, series, baselines in specs:
        path = save_plot(
            rows,
            run_dir,
            filename,
            title=title,
            ylabel=ylabel,
            series=series,
            baselines=baselines,
        )
        if path is not None:
            generated.append(path)

    if has_valid_series(rows, "eval_adv_vs_random_prey_touch_count"):
        path = save_plot(
            rows,
            run_dir,
            "touch_counts.png",
            title="Touch Count vs Random Baseline",
            ylabel="Touch Count",
            series=[
                ("eval_adv_vs_random_prey_touch_count", "Adversary vs random prey touch count"),
            ],
            baselines=[(RANDOM_TOUCH_BASELINE, "random touch baseline = 1.7")],
        )
        if path is not None:
            generated.append(path)

    return generated


def save_debug_plots(rows: list[dict], run_dir: Path) -> list[Path]:
    debug_specs = [
        (
            "train_returns.png",
            "Training Episode Returns",
            "Return",
            [
                ("adversary_team_return", "adversary team return"),
                ("prey_return", "prey return"),
            ],
            [],
        ),
        (
            "eval_self_play.png",
            "Self-play Evaluation",
            "Mean Eval Return",
            [
                ("eval_self_play_adversary_return", "self-play adversary"),
                ("eval_self_play_prey_return", "self-play prey"),
            ],
            [],
        ),
        (
            "eval_adversary_strength.png",
            "Adversary Strength Evaluation",
            "Mean Adversary Team Return",
            [
                ("eval_adv_vs_random_prey_adversary_return", "trained adversaries vs random prey"),
            ],
            [(RANDOM_ADVERSARY_BASELINE, "random adversary baseline = 17.0")],
        ),
        (
            "eval_prey_strength.png",
            "Prey Strength Evaluation",
            "Mean Prey Return",
            [
                ("eval_random_adv_vs_prey_prey_return", "trained prey vs random adversaries"),
            ],
            [(RANDOM_PREY_BASELINE, "random prey baseline = -181.9")],
        ),
        (
            "losses.png",
            "MADDPG Losses",
            "Loss",
            [
                ("mean_critic_loss", "mean critic loss"),
                ("mean_actor_loss", "mean actor loss"),
            ],
            [],
        ),
        (
            "epsilon.png",
            "Epsilon Schedule",
            "Epsilon",
            [("epsilon", "epsilon")],
            [],
        ),
        (
            "actor_entropy.png",
            "Actor Entropy",
            "Entropy",
            [("mean_actor_entropy", "mean actor entropy")],
            [],
        ),
    ]

    generated = []
    for filename, title, ylabel, series, baselines in debug_specs:
        path = save_plot(
            rows,
            run_dir,
            filename,
            title=title,
            ylabel=ylabel,
            series=series,
            baselines=baselines,
        )
        if path is not None:
            generated.append(path)
    return generated


def save_all_plots_from_rows(
    rows: list[dict],
    run_dir: Path,
    save_debug_plots_enabled: bool = False,
) -> list[Path]:
    run_dir = Path(run_dir)
    generated = save_core_plots(rows, run_dir)
    if save_debug_plots_enabled:
        generated.extend(save_debug_plots(rows, run_dir))
    return generated


def build_summary_text(rows: list[dict], run_dir: Path) -> tuple[str, dict[str, str]]:
    final_episode = latest_valid(rows, "episode")
    final_steps = latest_valid(rows, "total_steps")
    adv_return = latest_valid(rows, "eval_adv_vs_random_prey_adversary_return")
    prey_return = latest_valid(rows, "eval_random_adv_vs_prey_prey_return")
    touch_count = latest_valid(rows, "eval_adv_vs_random_prey_touch_count")
    critic_loss = latest_valid(rows, "mean_critic_loss")
    actor_loss = latest_valid(rows, "mean_actor_loss")
    actor_entropy = latest_valid(rows, "mean_actor_entropy")

    adv_status = better_text(adv_return, RANDOM_ADVERSARY_BASELINE)
    prey_status = better_text(prey_return, RANDOM_PREY_BASELINE)
    touch_status = better_text(touch_count, RANDOM_TOUCH_BASELINE)
    health = training_health(rows)
    collapse = policy_collapse_status(rows)

    lines = [
        f"run directory: {run_dir}",
        f"total episodes: {format_value(final_episode, digits=0)}",
        f"total steps: {format_value(final_steps, digits=0)}",
        f"final eval_adv_vs_random_prey_adversary_return: {format_value(adv_return)}",
        f"random adversary baseline = {RANDOM_ADVERSARY_BASELINE}",
        f"adversary better than random: {adv_status}",
        f"final eval_random_adv_vs_prey_prey_return: {format_value(prey_return)}",
        f"random prey baseline = {RANDOM_PREY_BASELINE}",
        f"prey better than random: {prey_status}",
        f"final eval_adv_vs_random_prey_touch_count: {format_value(touch_count)}",
        f"random touch baseline = {RANDOM_TOUCH_BASELINE}",
        f"final mean_critic_loss: {format_value(critic_loss)}",
        f"final mean_actor_loss: {format_value(actor_loss)}",
        f"final mean_actor_entropy: {format_value(actor_entropy)}",
        f"policy collapse risk: {collapse}",
        f"training health: {health}",
    ]
    note = actor_entropy_note(rows)
    if note:
        lines.append(note)

    summary = {
        "adv_status": adv_status,
        "prey_status": prey_status,
        "touch_status": touch_status,
        "health": health,
        "collapse": collapse,
        "adv_return": format_value(adv_return),
        "prey_return": format_value(prey_return),
        "touch_count": format_value(touch_count),
    }
    return "\n".join(lines) + "\n", summary


def format_value(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "not available"
    if digits == 0:
        return str(int(value))
    return f"{value:.{digits}f}"


def write_summary(rows: list[dict], run_dir: Path) -> dict[str, str]:
    text, summary = build_summary_text(rows, run_dir)
    summary_path = Path(run_dir) / "summary.txt"
    tmp_path = summary_path.with_name("summary.tmp.txt")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, summary_path)
    except OSError as error:
        print(f"warning: failed to save summary: {summary_path} ({error})")
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
    return summary


def save_all_outputs_from_rows(
    rows: list[dict],
    run_dir: Path,
    save_debug_plots_enabled: bool = False,
) -> tuple[list[Path], dict[str, str]]:
    generated = save_all_plots_from_rows(
        rows,
        run_dir,
        save_debug_plots_enabled=save_debug_plots_enabled,
    )
    summary = write_summary(rows, run_dir)
    return generated, summary


def save_all_outputs_from_csv(
    run_dir: Path,
    save_debug_plots_enabled: bool = False,
) -> tuple[list[Path], dict[str, str]]:
    run_dir = Path(run_dir)
    csv_path = run_dir / "train_log.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"train_log.csv not found: {csv_path}")
    rows = read_train_log(csv_path)
    return save_all_outputs_from_rows(
        rows,
        run_dir,
        save_debug_plots_enabled=save_debug_plots_enabled,
    )


def main() -> None:
    args = parse_args()
    generated, _ = save_all_outputs_from_csv(
        args.run_dir,
        save_debug_plots_enabled=args.save_debug_plots,
    )
    print("Generated outputs:")
    for path in generated:
        print(path)
    print(args.run_dir / "summary.txt")


if __name__ == "__main__":
    main()
