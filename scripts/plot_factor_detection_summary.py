"""Create publication-quality detection-rate and SNR figures for Mobicom factors.

The script reads the existing detection results only.  It never reruns video
processing or modifies the detection metrics.  Each figure is written as a
vector PDF in its corresponding factor directory.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data"

# Okabe-Ito colours: colour-blind safe and clear in print as well as on screen.
BLUE = "#0072B2"
ORANGE = "#D55E00"
INK = "#202124"
MUTED = "#8A8F98"


@dataclass(frozen=True)
class FactorSpec:
    display_name: str
    x_label: str
    mode: str  # "numeric", "category", or "recordings"
    unit: str = ""
    labels: dict[str, str] | None = None


FACTOR_SPECS: dict[str, FactorSpec] = {
    "distance": FactorSpec("Distance", "Distance (mm)", "numeric", "mm"),
    "brightness": FactorSpec("Illuminance", "Illuminance (lux)", "numeric", "lux"),
    "brightness_second_run": FactorSpec("Illuminance", "Illuminance (lux)", "numeric", "lux"),
    "material": FactorSpec(
        "Material",
        "Surface material",
        "category",
        labels={
            "wall": "Wall",
            "wood_board": "Wood board",
            "foam_board": "Foam board",
            "cardboard": "Cardboard",
            "black_cloth": "Black fabric",
        },
    ),
    # The raw videos do not encode their physical angle.  This keeps the output
    # honest by labelling their acquisition order rather than inventing degrees.
    "angle": FactorSpec("Viewing Angle", "Recording order", "recordings"),
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.unicode_minus": False,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot detection rate and SNR versus each experimental factor."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--factor",
        action="append",
        default=[],
        help="Factor directory to plot. May be supplied more than once.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def condition_video_rows(factor_dir: Path, condition: str) -> list[dict[str, str]]:
    path = factor_dir / condition / "detection_results" / "video_metrics.csv"
    return read_csv(path) if path.exists() else []


def direct_video_rows(factor_dir: Path) -> list[dict[str, str]]:
    path = factor_dir / "detection_results" / "video_metrics.csv"
    return read_csv(path) if path.exists() else []


def float_values(rows: Iterable[dict[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def bool_values(rows: Iterable[dict[str, str]], field: str) -> list[bool]:
    truthy = {"true", "1", "yes"}
    return [str(row.get(field, "")).strip().lower() in truthy for row in rows]


def bootstrap_mean_interval(values: list[float], seed: int) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (values[0], values[0])
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(6000, len(data)), replace=True).mean(axis=1)
    return tuple(np.percentile(samples, [2.5, 97.5]).tolist())


def condition_numeric_value(text: str) -> float:
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        raise ValueError(f"Cannot extract numeric value from condition name: {text}")
    return float(match.group())


def ordered_summary_rows(factor_dir: Path, spec: FactorSpec) -> list[dict[str, str]]:
    summary_path = factor_dir / "summary_detection.csv"
    rows = read_csv(summary_path) if summary_path.exists() else []
    if spec.mode == "numeric":
        return sorted(rows, key=lambda row: condition_numeric_value(row["condition"]))
    return rows


def collect_factor_data(factor_dir: Path, spec: FactorSpec) -> list[dict[str, object]]:
    if spec.mode == "recordings":
        rows = direct_video_rows(factor_dir)
        output = []
        for index, row in enumerate(rows, start=1):
            values = float_values([row], "raw_abs_snr_mean_db")
            detected = bool_values([row], "raw_abs_detected")
            output.append(
                {
                    "condition": row.get("video", f"Recording {index}"),
                    "x": float(index),
                    "label": f"Trial {index}",
                    "snr_values": values,
                    "detection_values": detected,
                    "snr_mean": values[0] if values else float("nan"),
                    "snr_ci": (values[0], values[0]) if values else (float("nan"), float("nan")),
                    "rate": float(detected[0]) if detected else float("nan"),
                }
            )
        return output

    output = []
    for index, summary in enumerate(ordered_summary_rows(factor_dir, spec), start=1):
        condition = summary["condition"]
        rows = condition_video_rows(factor_dir, condition)
        snr_values = float_values(rows, "raw_abs_snr_mean_db")
        detection_values = bool_values(rows, "raw_abs_detected")
        if not rows:
            raise FileNotFoundError(
                f"Missing video_metrics.csv for {factor_dir.name}/{condition}."
            )
        x = condition_numeric_value(condition) if spec.mode == "numeric" else float(index)
        label = spec.labels.get(condition, condition) if spec.labels else condition
        output.append(
            {
                "condition": condition,
                "x": x,
                "label": label,
                "snr_values": snr_values,
                "detection_values": detection_values,
                "snr_mean": mean(snr_values),
                "snr_ci": bootstrap_mean_interval(snr_values, 20260819 + index),
                "rate": sum(detection_values) / len(detection_values),
            }
        )
    return output


def prepare_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#6D7278")
        ax.spines[spine].set_linewidth(0.75)
    ax.tick_params(axis="both", colors="#43484E", width=0.75, length=3, pad=3)
    ax.grid(False)


def numeric_tick_positions(x: np.ndarray) -> np.ndarray:
    if len(x) <= 7:
        return x
    step = 100.0 if (x.max() - x.min()) >= 400 else 50.0
    start = math.ceil(x.min() / step) * step
    return np.arange(start, x.max() + step * 0.3, step)


def apply_x_axis(ax: plt.Axes, data: list[dict[str, object]], spec: FactorSpec) -> None:
    x = np.asarray([item["x"] for item in data], dtype=float)
    if spec.mode == "numeric":
        ax.set_xticks(numeric_tick_positions(x))
        margin = max((x.max() - x.min()) * 0.04, 1.0)
        ax.set_xlim(x.min() - margin, x.max() + margin)
    else:
        ax.set_xticks(x)
        labels = [str(item["label"]) for item in data]
        if spec.mode == "category":
            labels = [label.replace(" ", "\n", 1) if " " in label else label for label in labels]
        ax.set_xticklabels(labels)
        ax.set_xlim(x.min() - 0.45, x.max() + 0.45)


def plot_detection_rate(ax: plt.Axes, data: list[dict[str, object]], spec: FactorSpec) -> None:
    x = np.asarray([item["x"] for item in data], dtype=float)
    rate = np.asarray([item["rate"] for item in data], dtype=float)
    ax.plot(x, rate, color=BLUE, linewidth=2.0, zorder=2)
    ax.scatter(x, rate, s=32, facecolor="white", edgecolor=BLUE, linewidth=1.45, zorder=3)

    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Detection rate")
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    prepare_axes(ax)
    apply_x_axis(ax, data, spec)
    ax.set_xlabel(spec.x_label)
    ax.legend(
        handles=[
            Line2D(
                [0], [0], color=BLUE, marker="o", markersize=4.8,
                markerfacecolor="white", markeredgewidth=1.2, label="Detection rate"
            )
        ],
        loc="lower left",
        bbox_to_anchor=(0.012, 0.018),
        frameon=False,
        handlelength=1.8,
        handletextpad=0.45,
        borderaxespad=0.0,
    )


def plot_snr(ax: plt.Axes, data: list[dict[str, object]], spec: FactorSpec) -> None:
    x = np.asarray([item["x"] for item in data], dtype=float)
    snr_mean = np.asarray([item["snr_mean"] for item in data], dtype=float)
    ci = np.asarray([item["snr_ci"] for item in data], dtype=float)

    values_all = np.concatenate([np.asarray(item["snr_values"], dtype=float) for item in data])

    jitter_scale = (x.max() - x.min()) * 0.009 if spec.mode == "numeric" and len(x) > 1 else 0.06
    rng = np.random.default_rng(20260820)
    for item in data:
        values = np.asarray(item["snr_values"], dtype=float)
        if not len(values):
            continue
        jitter = rng.normal(0.0, jitter_scale, size=len(values))
        ax.scatter(
            float(item["x"]) + jitter,
            values,
            s=13,
            color=MUTED,
            alpha=0.44,
            linewidth=0,
            zorder=1,
        )

    lower = snr_mean - ci[:, 0]
    upper = ci[:, 1] - snr_mean
    ax.errorbar(
        x,
        snr_mean,
        yerr=np.vstack((lower, upper)),
        fmt="none",
        ecolor=ORANGE,
        elinewidth=1.15,
        capsize=2.7,
        capthick=1.15,
        zorder=3,
    )
    ax.plot(x, snr_mean, color=ORANGE, linewidth=2.0, zorder=4)
    ax.scatter(x, snr_mean, s=31, facecolor="white", edgecolor=ORANGE, linewidth=1.45, zorder=5)

    lower_y = float(np.nanmin(values_all))
    upper_y = float(np.nanmax(ci[:, 1]))
    span = max(upper_y - lower_y, 1.0)
    ax.set_ylim(lower_y - 0.12 * span, upper_y + 0.19 * span)
    ax.set_ylabel("SNR (dB)")
    prepare_axes(ax)
    apply_x_axis(ax, data, spec)
    ax.set_xlabel(spec.x_label)

    for index, (x_value, value, upper_ci) in enumerate(zip(x, snr_mean, ci[:, 1])):
        ax.annotate(
            f"{value:.1f}",
            xy=(x_value, upper_ci),
            xytext=(0, 5 + 2 * (index % 2)),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.1,
            fontweight="bold",
            color=ORANGE,
            zorder=6,
        )
    ax.legend(
        handles=[
            Line2D(
                [0], [0], color=MUTED, marker="o", linestyle="None", markersize=4.8,
                alpha=0.7, label="Individual recording"
            ),
            Line2D(
                [0], [0], color=ORANGE, marker="o", markersize=4.8,
                markerfacecolor="white", markeredgewidth=1.2,
                label="Mean SNR +/- 95% CI"
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0.012, 0.018),
        frameon=False,
        handlelength=1.8,
        handletextpad=0.45,
        borderaxespad=0.0,
    )


def save_detection_rate_figure(factor_dir: Path, data: list[dict[str, object]], spec: FactorSpec) -> Path:
    configure_style()
    fig, ax = plt.subplots(figsize=(4.25, 2.58))
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.25, top=0.965)
    plot_detection_rate(ax, data, spec)
    output = factor_dir / "detection_rate.pdf"
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    return output


def save_snr_figure(factor_dir: Path, data: list[dict[str, object]], spec: FactorSpec) -> Path:
    configure_style()
    fig, ax = plt.subplots(figsize=(4.25, 2.58))
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.25, top=0.965)
    plot_snr(ax, data, spec)
    output = factor_dir / "snr_db.pdf"
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    return output


def plot_factor(factor_dir: Path, spec: FactorSpec) -> tuple[Path, Path] | None:
    data = collect_factor_data(factor_dir, spec)
    if not data:
        return None
    return (
        save_detection_rate_figure(factor_dir, data, spec),
        save_snr_figure(factor_dir, data, spec),
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    requested = set(args.factor)
    targets = [
        (name, spec)
        for name, spec in FACTOR_SPECS.items()
        if not requested or name in requested
    ]
    outputs = []
    for factor_name, spec in targets:
        factor_dir = root / factor_name
        if not factor_dir.exists():
            print(f"[skip] Missing factor directory: {factor_dir}")
            continue
        try:
            output = plot_factor(factor_dir, spec)
        except FileNotFoundError as exc:
            print(f"[skip] {exc}")
            continue
        if output:
            outputs.extend(output)
            for path in output:
                print(f"[saved] {path}")
        else:
            print(f"[skip] No usable detection records for {factor_name}")
    print(f"Finished: {len(outputs)} figure(s).")


if __name__ == "__main__":
    main()
