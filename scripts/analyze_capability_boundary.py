"""Summarize the capability boundary from condition-level detection outputs.

It aggregates the released ``condition_summary.json`` files into a compact CSV
and factor-level markdown table.  The analysis uses the primary raw_abs method
unless a condition explicitly records a different selected method.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate CaGUS capability-boundary metrics.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.rglob("condition_summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        method = data.get("selected_method", "raw_abs")
        metrics = data.get("methods", {}).get(method) or data.get("methods", {}).get("raw_abs", {})
        if not metrics:
            continue
        relative = path.relative_to(root)
        factor = relative.parts[0] if relative.parts else "unknown"
        rows.append({
            "factor": factor,
            "condition": str(data.get("condition", relative.parent.parent.as_posix())),
            "method": method,
            "videos": int(metrics.get("total", data.get("num_videos", 0))),
            "detected": int(metrics.get("detected", 0)),
            "detection_rate": float(metrics.get("detection_rate", 0.0)),
            "mean_snr_db": float(metrics.get("mean_snr_db", np.nan)),
            "mean_active_signal": float(metrics.get("mean_active", np.nan)),
            "mean_reference_signal": float(metrics.get("mean_reference", np.nan)),
        })
    return rows


def write_outputs(rows: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["factor", "condition", "method", "videos", "detected", "detection_rate", "mean_snr_db", "mean_active_signal", "mean_reference_signal"]
    with (output / "capability_boundary_conditions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["factor"]].append(row)
    lines = ["# Capability Boundary Summary", "", "| factor | conditions | detection rate | mean SNR (dB) |", "|---|---:|---:|---:|"]
    for factor, entries in sorted(grouped.items()):
        total = sum(item["videos"] for item in entries)
        detected = sum(item["detected"] for item in entries)
        snr_values = [item["mean_snr_db"] for item in entries if np.isfinite(item["mean_snr_db"])]
        mean_snr = float(np.mean(snr_values)) if snr_values else float("nan")
        lines.append(f"| {factor} | {len(entries)} | {detected}/{total} ({100.0 * detected / max(total, 1):.1f}%) | {mean_snr:.2f} |")
    (output / "capability_boundary_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.root)
    if not rows:
        raise RuntimeError(f"No condition_summary.json files found under {args.root}")
    output = args.output or args.root / "capability_boundary_analysis"
    write_outputs(rows, output)
    print(f"Wrote {len(rows)} condition rows to {output}")


if __name__ == "__main__":
    main()
