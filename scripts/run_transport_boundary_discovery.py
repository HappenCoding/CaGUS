"""Discover image boundaries that carry temporal transport perturbations.

This implements the perturbation-driven boundary validation in Appendix B.
It operates on a commodity-camera video and does not require a known
occluder geometry.  Results are intended as candidate sensing structures for
subsequent calibrated or analytical transport projection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover active transport boundaries from a wall video.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reference-seconds", type=float, default=5.0)
    parser.add_argument("--resize", type=int, default=192)
    parser.add_argument("--min-length", type=int, default=36)
    parser.add_argument("--patch-width", type=int, default=7)
    parser.add_argument("--max-boundaries", type=int, default=12)
    parser.add_argument("--energy-z", type=float, default=1.5)
    return parser.parse_args()


def read_video(video: Path, resize: int) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        scale = float(resize) / max(height, width)
        frame = cv2.resize(frame, (max(2, round(width * scale)), max(2, round(height * scale))), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0)
    cap.release()
    if len(frames) < 10:
        raise RuntimeError("Video contains too few readable frames")
    return np.stack(frames), float(fps)


def line_patch_masks(shape: tuple[int, int], line: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = line.astype(np.float32)
    tangent = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    tangent /= max(float(np.linalg.norm(tangent)), 1e-6)
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    centre = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
    length = float(np.linalg.norm(np.array([x2 - x1, y2 - y1])))
    size = (max(2, int(round(length))), max(2, int(width)))

    def mask_for(sign: float) -> np.ndarray:
        c = centre + sign * (width / 2.0 + 1.0) * normal
        box = cv2.boxPoints(((float(c[0]), float(c[1])), size, float(np.degrees(np.arctan2(tangent[1], tangent[0])))))
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(box).astype(np.int32), 1)
        return mask.astype(bool)

    return mask_for(-1.0), mask_for(1.0)


def robust_z(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - median))
    return (values - median) / max(float(scale), 1e-8)


def main() -> None:
    args = parse_args()
    output = args.output or args.video.with_suffix("").with_name(args.video.stem + "_transport_boundaries")
    output.mkdir(parents=True, exist_ok=True)
    frames, fps = read_video(args.video, args.resize)
    ref_count = min(frames.shape[0], max(5, int(round(args.reference_seconds * fps))))
    reference = np.median(frames[:ref_count], axis=0)
    edges = cv2.Canny(np.uint8(np.clip(reference * 255.0, 0, 255)), 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=22, minLineLength=args.min_length, maxLineGap=8)
    if lines is None:
        raise RuntimeError("No candidate line segments were found; reduce --min-length or inspect the reference image")

    candidates = []
    for line in lines[:, 0, :]:
        mask_a, mask_b = line_patch_masks(reference.shape, line, args.patch_width)
        if mask_a.sum() < 10 or mask_b.sum() < 10:
            continue
        signal = frames[:, mask_a].mean(axis=1) - frames[:, mask_b].mean(axis=1)
        energy = float(np.var(signal - np.median(signal[:ref_count])))
        candidates.append((line, signal, energy))
    if not candidates:
        raise RuntimeError("No valid observation patches could be constructed")

    energies = np.array([item[2] for item in candidates], dtype=np.float32)
    selected = [item for item, score in zip(candidates, robust_z(energies)) if score >= args.energy_z]
    if not selected:
        selected = sorted(candidates, key=lambda item: item[2], reverse=True)[: max(1, min(args.max_boundaries, len(candidates)))]
    selected = sorted(selected, key=lambda item: item[2], reverse=True)[: args.max_boundaries]

    signals = np.vstack([item[1] for item in selected]).T.astype(np.float32)
    np.savez_compressed(output / "boundary_signals.npz", signals=signals, fps=fps, reference_frames=ref_count)
    overlay = cv2.cvtColor(np.uint8(np.clip(reference * 255.0, 0, 255)), cv2.COLOR_GRAY2BGR)
    metadata = []
    for index, (line, _, energy) in enumerate(selected):
        x1, y1, x2, y2 = map(int, line)
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, str(index + 1), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
        metadata.append({"index": index, "line_xyxy": [x1, y1, x2, y2], "perturbation_energy": energy})
    cv2.imwrite(str(output / "active_boundaries.png"), overlay)
    (output / "active_boundaries.json").write_text(json.dumps({"video": str(args.video), "fps": fps, "reference_frames": ref_count, "boundaries": metadata}, indent=2), encoding="utf-8")

    time = np.arange(signals.shape[0]) / fps
    fig, ax = plt.subplots(figsize=(10, 4))
    for index in range(signals.shape[1]):
        ax.plot(time, signals[:, index], linewidth=0.9, label=f"boundary {index + 1}")
    ax.axvspan(0, ref_count / fps, color="0.85", alpha=0.7, label="reference")
    ax.set(xlabel="time (s)", ylabel="two-sided boundary response", title="Candidate transport-boundary responses")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "boundary_response_curves.png", dpi=180)
    plt.close(fig)
    print(f"Selected {len(selected)} active boundaries. Results: {output}")


if __name__ == "__main__":
    main()
