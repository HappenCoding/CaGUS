"""Project differential wall-video observations into a hidden 2-D region.

The script uses the analytical kernel K_G = L_G V_G rho (Appendix A) and
reports the strongest hidden-space response and peak-to-sidelobe ratio (PSR)
over time.  It deliberately does not claim to reconstruct hidden appearance.
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

from transport_kernel import Segment, adjoint_project, build_analytical_kernel, grid_points, peak_to_sidelobe_ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run geometry-conditioned adjoint projection on a wall video.")
    parser.add_argument("video", type=Path)
    parser.add_argument("geometry", type=Path, help="JSON geometry file; see example_transport_geometry.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reference-seconds", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    return parser.parse_args()


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("wall_bounds_xz", "hidden_bounds_xz", "wall_grid", "hidden_grid"):
        if key not in config:
            raise ValueError(f"Geometry file is missing '{key}'")
    return config


def read_frames(path: Path, wall_grid: tuple[int, int], stride: int, max_seconds: float) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(max_seconds * fps) if max_seconds > 0 else None
    rows, cols = wall_grid
    frames: list[np.ndarray] = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames is not None and index >= max_frames):
            break
        if index % max(1, stride) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            frames.append(cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA))
        index += 1
    cap.release()
    if len(frames) < 5:
        raise RuntimeError("Too few frames after sampling")
    return np.stack(frames), float(fps / max(1, stride))


def main() -> None:
    args = parse_args()
    geometry = read_config(args.geometry)
    wall_rows, wall_cols = map(int, geometry["wall_grid"])
    hidden_rows, hidden_cols = map(int, geometry["hidden_grid"])
    frames, fps = read_frames(args.video, (wall_rows, wall_cols), args.stride, args.max_seconds)
    ref_count = min(frames.shape[0], max(5, int(round(args.reference_seconds * fps))))
    reference = np.median(frames[:ref_count], axis=0)
    perturbation = (frames - reference[None, :, :]).reshape(frames.shape[0], -1)

    occluders = [Segment(tuple(item["start"]), tuple(item["end"])) for item in geometry.get("occluders", [])]
    wall_points = grid_points(tuple(geometry["wall_bounds_xz"]), wall_rows, wall_cols)
    hidden_points = grid_points(tuple(geometry["hidden_bounds_xz"]), hidden_rows, hidden_cols)
    kernel = build_analytical_kernel(
        wall_points,
        hidden_points,
        occluders=occluders,
        wall_reflectance=float(geometry.get("wall_reflectance", 1.0)),
        attenuation_power=float(geometry.get("attenuation_power", 2.0)),
    )
    response = adjoint_project(kernel, perturbation, normalize_columns=True)
    response_maps = response.reshape(response.shape[0], hidden_rows, hidden_cols)
    peaks = np.empty((response.shape[0], 3), dtype=np.float32)
    for index, frame_response in enumerate(response):
        peak, psr = peak_to_sidelobe_ratio(np.abs(frame_response), exclusion_radius=max(1, hidden_cols // 16))
        peaks[index] = (peak % hidden_cols, peak // hidden_cols, psr)

    output = args.output or args.video.with_suffix("").with_name(args.video.stem + "_geometry_projection")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "projection_results.npz", response=response_maps, peaks=peaks, fps=fps, reference_frames=ref_count)
    time = np.arange(response.shape[0]) / fps
    activity = np.percentile(np.abs(response), 95, axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(reference, cmap="gray")
    axes[0].set_title("reference wall frame")
    axes[1].plot(time, activity, linewidth=1.0)
    axes[1].axvspan(0, ref_count / fps, color="0.85")
    axes[1].set(xlabel="time (s)", ylabel="95th percentile response", title="hidden-space activity")
    axes[2].scatter(peaks[:, 0], peaks[:, 1], c=time, s=8, cmap="viridis")
    axes[2].invert_yaxis()
    axes[2].set(xlabel="hidden grid x", ylabel="hidden grid z", title="peak trajectory")
    fig.tight_layout()
    fig.savefig(output / "projection_overview.png", dpi=180)
    plt.close(fig)
    summary = {
        "video": str(args.video),
        "geometry": str(args.geometry),
        "wall_grid": [wall_rows, wall_cols],
        "hidden_grid": [hidden_rows, hidden_cols],
        "frames": int(response.shape[0]),
        "reference_frames": int(ref_count),
        "median_psr": float(np.median(peaks[:, 2])),
        "max_psr": float(np.max(peaks[:, 2])),
    }
    (output / "projection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Projected {response.shape[0]} frames. Results: {output}")


if __name__ == "__main__":
    main()
