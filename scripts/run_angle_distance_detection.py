import importlib.util
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm


ROOT = Path(__file__).resolve().parents[1] / "data"
BASE_SCRIPT = Path(__file__).resolve().parent / "run_mobicom_detection.py"
FACTOR_NAME = "distance_angle"
METHOD = "raw_abs"
GROUP_SPECS = [
    {"name": "angle_0_180", "start": 0.0, "end": 180.0, "reverse_time": False},
    # This experiment was captured as 0 to -180 degrees, i.e. 360 -> 180 on a 0-360 circle.
    {"name": "angle_180_360", "start": 180.0, "end": 360.0, "reverse_time": True},
]
ANGLE_GRID_HALF = np.linspace(0.0, 180.0, 181, dtype=np.float32)
RAW_DISPLAY_GAMMA = 0.45
RAW_DISPLAY_VMAX_PERCENTILE = 99.5
SEAM_MATCH_WINDOW_DEGREES = 8
DENSE_RADIUS_SAMPLES = 240
RADIAL_SMOOTHING_SIGMA_SAMPLES = 2.0
SIGNAL_EPS = 1e-8
DISTANCE_LABELS_METERS = {1: "2m", 2: "4m", 3: "6m", 4: "8m", 5: "10m"}


def load_base_module():
    spec = importlib.util.spec_from_file_location("mobicom_detection", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def natural_distance_key(path):
    digits = "".join(ch for ch in path.name if ch.isdigit())
    return int(digits) if digits else path.name


def active_angle_curve(time, values, cfg, start_angle, end_angle, reverse_time=False):
    active = time >= cfg.active_start_seconds
    if int(active.sum()) < 2:
        return None, None
    active_time = time[active]
    active_values = values[active]
    denom = max(float(active_time[-1] - cfg.active_start_seconds), 1e-6)
    progress = (active_time - cfg.active_start_seconds) / denom
    if reverse_time:
        angles = end_angle - progress * (end_angle - start_angle)
    else:
        angles = start_angle + progress * (end_angle - start_angle)
    angles = np.clip(angles, start_angle, end_angle)
    return angles, active_values


def interpolate_to_half_grid(time, values, cfg, start_angle, end_angle, reverse_time=False):
    angles, active_values = active_angle_curve(time, values, cfg, start_angle, end_angle, reverse_time)
    if angles is None:
        return np.full_like(ANGLE_GRID_HALF, np.nan, dtype=np.float32)
    target_angles = np.linspace(start_angle, end_angle, len(ANGLE_GRID_HALF), dtype=np.float32)
    order = np.argsort(angles)
    return np.interp(target_angles, angles[order], active_values[order]).astype(np.float32)


def plot_angle_condition(
    png_path,
    condition_title,
    group_label,
    start_angle,
    end_angle,
    videos,
    curves,
    per_video,
    cfg,
    reverse_time=False,
):
    n = len(videos)
    cols = 2
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, max(5, 2.4 * rows)), squeeze=False)
    axes_flat = axes.ravel()
    by_name = {row["video"]: row for row in per_video}
    for ax, video in zip(axes_flat, videos):
        name = video.name
        data = curves[name][METHOD]
        angles, active_values = active_angle_curve(
            data["time"], data["value"], cfg, start_angle, end_angle, reverse_time
        )
        row = by_name[name]
        detected = bool(row[f"{METHOD}_detected"])
        snr = float(row[f"{METHOD}_snr_mean_db"])
        if angles is not None:
            ax.plot(angles, active_values, color="#1f77b4", linewidth=1.0)
        ax.set_xlim(start_angle, end_angle)
        ax.set_xticks(np.arange(start_angle, end_angle + 1e-6, 30))
        ax.set_title(f"{name} | {'detected' if detected else 'miss'} | SNR={snr:.1f} dB", fontsize=9)
        direction = "reversed" if reverse_time else "forward"
        ax.set_xlabel(f"angle after 5s ({group_label}, {direction})")
        ax.set_ylabel(METHOD)
        ax.grid(True, alpha=0.25)
    for ax in axes_flat[n:]:
        ax.axis("off")
    direction_title = "5s+ mapped backward" if reverse_time else "5s+ mapped forward"
    fig.suptitle(f"{condition_title} | {direction_title} to {group_label}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def stack_profiles(distance_profiles, row_normalize=False, drop_last=True):
    distances = sorted(distance_profiles)
    values = np.vstack([distance_profiles[d] for d in distances]).astype(np.float32)
    if drop_last:
        values = values[:, :-1]
    if row_normalize:
        row_min = np.nanmin(values, axis=1, keepdims=True)
        row_max = np.nanmax(values, axis=1, keepdims=True)
        values = (values - row_min) / np.maximum(row_max - row_min, 1e-8)
    return distances, values


def raw_display_norm(values):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    vmax = float(np.nanpercentile(finite, RAW_DISPLAY_VMAX_PERCENTILE))
    if vmax <= 0:
        vmax = float(np.nanmax(finite))
    if vmax <= 0:
        return None
    return PowerNorm(gamma=RAW_DISPLAY_GAMMA, vmin=0.0, vmax=vmax)


def plot_polar_heatmap(png_path, distance_profiles, title, start_angle, end_angle, row_normalize=False):
    distances, values = stack_profiles(distance_profiles, row_normalize=row_normalize, drop_last=True)
    if not distances:
        return

    theta_edges = np.deg2rad(np.linspace(start_angle, end_angle, values.shape[1] + 1))
    radius_edges = np.arange(0.5, len(distances) + 1.5, 1.0)
    theta_mesh, radius_mesh = np.meshgrid(theta_edges, radius_edges)

    fig = plt.figure(figsize=(9, 6 if end_angle - start_angle >= 300 else 5.5))
    ax = fig.add_subplot(111, projection="polar")
    norm = None if row_normalize else raw_display_norm(values)
    mesh = ax.pcolormesh(theta_mesh, radius_mesh, values, shading="auto", cmap="magma", norm=norm)
    ax.set_thetamin(start_angle)
    ax.set_thetamax(end_angle)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlim(0.5, len(distances) + 0.5)
    ax.set_rticks(range(1, len(distances) + 1))
    ax.set_yticklabels([f"D{d}" for d in distances])
    ax.set_rlabel_position(105 if end_angle <= 180 else 255 if start_angle >= 180 else 45)
    ax.set_thetagrids(np.arange(start_angle, end_angle + 1e-6, 30))
    ax.set_title(title, va="bottom")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.10, shrink=0.82)
    cbar.set_label(
        "row-normalized signal"
        if row_normalize
        else f"mean raw_abs signal (gamma={RAW_DISPLAY_GAMMA:g} display)"
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def write_group_summary(group_dir, group_label, condition_summaries, angle_plot_paths, heatmap_path, heatmap_norm_path):
    lines = [
        f"# {group_label} angle-distance detection summary",
        "",
        "## Condition Table",
        "",
        "| distance | raw detections | raw rate | raw SNR dB | selected | selected rate |",
        "|---|---:|---:|---:|---|---:|",
    ]
    total_detected = 0
    total_videos = 0
    for summary in condition_summaries:
        raw = summary["methods"][METHOD]
        selected = summary["methods"][summary["selected_method"]]
        total_detected += int(raw["detected"])
        total_videos += int(raw["total"])
        distance_label = summary["condition"].split("/")[-1]
        lines.append(
            f"| {distance_label} | {raw['detected']}/{raw['total']} | "
            f"{raw['detection_rate'] * 100:.1f}% | {raw['mean_snr_db']:.2f} | "
            f"{summary['selected_method']} | {selected['detection_rate'] * 100:.1f}% |"
        )

    lines.extend(
        [
            "",
            f"Total raw detection rate: {total_detected}/{total_videos} ({total_detected / max(1, total_videos) * 100:.1f}%).",
            "",
            "## Semicircle Heatmap",
            "",
            f"![]({heatmap_path.name})",
            "",
            f"![]({heatmap_norm_path.name})",
            "",
            "## Angle Curves",
            "",
        ]
    )
    for path in angle_plot_paths:
        lines.extend([f"### {path.stem}", "", f"![]({path.relative_to(group_dir).as_posix()})", ""])
    (group_dir / "angle_distance_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_angle_group(det, cfg, factor_dir, spec):
    group_label = spec["name"]
    group_dir = factor_dir / group_label
    if not group_dir.exists():
        print(f"[skip] Missing angle group: {group_dir}")
        return None

    condition_summaries = []
    angle_plot_paths = []
    distance_profiles = {}
    start_angle = float(spec["start"])
    end_angle = float(spec["end"])
    reverse_time = bool(spec.get("reverse_time", False))

    distance_dirs = sorted(
        [p for p in group_dir.iterdir() if p.is_dir() and det.find_videos(p)],
        key=natural_distance_key,
    )
    for distance_dir in distance_dirs:
        videos = det.find_videos(distance_dir)
        distance_num = natural_distance_key(distance_dir)
        condition_name = f"{group_label}/{distance_dir.name}"
        out_dir = distance_dir / "detection_results"
        out_dir.mkdir(exist_ok=True)
        print(f"[condition] {FACTOR_NAME}/{condition_name}: {len(videos)} videos")

        per_video = []
        curves = {}
        angle_profiles = []
        for idx, video_path in enumerate(videos, start=1):
            print(f"  [{idx:02d}/{len(videos):02d}] {video_path.name}")
            result = det.process_video(video_path, cfg)
            per_video.append(result["summary"])
            curves[video_path.name] = result["curves"]
            curve = result["curves"][METHOD]
            angle_profiles.append(
                interpolate_to_half_grid(curve["time"], curve["value"], cfg, start_angle, end_angle, reverse_time)
            )

        methods = sorted(next(iter(curves.values())).keys()) if curves else []
        det.write_video_metrics_csv(out_dir / "video_metrics.csv", per_video, methods)
        summary = det.summarize_condition(FACTOR_NAME, condition_name, distance_dir, per_video, methods)
        (out_dir / "condition_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        angle_png = out_dir / f"{group_label}_{distance_dir.name}_angle_curves.png"
        plot_angle_condition(
            angle_png,
            f"Distance {distance_num}",
            group_label,
            start_angle,
            end_angle,
            videos,
            curves,
            per_video,
            cfg,
            reverse_time,
        )
        raw = summary["methods"][METHOD]
        (distance_dir / "detection_report.md").write_text(
            "\n".join(
                [
                    f"![](detection_results/{angle_png.name})",
                    "",
                    f"Detection rate: {raw['detected']}/{raw['total']} ({raw['detection_rate'] * 100:.1f}%)",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        condition_summaries.append(summary)
        angle_plot_paths.append(angle_png)
        distance_profiles[distance_num] = np.nanmean(np.vstack(angle_profiles), axis=0)

    heatmap_path = group_dir / f"{group_label}_semicircle_heatmap_raw_abs.png"
    heatmap_norm_path = group_dir / f"{group_label}_semicircle_heatmap_row_normalized.png"
    plot_polar_heatmap(
        heatmap_path,
        distance_profiles,
        f"{group_label}: mean raw_abs by angle and distance",
        start_angle,
        end_angle,
    )
    plot_polar_heatmap(
        heatmap_norm_path,
        distance_profiles,
        f"{group_label}: row-normalized angular pattern",
        start_angle,
        end_angle,
        row_normalize=True,
    )
    write_group_summary(group_dir, group_label, condition_summaries, angle_plot_paths, heatmap_path, heatmap_norm_path)

    return {
        "group_label": group_label,
        "group_dir": group_dir,
        "condition_summaries": condition_summaries,
        "distance_profiles": distance_profiles,
        "heatmap_path": heatmap_path,
        "heatmap_norm_path": heatmap_norm_path,
    }


def boundary_mean(profile, side, window_degrees=SEAM_MATCH_WINDOW_DEGREES):
    count = max(1, int(round(window_degrees)) + 1)
    segment = profile[-count:] if side == "end" else profile[:count]
    return float(np.nanmean(segment))


def combine_full_circle_profiles(group_results, seam_correct=False):
    by_group = {result["group_label"]: result for result in group_results if result is not None}
    first = by_group.get("angle_0_180")
    second = by_group.get("angle_180_360")
    if first is None or second is None:
        return {}, {}

    distances = sorted(set(first["distance_profiles"]) & set(second["distance_profiles"]))
    full_profiles = {}
    seam_scales = {}
    for distance in distances:
        p0 = first["distance_profiles"][distance]
        p1 = second["distance_profiles"][distance].copy()
        seam_scale = 1.0
        if seam_correct:
            positive_180 = boundary_mean(p0, "end")
            negative_180 = boundary_mean(p1, "start")
            if negative_180 > SIGNAL_EPS:
                seam_scale = min(1.0, max(0.0, positive_180 / negative_180))
                p1 *= seam_scale
        full_profiles[distance] = np.concatenate([p0[:-1], p1[:-1], p1[-1:]]).astype(np.float32)
        seam_scales[distance] = seam_scale
    return full_profiles, seam_scales


def radial_edges_from_centers(radii):
    radii = np.asarray(radii, dtype=np.float32)
    if radii.size == 1:
        return np.array([radii[0] - 0.5, radii[0] + 0.5], dtype=np.float32)
    mids = 0.5 * (radii[:-1] + radii[1:])
    first = radii[0] - (mids[0] - radii[0])
    last = radii[-1] + (radii[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]]).astype(np.float32)


def gaussian_kernel1d(sigma):
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    radius = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def smooth_radial(values, sigma=RADIAL_SMOOTHING_SIGMA_SAMPLES):
    kernel = gaussian_kernel1d(sigma)
    if kernel.size == 1:
        return values
    pad = kernel.size // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.empty_like(values)
    for angle_idx in range(values.shape[1]):
        smoothed[:, angle_idx] = np.convolve(padded[:, angle_idx], kernel, mode="valid")
    return smoothed


def fit_distance_continuous_profiles(full_profiles):
    distances = np.array(sorted(full_profiles), dtype=np.float32)
    if distances.size == 0:
        return distances, np.empty((0, 0), dtype=np.float32)

    values = np.vstack([full_profiles[int(d)][:-1] for d in distances]).astype(np.float32)
    finite = values[np.isfinite(values)]
    eps = max(SIGNAL_EPS, float(np.nanpercentile(finite, 1)) * 0.05) if finite.size else SIGNAL_EPS
    log_values = np.log(np.maximum(values, eps))

    dense_radii = np.linspace(float(distances[0]), float(distances[-1]), DENSE_RADIUS_SAMPLES, dtype=np.float32)
    dense_log = np.empty((dense_radii.size, log_values.shape[1]), dtype=np.float32)
    for angle_idx in range(log_values.shape[1]):
        dense_log[:, angle_idx] = np.interp(dense_radii, distances, log_values[:, angle_idx])

    dense_values = np.exp(dense_log).astype(np.float32)
    dense_values = smooth_radial(dense_values)
    return dense_radii, dense_values


def plot_full_circle_heatmap(png_path, full_profiles, title, row_normalize=False):
    distances = sorted(full_profiles)
    if not distances:
        return
    values = np.vstack([full_profiles[d][:-1] for d in distances]).astype(np.float32)
    if row_normalize:
        row_min = np.nanmin(values, axis=1, keepdims=True)
        row_max = np.nanmax(values, axis=1, keepdims=True)
        values = (values - row_min) / np.maximum(row_max - row_min, 1e-8)

    theta_edges = np.deg2rad(np.linspace(0.0, 360.0, values.shape[1] + 1))
    radius_edges = np.arange(0.5, len(distances) + 1.5, 1.0)
    theta_mesh, radius_mesh = np.meshgrid(theta_edges, radius_edges)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")
    norm = None if row_normalize else raw_display_norm(values)
    mesh = ax.pcolormesh(theta_mesh, radius_mesh, values, shading="auto", cmap="magma", norm=norm)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlim(0.5, len(distances) + 0.5)
    ax.set_rticks(range(1, len(distances) + 1))
    ax.set_yticklabels([f"D{d}" for d in distances])
    ax.set_rlabel_position(45)
    ax.set_thetagrids(np.arange(0, 360, 30))
    ax.set_title(title, va="bottom")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.10, shrink=0.75)
    cbar.set_label(
        "row-normalized signal"
        if row_normalize
        else f"mean raw_abs signal (gamma={RAW_DISPLAY_GAMMA:g} display)"
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def plot_full_circle_heatmap_dense(png_path, radii, values, title):
    if len(radii) == 0 or values.size == 0:
        return

    theta_edges = np.deg2rad(np.linspace(0.0, 360.0, values.shape[1] + 1))
    radius_edges = radial_edges_from_centers(radii)
    theta_mesh, radius_mesh = np.meshgrid(theta_edges, radius_edges)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")
    mesh = ax.pcolormesh(
        theta_mesh,
        radius_mesh,
        values,
        shading="auto",
        cmap="magma",
        norm=raw_display_norm(values),
    )
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlim(float(radius_edges[0]), float(radius_edges[-1]))
    radius_ticks = list(range(1, int(np.ceil(float(radii[-1]))) + 1))
    ax.set_rticks(radius_ticks)
    ax.set_yticklabels([DISTANCE_LABELS_METERS.get(d, f"D{d}") for d in radius_ticks])
    ax.set_rlabel_position(45)
    ax.set_thetagrids(np.arange(0, 360, 30))
    ax.set_title(title, va="bottom")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.10, shrink=0.75)
    cbar.set_label(f"mean raw_abs signal (gamma={RAW_DISPLAY_GAMMA:g} display)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def write_factor_summary(
    factor_dir,
    group_results,
    full_heatmap_path,
    full_heatmap_norm_path,
    uncorrected_heatmap_path,
    continuous_heatmap_path,
    seam_scales,
):
    lines = [
        "# Distance + Angle Detection Summary",
        "",
        "## Group Summary",
        "",
        "| group | raw detections | raw rate | mean raw SNR dB | report |",
        "|---|---:|---:|---:|---|",
    ]
    total_detected = 0
    total_videos = 0
    weighted_snr_sum = 0.0
    weighted_snr_count = 0
    for result in group_results:
        detected = 0
        videos = 0
        snr_sum = 0.0
        snr_count = 0
        for summary in result["condition_summaries"]:
            raw = summary["methods"][METHOD]
            detected += int(raw["detected"])
            videos += int(raw["total"])
            snr_sum += float(raw["mean_snr_db"]) * int(raw["total"])
            snr_count += int(raw["total"])
        total_detected += detected
        total_videos += videos
        weighted_snr_sum += snr_sum
        weighted_snr_count += snr_count
        group_rate = detected / max(1, videos)
        group_snr = snr_sum / max(1, snr_count)
        report_rel = result["group_dir"].relative_to(factor_dir).as_posix() + "/angle_distance_summary.md"
        lines.append(
            f"| {result['group_label']} | {detected}/{videos} | {group_rate * 100:.1f}% | "
            f"{group_snr:.2f} | [{result['group_label']}]({report_rel}) |"
        )

    total_rate = total_detected / max(1, total_videos)
    total_snr = weighted_snr_sum / max(1, weighted_snr_count)
    lines.extend(
        [
            "",
            f"Total raw detection rate: {total_detected}/{total_videos} ({total_rate * 100:.1f}%).",
            f"Mean raw SNR dB: {total_snr:.2f}.",
            "",
            "## 0-360 Heatmap",
            "",
            f"![]({full_heatmap_path.name})",
            "",
            f"![]({continuous_heatmap_path.name})",
            "",
            f"![]({full_heatmap_norm_path.name})",
            "",
            "## Seam Correction",
            "",
            "The 180-360 degree half was scaled per distance so its 180 degree boundary matches the 0-180 degree half.",
            "",
            "| distance | negative-half scale |",
            "|---|---:|",
        ]
    )
    for distance in sorted(seam_scales):
        lines.append(f"| D{distance} | {seam_scales[distance]:.4f} |")
    lines.extend(["", "## Original Uncorrected Raw", "", f"![]({uncorrected_heatmap_path.name})", ""])
    (factor_dir / "angle_distance_0-360_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    det = load_base_module()
    cfg = det.Config(root=ROOT)
    factor_dir = ROOT / FACTOR_NAME
    if not factor_dir.exists():
        raise FileNotFoundError(f"Factor directory not found: {factor_dir}")

    group_results = []
    for spec in GROUP_SPECS:
        result = process_angle_group(det, cfg, factor_dir, spec)
        if result is not None:
            group_results.append(result)

    uncorrected_profiles, _ = combine_full_circle_profiles(group_results, seam_correct=False)
    full_profiles, seam_scales = combine_full_circle_profiles(group_results, seam_correct=True)
    if full_profiles:
        uncorrected_heatmap_path = factor_dir / "0-360_full_circle_heatmap_raw_abs_uncorrected.png"
        full_heatmap_path = factor_dir / "0-360_full_circle_heatmap_raw_abs.png"
        full_heatmap_norm_path = factor_dir / "0-360_full_circle_heatmap_row_normalized.png"
        continuous_heatmap_path = factor_dir / "0-360_full_circle_heatmap_raw_abs_seam_distance_continuous.png"
        continuous_heatmap_pdf_path = factor_dir / "0-360_full_circle_heatmap_raw_abs_seam_distance_continuous.pdf"
        if uncorrected_profiles:
            plot_full_circle_heatmap(
                uncorrected_heatmap_path,
                uncorrected_profiles,
                "0-360 deg: uncorrected mean raw_abs by angle and distance",
            )
        plot_full_circle_heatmap(
            full_heatmap_path,
            full_profiles,
            "0-360 deg: seam-corrected mean raw_abs by angle and distance",
        )
        plot_full_circle_heatmap(
            full_heatmap_norm_path,
            full_profiles,
            "0-360 deg: seam-corrected row-normalized angular pattern",
            row_normalize=True,
        )
        dense_radii, dense_values = fit_distance_continuous_profiles(full_profiles)
        plot_full_circle_heatmap_dense(
            continuous_heatmap_path,
            dense_radii,
            dense_values,
            "0-360 deg: seam-corrected distance-continuous raw_abs",
        )
        plot_full_circle_heatmap_dense(
            continuous_heatmap_pdf_path,
            dense_radii,
            dense_values,
            "0-360 deg: seam-corrected distance-continuous raw_abs",
        )
        write_factor_summary(
            factor_dir,
            group_results,
            full_heatmap_path,
            full_heatmap_norm_path,
            uncorrected_heatmap_path,
            continuous_heatmap_path,
            seam_scales,
        )

    total_videos = sum(sum(s["num_videos"] for s in r["condition_summaries"]) for r in group_results)
    total_distances = sum(len(r["condition_summaries"]) for r in group_results)
    print(f"Processed {total_videos} videos in {total_distances} angle-distance conditions.")
    if full_profiles:
        print(f"Wrote {factor_dir / 'angle_distance_0-360_summary.md'}")


if __name__ == "__main__":
    main()
