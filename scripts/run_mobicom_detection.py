import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".m4v"}


@dataclass
class Config:
    root: Path
    resize: int = 96
    frame_stride: int = 1
    reference_seconds: float = 5.0
    active_start_seconds: float = 5.0
    max_seconds: float = 0.0
    threshold_z: float = 5.0
    min_relative_gain: float = 1.15
    min_active_fraction: float = 0.50
    highpass_sigma: float = 5.0
    topk_fraction: float = 0.05
    clean_reference_spikes: bool = True
    reference_spike_z: float = 3.5
    max_reference_spike_seconds: float = 2.2
    reference_mode: str = "auto_platform"
    reference_platform_min_seconds: float = 1.0
    reference_transition_z: float = 4.0
    reference_segment_z: float = 3.5
    primary_method: str = "raw_abs"
    run_alternatives: bool = True
    only_factor: str = ""
    only_condition: str = ""
    limit_videos: int = 0


def main():
    parser = argparse.ArgumentParser(description="Batch wall-video differential detection for Mobicom experiments.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--resize", type=int, default=96)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--reference-seconds", type=float, default=5.0)
    parser.add_argument("--active-start-seconds", type=float, default=5.0)
    parser.add_argument("--max-seconds", type=float, default=0.0, help="0 means use the whole video.")
    parser.add_argument("--threshold-z", type=float, default=5.0)
    parser.add_argument("--min-relative-gain", type=float, default=1.15)
    parser.add_argument("--min-active-fraction", type=float, default=0.50)
    parser.add_argument("--no-clean-reference-spikes", action="store_true")
    parser.add_argument("--reference-spike-z", type=float, default=3.5)
    parser.add_argument("--max-reference-spike-seconds", type=float, default=2.2)
    parser.add_argument("--reference-mode", choices=("auto_platform", "spike_clean", "fixed"), default="auto_platform")
    parser.add_argument("--reference-platform-min-seconds", type=float, default=1.0)
    parser.add_argument("--reference-transition-z", type=float, default=4.0)
    parser.add_argument("--reference-segment-z", type=float, default=3.5)
    parser.add_argument("--no-alternatives", action="store_true")
    parser.add_argument("--only-factor", default="")
    parser.add_argument("--only-condition", default="")
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--summaries-only", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        root=Path(args.root),
        resize=args.resize,
        frame_stride=max(1, args.frame_stride),
        reference_seconds=args.reference_seconds,
        active_start_seconds=args.active_start_seconds,
        max_seconds=args.max_seconds,
        threshold_z=args.threshold_z,
        min_relative_gain=args.min_relative_gain,
        min_active_fraction=args.min_active_fraction,
        clean_reference_spikes=not args.no_clean_reference_spikes,
        reference_spike_z=args.reference_spike_z,
        max_reference_spike_seconds=args.max_reference_spike_seconds,
        reference_mode=args.reference_mode,
        reference_platform_min_seconds=args.reference_platform_min_seconds,
        reference_transition_z=args.reference_transition_z,
        reference_segment_z=args.reference_segment_z,
        run_alternatives=not args.no_alternatives,
        only_factor=args.only_factor,
        only_condition=args.only_condition,
        limit_videos=max(0, args.limit_videos),
    )

    if not cfg.root.exists():
        raise FileNotFoundError(f"Experiment root not found: {cfg.root}")

    if args.summaries_only:
        summaries = load_existing_summaries(cfg)
        by_factor = {}
        for summary in summaries:
            by_factor.setdefault(summary["factor"], []).append(summary)
        for factor_name, factor_summaries in by_factor.items():
            write_factor_summary(cfg.root / factor_name, factor_summaries, cfg)
        write_global_summary(cfg.root, summaries, cfg)
        print(f"Rebuilt summaries from {len(summaries)} existing condition summaries.")
        return

    all_condition_summaries = []
    for factor_dir in sorted([p for p in cfg.root.iterdir() if p.is_dir()], key=lambda p: p.name):
        if cfg.only_factor and factor_dir.name != cfg.only_factor:
            continue
        condition_summaries = []
        for condition_dir in find_condition_dirs(factor_dir):
            condition_name = condition_label(factor_dir, condition_dir)
            if cfg.only_condition and cfg.only_condition not in {condition_dir.name, condition_name}:
                continue
            videos = find_videos(condition_dir)
            if not videos:
                continue
            if cfg.limit_videos > 0:
                videos = videos[: cfg.limit_videos]
            print(f"[condition] {factor_dir.name}/{condition_name}: {len(videos)} videos")
            summary = process_condition(factor_dir.name, condition_dir, videos, cfg, condition_name)
            condition_summaries.append(summary)
            all_condition_summaries.append(summary)
        if condition_summaries:
            write_factor_summary(factor_dir, condition_summaries, cfg)

    write_global_summary(cfg.root, all_condition_summaries, cfg)
    print(f"Finished. Processed {sum(s['num_videos'] for s in all_condition_summaries)} videos.")


def load_existing_summaries(cfg):
    summaries = []
    for factor_dir in sorted([p for p in cfg.root.iterdir() if p.is_dir()], key=lambda p: p.name):
        if cfg.only_factor and factor_dir.name != cfg.only_factor:
            continue
        for condition_dir in find_condition_dirs(factor_dir):
            condition_name = condition_label(factor_dir, condition_dir)
            if cfg.only_condition and cfg.only_condition not in {condition_dir.name, condition_name}:
                continue
            path = condition_dir / "detection_results" / "condition_summary.json"
            if path.exists():
                summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return summaries


def natural_key(path):
    text = path.name if isinstance(path, Path) else str(path)
    parts = []
    cur = ""
    is_digit = False
    for ch in text:
        if ch.isdigit() != is_digit:
            if cur:
                parts.append(int(cur) if is_digit else cur.lower())
            cur = ch
            is_digit = ch.isdigit()
        else:
            cur += ch
    if cur:
        parts.append(int(cur) if is_digit else cur.lower())
    return parts


def find_videos(folder):
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES],
        key=natural_key,
    )


def find_condition_dirs(factor_dir):
    condition_dirs = []
    if find_videos(factor_dir):
        condition_dirs.append(factor_dir)
    for candidate in factor_dir.rglob("*"):
        if not candidate.is_dir():
            continue
        rel_parts = candidate.relative_to(factor_dir).parts
        if "__pycache__" in rel_parts or "detection_results" in rel_parts:
            continue
        if find_videos(candidate):
            condition_dirs.append(candidate)
    return sorted(condition_dirs, key=lambda p: natural_key(str(p.relative_to(factor_dir))))


def condition_label(factor_dir, condition_dir):
    if condition_dir == factor_dir:
        return factor_dir.name
    return str(condition_dir.relative_to(factor_dir)).replace(os.sep, "/")


def safe_path_name(text):
    return "".join("_" if ch in '<>:"/\\|?*' else ch for ch in str(text)).strip(" .") or "condition"


def process_condition(factor_name, condition_dir, videos, cfg, condition_name=None):
    out_dir = condition_dir / "detection_results"
    out_dir.mkdir(exist_ok=True)
    condition_name = condition_name or condition_dir.name

    per_video = []
    curves = {}
    for idx, video_path in enumerate(videos, start=1):
        print(f"  [{idx:02d}/{len(videos):02d}] {video_path.name}")
        result = process_video(video_path, cfg)
        per_video.append(result["summary"])
        curves[video_path.name] = result["curves"]

    methods = sorted(next(iter(curves.values())).keys()) if curves else []
    csv_path = out_dir / "video_metrics.csv"
    write_video_metrics_csv(csv_path, per_video, methods)

    summary = summarize_condition(factor_name, condition_name, condition_dir, per_video, methods)
    json_path = out_dir / "condition_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_name = safe_path_name(condition_name)
    primary_png = out_dir / f"{plot_name}_raw_abs_curves.png"
    plot_condition_curves(primary_png, condition_name, videos, curves, per_video, "raw_abs", cfg)

    selected_method = select_method(summary, cfg)
    selected_png = primary_png
    if selected_method != "raw_abs":
        selected_png = out_dir / f"{plot_name}_{selected_method}_curves.png"
        plot_condition_curves(selected_png, condition_name, videos, curves, per_video, selected_method, cfg)
        summary["selected_method"] = selected_method
        summary["selected_plot"] = str(selected_png)
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = condition_dir / "detection_report.md"
    write_condition_report(report_path, condition_dir, primary_png, selected_png, summary, cfg)
    return summary


def process_video(video_path, cfg):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-6 or math.isnan(fps):
        fps = 30.0
    effective_fps = fps / cfg.frame_stride
    max_raw_frames = int(cfg.max_seconds * fps) if cfg.max_seconds > 0 else None

    frames = []
    raw_frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_raw_frames is not None and raw_frame_id >= max_raw_frames:
            break
        if raw_frame_id % cfg.frame_stride == 0:
            frames.append(prepare_frame(frame, cfg.resize))
        raw_frame_id += 1
    cap.release()

    if len(frames) < max(5, int(effective_fps)):
        raise RuntimeError(f"Too few frames loaded from {video_path}: {len(frames)}")

    arr = np.stack(frames, axis=0).astype(np.float32)
    time = np.arange(arr.shape[0], dtype=np.float32) / max(effective_fps, 1e-6)

    ref_mask = time < cfg.reference_seconds
    active_mask = time >= cfg.active_start_seconds
    if ref_mask.sum() < 5:
        ref_mask[: max(5, int(0.2 * len(time)))] = True
    if active_mask.sum() < 5:
        active_mask[max(1, int(0.5 * len(time))) :] = True

    clean_ref_mask, spike_mask, spike_info = select_reference_mask(arr, time, ref_mask, cfg)
    ref = np.mean(arr[clean_ref_mask], axis=0)
    curves = compute_curves(arr, ref, cfg)
    summary = {
        "video": video_path.name,
        "path": str(video_path),
        "fps": float(fps),
        "effective_fps": float(effective_fps),
        "num_frames": int(arr.shape[0]),
        "duration_s": float(time[-1]) if len(time) else 0.0,
        "reference_frame_count": int(ref_mask.sum()),
        "clean_reference_frame_count": int(clean_ref_mask.sum()),
        "reference_spike_frame_count": int(spike_mask.sum()),
        "reference_spike_duration_s": float(spike_info["duration_s"]),
        "reference_spike_fraction": float(spike_info["fraction"]),
        "reference_mode": spike_info.get("mode", cfg.reference_mode),
        "reference_platform_start_s": float(spike_info.get("platform_start_s", 0.0)),
        "reference_platform_end_s": float(spike_info.get("platform_end_s", cfg.reference_seconds)),
        "reference_platform_duration_s": float(spike_info.get("platform_duration_s", 0.0)),
    }

    pre_ref_mask = time < float(spike_info.get("platform_start_s", 0.0))
    for method, values in curves.items():
        stats = score_curve(values, clean_ref_mask, active_mask, cfg, pre_ref_mask)
        summary.update({f"{method}_{k}": v for k, v in stats.items()})

    curves_with_time = {}
    for method, values in curves.items():
        curves_with_time[method] = {"time": time, "value": values}

    return {"summary": summary, "curves": curves_with_time}


def select_reference_mask(arr, time, ref_mask, cfg):
    if cfg.reference_mode == "fixed":
        spike_mask = np.zeros_like(ref_mask, dtype=bool)
        info = {
            "duration_s": 0.0,
            "fraction": 0.0,
            "mode": "fixed",
            "platform_start_s": float(time[np.flatnonzero(ref_mask)[0]]) if np.any(ref_mask) else 0.0,
            "platform_end_s": float(time[np.flatnonzero(ref_mask)[-1]]) if np.any(ref_mask) else cfg.reference_seconds,
            "platform_duration_s": float(ref_mask.sum()) * median_frame_dt(time),
        }
        return ref_mask.copy(), spike_mask, info
    if cfg.reference_mode == "spike_clean":
        return clean_reference_spike_mask(arr, time, ref_mask, cfg)
    return select_auto_platform_reference_mask(arr, time, ref_mask, cfg)


def select_auto_platform_reference_mask(arr, time, ref_mask, cfg):
    ref_indices = np.flatnonzero(ref_mask)
    spike_mask = np.zeros_like(ref_mask, dtype=bool)
    if ref_indices.size < 8:
        return clean_reference_spike_mask(arr, time, ref_mask, cfg)

    ref_stack = arr[ref_indices]
    dt = median_frame_dt(time[ref_indices])
    min_frames = max(5, int(math.ceil(cfg.reference_platform_min_seconds / max(dt, 1e-6))))

    frame_step = np.zeros(ref_indices.size, dtype=np.float64)
    frame_step[1:] = np.mean(np.abs(ref_stack[1:] - ref_stack[:-1]), axis=(1, 2))
    frame_mean = np.mean(ref_stack, axis=(1, 2))
    brightness_step = np.zeros(ref_indices.size, dtype=np.float64)
    brightness_step[1:] = np.abs(np.diff(frame_mean))

    step_thr = robust_high_threshold(frame_step[1:], cfg.reference_transition_z)
    brightness_thr = robust_high_threshold(brightness_step[1:], cfg.reference_transition_z)
    transition = (frame_step > step_thr) | (brightness_step > brightness_thr)
    transition[0] = False
    transition_points = np.flatnonzero(transition)
    groups = stable_groups_from_transition_points(
        ref_indices.size,
        transition_points,
        max(1, int(round(0.10 / max(dt, 1e-6)))),
    )
    candidates = []
    for start, end in groups:
        if end - start < min_frames:
            continue
        segment = ref_stack[start:end]
        median_frame = np.median(segment, axis=0)
        segment_scores = np.mean(np.abs(segment - median_frame[None, :, :]), axis=(1, 2))
        segment_center = float(np.median(segment_scores))
        segment_scale = robust_scale(segment_scores)
        segment_limit = segment_center + cfg.reference_segment_z * max(segment_scale, 1e-8)
        stable_fraction = float(np.mean(segment_scores <= segment_limit))
        segment_p90 = float(np.percentile(segment_scores, 90))
        segment_step_p90 = float(np.percentile(frame_step[start:end], 90))
        # Primary objective: use as many stable continuous frames as possible.
        # Ties are broken by lower internal variation, then by later end time.
        candidates.append({
            "start": start,
            "end": end,
            "length": end - start,
            "variation": segment_p90 + segment_step_p90,
            "end_time": float(time[ref_indices[end - 1]]),
        })

    if candidates:
        if transition_points.size > 0:
            last_transition = int(transition_points[-1])
            preferred = [item for item in candidates if item["start"] > last_transition]
        else:
            preferred = candidates
        if not preferred:
            preferred = candidates
        max_len = max(item["length"] for item in preferred)
        long_candidates = [item for item in preferred if item["length"] >= 0.85 * max_len]
        chosen_item = sorted(long_candidates, key=lambda item: (-item["length"], item["variation"], -item["end_time"]))[0]
        chosen = (chosen_item["start"], chosen_item["end"])
    else:
        end = ref_indices.size
        start = max(0, end - min_frames)
        chosen = (start, end)

    start, end = chosen
    clean_ref_mask = np.zeros_like(ref_mask, dtype=bool)
    clean_ref_mask[ref_indices[start:end]] = True

    if cfg.clean_reference_spikes:
        inner_clean, inner_spike, inner_info = clean_reference_spike_mask(arr, time, clean_ref_mask, cfg)
        clean_ref_mask = inner_clean
        spike_mask = inner_spike
        spike_duration = float(inner_info.get("duration_s", 0.0))
        spike_fraction = float(inner_info.get("fraction", 0.0))
    else:
        spike_duration = 0.0
        spike_fraction = 0.0

    selected_indices = np.flatnonzero(clean_ref_mask)
    if selected_indices.size == 0:
        clean_ref_mask[ref_indices[start:end]] = True
        selected_indices = np.flatnonzero(clean_ref_mask)

    platform_start = float(time[selected_indices[0]])
    platform_end = float(time[selected_indices[-1]] + dt)
    info = {
        "duration_s": spike_duration,
        "fraction": spike_fraction,
        "mode": "auto_platform",
        "platform_start_s": platform_start,
        "platform_end_s": platform_end,
        "platform_duration_s": float(selected_indices.size * dt),
    }
    return clean_ref_mask, spike_mask, info


def platform_threshold(curve, end_curve):
    end_med = float(np.median(end_curve))
    end_mad = float(np.median(np.abs(end_curve - end_med)))
    global_iqr = float(np.percentile(curve, 75) - np.percentile(curve, 25))
    global_range = float(np.percentile(curve, 95) - np.percentile(curve, 5))
    return max(3.5 * 1.4826 * end_mad, 0.20 * global_iqr, 0.08 * global_range, 1e-6)


def robust_high_threshold(values, z_value):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.inf
    center = float(np.median(values))
    scale = robust_scale(values)
    p95 = float(np.percentile(values, 95))
    return max(center + z_value * max(scale, 1e-8), p95)


def robust_scale(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    if scale <= 1e-8 and values.size > 1:
        scale = float(np.std(values))
    return scale


def moving_median(values, window):
    values = np.asarray(values)
    window = max(1, int(window))
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    out = np.empty_like(values)
    for ii in range(values.size):
        out[ii] = np.median(padded[ii:ii + window])
    return out


def fill_short_false_gaps(mask, max_gap):
    mask = np.asarray(mask, dtype=bool).copy()
    if max_gap <= 0:
        return mask
    false_groups = contiguous_true_groups(~mask)
    for start, end in false_groups:
        if end - start <= max_gap:
            mask[start:end] = True
    return mask


def stable_groups_from_transition_points(length, transition_points, guard_radius):
    blocked = np.zeros(length, dtype=bool)
    guard_radius = max(0, int(guard_radius))
    for point in np.asarray(transition_points, dtype=int):
        start = max(0, point - guard_radius)
        end = min(length, point + guard_radius + 1)
        blocked[start:end] = True
    return contiguous_true_groups(~blocked)


def expand_true_mask(mask, radius):
    mask = np.asarray(mask, dtype=bool)
    radius = max(0, int(radius))
    if radius == 0 or not np.any(mask):
        return mask.copy()
    expanded = mask.copy()
    true_idx = np.flatnonzero(mask)
    for idx in true_idx:
        start = max(0, idx - radius)
        end = min(mask.size, idx + radius + 1)
        expanded[start:end] = True
    return expanded


def median_frame_dt(time_values):
    time_values = np.asarray(time_values)
    if time_values.size > 1:
        return float(np.median(np.diff(time_values)))
    return 0.0


def clean_reference_spike_mask(arr, time, ref_mask, cfg):
    clean_ref_mask = ref_mask.copy()
    spike_mask = np.zeros_like(ref_mask, dtype=bool)
    info = {"duration_s": 0.0, "fraction": 0.0}
    if not cfg.clean_reference_spikes:
        return clean_ref_mask, spike_mask, info

    ref_indices = np.flatnonzero(ref_mask)
    if ref_indices.size < 8:
        return clean_ref_mask, spike_mask, info

    ref_stack = arr[ref_indices]
    robust_ref = np.median(ref_stack, axis=0)
    ref_scores = np.mean(np.abs(ref_stack - robust_ref[None, :, :]), axis=(1, 2))
    center = float(np.median(ref_scores))
    mad = float(np.median(np.abs(ref_scores - center)))
    scale = 1.4826 * mad
    if scale <= 1e-8 and ref_scores.size > 1:
        scale = float(np.std(ref_scores))
    if scale <= 1e-8:
        return clean_ref_mask, spike_mask, info

    z = (ref_scores - center) / scale

    frame_mean = np.mean(ref_stack, axis=(1, 2))
    mean_center = float(np.median(frame_mean))
    mean_mad = float(np.median(np.abs(frame_mean - mean_center)))
    mean_scale = 1.4826 * mean_mad
    if mean_scale <= 1e-8 and frame_mean.size > 1:
        mean_scale = float(np.std(frame_mean))
    if mean_scale > 1e-8:
        brightness_z = np.abs(frame_mean - mean_center) / mean_scale
    else:
        brightness_z = np.zeros_like(frame_mean)

    candidate_local = (z >= cfg.reference_spike_z) | (brightness_z >= cfg.reference_spike_z)
    if not np.any(candidate_local):
        return clean_ref_mask, spike_mask, info

    groups = contiguous_true_groups(candidate_local)
    for start, end in groups:
        group_indices = ref_indices[start:end]
        duration = float(time[group_indices[-1]] - time[group_indices[0]]) if group_indices.size > 1 else 0.0
        duration += float(np.median(np.diff(time[ref_indices]))) if ref_indices.size > 1 else 0.0
        if duration <= cfg.max_reference_spike_seconds:
            spike_mask[group_indices] = True

    kept = ref_mask & ~spike_mask
    min_kept = min(int(ref_mask.sum()), max(5, int(math.ceil(0.5 * ref_mask.sum()))))
    if int(kept.sum()) < min_kept:
        spike_mask[:] = False
        return clean_ref_mask, spike_mask, info

    clean_ref_mask = kept
    frame_dt = float(np.median(np.diff(time[ref_indices]))) if ref_indices.size > 1 else 0.0
    info["duration_s"] = float(spike_mask.sum() * frame_dt)
    info["fraction"] = float(spike_mask.sum() / max(1, ref_mask.sum()))
    return clean_ref_mask, spike_mask, info


def contiguous_true_groups(mask):
    mask_int = np.asarray(mask, dtype=np.int8)
    starts = np.flatnonzero(np.diff(np.r_[0, mask_int]) == 1)
    ends = np.flatnonzero(np.diff(np.r_[mask_int, 0]) == -1) + 1
    return list(zip(starts, ends))


def prepare_frame(frame, resize):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (resize, resize), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32) / 255.0


def compute_curves(arr, ref, cfg):
    diff = arr - ref[None, :, :]
    abs_diff = np.abs(diff)

    raw_abs = abs_diff.mean(axis=(1, 2))
    signed_abs_mean_removed = np.abs(diff - diff.mean(axis=(1, 2), keepdims=True)).mean(axis=(1, 2))

    ref_blur = cv2.GaussianBlur(ref, (0, 0), cfg.highpass_sigma)
    hp_ref = ref - ref_blur
    hp_values = []
    topk_values = []
    for frame in arr:
        hp = frame - cv2.GaussianBlur(frame, (0, 0), cfg.highpass_sigma)
        hp_abs = np.abs(hp - hp_ref)
        hp_values.append(float(hp_abs.mean()))
        k = max(1, int(hp_abs.size * cfg.topk_fraction))
        topk_values.append(float(np.partition(hp_abs.ravel(), -k)[-k:].mean()))

    prev = np.concatenate([arr[:1], arr[:-1]], axis=0)
    temporal_abs = np.abs(arr - prev).mean(axis=(1, 2))

    return {
        "raw_abs": raw_abs.astype(np.float32),
        "mean_removed_abs": signed_abs_mean_removed.astype(np.float32),
        "highpass_abs": np.asarray(hp_values, dtype=np.float32),
        "highpass_topk": np.asarray(topk_values, dtype=np.float32),
        "temporal_abs": temporal_abs.astype(np.float32),
    }


def score_curve(values, ref_mask, active_mask, cfg, pre_mask=None):
    base = values[ref_mask]
    active = values[active_mask]
    pre = values[pre_mask] if pre_mask is not None and np.any(pre_mask) else np.asarray([], dtype=values.dtype)
    base_median = float(np.median(base))
    base_mean = float(np.mean(base))
    base_std = float(np.std(base))
    mad = float(np.median(np.abs(base - base_median)))
    robust_scale = max(1.4826 * mad, base_std, 1e-8)
    active_mean = float(np.mean(active))
    active_median = float(np.median(active))
    active_p90 = float(np.percentile(active, 90))
    active_p95 = float(np.percentile(active, 95))
    delta_mean = active_mean - base_mean
    delta_median = active_median - base_median
    delta_p90 = active_p90 - base_median
    snr_mean = delta_mean / robust_scale
    snr_median = delta_median / robust_scale
    snr_p90 = delta_p90 / robust_scale
    snr_mean_db = linear_snr_to_db(max(snr_mean, 0.0))
    snr_median_db = linear_snr_to_db(max(snr_median, 0.0))
    snr_p90_db = linear_snr_to_db(max(snr_p90, 0.0))
    relative_gain = active_mean / max(base_mean, 1e-8)
    active_threshold = base_median + cfg.threshold_z * robust_scale
    active_fraction_above = float(np.mean(active >= active_threshold))
    active_peak = max(0.0, active_p95 - base_median)
    pre_transient_peak = max(0.0, float(np.percentile(pre, 99)) - base_median) if pre.size else 0.0
    active_peak_over_pre = active_peak / max(pre_transient_peak, 1e-8)
    peak_detected = bool(snr_p90 >= cfg.threshold_z and relative_gain >= cfg.min_relative_gain)
    pulse_detected = bool(
        snr_p90 >= cfg.threshold_z
        and relative_gain >= cfg.min_relative_gain
        and active_fraction_above >= 0.20
    )
    persistent_detected = bool(
        relative_gain >= cfg.min_relative_gain
        and (
            snr_median >= cfg.threshold_z
            or active_fraction_above >= cfg.min_active_fraction
        )
    )
    transient_suppressed = bool(
        pre_transient_peak >= 5.0 * robust_scale
        and active_peak_over_pre < 0.15
    )
    detected = bool((persistent_detected or pulse_detected) and not transient_suppressed)
    return {
        "base_mean": base_mean,
        "base_median": base_median,
        "base_std": base_std,
        "noise_scale": robust_scale,
        "active_mean": active_mean,
        "active_median": active_median,
        "active_p90": active_p90,
        "active_p95": active_p95,
        "delta_mean": float(delta_mean),
        "delta_median": float(delta_median),
        "delta_p90": float(delta_p90),
        "snr_mean": float(snr_mean),
        "snr_median": float(snr_median),
        "snr_p90": float(snr_p90),
        "snr_mean_db": float(snr_mean_db),
        "snr_median_db": float(snr_median_db),
        "snr_p90_db": float(snr_p90_db),
        "active_threshold": float(active_threshold),
        "active_fraction_above": active_fraction_above,
        "active_peak": float(active_peak),
        "pre_transient_peak": float(pre_transient_peak),
        "active_peak_over_pre": float(active_peak_over_pre),
        "relative_gain": float(relative_gain),
        "peak_detected": peak_detected,
        "pulse_detected": pulse_detected,
        "persistent_detected": persistent_detected,
        "transient_suppressed": transient_suppressed,
        "detected": detected,
    }


def linear_snr_to_db(value):
    return 20.0 * math.log10(max(float(value), 1e-6))


def summarize_condition(factor_name, condition_name, condition_dir, per_video, methods):
    summary = {
        "factor": factor_name,
        "condition": condition_name,
        "condition_dir": str(condition_dir),
        "num_videos": len(per_video),
        "mean_reference_spike_duration_s": float(np.mean([row.get("reference_spike_duration_s", 0.0) for row in per_video])),
        "total_reference_spike_frames": int(sum(row.get("reference_spike_frame_count", 0) for row in per_video)),
        "methods": {},
    }
    for method in methods:
        detections = [bool(row[f"{method}_detected"]) for row in per_video]
        summary["methods"][method] = {
            "detected": int(sum(detections)),
            "total": len(detections),
            "detection_rate": float(sum(detections) / max(1, len(detections))),
            "mean_snr_p90": float(np.mean([row[f"{method}_snr_p90"] for row in per_video])),
            "median_snr_p90": float(np.median([row[f"{method}_snr_p90"] for row in per_video])),
            "mean_snr_median": float(np.mean([row[f"{method}_snr_median"] for row in per_video])),
            "mean_snr_db": float(np.mean([row[f"{method}_snr_mean_db"] for row in per_video])),
            "median_snr_db": float(np.median([row[f"{method}_snr_mean_db"] for row in per_video])),
            "mean_snr_p90_db": float(np.mean([row[f"{method}_snr_p90_db"] for row in per_video])),
            "median_active_fraction_above": float(np.median([row[f"{method}_active_fraction_above"] for row in per_video])),
            "mean_active_mean": float(np.mean([row[f"{method}_active_mean"] for row in per_video])),
            "mean_base_mean": float(np.mean([row[f"{method}_base_mean"] for row in per_video])),
            "mean_delta_mean": float(np.mean([row[f"{method}_delta_mean"] for row in per_video])),
            "mean_relative_gain": float(np.mean([row[f"{method}_relative_gain"] for row in per_video])),
        }
    summary["selected_method"] = select_method(summary, None)
    return summary


def select_method(summary, cfg):
    ranked = sorted(
        summary["methods"].items(),
        key=lambda kv: (
            kv[1]["detection_rate"],
            kv[0] == "raw_abs",
            kv[1]["median_snr_p90"],
        ),
        reverse=True,
    )
    return ranked[0][0] if ranked else "raw_abs"


def write_video_metrics_csv(csv_path, per_video, methods):
    fieldnames = [
        "video",
        "path",
        "fps",
        "effective_fps",
        "num_frames",
        "duration_s",
        "reference_frame_count",
        "clean_reference_frame_count",
        "reference_spike_frame_count",
        "reference_spike_duration_s",
        "reference_spike_fraction",
        "reference_mode",
        "reference_platform_start_s",
        "reference_platform_end_s",
        "reference_platform_duration_s",
    ]
    metric_names = [
        "detected",
        "base_mean",
        "active_mean",
        "active_median",
        "delta_mean",
        "delta_median",
        "active_p90",
        "active_p95",
        "snr_mean",
        "snr_median",
        "snr_p90",
        "snr_mean_db",
        "snr_median_db",
        "snr_p90_db",
        "active_fraction_above",
        "active_peak",
        "pre_transient_peak",
        "active_peak_over_pre",
        "relative_gain",
        "peak_detected",
        "pulse_detected",
        "persistent_detected",
        "transient_suppressed",
    ]
    for method in methods:
        for metric in metric_names:
            fieldnames.append(f"{method}_{metric}")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in per_video:
            writer.writerow(row)


def plot_condition_curves(png_path, condition_name, videos, curves, per_video, method, cfg):
    n = len(videos)
    cols = 2
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, max(5, 2.4 * rows)), squeeze=False)
    axes_flat = axes.ravel()
    for ax, video in zip(axes_flat, videos):
        name = video.name
        data = curves[name][method]
        time = data["time"]
        value = data["value"]
        row = per_video[[v["video"] for v in per_video].index(name)]
        detected = row[f"{method}_detected"]
        snr = row[f"{method}_snr_mean_db"]
        ax.plot(time, value, color="#1f77b4", linewidth=1.0)
        ax.axvspan(0, cfg.reference_seconds, color="#cccccc", alpha=0.25)
        platform_start = float(row.get("reference_platform_start_s", 0.0))
        platform_end = float(row.get("reference_platform_end_s", cfg.reference_seconds))
        ax.axvspan(platform_start, platform_end, color="#2ca02c", alpha=0.28)
        ax.axvline(platform_start, color="#2ca02c", linestyle=":", linewidth=1.0)
        ax.axvline(cfg.active_start_seconds, color="#d62728", linestyle="--", linewidth=0.9)
        ax.set_title(f"{name} | {'detected' if detected else 'miss'} | SNR={snr:.1f} dB", fontsize=9)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(method)
        ax.grid(True, alpha=0.25)
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.suptitle(f"{condition_name} | {method}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def write_condition_report(report_path, condition_dir, primary_png, selected_png, summary, cfg):
    raw = summary["methods"]["raw_abs"]
    lines = [
        f"![](detection_results/{primary_png.name})",
        "",
        f"Detection rate: {raw['detected']}/{raw['total']} ({raw['detection_rate'] * 100:.1f}%)",
    ]
    if selected_png != primary_png:
        selected = summary["methods"][summary["selected_method"]]
        lines.extend(
            [
                "",
                f"![](detection_results/{selected_png.name})",
                "",
                f"Alternative detection rate: {selected['detected']}/{selected['total']} ({selected['detection_rate'] * 100:.1f}%)",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_factor_summary(factor_dir, condition_summaries, cfg):
    md_path = factor_dir / "summary_detection.md"
    csv_path = factor_dir / "summary_detection.csv"

    methods = sorted(condition_summaries[0]["methods"].keys()) if condition_summaries else []
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "factor",
            "condition",
            "selected_method",
            "raw_detection_rate",
            "raw_mean_snr_db",
            "raw_mean_snr_p90_linear",
            "raw_mean_delta_mean",
            "raw_mean_relative_gain",
            "selected_detection_rate",
            "selected_mean_snr_db",
            "selected_mean_snr_p90_linear",
            "selected_mean_delta_mean",
            "selected_mean_relative_gain",
            "mean_reference_spike_duration_s",
            "total_reference_spike_frames",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in condition_summaries:
            raw = s["methods"]["raw_abs"]
            sel = s["methods"][s["selected_method"]]
            writer.writerow(
                {
                    "factor": s["factor"],
                    "condition": s["condition"],
                    "selected_method": s["selected_method"],
                    "raw_detection_rate": raw["detection_rate"],
                    "raw_mean_snr_db": summary_snr_db(raw),
                    "raw_mean_snr_p90_linear": raw["mean_snr_p90"],
                    "raw_mean_delta_mean": raw["mean_delta_mean"],
                    "raw_mean_relative_gain": raw["mean_relative_gain"],
                    "selected_detection_rate": sel["detection_rate"],
                    "selected_mean_snr_db": summary_snr_db(sel),
                    "selected_mean_snr_p90_linear": sel["mean_snr_p90"],
                    "selected_mean_delta_mean": sel["mean_delta_mean"],
                    "selected_mean_relative_gain": sel["mean_relative_gain"],
                    "mean_reference_spike_duration_s": s.get("mean_reference_spike_duration_s", 0.0),
                    "total_reference_spike_frames": s.get("total_reference_spike_frames", 0),
                }
            )

    lines = [
        f"# {factor_dir.name} detection summary",
        "",
        "## Condition table",
        "",
        "| condition | raw rate | raw SNR dB | raw p90 SNR linear | raw delta | raw gain | ref spike s | selected | selected rate |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for s in condition_summaries:
        raw = s["methods"]["raw_abs"]
        sel = s["methods"][s["selected_method"]]
        lines.append(
            f"| {s['condition']} | {raw['detection_rate'] * 100:.1f}% | "
            f"{summary_snr_db(raw):.2f} | {raw['mean_snr_p90']:.2f} | {raw['mean_delta_mean']:.5f} | "
            f"{raw['mean_relative_gain']:.2f} | {s.get('mean_reference_spike_duration_s', 0.0):.2f} | {s['selected_method']} | "
            f"{sel['detection_rate'] * 100:.1f}% |"
        )

    methods = sorted(condition_summaries[0]["methods"].keys()) if condition_summaries else []
    lines.extend(["", "## Method comparison", ""])
    header = "| condition | " + " | ".join(methods) + " |"
    divider = "|---|" + "---:|" * len(methods)
    lines.append(header)
    lines.append(divider)
    for s in condition_summaries:
        parts = [f"| {s['condition']}"]
        for method in methods:
            parts.append(f"{s['methods'][method]['detection_rate'] * 100:.1f}%")
        lines.append(" | ".join(parts) + " |")

    total_detected = sum(s["methods"]["raw_abs"]["detected"] for s in condition_summaries)
    total_videos = sum(s["methods"]["raw_abs"]["total"] for s in condition_summaries)
    total_rate = total_detected / max(1, total_videos)
    lines.extend(["", "## Initial conclusion", ""])
    lines.extend(make_factor_conclusion(factor_dir.name, condition_summaries, total_detected, total_videos, total_rate))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_factor_conclusion(factor_name, summaries, total_detected, total_videos, total_rate):
    raw_rates = np.array([s["methods"]["raw_abs"]["detection_rate"] for s in summaries], dtype=float)
    raw_snr = np.array([summary_snr_db(s["methods"]["raw_abs"]) for s in summaries], dtype=float)
    raw_delta = np.array([s["methods"]["raw_abs"]["mean_delta_mean"] for s in summaries], dtype=float)
    raw_gain = np.array([s["methods"]["raw_abs"]["mean_relative_gain"] for s in summaries], dtype=float)
    conditions = [s["condition"] for s in summaries]

    best_idx = int(np.argmax(raw_rates + 1e-3 * raw_snr)) if len(summaries) else 0
    worst_idx = int(np.argmin(raw_rates + 1e-3 * raw_snr)) if len(summaries) else 0
    slope_text = describe_numeric_trend(conditions, raw_rates, raw_snr, raw_delta)

    lines = [
        f"Baseline raw-difference total detection rate is {total_detected}/{total_videos} ({total_rate * 100:.1f}%).",
        f"Best condition by detection/SNR is {conditions[best_idx]}: rate={raw_rates[best_idx] * 100:.1f}%, mean SNR={raw_snr[best_idx]:.2f} dB.",
        f"Worst condition by detection/SNR is {conditions[worst_idx]}: rate={raw_rates[worst_idx] * 100:.1f}%, mean SNR={raw_snr[worst_idx]:.2f} dB.",
        f"Mean active-minus-reference differential strength ranges from {raw_delta.min():.5f} to {raw_delta.max():.5f}.",
        f"Mean relative gain ranges from {raw_gain.min():.2f} to {raw_gain.max():.2f}.",
    ]
    if slope_text:
        lines.append(slope_text)

    alt_better = []
    for s in summaries:
        raw = s["methods"]["raw_abs"]
        sel = s["methods"][s["selected_method"]]
        if s["selected_method"] != "raw_abs" and sel["detection_rate"] > raw["detection_rate"]:
            alt_better.append(f"{s['condition']}->{s['selected_method']} ({sel['detection_rate'] * 100:.1f}%)")
    if alt_better:
        lines.append("For conditions where raw difference is weak, alternative methods improve detection: " + "; ".join(alt_better) + ".")
    if total_rate < 0.5:
        lines.append("Most videos fail under the simple raw-difference detector, so raw frame differencing alone is likely not reliable for this factor.")
    return lines


def summary_snr_db(method_summary):
    if "mean_snr_db" in method_summary:
        return float(method_summary["mean_snr_db"])
    return linear_snr_to_db(float(method_summary.get("mean_snr_p90", 0.0)))


def describe_numeric_trend(conditions, rates, snr, delta):
    parsed = []
    for c in conditions:
        digits = "".join(ch for ch in c if ch.isdigit() or ch == ".")
        if digits:
            try:
                parsed.append(float(digits))
            except ValueError:
                parsed.append(np.nan)
        else:
            parsed.append(np.nan)
    x = np.array(parsed, dtype=float)
    valid = np.isfinite(x)
    if valid.sum() < 3:
        return ""
    order = np.argsort(x[valid])
    xv = x[valid][order]
    rv = rates[valid][order]
    sv = snr[valid][order]
    dv = delta[valid][order]
    rate_corr = safe_corr(xv, rv)
    snr_corr = safe_corr(xv, sv)
    delta_corr = safe_corr(xv, dv)
    return (
        f"Numeric trend estimate: condition value vs detection-rate corr={rate_corr:.2f}, "
        f"vs SNR corr={snr_corr:.2f}, vs differential-strength corr={delta_corr:.2f}."
    )


def safe_corr(x, y):
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def write_global_summary(root, summaries, cfg):
    path = root / "global_detection_summary.md"
    total_videos = sum(s["num_videos"] for s in summaries)
    total_detected = sum(s["methods"]["raw_abs"]["detected"] for s in summaries)
    lines = [
        "# Global detection summary",
        "",
        f"Total videos: {total_videos}",
        f"Raw-difference detections: {total_detected}/{total_videos} ({total_detected / max(1, total_videos) * 100:.1f}%)",
        "",
        "| factor | condition | videos | raw rate | raw SNR dB | selected | selected rate |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for s in summaries:
        raw = s["methods"]["raw_abs"]
        sel = s["methods"][s["selected_method"]]
        lines.append(
            f"| {s['factor']} | {s['condition']} | {s['num_videos']} | "
            f"{raw['detection_rate'] * 100:.1f}% | {summary_snr_db(raw):.2f} | "
            f"{s['selected_method']} | {sel['detection_rate'] * 100:.1f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
