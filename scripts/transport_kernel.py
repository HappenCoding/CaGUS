"""Geometry-conditioned light transport primitives used by the CaGUS scripts.

The implementation follows the discrete form of the model in Section 2 and
Appendix A: a visible-wall perturbation is projected with the adjoint of a
geometry-conditioned transport kernel.  The response is a hidden-space
activity representation, not an image reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


EPS = 1e-8


@dataclass(frozen=True)
class Segment:
    """A 2-D occluding segment in the horizontal/vertical transport plane."""

    start: tuple[float, float]
    end: tuple[float, float]


def grid_points(bounds: tuple[float, float, float, float], rows: int, cols: int) -> np.ndarray:
    """Return cell-centre points with shape ``(rows * cols, 2)``."""
    x0, x1, z0, z1 = map(float, bounds)
    xs = np.linspace(x0, x1, cols, endpoint=False) + (x1 - x0) / (2.0 * cols)
    zs = np.linspace(z0, z1, rows, endpoint=False) + (z1 - z0) / (2.0 * rows)
    xx, zz = np.meshgrid(xs, zs)
    return np.column_stack((xx.ravel(), zz.ravel())).astype(np.float32)


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    return (
        min(a[0], c[0]) - EPS <= b[0] <= max(a[0], c[0]) + EPS
        and min(a[1], c[1]) - EPS <= b[1] <= max(a[1], c[1]) + EPS
    )


def segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    """Closed line-segment intersection test, including collinear contact."""
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True
    return (
        (abs(o1) <= EPS and _on_segment(a, c, b))
        or (abs(o2) <= EPS and _on_segment(a, d, b))
        or (abs(o3) <= EPS and _on_segment(c, a, d))
        or (abs(o4) <= EPS and _on_segment(c, b, d))
    )


def visibility_matrix(wall_points: np.ndarray, hidden_points: np.ndarray, occluders: Iterable[Segment]) -> np.ndarray:
    """Compute V_G(p, x) for a 2-D planar approximation of the scene."""
    segments = [(np.asarray(item.start, dtype=np.float32), np.asarray(item.end, dtype=np.float32)) for item in occluders]
    visible = np.ones((wall_points.shape[0], hidden_points.shape[0]), dtype=np.float32)
    if not segments:
        return visible
    for p_idx, wall_point in enumerate(wall_points):
        for x_idx, hidden_point in enumerate(hidden_points):
            if any(segments_intersect(hidden_point, wall_point, s0, s1) for s0, s1 in segments):
                visible[p_idx, x_idx] = 0.0
    return visible


def build_analytical_kernel(
    wall_points: np.ndarray,
    hidden_points: np.ndarray,
    occluders: Iterable[Segment] = (),
    wall_reflectance: float | np.ndarray = 1.0,
    attenuation_power: float = 2.0,
) -> np.ndarray:
    """Construct a discrete kernel K_G = L_G V_G rho for a planar scene.

    The simple inverse-distance term is deliberately explicit: it is a first
    order transport approximation and should be replaced by calibrated kernels
    in scenes dominated by higher-order reflection or scattering.
    """
    wall_points = np.asarray(wall_points, dtype=np.float32)
    hidden_points = np.asarray(hidden_points, dtype=np.float32)
    distance = np.linalg.norm(wall_points[:, None, :] - hidden_points[None, :, :], axis=2)
    attenuation = 1.0 / np.maximum(distance, 1e-3) ** float(attenuation_power)
    visibility = visibility_matrix(wall_points, hidden_points, occluders)
    rho = np.asarray(wall_reflectance, dtype=np.float32)
    if rho.ndim == 0:
        rho = np.full((wall_points.shape[0], 1), float(rho), dtype=np.float32)
    else:
        rho = rho.reshape(-1, 1)
        if rho.shape[0] != wall_points.shape[0]:
            raise ValueError("wall_reflectance must be scalar or have one value per wall point")
    return (attenuation * visibility * rho).astype(np.float32)


def normalize_dictionary(dictionary: np.ndarray) -> np.ndarray:
    """Column-normalize a transport dictionary before matched projection."""
    dictionary = np.asarray(dictionary, dtype=np.float32)
    return dictionary / np.maximum(np.linalg.norm(dictionary, axis=0, keepdims=True), EPS)


def adjoint_project(dictionary: np.ndarray, perturbation: np.ndarray, normalize_columns: bool = True) -> np.ndarray:
    """Compute R_G = S_G^T D_G for one or more visible-space perturbations."""
    kernel = normalize_dictionary(dictionary) if normalize_columns else np.asarray(dictionary, dtype=np.float32)
    delta = np.asarray(perturbation, dtype=np.float32)
    if delta.ndim == 1:
        if delta.shape[0] != kernel.shape[0]:
            raise ValueError("Perturbation length does not match the visible-space kernel dimension")
        return kernel.T @ delta
    if delta.ndim == 2:
        if delta.shape[1] != kernel.shape[0]:
            raise ValueError("Perturbation width does not match the visible-space kernel dimension")
        return delta @ kernel
    raise ValueError("perturbation must have shape (P,) or (T, P)")


def peak_to_sidelobe_ratio(response: np.ndarray, exclusion_radius: int = 1) -> tuple[int, float]:
    """Return the strongest response index and its local PSR estimate."""
    values = np.asarray(response, dtype=np.float32).ravel()
    peak = int(np.argmax(values))
    keep = np.ones(values.size, dtype=bool)
    keep[max(0, peak - exclusion_radius) : min(values.size, peak + exclusion_radius + 1)] = False
    sidelobes = values[keep]
    if sidelobes.size == 0:
        return peak, float("inf")
    return peak, float((values[peak] - np.mean(sidelobes)) / max(np.std(sidelobes), EPS))


def fuse_responses(responses: Iterable[np.ndarray], weights: Iterable[float] | None = None) -> np.ndarray:
    """Fuse independent projections after per-response scale normalization."""
    arrays = [np.asarray(item, dtype=np.float32) for item in responses]
    if not arrays:
        raise ValueError("At least one response map is required")
    if weights is None:
        weights = [1.0] * len(arrays)
    total = np.zeros_like(arrays[0], dtype=np.float32)
    weight_sum = 0.0
    for response, weight in zip(arrays, weights):
        if response.shape != total.shape:
            raise ValueError("All response maps must have the same shape")
        scale = float(np.percentile(np.abs(response), 95))
        total += float(weight) * response / max(scale, EPS)
        weight_sum += float(weight)
    return total / max(weight_sum, EPS)
