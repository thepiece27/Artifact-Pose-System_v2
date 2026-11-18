"""Stereo baseline DIRECTION from ORB matches for a pure slider translation (R6).

Between the golden left and right images only the slider moved and the servos
stayed still, so R = I. For normalised (undistorted) rays x1, x2 of a match the
pure-translation epipolar constraint is

    x2^T [t]x x1 = t . (x1 x x2) = 0.

The direction t_hat is estimated by RANSAC with 2-point minimal samples
(t_hat = normalise((x1a x x2a) x (x1b x x2b))), inliers by symmetric epipolar
distance in pixels, then refined by SVD on the inliers (smallest right
singular vector of the stacked (x1 x x2)^T). The sign is chosen by
cheirality. The metric length comes from the COMMANDED baseline.
"""
from __future__ import annotations

import numpy as np

from .geometry import DEFAULT_QUALITY, QualityConfig, camera_params, invert_pose, pose_matrix, refine_pose, triangulate, undistort


def normalized_rays(xy, K, D) -> np.ndarray:
    K, D = camera_params(K, D)
    u = undistort(np.asarray(xy, float).reshape(-1, 2), K, D)
    return np.c_[u, np.ones(len(u))] @ np.linalg.inv(K).T


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def epipolar_distance_px(t: np.ndarray, x1: np.ndarray, x2: np.ndarray, focal_px: float) -> np.ndarray:
    """Symmetric point-to-epipolar-line distance (pixels) for E = [t]x."""
    num = np.abs(np.cross(x1, x2) @ t)
    l2 = np.cross(t, x1)            # line in image 2
    l1 = np.cross(t, x2)            # line in image 1 (sign irrelevant)
    d2 = num / np.maximum(np.linalg.norm(l2[:, :2], axis=1), 1e-12)
    d1 = num / np.maximum(np.linalg.norm(l1[:, :2], axis=1), 1e-12)
    return 0.5 * (d1 + d2) * focal_px


def estimate_translation_direction(x1: np.ndarray, x2: np.ndarray, *, focal_px: float,
                                   max_epipolar_px: float = DEFAULT_QUALITY.max_epipolar_px,
                                   iterations: int = 1000, seed: int = 0) -> dict:
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    n = len(x1)
    if n < 8:
        raise ValueError(f"commanded_epipolar: only {n} matches")
    a = np.cross(x1, x2)
    rng = np.random.default_rng(seed)
    best_t, best_mask = None, None
    for _ in range(iterations):
        i, j = rng.choice(n, size=2, replace=False)
        t = np.cross(a[i], a[j])
        if np.linalg.norm(t) < 1e-12:
            continue
        t = _unit(t)
        mask = epipolar_distance_px(t, x1, x2, focal_px) <= max_epipolar_px
        if best_mask is None or mask.sum() > best_mask.sum():
            best_t, best_mask = t, mask
    if best_mask is None or best_mask.sum() < 8:
        raise ValueError("commanded_epipolar: RANSAC found no consensus direction")
    _, _, vt = np.linalg.svd(a[best_mask], full_matrices=False)
    t = _unit(vt[-1])
    mask = epipolar_distance_px(t, x1, x2, focal_px) <= max_epipolar_px
    d = epipolar_distance_px(t, x1[mask], x2[mask], focal_px)
    return {"t_dir": t, "inlier_mask": mask, "n_inliers": int(mask.sum()), "n_matches": int(n),
            "rms_epipolar_px": float(np.sqrt(np.mean(d ** 2))) if len(d) else None}


def transform_with_cheirality(t_dir: np.ndarray, baseline_m: float, kl, kr, K, D,
                              quality: QualityConfig = DEFAULT_QUALITY) -> tuple[np.ndarray, dict]:
    """T_right_left = [I | +/- b t_dir], sign with more points in front of both cameras."""
    best = None
    for sign in (1.0, -1.0):
        T = np.eye(4)
        T[:3, 3] = sign * baseline_m * np.asarray(t_dir, float)
        tri = triangulate(kl, kr, K, D, T, quality)
        if best is None or tri["num_valid"] > best[1]["num_valid"]:
            best = (T, tri)
    return best


def axis_angle_deg(u, v) -> float:
    """Angle between two directions, ignoring the sign (axes)."""
    u, v = _unit(np.asarray(u, float)), _unit(np.asarray(v, float))
    return float(np.rad2deg(np.arccos(np.clip(abs(float(u @ v)), 0.0, 1.0))))


def diamond_stereo_transform(left_diamond: dict | None, right_image, K, D, quality: QualityConfig = DEFAULT_QUALITY,
                             native=None) -> np.ndarray | None:
    """T_right_left from the two Diamond poses (the cita-baseline estimate), or None."""
    from .common import detect_diamond
    if left_diamond is None:
        return None
    rd = detect_diamond(right_image, K, D, quality)
    if rd is None or rd["ambiguous"]:
        return None
    fit = refine_pose(rd["rvec"], rd["tvec"], rd["obj_pts"], rd["corners"], np.empty((0, 3)), np.empty((0, 2)),
                      K, D, quality, native=native)
    if not fit["accepted"]:
        return None
    return pose_matrix(fit["rvec"], fit["tvec"]) @ invert_pose(pose_matrix(left_diamond["rvec"], left_diamond["tvec"]))
