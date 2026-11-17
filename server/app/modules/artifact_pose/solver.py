"""Pose-only robust least squares over stacked observation blocks.

Generalises ``geometry.refine_pose`` (which it reproduces exactly for one
Diamond block + one ORB block, scipy backend) to

* any number of Diamond blocks (M5: one per static frame, same camera pose),
* ORB-only problems (M2, Hung et al.), and
* the ``huber: false`` ablation.

Residuals are whitened by the block sigma; robust blocks use the same vector
Huber as ``refine_pose`` (cost per point = rho(|r|^2), rho(s) = s for
sqrt(s) <= delta, 2 delta sqrt(s) - delta^2 otherwise).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .geometry import (DEFAULT_QUALITY, QualityConfig, camera_params, points, pose_matrix,
                       project, transform_points, undistort, vector3)


@dataclass
class Block:
    xyz: np.ndarray          # (N,3) fixed landmarks, world (or golden-camera) frame
    xy: np.ndarray           # (N,2) DISTORTED pixel observations
    sigma_px: float
    robust: bool
    kind: str = "orb"        # diamond | orb
    frame: int = 0

    def __post_init__(self):
        self.xyz = points(self.xyz, 3)
        self.xy = points(self.xy, 2)
        if len(self.xyz) != len(self.xy):
            raise ValueError("Block 3D/2D counts differ")
        if not self.sigma_px > 0:
            raise ValueError("sigma_px must be positive")


def huber_scale(norms: np.ndarray, delta: float) -> np.ndarray:
    rho = np.where(norms <= delta, norms ** 2, 2 * delta * norms - delta ** 2)
    return np.sqrt(np.maximum(rho, 0)) / np.maximum(norms, 1e-15)


def huber_weights(norms: np.ndarray, delta: float) -> np.ndarray:
    """IRLS weight rho'(s) with s = |r|^2: 1 inside, delta/|r| outside."""
    return np.where(norms <= delta, 1.0, delta / np.maximum(norms, 1e-15))


def whitened_residuals(x: np.ndarray, blocks: list[Block], undist: list[np.ndarray], K) -> list[np.ndarray]:
    return [(project(b.xyz, x[:3], x[3:], K, None) - u) / b.sigma_px for b, u in zip(blocks, undist)]


def stacked_residual(x, blocks, undist, K, huber_delta: float, huber: bool) -> np.ndarray:
    out = []
    for b, r in zip(blocks, whitened_residuals(x, blocks, undist, K)):
        if b.robust and huber and len(r):
            r = r * huber_scale(np.linalg.norm(r, axis=1), huber_delta)[:, None]
        out.append(r.ravel())
    return np.concatenate(out) if out else np.empty(0)


def refine_pose_blocks(rvec, tvec, blocks: list[Block], K, D, quality: QualityConfig = DEFAULT_QUALITY, *,
                       huber_delta: float | None = None, huber: bool = True,
                       min_orb_inliers: int | None = None) -> dict:
    """Same contract/keys as ``geometry.refine_pose`` (scipy backend)."""
    K, D = camera_params(K, D)
    huber_delta = quality.huber_delta if huber_delta is None else huber_delta
    min_orb_inliers = quality.min_orb_inliers if min_orb_inliers is None else min_orb_inliers
    blocks = [b for b in blocks if len(b.xyz)]
    if not blocks:
        raise ValueError("No observations")
    undist = [undistort(b.xy, K, D) for b in blocks]
    x0 = np.r_[vector3(rvec), vector3(tvec)]
    all_xyz = np.vstack([b.xyz for b in blocks])
    if np.any(transform_points(pose_matrix(x0[:3], x0[3:]), all_xyz)[:, 2] <= 1e-6):
        raise ValueError("Nonpositive initial projection depth")

    def residual(x):
        return stacked_residual(x, blocks, undist, K, huber_delta, huber)

    initial_cost = float(residual(x0) @ residual(x0))
    fit = least_squares(residual, x0, method="trf", x_scale="jac", max_nfev=quality.max_iterations,
                        ftol=1e-10, xtol=1e-10, gtol=1e-8)
    x = fit.x
    finite = bool(np.isfinite(x).all())
    final_cost = float(residual(x) @ residual(x)) if finite else float("inf")
    positive = finite and bool(np.all(transform_points(pose_matrix(x[:3], x[3:]), all_xyz)[:, 2] > quality.min_depth_m))
    errors = [np.linalg.norm(project(b.xyz, x[:3], x[3:], K, D) - b.xy, axis=1) if finite
              else np.full(len(b.xyz), np.inf) for b in blocks]
    diamond_err = [e for b, e in zip(blocks, errors) if b.kind == "diamond"]
    orb_err = [e for b, e in zip(blocks, errors) if b.kind != "diamond"]
    derr = np.concatenate(diamond_err) if diamond_err else np.empty(0)
    oerr = np.concatenate(orb_err) if orb_err else np.empty(0)
    inliers = oerr <= quality.max_reprojection_px
    # Per-frame Diamond RMS must pass the same gate as refine_pose.
    diamond_rms_frames = [float(np.sqrt(np.mean(e ** 2))) for e in diamond_err]
    condition, rank = float("inf"), 0
    if finite and positive:
        J_rows = []
        for b, e in zip(blocks, errors):
            keep = np.ones(len(b.xyz), bool) if b.kind == "diamond" else e <= quality.max_reprojection_px
            if not keep.any():
                continue
            import cv2
            _, jac = cv2.projectPoints(b.xyz[keep], x[:3], x[3:], K, None)
            J_rows.append(jac[:, :6] / b.sigma_px)
        if J_rows:
            J = np.vstack(J_rows)
            J = J / np.maximum(np.linalg.norm(J, axis=0), 1e-15)
            singular = np.linalg.svd(J, compute_uv=False)
            rank = int(np.count_nonzero(singular > singular[0] * 1e-10))
            condition = float(singular[0] / max(singular[-1], 1e-30))
    accepted = bool(finite and positive and fit.success and rank == 6
                    and condition <= quality.max_scaled_jacobian_condition
                    and final_cost <= initial_cost + 1e-8 * max(1, initial_cost)
                    and all(r <= quality.max_diamond_rms_px for r in diamond_rms_frames)
                    and (not len(oerr) or int(inliers.sum()) >= min_orb_inliers))
    return {"rvec": x[:3], "tvec": x[3:], "accepted": accepted, "converged": bool(fit.success),
            "status": str(fit.message), "backend": "scipy",
            "initial_robust_cost": initial_cost, "final_robust_cost": final_cost,
            "diamond_rms_px": float(np.sqrt(np.mean(derr ** 2))) if len(derr) else None,
            "diamond_rms_px_frames": diamond_rms_frames,
            "orb_mean_error_px": float(np.mean(oerr[inliers])) if inliers.any() else None,
            "orb_rms_px": float(np.sqrt(np.mean(oerr[inliers] ** 2))) if inliers.any() else None,
            "num_orb_inliers": int(inliers.sum()), "num_orb_outliers": int((~inliers).sum()),
            "iterations": int(fit.nfev), "optimality": float(fit.optimality),
            "jacobian_rank": rank, "scaled_jacobian_condition": condition,
            "num_residuals": int(sum(2 * len(b.xyz) for b in blocks))}


def refine_pose_points_only(rvec, tvec, xyz, xy, K, D, *, sigma_px: float, huber_delta: float,
                            huber: bool = True, quality: QualityConfig = DEFAULT_QUALITY,
                            min_inliers: int = 6) -> dict:
    """M2: ORB-only pose refinement (Hung et al.: EPnP init + pose-only g2o)."""
    block = Block(xyz, xy, sigma_px, robust=True, kind="orb")
    return refine_pose_blocks(rvec, tvec, [block], K, D, quality, huber_delta=huber_delta, huber=huber,
                              min_orb_inliers=min_inliers)
