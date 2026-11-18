"""Variance-component estimation (VCE) of the Diamond / ORB pixel noise (R2).

Helmert / Foerstner simplified, on the SAME definition for both groups:
both sigmas are the scale of the post-fit residuals of the hybrid pose-only
problem, so systematic errors (print errors, residual distortion, corner
localisation bias for the Diamond; landmark triangulation error for ORB) are
included in both. The static-frame jitter only measures temporal noise and is
kept as side information (``diamond_jitter_sigma``).

For each iteration, every frame is solved with the current sigmas
(``solver.refine_pose_blocks``, Huber on ORB). At the solution, with the
whitened Jacobian J (m x 6) and the IRLS weights W, the hat matrix is
H = J (J^T W J)^-1 J^T W; the redundancy of group g is
red_g = n_g - sum_{i in g} H_ii (n_g = number of 2D residual components) and

    sigma_g^2 <- sigma_g^2 * (sum_frames r_g^T W_g r_g) / (sum_frames red_g)

with r_g the WHITENED residuals. ORB components only count when their 2D
whitened norm is <= ``orb_inlier_whitened`` (outlier rejection).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import DEFAULT_QUALITY, QualityConfig, camera_params, project, undistort
from .solver import Block, huber_weights, refine_pose_blocks

MAD_K = 1.4826


@dataclass
class FrameObservation:
    diamond_xyz: np.ndarray          # (4,3) Diamond object points (world)
    diamond_xy: np.ndarray           # (4,2) detected corners (distorted px)
    orb_xyz: np.ndarray              # (N,3) verified golden landmarks
    orb_xy: np.ndarray               # (N,2) matched keypoints (distorted px)
    rvec0: np.ndarray                # initial pose (Diamond IPPE)
    tvec0: np.ndarray
    session_id: str = ""
    frame_id: str = ""
    meta: dict = field(default_factory=dict)


def diamond_jitter_sigma(corner_frames) -> float | None:
    """1.4826 MAD of corner coordinates around the per-corner median over
    STATIC frames (temporal noise only; x and y pooled)."""
    c = np.asarray(corner_frames, dtype=np.float64)
    if c.ndim != 3 or len(c) < 3:
        return None
    dev = (c - np.median(c, axis=0, keepdims=True)).ravel()
    return float(MAD_K * np.median(np.abs(dev - np.median(dev))))


def frame_contribution(obs: FrameObservation, sigma_d: float, sigma_o: float, K, D, *,
                       quality: QualityConfig = DEFAULT_QUALITY, huber_delta: float = 2.0,
                       orb_inlier_whitened: float = 3.0) -> dict | None:
    """Solve one frame with the current sigmas; return the VCE sums."""
    K, D = camera_params(K, D)
    blocks = [Block(obs.diamond_xyz, obs.diamond_xy, sigma_d, False, "diamond")]
    if len(obs.orb_xyz):
        blocks.append(Block(obs.orb_xyz, obs.orb_xy, sigma_o, True, "orb"))
    try:
        fit = refine_pose_blocks(obs.rvec0, obs.tvec0, blocks, K, D, quality, huber_delta=huber_delta, huber=True,
                                 min_orb_inliers=0)
    except ValueError:
        return None
    r, t = np.asarray(fit["rvec"], float), np.asarray(fit["tvec"], float)
    if not (np.isfinite(r).all() and np.isfinite(t).all()):
        return None
    d_u = undistort(obs.diamond_xy, K, D)
    rd = (project(obs.diamond_xyz, r, t, K, None) - d_u) / sigma_d                     # (4,2)
    xyz_rows, scale_rows, groups, w_rows, res_rows = [obs.diamond_xyz], [np.full(4, sigma_d)], [np.zeros(4, int)], \
        [np.ones(4)], [rd]
    n_orb_used = 0
    if len(obs.orb_xyz):
        o_u = undistort(obs.orb_xy, K, D)
        ro = (project(obs.orb_xyz, r, t, K, None) - o_u) / sigma_o
        norms = np.linalg.norm(ro, axis=1)
        keep = norms <= orb_inlier_whitened
        n_orb_used = int(keep.sum())
        if n_orb_used:
            xyz_rows.append(obs.orb_xyz[keep])
            scale_rows.append(np.full(n_orb_used, sigma_o))
            groups.append(np.ones(n_orb_used, int))
            w_rows.append(huber_weights(norms[keep], huber_delta))
            res_rows.append(ro[keep])
    xyz = np.vstack(xyz_rows)
    _, jac = cv2.projectPoints(xyz, r, t, K, None)
    J = jac[:, :6] / np.repeat(np.concatenate(scale_rows), 2)[:, None]       # whitened, rows x0,y0,x1,...
    w = np.repeat(np.concatenate(w_rows), 2)
    g = np.repeat(np.concatenate(groups), 2)
    res = np.vstack(res_rows).ravel()
    A = J.T @ (w[:, None] * J)
    if np.linalg.matrix_rank(A) < 6:
        return None
    M = np.linalg.inv(A)
    h = np.einsum("ij,jk,ik->i", J, M, J) * w                                 # diag of H = J M J^T W
    return {"num_d": float(np.sum(w[g == 0] * res[g == 0] ** 2)), "red_d": float(np.sum(g == 0) - h[g == 0].sum()),
            "num_o": float(np.sum(w[g == 1] * res[g == 1] ** 2)), "red_o": float(np.sum(g == 1) - h[g == 1].sum()),
            "n_d": int(np.sum(g == 0)), "n_o": int(np.sum(g == 1)), "n_orb_used": n_orb_used,
            "rvec": r, "tvec": t, "accepted": bool(fit["accepted"])}


def estimate_vce(frames: list[FrameObservation], K, D, *, sigma_d0: float = 0.5, sigma_o0: float = 1.5,
                 max_iter: int = 20, rel_tol: float = 0.01, quality: QualityConfig = DEFAULT_QUALITY,
                 huber_delta: float = 2.0, orb_inlier_whitened: float = 3.0,
                 min_diamond_redundancy: float = 20.0) -> dict:
    sd, so = float(sigma_d0), float(sigma_o0)
    history = [{"iteration": 0, "sigma_d_px": sd, "sigma_o_px": so}]
    converged = False
    last: list[dict] = []
    for it in range(1, max_iter + 1):
        contribs = []
        for idx, f in enumerate(frames):
            c = frame_contribution(f, sd, so, K, D, quality=quality, huber_delta=huber_delta,
                                   orb_inlier_whitened=orb_inlier_whitened)
            if c:
                c["index"] = idx
                contribs.append(c)
        if not contribs:
            raise ValueError("VCE: no frame could be solved")
        last = contribs
        num_d, red_d = sum(c["num_d"] for c in contribs), sum(c["red_d"] for c in contribs)
        num_o, red_o = sum(c["num_o"] for c in contribs), sum(c["red_o"] for c in contribs)
        new_d = sd * np.sqrt(num_d / red_d) if red_d > 0 else sd
        new_o = so * np.sqrt(num_o / red_o) if red_o > 0 else so
        change = max(abs(new_d - sd) / sd, abs(new_o - so) / so)
        sd, so = float(new_d), float(new_o)
        history.append({"iteration": it, "sigma_d_px": sd, "sigma_o_px": so, "redundancy_d": red_d,
                        "redundancy_o": red_o, "n_frames": len(contribs), "max_rel_change": float(change)})
        if change < rel_tol:
            converged = True
            break
    red_d_total = history[-1].get("redundancy_d", 0.0)
    warning = None
    if red_d_total < min_diamond_redundancy:
        warning = (f"total Diamond redundancy {red_d_total:.1f} < {min_diamond_redundancy:.0f}: "
                   "sigma_d is poorly determined, capture more E0.3 frames")
    return {"sigma_d_px": sd, "sigma_o_px": so, "converged": converged, "n_iterations": len(history) - 1,
            "iterations": history, "redundancy_d": red_d_total, "redundancy_o": history[-1].get("redundancy_o"),
            "n_frames": len(last), "n_residuals_d": int(sum(c["n_d"] for c in last)),
            "n_residuals_o": int(sum(c["n_o"] for c in last)), "warning": warning,
            "final_poses": [(c["index"], c["rvec"], c["tvec"]) for c in last]}
