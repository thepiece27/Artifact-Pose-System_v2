"""First-order pose covariance (I1, CITA Appendix A.3).

At the solution T* (camera <- world) the whitened residuals r(xi) of the pose
``exp(xi) T*`` are differentiated numerically (central differences) w.r.t. the
left perturbation xi = [omega (rad), v (m)]. With IRLS weights of the Huber
kernel W = diag(rho'(s_i)):

    Sigma_xi      = (J^T W J)^-1                    (a priori)
    Sigma_xi_post = s^2 Sigma_xi, s^2 = cost / (m - 6)   (a posteriori)
    Sigma_c       = G Sigma_xi G^T, G = dc/dxi           (camera centre, golden frame)
    Sigma_theta   = H Sigma_xi H^T, H = dtheta/dxi       (rotation vector of dR)

This is a linearisation; it is only meaningful if the noise model (sigmas) is
right. It is computed in Python from the final pose whatever the optimisation
backend was (g2o or scipy), so both backends are comparable.
"""
from __future__ import annotations

import numpy as np
import cv2

from .geometry import camera_params, project, undistort
from .motion import camera_center_in_golden, invert, rotvec, se3_left_perturb
from .solver import Block, huber_scale, huber_weights


def _pose_vec(T: np.ndarray) -> np.ndarray:
    return np.r_[cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]]


def _numeric_jacobian(fun, n_out: int, h: float) -> np.ndarray:
    J = np.empty((n_out, 6))
    for k in range(6):
        e = np.zeros(6)
        e[k] = h
        J[:, k] = (fun(e) - fun(-e)) / (2 * h)
    return J


def pose_covariance(T_cw: np.ndarray, blocks: list[Block], K, D, *, huber_delta: float, huber: bool = True,
                    T_gw: np.ndarray | None = None, max_reprojection_px: float | None = None,
                    step: float = 1e-6) -> dict:
    """Covariance of the pose ``T_cw`` and of (c, theta) relative to ``T_gw``.

    ``T_gw`` = golden camera <- world. When None, ``T_cw`` is already the
    relative motion dT (M2: landmarks live in the golden camera frame).
    ORB points whose final reprojection error exceeds ``max_reprojection_px``
    are excluded (they are outliers for the pose acceptance test too).
    """
    K, D = camera_params(K, D)
    T_cw = np.asarray(T_cw, dtype=np.float64)
    T_ref_inv = invert(np.asarray(T_gw, dtype=np.float64)) if T_gw is not None else np.eye(4)
    used, undist = [], []
    x0 = _pose_vec(T_cw)
    for b in blocks:
        if not len(b.xyz):
            continue
        keep = np.ones(len(b.xyz), bool)
        if b.kind != "diamond" and max_reprojection_px is not None:
            keep = np.linalg.norm(project(b.xyz, x0[:3], x0[3:], K, D) - b.xy, axis=1) <= max_reprojection_px
        if not keep.any():
            continue
        bb = Block(b.xyz[keep], b.xy[keep], b.sigma_px, b.robust, b.kind, b.frame)
        used.append(bb)
        undist.append(undistort(bb.xy, K, D))
    if not used:
        return {"ok": False, "reason": "no_observations"}

    def whitened(xi):
        T = se3_left_perturb(T_cw, xi)
        x = _pose_vec(T)
        return np.concatenate([((project(b.xyz, x[:3], x[3:], K, None) - u) / b.sigma_px).ravel()
                               for b, u in zip(used, undist)])

    r0 = whitened(np.zeros(6))
    m = len(r0)
    J = _numeric_jacobian(whitened, m, step)
    # IRLS weights (per 2D residual, both components) and robust cost.
    w = np.ones(m)
    cost = 0.0
    offset = 0
    for b in used:
        n = 2 * len(b.xyz)
        r = r0[offset:offset + n].reshape(-1, 2)
        norms = np.linalg.norm(r, axis=1)
        if b.robust and huber:
            w[offset:offset + n] = np.repeat(huber_weights(norms, huber_delta), 2)
            cost += float(np.sum((r * huber_scale(norms, huber_delta)[:, None]) ** 2))
        else:
            cost += float(np.sum(r ** 2))
        offset += n
    info = J.T @ (w[:, None] * J)
    singular = np.linalg.svd(info, compute_uv=False)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-12)) if singular[0] > 0 else 0
    cov = np.linalg.pinv(info) if rank < 6 else np.linalg.inv(info)
    dof = m - 6
    s2 = cost / dof if dof > 0 else float("nan")
    cov_post = s2 * cov if dof > 0 else np.full((6, 6), np.nan)

    def center_and_theta(xi):
        dT = se3_left_perturb(T_cw, xi) @ T_ref_inv
        return np.r_[camera_center_in_golden(dT), rotvec(dT[:3, :3])]

    GH = _numeric_jacobian(center_and_theta, 6, step)
    G, H = GH[:3], GH[3:]
    cov_c, cov_th = G @ cov @ G.T, H @ cov @ H.T
    cov_c_post, cov_th_post = G @ cov_post @ G.T, H @ cov_post @ H.T
    sd = lambda C, scale: (np.sqrt(np.clip(np.diag(C), 0, None)) * scale).tolist()  # noqa: E731
    return {
        "ok": rank == 6, "rank": rank, "num_residuals": m, "dof": dof, "robust_cost": cost, "s2": s2,
        "cov_xi": cov.tolist(), "cov_xi_post": cov_post.tolist(),
        "cov_center_mm2": (cov_c * 1e6).tolist(), "cov_center_mm2_post": (cov_c_post * 1e6).tolist(),
        "cov_rot_deg2": (cov_th * (180 / np.pi) ** 2).tolist(),
        "sigma_center_mm": sd(cov_c, 1000.0), "sigma_center_mm_post": sd(cov_c_post, 1000.0),
        "sigma_rot_deg": sd(cov_th, 180 / np.pi), "sigma_rot_deg_post": sd(cov_th_post, 180 / np.pi),
        "xi_order": "omega_rad(3), v_m(3); left perturbation of T_cw",
    }
