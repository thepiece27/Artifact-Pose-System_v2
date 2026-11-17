"""Canonical relative motion and pose errors (CITA Appendix A.1/A.2).

``dT = T_cw(current) @ inv(T_cw(golden))`` maps golden-camera coordinates to
current-camera coordinates. ``c = -dR^T dt`` is the current camera centre in
the golden camera frame (metres) and ``theta = rotvec(dR)`` (radians).
Pure numpy/OpenCV; no I/O.
"""
from __future__ import annotations

import numpy as np
import cv2


def as_T(rvec=None, tvec=None, T=None) -> np.ndarray:
    if T is not None:
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError("T must be 4x4")
        return T
    out = np.eye(4)
    out[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))[0]
    out[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return out


def invert(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def relative_motion(T_cw_golden: np.ndarray, T_cw_current: np.ndarray) -> np.ndarray:
    """dT (golden camera -> current camera)."""
    return np.asarray(T_cw_current, dtype=np.float64) @ invert(np.asarray(T_cw_golden, dtype=np.float64))


def camera_center_in_golden(dT: np.ndarray) -> np.ndarray:
    return -dT[:3, :3].T @ dT[:3, 3]


def rotvec(R: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(R, dtype=np.float64))[0].reshape(3)


def motion_summary(dT: np.ndarray) -> dict:
    """Boundary units (mm / degrees) for logs."""
    c = camera_center_in_golden(dT)
    th = rotvec(dT[:3, :3])
    return {"c_mm": (c * 1000).tolist(), "c_norm_mm": float(np.linalg.norm(c) * 1000),
            "rotvec_deg": np.rad2deg(th).tolist(), "rot_deg": float(np.rad2deg(np.linalg.norm(th))),
            "delta_R": dT[:3, :3].tolist(), "delta_t_m": dT[:3, 3].tolist()}


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    cos = (np.trace(np.asarray(R_est) @ np.asarray(R_gt).T) - 1.0) / 2.0
    return float(np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0))))


def pose_errors(dT_est: np.ndarray, dT_gt: np.ndarray) -> dict:
    """A.2: e_t (mm), e_R (deg) and signed per-axis error of c (mm, golden camera frame)."""
    c_est, c_gt = camera_center_in_golden(dT_est), camera_center_in_golden(dT_gt)
    diff_mm = (c_est - c_gt) * 1000
    return {"e_t_mm": float(np.linalg.norm(diff_mm)),
            "e_R_deg": rotation_error_deg(dT_est[:3, :3], dT_gt[:3, :3]),
            "c_err_mm": diff_mm.tolist()}


def se3_left_perturb(T: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """exp(xi) @ T with xi = [omega (rad), v (m)]; first-order equal to SE(3) exp."""
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    R = cv2.Rodrigues(xi[:3].reshape(3, 1))[0]
    out = np.eye(4)
    out[:3, :3] = R @ T[:3, :3]
    out[:3, 3] = R @ T[:3, 3] + xi[3:]
    return out


def average_poses(Ts: list[np.ndarray]) -> np.ndarray:
    """Chordal L2 rotation mean (quaternion eigenvector) + arithmetic translation mean."""
    if not Ts:
        raise ValueError("No poses to average")
    from scipy.spatial.transform import Rotation
    quats = Rotation.from_matrix(np.stack([T[:3, :3] for T in Ts])).as_quat()
    M = quats.T @ quats
    q = np.linalg.eigh(M)[1][:, -1]
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(q).as_matrix()
    out[:3, 3] = np.mean([T[:3, 3] for T in Ts], axis=0)
    return out
