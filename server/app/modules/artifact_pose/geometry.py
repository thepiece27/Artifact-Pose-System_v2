"""Validated geometry. T_cw maps Diamond/world coordinates into camera coordinates.

Lengths are metres, observations are distorted pixels unless stated otherwise.
Thresholds are engineering defaults, not measured sensor covariance/accuracy.
"""
from dataclasses import dataclass
import json
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class QualityConfig:
    diamond_sigma_px: float = 0.5
    orb_sigma_px: float = 1.5
    huber_delta: float = 2.0  # whitened residual norm, NOT pixels
    max_diamond_rms_px: float = 2.0
    ambiguity_gap_px: float = 0.15
    ambiguity_rotation_deg: float = 2.0
    max_reprojection_px: float = 2.0
    max_epipolar_px: float = 1.5
    min_parallax_deg: float = 1.0
    min_depth_m: float = 0.05
    max_depth_m: float = 10.0
    min_orb_inliers: int = 10
    max_iterations: int = 100
    translation_tolerance_m: float = 0.01
    rotation_tolerance_deg: float = 1.0
    max_scaled_jacobian_condition: float = 1e8

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.min_depth_m >= self.max_depth_m:
            raise ValueError("min_depth_m must be smaller than max_depth_m")
        for name in ("min_orb_inliers", "max_iterations"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
        if self.min_orb_inliers < 6:
            raise ValueError("At least six ORB inliers are required")


DEFAULT_QUALITY = QualityConfig()


def load_quality(path=None):
    if path is None:
        return DEFAULT_QUALITY
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Quality configuration must be a JSON object")
    try:
        return QualityConfig(**data)
    except TypeError as exc:
        raise ValueError(f"Invalid quality configuration: {exc}") from exc


def points(value, columns, name="points"):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != columns or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite (N,{columns})")
    return np.ascontiguousarray(a)


def vector3(value, name="vector"):
    a = np.asarray(value, dtype=np.float64)
    if a.shape not in ((3,), (3, 1), (1, 3)) or not np.isfinite(a).all():
        raise ValueError(f"{name} must contain exactly three finite values")
    return a.reshape(3)


def camera_params(K, D):
    K = np.asarray(K, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or
            K[0, 0] <= 0 or K[1, 1] <= 0 or
            not np.allclose(K[2], [0, 0, 1]) or
            abs(K[0, 1]) > 1e-12 or abs(K[1, 0]) > 1e-12):
        raise ValueError("K must be a finite zero-skew pinhole camera matrix")
    if D.ndim > 2 or D.size not in (4, 5, 8, 12, 14) or not np.isfinite(D).all():
        raise ValueError("D must contain 4, 5, 8, 12 or 14 finite OpenCV coefficients")
    if D.ndim == 2 and 1 not in D.shape:
        raise ValueError("D must be a vector")
    return np.ascontiguousarray(K), np.ascontiguousarray(D.reshape(-1))


def pose_matrix(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(vector3(rvec, "rvec"))[0]
    T[:3, 3] = vector3(tvec, "tvec")
    return T


def validate_transform(T):
    T = np.asarray(T, dtype=np.float64)
    if (T.shape != (4, 4) or not np.isfinite(T).all() or
            not np.allclose(T[3], [0, 0, 0, 1], atol=1e-9) or
            not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-7) or
            not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-7)):
        raise ValueError("Expected a finite rigid SE(3) transform")
    return T


def invert_pose(T):
    T = validate_transform(T)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def transform_points(T, xyz):
    T = validate_transform(T)
    xyz = points(xyz, 3)
    return xyz @ T[:3, :3].T + T[:3, 3]


def rotation_angle(R):
    return float(np.rad2deg(Rotation.from_matrix(R).magnitude()))


def camera_deviation(r_current, t_current, r_golden, t_golden,
                     quality=DEFAULT_QUALITY):
    current = pose_matrix(r_current, t_current)
    golden = pose_matrix(r_golden, t_golden)
    camera_current, camera_golden = invert_pose(current), invert_pose(golden)
    delta_world = camera_current[:3, 3] - camera_golden[:3, 3]
    # Desired camera pose expressed in CURRENT camera coordinates.
    correction = current @ camera_golden
    angle = rotation_angle(correction[:3, :3])
    distance = float(np.linalg.norm(delta_world))
    return {
        "camera_position_current_world_m": camera_current[:3, 3],
        "camera_position_golden_world_m": camera_golden[:3, 3],
        "translation_world_m": delta_world,
        "translation_correction_world_m": -delta_world,
        "translation_magnitude": distance,
        "rotation_magnitude": angle,
        "rotation_vector_correction_camera_rad": Rotation.from_matrix(
            correction[:3, :3]).as_rotvec(),
        "T_current_camera_desired_camera": correction,
        "within_translation_tolerance": distance <= quality.translation_tolerance_m,
        "within_rotation_tolerance": angle <= quality.rotation_tolerance_deg,
        "within_tolerance": (distance <= quality.translation_tolerance_m and
                             angle <= quality.rotation_tolerance_deg),
    }


def project(xyz, rvec, tvec, K, D):
    xyz = points(xyz, 3)
    if not len(xyz):
        return np.empty((0, 2))
    return cv2.projectPoints(xyz, vector3(rvec), vector3(tvec), K, D)[0].reshape(-1, 2)


def undistort(xy, K, D):
    xy = points(xy, 2)
    if not len(xy):
        return xy.copy()
    # More iterations than the default 5 for stronger calibrated distortion.
    return cv2.undistortPointsIter(xy.reshape(-1, 1, 2), K, D, None, K,
                                  (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                                   50, 1e-12)).reshape(-1, 2)


def square_pose_candidates(obj, corners, K, D, quality=DEFAULT_QUALITY):
    obj, corners = points(obj, 3), points(corners, 2)
    K, D = camera_params(K, D)
    if obj.shape != (4, 3) or corners.shape != (4, 2):
        raise ValueError("IPPE_SQUARE requires exactly four ordered corners")
    h = obj[1, 0]
    expected = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    if h <= 0 or not np.allclose(obj, expected):
        raise ValueError("Object corners do not follow OpenCV IPPE_SQUARE order")
    answer = cv2.solvePnPGeneric(obj, corners, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    candidates = []
    if not answer[0]:
        return candidates
    for r, t in zip(answer[1], answer[2]):
        if not np.isfinite(r).all() or not np.isfinite(t).all():
            continue
        depths = transform_points(pose_matrix(r, t), obj)[:, 2]
        rms = float(np.sqrt(np.mean(np.sum((project(obj, r, t, K, D) - corners)**2, axis=1))))
        # Keep the second physical solution even across the quality cutoff:
        # discarding it early could hide a near-tied planar ambiguity.
        if np.min(depths) > quality.min_depth_m and np.isfinite(rms):
            candidates.append({"rvec": r.ravel(), "tvec": t.ravel(), "rms_px": rms})
    return sorted(candidates, key=lambda c: c["rms_px"])


def pose_is_ambiguous(candidates, quality=DEFAULT_QUALITY):
    if len(candidates) < 2:
        return False
    a, b = candidates[:2]
    gap = b["rms_px"] - a["rms_px"]
    Ra, Rb = cv2.Rodrigues(a["rvec"])[0], cv2.Rodrigues(b["rvec"])[0]
    return bool(gap < quality.ambiguity_gap_px and
                rotation_angle(Ra @ Rb.T) > quality.ambiguity_rotation_deg)


def descriptors(value, name="descriptors"):
    a = np.asarray(value)
    if a.dtype != np.uint8 or a.ndim != 2 or a.shape[1] != 32:
        raise ValueError(f"{name} must be uint8 (N,32) ORB descriptors (WTA_K=2)")
    return np.ascontiguousarray(a)


def match_descriptors(query, train, ratio=0.75, max_hamming=64):
    """Bidirectional ratio test: mutual, unique train and query indices."""
    query, train = descriptors(query), descriptors(train)
    if not 0 < ratio < 1 or not 0 < max_hamming <= 256:
        raise ValueError("Invalid matching thresholds")
    if len(query) < 2 or len(train) < 2:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    def good(a, b):
        return {m[0].queryIdx: m[0] for m in bf.knnMatch(a, b, k=2)
                if len(m) == 2 and m[0].distance < ratio * m[1].distance
                and m[0].distance <= max_hamming}
    forward, reverse = good(query, train), good(train, query)
    return [m for q, m in sorted(forward.items())
            if m.trainIdx in reverse and reverse[m.trainIdx].trainIdx == q]


def triangulate(left, right, K, D, T_right_left, quality=DEFAULT_QUALITY):
    """General calibrated stereo; results in LEFT camera frame, mask aligned to input."""
    left, right = points(left, 2), points(right, 2)
    K, D = camera_params(K, D)
    T = validate_transform(T_right_left)
    if len(left) != len(right):
        raise ValueError("Stereo observation counts differ")
    R, t = T[:3, :3], T[:3, 3]
    if np.linalg.norm(t) < 1e-6:
        raise ValueError("Stereo baseline is degenerate")
    n = len(left)
    xyz, valid = np.full((n, 3), np.nan), np.zeros(n, dtype=bool)
    if not n:
        return {"points_3d": xyz, "valid_mask": valid, "num_valid": 0}
    ul, ur = undistort(left, K, D), undistort(right, K, D)
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    Ki = np.linalg.inv(K)
    F = Ki.T @ tx @ R @ Ki
    xl, xr = np.c_[ul, np.ones(n)], np.c_[ur, np.ones(n)]
    lines_r, lines_l = xl @ F.T, xr @ F
    denom = np.sum(lines_r[:, :2]**2 + lines_l[:, :2]**2, axis=1)
    sampson = np.abs(np.sum(xr * lines_r, axis=1)) / np.sqrt(np.maximum(denom, 1e-30))
    homogeneous = cv2.triangulatePoints(K @ np.eye(3, 4), K @ T[:3], ul.T, ur.T).T
    finite = np.abs(homogeneous[:, 3]) > 1e-12
    xyz[finite] = homogeneous[finite, :3] / homogeneous[finite, 3:4]
    right_xyz = xyz @ R.T + t
    rays_l, rays_r = xl @ Ki.T, xr @ Ki.T @ R
    cos_angle = np.sum(rays_l * rays_r, axis=1) / (
        np.linalg.norm(rays_l, axis=1) * np.linalg.norm(rays_r, axis=1))
    parallax = np.rad2deg(np.arccos(np.clip(cos_angle, -1, 1)))
    safe_xyz = np.where(np.isfinite(xyz), xyz, 0)
    err_l = np.linalg.norm(project(safe_xyz, np.zeros(3), np.zeros(3), K, D) - left, axis=1)
    err_r = np.linalg.norm(project(safe_xyz, cv2.Rodrigues(R)[0], t, K, D) - right, axis=1)
    valid = (finite & np.isfinite(xyz).all(axis=1) &
             (xyz[:, 2] >= quality.min_depth_m) & (xyz[:, 2] <= quality.max_depth_m) &
             (right_xyz[:, 2] >= quality.min_depth_m) & (right_xyz[:, 2] <= quality.max_depth_m) &
             (sampson <= quality.max_epipolar_px) & (parallax >= quality.min_parallax_deg) &
             (err_l <= quality.max_reprojection_px) & (err_r <= quality.max_reprojection_px))
    return {"points_3d": xyz, "valid_mask": valid, "num_valid": int(valid.sum()),
            "sampson_px": sampson, "parallax_deg": parallax,
            "reprojection_left_px": err_l, "reprojection_right_px": err_r}


def verify_correspondences(xyz, xy, rvec, tvec, K, D, quality=DEFAULT_QUALITY):
    """Independent PnP RANSAC intersected with the identified Diamond pose gate."""
    xyz, xy = points(xyz, 3), points(xy, 2)
    if len(xyz) != len(xy):
        raise ValueError("3D/2D counts differ")
    keep = np.zeros(len(xyz), dtype=bool)
    if len(xyz) < quality.min_orb_inliers:
        return keep
    ok, _, _, inliers = cv2.solvePnPRansac(
        xyz, xy, K, D, iterationsCount=200, reprojectionError=quality.max_reprojection_px,
        confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None:
        return keep
    keep[inliers.ravel()] = True
    # A wider Diamond gate accounts for its small spatial extent.
    prior_error = np.linalg.norm(project(xyz, rvec, tvec, K, D) - xy, axis=1)
    keep &= prior_error <= 3 * quality.max_reprojection_px
    keep &= transform_points(pose_matrix(rvec, tvec), xyz)[:, 2] > quality.min_depth_m
    return keep


def refine_pose(rvec, tvec, diamond_xyz, diamond_xy, orb_xyz, orb_xy, K, D,
                quality=DEFAULT_QUALITY, native=None):
    """Pose-only robust least squares, with identical undistortion for both backends.

    Fixed landmarks: this is NOT joint bundle adjustment. No physical covariance
    is claimed. Vector Huber acts on each whitened 2D ORB residual.
    """
    K, D = camera_params(K, D)
    d3, d2 = points(diamond_xyz, 3), points(diamond_xy, 2)
    o3, o2 = points(orb_xyz, 3), points(orb_xy, 2)
    if len(d3) != 4 or len(d2) != 4 or len(o3) != len(o2):
        raise ValueError("Expected 4 Diamond pairs and aligned ORB pairs")
    d2u, o2u = undistort(d2, K, D), undistort(o2, K, D)
    x0 = np.r_[vector3(rvec), vector3(tvec)]
    if np.any(transform_points(pose_matrix(x0[:3], x0[3:]), np.vstack((d3, o3)))[:, 2] <= 1e-6):
        raise ValueError("Nonpositive initial projection depth")
    def residual(x):
        diamond = (project(d3, x[:3], x[3:], K, None) - d2u) / quality.diamond_sigma_px
        orb = (project(o3, x[:3], x[3:], K, None) - o2u) / quality.orb_sigma_px
        norms = np.linalg.norm(orb, axis=1)
        rho = np.where(norms <= quality.huber_delta, norms**2,
                       2 * quality.huber_delta * norms - quality.huber_delta**2)
        scale = np.sqrt(np.maximum(rho, 0)) / np.maximum(norms, 1e-15)
        return np.r_[diamond.ravel(), (orb * scale[:, None]).ravel()]
    initial_cost = float(residual(x0) @ residual(x0))
    if native is None:
        fit = least_squares(residual, x0, method="trf", x_scale="jac",
                            max_nfev=quality.max_iterations, ftol=1e-10,
                            xtol=1e-10, gtol=1e-8)
        x = fit.x
        solver_success, status = bool(fit.success), str(fit.message)
        evaluations = int(fit.nfev)
        optimality = float(fit.optimality)
    else:
        # Native v2 accepts DISTORTED pixels and handles undistortion internally.
        fit = native.hybrid_optimize(x0[:3], x0[3:], d3, d2, o3, o2, K, D,
                                    1 / quality.diamond_sigma_px**2,
                                    1 / quality.orb_sigma_px**2,
                                    quality.huber_delta, quality.max_iterations)
        x = np.r_[fit["rvec"], fit["tvec"]]
        solver_success, status = bool(fit["converged"]), fit["status"]
        evaluations, optimality = int(fit["iterations"]), None
    finite = np.isfinite(x).all()
    final_cost = float(residual(x) @ residual(x)) if finite else float("inf")
    derror = np.linalg.norm(project(d3, x[:3], x[3:], K, D) - d2, axis=1) if finite else np.full(4, np.inf)
    oerror = np.linalg.norm(project(o3, x[:3], x[3:], K, D) - o2, axis=1) if finite else np.full(len(o3), np.inf)
    positive = finite and np.all(transform_points(pose_matrix(x[:3], x[3:]), np.vstack((d3, o3)))[:, 2] > quality.min_depth_m)
    inliers = oerror <= quality.max_reprojection_px
    # Dimensionless column-scaled local Jacobian conditioning, not pose covariance.
    condition, rank = float("inf"), 0
    if finite and positive:
        _, jac = cv2.projectPoints(np.vstack((d3, o3[inliers])), x[:3], x[3:], K, None)
        J = jac[:, :6]
        J[:8] /= quality.diamond_sigma_px
        J[8:] /= quality.orb_sigma_px
        J = J / np.maximum(np.linalg.norm(J, axis=0), 1e-15)
        singular = np.linalg.svd(J, compute_uv=False)
        rank = int(np.count_nonzero(singular > singular[0] * 1e-10))
        condition = float(singular[0] / max(singular[-1], 1e-30))
    accepted = bool(finite and positive and solver_success and
                    rank == 6 and condition <= quality.max_scaled_jacobian_condition and
                    final_cost <= initial_cost + 1e-8 * max(1, initial_cost) and
                    np.sqrt(np.mean(derror**2)) <= quality.max_diamond_rms_px and
                    (not len(o3) or int(inliers.sum()) >= quality.min_orb_inliers))
    return {"rvec": x[:3], "tvec": x[3:], "accepted": accepted,
            "converged": solver_success, "status": status,
            "backend": "g2o" if native is not None else "scipy",
            "initial_robust_cost": initial_cost, "final_robust_cost": final_cost,
            "diamond_rms_px": float(np.sqrt(np.mean(derror**2))),
            "orb_mean_error_px": float(np.mean(oerror[inliers])) if inliers.any() else None,
            "num_orb_inliers": int(inliers.sum()), "num_orb_outliers": int((~inliers).sum()),
            "iterations": evaluations, "optimality": optimality,
            "jacobian_rank": rank, "scaled_jacobian_condition": condition}
