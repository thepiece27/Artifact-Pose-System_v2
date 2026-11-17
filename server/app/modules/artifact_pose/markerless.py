"""M2: marker-free pose after Hung et al., Electronics 14(18):3694, 2025.

Golden: ORB + QuadTree on a simulated stereo pair (slider X moved by the
COMMANDED baseline, 100 mm), Hamming matching with ratio test (0.6),
triangulation with T_right_left = [I | (-s b, 0, 0)] -> landmarks in the
golden (left) camera frame, metres set by the commanded baseline. The sign s
of the slider direction in the camera x axis is chosen by cheirality (the
wrong sign puts the points behind the cameras). The Diamond is never used,
except for an optional logged ``baseline_check``.

Correction: ORB on the current image <-> golden descriptors (ratio 0.6),
EPnP inside RANSAC, reject when <= 5 inliers, then ORB-only pose refinement
with Huber. The result is directly dT (golden camera -> current camera).
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .common import detect_diamond, extract_orb, validate_image
from .geometry import DEFAULT_QUALITY, QualityConfig, camera_params, match_descriptors, triangulate
from .motion import as_T, invert
from .solver import refine_pose_points_only


def scale_intrinsics(K, size_from, size_to) -> np.ndarray:
    """Pixel-centre-consistent scaling of K (D is unchanged: normalised coords)."""
    K = np.asarray(K, dtype=np.float64).copy()
    sx, sy = size_to[0] / size_from[0], size_to[1] / size_from[1]
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return K


def prepare(image, K, resize_to, mask=None):
    """Resize image (+ mask) and K to ``resize_to`` = (w, h); None = full resolution."""
    image = validate_image(image)
    if not resize_to:
        return image, np.asarray(K, dtype=np.float64), mask
    w, h = int(resize_to[0]), int(resize_to[1])
    size_from = (image.shape[1], image.shape[0])
    small = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    m = None if mask is None else cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return small, scale_intrinsics(K, size_from, (w, h)), m


def build_markerless_golden(left, right, K, D, *, baseline_m: float = 0.1, ratio: float = 0.6,
                            resize_to=None, masks=(None, None), quality: QualityConfig = DEFAULT_QUALITY,
                            backend: str = "python", diamond_check: bool = True) -> dict:
    K, D = camera_params(K, D)
    full_size = (left.shape[1], left.shape[0])
    L, Ks, mL = prepare(left, K, resize_to, masks[0])
    R, _, mR = prepare(right, K, resize_to, masks[1])
    _, dl, kl = extract_orb(L, backend=backend, mask=mL)
    _, dr, kr = extract_orb(R, backend=backend, mask=mR)
    matches = match_descriptors(dl, dr, ratio=ratio)
    il = np.array([m.queryIdx for m in matches], dtype=int)
    ir = np.array([m.trainIdx for m in matches], dtype=int)
    best = None
    for sign in (1.0, -1.0):
        T = np.eye(4)
        T[0, 3] = -sign * baseline_m
        tri = triangulate(kl[il], kr[ir], Ks, D, T, quality) if len(il) else {"num_valid": 0}
        if best is None or tri["num_valid"] > best[1]["num_valid"]:
            best = (sign, tri, T)
    sign, tri, T = best
    if tri["num_valid"] < quality.min_orb_inliers:
        raise ValueError(f"M2 golden: only {tri['num_valid']} valid stereo landmarks")
    valid = tri["valid_mask"]
    check = None
    if diamond_check:
        dl_ = detect_diamond(left, K, D, quality)
        dr_ = detect_diamond(right, K, D, quality)
        if dl_ is not None and dr_ is not None:
            Td = as_T(dr_["rvec"], dr_["tvec"]) @ invert(as_T(dl_["rvec"], dl_["tvec"]))
            check = {"diamond_baseline_mm": float(np.linalg.norm(Td[:3, 3]) * 1000),
                     "commanded_baseline_mm": baseline_m * 1000,
                     "diamond_direction_camera": (Td[:3, 3] / max(np.linalg.norm(Td[:3, 3]), 1e-12)).tolist(),
                     "note": "log only; never used by M2"}
    return {"points_3d": tri["points_3d"][valid], "points_2d": kl[il][valid], "descriptors": dl[il][valid],
            "K_used": Ks, "D": D, "image_size_full": full_size,
            "image_size_used": (L.shape[1], L.shape[0]), "resize_to": list(resize_to) if resize_to else None,
            "baseline_m": baseline_m, "slider_x_sign_in_camera": sign, "T_right_left": T.tolist(),
            "n_matches": int(len(matches)), "n_valid": int(valid.sum()), "ratio": ratio,
            "baseline_check": check}


def estimate_markerless(image, golden: dict, K, D, *, ratio: float = 0.6, sigma_px: float = 1.0,
                        huber_delta: float = 2.0, huber: bool = True, min_inliers: int = 6,
                        quality: QualityConfig = DEFAULT_QUALITY, mask=None, backend: str = "python") -> dict:
    """Returns ``{ok, status, dT, refine, n_matches, n_inliers, K_used, blocks, timings_ms}``."""
    t0 = time.perf_counter()
    K, D = camera_params(K, D)
    if (image.shape[1], image.shape[0]) != tuple(golden["image_size_full"]):
        raise ValueError("M2: image size differs from the golden images")
    img, Ks, m = prepare(image, K, golden["resize_to"], mask)
    _, desc, kp = extract_orb(img, backend=backend, mask=m)
    t1 = time.perf_counter()
    matches = match_descriptors(desc, golden["descriptors"], ratio=ratio)
    out = {"ok": False, "status": "too_few_matches", "dT": None, "refine": None, "n_matches": len(matches),
           "n_inliers": 0, "K_used": Ks, "timings_ms": {}}
    if len(matches) < 4:
        out["timings_ms"] = {"features": (t1 - t0) * 1000, "total": (time.perf_counter() - t0) * 1000}
        return out
    ci = np.array([mm.queryIdx for mm in matches], dtype=int)
    gi = np.array([mm.trainIdx for mm in matches], dtype=int)
    xyz, xy = golden["points_3d"][gi], kp[ci]
    ok, rvec, tvec, inl = cv2.solvePnPRansac(xyz, xy, Ks, D, iterationsCount=200,
                                             reprojectionError=quality.max_reprojection_px,
                                             confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    t2 = time.perf_counter()
    n_inl = 0 if inl is None else int(len(inl))
    out["n_inliers"] = n_inl
    if not ok or n_inl < max(min_inliers, 6):
        out["status"] = "pnp_rejected_too_few_inliers"
        out["timings_ms"] = {"features": (t1 - t0) * 1000, "pnp": (t2 - t1) * 1000,
                             "total": (time.perf_counter() - t0) * 1000}
        return out
    inl = inl.ravel()
    fit = refine_pose_points_only(rvec.ravel(), tvec.ravel(), xyz[inl], xy[inl], Ks, D, sigma_px=sigma_px,
                                  huber_delta=huber_delta, huber=huber, quality=quality, min_inliers=min_inliers)
    t3 = time.perf_counter()
    out.update(refine=fit, dT=as_T(fit["rvec"], fit["tvec"]), ok=bool(fit["accepted"]),
               status="accepted" if fit["accepted"] else "refinement_rejected",
               landmarks=(xyz[inl], xy[inl]),
               timings_ms={"features": (t1 - t0) * 1000, "pnp": (t2 - t1) * 1000,
                           "refine": (t3 - t2) * 1000, "total": (t3 - t0) * 1000})
    return out
