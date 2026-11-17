"""Unified pose interface for the CITA branches M1-M5 (and ``legacy``).

    estimate_pose(images, golden, cfg, K, D, masks=None) -> PoseEstimate

* ``legacy``: the hybrid pipeline of ``correction.run_correction_step``
  (Diamond -> ORB match -> PnP/Diamond gate -> robust refinement, Diamond-only
  fallback) with the QualityConfig sigmas (0.5 / 1.5 px).
* M1: Diamond only (IPPE_SQUARE + ambiguity check), no refinement.
* M2: marker-free (``markerless.py``).
* M3: hybrid with PBL weights (sigma_d 0.01 px, sigma_o 1 px, Huber 2).
* M4: hybrid with measured sigmas (``noise_model.yaml``) + covariance.
* M5: M4 on k static frames, stacked residuals (or SE(3) average).

All methods honour ``masks`` (255 = usable): the reference-board region is
removed from ORB extraction so the ground truth stays independent.
Relative motion follows Appendix A.1 (``motion.py``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .common import (calibration_fingerprint, detect_diamond, extract_orb, load_golden_pose,
                     resolve_backend, save_golden_pose, validate_image)
from .covariance import pose_covariance
from .geometry import (DEFAULT_QUALITY, QualityConfig, camera_deviation, invert_pose, match_descriptors,
                       pose_matrix, refine_pose, transform_points, triangulate, validate_transform,
                       verify_correspondences)
from .markerless import build_markerless_golden, estimate_markerless
from .motion import as_T, average_poses, camera_center_in_golden, invert, relative_motion, rotvec
from .solver import Block, refine_pose_blocks

METHODS = ("legacy", "M1", "M2", "M3", "M4", "M5")
NOISE_SCHEMA = "cita.noise_model.v1"


class NoiseModelError(ValueError):
    pass


def load_noise_model(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise NoiseModelError(
            f"Noise model {path} not found. Run tools/experiments/e0_noise.py on E0.2/E0.3 sessions "
            "first; M4/M5 never fall back to default sigmas.")
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if data.get("schema") != NOISE_SCHEMA:
        raise NoiseModelError(f"{path}: schema must be {NOISE_SCHEMA}")
    for key in ("diamond_sigma_px", "orb_sigma_px"):
        v = data.get(key)
        if not isinstance(v, (int, float)) or not v > 0:
            raise NoiseModelError(f"{path}: {key} must be a positive number")
    from app.experiments.trial_log import sha256_file
    data["_path"], data["_sha256"] = str(path), sha256_file(path)
    return data


@dataclass(frozen=True)
class PoseMethodConfig:
    method: str = "legacy"
    diamond_sigma_px: float = 0.5
    orb_sigma_px: float = 1.5
    huber_delta: float = 2.0
    huber: bool = True
    backend: str = "auto"
    orb_region: str = "full"
    frames: int = 1
    fusion: str = "stacked"
    covariance: bool = False
    resize_to: tuple | None = None
    ratio: float | None = None
    min_pnp_inliers: int = 6
    quality: QualityConfig = DEFAULT_QUALITY
    noise_model: dict | None = None

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"Unknown pose method {self.method}")

    @classmethod
    def from_profile(cls, profile, quality: QualityConfig = DEFAULT_QUALITY) -> "PoseMethodConfig":
        p = profile.pose
        noise = None
        dsig, osig = p.diamond_sigma_px, p.orb_sigma_px
        if p.method in ("M4", "M5"):
            if not p.noise_model:
                raise NoiseModelError(f"{p.method} requires pose.noise_model in the profile")
            noise = load_noise_model(profile.resolve_path(p.noise_model))
            dsig, osig = float(noise["diamond_sigma_px"]), float(noise["orb_sigma_px"])
        return cls(method=p.method, diamond_sigma_px=dsig, orb_sigma_px=osig, huber_delta=p.huber_delta,
                   huber=p.huber, backend=p.backend, orb_region=p.orb_region, frames=p.frames,
                   fusion=p.fusion, covariance=p.covariance,
                   resize_to=tuple(p.resize_to) if p.resize_to else None, ratio=p.ratio,
                   min_pnp_inliers=p.min_pnp_inliers, quality=quality, noise_model=noise)

    def quality_for_solver(self) -> QualityConfig:
        return replace(self.quality, diamond_sigma_px=self.diamond_sigma_px,
                       orb_sigma_px=self.orb_sigma_px, huber_delta=self.huber_delta)


@dataclass
class GoldenModel:
    K: np.ndarray
    D: np.ndarray
    image_size: tuple
    hybrid: dict | None = None          # load_golden_pose() output (Diamond world frame)
    markerless: dict | None = None      # build_markerless_golden() output (golden camera frame)
    roi_polygon_px: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def T_gw(self) -> np.ndarray | None:
        if self.hybrid is None:
            return None
        return as_T(self.hybrid["rvec"], self.hybrid["tvec"])


@dataclass
class PoseEstimate:
    ok: bool
    status: str
    method: str
    rvec: np.ndarray | None = None              # camera <- world (Diamond), if available
    tvec: np.ndarray | None = None
    delta_R: np.ndarray | None = None           # golden camera -> current camera
    delta_t: np.ndarray | None = None
    camera_center_in_golden_m: np.ndarray | None = None
    rotvec_deg: np.ndarray | None = None
    cov_xi: np.ndarray | None = None
    sigma_center_mm: list | None = None
    sigma_rot_deg: list | None = None
    n_diamond: int = 0
    n_orb_matches: int = 0
    n_orb_inliers: int = 0
    diamond_rms_px: float | None = None
    orb_rms_px: float | None = None
    cost_initial: float | None = None
    cost_final: float | None = None
    jacobian_condition: float | None = None
    timings_ms: dict = field(default_factory=dict)
    strategy_used: str | None = None
    frames_used: int = 1
    covariance: dict | None = None
    deviation: dict | None = None               # geometry.camera_deviation (legacy contract)
    extra: dict = field(default_factory=dict)

    @property
    def dT(self) -> np.ndarray | None:
        if self.delta_R is None:
            return None
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.delta_R, self.delta_t
        return T

    def to_json(self) -> dict:
        def j(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.floating, np.integer)):
                return v.item()
            if isinstance(v, dict):
                return {k: j(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [j(x) for x in v]
            if isinstance(v, float) and not np.isfinite(v):
                return None
            return v
        out = {k: j(getattr(self, k)) for k in self.__dataclass_fields__}
        if self.camera_center_in_golden_m is not None:
            out["c_mm"] = (np.asarray(self.camera_center_in_golden_m) * 1000).tolist()
            out["c_norm_mm"] = float(np.linalg.norm(self.camera_center_in_golden_m) * 1000)
            out["rot_deg"] = float(np.linalg.norm(self.rotvec_deg))
        return out


# ---------------------------------------------------------------------------
# Golden construction
# ---------------------------------------------------------------------------

def commanded_stereo_transform(baseline_m: float, sign: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3] = -sign * baseline_m
    return T


def build_hybrid_golden(left, right, K, D, output: str | Path, *, stereo_source: str = "diamond",
                        baseline_m: float = 0.1, quality: QualityConfig = DEFAULT_QUALITY,
                        backend: str = "python", masks=(None, None), ratio: float = 0.75) -> dict:
    """``initialize.run_initialization(strategy='hybrid')`` + masks + stereo source.

    With ``stereo_source='diamond'`` and no masks it performs exactly the same
    steps as ``run_initialization`` (verified by tests).
    """
    left = validate_image(left)
    right = validate_image(right, (left.shape[1], left.shape[0]))
    native = resolve_backend(backend)
    diamond = detect_diamond(left, K, D, quality)
    if diamond is None:
        raise ValueError("Expected Diamond IDs not detected, or pose quality rejected")
    if diamond["ambiguous"]:
        raise ValueError("Ambiguous planar reference pose")
    refined = refine_pose(diamond["rvec"], diamond["tvec"], diamond["obj_pts"], diamond["corners"],
                          np.empty((0, 3)), np.empty((0, 2)), K, D, quality, native=native)
    if not refined["accepted"]:
        raise ValueError(f"Reference refinement rejected: {refined['status']}")
    diamond.update(rvec=refined["rvec"], tvec=refined["tvec"])
    T_left_world = pose_matrix(diamond["rvec"], diamond["tvec"])
    _, dl, kl = extract_orb(left, backend=backend, mask=masks[0])
    _, dr, kr = extract_orb(right, backend=backend, mask=masks[1])
    matches = match_descriptors(dl, dr, ratio=ratio)
    il = np.array([m.queryIdx for m in matches], dtype=int)
    ir = np.array([m.trainIdx for m in matches], dtype=int)
    if stereo_source == "diamond":
        right_diamond = detect_diamond(right, K, D, quality)
        if right_diamond is None or right_diamond["ambiguous"]:
            raise ValueError("Right Diamond missing/ambiguous")
        right_fit = refine_pose(right_diamond["rvec"], right_diamond["tvec"], right_diamond["obj_pts"],
                                right_diamond["corners"], np.empty((0, 3)), np.empty((0, 2)), K, D,
                                quality, native=native)
        if not right_fit["accepted"]:
            raise ValueError("Right Diamond refinement rejected")
        T = pose_matrix(right_fit["rvec"], right_fit["tvec"]) @ invert_pose(T_left_world)
        tri = triangulate(kl[il], kr[ir], K, D, T, quality)
        source = "diamond_estimate_not_independent_metrology"
    elif stereo_source == "commanded":
        best = None
        for sign in (1.0, -1.0):
            T_try = commanded_stereo_transform(baseline_m, sign)
            tri_try = triangulate(kl[il], kr[ir], K, D, T_try, quality)
            if best is None or tri_try["num_valid"] > best[1]["num_valid"]:
                best = (T_try, tri_try)
        T, tri = best
        source = "commanded_slider_baseline"
    else:
        raise ValueError(f"Unknown stereo_source {stereo_source}")
    if tri["num_valid"] < quality.min_orb_inliers:
        raise ValueError(f"Only {tri['num_valid']} geometrically valid stereo landmarks")
    valid = tri["valid_mask"]
    xyz = transform_points(invert_pose(T_left_world), tri["points_3d"][valid])
    xy, desc = kl[il][valid], dl[il][valid]
    stereo_meta = {"source": source, "T_right_left": T.tolist(), "baseline_m": float(np.linalg.norm(T[:3, 3])),
                   "matched": len(matches), "accepted": len(xyz), "commanded_baseline_m": baseline_m}
    save_golden_pose(output, diamond, xyz, xy, desc, (left.shape[1], left.shape[0]), K=K, D=D,
                     strategy="hybrid", stereo=stereo_meta, min_orb_inliers=quality.min_orb_inliers)
    return load_golden_pose(output, K=K, D=D, image_size=(left.shape[1], left.shape[0]))


def build_golden_model(left, right, K, D, *, cfg: PoseMethodConfig, workdir: str | Path,
                       stereo_source: str = "diamond", baseline_m: float = 0.1, masks=(None, None),
                       build_hybrid: bool = True, build_markerless: bool = False,
                       roi_polygon_px=None) -> GoldenModel:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    gm = GoldenModel(K=np.asarray(K, float), D=np.asarray(D, float).reshape(-1),
                     image_size=(left.shape[1], left.shape[0]),
                     roi_polygon_px=None if roi_polygon_px is None else np.asarray(roi_polygon_px, float))
    backend = "python" if cfg.backend == "auto" and cfg.method != "legacy" else cfg.backend
    if build_hybrid:
        gm.hybrid = build_hybrid_golden(left, right, K, D, workdir / "golden_pose.yaml",
                                        stereo_source=stereo_source, baseline_m=baseline_m,
                                        quality=cfg.quality, backend=backend, masks=masks)
        gm.meta["hybrid_stereo"] = gm.hybrid["metadata"].get("stereo")
    if build_markerless:
        gm.markerless = build_markerless_golden(left, right, K, D, baseline_m=baseline_m,
                                                ratio=cfg.ratio or 0.6, resize_to=cfg.resize_to,
                                                masks=masks, quality=cfg.quality, backend=backend)
        gm.meta["markerless"] = {k: gm.markerless[k] for k in ("n_matches", "n_valid", "baseline_m",
                                                               "slider_x_sign_in_camera", "baseline_check",
                                                               "resize_to")}
    return gm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finish(est: PoseEstimate, T_cw: np.ndarray | None, golden: GoldenModel, dT: np.ndarray | None = None,
            quality: QualityConfig = DEFAULT_QUALITY) -> PoseEstimate:
    T_gw = golden.T_gw
    if dT is None and T_cw is not None and T_gw is not None:
        dT = relative_motion(T_gw, T_cw)
    if T_cw is None and dT is not None and T_gw is not None:
        T_cw = dT @ T_gw
    if T_cw is not None:
        est.rvec = cv2.Rodrigues(T_cw[:3, :3])[0].ravel()
        est.tvec = T_cw[:3, 3].copy()
        if T_gw is not None:
            est.deviation = camera_deviation(est.rvec, est.tvec, golden.hybrid["rvec"], golden.hybrid["tvec"], quality)
    if dT is not None:
        est.delta_R, est.delta_t = dT[:3, :3].copy(), dT[:3, 3].copy()
        est.camera_center_in_golden_m = camera_center_in_golden(dT)
        est.rotvec_deg = np.rad2deg(rotvec(dT[:3, :3]))
    return est


def _attach_covariance(est: PoseEstimate, T_cw: np.ndarray, blocks: list[Block], cfg: PoseMethodConfig,
                       K, D, T_ref: np.ndarray | None) -> None:
    cov = pose_covariance(T_cw, blocks, K, D, huber_delta=cfg.huber_delta, huber=cfg.huber, T_gw=T_ref,
                          max_reprojection_px=cfg.quality.max_reprojection_px)
    est.covariance = cov
    if cov.get("cov_xi") is not None:
        est.cov_xi = np.asarray(cov["cov_xi"])
        est.sigma_center_mm = cov["sigma_center_mm"]
        est.sigma_rot_deg = cov["sigma_rot_deg"]


def project_golden_polygon(poly_px, T_gw, T_cw, K) -> np.ndarray:
    """Golden-image polygon -> current image through the Diamond-plane
    homography H = K (dR - dt n^T / d) K^-1 (Appendix A.5)."""
    poly = np.asarray(poly_px, dtype=np.float64).reshape(-1, 2)
    dT = relative_motion(T_gw, T_cw)
    n = T_gw[:3, :3] @ np.array([0.0, 0.0, 1.0])            # plane normal in golden camera frame
    d = float(n @ T_gw[:3, 3])                               # n.X = d for points X of the plane
    H = K @ (dT[:3, :3] + np.outer(dT[:3, 3], n) / d) @ np.linalg.inv(K)
    out = cv2.perspectiveTransform(poly.reshape(-1, 1, 2), H).reshape(-1, 2)
    return out


def roi_mask(shape, polygon_px, margin_px: int = 40) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.rint(np.asarray(polygon_px)).astype(np.int32)], 255)
    if margin_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1))
        mask = cv2.dilate(mask, k)
    return mask


def _combine_masks(*masks):
    out = None
    for m in masks:
        if m is None:
            continue
        out = m.copy() if out is None else cv2.bitwise_and(out, m)
    return out


# ---------------------------------------------------------------------------
# Hybrid (legacy, M3, M4) single frame
# ---------------------------------------------------------------------------

def _hybrid_observations(image, golden: GoldenModel, cfg: PoseMethodConfig, mask, backend: str) -> dict:
    """Diamond detection + ORB matching + verification (no refinement)."""
    K, D, quality = golden.K, golden.D, cfg.quality
    obs = {"diamond": detect_diamond(image, K, D, quality), "o3": np.empty((0, 3)), "o2": np.empty((0, 2)),
           "num_matches": 0}
    d = obs["diamond"]
    if d is None or d["ambiguous"]:
        return obs
    roi = None
    if cfg.orb_region == "roi" and golden.roi_polygon_px is not None:
        poly = project_golden_polygon(golden.roi_polygon_px, golden.T_gw, as_T(d["rvec"], d["tvec"]), K)
        roi = roi_mask(image.shape, poly)
    m = _combine_masks(mask, roi)
    _, desc, kp = extract_orb(image, backend=backend, mask=m)
    matches = match_descriptors(desc, golden.hybrid["descriptors"])
    obs["num_matches"] = len(matches)
    ci = np.array([mm.queryIdx for mm in matches], dtype=int)
    gi = np.array([mm.trainIdx for mm in matches], dtype=int)
    matched3, matched2 = golden.hybrid["points_3d"][gi], kp[ci]
    keep = verify_correspondences(matched3, matched2, d["rvec"], d["tvec"], K, D, quality)
    if int(keep.sum()) >= quality.min_orb_inliers:
        obs["o3"], obs["o2"] = matched3[keep], matched2[keep]
    return obs


def _hybrid_single(image, golden: GoldenModel, cfg: PoseMethodConfig, mask) -> PoseEstimate:
    t0 = time.perf_counter()
    K, D = golden.K, golden.D
    quality = cfg.quality_for_solver()
    backend = cfg.backend
    native = resolve_backend(backend)
    est = PoseEstimate(ok=False, status="diamond_not_found_or_rejected", method=cfg.method)
    obs = _hybrid_observations(image, golden, cfg, mask, backend)
    t1 = time.perf_counter()
    d = obs["diamond"]
    if d is None:
        est.timings_ms = {"detect_match": (t1 - t0) * 1000, "total": (t1 - t0) * 1000}
        return est
    est.n_diamond = 4
    if d["ambiguous"]:
        est.status = "ambiguous_diamond_pose"
        est.timings_ms = {"detect_match": (t1 - t0) * 1000, "total": (t1 - t0) * 1000}
        return est
    o3, o2 = obs["o3"], obs["o2"]
    est.n_orb_matches = obs["num_matches"]

    def solve(o3_, o2_):
        if cfg.huber:
            return refine_pose(d["rvec"], d["tvec"], d["obj_pts"], d["corners"], o3_, o2_, K, D, quality,
                               native=native)
        blocks = [Block(d["obj_pts"], d["corners"], quality.diamond_sigma_px, False, "diamond")]
        if len(o3_):
            blocks.append(Block(o3_, o2_, quality.orb_sigma_px, True, "orb"))
        return refine_pose_blocks(d["rvec"], d["tvec"], blocks, K, D, quality, huber=False)

    fit = solve(o3, o2)
    est.strategy_used = "hybrid" if len(o3) else "diamond"
    if not fit["accepted"] and len(o3):
        est.extra["hybrid_rejection"] = {k: fit[k] for k in ("status", "final_robust_cost", "num_orb_inliers")}
        o3, o2 = np.empty((0, 3)), np.empty((0, 2))
        fit = solve(o3, o2)
        est.strategy_used = "diamond"
    t2 = time.perf_counter()
    est.cost_initial, est.cost_final = fit["initial_robust_cost"], fit["final_robust_cost"]
    est.diamond_rms_px = fit["diamond_rms_px"]
    est.orb_rms_px = fit.get("orb_mean_error_px")
    est.n_orb_inliers = fit["num_orb_inliers"] if len(o3) else 0
    est.jacobian_condition = fit["scaled_jacobian_condition"]
    est.extra.update(backend=fit["backend"], iterations=fit["iterations"], num_verified_matches=len(o3))
    if not fit["accepted"]:
        est.status = "refinement_rejected"
        est.timings_ms = {"detect_match": (t1 - t0) * 1000, "refine": (t2 - t1) * 1000,
                          "total": (time.perf_counter() - t0) * 1000}
        return est
    est.ok = True
    est.status = "accepted" if est.strategy_used == "hybrid" else "accepted_diamond_fallback"
    T_cw = as_T(fit["rvec"], fit["tvec"])
    _finish(est, T_cw, golden, quality=cfg.quality)
    if cfg.covariance:
        blocks = [Block(d["obj_pts"], d["corners"], quality.diamond_sigma_px, False, "diamond")]
        if len(o3):
            blocks.append(Block(o3, o2, quality.orb_sigma_px, True, "orb"))
        _attach_covariance(est, T_cw, blocks, cfg, K, D, golden.T_gw)
    est.timings_ms = {"detect_match": (t1 - t0) * 1000, "refine": (t2 - t1) * 1000,
                      "total": (time.perf_counter() - t0) * 1000}
    return est


# ---------------------------------------------------------------------------
# M1, M2, M5
# ---------------------------------------------------------------------------

def _diamond_only(image, golden: GoldenModel, cfg: PoseMethodConfig) -> PoseEstimate:
    t0 = time.perf_counter()
    d = detect_diamond(image, golden.K, golden.D, cfg.quality)
    est = PoseEstimate(ok=False, status="diamond_not_found_or_rejected", method="M1", strategy_used="diamond")
    if d is not None:
        est.n_diamond = 4
        est.diamond_rms_px = d["rms_px"]
        if d["ambiguous"]:
            est.status = "ambiguous_diamond_pose"
        else:
            est.ok, est.status = True, "accepted"
            T_cw = as_T(d["rvec"], d["tvec"])
            _finish(est, T_cw, golden, quality=cfg.quality)
            if cfg.covariance:
                _attach_covariance(est, T_cw, [Block(d["obj_pts"], d["corners"], cfg.diamond_sigma_px,
                                                     False, "diamond")], cfg, golden.K, golden.D, golden.T_gw)
    est.timings_ms = {"total": (time.perf_counter() - t0) * 1000}
    return est


def _markerless(image, golden: GoldenModel, cfg: PoseMethodConfig, mask) -> PoseEstimate:
    if golden.markerless is None:
        raise ValueError("M2 requires a markerless golden (golden.build_markerless)")
    res = estimate_markerless(image, golden.markerless, golden.K, golden.D, ratio=cfg.ratio or 0.6,
                              sigma_px=cfg.orb_sigma_px, huber_delta=cfg.huber_delta, huber=cfg.huber,
                              min_inliers=cfg.min_pnp_inliers, quality=cfg.quality, mask=mask,
                              backend="python" if cfg.backend == "auto" else cfg.backend)
    est = PoseEstimate(ok=res["ok"], status=res["status"], method="M2", strategy_used="markerless",
                       n_orb_matches=res["n_matches"], timings_ms=res["timings_ms"])
    fit = res.get("refine")
    if fit is not None:
        est.n_orb_inliers = fit["num_orb_inliers"]
        est.orb_rms_px = fit.get("orb_rms_px")
        est.cost_initial, est.cost_final = fit["initial_robust_cost"], fit["final_robust_cost"]
        est.jacobian_condition = fit["scaled_jacobian_condition"]
    est.extra["pnp_inliers"] = res["n_inliers"]
    if res["ok"]:
        dT = res["dT"]
        _finish(est, None, golden, dT=dT, quality=cfg.quality)
        if cfg.covariance:
            xyz, xy = res["landmarks"]
            _attach_covariance(est, dT, [Block(xyz, xy, cfg.orb_sigma_px, True, "orb")], cfg,
                               res["K_used"], golden.D, None)
    return est


def _stacked(images, golden: GoldenModel, cfg: PoseMethodConfig, masks) -> PoseEstimate:
    t0 = time.perf_counter()
    K, D = golden.K, golden.D
    quality = cfg.quality_for_solver()
    est = PoseEstimate(ok=False, status="diamond_not_found_or_rejected", method="M5", frames_used=0)
    diamond_blocks, orb_blocks, x0 = [], [], None
    n_matches = 0
    for k, (img, m) in enumerate(zip(images, masks)):
        obs = _hybrid_observations(img, golden, cfg, m, "python")
        d = obs["diamond"]
        if d is None or d["ambiguous"]:
            continue
        diamond_blocks.append(Block(d["obj_pts"], d["corners"], quality.diamond_sigma_px, False, "diamond", k))
        n_matches += obs["num_matches"]
        if len(obs["o3"]):
            orb_blocks.append(Block(obs["o3"], obs["o2"], quality.orb_sigma_px, True, "orb", k))
        if x0 is None:
            x0 = (d["rvec"], d["tvec"])
    t1 = time.perf_counter()
    est.frames_used = len(diamond_blocks)
    est.n_diamond = 4 * len(diamond_blocks)
    est.n_orb_matches = n_matches
    if x0 is None:
        est.timings_ms = {"detect_match": (t1 - t0) * 1000, "total": (t1 - t0) * 1000}
        return est
    blocks = diamond_blocks + orb_blocks
    fit = refine_pose_blocks(x0[0], x0[1], blocks, K, D, quality, huber=cfg.huber)
    est.strategy_used = "hybrid" if orb_blocks else "diamond"
    if not fit["accepted"] and orb_blocks:
        est.extra["hybrid_rejection"] = {k: fit[k] for k in ("status", "final_robust_cost", "num_orb_inliers")}
        blocks = diamond_blocks
        fit = refine_pose_blocks(x0[0], x0[1], blocks, K, D, quality, huber=cfg.huber)
        est.strategy_used = "diamond"
    t2 = time.perf_counter()
    est.cost_initial, est.cost_final = fit["initial_robust_cost"], fit["final_robust_cost"]
    est.diamond_rms_px = fit["diamond_rms_px"]
    est.orb_rms_px = fit.get("orb_rms_px")
    est.n_orb_inliers = fit["num_orb_inliers"] if est.strategy_used == "hybrid" else 0
    est.jacobian_condition = fit["scaled_jacobian_condition"]
    est.extra.update(backend="scipy", iterations=fit["iterations"],
                     diamond_rms_px_frames=fit["diamond_rms_px_frames"])
    if fit["accepted"]:
        est.ok = True
        est.status = "accepted" if est.strategy_used == "hybrid" else "accepted_diamond_fallback"
        T_cw = as_T(fit["rvec"], fit["tvec"])
        _finish(est, T_cw, golden, quality=cfg.quality)
        if cfg.covariance:
            _attach_covariance(est, T_cw, blocks, cfg, K, D, golden.T_gw)
    else:
        est.status = "refinement_rejected"
    est.timings_ms = {"detect_match": (t1 - t0) * 1000, "refine": (t2 - t1) * 1000,
                      "total": (time.perf_counter() - t0) * 1000}
    return est


def _averaged(images, golden: GoldenModel, cfg: PoseMethodConfig, masks) -> PoseEstimate:
    t0 = time.perf_counter()
    single = replace(cfg, method="M4", frames=1)
    per = [_hybrid_single(img, golden, single, m) for img, m in zip(images, masks)]
    good = [e for e in per if e.ok]
    est = PoseEstimate(ok=bool(good), status="accepted" if good else "all_frames_rejected", method="M5",
                       frames_used=len(good), strategy_used="average")
    est.extra["per_frame_status"] = [e.status for e in per]
    if good:
        T_cw = average_poses([as_T(e.rvec, e.tvec) for e in good])
        _finish(est, T_cw, golden, quality=cfg.quality)
        est.n_orb_inliers = int(sum(e.n_orb_inliers for e in good))
        est.n_diamond = 4 * len(good)
        if cfg.covariance and all(e.cov_xi is not None for e in good):
            # Mean of n independent estimates: Sigma / n (approximation).
            est.sigma_center_mm = (np.mean([e.sigma_center_mm for e in good], axis=0) / np.sqrt(len(good))).tolist()
            est.sigma_rot_deg = (np.mean([e.sigma_rot_deg for e in good], axis=0) / np.sqrt(len(good))).tolist()
    est.timings_ms = {"total": (time.perf_counter() - t0) * 1000}
    return est


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def estimate_pose(images: list[np.ndarray], golden: GoldenModel, cfg: PoseMethodConfig, K=None, D=None,
                  masks: list[np.ndarray | None] | None = None) -> PoseEstimate:
    if isinstance(images, np.ndarray):
        images = [images]
    images = [validate_image(img) for img in images]
    if not images:
        raise ValueError("At least one image is required")
    if K is not None or D is not None:
        if calibration_fingerprint(K, D, golden.image_size) != calibration_fingerprint(golden.K, golden.D, golden.image_size):
            raise ValueError("K/D differ from the golden model calibration")
    for img in images:
        if (img.shape[1], img.shape[0]) != tuple(golden.image_size):
            raise ValueError("Image size differs from the golden image size")
    masks = list(masks) if masks is not None else [None] * len(images)
    if len(masks) != len(images):
        raise ValueError("masks must align with images")
    method = cfg.method
    if method == "M5":
        if len(images) != cfg.frames:
            raise ValueError(f"M5 expects {cfg.frames} frames, got {len(images)}")
        return _stacked(images, golden, cfg, masks) if cfg.fusion == "stacked" else _averaged(images, golden, cfg, masks)
    if len(images) != 1:
        raise ValueError(f"{method} uses exactly one image")
    if method == "M1":
        return _diamond_only(images[0], golden, cfg)
    if method == "M2":
        return _markerless(images[0], golden, cfg, masks[0])
    if golden.hybrid is None:
        raise ValueError(f"{method} requires a hybrid golden model")
    return _hybrid_single(images[0], golden, cfg, masks[0])


def golden_from_file(path: str | Path, K, D, image_size, roi_polygon_px=None) -> GoldenModel:
    """Wrap an existing per-artifact ``golden_pose.yaml`` (server path)."""
    hybrid = load_golden_pose(path, K=K, D=D, image_size=image_size)
    return GoldenModel(K=np.asarray(K, float), D=np.asarray(D, float).reshape(-1), image_size=tuple(image_size),
                       hybrid=hybrid, roi_polygon_px=roi_polygon_px)


def estimate_to_correction_result(est: PoseEstimate, quality: QualityConfig) -> dict:
    """Map a PoseEstimate to the dict returned by ``run_correction_step``
    (keys used by inspection_service / the app), for non-legacy profiles."""
    result = {"accepted": est.ok, "status": est.status, "diamond": None, "hybrid": None, "deviation": None,
              "num_matches": est.n_orb_matches, "num_verified_matches": est.extra.get("num_verified_matches"),
              "motor_command": {"enabled": False, "reason": "computed by the alignment controller"},
              "strategy_requested": est.method, "strategy_used": est.strategy_used,
              "elapsed_ms": est.timings_ms.get("total"), "cita_pose": est.to_json()}
    if est.ok and est.deviation is not None:
        deviation = est.deviation
        correction = np.asarray(deviation["translation_correction_world_m"], dtype=float)
        deviation.update({
            "delta_x": float(correction[0]), "delta_y": float(correction[1]), "delta_z": float(correction[2]),
            "translation_mag": float(deviation["translation_magnitude"]),
            "rotation_mag": float(deviation["rotation_magnitude"]),
            "within_trans_tolerance": bool(deviation["within_translation_tolerance"]),
            "within_rot_tolerance": bool(deviation["within_rotation_tolerance"]),
            "motor_command": result["motor_command"],
        })
        result["deviation"] = deviation
        result["hybrid"] = {"rvec": est.rvec, "tvec": est.tvec, "accepted": True}
    return result
