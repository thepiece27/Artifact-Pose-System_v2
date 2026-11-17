"""Image registration current -> reference (I4).

* ``legacy_whole_frame``: the cita-baseline SIFT + FLANN (ratio 0.65) +
  RANSAC homography on the whole frame (same code as ``legacy.py``), here
  also returning H for coordinate mapping.
* ``roi_homography``: matches restricted to the painting ROI (reference side,
  eroded by a few px), RANSAC 3 px with >= 15 inliers, then ECC refinement
  (``cv2.findTransformECC``, MOTION_HOMOGRAPHY, ROI mask, 2-3 pyramid levels,
  H rescaled between levels with S H S^-1). ECC failure keeps the RANSAC H.

ROI-AFD (Appendix A.4): mean |p_ref - p_cur| over SIFT inlier pairs whose
reference point lies in the ROI, measured on the UNWARPED images.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .roi import point_in_polygon, roi_mask


@dataclass
class Registration:
    method: str
    H_ref_from_cur: np.ndarray | None
    aligned: np.ndarray | None
    valid_mask: np.ndarray | None
    n_matches: int = 0
    n_inliers: int = 0
    ecc_ok: bool | None = None
    ecc_rho: float | None = None
    roi_afd_px: float | None = None
    quality: str = "resize_fallback"
    notes: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"method": self.method, "H_ref_from_cur": None if self.H_ref_from_cur is None else self.H_ref_from_cur.tolist(),
                "n_matches": self.n_matches, "n_inliers": self.n_inliers, "ecc_ok": self.ecc_ok,
                "ecc_rho": self.ecc_rho, "roi_afd_px": self.roi_afd_px, "quality": self.quality, **self.notes}


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _sift_matches(cur, ref, ratio=0.65, nfeatures=3000):
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kp1, des1 = sift.detectAndCompute(_gray(cur), None)
    kp2, des2 = sift.detectAndCompute(_gray(ref), None)
    if des1 is None or des2 is None or len(kp1) < 15 or len(kp2) < 15:
        return np.empty((0, 2)), np.empty((0, 2))
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
    good = [m for pair in flann.knnMatch(des1, des2, k=2) if len(pair) == 2
            for m, n in [pair] if m.distance < ratio * n.distance]
    src = np.float64([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    dst = np.float64([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return src, dst


def _warp(cur, ref_shape, H):
    h, w = ref_shape[:2]
    aligned = cv2.warpPerspective(cur, H, (w, h))
    ones = np.ones(cur.shape[:2], dtype=np.uint8) * 255
    valid = cv2.warpPerspective(ones, H, (w, h))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    return aligned, cv2.erode(valid, kernel, iterations=2)


def roi_afd(src_cur, dst_ref, inlier_mask, roi_polygon=None) -> float | None:
    if inlier_mask is None or not len(src_cur):
        return None
    keep = np.asarray(inlier_mask, bool).ravel()
    if roi_polygon is not None:
        keep &= point_in_polygon(dst_ref, roi_polygon)
    if not keep.any():
        return None
    return float(np.mean(np.linalg.norm(dst_ref[keep] - src_cur[keep], axis=1)))


def legacy_whole_frame(cur, ref, roi_polygon=None) -> Registration:
    """Same computation as ``legacy.sift_align_with_mask`` plus H and ROI-AFD."""
    src, dst = _sift_matches(cur, ref)
    reg = Registration("legacy_whole_frame", None, None, None, n_matches=len(src))
    if len(src) < 15:
        reg.aligned = cv2.resize(cur, (ref.shape[1], ref.shape[0]))
        return reg
    H, inl = cv2.findHomography(np.float32(src).reshape(-1, 1, 2), np.float32(dst).reshape(-1, 1, 2), cv2.RANSAC, 3.0)
    if H is None:
        reg.aligned = cv2.resize(cur, (ref.shape[1], ref.shape[0]))
        return reg
    reg.H_ref_from_cur = H
    reg.n_inliers = int(inl.sum()) if inl is not None else 0
    reg.aligned, reg.valid_mask = _warp(cur, ref.shape, H)
    reg.quality = "sift_homography" if reg.n_inliers >= 15 else "resize_fallback"
    reg.roi_afd_px = roi_afd(src, dst, inl, roi_polygon)
    return reg


def _ecc(ref, cur, H0, mask, levels: int, max_iter: int, eps: float, max_side_px: int = 2000):
    """ECC on a pyramid. findTransformECC estimates W: template(ref) -> input(cur).
    The finest level is capped at ``max_side_px`` (long side)."""
    W = np.linalg.inv(H0)
    W = W / W[2, 2]
    rho = None
    g_ref, g_cur = np.float32(_gray(ref)), np.float32(_gray(cur))
    base = min(1.0, max_side_px / float(max(ref.shape[:2])))
    for level in reversed(range(levels)):
        s = base / (2 ** level)
        S = np.diag([s, s, 1.0])
        Wl = (S @ W @ np.linalg.inv(S)).astype(np.float32)
        r = cv2.resize(g_ref, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g_ref
        c = cv2.resize(g_cur, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g_cur
        m = cv2.resize(mask, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_NEAREST) if s < 1 else mask
        rho, Wl = cv2.findTransformECC(r, c, Wl, cv2.MOTION_HOMOGRAPHY,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps), m, 5)
        W = np.linalg.inv(S) @ Wl.astype(np.float64) @ S
    H = np.linalg.inv(W)
    return H / H[2, 2], float(rho)


def roi_homography(cur, ref, roi_polygon, *, ecc: bool = True, levels: int = 3, max_iter: int = 100,
                   eps: float = 1e-6, erode_px: int = 4, ratio: float = 0.65, min_inliers: int = 15,
                   max_side_px: int = 2000) -> Registration:
    src, dst = _sift_matches(cur, ref, ratio=ratio)
    reg = Registration("roi_homography", None, None, None, n_matches=len(src))
    keep = point_in_polygon(dst, roi_polygon, margin_px=erode_px) if len(dst) else np.zeros(0, bool)
    src_r, dst_r = src[keep], dst[keep]
    reg.notes["n_roi_matches"] = int(len(src_r))
    if len(src_r) < min_inliers:
        reg.aligned = cv2.resize(cur, (ref.shape[1], ref.shape[0]))
        reg.notes["failure"] = "too_few_roi_matches"
        return reg
    H, inl = cv2.findHomography(np.float32(src_r).reshape(-1, 1, 2), np.float32(dst_r).reshape(-1, 1, 2),
                                cv2.RANSAC, 3.0)
    reg.n_inliers = 0 if inl is None else int(inl.sum())
    if H is None or reg.n_inliers < min_inliers:
        reg.aligned = cv2.resize(cur, (ref.shape[1], ref.shape[0]))
        reg.notes["failure"] = "ransac_failed"
        return reg
    reg.roi_afd_px = roi_afd(src_r, dst_r, inl, None)
    reg.ecc_ok = False
    if ecc:
        try:
            H_ecc, rho = _ecc(ref, cur, H, roi_mask(ref.shape, roi_polygon, erode_px), levels, max_iter, eps,
                              max_side_px)
            # Reject ECC solutions that drift far from the feature-based H.
            corners = np.float64(roi_polygon).reshape(-1, 1, 2)
            drift = np.max(np.linalg.norm(cv2.perspectiveTransform(corners, H_ecc) - cv2.perspectiveTransform(corners, H), axis=2))
            if np.isfinite(H_ecc).all() and drift < 20.0:
                H, reg.ecc_ok, reg.ecc_rho = H_ecc, True, rho
                reg.notes["ecc_drift_px"] = float(drift)
            else:
                reg.notes["ecc_rejected_drift_px"] = float(drift)
        except cv2.error as exc:
            reg.notes["ecc_error"] = str(exc).splitlines()[-1][:200]
    reg.H_ref_from_cur = H
    reg.aligned, warp_valid = _warp(cur, ref.shape, H)
    reg.valid_mask = cv2.bitwise_and(warp_valid, roi_mask(ref.shape, roi_polygon, erode_px))
    reg.quality = "roi_homography_ecc" if reg.ecc_ok else "roi_homography"
    return reg
