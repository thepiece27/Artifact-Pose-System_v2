"""Difference map -> candidate regions (reference frame).

Thresholds: ``otsu_legacy`` (Otsu on the blurred uint8 map, fallback 60 when
Otsu < 30, exactly as cita-baseline) or ``fixed`` (tau on d = 1 - SSIM, chosen
on VALIDATION data only by calibrate_thresholds.py). Morphology as before.
Minimum area: legacy max(500, 1e-4 h w) px^2, or ``min_area_mm2`` converted
with px_per_mm.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

OTSU_FALLBACK = 60
CONTOUR_CAP = 50
MIN_AREA_FACTOR = 0.0001
THRESHOLDS_SCHEMA = "cita.thresholds.v1"


def load_thresholds(path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found: run tools/experiments/calibrate_thresholds.py --split val first "
            "(fixed thresholds are never guessed)")
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if data.get("schema") != THRESHOLDS_SCHEMA or not 0 < float(data.get("tau", -1)) < 1:
        raise ValueError(f"{path}: invalid thresholds file")
    from app.experiments.trial_log import sha256_file
    data["_sha256"] = sha256_file(path)
    data["_path"] = str(path)
    return data


def morphology(mask: np.ndarray) -> np.ndarray:
    kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_s, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_b, iterations=2)


def threshold_map(d: np.ndarray, diff_uint8: np.ndarray, mode: str, tau: float | None = None) -> tuple[np.ndarray, dict]:
    blurred = cv2.GaussianBlur(diff_uint8, (5, 5), 0)
    if mode == "otsu_legacy":
        otsu, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        info = {"mode": mode, "otsu": float(otsu)}
        if otsu < 30:
            _, mask = cv2.threshold(blurred, OTSU_FALLBACK, 255, cv2.THRESH_BINARY)
            info["fallback"] = OTSU_FALLBACK
        return morphology(mask), info
    if mode == "fixed":
        if tau is None:
            raise ValueError("fixed threshold requires tau")
        db = cv2.GaussianBlur(d.astype(np.float32), (5, 5), 0)
        mask = np.where(db > tau, 255, 0).astype(np.uint8)
        return morphology(mask), {"mode": mode, "tau": float(tau)}
    raise ValueError(f"Unknown threshold mode {mode}")


def min_area_px(shape, *, min_area_mm2: float | None, px_per_mm: float | None) -> float:
    h, w = shape[:2]
    if min_area_mm2 is not None and px_per_mm:
        return float(min_area_mm2 * px_per_mm ** 2)
    return float(max(500, (h * w) * MIN_AREA_FACTOR))


def extract_regions(mask: np.ndarray, d: np.ndarray, *, min_area: float, px_per_mm: float | None,
                    cap: int = CONTOUR_CAP) -> list[dict]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = sorted([c for c in contours if cv2.contourArea(c) > min_area], key=cv2.contourArea, reverse=True)[:cap]
    regions = []
    for i, c in enumerate(big):
        x, y, w, h = cv2.boundingRect(c)
        m = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(m, [c], -1, 255, -1)
        vals = d[m > 0]
        area = float(cv2.contourArea(c))
        regions.append({"region_id": i, "bbox_ref_xyxy": [int(x), int(y), int(x + w), int(y + h)],
                        "area_px": area, "area_mm2": area / px_per_mm ** 2 if px_per_mm else None,
                        "mean_d": float(vals.mean()) if len(vals) else 0.0,
                        "max_d": float(vals.max()) if len(vals) else 0.0,
                        "score": float(vals.max()) if len(vals) else 0.0, "contour": c})
    return regions


def image_score(d: np.ndarray, valid_mask: np.ndarray | None, q: float = 99.9) -> float:
    """Threshold-free image-level change score (ROC): high quantile of the blurred map."""
    db = cv2.GaussianBlur(d.astype(np.float32), (5, 5), 0)
    vals = db[valid_mask > 0] if valid_mask is not None else db.ravel()
    return float(np.percentile(vals, q)) if len(vals) else 0.0
