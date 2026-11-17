"""Painting ROI in the REFERENCE (golden) image frame.

``roi.json`` = {"schema": "cita.roi.v1", "polygon_ref_px": [[x, y], ...],
"size_mm": [w, h] | null, "source": "auto_quad" | "manual" | "diamond_projection" | ...}.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROI_SCHEMA = "cita.roi.v1"


def save_roi(path, polygon_ref_px, *, size_mm=None, source="manual", extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"schema": ROI_SCHEMA, "polygon_ref_px": np.asarray(polygon_ref_px, float).tolist(),
            "size_mm": list(size_mm) if size_mm else None, "source": source, **(extra or {})}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_roi(path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") not in (ROI_SCHEMA, None) or len(data.get("polygon_ref_px") or []) < 3:
        raise ValueError(f"Invalid ROI file {path}")
    data["polygon_ref_px"] = np.asarray(data["polygon_ref_px"], float)
    return data


def roi_mask(shape, polygon, erode_px: int = 0) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.rint(np.asarray(polygon, float)).astype(np.int32)], 255)
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        mask = cv2.erode(mask, k)
    return mask


def point_in_polygon(points, polygon, margin_px: float = 0.0) -> np.ndarray:
    poly = np.asarray(polygon, np.float32).reshape(-1, 1, 2)
    return np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), True) >= margin_px
                     for x, y in np.asarray(points, float).reshape(-1, 2)], dtype=bool)


def auto_detect_quad(image: np.ndarray, *, min_area_frac: float = 0.03) -> np.ndarray | None:
    """Largest convex quadrilateral (e.g. an A4 sheet on the formex)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    best, best_area = None, min_area_frac * gray.size
    for edges in (cv2.Canny(blur, 40, 120),
                  cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 5)):
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            area = cv2.contourArea(approx)
            if len(approx) == 4 and cv2.isContourConvex(approx) and area > best_area and area < 0.95 * gray.size:
                best, best_area = approx.reshape(4, 2).astype(float), area
    if best is None:
        return None
    # Order: top-left, top-right, bottom-right, bottom-left.
    s, d = best.sum(1), np.diff(best, axis=1).ravel()
    return np.array([best[np.argmin(s)], best[np.argmin(d)], best[np.argmax(s)], best[np.argmax(d)]])


def roi_from_diamond(K, D, rvec, tvec, rect_mm) -> np.ndarray:
    """Project a rectangle given in Diamond-plane coordinates (mm,
    [x0, y0, x1, y1], Y up) into the reference image."""
    x0, y0, x1, y1 = (v / 1000.0 for v in rect_mm)
    obj = np.array([[x0, y1, 0], [x1, y1, 0], [x1, y0, 0], [x0, y0, 0]], float)
    return cv2.projectPoints(obj, np.asarray(rvec, float), np.asarray(tvec, float), K, D)[0].reshape(-1, 2)


def px_per_mm(roi: dict | None = None, diamond_corners=None) -> float | None:
    """Scale in the reference image: from the ROI physical size, else from the
    40 mm Diamond square, else None."""
    if roi is not None and roi.get("size_mm"):
        poly = np.asarray(roi["polygon_ref_px"], float)
        if len(poly) == 4:
            w_px = (np.linalg.norm(poly[1] - poly[0]) + np.linalg.norm(poly[2] - poly[3])) / 2
            h_px = (np.linalg.norm(poly[3] - poly[0]) + np.linalg.norm(poly[2] - poly[1])) / 2
            w_mm, h_mm = roi["size_mm"]
            return float((w_px / w_mm + h_px / h_mm) / 2)
    if diamond_corners is not None:
        c = np.asarray(diamond_corners, float).reshape(4, 2)
        side = np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)])
        return float(side / 40.0)
    return None
