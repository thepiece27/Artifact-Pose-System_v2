"""SSIM difference maps (Wang et al. 2004).

* ``legacy``: cita-baseline formula (gray 0.6 + colour 0.4 score, diff =
  max(1 - SSIM_gray, mean_c(1 - SSIM_c)), full resolution, win_size 7).
* ``standard``: working resolution (long side ``work_long_side_px``),
  Gaussian window sigma 1.5, population covariance, data_range 255; the map
  d = 1 - SSIM is upsampled back to the reference resolution and only
  evaluated inside ``valid_mask``.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def legacy_ssim(ref: np.ndarray, src: np.ndarray) -> dict:
    gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    score_gray, diff_gray = structural_similarity(gray_ref, gray_src, full=True, win_size=7)
    score_channels = []
    diff_color = np.zeros_like(gray_src, dtype=np.float64)
    for c in range(3):
        sc, dc = structural_similarity(ref[:, :, c], src[:, :, c], full=True, win_size=7)
        score_channels.append(sc)
        diff_color += (1.0 - dc)
    diff_color /= 3.0
    score_color = float(np.mean(score_channels))
    diff = np.maximum(1.0 - diff_gray, diff_color)
    return {"ssim": 0.6 * float(score_gray) + 0.4 * score_color, "ssim_gray": float(score_gray),
            "ssim_color": score_color, "d": diff, "diff_uint8": (diff * 255).astype(np.uint8)}


def _ssim_map(a, b):
    _, m = structural_similarity(a, b, full=True, gaussian_weights=True, sigma=1.5,
                                 use_sample_covariance=False, data_range=255)
    return m


def standard_ssim(ref: np.ndarray, src: np.ndarray, valid_mask: np.ndarray | None, *,
                  work_long_side_px: int = 2000, channels: str = "gray_color") -> dict:
    h, w = ref.shape[:2]
    s = min(1.0, work_long_side_px / max(h, w))
    if s < 1.0:
        size = (max(16, int(round(w * s))), max(16, int(round(h * s))))
        r = cv2.resize(ref, size, interpolation=cv2.INTER_AREA)
        c = cv2.resize(src, size, interpolation=cv2.INTER_AREA)
    else:
        r, c = ref, src
    if channels == "lab":
        rl, cl = cv2.cvtColor(r, cv2.COLOR_BGR2LAB), cv2.cvtColor(c, cv2.COLOR_BGR2LAB)
        maps = [_ssim_map(rl[..., k], cl[..., k]) for k in range(3)]
        d = np.mean([1.0 - m for m in maps], axis=0)
    else:
        mg = _ssim_map(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY), cv2.cvtColor(c, cv2.COLOR_BGR2GRAY))
        d = 1.0 - mg
        if channels == "gray_color":
            dc = np.mean([1.0 - _ssim_map(r[..., k], c[..., k]) for k in range(3)], axis=0)
            d = np.maximum(d, dc)
    d = np.clip(d, 0.0, 1.0).astype(np.float32)
    if s < 1.0:
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    if valid_mask is not None:
        d = np.where(valid_mask > 0, d, 0.0).astype(np.float32)
        inside = valid_mask > 0
        ssim_roi = float(1.0 - d[inside].mean()) if inside.any() else None
    else:
        ssim_roi = float(1.0 - d.mean())
    return {"ssim": ssim_roi, "d": d, "diff_uint8": (d * 255).astype(np.uint8), "work_scale": s}
