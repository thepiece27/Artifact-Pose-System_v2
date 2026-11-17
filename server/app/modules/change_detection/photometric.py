"""Photometric normalisation of the aligned current image to the reference.

* ``none``
* ``gain_offset``: per channel cur' = a cur + b fitted inside the valid ROI by
  least squares, dropping the 10 % largest residuals, 2 iterations.
* ``clahe_lab``: CLAHE on the L channel of Lab, both images (Nishidate et al. 2023).
"""
from __future__ import annotations

import cv2
import numpy as np


def fit_gain_offset(cur: np.ndarray, ref: np.ndarray, mask: np.ndarray | None, *, trim: float = 0.10,
                    iterations: int = 2, max_samples: int = 200_000, seed: int = 0) -> list[tuple[float, float]]:
    sel = (mask > 0) if mask is not None else np.ones(cur.shape[:2], bool)
    idx = np.flatnonzero(sel.ravel())
    if len(idx) > max_samples:
        idx = np.random.default_rng(seed).choice(idx, max_samples, replace=False)
    params = []
    channels = 1 if cur.ndim == 2 else cur.shape[2]
    for ch in range(channels):
        x = (cur if cur.ndim == 2 else cur[..., ch]).ravel()[idx].astype(np.float64)
        y = (ref if ref.ndim == 2 else ref[..., ch]).ravel()[idx].astype(np.float64)
        keep = np.ones(len(x), bool)
        a, b = 1.0, 0.0
        for _ in range(iterations + 1):
            if keep.sum() < 10:
                break
            A = np.c_[x[keep], np.ones(keep.sum())]
            a, b = np.linalg.lstsq(A, y[keep], rcond=None)[0]
            res = np.abs(a * x + b - y)
            keep = res <= np.quantile(res, 1.0 - trim)
        params.append((float(a), float(b)))
    return params


def normalize(cur: np.ndarray, ref: np.ndarray, mask: np.ndarray | None, mode: str = "none"):
    """Returns (cur_n, ref_n, info)."""
    if mode == "none":
        return cur, ref, {"mode": "none"}
    if mode == "gain_offset":
        params = fit_gain_offset(cur, ref, mask)
        out = cur.astype(np.float32)
        if cur.ndim == 2:
            out = out * params[0][0] + params[0][1]
        else:
            for ch, (a, b) in enumerate(params):
                out[..., ch] = out[..., ch] * a + b
        return np.clip(np.rint(out), 0, 255).astype(np.uint8), ref, {"mode": mode, "gain_offset": params}
    if mode == "clahe_lab":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        def f(img):
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab[..., 0] = clahe.apply(lab[..., 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return f(cur), f(ref), {"mode": mode, "clip_limit": 2.0, "tile": 8}
    raise ValueError(f"Unknown photometric mode {mode}")
