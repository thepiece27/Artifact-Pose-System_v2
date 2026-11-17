#!/usr/bin/env python3
"""E0.4 settling time t_s from a ``capture_session.py settle`` session.

For every trial the image shift between consecutive frames is measured with
``cv2.phaseCorrelate`` inside a ROI. A trial is settled at the first frame
after which every consecutive shift stays below ``--threshold-px`` (0.3 px).
t_s = the ``--quantile`` (95 %) of the per-trial settling times.

    python tools/experiments/e0_settle.py --session data/sessions/E0.4/S1 --out results/E0.4/S1_settle.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_common_args, make_logger, setup, write_json  # noqa: E402

from app.experiments.session import input_record, load_session  # noqa: E402


def _roi(shape, roi: str | None):
    h, w = shape[:2]
    if roi:
        x, y, rw, rh = (int(v) for v in roi.split(","))
        return x, y, rw, rh
    return w // 4, h // 4, w // 2, h // 2


def shift_px(a: np.ndarray, b: np.ndarray, roi) -> float:
    x, y, w, h = roi
    ga = a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = b if b.ndim == 2 else cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    pa = np.float32(ga[y:y + h, x:x + w])
    pb = np.float32(gb[y:y + h, x:x + w])
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(pa, pb, win)
    return float(np.hypot(dx, dy))


def analyse(session, threshold_px: float, quantile: float, roi_text: str | None) -> dict:
    trials: dict = {}
    for f in session.frames:
        if f.role == "settle":
            trials.setdefault(f.get("trial", f.pose_index), []).append(f)
    per_trial = []
    for trial, frames in sorted(trials.items()):
        frames = sorted(frames, key=lambda fr: fr.get("rep", 0))
        imgs = [fr.load(cv2.IMREAD_GRAYSCALE) for fr in frames]
        roi = _roi(imgs[0].shape, roi_text)
        times = [fr.get("t_since_move_s") for fr in frames]
        shifts = [shift_px(imgs[k], imgs[k + 1], roi) for k in range(len(imgs) - 1)]
        settled_at = None
        for k in range(len(shifts)):
            if all(s < threshold_px for s in shifts[k:]):
                settled_at = times[k] if times[k] is not None else None
                break
        per_trial.append({"trial": trial, "n_frames": len(frames), "t_since_move_s": times,
                          "shift_px": shifts, "settled_at_s": settled_at,
                          "settled": settled_at is not None})
    settled = [t["settled_at_s"] for t in per_trial if t["settled_at_s"] is not None]
    t_s = float(np.quantile(settled, quantile)) if settled else None
    return {"threshold_px": threshold_px, "quantile": quantile, "n_trials": len(per_trial),
            "n_settled": len(settled), "t_s": t_s,
            "note": "trials never settling within the recorded window count as failures"
                    if len(settled) < len(per_trial) else None,
            "trials": per_trial}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold-px", type=float, default=0.3)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--roi", default=None, help="x,y,w,h (default: central half)")
    add_common_args(parser)
    args = parser.parse_args(argv)
    profile, seed, _ = setup(args)
    session = load_session(args.session, verify=True)
    log = make_logger("E0.4", args, profile)
    with log.trial(arm="E0.4-settle", input=input_record(session, session.frames)) as out:
        result = analyse(session, args.threshold_px, args.quantile, args.roi)
        out.update({k: v for k, v in result.items() if k != "trials"})
        out["result_file"] = str(write_json(Path(args.out), {"session_id": session.session_id, **result}))
    print(f"t_s = {result['t_s']} s  ({result['n_settled']}/{result['n_trials']} trials settled); log {log.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
