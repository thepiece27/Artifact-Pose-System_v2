#!/usr/bin/env python3
"""E0.4 mechanical repeatability, backlash (hysteresis) and servo resolution.

* ``repeat`` sessions: every frame is taken at the start point after an
  excursion of ``level`` on side +1/-1. Per (level, side): std of the measured
  pose = repeatability; |mean(+) - mean(-)| = hysteresis (backlash).
* ``grid``/``response`` sessions along one axis: measured response vs
  commanded offset; the smallest commanded step whose response exceeds
  ``k`` x the static noise is reported as the resolvable step.

Measurement (``--measure``):
  board        reference ChArUco board -> camera centre (mm) + rotation (deg)
               relative to the session's first golden frame (independent GT);
  image_shift  phase-correlation shift (px) w.r.t. the golden frame (proxy for
               dry-run data or when no board is visible).

    python tools/experiments/e0_repeat.py --session data/sessions/E0.4/R1 --out results/E0.4/R1.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_common_args, load_calibration, make_logger, resolve_calibration, setup, write_json  # noqa: E402

from app.experiments.session import input_record, load_session  # noqa: E402
from app.modules.artifact_pose.motion import camera_center_in_golden, rotvec  # noqa: E402
from app.modules.ground_truth.reference_board import BoardSpec, detect_reference_board, relative_motion_gt  # noqa: E402


def measure_frames(session, frames, mode: str, cal: dict | None, spec: BoardSpec | None) -> dict:
    golden = session.golden_frames()[0] if session.golden_frames() else frames[0]
    gimg = golden.load()
    out = {}
    if mode == "board":
        g = detect_reference_board(gimg, cal["K"], cal["D"], spec)
        if not g["ok"]:
            raise SystemExit(f"Reference board not detected in golden frame {golden.frame_id}: {g['reason']}")
        for f in frames:
            det = detect_reference_board(f.load(), cal["K"], cal["D"], spec)
            if not det["ok"]:
                out[f.frame_id] = None
                continue
            dT = relative_motion_gt(g, det)
            out[f.frame_id] = np.r_[camera_center_in_golden(dT) * 1000, np.rad2deg(rotvec(dT[:3, :3]))]
    else:
        roi = (gimg.shape[1] // 4, gimg.shape[0] // 4, gimg.shape[1] // 2, gimg.shape[0] // 2)
        gg = cv2.cvtColor(gimg, cv2.COLOR_BGR2GRAY)
        for f in frames:
            img = f.load(cv2.IMREAD_GRAYSCALE)
            x, y, w, h = roi
            win = cv2.createHanningWindow((w, h), cv2.CV_32F)
            (dx, dy), _ = cv2.phaseCorrelate(np.float32(gg[y:y + h, x:x + w]), np.float32(img[y:y + h, x:x + w]), win)
            out[f.frame_id] = np.array([dx, dy])
    return out


def analyse_repeat(session, meas: dict, labels: list[str]) -> dict:
    groups: dict = {}
    for f in session.frames:
        if f.role != "repeat" or meas.get(f.frame_id) is None:
            continue
        groups.setdefault((f.get("level"), f.get("approach_side")), []).append(meas[f.frame_id])
    rows = []
    for (level, side), vals in sorted(groups.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
        v = np.asarray(vals)
        rows.append({"level": level, "approach_side": side, "n": len(v),
                     "mean": dict(zip(labels, v.mean(axis=0).tolist())),
                     "std": dict(zip(labels, v.std(axis=0, ddof=1).tolist() if len(v) > 1 else [None] * len(labels)))})
    hysteresis = []
    for level in sorted({k[0] for k in groups}):
        if (level, 1) in groups and (level, -1) in groups:
            d = np.mean(groups[(level, 1)], axis=0) - np.mean(groups[(level, -1)], axis=0)
            hysteresis.append({"level": level, "diff_plus_minus": dict(zip(labels, d.tolist())),
                               "norm": float(np.linalg.norm(d[:3] if len(d) >= 3 else d))})
    allv = np.asarray([m for k, g in groups.items() for m in g]) if groups else np.empty((0, len(labels)))
    return {"groups": rows, "hysteresis": hysteresis,
            "pooled_std": dict(zip(labels, allv.std(axis=0, ddof=1).tolist())) if len(allv) > 1 else None}


def analyse_response(session, meas: dict, labels: list[str], axis: str, k_sigma: float) -> dict:
    key = {"x": "x_mm", "z": "z_mm", "pan": "pan_deg", "tilt": "tilt_deg"}[axis]
    golden_ids = {f.frame_id for f in session.golden_frames()}
    static = np.asarray([meas[i] for i in golden_ids if meas.get(i) is not None])
    by_cmd: dict = {}
    for f in session.frames:
        if f.frame_id in golden_ids or meas.get(f.frame_id) is None or not f.get("target"):
            continue
        by_cmd.setdefault(float(f.get("target")[key]), []).append(meas[f.frame_id])
    noise = float(np.linalg.norm(static.std(axis=0, ddof=1))) if len(static) > 1 else None
    base = static.mean(axis=0) if len(static) else np.zeros(len(labels))
    rows, resolvable = [], None
    for cmd in sorted(by_cmd, key=abs):
        v = np.asarray(by_cmd[cmd]).mean(axis=0) - base
        mag = float(np.linalg.norm(v))
        rows.append({"commanded": cmd, "n": len(by_cmd[cmd]), "response": dict(zip(labels, v.tolist())),
                     "magnitude": mag})
        if resolvable is None and cmd != 0 and noise is not None and mag > k_sigma * noise:
            resolvable = abs(cmd)
    return {"axis": axis, "static_noise_norm": noise, "k_sigma": k_sigma, "rows": rows,
            "min_resolvable_step": resolvable}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--measure", choices=("board", "image_shift"), default="board")
    parser.add_argument("--mode", choices=("auto", "repeat", "response"), default="auto")
    parser.add_argument("--axis", choices=("x", "z", "pan", "tilt"), default=None, help="for --mode response")
    parser.add_argument("--k-sigma", type=float, default=3.0)
    add_common_args(parser)
    args = parser.parse_args(argv)
    profile, seed, _ = setup(args)
    session = load_session(args.session, verify=True)
    mode = args.mode
    if mode == "auto":
        mode = "repeat" if any(f.role == "repeat" for f in session.frames) else "response"
    cal = load_calibration(resolve_calibration(session, args.calib)) if args.measure == "board" else None
    spec = BoardSpec.from_profile(profile.ground_truth.board)
    labels = (["cx_mm", "cy_mm", "cz_mm", "rx_deg", "ry_deg", "rz_deg"] if args.measure == "board"
              else ["dx_px", "dy_px"])
    log = make_logger("E0.4", args, profile)
    with log.trial(arm=f"E0.4-{mode}", input=input_record(session, session.frames)) as out:
        meas = measure_frames(session, session.frames, args.measure, cal, spec)
        if mode == "repeat":
            result = analyse_repeat(session, meas, labels)
        else:
            axis = args.axis or (session.manifest.get("grid") or {}).get("axis")
            if axis is None:
                raise SystemExit("--axis is required for --mode response")
            result = analyse_response(session, meas, labels, axis, args.k_sigma)
        result.update(mode=mode, measure=args.measure, labels=labels,
                      n_unmeasured=sum(v is None for v in meas.values()))
        out.update({"mode": mode, "measure": args.measure, "n_unmeasured": result["n_unmeasured"]})
        out["result_file"] = str(write_json(Path(args.out), {"session_id": session.session_id, **result}))
    print(f"E0.4 {mode}: wrote {args.out}; log {log.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
