#!/usr/bin/env python3
"""E0 stereo check (R6): which golden stereo source should M4/M5/proposed use?

    python tools/experiments/e0_stereo_check.py --sessions data/sessions/E0.3/E0.3_P000_* [--out results/E0/stereo_check.json]

For every session with a golden + stereo-right frame (``grid --with-stereo``)
it reports the Diamond-estimated baseline vs the commanded one (mm, %), and the
angles between the three directions: Diamond (T_right_left of the two Diamond
poses), epipolar (pure-translation RANSAC on ORB matches) and the camera x axis,
plus the number of epipolar inliers.

DECISION RULE (fixed before any E1 evaluation, write it into CHANGELOG_CITA.md):
if median |baseline_Diamond - commanded| > 2 mm OR median angle(Diamond,
epipolar) > 1 deg, then proposed / M4 / M5 use ``golden.stereo_source:
commanded_epipolar``; otherwise keep ``diamond``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SERVER, add_common_args, load_calibration, make_logger, resolve_calibration, setup, write_json  # noqa: E402

from app.experiments.session import input_record, load_session  # noqa: E402
from app.modules.artifact_pose.common import detect_diamond, extract_orb  # noqa: E402
from app.modules.artifact_pose.geometry import DEFAULT_QUALITY, match_descriptors, refine_pose  # noqa: E402
from app.modules.artifact_pose.stereo_direction import (axis_angle_deg, diamond_stereo_transform,  # noqa: E402
                                                        estimate_translation_direction, normalized_rays)
from app.modules.ground_truth.reference_board import BoardSpec, board_mask, detect_reference_board  # noqa: E402

BASELINE_TOL_MM = 2.0
ANGLE_TOL_DEG = 1.0


def check_session(session, cal, spec: BoardSpec, commanded_mm: float) -> dict:
    left_f, right_f = session.golden_frames()[0], session.stereo_right()
    if right_f is None:
        return {"session_id": session.session_id, "error": "no stereo_right_frame_id"}
    L, R = left_f.load(), right_f.load()
    K, D = cal["K"], cal["D"]
    out = {"session_id": session.session_id, "painting_id": session.manifest.get("painting_id"),
           "commanded_baseline_mm": commanded_mm}
    dl = detect_diamond(L, K, D, DEFAULT_QUALITY)
    if dl is not None and not dl["ambiguous"]:
        fit = refine_pose(dl["rvec"], dl["tvec"], dl["obj_pts"], dl["corners"], np.empty((0, 3)), np.empty((0, 2)),
                          K, D, DEFAULT_QUALITY)
        dl = {**dl, "rvec": fit["rvec"], "tvec": fit["tvec"]} if fit["accepted"] else None
    T_d = diamond_stereo_transform(dl, R, K, D, DEFAULT_QUALITY) if dl is not None and not dl.get("ambiguous") else None
    masks = [board_mask(im.shape, detect_reference_board(im, K, D, spec)["hull_polygon_px"], spec.mask_dilate_px)
             for im in (L, R)]
    _, d1, k1 = extract_orb(L, backend="python", mask=masks[0])
    _, d2, k2 = extract_orb(R, backend="python", mask=masks[1])
    matches = match_descriptors(d1, d2)
    il = np.array([m.queryIdx for m in matches], dtype=int)
    ir = np.array([m.trainIdx for m in matches], dtype=int)
    est = estimate_translation_direction(normalized_rays(k1[il], K, D), normalized_rays(k2[ir], K, D),
                                         focal_px=float((K[0, 0] + K[1, 1]) / 2))
    t_epi = est["t_dir"]
    out.update(epipolar_inliers=est["n_inliers"], epipolar_matches=est["n_matches"],
               epipolar_rms_px=est["rms_epipolar_px"], epipolar_direction=t_epi.tolist(),
               angle_epipolar_x_deg=axis_angle_deg(t_epi, [1.0, 0.0, 0.0]))
    if T_d is not None:
        bd = float(np.linalg.norm(T_d[:3, 3]) * 1000)
        out.update(diamond_baseline_mm=bd, diamond_baseline_error_mm=bd - commanded_mm,
                   diamond_baseline_error_pct=100.0 * (bd - commanded_mm) / commanded_mm,
                   diamond_direction=(T_d[:3, 3] / np.linalg.norm(T_d[:3, 3])).tolist(),
                   angle_diamond_epipolar_deg=axis_angle_deg(T_d[:3, 3], t_epi),
                   angle_diamond_x_deg=axis_angle_deg(T_d[:3, 3], [1.0, 0.0, 0.0]))
    else:
        out["diamond"] = "not detected / ambiguous in the left or right golden frame"
    return out


def decide(rows: list[dict]) -> dict:
    base = [abs(r["diamond_baseline_error_mm"]) for r in rows if r.get("diamond_baseline_error_mm") is not None]
    ang = [r["angle_diamond_epipolar_deg"] for r in rows if r.get("angle_diamond_epipolar_deg") is not None]
    med_b = float(np.median(base)) if base else None
    med_a = float(np.median(ang)) if ang else None
    use_epi = (med_b is not None and med_b > BASELINE_TOL_MM) or (med_a is not None and med_a > ANGLE_TOL_DEG)
    return {"median_abs_diamond_baseline_error_mm": med_b, "median_angle_diamond_epipolar_deg": med_a,
            "rule": f"commanded_epipolar if median |b_Diamond - b_cmd| > {BASELINE_TOL_MM} mm "
                    f"or median angle(Diamond, epipolar) > {ANGLE_TOL_DEG} deg, else diamond",
            "recommended_stereo_source": "commanded_epipolar" if use_epi else "diamond",
            "n_sessions_with_diamond": len(base)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions", nargs="+", required=True)
    parser.add_argument("--out", default=str(SERVER / "results" / "E0" / "stereo_check.json"))
    add_common_args(parser)
    args = parser.parse_args(argv)
    profile, seed, _ = setup(args)
    sessions = [load_session(p, verify=True) for p in args.sessions]
    cal = load_calibration(resolve_calibration(sessions[0], args.calib))
    spec = BoardSpec.from_profile(profile.ground_truth.board)
    log = make_logger("E0.3", args, profile)
    with log.trial(arm="E0-stereo-check", input={"sessions": [s.session_id for s in sessions],
                                                  **input_record(sessions[0], [f for s in sessions for f in s.frames])}) as rec:
        rows = [check_session(s, cal, spec, profile.golden.baseline_mm) for s in sessions]
        decision = decide(rows)
        result = {"experiment": "E0", "sessions": rows, "decision": decision}
        path = write_json(Path(args.out), result)
        rec.update(decision=decision, result_file=str(path))
    for r in rows:
        print(f"{r['session_id']}: Diamond baseline {r.get('diamond_baseline_mm')} mm, "
              f"angle(Diamond, epipolar) {r.get('angle_diamond_epipolar_deg')} deg, "
              f"angle(epipolar, x) {r.get('angle_epipolar_x_deg')} deg, inliers {r.get('epipolar_inliers')}")
    print(f"RULE: {decision['rule']}")
    print(f"=> recommended golden.stereo_source = {decision['recommended_stereo_source']} "
          f"(record this decision in docs/cita/CHANGELOG_CITA.md BEFORE evaluating E1); {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
