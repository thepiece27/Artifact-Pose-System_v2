#!/usr/bin/env python3
"""E0.2 / E0.3 / E0.5 measurement noise -> ``configs/noise_model.yaml``.

* sigma_d (E0.2): robust std (1.4826 x MAD) of the Diamond corner coordinates
  over static ``burst`` frames, x and y pooled, per session/distance. The
  corners are exactly those used by the pose pipeline (``detect_diamond``).
* sigma_o (E0.3): 1.4826 x MAD of the residual components of ORB inliers
  after an ORB-only robust pose fit (Diamond pose as initial guess, PnP/Diamond
  gate as in the pipeline) on frames near golden. Needs ``--golden-session``
  (golden frame + stereo right frame). The keypoint jitter between static
  frames is also reported (detection repeatability only).
* E0.5: jitter of the reference-board pose (camera centre mm, rotation deg).

    python tools/experiments/e0_noise.py --burst data/sessions/E0.2/B65 data/sessions/E0.2/B80 \
        --golden-session data/sessions/E1/E1_P001_L0_... --out configs/noise_model.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_common_args, load_calibration, make_logger, resolve_calibration, setup, write_json  # noqa: E402

from app.experiments.session import input_record, load_session  # noqa: E402
from app.experiments.trial_log import git_info, json_safe, sha256_file  # noqa: E402
from app.modules.artifact_pose.common import detect_diamond, extract_orb  # noqa: E402
from app.modules.artifact_pose.geometry import DEFAULT_QUALITY, match_descriptors, project  # noqa: E402
from app.modules.artifact_pose.methods import PoseMethodConfig, _hybrid_observations, build_golden_model  # noqa: E402
from app.modules.artifact_pose.motion import camera_center_in_golden, rotvec  # noqa: E402
from app.modules.artifact_pose.solver import refine_pose_points_only  # noqa: E402
from app.modules.ground_truth.reference_board import (BoardSpec, board_mask, detect_reference_board,  # noqa: E402
                                                      relative_motion_gt)

MAD_K = 1.4826


def robust_sigma(values) -> float | None:
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if len(v) < 4:
        return None
    return float(MAD_K * np.median(np.abs(v - np.median(v))))


def static_frames(session) -> list:
    """Frames taken at the static (golden) pose: no stereo-right frame, no
    commanded offset."""
    out = []
    for f in session.frames:
        target = f.get("target") or {}
        if f.role == "stereo_right" or any(abs(float(v or 0)) > 0 for v in target.values()):
            continue
        out.append(f)
    return out


def diamond_sigma(session, cal) -> dict:
    corners, tz = [], []
    for f in static_frames(session):
        d = detect_diamond(f.load(), cal["K"], cal["D"])
        if d is not None and not d["ambiguous"]:
            corners.append(d["corners"])
            tz.append(float(np.linalg.norm(d["tvec"])))
    if len(corners) < 3:
        return {"n_frames": len(corners), "sigma_px": None}
    c = np.asarray(corners)                       # (n, 4, 2)
    dev = c - np.median(c, axis=0, keepdims=True)
    return {"n_frames": len(corners), "sigma_px": robust_sigma(dev),
            "sigma_x_px": robust_sigma(dev[..., 0]), "sigma_y_px": robust_sigma(dev[..., 1]),
            "std_px": float(np.std(dev)), "diamond_distance_m": float(np.median(tz))}


def board_jitter(session, cal, spec: BoardSpec) -> dict:
    poses = []
    for f in static_frames(session):
        det = detect_reference_board(f.load(), cal["K"], cal["D"], spec)
        if det["ok"]:
            poses.append(det)
    if len(poses) < 3:
        return {"n_frames": len(poses)}
    c, th = [], []
    for p in poses[1:]:
        dT = relative_motion_gt(poses[0], p)
        c.append(camera_center_in_golden(dT) * 1000)
        th.append(np.rad2deg(rotvec(dT[:3, :3])))
    c, th = np.asarray(c), np.asarray(th)
    return {"n_frames": len(poses), "c_std_mm": c.std(axis=0, ddof=1).tolist(),
            "c_norm_rms_mm": float(np.sqrt(np.mean(np.sum((c - c.mean(0)) ** 2, axis=1)))),
            "rot_std_deg": th.std(axis=0, ddof=1).tolist(),
            "reproj_rms_px_median": float(np.median([p["reproj_rms_px"] for p in poses]))}


def keypoint_jitter(session, cal, max_frames: int = 10) -> float | None:
    frames = static_frames(session)[:max_frames]
    feats = [extract_orb(f.load(), backend="python")[1:] for f in frames]
    disp = []
    for (d0, k0), (d1, k1) in zip(feats[:-1], feats[1:]):
        for m in match_descriptors(d0, d1, ratio=0.7, max_hamming=40):
            disp.append(k1[m.trainIdx] - k0[m.queryIdx])
    if len(disp) < 10:
        return None
    s = robust_sigma(np.asarray(disp))
    return None if s is None else s / np.sqrt(2.0)   # difference of two noisy positions


def orb_sigma(burst_sessions, golden_session, cal, profile, spec: BoardSpec) -> dict:
    gl = golden_session.golden_frames()[0].load()
    right_frame = golden_session.stereo_right()
    if right_frame is None:
        raise SystemExit("--golden-session needs a stereo_right_frame_id (capture_session.py stereo/grid)")
    gr = right_frame.load()
    K, D = cal["K"], cal["D"]
    masks = tuple(board_mask(im.shape, detect_reference_board(im, K, D, spec)["hull_polygon_px"],
                             spec.mask_dilate_px) for im in (gl, gr))
    cfg = PoseMethodConfig(method="M3", backend="python")
    import tempfile
    gm = build_golden_model(gl, gr, K, D, cfg=cfg, workdir=tempfile.mkdtemp(prefix="e0noise_"),
                            stereo_source=profile.golden.stereo_source,
                            baseline_m=profile.golden.baseline_mm / 1000.0, masks=masks)
    residuals, per_frame = [], []
    for s in burst_sessions:
        for f in static_frames(s):
            img = f.load()
            m = board_mask(img.shape, detect_reference_board(img, K, D, spec)["hull_polygon_px"], spec.mask_dilate_px)
            obs = _hybrid_observations(img, gm, cfg, m, "python")
            d = obs["diamond"]
            if d is None or d["ambiguous"] or len(obs["o3"]) < DEFAULT_QUALITY.min_orb_inliers:
                continue
            fit = refine_pose_points_only(d["rvec"], d["tvec"], obs["o3"], obs["o2"], K, D, sigma_px=1.0,
                                          huber_delta=2.0, quality=DEFAULT_QUALITY)
            if not fit["accepted"]:
                continue
            res = project(obs["o3"], fit["rvec"], fit["tvec"], K, D) - obs["o2"]
            keep = np.linalg.norm(res, axis=1) <= 3 * DEFAULT_QUALITY.max_reprojection_px
            residuals.append(res[keep])
            per_frame.append({"frame_id": f.frame_id, "n_inliers": int(keep.sum()),
                              "sigma_px": robust_sigma(res[keep])})
    if not residuals:
        return {"sigma_px": None, "n_frames": 0}
    allres = np.vstack(residuals)
    return {"sigma_px": robust_sigma(allres), "n_frames": len(per_frame), "n_residuals": int(len(allres)),
            "golden_stereo_source": profile.golden.stereo_source, "per_frame": per_frame}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--burst", nargs="+", required=True, help="burst session directories (E0.2/E0.5)")
    parser.add_argument("--golden-session", default=None, help="session with golden + stereo right (E0.3)")
    parser.add_argument("--out", default="configs/noise_model.yaml")
    parser.add_argument("--report", default=None, help="JSON report (default: <out>.report.json)")
    parser.add_argument("--allow-partial", action="store_true", help="write even if sigma_o is missing")
    parser.add_argument("--force", action="store_true", help="overwrite an existing noise model")
    add_common_args(parser)
    args = parser.parse_args(argv)
    profile, seed, _ = setup(args)
    sessions = [load_session(p, verify=True) for p in args.burst]
    golden = load_session(args.golden_session, verify=True) if args.golden_session else None
    cal = load_calibration(resolve_calibration(sessions[0], args.calib))
    spec = BoardSpec.from_profile(profile.ground_truth.board)
    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[2] / out
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force to replace it (the old file is kept as .bak)")
    log = make_logger("E0.2", args, profile)
    frames = [f for s in sessions for f in s.frames]
    with log.trial(arm="E0-noise", input={"sessions": [s.session_id for s in sessions],
                                          "golden_session": golden.session_id if golden else None,
                                          **input_record(sessions[0], frames)}) as rec:
        per_session = []
        for s in sessions:
            per_session.append({"session_id": s.session_id, "distance_cm": s.manifest.get("distance_cm"),
                                "lighting": s.manifest.get("lighting"),
                                "diamond": diamond_sigma(s, cal), "board_jitter": board_jitter(s, cal, spec),
                                "orb_keypoint_jitter_px": keypoint_jitter(s, cal)})
        sig_d = [p["diamond"]["sigma_px"] for p in per_session if p["diamond"]["sigma_px"]]
        diamond_all = float(np.median(sig_d)) if sig_d else None
        orb = orb_sigma(sessions, golden, cal, profile, spec) if golden else {"sigma_px": None}
        if diamond_all is None:
            raise SystemExit("No Diamond detected in the burst sessions")
        if orb["sigma_px"] is None and not args.allow_partial:
            raise SystemExit("sigma_o unavailable (give --golden-session) or pass --allow-partial")
        model = {
            "schema": "cita.noise_model.v1",
            "diamond_sigma_px": diamond_all,
            "orb_sigma_px": orb["sigma_px"],
            "estimator": "1.4826*MAD; diamond = median over sessions; orb = pooled ORB-only fit residuals",
            "by_distance": [{"session_id": p["session_id"], "distance_cm": p["distance_cm"],
                             "diamond_sigma_px": p["diamond"]["sigma_px"],
                             "orb_keypoint_jitter_px": p["orb_keypoint_jitter_px"]} for p in per_session],
            "board_jitter": {p["session_id"]: p["board_jitter"] for p in per_session},
            "sources": [{"session_id": s.session_id, "manifest_sha256": sha256_file(s.root / "manifest.json")}
                        for s in sessions + ([golden] if golden else [])],
            "calibration_sha256": cal["sha256"],
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tool_commit": git_info()["commit"],
        }
        body = json.dumps(model, sort_keys=True, default=str)
        model["content_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
        import yaml
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            out.replace(out.with_suffix(out.suffix + ".bak"))
        model = json_safe(model)
        out.write_text(yaml.safe_dump(model, sort_keys=False, allow_unicode=True), encoding="utf-8")
        report = Path(args.report) if args.report else out.with_suffix(".report.json")
        write_json(report, {"model": model, "per_session": per_session, "orb": orb})
        rec.update({"diamond_sigma_px": diamond_all, "orb_sigma_px": orb["sigma_px"],
                    "noise_model": str(out), "report": str(report), "content_sha256": model["content_sha256"]})
    print(f"sigma_d = {diamond_all}, sigma_o = {orb['sigma_px']} -> {out}; log {log.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
