#!/usr/bin/env python3
"""E0.3 (+ E0.2 / E0.5) pixel noise model -> ``configs/noise_model.yaml`` (schema v2).

``--estimator vce`` (default, R2): variance-component estimation on every
non-golden frame of the E0.3 sessions (calibration painting P000, small grid,
golden + stereo). sigma_d and sigma_o are the scales of the post-fit residuals
of the SAME hybrid pose problem, so both include systematic errors.

``--estimator jitter`` (ablation, profile ``M4-jitter``): sigma_d = static
Diamond corner jitter (temporal noise only), sigma_o = ORB-only fit residuals
(the v1.1 behaviour). Kept to show in the paper why jitter is the wrong scale.

Side information always reported: ``diamond_jitter_px``,
``orb_keypoint_jitter_px``, board-pose jitter (E0.5) from ``--burst`` sessions
(or the static frames of the E0.3 sessions). The golden is built with
``--stereo-source`` (default: profile ``golden.stereo_source``), recorded as
``golden_stereo_source``; M4/M5 refuse a model built with another source.

    python tools/experiments/e0_noise.py --golden-session data/sessions/E0.3/E0.3_P000_* \
        [--burst data/sessions/E0.2/*] --out configs/noise_model.yaml
    python tools/experiments/e0_noise.py --estimator jitter --golden-session ... --out configs/noise_model_jitter.yaml
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
from app.modules.artifact_pose.noise_vce import FrameObservation, estimate_vce  # noqa: E402
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


def golden_for(session, cal, stereo_source: str, baseline_mm: float, spec: BoardSpec):
    """Hybrid golden of a session (golden + stereo-right frames), board masked."""
    import tempfile
    gl = session.golden_frames()[0].load()
    right_frame = session.stereo_right()
    if right_frame is None:
        raise SystemExit(f"{session.session_id}: needs a stereo_right_frame_id (capture_session.py grid --with-stereo)")
    gr = right_frame.load()
    K, D = cal["K"], cal["D"]
    masks = tuple(board_mask(im.shape, detect_reference_board(im, K, D, spec)["hull_polygon_px"],
                             spec.mask_dilate_px) for im in (gl, gr))
    cfg = PoseMethodConfig(method="M3", backend="python")
    gm = build_golden_model(gl, gr, K, D, cfg=cfg, workdir=tempfile.mkdtemp(prefix="e0noise_"),
                            stereo_source=stereo_source, baseline_m=baseline_mm / 1000.0, masks=masks)
    return gm, cfg


def observations(session, gm, cfg, cal, spec: BoardSpec, frames) -> list:
    out = []
    K, D = cal["K"], cal["D"]
    for f in frames:
        img = f.load()
        m = board_mask(img.shape, detect_reference_board(img, K, D, spec)["hull_polygon_px"], spec.mask_dilate_px)
        obs = _hybrid_observations(img, gm, cfg, m, "python")
        d = obs["diamond"]
        if d is None or d["ambiguous"] or len(obs["o3"]) < DEFAULT_QUALITY.min_orb_inliers:
            continue
        out.append(FrameObservation(d["obj_pts"], d["corners"], obs["o3"], obs["o2"], d["rvec"], d["tvec"],
                                    session_id=session.session_id, frame_id=f.frame_id,
                                    meta={"T_gw": gm.T_gw}))
    return out


def orb_sigma_jitter_estimator(obs_list: list, cal) -> dict:
    """v1.1 sigma_o: 1.4826 MAD of ORB residuals after an ORB-only fit."""
    K, D = cal["K"], cal["D"]
    residuals, per_frame = [], []
    for o in obs_list:
        fit = refine_pose_points_only(o.rvec0, o.tvec0, o.orb_xyz, o.orb_xy, K, D, sigma_px=1.0,
                                      huber_delta=2.0, quality=DEFAULT_QUALITY)
        if not fit["accepted"]:
            continue
        res = project(o.orb_xyz, fit["rvec"], fit["tvec"], K, D) - o.orb_xy
        keep = np.linalg.norm(res, axis=1) <= 3 * DEFAULT_QUALITY.max_reprojection_px
        residuals.append(res[keep])
        per_frame.append({"frame_id": o.frame_id, "n_inliers": int(keep.sum()), "sigma_px": robust_sigma(res[keep])})
    if not residuals:
        return {"sigma_px": None, "n_frames": 0}
    allres = np.vstack(residuals)
    return {"sigma_px": robust_sigma(allres), "n_frames": len(per_frame), "n_residuals": int(len(allres)),
            "per_frame": per_frame}


def pose_sigma_summary(obs_list, vce: dict, cal) -> dict:
    """Median single-frame camera-centre / rotation sigma at the VCE solution
    (used by run_reloc_trials --noise-from)."""
    from app.modules.artifact_pose.covariance import pose_covariance
    from app.modules.artifact_pose.motion import as_T
    from app.modules.artifact_pose.solver import Block
    sc, sr = [], []
    for idx, r, t in vce["final_poses"]:
        o = obs_list[idx]
        blocks = [Block(o.diamond_xyz, o.diamond_xy, vce["sigma_d_px"], False, "diamond"),
                  Block(o.orb_xyz, o.orb_xy, vce["sigma_o_px"], True, "orb")]
        cov = pose_covariance(as_T(r, t), blocks, cal["K"], cal["D"], huber_delta=2.0, T_gw=o.meta["T_gw"],
                              max_reprojection_px=DEFAULT_QUALITY.max_reprojection_px)
        if cov.get("ok"):
            sc.append(cov["sigma_center_mm"])
            sr.append(cov["sigma_rot_deg"])
    if not sc:
        return {}
    return {"pose_sigma_center_mm_median": np.median(sc, axis=0).tolist(),
            "pose_sigma_rot_deg_median": np.median(sr, axis=0).tolist(), "pose_sigma_n_frames": len(sc)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden-session", nargs="+", default=[],
                        help="E0.3 sessions with golden + stereo right (grid --with-stereo on painting P000)")
    parser.add_argument("--burst", nargs="*", default=[], help="optional burst sessions (E0.2/E0.5 jitter, board)")
    parser.add_argument("--estimator", choices=("vce", "jitter"), default="vce")
    parser.add_argument("--stereo-source", default=None, choices=("diamond", "commanded", "commanded_epipolar"),
                        help="golden.stereo_source used to build the golden (default: profile)")
    parser.add_argument("--out", default="configs/noise_model.yaml")
    parser.add_argument("--report", default=None, help="JSON report (default: <out>.report.json)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing noise model")
    add_common_args(parser)
    args = parser.parse_args(argv)
    profile, seed, _ = setup(args)
    if not args.golden_session:
        raise SystemExit("--golden-session (E0.3 session with stereo) is required for both estimators")
    goldens = [load_session(p, verify=True) for p in args.golden_session]
    bursts = [load_session(p, verify=True) for p in args.burst]
    cal = load_calibration(resolve_calibration(goldens[0], args.calib))
    spec = BoardSpec.from_profile(profile.ground_truth.board)
    stereo_source = args.stereo_source or profile.golden.stereo_source
    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[2] / out
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force to replace it (the old file is kept as .bak)")
    log = make_logger("E0.3", args, profile)
    all_sessions = goldens + bursts
    frames_all = [f for s in all_sessions for f in s.frames]
    with log.trial(arm=f"E0-noise-{args.estimator}", input={
            "golden_sessions": [s.session_id for s in goldens], "bursts": [s.session_id for s in bursts],
            "estimator": args.estimator, "golden_stereo_source": stereo_source,
            **input_record(goldens[0], frames_all)}) as rec:
        # Side information: temporal jitter, keypoint jitter, board jitter (E0.5).
        per_session = []
        for s in (bursts or goldens):
            per_session.append({"session_id": s.session_id, "distance_cm": s.manifest.get("distance_cm"),
                                "lighting": s.manifest.get("lighting"),
                                "diamond": diamond_sigma(s, cal), "board_jitter": board_jitter(s, cal, spec),
                                "orb_keypoint_jitter_px": keypoint_jitter(s, cal)})
        jit = [p["diamond"]["sigma_px"] for p in per_session if p["diamond"]["sigma_px"] is not None]
        kp = [p["orb_keypoint_jitter_px"] for p in per_session if p["orb_keypoint_jitter_px"] is not None]
        diamond_jitter = float(np.median(jit)) if jit else None
        keypoint_jit = float(np.median(kp)) if kp else None
        # Observations of every non-golden frame of the E0.3 sessions.
        obs_list = []
        for s in goldens:
            gm, cfg = golden_for(s, cal, stereo_source, profile.golden.baseline_mm, spec)
            obs_list += observations(s, gm, cfg, cal, spec, s.evaluation_frames())
        if not obs_list:
            raise SystemExit("No usable frame (Diamond + verified ORB) in the E0.3 sessions")
        vce, orb_jit = None, None
        if args.estimator == "vce":
            vce = estimate_vce(obs_list, cal["K"], cal["D"], quality=DEFAULT_QUALITY)
            sigma_d, sigma_o = vce["sigma_d_px"], vce["sigma_o_px"]
            if vce["warning"]:
                print(f"WARNING: {vce['warning']}", file=sys.stderr)
        else:
            orb_jit = orb_sigma_jitter_estimator(obs_list, cal)
            sigma_d, sigma_o = diamond_jitter, orb_jit["sigma_px"]
            if sigma_d is None or sigma_o is None:
                raise SystemExit("jitter estimator: Diamond jitter or ORB residual sigma unavailable")
        model = {
            "schema": "cita.noise_model.v2",
            "estimator": args.estimator,
            "golden_stereo_source": stereo_source,
            "diamond_sigma_px": float(sigma_d),
            "orb_sigma_px": float(sigma_o),
            "diamond_jitter_px": diamond_jitter,
            "orb_keypoint_jitter_px": keypoint_jit,
            "vce": None if vce is None else {k: vce[k] for k in (
                "converged", "n_iterations", "iterations", "redundancy_d", "redundancy_o", "n_frames",
                "n_residuals_d", "n_residuals_o", "warning")},
            "by_distance": [{"session_id": p["session_id"], "distance_cm": p["distance_cm"],
                             "diamond_jitter_px": p["diamond"]["sigma_px"],
                             "orb_keypoint_jitter_px": p["orb_keypoint_jitter_px"]} for p in per_session],
            "board_jitter": {p["session_id"]: p["board_jitter"] for p in per_session},
            "sources": [{"session_id": s.session_id, "manifest_sha256": sha256_file(s.root / "manifest.json"),
                         "painting_id": s.manifest.get("painting_id")} for s in all_sessions],
            "calibration_sha256": cal["sha256"],
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tool_commit": git_info()["commit"],
        }
        if vce is not None:
            model.update(pose_sigma_summary(obs_list, vce, cal))
        body = json.dumps(json_safe(model), sort_keys=True, default=str)
        model["content_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
        import yaml
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            out.replace(out.with_suffix(out.suffix + ".bak"))
        model = json_safe(model)
        out.write_text(yaml.safe_dump(model, sort_keys=False, allow_unicode=True), encoding="utf-8")
        report = Path(args.report) if args.report else out.with_suffix(".report.json")
        write_json(report, {"model": model, "per_session": per_session, "orb_jitter_estimator": orb_jit})
        rec.update({"estimator": args.estimator, "diamond_sigma_px": sigma_d, "orb_sigma_px": sigma_o,
                    "diamond_jitter_px": diamond_jitter, "golden_stereo_source": stereo_source,
                    "noise_model": str(out), "report": str(report), "content_sha256": model["content_sha256"]})
    print(f"[{args.estimator}] sigma_d = {sigma_d:.4f} px, sigma_o = {sigma_o:.4f} px "
          f"(diamond jitter {diamond_jitter}) -> {out}; log {log.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
