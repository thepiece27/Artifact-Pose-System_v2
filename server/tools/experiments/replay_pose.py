#!/usr/bin/env python3
"""E1 offline replay: run pose methods on a captured session.

    python tools/experiments/replay_pose.py --session data/sessions/E1/E1_P001_L0_... \
        --methods M1,M2,M2-full,M3,M4,M5 [--noise-model configs/noise_model.yaml] \
        [--override golden.stereo_source=commanded] [--run-id E1_P001]

One JSONL record (cita.trial.v1) per (frame, method) in
``logs/experiments/E1/<run_id>.jsonl``; arm ``GT`` holds the reference-board
ground truth of each frame. M5 uses groups of ``pose.frames`` static frames
(same ``pose_index``). The board region is masked out of every method.
Each method uses its registry profile (configs/profiles/<arm>.yaml).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_common_args, load_calibration, make_logger, resolve_calibration, setup  # noqa: E402

from app.experiments import registry  # noqa: E402
from app.experiments.profiles import load_profile  # noqa: E402
from app.experiments.session import input_record, load_session  # noqa: E402
from app.modules.artifact_pose.geometry import DEFAULT_QUALITY  # noqa: E402
from app.modules.artifact_pose.methods import PoseMethodConfig, build_golden_model, estimate_pose  # noqa: E402
from app.modules.artifact_pose.motion import average_poses, motion_summary  # noqa: E402
from app.modules.ground_truth.reference_board import (BoardSpec, board_mask, detect_reference_board,  # noqa: E402
                                                      relative_motion_gt)


def parse_overrides(items: list[str]) -> dict:
    import yaml
    out: dict = {}
    for item in items or []:
        key, _, value = item.partition("=")
        if not key or not _:
            raise SystemExit(f"--override expects key=value, got {item!r}")
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out


def method_profile(arm: str, overrides: dict, noise_model: str | None):
    """Registry profile of ``arm`` + CLI overrides (noise model only matters for M4/M5)."""
    ov = dict(overrides)
    if noise_model:
        ov["pose"] = {**ov.get("pose", {}), "noise_model": noise_model}
    return load_profile(registry.get(arm).default_profile, overrides=ov or None)


def board_state(frames, K, D, spec: BoardSpec) -> dict:
    out = {}
    for f in frames:
        img = f.load()
        det = detect_reference_board(img, K, D, spec)
        out[f.frame_id] = {"det": det, "mask": board_mask(img.shape, det.get("hull_polygon_px"),
                                                          spec.mask_dilate_px)}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True)
    parser.add_argument("--methods", default="M1,M2,M3,M4,M5")
    parser.add_argument("--noise-model", default=None, help="override pose.noise_model of M4/M5")
    parser.add_argument("--override", action="append", default=[],
                        help="profile override applied to every method, e.g. golden.stereo_source=commanded")
    parser.add_argument("--max-groups", type=int, default=None, help="limit pose groups (smoke tests)")
    parser.add_argument("--no-verify", action="store_true", help="skip SHA-256 verification of images")
    add_common_args(parser)
    args = parser.parse_args(argv)
    base, seed, _ = setup(args)
    session = load_session(args.session, verify=not args.no_verify)
    cal = load_calibration(resolve_calibration(session, args.calib))
    K, D = cal["K"], cal["D"]
    spec = BoardSpec.from_profile(base.ground_truth.board)
    overrides = parse_overrides(args.override)
    arms = [a.strip() for a in args.methods.split(",") if a.strip()]
    profiles = {a: method_profile(a, overrides, args.noise_model) for a in arms}
    log = make_logger("E1", args, base)

    golden_frames = session.golden_frames()
    right = session.stereo_right()
    if not golden_frames or right is None:
        raise SystemExit("Session needs golden.frame_ids and golden.stereo_right_frame_id")
    left_img, right_img = golden_frames[0].load(), right.load()
    boards = board_state(session.frames, K, D, spec)
    golden_dets = [boards[f.frame_id]["det"] for f in golden_frames if boards[f.frame_id]["det"]["ok"]]
    T_cb_golden = average_poses([np.asarray(d["T_cb"]) for d in golden_dets]) if golden_dets else None
    masks_golden = (boards[golden_frames[0].frame_id]["mask"], boards[right.frame_id]["mask"])

    # ---- ground truth records ----------------------------------------------
    groups = session.groups()
    keys = sorted(groups, key=lambda k: (k is None, k))[: args.max_groups]
    for key in keys:
        for f in groups[key]:
            det = boards[f.frame_id]["det"]
            out = {"board_ok": bool(det["ok"]), "n_corners": det["n_corners"],
                   "reproj_rms_px": det["reproj_rms_px"], "reason": det["reason"]}
            status = "ok"
            if det["ok"] and T_cb_golden is not None:
                dT = relative_motion_gt(T_cb_golden, det)
                out.update(motion_summary(dT))
            else:
                status = "no_gt"
            log.log(arm="GT", input={**input_record(session, [f]), "pose_index": f.pose_index,
                                     "reps": [f.get("rep")], "target": f.get("target"),
                                     "odometry": f.get("odometry")},
                    output=out, status=status)

    # ---- methods --------------------------------------------------------------
    cache: dict = {}
    workroot = Path(tempfile.mkdtemp(prefix="replay_pose_"))
    for arm in arms:
        prof = profiles[arm]
        try:
            cfg = PoseMethodConfig.from_profile(prof, quality=DEFAULT_QUALITY)
        except ValueError as exc:
            log.log(arm=arm, input=input_record(session, []), status="error", error=str(exc))
            print(f"[{arm}] configuration error: {exc}", file=sys.stderr)
            continue
        gkey = (prof.golden.stereo_source, prof.golden.baseline_mm, cfg.method == "M2", cfg.resize_to,
                cfg.ratio if cfg.method == "M2" else None)
        t0 = time.perf_counter()
        if gkey not in cache:
            try:
                cache[gkey] = build_golden_model(
                    left_img, right_img, K, D, cfg=cfg, workdir=workroot / f"g{len(cache)}",
                    stereo_source=prof.golden.stereo_source, baseline_m=prof.golden.baseline_mm / 1000,
                    masks=masks_golden, build_hybrid=True, build_markerless=(cfg.method == "M2"))
            except ValueError as exc:
                cache[gkey] = exc
        golden = cache[gkey]
        if isinstance(golden, Exception):
            log.log(arm=arm, input=input_record(session, golden_frames + [right]), status="golden_failed",
                    error=str(golden), output={"profile": prof.name})
            continue
        log.log(arm=f"{arm}:golden", input=input_record(session, golden_frames + [right]),
                output={"profile": prof.name, "profile_hash": prof.profile_hash, **golden.meta,
                        "noise_model": cfg.noise_model and {k: cfg.noise_model[k] for k in
                                                           ("diamond_sigma_px", "orb_sigma_px", "_sha256")},
                        "sigmas": [cfg.diamond_sigma_px, cfg.orb_sigma_px]},
                timing_ms={"total": (time.perf_counter() - t0) * 1000})
        for key in keys:
            frames = groups[key]
            batches = ([frames[:cfg.frames]] if cfg.method == "M5" else [[f] for f in frames])
            for batch in batches:
                if len(batch) < cfg.frames:
                    log.log(arm=arm, input=input_record(session, batch), status="insufficient_frames")
                    continue
                imgs = [f.load() for f in batch]
                masks = [boards[f.frame_id]["mask"] for f in batch]
                inp = {**input_record(session, batch), "pose_index": key, "target": batch[0].get("target"),
                       "reps": [f.get("rep") for f in batch]}
                with log.trial(arm=arm, input=inp, profile_name=prof.name, profile_hash_arm=prof.profile_hash) as out:
                    est = estimate_pose(imgs, golden, cfg, masks=masks)
                    out.update(est.to_json())
                    out["_status"] = "ok" if est.ok else est.status
                    out["_timings"] = {k: v for k, v in est.timings_ms.items() if k != "total"}
    print(f"E1 replay written: {log.path} ({log.count} records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
