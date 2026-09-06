#!/usr/bin/env python3
"""One-image correction proposal for a moving camera and a fixed artifact.

No motor actuation. Pose acceptance is not authorization to move hardware.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from .common import (GOLDEN_POSE_FILE, PARAMS_FILE, VIS_DIR,
                     calibration_fingerprint, detect_diamond, extract_orb, load_golden_pose,
                     read_calibration, resolve_backend, validate_image)
from .geometry import (DEFAULT_QUALITY, camera_deviation, load_quality, match_descriptors,
                       refine_pose, verify_correspondences)


def run_correction_step(image, K, D, golden, *, strategy=None, backend="python",
                        quality=DEFAULT_QUALITY):
    started = time.perf_counter()
    image = validate_image(image)
    native = resolve_backend(backend)
    meta = golden["metadata"]
    if calibration_fingerprint(K, D, (image.shape[1], image.shape[0])) != meta["calibration_sha256"]:
        raise ValueError("Current image/calibration does not match golden")
    strategy = strategy or meta["strategy"]
    if strategy not in ("diamond", "hybrid") or (strategy == "hybrid" and meta["strategy"] != "hybrid"):
        raise ValueError("Hybrid correction requires a hybrid reference")
    result = {"accepted": False, "status": "diamond_not_found_or_rejected", "diamond": None,
              "hybrid": None, "deviation": None, "num_matches": 0,
              "motor_command": {"enabled": False, "reason": "Actuator axes, signs, pivot and hand-eye transform are not calibrated; no hardware driver"},
              "strategy_requested": strategy, "strategy_used": None}
    def finish():
        result["elapsed_ms"] = (time.perf_counter() - started) * 1000
        return result
    diamond = detect_diamond(image, K, D, quality)
    result["diamond"] = diamond
    if diamond is None:
        return finish()
    # Conservative: no previous-pose prior silently resolves a planar flip.
    if diamond["ambiguous"]:
        result["status"] = "ambiguous_diamond_pose"
        return finish()
    o3, o2 = np.empty((0, 3)), np.empty((0, 2))
    if strategy == "hybrid":
        _, desc, kp = extract_orb(image, backend=backend)
        matches = match_descriptors(desc, golden["descriptors"])
        result["num_matches"] = len(matches)
        ci = np.array([m.queryIdx for m in matches], dtype=int)
        gi = np.array([m.trainIdx for m in matches], dtype=int)
        matched3, matched2 = golden["points_3d"][gi], kp[ci]
        keep = verify_correspondences(matched3, matched2, diamond["rvec"], diamond["tvec"], K, D, quality)
        if int(keep.sum()) >= quality.min_orb_inliers:
            o3, o2 = matched3[keep], matched2[keep]
    result["num_verified_matches"] = len(o3)
    fit = refine_pose(diamond["rvec"], diamond["tvec"], diamond["obj_pts"], diamond["corners"],
                      o3, o2, K, D, quality, native=native)
    result["strategy_used"] = "hybrid" if len(o3) else "diamond"
    if not fit["accepted"] and len(o3):
        result["hybrid_rejection"] = fit
        fit = refine_pose(diamond["rvec"], diamond["tvec"], diamond["obj_pts"], diamond["corners"],
                          np.empty((0, 3)), np.empty((0, 2)), K, D, quality, native=native)
        result["strategy_used"] = "diamond"
    result["hybrid"] = fit  # Retained result key; strategy_used distinguishes the objective.
    if not fit["accepted"]:
        result["status"] = "refinement_rejected"
        return finish()
    result["accepted"] = True
    result["status"] = "accepted_diamond_fallback" if strategy != result["strategy_used"] else "accepted"
    result["deviation"] = camera_deviation(fit["rvec"], fit["tvec"], golden["rvec"], golden["tvec"], quality)
    # Keep the server/UI contract while exposing the corrected camera-frame
    # geometry as the authoritative fields. Hardware remains fail-closed.
    deviation = result["deviation"]
    correction = np.asarray(deviation["translation_correction_world_m"], dtype=float)
    deviation.update({
        "delta_x": float(correction[0]), "delta_y": float(correction[1]),
        "delta_z": float(correction[2]),
        "translation_mag": float(deviation["translation_magnitude"]),
        "rotation_mag": float(deviation["rotation_magnitude"]),
        "within_trans_tolerance": bool(deviation["within_translation_tolerance"]),
        "within_rot_tolerance": bool(deviation["within_rotation_tolerance"]),
        "motor_command": result["motor_command"],
    })
    return finish()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_visualization(image, result, K, D, path):
    preview = image.copy()
    if result["diamond"] is not None:
        cv2.polylines(preview, [np.rint(result["diamond"]["corners"]).astype(np.int32)], True, (0, 255, 255), 2)
    if result["accepted"]:
        pose = result["hybrid"]
        cv2.drawFrameAxes(preview, K, D, pose["rvec"], pose["tvec"], 0.04)
    cv2.putText(preview, result["status"], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 200, 0) if result["accepted"] else (0, 0, 255), 2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), preview):
        raise OSError(f"Cannot write {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--params", default=PARAMS_FILE)
    parser.add_argument("--golden", default=GOLDEN_POSE_FILE)
    parser.add_argument("--strategy", choices=("diamond", "hybrid"), default=None)
    parser.add_argument("--backend", choices=("python", "g2o", "auto"), default="python")
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--quality", help="JSON overrides for QualityConfig thresholds")
    parser.add_argument("--json-output", help="Optional machine-readable result file")
    args = parser.parse_args(argv)
    try:
        cal = read_calibration(args.params)
        image = validate_image(cv2.imread(args.image), cal["image_size"])
        golden = load_golden_pose(args.golden, K=cal["K"], D=cal["D"], image_size=cal["image_size"])
        result = run_correction_step(image, cal["K"], cal["D"], golden,
                                     strategy=args.strategy, backend=args.backend, quality=load_quality(args.quality))
        payload = json.dumps(json_safe(result), ensure_ascii=False, allow_nan=False, indent=2)
        print(payload)
        if args.json_output:
            path = Path(args.json_output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload + "\n", encoding="utf-8")
        if args.visual:
            save_visualization(image, result, cal["K"], cal["D"], Path(VIS_DIR) / "correction.png")
        return 0 if result["accepted"] else 2
    except (ValueError, OSError, KeyError, cv2.error) as exc:
        print(f"Correction failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
