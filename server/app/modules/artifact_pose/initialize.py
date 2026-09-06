#!/usr/bin/env python3
"""Initialize a versioned Diamond-only or geometrically verified stereo reference."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from .common import (GOLDEN_POSE_FILE, PARAMS_FILE, VIS_DIR, detect_diamond,
                     extract_orb, read_calibration, resolve_backend, save_golden_pose, validate_image)
from .geometry import (DEFAULT_QUALITY, invert_pose, load_quality, match_descriptors, pose_matrix,
                       refine_pose, transform_points, triangulate, validate_transform)


def run_initialization(left, right, K, D, visual=False, *, output=GOLDEN_POSE_FILE,
                       strategy="diamond", stereo_transform=None, backend="python",
                       quality=DEFAULT_QUALITY):
    left = validate_image(left)
    native = resolve_backend(backend)
    diamond = detect_diamond(left, K, D, quality)
    if diamond is None:
        raise ValueError("Expected Diamond IDs not detected, or pose quality rejected")
    if diamond["ambiguous"]:
        raise ValueError("Ambiguous planar reference pose; use a closer/tilted view or a larger multi-marker target")
    refined = refine_pose(diamond["rvec"], diamond["tvec"], diamond["obj_pts"], diamond["corners"],
                          np.empty((0, 3)), np.empty((0, 2)), K, D, quality, native=native)
    if not refined["accepted"]:
        raise ValueError(f"Reference refinement rejected: {refined['status']}")
    diamond.update(rvec=refined["rvec"], tvec=refined["tvec"])
    xyz, xy, desc = np.empty((0, 3)), np.empty((0, 2)), np.empty((0, 32), np.uint8)
    stereo_meta = None
    if strategy == "hybrid":
        right = validate_image(right, (left.shape[1], left.shape[0]))
        T_left_world = pose_matrix(diamond["rvec"], diamond["tvec"])
        if stereo_transform is None:
            right_diamond = detect_diamond(right, K, D, quality)
            if right_diamond is None or right_diamond["ambiguous"]:
                raise ValueError("Right Diamond missing/ambiguous; supply calibrated --stereo or improve the view")
            right_fit = refine_pose(right_diamond["rvec"], right_diamond["tvec"],
                                    right_diamond["obj_pts"], right_diamond["corners"],
                                    np.empty((0, 3)), np.empty((0, 2)), K, D, quality, native=native)
            if not right_fit["accepted"]:
                raise ValueError("Right Diamond refinement rejected")
            T = pose_matrix(right_fit["rvec"], right_fit["tvec"]) @ invert_pose(T_left_world)
            source = "diamond_estimate_not_independent_metrology"
        else:
            T, source = validate_transform(stereo_transform), "user_calibrated_extrinsic"
        _, dl, kl = extract_orb(left, backend=backend)
        _, dr, kr = extract_orb(right, backend=backend)
        matches = match_descriptors(dl, dr)
        il = np.array([m.queryIdx for m in matches], dtype=int)
        ir = np.array([m.trainIdx for m in matches], dtype=int)
        tri = triangulate(kl[il], kr[ir], K, D, T, quality)
        if tri["num_valid"] < quality.min_orb_inliers:
            raise ValueError(f"Only {tri['num_valid']} geometrically valid stereo landmarks")
        valid = tri["valid_mask"]
        xyz = transform_points(invert_pose(T_left_world), tri["points_3d"][valid])
        xy, desc = kl[il][valid], dl[il][valid]
        stereo_meta = {"source": source, "T_right_left": T.tolist(),
                       "baseline_m": float(np.linalg.norm(T[:3, 3])),
                       "matched": len(matches), "accepted": len(xyz)}
    elif strategy != "diamond":
        raise ValueError("strategy must be diamond or hybrid")
    save_golden_pose(output, diamond, xyz, xy, desc, (left.shape[1], left.shape[0]),
                     K=K, D=D, strategy=strategy, stereo=stereo_meta,
                     min_orb_inliers=quality.min_orb_inliers)
    if visual:
        preview = left.copy()
        cv2.drawFrameAxes(preview, K, D, diamond["rvec"], diamond["tvec"], 0.04)
        for p in xy:
            cv2.circle(preview, tuple(np.rint(p).astype(int)), 3, (0, 255, 0), -1)
        path = Path(VIS_DIR) / "golden_initialization.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), preview):
            raise OSError(f"Cannot write visualization {path}")
    return {"diamond": diamond, "points_3d": xyz, "points_2d": xy,
            "descriptors": desc, "stereo": stereo_meta, "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", help="Required for --strategy hybrid")
    parser.add_argument("--strategy", choices=("diamond", "hybrid"), default="diamond")
    parser.add_argument("--stereo", help="JSON with T_right_left (4x4, metres), same camera intrinsics")
    parser.add_argument("--params", default=PARAMS_FILE)
    parser.add_argument("--output", default=GOLDEN_POSE_FILE)
    parser.add_argument("--backend", choices=("python", "g2o", "auto"), default="python")
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--quality", help="JSON overrides for QualityConfig thresholds")
    args = parser.parse_args(argv)
    try:
        if args.strategy == "hybrid" and not args.right:
            raise ValueError("--strategy hybrid requires --right")
        if args.strategy == "diamond" and (args.right or args.stereo):
            raise ValueError("--right/--stereo require --strategy hybrid")
        calibration = read_calibration(args.params)
        left = validate_image(cv2.imread(args.left), calibration["image_size"])
        right = validate_image(cv2.imread(args.right), calibration["image_size"]) if args.right else None
        stereo = None
        if args.stereo:
            with open(args.stereo, encoding="utf-8") as f:
                stereo = validate_transform(json.load(f)["T_right_left"])
        result = run_initialization(left, right, calibration["K"], calibration["D"], args.visual,
                                    output=args.output, strategy=args.strategy,
                                    stereo_transform=stereo, backend=args.backend, quality=load_quality(args.quality))
        print(f"Saved {result['output']}; strategy={args.strategy}; landmarks={len(result['points_3d'])}")
        return 0
    except (ValueError, OSError, KeyError, cv2.error) as exc:
        print(f"Initialization failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
