#!/usr/bin/env python3
"""Choose the fixed SSIM threshold tau (P2/P3) on VALIDATION pairs only.

    python tools/experiments/calibrate_thresholds.py --pairs data/E3/val_* --split val \
        --target-far 0.05 --out configs/thresholds.yaml [--min-area-grid 2,4,8]

tau = the smallest value whose false-alarm rate (fraction of UNCHANGED pairs
with >= 1 detected region) is <= ``--target-far``. With ``--min-area-grid``
the (tau, min_area_mm2) pair with the best region recall on CHANGED
validation pairs is kept (ties -> larger area). Pairs whose split is not
``val`` are refused: test data is never used for tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_common_args, make_logger, setup  # noqa: E402
from _pairs import PairSet  # noqa: E402

from app.experiments.profiles import load_profile  # noqa: E402
from app.experiments.trial_log import git_info, json_safe, sha256_file  # noqa: E402
from app.modules.change_detection.pipelines import compare  # noqa: E402
from app.modules.change_detection.regions import extract_regions, min_area_px, threshold_map  # noqa: E402
from app.modules.change_detection.roi import px_per_mm as roi_ppm  # noqa: E402


def region_hits(regions, gt_changes) -> int:
    hits = 0
    for g in gt_changes:
        gx1, gy1, gx2, gy2 = g["bbox_ref_xyxy"]
        for r in regions:
            x1, y1, x2, y2 = r["bbox_ref_xyxy"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if gx1 <= cx <= gx2 and gy1 <= cy <= gy2:
                hits += 1
                break
    return hits


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", nargs="+", required=True)
    p.add_argument("--split", required=True, choices=("val",), help="must be 'val'")
    p.add_argument("--pipeline", default="P2", choices=("P2", "P3"))
    p.add_argument("--target-far", type=float, default=0.05)
    p.add_argument("--tau-grid", default="0.05:0.95:0.005", help="start:stop:step")
    p.add_argument("--min-area-grid", default=None, help="comma list of min_area_mm2 to search")
    p.add_argument("--out", default="configs/thresholds.yaml")
    p.add_argument("--force", action="store_true")
    add_common_args(p, profile_default="P2")
    args = p.parse_args(argv)
    base, seed, _ = setup(args)
    sets = [PairSet(Path(x)) for x in args.pairs]
    for s in sets:
        bad = [q["pair_id"] for q in s.pairs() if q["split"] != "val"] + ([s.session_id] if s.split != "val" else [])
        if bad:
            raise SystemExit(f"Refusing non-validation data ({bad[:5]}): thresholds are tuned on split=val only")
    prof = load_profile(args.pipeline)
    cfg = prof.inspection
    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[2] / out
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force to replace it")
    log = make_logger("E3", args, base)
    maps = []
    for s in sets:
        roi = s.roi()
        ppm = s.manifest.get("px_per_mm") or roi_ppm(roi)
        for q in s.pairs():
            cv2.setRNGSeed(seed)
            ref, cur = cv2.imread(str(q["golden_path"])), cv2.imread(str(q["current_path"]))
            comp = compare(args.pipeline, ref, cur, cfg, roi["polygon_ref_px"] if roi else
                           np.array([[0, 0], [ref.shape[1] - 1, 0], [ref.shape[1] - 1, ref.shape[0] - 1], [0, ref.shape[0] - 1]], float))
            maps.append({"pair_id": q["pair_id"], "changes": (q["gt_data"] or {}).get("changes", []), "d": comp["d"],
                         "diff": comp["diff_uint8"], "valid": comp["valid"], "ppm": ppm, "shape": ref.shape,
                         "sha256": [sha256_file(q["golden_path"]), sha256_file(q["current_path"])]})
    unchanged = [m for m in maps if not m["changes"]]
    changed = [m for m in maps if m["changes"]]
    if not unchanged:
        raise SystemExit("Need unchanged validation pairs to measure false alarms")
    start, stop, step = (float(v) for v in args.tau_grid.split(":"))
    taus = np.arange(start, stop + 1e-12, step)
    areas = [float(v) for v in args.min_area_grid.split(",")] if args.min_area_grid else [cfg.min_area_mm2]

    def count_regions(m, tau, area):
        mask, _ = threshold_map(m["d"], m["diff"], "fixed", float(tau))
        mask = cv2.bitwise_and(mask, m["valid"])
        return extract_regions(mask, m["d"], min_area=min_area_px(m["shape"], min_area_mm2=area, px_per_mm=m["ppm"]),
                               px_per_mm=m["ppm"])

    candidates = []
    for area in areas:
        chosen = None
        for tau in taus:
            far = np.mean([len(count_regions(m, tau, area)) > 0 for m in unchanged])
            if far <= args.target_far:
                chosen = (float(tau), float(far))
                break
        if chosen is None:
            continue
        hits = sum(region_hits(count_regions(m, chosen[0], area), m["changes"]) for m in changed)
        total = sum(len(m["changes"]) for m in changed)
        candidates.append({"tau": chosen[0], "far": chosen[1], "min_area_mm2": area,
                           "val_region_recall": hits / total if total else None})
    if not candidates:
        raise SystemExit("No tau reaches the target false-alarm rate on validation")
    best = max(candidates, key=lambda c: ((c["val_region_recall"] or 0.0), c["min_area_mm2"] or 0.0, -c["tau"]))
    warning = None
    if len(unchanged) < int(np.ceil(1.0 / max(args.target_far, 1e-9))):
        warning = (f"only {len(unchanged)} unchanged validation pairs: a false-alarm target of "
                   f"{args.target_far:.0%} cannot be resolved (needs >= {int(np.ceil(1 / args.target_far))}); "
                   "tau is optimistic, collect more unchanged validation pairs")
        print(f"WARNING: {warning}", file=sys.stderr)
    data = {"schema": "cita.thresholds.v1", "pipeline": args.pipeline, "tau": best["tau"],
            "min_area_mm2": best["min_area_mm2"], "target_far": args.target_far, "achieved_far_val": best["far"],
            "val_region_recall": best["val_region_recall"], "n_unchanged_val": len(unchanged),
            "n_changed_val": len(changed), "candidates": candidates, "warning": warning,
            "sources": [{"pairs": s.session_id, "split": s.split} for s in sets],
            "data_sha256": hashlib.sha256("".join(sorted(h for m in maps for h in m["sha256"])).encode()).hexdigest(),
            "profile_hash": prof.profile_hash, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tool_commit": git_info()["commit"]}
    import yaml
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(json_safe(data), sort_keys=False), encoding="utf-8")
    log.log(arm=f"{args.pipeline}-threshold", input={"pairs": [s.session_id for s in sets], "split": "val"},
            output={k: v for k, v in data.items() if k != "candidates"})
    print(f"tau = {best['tau']:.3f}, min_area_mm2 = {best['min_area_mm2']}, val FAR = {best['far']:.3f} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
