#!/usr/bin/env python3
"""E1 evaluation: replay_pose JSONL + board GT -> errors, statistics, CSV/JSON.

    python tools/experiments/eval_pose.py --logs logs/experiments/E1/*.jsonl --out results/E1/eval

Per method: failure rate, e_t / e_R (median, RMSE, p95 + bootstrap 95 % CI),
per-axis bias of c (golden camera frame), 2-sigma coverage of the reported
uncertainty (M4/M5), timings; Friedman + Wilcoxon-Holm paired by unit
(frame, or frame group for M5); odometry vs board cross-check.
Accuracy is ALWAYS measured against the independent board GT, never against
reprojection error.
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_log_args, expand, log_tool_run, write_json  # noqa: E402

from app.experiments.stats import bootstrap_ci, coverage, describe, friedman_wilcoxon, rmse  # noqa: E402
from app.experiments.trial_log import read_jsonl  # noqa: E402
from app.modules.artifact_pose.motion import average_poses, pose_errors  # noqa: E402

SUCCESS = {"ok"}


def _T(summary: dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(summary["delta_R"])
    T[:3, 3] = np.asarray(summary["delta_t"] if "delta_t" in summary else summary["delta_t_m"])
    return T


def load_records(paths: list[str]) -> list[dict]:
    rows = []
    for pattern in paths:
        for p in sorted(glob.glob(pattern)) or [pattern]:
            rows.extend(read_jsonl(p))
    return rows


def evaluate(rows: list[dict], *, seed: int = 2027, n_boot: int = 2000) -> dict:
    gt = {}
    for r in rows:
        if r.get("arm") == "GT" and r.get("status") == "ok":
            fid = r["input"]["frame_ids"][0]
            gt[(r["input"]["session_id"], fid)] = r
    per_unit = []
    for r in rows:
        arm = r.get("arm", "")
        if arm == "GT" or ":" in arm or not r.get("input", {}).get("frame_ids"):
            continue
        sid, fids = r["input"]["session_id"], r["input"]["frame_ids"]
        gts = [gt.get((sid, f)) for f in fids]
        unit = {"arm": arm, "session_id": sid, "unit": "+".join(fids), "pose_index": r["input"].get("pose_index"),
                "status": r.get("status"), "has_gt": all(g is not None for g in gts),
                "time_ms": (r.get("timing_ms") or {}).get("total")}
        out = r.get("output") or {}
        if unit["has_gt"]:
            dT_gt = average_poses([_T(g["output"]) for g in gts])
            unit["c_gt_mm"] = (-(dT_gt[:3, :3].T @ dT_gt[:3, 3]) * 1000).tolist()
            target = r["input"].get("target") or {}
            unit["target"] = target
        if r.get("status") in SUCCESS and out.get("delta_R") is not None and unit["has_gt"]:
            e = pose_errors(_T(out), dT_gt)
            unit.update(e)
            if out.get("sigma_center_mm"):
                unit["sigma_center_mm"] = out["sigma_center_mm"]
        per_unit.append(unit)

    methods = sorted({u["arm"] for u in per_unit})
    summary = {}
    for m in methods:
        units = [u for u in per_unit if u["arm"] == m]
        with_gt = [u for u in units if u["has_gt"]]
        good = [u for u in with_gt if "e_t_mm" in u]
        et = [u["e_t_mm"] for u in good]
        er = [u["e_R_deg"] for u in good]
        cerr = np.asarray([u["c_err_mm"] for u in good]) if good else np.empty((0, 3))
        sig = np.asarray([u["sigma_center_mm"] for u in good if "sigma_center_mm" in u])
        cov2 = None
        if len(sig) == len(good) and len(sig):
            cov2 = coverage(cerr.ravel(), sig.ravel(), 2.0)
        summary[m] = {
            "n_units": len(units), "n_with_gt": len(with_gt), "n_success": len(good),
            "failure_rate": (1 - len(good) / len(with_gt)) if with_gt else None,
            "e_t_mm": {**describe(et), "median_ci95": bootstrap_ci(et, np.median, seed=seed, n_boot=n_boot),
                       "rmse_ci95": bootstrap_ci(et, rmse, seed=seed, n_boot=n_boot)},
            "e_R_deg": {**describe(er), "median_ci95": bootstrap_ci(er, np.median, seed=seed, n_boot=n_boot)},
            "bias_mm": cerr.mean(axis=0).tolist() if len(cerr) else None,
            "coverage_2sigma": cov2,
            "time_ms": describe([u["time_ms"] for u in units if u.get("time_ms") is not None]),
        }
    # Paired tests on units every method solved (frames; M5 units differ -> its own family).
    paired = {}
    singles = [m for m in methods if m != "M5"]
    keyed = {m: {(u["session_id"], u["unit"]): u.get("e_t_mm") for u in per_unit if u["arm"] == m} for m in singles}
    common = set.intersection(*(set(k for k, v in d.items() if v is not None) for d in keyed.values())) if keyed else set()
    common = sorted(common)
    if len(singles) >= 2 and common:
        paired["e_t_mm"] = friedman_wilcoxon({m: [keyed[m][k] for k in common] for m in singles})
    # Odometry cross-check on pure slider moves.
    odo = []
    for u in per_unit:
        t = u.get("target") or {}
        if u["arm"] != methods[0] or not u.get("c_gt_mm") or not t:
            continue
        if abs(t.get("pan_deg", 0) or 0) < 1e-9 and abs(t.get("tilt_deg", 0) or 0) < 1e-9:
            cmd = float(np.hypot(t.get("x_mm", 0) or 0, t.get("z_mm", 0) or 0))
            if cmd > 0:
                odo.append(float(np.linalg.norm(u["c_gt_mm"])) / cmd)
    return {"methods": methods, "summary": summary, "paired_tests": paired,
            "odometry_scale_board_over_command": describe(odo), "per_unit": per_unit}


def write_csv(path: Path, units: list[dict]) -> None:
    keys = ["arm", "session_id", "unit", "pose_index", "status", "has_gt", "e_t_mm", "e_R_deg", "c_err_mm",
            "sigma_center_mm", "c_gt_mm", "time_ms"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for u in units:
            w.writerow({k: u.get(k) for k in keys})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", nargs="+", required=True, help="replay_pose JSONL files / globs")
    parser.add_argument("--out", required=True, help="output prefix (writes <out>.json, <out>_units.csv)")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--n-boot", type=int, default=2000)
    add_log_args(parser)
    args = parser.parse_args(argv)
    result = evaluate(load_records(args.logs), seed=args.seed, n_boot=args.n_boot)
    out = Path(args.out)
    write_csv(out.with_name(out.name + "_units.csv"), result["per_unit"])
    write_json(out.with_suffix(".json"), {"experiment": "E1", "seed": args.seed, "logs": args.logs,
                                          **{k: v for k, v in result.items() if k != "per_unit"}})
    log_tool_run("E1", "eval_pose", inputs=expand(args.logs), log_root=args.log_root, run_id=args.run_id,
                 seed=args.seed, outputs={"result": str(out.with_suffix(".json")),
                                          "units_csv": str(out.with_name(out.name + "_units.csv")),
                                          "e_t_median_mm": {m: result["summary"][m]["e_t_mm"]["median"]
                                                            for m in result["methods"]}})
    for m in result["methods"]:
        s = result["summary"][m]
        print(f"{m:8s} n={s['n_with_gt']:4d} fail={s['failure_rate']} e_t med={s['e_t_mm']['median']} "
              f"rmse={s['e_t_mm']['rmse']} e_R med={s['e_R_deg']['median']} cov2={s['coverage_2sigma']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
