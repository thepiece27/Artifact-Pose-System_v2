#!/usr/bin/env python3
"""E1 evaluation (v1.2, R5): replay_pose JSONL + board GT -> per-POSITION errors and statistics.

    python tools/experiments/eval_pose.py --logs logs/experiments/E1/*.jsonl --out results/E1/eval \
        [--contrasts M3-M1,M4-M3,M5-M4,M4-M2,M4-M4-jitter] [--cluster session|none]

Unit = rig position ``(session_id, pose_index)``:
* single-frame methods (M1-M4, M2-full, M4-jitter): ``e_t_first`` = error of the
  frame with the smallest rep (rep 0, "one shot", PRIMARY), ``e_t_med`` = median
  over the frames of the position (secondary);
* M5: one value per position (its own frame group);
* GT of the unit = SE(3) mean of the board GT of all frames of the position.

Paired tests on ``e_t_first`` over the units EVERY method solved (M5 included):
Friedman + Wilcoxon on the pre-registered ``--contrasts`` (Holm over that set);
all pairs are reported as an appendix. CIs: two-stage cluster bootstrap
(sessions, then units) and the iid bootstrap (both reported; a warning is
printed with < 5 sessions). Per method: failure rate, Diamond-fallback rate,
ORB inliers (median, p10), e_t split hybrid / fallback, 2-sigma coverage of
the a-priori and a-posteriori covariance. Accuracy is ALWAYS measured against
the independent board GT, never against reprojection error.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_log_args, expand, log_tool_run, output_file, write_json  # noqa: E402

from app.experiments.stats import (bootstrap_ci, cluster_bootstrap_ci, coverage, describe, friedman_wilcoxon,  # noqa: E402
                                   rmse, wilcoxon_contrasts)
from app.experiments.trial_log import read_jsonl  # noqa: E402
from app.modules.artifact_pose.motion import average_poses, pose_errors  # noqa: E402

DEFAULT_CONTRASTS = "M3-M1,M4-M3,M5-M4,M4-M2,M4-M4-jitter"


def _T(summary: dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(summary["delta_R"])
    T[:3, 3] = np.asarray(summary["delta_t"] if "delta_t" in summary else summary["delta_t_m"])
    return T


def load_records(paths: list[str]) -> list[dict]:
    return [r for f in expand(paths) for r in read_jsonl(f)]


def parse_contrasts(text: str, methods: set[str]) -> list[tuple[str, str]]:
    """``A-B`` where A and B may contain '-' (M2-full, M4-jitter): split at the
    dash that leaves two known names. Unknown/missing pairs are skipped."""
    out = []
    for item in [c.strip() for c in text.split(",") if c.strip()]:
        for i, ch in enumerate(item):
            if ch == "-" and item[:i] in methods and item[i + 1:] in methods:
                out.append((item[:i], item[i + 1:]))
                break
    return out


def is_fallback(out: dict) -> bool:
    return out.get("strategy_used") == "diamond" or out.get("status") == "accepted_diamond_fallback"


def build_units(rows: list[dict]) -> tuple[dict, list[dict]]:
    gt_frames: dict = {}
    for r in rows:
        if r.get("arm") == "GT" and r.get("status") == "ok":
            inp = r["input"]
            gt_frames.setdefault((inp["session_id"], inp.get("pose_index")), []).append(_T(r["output"]))
    gt = {k: average_poses(v) for k, v in gt_frames.items()}
    grouped: dict = {}
    for r in rows:
        arm = r.get("arm", "")
        inp = r.get("input") or {}
        if arm == "GT" or ":" in arm or r.get("record_type") == "tool" or not inp.get("frame_ids"):
            continue
        key = (inp["session_id"], inp.get("pose_index"))
        grouped.setdefault(arm, {}).setdefault(key, []).append(r)
    units = []
    for arm, by_key in grouped.items():
        for key, recs in by_key.items():
            recs = sorted(recs, key=lambda r: min(x for x in (r["input"].get("reps") or [0]) if x is not None)
                          if r["input"].get("reps") else 0)
            first = recs[0]
            u = {"arm": arm, "session_id": key[0], "pose_index": key[1], "n_frames": sum(len(r["input"]["frame_ids"]) for r in recs),
                 "has_gt": key in gt, "status_first": first.get("status"),
                 "time_ms": (first.get("timing_ms") or {}).get("total"),
                 "target": first["input"].get("target") or {}}
            out = first.get("output") or {}
            u["fallback"] = is_fallback(out) if first.get("status") == "ok" else None
            u["n_orb_inliers"] = out.get("n_orb_inliers")
            if key in gt:
                dT_gt = gt[key]
                u["c_gt_mm"] = (-(dT_gt[:3, :3].T @ dT_gt[:3, 3]) * 1000).tolist()
                if first.get("status") == "ok" and out.get("delta_R") is not None:
                    e = pose_errors(_T(out), dT_gt)
                    u.update(e_t_first=e["e_t_mm"], e_R_first=e["e_R_deg"], c_err_mm=e["c_err_mm"])
                    if out.get("sigma_center_mm"):
                        u["sigma_center_mm"] = out["sigma_center_mm"]
                        u["sigma_rot_deg"] = out.get("sigma_rot_deg")
                    post = (out.get("covariance") or {}).get("sigma_center_mm_post")
                    if post:
                        u["sigma_center_mm_post"] = post
                errs = [pose_errors(_T(r["output"]), dT_gt)["e_t_mm"] for r in recs
                        if r.get("status") == "ok" and (r.get("output") or {}).get("delta_R") is not None]
                u["e_t_med"] = float(np.median(errs)) if errs else None
            units.append(u)
    return gt, units


def summarise(units: list[dict], *, seed: int, n_boot: int, cluster: str) -> dict:
    out = {}
    for m in sorted({u["arm"] for u in units}):
        us = [u for u in units if u["arm"] == m]
        with_gt = [u for u in us if u["has_gt"]]
        good = [u for u in with_gt if "e_t_first" in u]
        et = [u["e_t_first"] for u in good]
        by_session: dict = {}
        for u in good:
            by_session.setdefault(u["session_id"], []).append(u["e_t_first"])
        cerr = np.asarray([u["c_err_mm"] for u in good]) if good else np.empty((0, 3))
        cov_prior = cov_post = None
        sig = [u for u in good if u.get("sigma_center_mm")]
        if sig:
            cov_prior = coverage(np.ravel([u["c_err_mm"] for u in sig]), np.ravel([u["sigma_center_mm"] for u in sig]), 2.0)
        sigp = [u for u in good if u.get("sigma_center_mm_post")]
        if sigp:
            cov_post = coverage(np.ravel([u["c_err_mm"] for u in sigp]),
                                np.ravel([u["sigma_center_mm_post"] for u in sigp]), 2.0)
        hyb = [u["e_t_first"] for u in good if u.get("fallback") is False]
        fb = [u["e_t_first"] for u in good if u.get("fallback")]
        inl = [u["n_orb_inliers"] for u in good if u.get("n_orb_inliers") is not None]
        e_t = {**describe(et),
               "median_ci95_cluster": cluster_bootstrap_ci(by_session, np.median, seed=seed, n_boot=n_boot),
               "median_ci95_iid": bootstrap_ci(et, np.median, seed=seed, n_boot=n_boot),
               "rmse_ci95_cluster": cluster_bootstrap_ci(by_session, rmse, seed=seed, n_boot=n_boot),
               "n_sessions": len(by_session)}
        e_t["median_ci95"] = e_t["median_ci95_cluster"] if cluster == "session" else e_t["median_ci95_iid"]
        out[m] = {
            "n_units": len(us), "n_with_gt": len(with_gt), "n_success": len(good),
            "failure_rate": (1 - len(good) / len(with_gt)) if with_gt else None,
            "fallback_rate": (len(fb) / len(good)) if good else None,
            "e_t_first": e_t,
            "e_t_med": describe([u.get("e_t_med") for u in with_gt]),
            "e_R_first": {**describe([u["e_R_first"] for u in good]),
                          "median_ci95": bootstrap_ci([u["e_R_first"] for u in good], np.median, seed=seed, n_boot=n_boot)},
            "e_t_hybrid": describe(hyb), "e_t_fallback": describe(fb),
            "n_orb_inliers": {"median": float(np.median(inl)) if inl else None,
                              "p10": float(np.percentile(inl, 10)) if inl else None},
            "bias_mm": cerr.mean(axis=0).tolist() if len(cerr) else None,
            "coverage_2sigma": cov_prior, "coverage_2sigma_post": cov_post,
            "sigma_center_mm_median": np.median([u["sigma_center_mm"] for u in sig], axis=0).tolist() if sig else None,
            "sigma_rot_deg_median": np.median([u["sigma_rot_deg"] for u in sig if u.get("sigma_rot_deg")], axis=0).tolist()
            if any(u.get("sigma_rot_deg") for u in sig) else None,
            "time_ms": describe([u["time_ms"] for u in us if u.get("time_ms") is not None]),
        }
    return out


def evaluate(rows: list[dict], *, seed: int = 2027, n_boot: int = 2000, contrasts: str = DEFAULT_CONTRASTS,
             cluster: str = "session") -> dict:
    gt, units = build_units(rows)
    methods = sorted({u["arm"] for u in units})
    summary = summarise(units, seed=seed, n_boot=n_boot, cluster=cluster)
    warnings = []
    n_sessions = len({u["session_id"] for u in units})
    if n_sessions < 5:
        warnings.append(f"only {n_sessions} session(s): cluster-bootstrap CIs are wide (this is correct)")
    keyed = {m: {(u["session_id"], u["pose_index"]): u.get("e_t_first") for u in units if u["arm"] == m}
             for m in methods}
    common = sorted(set.intersection(*(set(k for k, v in d.items() if v is not None) for d in keyed.values()))
                    if keyed else set(), key=str)
    paired: dict = {"n_common_units": len(common), "methods": methods}
    if len(methods) >= 2 and len(common) >= 3:
        table = {m: [keyed[m][k] for k in common] for m in methods}
        paired["all_pairs"] = friedman_wilcoxon(table)
        paired["contrasts"] = wilcoxon_contrasts(table, parse_contrasts(contrasts, set(methods)))
    odo = []
    for u in units:
        t = u.get("target") or {}
        if u["arm"] != methods[0] or not u.get("c_gt_mm") or not t:
            continue
        if abs(t.get("pan_deg", 0) or 0) < 1e-9 and abs(t.get("tilt_deg", 0) or 0) < 1e-9:
            cmd = float(np.hypot(t.get("x_mm", 0) or 0, t.get("z_mm", 0) or 0))
            if cmd > 0:
                odo.append(float(np.linalg.norm(u["c_gt_mm"])) / cmd)
    return {"methods": methods, "unit": "position (session_id, pose_index)", "primary_metric": "e_t_first",
            "cluster_bootstrap": cluster, "n_sessions": n_sessions, "warnings": warnings, "summary": summary,
            "paired_tests": paired, "odometry_scale_board_over_command": describe(odo), "units": units}


def write_csv(path: Path, units: list[dict]) -> None:
    keys = ["arm", "session_id", "pose_index", "n_frames", "status_first", "has_gt", "e_t_first", "e_t_med",
            "e_R_first", "fallback", "n_orb_inliers", "c_err_mm", "sigma_center_mm", "sigma_center_mm_post",
            "c_gt_mm", "time_ms"]
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
    parser.add_argument("--contrasts", default=DEFAULT_CONTRASTS, help="pre-registered contrasts A-B (Holm)")
    parser.add_argument("--cluster", choices=("session", "none"), default="session",
                        help="CI reported as median_ci95 (both are always written)")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--n-boot", type=int, default=2000)
    add_log_args(parser)
    args = parser.parse_args(argv)
    result = evaluate(load_records(args.logs), seed=args.seed, n_boot=args.n_boot, contrasts=args.contrasts,
                      cluster=args.cluster)
    out = Path(args.out)
    write_csv(out.with_name(out.name + "_units.csv"), result["units"])
    write_json(output_file(out, ".json"), {"experiment": "E1", "seed": args.seed, "logs": args.logs,
                                          "contrasts_requested": args.contrasts,
                                          **{k: v for k, v in result.items() if k != "units"}})
    log_tool_run("E1", "eval_pose", inputs=expand(args.logs), log_root=args.log_root, run_id=args.run_id,
                 seed=args.seed, outputs={"result": str(output_file(out, ".json")),
                                          "units_csv": str(out.with_name(out.name + "_units.csv")),
                                          "e_t_first_median_mm": {m: result["summary"][m]["e_t_first"]["median"]
                                                                  for m in result["methods"]},
                                          "warnings": result["warnings"]})
    for w in result["warnings"]:
        print(f"WARNING: {w}", file=sys.stderr)
    for m in result["methods"]:
        s = result["summary"][m]
        print(f"{m:10s} units={s['n_with_gt']:4d} fail={s['failure_rate']} fallback={s['fallback_rate']} "
              f"e_t_first med={s['e_t_first']['median']} CI(cluster)={s['e_t_first']['median_ci95_cluster']} "
              f"cov2={s['coverage_2sigma']}/{s['coverage_2sigma_post']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
