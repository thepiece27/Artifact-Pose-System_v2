#!/usr/bin/env python3
"""E2 evaluation of run_reloc_trials logs, grouped by (mount condition, arm).

    python tools/experiments/eval_reloc.py --logs logs/experiments/E2/<run>.jsonl --out results/E2/eval \
        --tol-t-mm 2 --tol-R-deg 1 [--golden-image golden_left.png --calib cal.yaml --roi roi.json]

Primary metrics, in this order: ``total_moves`` (control iterations + actual
probe moves of the trial; median + bootstrap CI), time (s), success rate by
GT, final e_t / e_R / ROI-AFD. Probe cost is reported as probe moves per trial
(total / n_trials) and number of trials that probed (R3).

Success by GT: virtual trials use the true final pose of the virtual rig;
real trials use the reference ChArUco board between the golden image and the
final aligned image (``final_aligned_path``), plus ROI-AFD.

Tests, inside each mount condition, on blocks paired by (painting,
perturbation): McNemar on GT success (all arm pairs) and Wilcoxon on
total_moves for C-A vs C-D and C-B vs C-D (Holm). The tolerances are
mandatory unless every log record carries the same tau.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_log_args, expand, load_calibration, log_tool_run, write_json  # noqa: E402

from app.experiments.stats import bootstrap_ci, describe, holm, mcnemar  # noqa: E402
from app.experiments.trial_log import read_jsonl  # noqa: E402

WILCOXON_CONTRASTS = (("C-A", "C-D"), ("C-B", "C-D"))


def real_gt(summary: dict, golden_img, cal, roi_polygon, spec) -> dict:
    from app.modules.artifact_pose.motion import camera_center_in_golden, rotvec
    from app.modules.change_detection.registration import roi_homography
    from app.modules.ground_truth.reference_board import detect_reference_board, relative_motion_gt
    path = summary.get("final_aligned_path")
    img = cv2.imread(str(path)) if path else None
    if img is None:
        return {}
    out = {}
    g = detect_reference_board(golden_img, cal["K"], cal["D"], spec)
    c = detect_reference_board(img, cal["K"], cal["D"], spec)
    if g["ok"] and c["ok"]:
        dT = relative_motion_gt(g, c)
        out["final_e_t_mm"] = float(np.linalg.norm(camera_center_in_golden(dT)) * 1000)
        out["final_e_R_deg"] = float(np.rad2deg(np.linalg.norm(rotvec(dT[:3, :3]))))
    if roi_polygon is not None:
        out["final_roi_afd_px"] = roi_homography(img, golden_img, roi_polygon, ecc=False).roi_afd_px
    return out


def resolve_tolerances(summaries: list[dict], tol_t_mm: float | None, tol_R_deg: float | None) -> tuple[float, float]:
    if tol_t_mm is not None and tol_R_deg is not None:
        return float(tol_t_mm), float(tol_R_deg)
    taus = {(s["output"].get("tol_t_mm"), s["output"].get("tol_R_deg")) for s in summaries}
    if len(taus) != 1 or None in next(iter(taus)):
        raise SystemExit(f"Pass --tol-t-mm and --tol-R-deg: the logs carry no single common tau ({sorted(map(str, taus))})")
    t, r = next(iter(taus))
    return float(tol_t_mm if tol_t_mm is not None else t), float(tol_R_deg if tol_R_deg is not None else r)


def evaluate(rows: list[dict], *, tol_t_mm: float | None, tol_R_deg: float | None, golden=None, cal=None,
             roi_polygon=None, spec=None, seed: int = 2027, n_boot: int = 2000) -> dict:
    summaries = [r for r in rows if r.get("record_type") == "summary"]
    tt, tr = resolve_tolerances(summaries, tol_t_mm, tol_R_deg)
    trials = []
    for r in summaries:
        out, inp = r.get("output") or {}, r.get("input") or {}
        t = {"arm": r["arm"], "mount_condition": inp.get("mount_condition") or "unspecified",
             "painting_yaw_deg": inp.get("painting_yaw_deg"), "probe_reuse": inp.get("probe_reuse"),
             "key": (inp.get("painting_id"), inp.get("perturbation_index")),
             "success_est": bool(out.get("success_est")), "iterations": out.get("control_iterations"),
             "probe_moves": int(out.get("n_probe_moves") or 0),
             "total_moves": out.get("total_moves", (out.get("control_iterations") or 0) + (out.get("n_probe_moves") or 0)),
             "time_s": out.get("time_s", out.get("wall_time_s")), "stop_reason": out.get("stop_reason")}
        if out.get("true_final"):
            t["final_e_t_mm"] = out["true_final"]["c_norm_mm"]
            t["final_e_R_deg"] = out["true_final"]["rot_deg"]
        elif golden is not None:
            t.update(real_gt(out, golden, cal, roi_polygon, spec))
        t["success_gt"] = (t.get("final_e_t_mm") is not None and t["final_e_t_mm"] <= tt
                           and t.get("final_e_R_deg") is not None and t["final_e_R_deg"] <= tr)
        trials.append(t)
    conditions = sorted({t["mount_condition"] for t in trials})
    summary, paired = {}, {}
    for cond in conditions:
        tc = [t for t in trials if t["mount_condition"] == cond]
        arms = sorted({t["arm"] for t in tc})
        summary[cond] = {}
        for a in arms:
            ts = [t for t in tc if t["arm"] == a]
            succ = [float(t["success_gt"]) for t in ts]
            moves = [t["total_moves"] for t in ts]
            summary[cond][a] = {
                "n_trials": len(ts),
                "total_moves": {**describe(moves), "median_ci95": bootstrap_ci(moves, np.median, seed=seed, n_boot=n_boot)},
                "time_s": describe([t["time_s"] for t in ts]),
                "success_gt_rate": float(np.mean(succ)) if ts else None,
                "success_gt_ci95": bootstrap_ci(succ, np.mean, seed=seed, n_boot=n_boot),
                "success_est_rate": float(np.mean([t["success_est"] for t in ts])) if ts else None,
                "probe_moves_total": int(sum(t["probe_moves"] for t in ts)),
                "probe_moves_per_trial": float(sum(t["probe_moves"] for t in ts) / len(ts)) if ts else None,
                "n_trials_with_probe": int(sum(1 for t in ts if t["probe_moves"] > 0)),
                "iterations": describe([t["iterations"] for t in ts]),
                "final_e_t_mm": describe([t.get("final_e_t_mm") for t in ts]),
                "final_e_R_deg": describe([t.get("final_e_R_deg") for t in ts]),
                "final_roi_afd_px": describe([t.get("final_roi_afd_px") for t in ts]),
                "painting_yaw_deg": sorted({t["painting_yaw_deg"] for t in ts if t["painting_yaw_deg"] is not None}),
                "probe_reuse": sorted({t["probe_reuse"] for t in ts if t["probe_reuse"]}),
                "stop_reasons": {k: sum(t["stop_reason"] == k for t in ts) for k in {t["stop_reason"] for t in ts}},
            }
        by_key: dict = {}
        for t in tc:
            by_key.setdefault(t["key"], {})[t["arm"]] = t
        complete = [v for v in by_key.values() if all(a in v for a in arms)]
        tests = {"n_paired_blocks": len(complete), "mcnemar_success_gt": [], "wilcoxon_total_moves": []}
        for a, b in combinations(arms, 2):
            tests["mcnemar_success_gt"].append({"a": a, "b": b, **mcnemar([v[a]["success_gt"] for v in complete],
                                                                          [v[b]["success_gt"] for v in complete])})
        raw = {}
        for a, b in WILCOXON_CONTRASTS:
            if a in arms and b in arms and len(complete) >= 3:
                x = np.array([v[a]["total_moves"] for v in complete], float)
                y = np.array([v[b]["total_moves"] for v in complete], float)
                raw[(a, b)] = (1.0 if np.allclose(x, y) else float(sstats.wilcoxon(x, y).pvalue),
                               float(np.median(x - y)))
        adj = holm({f"{a}|{b}": p for (a, b), (p, _) in raw.items()})
        tests["wilcoxon_total_moves"] = [{"a": a, "b": b, "p": p, "p_holm": adj[f"{a}|{b}"], "median_diff": d}
                                         for (a, b), (p, d) in raw.items()]
        paired[cond] = tests
    return {"conditions": conditions, "tol_t_mm": tt, "tol_R_deg": tr, "summary": summary,
            "paired_tests": paired, "trials": trials}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tol-t-mm", type=float, default=None, help="GT success threshold (mandatory unless logs agree)")
    p.add_argument("--tol-R-deg", type=float, default=None)
    p.add_argument("--golden-image", default=None, help="real trials: golden image for board GT / ROI-AFD")
    p.add_argument("--calib", default=None)
    p.add_argument("--roi", default=None)
    p.add_argument("--profile", default="pbl_legacy", help="ground_truth.board settings")
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--n-boot", type=int, default=2000)
    add_log_args(p)
    args = p.parse_args(argv)
    files = expand(args.logs)
    rows = [r for f in files for r in read_jsonl(f)]
    golden = cal = roi_poly = spec = None
    if args.golden_image:
        from _common import DEFAULT_CALIBRATION
        from app.experiments.profiles import load_profile
        from app.modules.ground_truth.reference_board import BoardSpec
        golden = cv2.imread(args.golden_image)
        cal = load_calibration(Path(args.calib) if args.calib else DEFAULT_CALIBRATION)
        spec = BoardSpec.from_profile(load_profile(args.profile).ground_truth.board)
        if args.roi:
            from app.modules.change_detection.roi import load_roi
            roi_poly = load_roi(args.roi)["polygon_ref_px"]
    res = evaluate(rows, tol_t_mm=args.tol_t_mm, tol_R_deg=args.tol_R_deg, golden=golden, cal=cal,
                   roi_polygon=roi_poly, spec=spec, seed=args.seed, n_boot=args.n_boot)
    out = Path(args.out)
    write_json(out.with_suffix(".json"), {"experiment": "E2", "logs": args.logs, "seed": args.seed,
                                          **{k: v for k, v in res.items() if k != "trials"}})
    write_json(out.with_name(out.name + "_trials.json"), res["trials"])
    log_tool_run("E2", "eval_reloc", inputs=files + ([args.golden_image] if args.golden_image else []),
                 log_root=args.log_root, run_id=args.run_id, seed=args.seed,
                 outputs={"result": str(out.with_suffix(".json")), "tol_t_mm": res["tol_t_mm"],
                          "tol_R_deg": res["tol_R_deg"],
                          "total_moves_median": {c: {a: s["total_moves"]["median"] for a, s in res["summary"][c].items()}
                                                 for c in res["conditions"]}})
    for cond in res["conditions"]:
        for a, s in res["summary"][cond].items():
            print(f"[{cond}] {a}: total moves median {s['total_moves']['median']} "
                  f"(probe/trial {s['probe_moves_per_trial']:.2f}) success GT {s['success_gt_rate']:.2f} "
                  f"final e_t median {s['final_e_t_mm']['median']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
