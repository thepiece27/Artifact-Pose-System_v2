#!/usr/bin/env python3
"""E3 evaluation of replay_change JSONL against GT (cita.change_gt.v1).

    python tools/experiments/eval_change.py --logs logs/experiments/E3/<run>.jsonl --pairs data/E3/test_* \
        --out results/E3/eval

Image level: ROC-AUC (image_score), TPR at FPR 5 %, false-alarm rate on
unchanged pairs (>= 1 region). Region level: a prediction matches a GT box if
IoU >= 0.25 OR its centre lies inside the GT box (greedy by score); P/R/F1;
recall by size (small < 5 mm, medium 5-15 mm, large > 15 mm); class macro-F1
on matched regions for pipelines with YOLO. Sensitivity curve vs deliberate
misalignment (offset_mm/offset_deg). Bootstrap CIs over pairs (seeded).
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
from _pairs import PairSet  # noqa: E402

from app.experiments.stats import bootstrap_ci, roc_auc, tpr_at_fpr  # noqa: E402
from app.experiments.trial_log import read_jsonl  # noqa: E402

SIZE_BINS = (("small", 0.0, 5.0), ("medium", 5.0, 15.0), ("large", 15.0, float("inf")))


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def is_match(pred, gt, iou_thr: float = 0.25) -> bool:
    cx, cy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
    return iou(pred, gt) >= iou_thr or (gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3])


def match(preds: list[dict], gts: list[dict], iou_thr: float = 0.25):
    """Greedy by score; returns (tp_pred_idx->gt_idx dict)."""
    order = sorted(range(len(preds)), key=lambda i: -(preds[i].get("score") or 0.0))
    used, pairs = set(), {}
    for i in order:
        best, best_iou = None, -1.0
        for j, g in enumerate(gts):
            if j in used or not is_match(preds[i]["bbox_ref_xyxy"], g["bbox_ref_xyxy"], iou_thr):
                continue
            v = iou(preds[i]["bbox_ref_xyxy"], g["bbox_ref_xyxy"])
            if v > best_iou:
                best, best_iou = j, v
        if best is not None:
            used.add(best)
            pairs[i] = best
    return pairs


def size_bin(mm: float | None) -> str | None:
    if mm is None:
        return None
    for name, lo, hi in SIZE_BINS:
        if lo <= mm < hi:
            return name
    return None


def macro_f1(y_true: list[str], y_pred: list[str]) -> float | None:
    classes = sorted(set(y_true))
    if not classes:
        return None
    f1s = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def evaluate(rows: list[dict], gts: dict, *, seed: int = 2027, n_boot: int = 2000) -> dict:
    per_pair, summary = [], {}
    pipelines = sorted({r["arm"] for r in rows if r.get("input", {}).get("pair_id") in gts})
    for code in pipelines:
        recs = [r for r in rows if r["arm"] == code and r["input"].get("pair_id") in gts]
        pos, neg, unchanged_alarm = [], [], []
        tp = fp = fn = 0
        by_size = {name: [0, 0] for name, _, _ in SIZE_BINS}
        cls_true, cls_pred = [], []
        offsets = []
        for r in recs:
            gt = gts[r["input"]["pair_id"]]
            out = r.get("output") or {}
            regions = out.get("regions") or []
            dets = out.get("detections") or []
            changed = bool(gt["changes"])
            score = out.get("image_score")
            (pos if changed else neg).append(score if score is not None else 0.0)
            if not changed:
                unchanged_alarm.append(len(regions) > 0)
            m = match(regions, gt["changes"])
            ptp = len(m)
            tp += ptp
            fp += len(regions) - ptp
            fn += len(gt["changes"]) - ptp
            matched_gt = set(m.values())
            for j, g in enumerate(gt["changes"]):
                b = size_bin(g.get("size_mm"))
                if b:
                    by_size[b][1] += 1
                    by_size[b][0] += int(j in matched_gt)
                if dets and j in matched_gt:
                    cand = [d for d in dets if is_match(d.get("bbox_ref_xyxy", d.get("bbox_xyxy")), g["bbox_ref_xyxy"])]
                    cls_true.append(g.get("class", "unknown"))
                    cls_pred.append(max(cand, key=lambda d: d["confidence"])["class_name"] if cand else "none")
            rec = {"pipeline": code, "pair_id": r["input"]["pair_id"], "changed": changed, "image_score": score,
                   "n_regions": len(regions), "n_gt": len(gt["changes"]), "tp": ptp,
                   "offset_mm": r["input"].get("offset_mm"), "offset_deg": r["input"].get("offset_deg"),
                   "roi_afd_px": out.get("roi_afd_px")}
            per_pair.append(rec)
            offsets.append(rec)
        prec = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * prec * recall / (prec + recall) if prec and recall else None
        recalls = [p["tp"] / p["n_gt"] for p in per_pair if p["pipeline"] == code and p["n_gt"]]
        # Sensitivity to deliberate misalignment: false alarms / recall per offset bin
        # (translation offset in mm and rotation offset in degrees, from the pair manifest).
        def sensitivity(key: str, bins) -> list:
            out = []
            for lo, hi in bins:
                sel = [o for o in offsets if o.get(key) is not None and lo <= o[key] < hi]
                if sel:
                    un = [o for o in sel if not o["changed"]]
                    ch = [o for o in sel if o["changed"]]
                    out.append({f"{key}_bin": [lo, hi], "n": len(sel),
                                "false_alarm_rate": float(np.mean([o["n_regions"] > 0 for o in un])) if un else None,
                                "recall": float(np.mean([o["tp"] / o["n_gt"] for o in ch])) if ch else None})
            return out
        curve = sensitivity("offset_mm", ((0, 1), (1, 2), (2, 4), (4, 8), (8, float("inf"))))
        curve_deg = sensitivity("offset_deg", ((0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, float("inf"))))
        summary[code] = {
            "n_pairs": len(recs), "n_changed": len(pos), "n_unchanged": len(neg),
            "roc_auc": roc_auc(pos, neg), "tpr_at_fpr5": tpr_at_fpr(pos, neg, 0.05),
            "false_alarm_rate_unchanged": float(np.mean(unchanged_alarm)) if unchanged_alarm else None,
            "false_alarm_ci95": bootstrap_ci([float(a) for a in unchanged_alarm], np.mean, seed=seed, n_boot=n_boot),
            "precision": prec, "recall": recall, "f1": f1,
            "recall_ci95": bootstrap_ci(recalls, np.mean, seed=seed, n_boot=n_boot),
            "recall_by_size": {k: (v[0] / v[1] if v[1] else None) for k, v in by_size.items()},
            "n_by_size": {k: v[1] for k, v in by_size.items()},
            "class_macro_f1": macro_f1(cls_true, cls_pred) if cls_true else None,
            "sensitivity_vs_offset": curve,
            "sensitivity_vs_offset_deg": curve_deg,
            "roi_afd_px_median": float(np.median([p["roi_afd_px"] for p in per_pair if p["pipeline"] == code
                                                  and p["roi_afd_px"] is not None]))
            if any(p["roi_afd_px"] is not None for p in per_pair if p["pipeline"] == code) else None,
        }
    return {"pipelines": pipelines, "summary": summary, "per_pair": per_pair}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", nargs="+", required=True)
    p.add_argument("--pairs", nargs="+", required=True, help="pair sets holding the GT files")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--n-boot", type=int, default=2000)
    add_log_args(p)
    args = p.parse_args(argv)
    rows = []
    for pattern in args.logs:
        for f in sorted(glob.glob(pattern)) or [pattern]:
            rows.extend(read_jsonl(f))
    gts = {}
    for path in args.pairs:
        for q in PairSet(Path(path)).pairs():
            if q["gt_data"] is not None:
                gts[q["pair_id"]] = q["gt_data"]
    res = evaluate(rows, gts, seed=args.seed, n_boot=args.n_boot)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_name(out.name + "_pairs.csv"), "w", newline="", encoding="utf-8") as f:
        keys = ["pipeline", "pair_id", "changed", "image_score", "n_regions", "n_gt", "tp", "offset_mm",
                "offset_deg", "roi_afd_px"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(res["per_pair"])
    write_json(out.with_suffix(".json"), {"experiment": "E3", "logs": args.logs, "seed": args.seed,
                                          **{k: v for k, v in res.items() if k != "per_pair"}})
    log_tool_run("E3", "eval_change", inputs=expand(args.logs) + [str(Path(p) / "pairs.json") for p in args.pairs],
                 log_root=args.log_root, run_id=args.run_id, seed=args.seed,
                 outputs={"result": str(out.with_suffix(".json")),
                          "f1": {c: res["summary"][c]["f1"] for c in res["pipelines"]},
                          "false_alarm_rate_unchanged": {c: res["summary"][c]["false_alarm_rate_unchanged"]
                                                         for c in res["pipelines"]}})
    for code in res["pipelines"]:
        s = res["summary"][code]
        print(f"{code}: AUC={s['roc_auc']} FA={s['false_alarm_rate_unchanged']} P={s['precision']} "
              f"R={s['recall']} F1={s['f1']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
