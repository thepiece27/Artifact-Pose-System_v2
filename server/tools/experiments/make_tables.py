#!/usr/bin/env python3
"""Paper tables (Markdown + LaTeX) generated ONLY from evaluation outputs.

    python tools/experiments/make_tables.py --eval results/E1/eval.json --out results/tables/E1
    python tools/experiments/make_tables.py --eval results/E2/eval.json results/E3/eval.json --out results/tables/all

The experiment is read from the ``experiment`` field of each eval JSON
(eval_pose.py -> E1, eval_reloc.py -> E2, eval_change.py -> E3). Numbers are
never typed by hand; re-run this script after every re-evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_log_args, log_tool_run  # noqa: E402


def fmt(v, nd=2):
    if v is None:
        return "–"
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return f"[{fmt(v[0], nd)}, {fmt(v[1], nd)}]"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def pct(v):
    return "–" if v is None else f"{100 * v:.1f}"


def e1_rows(ev: dict) -> tuple[list[str], list[list[str]]]:
    head = ["Method", "n", "Fail (%)", "e_t median (mm) [95% CI]", "e_t RMSE (mm)", "e_t p95 (mm)",
            "e_R median (°)", "2σ coverage (%)", "Time median (ms)"]
    rows = []
    order = ["M1", "M2", "M2-full", "M3", "M4", "M5"]
    for m in sorted(ev["summary"], key=lambda x: order.index(x) if x in order else 99):
        s = ev["summary"][m]
        et, er = s["e_t_mm"], s["e_R_deg"]
        rows.append([m, str(s["n_with_gt"]), pct(s["failure_rate"]),
                     f"{fmt(et['median'])} {fmt(et.get('median_ci95'))}", fmt(et["rmse"]), fmt(et["p95"]),
                     fmt(er["median"], 3), pct(s.get("coverage_2sigma")), fmt(s["time_ms"]["median"], 0)])
    return head, rows


def e2_rows(ev: dict) -> tuple[list[str], list[list[str]]]:
    head = ["Arm", "Trials", "Success GT (%) [95% CI]", "Success est. (%)", "Iterations median",
            "Time median (s)", "Final e_t median (mm)", "Final e_R median (°)", "Final ROI-AFD (px)"]
    rows = []
    for arm in sorted(ev["summary"]):
        s = ev["summary"][arm]
        rows.append([arm, str(s["n_trials"]), f"{pct(s['success_gt_rate'])} {fmt(s.get('success_gt_ci95'))}",
                     pct(s["success_est_rate"]), fmt(s["iterations"]["median"], 1), fmt(s["time_s"]["median"], 1),
                     fmt(s["final_e_t_mm"]["median"]), fmt(s["final_e_R_deg"]["median"], 3),
                     fmt(s.get("final_roi_afd_px", {}).get("median"))])
    return head, rows


def e3_rows(ev: dict) -> tuple[list[str], list[list[str]]]:
    head = ["Pipeline", "Pairs", "ROC-AUC", "TPR@FPR5%", "FA rate unchanged (%)", "Region P", "Region R",
            "Region F1", "Recall small/med/large", "Class macro-F1"]
    rows = []
    for p in sorted(ev["summary"]):
        s = ev["summary"][p]
        by = s.get("recall_by_size", {})
        rows.append([p, str(s["n_pairs"]), fmt(s.get("roc_auc"), 3), fmt(s.get("tpr_at_fpr5", {}).get("tpr"), 3),
                     pct(s.get("false_alarm_rate_unchanged")), fmt(s.get("precision"), 3), fmt(s.get("recall"), 3),
                     fmt(s.get("f1"), 3),
                     "/".join(fmt(by.get(k), 2) for k in ("small", "medium", "large")),
                     fmt(s.get("class_macro_f1"), 3)])
    return head, rows


BUILDERS = {"E1": e1_rows, "E2": e2_rows, "E3": e3_rows}
CAPTIONS = {"E1": "E1 – pose accuracy against the independent ChArUco ground truth",
            "E2": "E2 – relocalisation (arms C-A…C-D)",
            "E3": "E3 – change detection (P0–P4)"}


def markdown(head, rows, caption) -> str:
    out = [f"**{caption}**", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def latex(head, rows, caption, label) -> str:
    esc = lambda s: s.replace("%", r"\%").replace("_", r"\_").replace("σ", r"$\sigma$").replace("°", r"$^\circ$")  # noqa: E731
    cols = "l" + "r" * (len(head) - 1)
    out = [r"\begin{table}[t]", r"\centering", rf"\caption{{{esc(caption)}}}", rf"\label{{{label}}}",
           rf"\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(esc(h) for h in head) + r" \\", r"\midrule"]
    out += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", nargs="+", required=True)
    parser.add_argument("--out", required=True, help="output prefix: writes <out>.md and <out>.tex")
    add_log_args(parser)
    args = parser.parse_args(argv)
    md, tex = [], []
    for path in args.eval:
        ev = json.loads(Path(path).read_text(encoding="utf-8"))
        exp = ev.get("experiment")
        if exp not in BUILDERS:
            raise SystemExit(f"{path}: unknown experiment {exp!r}")
        head, rows = BUILDERS[exp](ev)
        caption = f"{CAPTIONS[exp]} (source: {Path(path).name})"
        md.append(markdown(head, rows, caption))
        tex.append(latex(head, rows, caption, f"tab:{exp.lower()}"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    out.with_suffix(".tex").write_text("\n".join(tex), encoding="utf-8")
    exps = sorted({json.loads(Path(p).read_text(encoding="utf-8")).get("experiment") for p in args.eval})
    log_tool_run("+".join(exps) if len(exps) > 1 else exps[0], "make_tables", inputs=args.eval,
                 log_root=args.log_root, run_id=args.run_id,
                 outputs={"markdown": str(out.with_suffix(".md")), "latex": str(out.with_suffix(".tex"))})
    print(f"wrote {out.with_suffix('.md')} and {out.with_suffix('.tex')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
