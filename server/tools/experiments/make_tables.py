#!/usr/bin/env python3
"""Paper tables (Markdown + LaTeX) generated ONLY from evaluation outputs.

    python tools/experiments/make_tables.py --eval results/E1/eval.json --out results/tables/E1
    python tools/experiments/make_tables.py --eval results/E2/eval.json results/E3/eval.json --out results/tables/all [--lang en]

The experiment is read from the ``experiment`` field of each eval JSON
(eval_pose.py -> E1, eval_reloc.py -> E2, eval_change.py -> E3). Numbers are
never typed by hand; re-run this script after every re-evaluation.

Columns (v1.2, R5.6 / R7.5 / R10):
* E1: method | e_t median [CI] | e_t p95 | e_R median | failure % | Diamond fallback % | time,
  plus the pre-registered contrasts (median difference, p_Holm);
* E2: one block per mount condition, rows C-A ... C-D: success % | total moves (median) |
  probe moves / trial | control iterations | time | final e_t | final e_R | final ROI-AFD;
* E3: rows P0, P1, P2, P3, P3v, P4: FAR on unchanged pairs [CI] | image AUC | region P / R / F1 |
  recall small (< 5 mm) | class macro-F1. Cells that do not apply are "—".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_log_args, log_tool_run  # noqa: E402

NA = "—"
E1_ORDER = ["M1", "M2", "M2-full", "M3", "M4", "M4-jitter", "M5"]
E2_ORDER = ["C-A", "C-B", "C-C", "C-D"]
E3_ORDER = ["P0", "P1", "P2", "P3", "P3v", "P4"]

HEADERS = {
    "vi": {
        "E1": ["Phương pháp", "e_t trung vị (mm) [CI]", "e_t p95 (mm)", "e_R trung vị (°)", "Thất bại (%)",
               "Rơi về Diamond (%)", "Thời gian (ms)"],
        "E1c": ["Phép so sánh", "Số đơn vị", "Chênh trung vị e_t (mm)", "p", "p_Holm"],
        "E2": ["Cấu hình", "Thành công (%)", "Tổng di chuyển (trung vị)", "Thăm dò/lượt", "Vòng điều khiển",
               "Thời gian (s)", "e_t cuối (mm)", "e_R cuối (°)", "ROI-AFD cuối (px)"],
        "E3": ["Pipeline", "FAR cặp không đổi (%) [CI]", "AUC mức ảnh", "P vùng", "R vùng", "F1 vùng",
               "Recall vùng nhỏ < 5 mm", "Macro-F1 lớp"],
    },
    "en": {
        "E1": ["Method", "e_t median (mm) [CI]", "e_t p95 (mm)", "e_R median (°)", "Failure (%)",
               "Diamond fallback (%)", "Time (ms)"],
        "E1c": ["Contrast", "Units", "Median e_t difference (mm)", "p", "p_Holm"],
        "E2": ["Configuration", "Success (%)", "Total moves (median)", "Probe moves/trial", "Control iterations",
               "Time (s)", "Final e_t (mm)", "Final e_R (°)", "Final ROI-AFD (px)"],
        "E3": ["Pipeline", "FAR unchanged pairs (%) [CI]", "Image AUC", "Region P", "Region R", "Region F1",
               "Recall small < 5 mm", "Class macro-F1"],
    },
}
CAPTIONS = {
    "vi": {"E1": "E1 – độ chính xác pose so với GT bảng ChArUco độc lập (đơn vị = vị trí, khung đầu tiên; CI bootstrap theo phiên)",
           "E1c": "E1 – các phép so sánh định trước (Wilcoxon ghép cặp theo vị trí, hiệu chỉnh Holm)",
           "E2": "E2 – định vị lại theo điều kiện lắp (nhánh C-A…C-D)",
           "E3": "E3 – phát hiện thay đổi. P3 = vùng SSIM; P3v = vùng SSIM được YOLO xác nhận"},
    "en": {"E1": "E1 – pose accuracy against the independent ChArUco GT (unit = position, first frame; session-cluster bootstrap CI)",
           "E1c": "E1 – pre-registered contrasts (paired Wilcoxon over positions, Holm-adjusted)",
           "E2": "E2 – relocalisation per mount condition (arms C-A…C-D)",
           "E3": "E3 – change detection. P3 = SSIM regions; P3v = SSIM regions confirmed by YOLO"},
}


def fmt(v, nd=2):
    if v is None:
        return NA
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return f"[{fmt(v[0], nd)}, {fmt(v[1], nd)}]"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def pct(v, ci=None):
    if v is None:
        return NA
    s = f"{100 * v:.1f}"
    if ci:
        s += f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"
    return s


def ordered(keys, order):
    return [k for k in order if k in keys] + sorted(k for k in keys if k not in order)


def e1_tables(ev: dict, lang: str) -> list[tuple]:
    rows = []
    for m in ordered(ev["summary"], E1_ORDER):
        s = ev["summary"][m]
        et, er = s["e_t_first"], s["e_R_first"]
        ci = et.get("median_ci95_cluster") or et.get("median_ci95")
        rows.append([m, f"{fmt(et['median'])} {fmt(ci)}", fmt(et["p95"]), fmt(er["median"], 3),
                     pct(s["failure_rate"]), pct(s.get("fallback_rate")), fmt((s.get("time_ms") or {}).get("median"), 0)])
    out = [("E1", HEADERS[lang]["E1"], rows)]
    contrasts = (ev.get("paired_tests") or {}).get("contrasts") or []
    if contrasts:
        crow = [[c["contrast"], str(c["n_units"]), fmt(c["median_diff"]), fmt(c["p"], 4), fmt(c["p_holm"], 4)]
                for c in contrasts]
        out.append(("E1c", HEADERS[lang]["E1c"], crow))
    return out


def e2_tables(ev: dict, lang: str) -> list[tuple]:
    rows = []
    for cond in ev["conditions"]:
        for arm in ordered(ev["summary"][cond], E2_ORDER):
            s = ev["summary"][cond][arm]
            rows.append([f"{cond} · {arm}", pct(s["success_gt_rate"], s.get("success_gt_ci95")),
                         fmt(s["total_moves"]["median"], 1), fmt(s["probe_moves_per_trial"], 2),
                         fmt(s["iterations"]["median"], 1), fmt(s["time_s"]["median"], 1),
                         fmt(s["final_e_t_mm"]["median"]), fmt(s["final_e_R_deg"]["median"], 3),
                         fmt(s["final_roi_afd_px"]["median"])])
    return [("E2", HEADERS[lang]["E2"], rows)]


def e3_tables(ev: dict, lang: str) -> list[tuple]:
    rows = []
    for p in ordered(ev["summary"], E3_ORDER):
        s = ev["summary"][p]
        rows.append([p, pct(s.get("false_alarm_rate_unchanged"), s.get("false_alarm_ci95")), fmt(s.get("roc_auc"), 3),
                     fmt(s.get("precision"), 3), fmt(s.get("recall"), 3), fmt(s.get("f1"), 3),
                     fmt((s.get("recall_by_size") or {}).get("small"), 3),
                     fmt(s.get("class_macro_f1"), 3) if s.get("has_classes") else NA])
    return [("E3", HEADERS[lang]["E3"], rows)]


BUILDERS = {"E1": e1_tables, "E2": e2_tables, "E3": e3_tables}


def markdown(head, rows, caption) -> str:
    out = [f"**{caption}**", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def latex(head, rows, caption, label) -> str:
    def esc(s: str) -> str:
        return (s.replace("%", r"\%").replace("_", r"\_").replace("σ", r"$\sigma$").replace("°", r"$^\circ$")
                .replace("—", "---").replace("·", r"$\cdot$").replace("<", r"$<$"))
    cols = "l" + "r" * (len(head) - 1)
    out = [r"\begin{table}[t]", r"\centering", rf"\caption{{{esc(caption)}}}", rf"\label{{{label}}}",
           rf"\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(esc(h) for h in head) + r" \\", r"\midrule"]
    out += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def render(evals: list[dict], lang: str = "vi") -> tuple[str, str]:
    md, tex = [], []
    for ev in evals:
        exp = ev.get("experiment")
        if exp not in BUILDERS:
            raise SystemExit(f"unknown experiment {exp!r}")
        for key, head, rows in BUILDERS[exp](ev, lang):
            caption = CAPTIONS[lang][key]
            md.append(markdown(head, rows, caption))
            tex.append(latex(head, rows, caption, f"tab:{key.lower()}"))
    return "\n".join(md), "\n".join(tex)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", nargs="+", required=True)
    parser.add_argument("--out", required=True, help="output prefix: writes <out>.md and <out>.tex")
    parser.add_argument("--lang", choices=("vi", "en"), default="vi")
    add_log_args(parser)
    args = parser.parse_args(argv)
    evals = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.eval]
    md, tex = render(evals, args.lang)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    out.with_suffix(".tex").write_text(tex, encoding="utf-8")
    exps = sorted({e.get("experiment") for e in evals})
    log_tool_run("+".join(exps), "make_tables", inputs=args.eval, log_root=args.log_root, run_id=args.run_id,
                 outputs={"markdown": str(out.with_suffix(".md")), "latex": str(out.with_suffix(".tex")),
                          "lang": args.lang})
    print(f"wrote {out.with_suffix('.md')} and {out.with_suffix('.tex')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
