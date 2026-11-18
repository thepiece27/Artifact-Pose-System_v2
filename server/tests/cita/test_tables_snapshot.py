"""R10: snapshot tests for the paper tables (E1, E2, E3) from fake eval JSONs.

The columns are the ones fixed in the v1.2 spec (R5.6 E1, R7.5 E2, R10 E3);
rows must come out in the canonical order whatever the input order; cells that
do not apply are "—". Both the Markdown and the LaTeX outputs are pinned.

To regenerate after an INTENDED change of the table format:
    CITA_UPDATE_SNAPSHOTS=1 pytest tests/cita/test_tables_snapshot.py
and review the diff of tests/cita/snapshots/ before committing.
"""
import json
import os
import sys
from pathlib import Path

import pytest

SERVER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SERVER / "tools" / "experiments"))
import make_tables  # noqa: E402

SNAP = Path(__file__).resolve().parent / "snapshots"
NA = "—"


def _dist(median, p95=None, ci=None):
    d = {"n": 12, "median": median, "mean": median, "rmse": median, "p95": p95 if p95 is not None else median,
         "max": median}
    if ci is not None:
        d["median_ci95_cluster"] = ci
    return d


def fake_e1() -> dict:
    def m(et, ci, p95, er, fail, fb, t):
        return {"e_t_first": _dist(et, p95, ci), "e_R_first": _dist(er), "failure_rate": fail,
                "fallback_rate": fb, "time_ms": _dist(t)}
    summary = {  # deliberately NOT in table order
        "M5": m(0.61, [0.42, 0.93], 1.9, 0.0412, 0.0, 0.0, 812.0),
        "M2": m(3.2, [2.1, 4.6], 9.75, 0.21, 0.0833, None, 140.0),
        "M4-jitter": m(0.95, [0.7, 1.31], 2.6, 0.0533, 0.0, 0.0833, 96.0),
        "M1": m(7.4, [5.9, 9.2], 15.1, 0.388, 0.0, None, 31.0),
        "M4": m(0.81, [0.6, 1.12], 2.25, 0.0491, 0.0, 0.0833, 95.0),
        "M3": m(1.4, [1.0, 1.9], 3.8, 0.0702, 0.0, 0.1667, 88.0),
    }
    contrasts = [
        {"contrast": "M3-M1", "a": "M3", "b": "M1", "n_units": 11, "median_diff": -5.9, "p": 0.00098, "p_holm": 0.0049},
        {"contrast": "M4-M3", "a": "M4", "b": "M3", "n_units": 11, "median_diff": -0.55, "p": 0.0034, "p_holm": 0.0136},
        {"contrast": "M5-M4", "a": "M5", "b": "M4", "n_units": 11, "median_diff": -0.18, "p": 0.042, "p_holm": 0.126},
        {"contrast": "M4-M2", "a": "M4", "b": "M2", "n_units": 11, "median_diff": -2.4, "p": 0.0019, "p_holm": 0.0095},
        {"contrast": "M4-M4-jitter", "a": "M4", "b": "M4-jitter", "n_units": 11, "median_diff": -0.12, "p": 0.31,
         "p_holm": 0.31},
    ]
    return {"experiment": "E1", "summary": summary, "paired_tests": {"n_common_units": 11, "contrasts": contrasts}}


def fake_e2() -> dict:
    def a(succ, ci, moves, probe, it, t, et, er, afd):
        return {"n_trials": 30, "total_moves": _dist(moves), "time_s": _dist(t), "success_gt_rate": succ,
                "success_gt_ci95": ci, "probe_moves_per_trial": probe, "iterations": _dist(it),
                "final_e_t_mm": _dist(et), "final_e_R_deg": _dist(er), "final_roi_afd_px": _dist(afd)}
    return {"experiment": "E2", "conditions": ["aligned", "yawed25"], "summary": {
        "aligned": {"C-D": a(0.9667, [0.9, 1.0], 3.0, 0.0667, 3.0, 21.4, 0.84, 0.112, 1.9),
                    "C-A": a(0.9333, [0.83, 1.0], 3.0, 0.0, 3.0, 19.8, 0.97, 0.205, 2.4),
                    "C-C": a(0.9333, [0.83, 1.0], 3.0, 0.0667, 3.0, 22.0, 1.02, 0.198, None),
                    "C-B": a(0.9, [0.8, 1.0], 3.0, 0.0, 3.0, 20.5, 0.91, 0.121, 2.2)},
        "yawed25": {"C-B": a(0.9667, [0.9, 1.0], 5.0, 0.0, 5.0, 33.1, 0.88, 0.117, 2.1),
                    "C-A": a(0.8667, [0.73, 0.97], 5.0, 0.0, 5.0, 31.7, 1.31, 0.244, 2.9),
                    "C-D": a(0.9667, [0.9, 1.0], 3.0, 0.0667, 3.0, 22.9, 0.86, 0.109, 1.8),
                    "C-C": a(0.9333, [0.83, 1.0], 3.0, 0.0667, 3.0, 23.5, 1.05, 0.201, None)},
    }}


def fake_e3() -> dict:
    def p(far, ci, auc, prec, rec, f1, small, mf1, has_classes):
        return {"false_alarm_rate_unchanged": far, "false_alarm_ci95": ci, "roc_auc": auc, "precision": prec,
                "recall": rec, "f1": f1, "recall_by_size": {"small": small, "medium": 0.9, "large": 1.0},
                "class_macro_f1": mf1, "has_classes": has_classes}
    return {"experiment": "E3", "summary": {  # deliberately NOT in table order
        "P4": p(0.04, [0.0, 0.1], 0.962, 0.881, 0.842, 0.861, 0.611, 0.702, True),
        "P3v": p(0.02, [0.0, 0.06], 0.948, 0.912, 0.774, 0.837, 0.5, 0.688, True),
        "P0": p(0.18, [0.08, 0.3], 0.811, 0.512, 0.433, 0.469, 0.0, None, False),
        "P2": p(0.08, [0.02, 0.16], 0.931, 0.803, 0.838, 0.82, 0.611, None, False),
        "P1": p(0.16, [0.06, 0.28], 0.822, 0.55, 0.47, 0.507, 0.0, 0.41, True),
        "P3": p(0.08, [0.02, 0.16], 0.931, 0.803, 0.838, 0.82, 0.611, 0.688, True),
    }}


FAKES = {"E1": fake_e1, "E2": fake_e2, "E3": fake_e3}


def _table_rows(md: str, caption_start: str) -> tuple[list[str], list[list[str]]]:
    lines = md.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(f"**{caption_start}"))
    block = []
    for l in lines[i + 2:]:
        if not l.startswith("|"):
            break
        block.append([c.strip() for c in l.strip().strip("|").split("|")])
    head, rows = block[0], block[2:]
    return head, rows


@pytest.mark.parametrize("exp", ["E1", "E2", "E3"])
@pytest.mark.parametrize("lang", ["vi", "en"])
def test_table_snapshots(exp, lang):
    md, tex = make_tables.render([FAKES[exp]()], lang)
    for ext, text in ((".md", md), (".tex", tex)):
        snap = SNAP / f"{exp}_{lang}{ext}"
        if os.environ.get("CITA_UPDATE_SNAPSHOTS") == "1":
            snap.parent.mkdir(parents=True, exist_ok=True)
            snap.write_text(text, encoding="utf-8", newline="\n")
        assert snap.is_file(), f"missing snapshot {snap.name}; run with CITA_UPDATE_SNAPSHOTS=1"
        assert text == snap.read_text(encoding="utf-8"), f"{snap.name} changed"


def test_e1_columns_rows_and_contrasts():
    md, tex = make_tables.render([fake_e1()], "vi")
    head, rows = _table_rows(md, "E1 – độ chính xác")
    assert head == ["Phương pháp", "e_t trung vị (mm) [CI]", "e_t p95 (mm)", "e_R trung vị (°)", "Thất bại (%)",
                    "Rơi về Diamond (%)", "Thời gian (ms)"]
    assert [r[0] for r in rows] == ["M1", "M2", "M3", "M4", "M4-jitter", "M5"]
    by = {r[0]: r for r in rows}
    assert by["M4"][1] == "0.81 [0.60, 1.12]"            # cluster CI
    assert by["M1"][5] == NA and by["M2"][5] == NA        # no Diamond fallback for M1/M2
    assert by["M3"][5] == "16.7"
    head_c, rows_c = _table_rows(md, "E1 – các phép so sánh")
    assert head_c == ["Phép so sánh", "Số đơn vị", "Chênh trung vị e_t (mm)", "p", "p_Holm"]
    assert [r[0] for r in rows_c] == ["M3-M1", "M4-M3", "M5-M4", "M4-M2", "M4-M4-jitter"]
    assert r"\label{tab:e1}" in tex and r"\label{tab:e1c}" in tex


def test_e2_blocks_per_condition():
    md, tex = make_tables.render([fake_e2()], "vi")
    head, rows = _table_rows(md, "E2 –")
    assert head == ["Cấu hình", "Thành công (%)", "Tổng di chuyển (trung vị)", "Thăm dò/lượt", "Vòng điều khiển",
                    "Thời gian (s)", "e_t cuối (mm)", "e_R cuối (°)", "ROI-AFD cuối (px)"]
    assert [r[0] for r in rows] == [f"{c} · {a}" for c in ("aligned", "yawed25") for a in ("C-A", "C-B", "C-C", "C-D")]
    by = {r[0]: r for r in rows}
    assert by["yawed25 · C-A"][1] == "86.7 [73.0, 97.0]" and by["yawed25 · C-A"][2] == "5.0"
    assert by["aligned · C-C"][8] == NA                    # no ROI-AFD -> dash
    assert r"aligned $\cdot$ C-A" in tex


def test_e3_columns_row_order_and_dashes():
    md, tex = make_tables.render([fake_e3()], "vi")
    head, rows = _table_rows(md, "E3 –")
    assert head == ["Pipeline", "FAR cặp không đổi (%) [CI]", "AUC mức ảnh", "P vùng", "R vùng", "F1 vùng",
                    "Recall vùng nhỏ < 5 mm", "Macro-F1 lớp"]
    assert [r[0] for r in rows] == ["P0", "P1", "P2", "P3", "P3v", "P4"]
    by = {r[0]: r for r in rows}
    assert by["P2"][7] == NA and by["P0"][7] == NA        # no classes -> dash
    assert by["P3v"][7] == "0.688" and by["P3v"][1] == "2.0 [0.0, 6.0]"
    assert "P3 = vùng SSIM; P3v = vùng SSIM được YOLO xác nhận" in md
    assert "---" in tex and NA not in tex                 # LaTeX dash


def test_cli_writes_md_and_tex_with_dotted_prefix(tmp_path):
    evals = []
    for exp, fake in FAKES.items():
        f = tmp_path / f"{exp}.json"
        f.write_text(json.dumps(fake()), encoding="utf-8")
        evals.append(str(f))
    out = tmp_path / "tables" / "all_v1.2"
    assert make_tables.main(["--eval", *evals, "--out", str(out), "--log-root", str(tmp_path / "logs")]) == 0
    md = (tmp_path / "tables" / "all_v1.2.md").read_text(encoding="utf-8")
    tex = (tmp_path / "tables" / "all_v1.2.tex").read_text(encoding="utf-8")
    assert not (tmp_path / "tables" / "all_v1.md").exists()
    expected_md, expected_tex = make_tables.render([fake() for fake in FAKES.values()], "vi")
    assert md == expected_md and tex == expected_tex
    for label in ("tab:e1", "tab:e1c", "tab:e2", "tab:e3"):
        assert label in tex


def test_output_file_keeps_dots_in_prefixes():
    from _common import output_file
    assert output_file("r/eval", ".json") == Path("r/eval.json")
    assert output_file("r/eval.json", ".json") == Path("r/eval.json")
    assert output_file("r/ev_aligned_0.586_1", ".json") == Path("r/ev_aligned_0.586_1.json")
    assert output_file("r/eval.json", "_trials.json") == Path("r/eval_trials.json")
    assert output_file("r/all.md", ".tex") == Path("r/all.tex")
