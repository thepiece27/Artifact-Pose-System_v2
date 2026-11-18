"""R7: mount conditions, total_moves as primary metric, mandatory tau, PWM step, --noise-from."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "server" / "tools" / "experiments"))
import eval_reloc  # noqa: E402
import make_tables  # noqa: E402
import run_reloc_trials  # noqa: E402
from app.experiments.trial_log import read_jsonl  # noqa: E402
from app.modules.sim.virtual_rig import RigParams, VirtualRig, noise_from_file  # noqa: E402


def test_mount_presets_and_labels(tmp_path):
    assert RigParams.preset("aligned").slider_misalign_deg == 3.0
    assert RigParams.preset("yawed25", pivot_offset_mm=(0, 0, -40)).slider_misalign_deg == 25.0
    with pytest.raises(ValueError):
        RigParams.preset("sideways")
    for mount in ("aligned", "yawed25"):
        assert run_reloc_trials.main(["--virtual", "--mount", mount, "--paintings", "P001", "--n-perturb",
                                      "10" if mount == "yawed25" else "3", "--arms", "C-A,C-B,C-C,C-D",
                                      "--log-root", str(tmp_path), "--run-id", f"r_{mount}",
                                      "--jacobian-dir", str(tmp_path / f"jac_{mount}")]) == 0
    rows = read_jsonl(tmp_path / "E2" / "r_yawed25.jsonl")
    summ = [r for r in rows if r.get("record_type") == "summary"]
    assert len(summ) == 40 and {r["arm"] for r in summ} == {"C-A", "C-B", "C-C", "C-D"}
    for r in rows:
        if r.get("record_type") in ("summary", "iteration"):
            assert r["input"]["mount_condition"] == "yawed25" and r["input"]["painting_yaw_deg"] == 25.0
            assert r["input"]["rig_params"]["slider_misalign_deg"] == 25.0
    logs = [str(tmp_path / "E2" / "r_aligned.jsonl"), str(tmp_path / "E2" / "r_yawed25.jsonl")]
    assert eval_reloc.main(["--logs", *logs, "--out", str(tmp_path / "ev"), "--tol-t-mm", "2", "--tol-R-deg", "1",
                            "--n-boot", "100", "--log-root", str(tmp_path)]) == 0
    ev = json.loads((tmp_path / "ev.json").read_text())
    assert ev["conditions"] == ["aligned", "yawed25"] and ev["tol_t_mm"] == 2.0
    s = ev["summary"]["yawed25"]["C-D"]
    assert s["n_trials"] == 10 and s["total_moves"]["median_ci95"] is not None
    assert list(s)[:3] == ["n_trials", "total_moves", "time_s"]          # primary metrics first
    w = ev["paired_tests"]["yawed25"]["wilcoxon_total_moves"]
    assert {(x["a"], x["b"]) for x in w} == {("C-A", "C-D"), ("C-B", "C-D")} and all("p_holm" in x for x in w)
    assert make_tables.main(["--eval", str(tmp_path / "ev.json"), "--out", str(tmp_path / "t"),
                             "--log-root", str(tmp_path)]) == 0
    md = (tmp_path / "t.md").read_text(encoding="utf-8")
    assert "| aligned · C-A |" in md and "| yawed25 · C-D |" in md and "Thăm dò/lượt" in md


def test_tau_is_mandatory_when_logs_disagree(tmp_path):
    for arm, tol in (("C-A", 10.0), ("C-D", 2.0)):
        assert run_reloc_trials.main(["--virtual", "--n-perturb", "2", "--arms", arm, "--log-root", str(tmp_path),
                                      "--run-id", f"t_{arm}", "--jacobian-dir", str(tmp_path / "jac"),
                                      "--overrides", json.dumps({"control": {"tol_translation_mm": tol}})]) == 0
    logs = [str(tmp_path / "E2" / "t_C-A.jsonl"), str(tmp_path / "E2" / "t_C-D.jsonl")]
    with pytest.raises(SystemExit, match="tol-t-mm"):
        eval_reloc.main(["--logs", *logs, "--out", str(tmp_path / "ev"), "--log-root", str(tmp_path)])
    assert eval_reloc.main(["--logs", *logs, "--out", str(tmp_path / "ev"), "--tol-t-mm", "5", "--tol-R-deg", "1",
                            "--n-boot", "50", "--log-root", str(tmp_path)]) == 0
    # C-A ran with 10 mm > 5 mm evaluation tau -> explicit warning (C-D at 2 mm is fine).
    warnings = json.loads((tmp_path / "ev.json").read_text(encoding="utf-8"))["warnings"]
    assert len(warnings) == 1 and "C-A (10.0 mm" in warnings[0] and "C-D" not in warnings[0]
    assert eval_reloc.main(["--logs", logs[1], "--out", str(tmp_path / "ev2"), "--tol-t-mm", "2", "--tol-R-deg", "1",
                            "--n-boot", "50", "--log-root", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "ev2.json").read_text(encoding="utf-8"))["warnings"] == []


def test_pwm_step_rounding():
    p = RigParams(slider_misalign_deg=0, pivot_offset_mm=(0, 0, 0), deadband_deg=0, backlash_deg=0,
                  noise_c_mm=0, noise_rot_deg=0, pwm_step_deg=0.586)
    rig = VirtualRig(p)
    rig.move({"yaw_delta": 1.0, "pitch_delta": 0.0, "motion_profile": {"float_angles": True}})
    home = 110.0
    expected = round((home + 1.0) / 0.586) * 0.586 - round(home / 0.586) * 0.586
    assert rig.pan.actual == pytest.approx(expected)
    # home and target are both quantised: the relative error is at most ONE PWM step.
    assert abs(rig.pan.actual - 1.0) <= 0.586 + 1e-9
    q = lambda a: round(a / 0.586) * 0.586  # noqa: E731
    # +0.02 deg: 111.02 deg stays on the same PWM tick as 111.00 deg -> no motion.
    rig.move({"yaw_delta": 0.02, "pitch_delta": 0.0, "motion_profile": {"float_angles": True}})
    assert rig.pan.actual == pytest.approx(expected)
    # +0.1 more (1.12 deg total): next tick.
    rig.move({"yaw_delta": 0.1, "pitch_delta": 0.0, "motion_profile": {"float_angles": True}})
    assert rig.pan.actual == pytest.approx(q(home + 1.12) - q(home)) and rig.pan.actual > expected
    assert VirtualRig(RigParams(pwm_step_deg=0.0)).pan.pwm_step == 0.0


def test_noise_from_both_file_types(tmp_path):
    nm = tmp_path / "noise_model.yaml"
    nm.write_text("schema: cita.noise_model.v2\nestimator: vce\ngolden_stereo_source: diamond\n"
                  "diamond_sigma_px: 0.3\norb_sigma_px: 1.0\npose_sigma_center_mm_median: [1.0, 2.0, 3.0]\n"
                  "pose_sigma_rot_deg_median: [0.03, 0.06, 0.03]\n")
    a = noise_from_file(nm)
    assert a["noise_c_mm"] == pytest.approx(2.0) and a["noise_rot_deg"] == pytest.approx(0.04)
    ev = tmp_path / "eval.json"
    ev.write_text(json.dumps({"experiment": "E1", "summary": {"M5": {"sigma_center_mm_median": [0.5, 0.7, 0.9],
                                                                    "sigma_rot_deg_median": [0.01, 0.02, 0.03]}}}))
    b = noise_from_file(ev)
    assert b["noise_c_mm"] == pytest.approx(0.7) and b["noise_rot_deg"] == pytest.approx(0.02)
    assert run_reloc_trials.main(["--virtual", "--n-perturb", "2", "--arms", "C-D", "--noise-from", str(nm),
                                  "--pwm-step-deg", "0.586", "--log-root", str(tmp_path), "--run-id", "nf",
                                  "--jacobian-dir", str(tmp_path / "jac")]) == 0
    r = [x for x in read_jsonl(tmp_path / "E2" / "nf.jsonl") if x.get("record_type") == "summary"][0]
    assert r["input"]["rig_params"]["noise_c_mm"] == pytest.approx(2.0)
    assert r["input"]["rig_params"]["pwm_step_deg"] == 0.586 and r["input"]["noise_from"]["source"].startswith("noise_model")
    with pytest.raises(ValueError):
        noise_from_file(_bad(tmp_path))


def _bad(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("schema: something.else\n")
    return p
