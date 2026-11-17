"""replay_pose -> eval_pose -> make_tables on a small synthetic session."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "server" / "tools" / "experiments"))

import eval_pose  # noqa: E402
import make_tables  # noqa: E402
import replay_pose  # noqa: E402
import synth  # noqa: E402
from app.experiments.stats import bootstrap_ci, friedman_wilcoxon, holm, mcnemar, roc_auc, tpr_at_fpr  # noqa: E402
from app.experiments.trial_log import read_jsonl  # noqa: E402


def test_replay_eval_tables(tmp_path):
    session = synth.make_pose_session(tmp_path, session_id="E1S", seed=4, frames_per_pose=3,
                                      offsets=[{"x_mm": 0, "z_mm": 0, "pan_deg": 0, "tilt_deg": 0},
                                               {"x_mm": 10, "z_mm": 0, "pan_deg": 0, "tilt_deg": 0},
                                               {"x_mm": -5, "z_mm": 10, "pan_deg": 1, "tilt_deg": 0}])
    (tmp_path / "nm.yaml").write_text("schema: cita.noise_model.v1\ndiamond_sigma_px: 0.15\norb_sigma_px: 0.7\n")
    logs = tmp_path / "logs"
    assert replay_pose.main(["--session", str(session), "--methods", "M1,M2-full,M3,M4,M5",
                             "--noise-model", str(tmp_path / "nm.yaml"), "--log-root", str(logs),
                             "--run-id", "r1"]) == 0
    rows = read_jsonl(logs / "E1" / "r1.jsonl")
    arms = [r["arm"] for r in rows]
    assert arms.count("GT") == 9 and arms.count("M1") == 3 * 3 and arms.count("M5") == 3
    assert {r["status"] for r in rows if r["arm"] == "GT"} == {"ok"}
    golden = [r for r in rows if r["arm"] == "M4:golden"][0]
    assert golden["output"]["sigmas"] == [0.15, 0.7]
    # Every record carries the reproducibility fields.
    for r in rows:
        assert r["schema"] == "cita.trial.v1" and r["profile_hash"] and "git_commit" in r and r["seed"] == 2027
    out = tmp_path / "eval"
    assert eval_pose.main(["--logs", str(logs / "E1" / "r1.jsonl"), "--out", str(out), "--n-boot", "200"]) == 0
    ev = json.loads(out.with_suffix(".json").read_text())
    assert set(ev["methods"]) == {"M1", "M2-full", "M3", "M4", "M5"}
    for m in ev["methods"]:
        s = ev["summary"][m]
        assert s["n_with_gt"] > 0 and s["failure_rate"] == 0.0
        assert s["e_t_mm"]["median"] < 15.0
    assert ev["summary"]["M2-full"]["e_t_mm"]["median"] < 4.0
    assert ev["summary"]["M4"]["coverage_2sigma"] is not None
    assert "friedman" in ev["paired_tests"]["e_t_mm"]
    assert (tmp_path / "eval_units.csv").exists()
    assert make_tables.main(["--eval", str(out.with_suffix(".json")), "--out", str(tmp_path / "t")]) == 0
    md = (tmp_path / "t.md").read_text(encoding="utf-8")
    assert "| M4 |" in md and r"\begin{tabular}" in (tmp_path / "t.tex").read_text(encoding="utf-8")


def test_stats_helpers():
    rng = np.random.default_rng(0)
    a = rng.normal(1, 0.1, 50)
    lo, hi = bootstrap_ci(a, np.median, seed=1)
    assert lo < np.median(a) < hi
    assert bootstrap_ci(a, np.median, seed=1) == (lo, hi)
    res = friedman_wilcoxon({"x": a, "y": a + 0.5, "z": a + 0.01 * rng.normal(size=50)})
    assert res["friedman"]["p"] < 1e-6
    pairs = {(r["a"], r["b"]): r for r in res["wilcoxon_holm"]}
    assert pairs[("x", "y")]["p_holm"] < 1e-6
    assert holm({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}
    assert mcnemar([1, 1, 0, 1], [0, 1, 0, 0])["a_only"] == 2
    assert roc_auc([2, 3], [0, 1]) == 1.0 and roc_auc([0], [0]) == 0.5
    t = tpr_at_fpr(np.arange(10, 20), np.arange(20), 0.05)
    assert t["fpr"] <= 0.05
