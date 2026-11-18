"""R5: E1 statistics with position units, M5 in the paired tests, cluster bootstrap, fallback rate."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "server" / "tools" / "experiments"))
import eval_pose  # noqa: E402
import synth  # noqa: E402
from app.modules.artifact_pose.motion import motion_summary  # noqa: E402

SESSIONS, POSITIONS, REPS = 3, 4, 3


def dT_from(c_mm, rot_deg):
    R = synth.rotvec_deg_to_R(rot_deg)
    return synth.make_T(R, -R @ (np.asarray(c_mm, float) / 1000.0))


def make_records(seed=0, session_bias_mm=3.0, noise_mm=0.5, fallback_positions=((0, 0), (1, 2), (2, 3))):
    rng = np.random.default_rng(seed)
    rows = []
    bias = {s: rng.normal(0, session_bias_mm, 3) for s in range(SESSIONS)}
    for s in range(SESSIONS):
        sid = f"S{s}"
        for p in range(POSITIONS):
            c_gt, rot = rng.uniform(-20, 20, 3), rng.uniform(-2, 2, 3)
            fids = [f"{sid}_p{p}_{k}" for k in range(REPS)]
            for k, fid in enumerate(fids):
                rows.append({"arm": "GT", "status": "ok",
                             "input": {"session_id": sid, "frame_ids": [fid], "pose_index": p, "reps": [k]},
                             "output": motion_summary(dT_from(c_gt, rot))})

            def est(method, fids_, reps, fallback=False):
                c = c_gt + bias[s] * (1.0 if method != "M1" else 2.0) + rng.normal(0, noise_mm, 3)
                dT = dT_from(c, rot)
                out = {"delta_R": dT[:3, :3].tolist(), "delta_t": dT[:3, 3].tolist(),
                       "status": "accepted_diamond_fallback" if fallback else "accepted",
                       "strategy_used": "diamond" if fallback else "hybrid", "n_orb_inliers": 0 if fallback else 80,
                       "sigma_center_mm": [1.0, 1.0, 1.0], "covariance": {"sigma_center_mm_post": [2.0, 2.0, 2.0]}}
                return {"arm": method, "status": "ok", "timing_ms": {"total": 10.0}, "output": out,
                        "input": {"session_id": sid, "frame_ids": fids_, "pose_index": p, "reps": reps}}
            for m in ("M1", "M3", "M4"):
                for k, fid in enumerate(fids):
                    fb = m == "M3" and (s, p) in fallback_positions and k == 0
                    rows.append(est(m, [fid], [k], fallback=fb))
            rows.append(est("M5", fids, list(range(REPS))))
    return rows


def test_units_are_positions_and_m5_is_paired():
    res = eval_pose.evaluate(make_records(), n_boot=300)
    assert set(res["methods"]) == {"M1", "M3", "M4", "M5"}
    for m in res["methods"]:
        assert res["summary"][m]["n_units"] == SESSIONS * POSITIONS == 12
    paired = res["paired_tests"]
    assert paired["n_common_units"] == 12 and "M5" in paired["all_pairs"]["methods"]
    names = {c["contrast"] for c in paired["contrasts"]}
    assert names == {"M3-M1", "M4-M3", "M5-M4"}          # M2 / M4-jitter absent -> skipped
    assert all(0 <= c["p_holm"] <= 1 and c["n_units"] == 12 for c in paired["contrasts"])
    assert res["warnings"] and "session" in res["warnings"][0]


def test_cluster_bootstrap_is_wider_with_session_effects():
    res = eval_pose.evaluate(make_records(session_bias_mm=4.0, noise_mm=0.3), n_boot=1000)
    e = res["summary"]["M4"]["e_t_first"]
    width = lambda ci: ci[1] - ci[0]  # noqa: E731
    assert width(e["median_ci95_cluster"]) > width(e["median_ci95_iid"])
    assert e["median_ci95"] == e["median_ci95_cluster"] and e["n_sessions"] == 3


def test_fallback_rate_and_split():
    res = eval_pose.evaluate(make_records(), n_boot=200)
    s3 = res["summary"]["M3"]
    assert s3["fallback_rate"] == pytest.approx(3 / 12)
    assert s3["e_t_fallback"]["n"] == 3 and s3["e_t_hybrid"]["n"] == 9
    assert res["summary"]["M4"]["fallback_rate"] == 0.0
    # M1 is Diamond-only by design: its fallback rate is "not applicable", not 100 %.
    m1 = [r for r in make_records() if r["arm"] == "M1"]
    for r in m1:
        r["output"].update(strategy_used="diamond", status="accepted")
    res1 = eval_pose.evaluate(m1 + [r for r in make_records() if r["arm"] == "GT"], n_boot=50)
    assert res1["summary"]["M1"]["fallback_rate"] is None
    assert res1["summary"]["M1"]["e_t_fallback"]["n"] == 0
    assert s3["n_orb_inliers"]["p10"] == 0.0 and s3["n_orb_inliers"]["median"] == 80.0
    # Coverage prior (sigma 1 mm) vs posterior (2 mm) on the same errors.
    s4 = res["summary"]["M4"]
    assert s4["coverage_2sigma_post"] >= s4["coverage_2sigma"]


def test_contrast_parsing_with_dashed_names():
    known = {"M2", "M2-full", "M4", "M4-jitter", "M5"}
    assert eval_pose.parse_contrasts("M4-M4-jitter,M4-M2-full,M2-full-M4,M5-M9", known) == [
        ("M4", "M4-jitter"), ("M4", "M2-full"), ("M2-full", "M4")]
