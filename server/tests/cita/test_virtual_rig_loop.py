"""Phase 3 acceptance on the virtual rig (10 deg slider misalignment, pivot
offset 30 mm, backlash, dead-band, noise)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from app.experiments.profiles import load_profile
from app.modules.artifact_pose.alignment_loop import build_session, run_alignment
from app.modules.sim.virtual_rig import RigParams, VirtualPort, VirtualRig, nominal_fixed_signs

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "server" / "tools" / "experiments"))
import run_reloc_trials  # noqa: E402
from app.experiments.trial_log import read_jsonl  # noqa: E402

PARAMS = RigParams(slider_misalign_deg=10.0, pivot_offset_mm=(0.0, 10.0, -30.0))


def trials(arm, overrides, n=100, tmp=None):
    prof = load_profile(arm, overrides=overrides)
    rng = np.random.default_rng(2027)
    out = []
    for k in range(n):
        rig = VirtualRig(PARAMS, seed=k)
        rig.perturb(**run_reloc_trials.sample_perturbation(rng))
        sess = build_session(prof, arm=arm, server_root=tmp, reuse_jacobian=False)
        res = run_alignment(VirtualPort(rig), sess)
        res["true"] = rig.true_errors()
        out.append(res)
    return out


def test_virtual_rig_kinematics():
    rig = VirtualRig(RigParams.ideal())
    rig.perturb(x_mm=10)
    assert np.allclose(rig.true_errors()["c_mm"], [10, 0, 0], atol=1e-9)
    rig = VirtualRig(RigParams(slider_misalign_deg=10.0, pivot_offset_mm=(0, 0, 0), step_scale_error=(0, 0),
                               deadband_deg=0, backlash_deg=0))
    rig.perturb(x_mm=10)
    c = rig.true_errors()["c_mm"]
    assert c[2] == pytest.approx(-10 * np.sin(np.deg2rad(10)), abs=1e-6)
    rig = VirtualRig(RigParams(slider_misalign_deg=0, pivot_offset_mm=(0, 0, -30), deadband_deg=0, backlash_deg=0))
    rig.perturb(pan_deg=4)
    assert rig.true_errors()["c_norm_mm"] == pytest.approx(2 * 30 * np.sin(np.deg2rad(2)), rel=1e-6)
    # Legacy integer servo angles quantise small commands.
    rig = VirtualRig(RigParams.ideal())
    rig.move({"yaw_delta": 0.4, "pitch_delta": 0.0})
    assert rig.true_errors()["rot_deg"] == 0.0
    rig.move({"yaw_delta": 0.4, "pitch_delta": 0.0, "motion_profile": {"float_angles": True}})
    assert rig.true_errors()["rot_deg"] == pytest.approx(0.4, abs=1e-9)   # 0 (rounded) + 0.4 (float)
    assert nominal_fixed_signs(PARAMS) == {"x": 1, "z": -1, "pan": -1, "tilt": 1}


def test_jacobian_controller_acceptance(tmp_path):
    """Spec 8 acceptance: >= 95 % convergence, median <= 6 iterations, 100 seeded trials."""
    tight = {"control": {"tol_translation_mm": 2.0, "tol_rotation_deg": 0.3}}
    res = trials("C-D", tight, tmp=tmp_path)
    success = np.mean([r["success_est"] for r in res])
    true_success = np.mean([r["true"]["c_norm_mm"] <= 2.0 and r["true"]["rot_deg"] <= 0.3 for r in res])
    iters = [r["control_iterations"] for r in res if r["success_est"]]
    assert success >= 0.95 and true_success >= 0.90
    assert np.median(iters) <= 6
    # Default profile tolerances too (C-C, no I2).
    res = trials("C-C", None, n=40, tmp=tmp_path)
    assert np.mean([r["success_est"] for r in res]) >= 0.95


def test_fixed_mapping_logged_faithfully(tmp_path):
    signs = nominal_fixed_signs(PARAMS)
    res = trials("C-A", {"control": {"signs": signs, "tol_translation_mm": 2.0, "tol_rotation_deg": 0.3}},
                 n=10, tmp=tmp_path)
    for r in res:
        assert r["stop_reason"] in ("converged", "no_improvement", "max_iterations")
        assert r["n_probe_moves"] == 0 and r["n_observations"] >= 1


def test_run_reloc_trials_virtual_all_arms(tmp_path):
    assert run_reloc_trials.main(["--virtual", "--paintings", "P001,P002", "--n-perturb", "3",
                                  "--arms", "C-A,C-B,C-C,C-D", "--seed", "7", "--log-root", str(tmp_path),
                                  "--run-id", "e2", "--jacobian-dir", str(tmp_path / "jac")]) == 0
    rows = read_jsonl(tmp_path / "E2" / "e2.jsonl")
    summaries = [r for r in rows if r.get("record_type") == "summary"]
    assert len(summaries) == 2 * 3 * 4
    assert {r["arm"] for r in summaries} == {"C-A", "C-B", "C-C", "C-D"}
    for r in summaries:
        assert "true_final" in r["output"] and r["input"]["mode"] == "virtual"
    iters = [r for r in rows if r.get("record_type") == "iteration"]
    assert iters and all("iteration" in r["output"] for r in iters)
    # Block-randomised arm order differs across blocks (seeded).
    orders = []
    for i in range(0, len(summaries), 4):
        orders.append(tuple(r["arm"] for r in summaries[i:i + 4]))
    assert len(set(orders)) > 1
    # Jacobian probed once per (painting, arm) then reused.
    probe = [r["output"]["n_probe_moves"] for r in summaries if r["arm"] == "C-C"]
    assert sum(1 for p in probe if p > 0) == 2
