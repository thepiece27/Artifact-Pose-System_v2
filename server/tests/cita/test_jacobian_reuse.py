"""R3: Jacobian files are per arm; probe.reuse = ttl | per_block | never."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from app.experiments.profiles import load_profile
from app.modules.artifact_pose.control import ControlState, make_controller

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "server" / "tools" / "experiments"))
import eval_reloc  # noqa: E402
import run_reloc_trials  # noqa: E402
from app.experiments.trial_log import read_jsonl  # noqa: E402


def probe_to_completion(ctl, rng):
    J_true = rng.normal(0, 1, (6, 4)) + np.vstack([np.eye(4) * 3, np.zeros((2, 4))])
    e0 = rng.normal(0, 5, 6)
    q = np.zeros(4)
    for it in range(10):
        e = e0 + J_true @ q
        cmd = ctl.step(ControlState(iteration=it, c_mm=e[:3] * ctl.scale[:3], theta_deg=e[3:] * ctl.scale[3:]))
        if cmd.kind != "probe":
            return
        q = q + cmd.dq


def test_arms_never_share_a_jacobian(tmp_path):
    rng = np.random.default_rng(0)
    cc = make_controller(load_profile("C-C"), None, device_id="dev", artifact_id="P1", server_root=tmp_path)
    cd = make_controller(load_profile("C-D"), None, device_id="dev", artifact_id="P1", server_root=tmp_path)
    assert cc.store_path != cd.store_path and cc.store_path.name == "jacobian_dev_P1_C-C.json"
    cc.reset({"probe_reuse": "ttl"})
    probe_to_completion(cc, rng)
    saved = json.loads(cc.store_path.read_text())
    assert saved["arm"] == "C-C" and saved["pose_method"] == "M3" and saved["profile_hash"]
    cd.reset({"probe_reuse": "ttl"})
    assert cd.mode == "probe"                      # its own file does not exist
    shutil.copy(cc.store_path, cd.store_path)      # a C-C file where C-D looks
    cd.reset({"probe_reuse": "ttl"})
    assert cd.mode == "probe"
    assert any(e["event"] == "jacobian_rejected_other_arm" for e in cd.events)
    again = make_controller(load_profile("C-C"), None, device_id="dev", artifact_id="P1", server_root=tmp_path)
    again.reset({"probe_reuse": "ttl"})
    assert again.mode == "control" and again.J_source == "stored"


def test_per_block_ignores_ttl_but_not_block(tmp_path):
    rng = np.random.default_rng(1)
    prof = load_profile("C-C", overrides={"control": {"probe": {"ttl_s": 0}}})
    a = make_controller(prof, None, device_id="d", artifact_id="P", server_root=tmp_path)
    a.reset({"probe_reuse": "per_block", "block_id": "run1:P:C-C"})
    probe_to_completion(a, rng)
    b = make_controller(prof, None, device_id="d", artifact_id="P", server_root=tmp_path)
    b.reset({"probe_reuse": "per_block", "block_id": "run1:P:C-C"})
    assert b.mode == "control"                     # TTL 0 s ignored inside the block
    c = make_controller(prof, None, device_id="d", artifact_id="P", server_root=tmp_path)
    c.reset({"probe_reuse": "per_block", "block_id": "run2:P:C-C"})
    assert c.mode == "probe" and any(e["event"] == "jacobian_rejected_other_block" for e in c.events)
    d = make_controller(prof, None, device_id="d", artifact_id="P", server_root=tmp_path)
    d.reset({"probe_reuse": "never", "block_id": "run1:P:C-C"})
    assert d.mode == "probe"
    state = d.to_state()
    assert state["block_id"] == "run1:P:C-C" and state["probe_reuse"] == "never"


@pytest.mark.parametrize("mode,expected", [("per_block", 2), ("never", 10)])
def test_probe_counts_per_arm(tmp_path, mode, expected):
    assert run_reloc_trials.main(["--virtual", "--paintings", "P001,P002", "--n-perturb", "5", "--arms", "C-C,C-D",
                                  "--probe-reuse", mode, "--log-root", str(tmp_path), "--run-id", "r",
                                  "--jacobian-dir", str(tmp_path / "jac")]) == 0
    rows = [r for r in read_jsonl(tmp_path / "E2" / "r.jsonl") if r.get("record_type") == "summary"]
    for arm in ("C-C", "C-D"):
        mine = [r for r in rows if r["arm"] == arm]
        assert len(mine) == 10
        assert sum(1 for r in mine if r["output"]["n_probe_moves"] > 0) == expected, arm
        assert all(r["input"]["probe_reuse"] == mode and r["input"]["block_id"].endswith(arm) for r in mine)
        assert all(r["output"]["total_moves"] == r["output"]["control_iterations"] + r["output"]["n_probe_moves"]
                   for r in mine)
    assert eval_reloc.main(["--logs", str(tmp_path / "E2" / "r.jsonl"), "--out", str(tmp_path / "ev"),
                            "--tol-t-mm", "10", "--tol-R-deg", "1", "--n-boot", "100",
                            "--log-root", str(tmp_path)]) == 0
    ev = json.loads((tmp_path / "ev.json").read_text())
    s = ev["summary"]["aligned"]["C-D"]
    assert s["n_trials_with_probe"] == expected
    assert s["probe_moves_per_trial"] == pytest.approx(s["probe_moves_total"] / s["n_trials"])
    assert {"total_moves", "iterations", "probe_moves_total", "time_s"} <= set(s)
