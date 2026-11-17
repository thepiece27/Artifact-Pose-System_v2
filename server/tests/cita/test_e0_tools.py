"""E0 analysis CLIs on dry-run / synthetic sessions."""
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "server" / "tools" / "experiments"))
sys.path.insert(0, str(ROOT / "embed" / "tools"))

import capture_session  # noqa: E402
import e0_noise  # noqa: E402
import e0_repeat  # noqa: E402
import e0_settle  # noqa: E402
import synth  # noqa: E402
from app.experiments.trial_log import read_jsonl  # noqa: E402


def test_e0_settle_and_repeat_on_dry_run(tmp_path):
    out = tmp_path / "sessions"
    capture_session.main(["--out", str(out), "--exp", "E0.4", "--session-id", "S", "--dry-run", "settle",
                          "--axis", "x", "--delta", "5", "--n-frames", "12", "--trials", "2",
                          "--interval-s", "0.08", "--rest-s", "0"])
    capture_session.main(["--out", str(out), "--exp", "E0.4", "--session-id", "R", "--dry-run", "repeat",
                          "--axis", "pan", "--levels", "1,3", "--reps", "2", "--both-sides"])
    assert e0_settle.main(["--session", str(out / "E0.4" / "S"), "--out", str(tmp_path / "s.json"),
                           "--log-root", str(tmp_path / "logs")]) == 0
    res = json.loads((tmp_path / "s.json").read_text())
    assert res["n_trials"] == 2 and all(len(t["shift_px"]) == 11 for t in res["trials"])
    assert e0_repeat.main(["--session", str(out / "E0.4" / "R"), "--out", str(tmp_path / "r.json"),
                           "--measure", "image_shift", "--log-root", str(tmp_path / "logs")]) == 0
    res = json.loads((tmp_path / "r.json").read_text())
    assert res["mode"] == "repeat" and len(res["groups"]) == 4 and len(res["hysteresis"]) == 2
    logs = list((tmp_path / "logs" / "E0.4").glob("*.jsonl"))
    assert len(logs) == 2 and all(read_jsonl(p)[0]["status"] == "ok" for p in logs)


def test_e0_noise_writes_noise_model(tmp_path):
    session = synth.make_pose_session(tmp_path, session_id="B", frames_per_pose=5,
                                      offsets=[{"x_mm": 0, "z_mm": 0, "pan_deg": 0, "tilt_deg": 0}])
    out = tmp_path / "noise_model.yaml"
    assert e0_noise.main(["--burst", str(session), "--golden-session", str(session), "--out", str(out),
                          "--log-root", str(tmp_path / "logs")]) == 0
    model = yaml.safe_load(out.read_text())
    assert model["schema"] == "cita.noise_model.v1"
    assert 0 < model["diamond_sigma_px"] < 0.5 and 0 < model["orb_sigma_px"] < 1.5
    assert model["sources"][0]["session_id"] == "B" and len(model["content_sha256"]) == 64
    with pytest.raises(SystemExit):
        e0_noise.main(["--burst", str(session), "--out", str(out), "--log-root", str(tmp_path / "logs")])
