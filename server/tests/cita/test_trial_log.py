import json
import random

import numpy as np
import pytest

from app.experiments import profiles as P
from app.experiments.session import SessionError, load_session
from app.experiments.trial_log import (TrialLogger, git_info, read_jsonl, seed_everything, sha256_file)

REQUIRED = {"schema", "run_id", "exp", "arm", "profile", "profile_hash", "git_commit", "git_dirty", "seed",
            "env", "input", "output", "timing_ms", "status", "error", "t_start", "t_end"}


def test_logger_appends_and_never_overwrites(tmp_path):
    prof = P.load_profile("M3")
    log = TrialLogger("E1", root=tmp_path, run_id="r1", profile=prof)
    log.log(arm="M3", input={"frame_ids": ["a"]}, output={"x": np.float64(1.5), "v": np.arange(3)})
    with log.trial(arm="M4", input={"frame_ids": ["b"]}) as out:
        out["c_mm"] = [1, 2, 3]
    rows = read_jsonl(log.path)
    assert [r["arm"] for r in rows] == ["M3", "M4"]
    for r in rows:
        assert REQUIRED <= set(r)
        assert r["schema"] == "cita.trial.v1" and r["profile_hash"] == prof.profile_hash
        assert r["env"]["opencv"] and r["env"]["numpy"]
    assert rows[0]["output"] == {"x": 1.5, "v": [0, 1, 2]}
    assert rows[1]["timing_ms"]["total"] >= 0
    with pytest.raises(FileExistsError):
        TrialLogger("E1", root=tmp_path, run_id="r1", profile=prof)
    resumed = TrialLogger("E1", root=tmp_path, run_id="r1", profile=prof, resume=True)
    resumed.log(arm="M5", input={})
    assert len(read_jsonl(log.path)) == 3


def test_trial_context_logs_errors(tmp_path):
    log = TrialLogger("E1", root=tmp_path)
    with pytest.raises(RuntimeError):
        with log.trial(arm="M1", input={}):
            raise RuntimeError("boom")
    row = read_jsonl(log.path)[-1]
    assert row["status"] == "error" and "boom" in row["error"]


def test_nan_is_serialised_as_null(tmp_path):
    log = TrialLogger("E1", root=tmp_path)
    log.log(arm="M1", input={}, output={"e": float("nan"), "f": float("inf")})
    assert read_jsonl(log.path)[0]["output"] == {"e": None, "f": None}


def test_seed_everything_is_deterministic():
    seed_everything(7)
    a = (random.random(), np.random.rand(), np.random.default_rng(7).random())
    seed_everything(7)
    b = (random.random(), np.random.rand(), np.random.default_rng(7).random())
    assert a == b


def test_git_info_and_hash(tmp_path):
    info = git_info()
    assert set(info) == {"commit", "branch", "dirty"}
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def _manifest(tmp_path, frames, **extra):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema": "cita.session.v1", "session_id": "S", "experiment": "E1", "frames": frames, **extra}))
    return tmp_path


def test_session_validation(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    ok = _manifest(tmp_path, [{"frame_id": "a", "file": "a.png", "sha256": sha256_file(tmp_path / "a.png")}],
                   golden={"frame_ids": ["a"], "stereo_right_frame_id": None})
    s = load_session(ok, verify=True)
    assert s.golden_frames()[0].frame_id == "a" and s.evaluation_frames() == []
    (tmp_path / "a.png").write_bytes(b"y")
    with pytest.raises(SessionError, match="Hash"):
        load_session(ok, verify=True)
    _manifest(tmp_path, [{"frame_id": "a", "file": "../a.png"}])
    with pytest.raises(SessionError, match="bare file"):
        load_session(tmp_path)
    _manifest(tmp_path, [{"frame_id": "a", "file": "a.png"}, {"frame_id": "a", "file": "a.png"}])
    with pytest.raises(SessionError, match="Duplicate"):
        load_session(tmp_path)
    _manifest(tmp_path, [{"frame_id": "a", "file": "a.png"}], golden={"frame_ids": ["zz"]})
    with pytest.raises(SessionError, match="unknown frame"):
        load_session(tmp_path)
