from pathlib import Path

import pytest

from app.experiments import profiles as P
from app.experiments import registry

ALL = ["pbl_legacy", "proposed", "M1", "M2", "M2-full", "M3", "M4", "M5",
       "C-A", "C-B", "C-C", "C-D", "P0", "P1", "P2", "P3", "P4"]


@pytest.mark.parametrize("name", ALL)
def test_every_shipped_profile_loads(name):
    prof = P.load_profile(name)
    assert prof.name == name
    assert len(prof.profile_hash) == 64


def test_pbl_legacy_equals_cita_baseline_defaults():
    prof = P.load_profile("pbl_legacy")
    builtin = P.legacy_profile()
    # Behavioural sections are identical to the built-in (code) defaults.
    for section in ("pose", "golden", "control", "motion_profile", "inspection", "ground_truth"):
        assert getattr(prof, section) == getattr(builtin, section), section
    assert prof.pose.method == "legacy"
    assert (prof.pose.diamond_sigma_px, prof.pose.orb_sigma_px, prof.pose.huber_delta) == (0.5, 1.5, 2.0)
    assert prof.control.max_iterations is None and prof.control.signs is None
    assert prof.motion_payload() == {}
    assert prof.inspection.pipeline == "P0"


def test_branch_semantics():
    assert P.load_profile("M3").pose.diamond_sigma_px == 0.01
    assert P.load_profile("M3").pose.orb_sigma_px == 1.0
    m5 = P.load_profile("M5")
    assert (m5.pose.method, m5.pose.frames, m5.pose.covariance) == ("M5", 3, True)
    assert m5.pose.noise_model == "configs/noise_model.yaml"
    assert P.load_profile("M2").pose.resize_to == [848, 480]
    assert P.load_profile("M2-full").pose.resize_to is None
    assert P.load_profile("C-C").control.controller == "jacobian"
    assert P.load_profile("C-B").motion_payload()["float_angles"] is True
    assert P.load_profile("C-A").motion_payload() == {}
    stop = {(P.load_profile(a).control.tol_translation_mm, P.load_profile(a).control.tol_rotation_deg)
            for a in ("C-A", "C-B", "C-C", "C-D")}
    assert len(stop) == 1, "all E2 arms must share the same stop thresholds"


def _write(tmp: Path, name: str, text: str) -> Path:
    p = tmp / f"{name}.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_extends_deep_merge_and_hash(tmp_path):
    _write(tmp_path, "base", "name: base\npose: {method: M3, diamond_sigma_px: 0.01, orb_sigma_px: 1.0}\n")
    _write(tmp_path, "child", "extends: base\npose: {orb_sigma_px: 2.0}\n")
    child = P.load_profile("child", profile_dir=tmp_path)
    assert child.name == "child"
    assert child.pose.method == "M3" and child.pose.diamond_sigma_px == 0.01 and child.pose.orb_sigma_px == 2.0
    again = P.load_profile("child", profile_dir=tmp_path)
    assert again.profile_hash == child.profile_hash
    _write(tmp_path, "child", "extends: base\npose: {orb_sigma_px: 2.5}\n")
    assert P.load_profile("child", profile_dir=tmp_path).profile_hash != child.profile_hash


@pytest.mark.parametrize("text,needle", [
    ("pose: {methd: M3}\n", "Unknown key"),
    ("pose: {method: M9}\n", "pose.method"),
    ("pose: {orb_sigma_px: -1}\n", "orb_sigma_px"),
    ("pose: {frames: two}\n", "integer"),
    ("pose: {covariance: 1}\n", "boolean"),
    ("control: {gain: 3}\n", "gain"),
    ("motion_profile: {settle_s: 100}\n", "safety"),
    ("profile_hash: abc\n", "computed"),
])
def test_validation_errors(tmp_path, text, needle):
    _write(tmp_path, "bad", "name: bad\n" + text)
    with pytest.raises(P.ProfileError, match=needle):
        P.load_profile("bad", profile_dir=tmp_path)


def test_cyclic_extends_rejected(tmp_path):
    _write(tmp_path, "a", "extends: b\n")
    _write(tmp_path, "b", "extends: a\n")
    with pytest.raises(P.ProfileError, match="Cyclic"):
        P.load_profile("a", profile_dir=tmp_path)


def test_active_profile_env_and_fallback(monkeypatch):
    assert P.active_profile({}).name == "pbl_legacy"
    assert P.active_profile({"CITA_PROFILE": "M3"}).pose.method == "M3"
    with pytest.raises(P.ProfileError):
        P.active_profile({"CITA_PROFILE": "does-not-exist"})
    monkeypatch.setattr(P, "PROFILE_DIR", Path("/nonexistent"))
    monkeypatch.setattr(P, "load_profile", lambda name: (_ for _ in ()).throw(P.ProfileError("missing")))
    assert P.active_profile({}).pose == P.legacy_profile().pose


def test_registry_covers_all_arms():
    codes = {e.code for e in registry.arms()}
    assert set(registry.POSE_METHODS) | set(registry.CONTROL_ARMS) | set(registry.PIPELINES) <= codes
    import importlib.util
    for e in registry.arms():
        assert P.load_profile(e.default_profile)
        try:
            present = importlib.util.find_spec(e.target.partition(":")[0]) is not None
        except ModuleNotFoundError:
            present = False
        if present:
            assert callable(e.resolve())
