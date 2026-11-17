"""``pbl_legacy`` (and no profile) reproduces tag cita-baseline on the fixture.

Reference: tests/fixtures/expected_baseline/*.json, produced by
tests/fixtures/capture_expected_baseline.py on the UNMODIFIED code.
Tolerances are numerical only (cross-platform BLAS / float noise).
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.experiments.profiles import load_profile
from app.modules.artifact_pose import common, correction
from app.modules.artifact_pose import methods as M
from app.services.pose_service import PoseService
from mocks import fake_settings

FIX = Path(__file__).resolve().parents[1] / "fixtures"
SYN = FIX / "synthetic"
EXPECTED = FIX / "expected_baseline"
ARTIFACT = "fixture01"


def load(name):
    return json.loads((EXPECTED / f"{name}.json").read_text(encoding="utf-8"))


def rot(rvec):
    return cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))[0]


@pytest.fixture(scope="module", params=["none", "pbl_legacy"])
def legacy_pose_service(request, tmp_path_factory):
    profile = None if request.param == "none" else load_profile("pbl_legacy")
    settings = fake_settings(tmp_path_factory.mktemp(request.param),
                             artifact_camera_params=SYN / "calibration.yaml")
    return PoseService(settings, profile=profile)


def test_golden_initialization_matches_baseline(legacy_pose_service):
    exp = load("pose")
    init = legacy_pose_service.initialize_golden(SYN / "golden_left.png", SYN / "golden_right.png",
                                                 artifact_id=ARTIFACT)
    assert len(init["points_3d"]) == len(exp["initialize"]["points_3d"])
    np.testing.assert_allclose(init["points_3d"], exp["initialize"]["points_3d"], atol=1e-7)
    np.testing.assert_allclose(init["points_2d"], exp["initialize"]["points_2d"], atol=1e-9)
    assert init["stereo"]["baseline_m"] == pytest.approx(exp["initialize"]["stereo"]["baseline_m"], abs=1e-9)
    np.testing.assert_allclose(rot(init["diamond"]["rvec"]), rot(exp["initialize_diamond"]["rvec"]), atol=1e-8)


@pytest.mark.parametrize("name", ["current_00", "current_01", "current_02"])
def test_corrections_match_baseline(legacy_pose_service, name):
    exp = load("pose")["corrections"][name]
    golden = legacy_pose_service._golden_pose_path(ARTIFACT)
    if not golden.exists():
        legacy_pose_service.initialize_golden(SYN / "golden_left.png", SYN / "golden_right.png",
                                              artifact_id=ARTIFACT)
    res = legacy_pose_service.correct_image(SYN / f"{name}.png", artifact_id=ARTIFACT)
    for key in ("accepted", "status", "strategy_used", "strategy_requested", "num_matches",
                "num_verified_matches"):
        assert res[key] == exp[key], key
    dev, edev = res["deviation"], exp["deviation"]
    assert set(edev) <= set(dev)
    np.testing.assert_allclose(dev["translation_world_m"], edev["translation_world_m"], atol=1e-7)
    for key in ("delta_x", "delta_y", "delta_z", "translation_mag"):
        assert dev[key] == pytest.approx(edev[key], abs=1e-7), key
    for key in ("delta_pan", "delta_tilt", "delta_roll", "rotation_mag"):
        assert dev[key] == pytest.approx(edev[key], abs=1e-6), key
    for key in ("within_tolerance", "within_trans_tolerance", "within_rot_tolerance"):
        assert dev[key] == edev[key], key
    np.testing.assert_allclose(rot(res["hybrid"]["rvec"]), rot(exp["hybrid"]["rvec"]), atol=1e-8)
    np.testing.assert_allclose(res["hybrid"]["tvec"], exp["hybrid"]["tvec"], atol=1e-8)


def test_estimate_pose_legacy_equals_run_correction_step(tmp_path):
    cal = common.read_calibration(SYN / "calibration.yaml")
    K, D = cal["K"], cal["D"]
    left, right = cv2.imread(str(SYN / "golden_left.png")), cv2.imread(str(SYN / "golden_right.png"))
    cfg = M.PoseMethodConfig(method="legacy", backend="python")
    golden = M.build_golden_model(left, right, K, D, cfg=cfg, workdir=tmp_path)
    exp_init = load("pose")["initialize"]
    np.testing.assert_allclose(golden.hybrid["points_3d"], exp_init["points_3d"], atol=1e-7)
    for name in ("current_01", "current_02"):
        img = cv2.imread(str(SYN / f"{name}.png"))
        ref = correction.run_correction_step(img, K, D, golden.hybrid, backend="python")
        est = M.estimate_pose([img], golden, cfg)
        assert est.ok == ref["accepted"] and est.strategy_used == ref["strategy_used"]
        np.testing.assert_allclose(rot(est.rvec), rot(ref["hybrid"]["rvec"]), atol=1e-9)
        np.testing.assert_allclose(est.tvec, ref["hybrid"]["tvec"], atol=1e-10)
        np.testing.assert_allclose(est.deviation["translation_world_m"],
                                   ref["deviation"]["translation_world_m"], atol=1e-10)
