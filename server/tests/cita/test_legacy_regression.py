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


# ---------------------------------------------------------------------------
# Inspection (P0) and alignment loop (C-A logic)
# ---------------------------------------------------------------------------

def _inspection_service(root, profile=None, pose_service=None, mqtt=None):
    from app.services.command_service import CommandService
    from app.services.inspection_service import InspectionService
    from mocks import MockModelService
    return InspectionService(settings=fake_settings(root), pose_service=pose_service,
                             model_service=MockModelService(), command_service=CommandService(),
                             mqtt_bridge=mqtt, profile=profile)


@pytest.mark.parametrize("profile_name", [None, "pbl_legacy", "P0"])
@pytest.mark.parametrize("name", ["inspect_unchanged", "inspect_damaged"])
def test_p0_inspection_matches_baseline(tmp_path, profile_name, name):
    exp = load("inspection")[name]
    service = _inspection_service(tmp_path, load_profile(profile_name) if profile_name else None)
    # The legacy SIFT/FLANN matcher draws from cv::theRNG(); the baseline was
    # captured in a fresh process (default state = seed 0) analysing
    # inspect_unchanged first, then inspect_damaged. Replay that order.
    cv2.setRNGSeed(0)
    if name == "inspect_damaged":
        service._analyze_against_reference(current_path=SYN / "inspect_unchanged.png",
                                           reference_path=SYN / "golden_left.png", artifact_id=ARTIFACT, ts_ms=0)
    res = service._analyze_against_reference(current_path=SYN / f"{name}.png",
                                             reference_path=SYN / "golden_left.png", artifact_id=ARTIFACT, ts_ms=0)
    for key in ("analysis_status", "analysis_error", "alignment_quality", "sift_inliers", "damage_pct",
                "auto_description"):
        assert res[key] == exp[key], key
    # Floating-point SSIM differs across platforms (Windows capture vs Linux:
    # ~4e-6, reproducible, not noise); integer / categorical keys and the
    # detection list stay exact. expected_baseline is never re-recorded.
    for key in ("ssim", "ssim_gray", "ssim_color"):
        assert res[key] == pytest.approx(exp[key], abs=1e-5), key
    # valid_mask_ratio = warped+eroded mask pixel count / image size. Measured on
    # Linux (OpenCV 4.14.0, numpy 2.4.6): 1.3e-5 (unchanged pair) and 4e-4 (damaged
    # pair, ~210 border pixels of 518 400, i.e. the border moves < 0.1 px) because
    # H differs in its last bits. 1e-3 ~ a 0.15 px shift of the warp border.
    assert res["valid_mask_ratio"] == pytest.approx(exp["valid_mask_ratio"], abs=1e-3)
    det = json.loads(res["detections_json"])

    def same_detections(got, want):
        # Class, confidence, box and crop phase must be identical. region_ssim is
        # the SSIM of a small (~50x50 px) crop of the aligned image, rounded to 4
        # decimals; measured Linux-vs-Windows max difference over the 20 damaged-pair
        # detections: 1.7e-3 (all other fields equal) -> abs 5e-3.
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert {k: v for k, v in g.items() if k != "region_ssim"} == \
                   {k: v for k, v in w.items() if k != "region_ssim"}
            assert g.get("region_ssim") == pytest.approx(w.get("region_ssim"), abs=5e-3)

    same_detections(det["all_detections"], exp["detections_json"]["all_detections"])
    same_detections(res["all_detections"], exp["all_detections"])
    assert Path(res["heatmap_path"]).name == exp["heatmap_path"]
    assert set(exp) <= set(res)


class _Recorder:
    def __init__(self):
        self.published = []

    def publish_command(self, device_id, payload):
        self.published.append(json.loads(json.dumps(payload)))
        return True, f"cmd/{device_id}"


class _StubPose:
    def __init__(self, deviations):
        self._devs = [dict(d) for d in deviations]

    def correct_image(self, image_path, artifact_id=None, **kw):
        return {"deviation": dict(self._devs.pop(0)), "accepted": True}


@pytest.mark.parametrize("profile_name", [None, "pbl_legacy"])
def test_alignment_loop_commands_match_baseline(tmp_path, profile_name):
    import asyncio
    import io
    from starlette.datastructures import UploadFile
    exp = load("alignment")
    recorder = _Recorder()
    service = _inspection_service(tmp_path, load_profile(profile_name) if profile_name else None,
                                  pose_service=_StubPose(exp["deviations"]), mqtt=recorder)
    png = (SYN / "current_00.png").read_bytes()
    meta = json.dumps({"device_id": "dev01", "artifact_id": ARTIFACT,
                       "calibration_data": {"workflow": {"auto_alignment_loop": True}}})
    for i, step in enumerate(exp["steps"]):
        before = len(recorder.published)
        resp = asyncio.run(service.handle_upload(UploadFile(io.BytesIO(png), filename=f"u{i}.png"), meta))
        cmds = recorder.published[before:]
        for c in cmds:
            c.pop("task_id", None)
        assert cmds == step["commands"], i
        dispatch = resp.get("correction_dispatch") or {}
        assert dispatch.get("status") == step["dispatch_status"]
        assert dispatch.get("alignment_phase") == step["dispatch_phase"]
        assert (resp.get("pose_result") or {}).get("motor_command") == step["motor_command"]


def test_fixed_mapping_controller_equals_inline_legacy_formula():
    from app.modules.artifact_pose.control import FixedMappingController
    rng = np.random.default_rng(0)
    ctl = FixedMappingController(steps_per_mm=800.0, sign_x=-1, sign_z=-1, sign_pan=-1, sign_tilt=-1)
    for _ in range(500):
        dev = {"delta_x": float(rng.normal(0, 0.02)), "delta_z": float(rng.normal(0, 0.02)),
               "delta_pan": float(rng.normal(0, 3)), "delta_tilt": float(rng.normal(0, 3)),
               "within_trans_tolerance": bool(rng.integers(2)), "within_rot_tolerance": bool(rng.integers(2))}
        phase = int(rng.integers(2))
        out = ctl.legacy_step(dev, phase)
        # cita-baseline inline formula (inspection_service.py @ eaed45e, lines 977-1060)
        rx = dev["delta_x"] * 1000.0 * 800.0 * -1
        rz = dev["delta_z"] * 1000.0 * 800.0 * -1
        rp, rt = dev["delta_pan"] * -1, dev["delta_tilt"] * -1
        if phase == 0 and dev["within_trans_tolerance"] and not dev["within_rot_tolerance"]:
            phase = 1
        elif phase == 1 and dev["within_rot_tolerance"] and not dev["within_trans_tolerance"]:
            phase = 0
        if phase == 0:
            exp = {"x_steps": abs(int(round(rx))), "x_dir": 1 if rx >= 0 else -1, "z_steps": abs(int(round(rz))),
                   "z_dir": 1 if rz >= 0 else -1, "yaw_delta": 0.0, "pitch_delta": 0.0, "alignment_phase": "translation"}
        else:
            exp = {"x_steps": 0, "x_dir": 1, "z_steps": 0, "z_dir": 1, "yaw_delta": rp, "pitch_delta": rt,
                   "alignment_phase": "rotation"}
        assert out["fields"] == exp and out["next_phase"] == 1 - phase
        assert out["motor_cmd"] == {"enabled": True, "move_x": rx, "move_z": rz, "rotate_pan": rp, "rotate_tilt": rt}
