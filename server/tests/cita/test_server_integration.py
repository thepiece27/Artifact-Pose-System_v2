"""InspectionService with CITA profiles: inspection pipelines and alignment sessions."""
import asyncio
import io
import json
from pathlib import Path

import numpy as np
import pytest
from starlette.datastructures import UploadFile

from app.experiments.profiles import load_profile
from app.services.command_service import CommandService
from app.services.inspection_service import InspectionService
from mocks import MockModelService, fake_settings

SYN = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic"
LEGACY_KEYS = {"ssim", "ssim_gray", "ssim_color", "damage_pct", "heatmap_path", "auto_description",
               "detections_json", "all_detections", "analysis_status", "analysis_error", "alignment_quality",
               "sift_inliers", "valid_mask_ratio"}


@pytest.mark.parametrize("code", ["P1", "P2", "P3", "P4"])
def test_inspection_pipelines_keep_the_result_contract(tmp_path, code):
    prof = load_profile(code, overrides={"inspection": {"threshold": {"tau": 0.3}}} if code in ("P2", "P3") else None)
    svc = InspectionService(settings=fake_settings(tmp_path), pose_service=None, model_service=MockModelService(),
                            command_service=CommandService(), mqtt_bridge=None, profile=prof)
    res = svc._analyze_against_reference(current_path=SYN / "inspect_damaged.png",
                                         reference_path=SYN / "golden_left.png", artifact_id="fx", ts_ms=1)
    assert LEGACY_KEYS <= set(res) and res["analysis_status"] == "ok", res.get("analysis_error")
    assert res["pipeline"] == code and {"roi_afd_px", "ecc_ok", "ai_input_package_path"} <= set(res)
    det = json.loads(res["detections_json"])
    assert det["pipeline"] == code
    if code in ("P2", "P3"):
        assert res["ssim"] is not None and res["heatmap_path"] and Path(res["heatmap_path"]).exists()
    status = InspectionService._classify_damage_status(res["all_detections"], res["ssim"], res["damage_pct"])
    assert status.value in ("good", "warning", "damaged")


class Recorder:
    def __init__(self):
        self.published = []

    def publish_command(self, device_id, payload):
        self.published.append(json.loads(json.dumps(payload)))
        return True, f"cmd/{device_id}"


class StubPose:
    """Returns a PoseEstimate-like result; error shrinks after every move."""

    def __init__(self):
        self.calls = []
        self.c = np.array([0.030, 0.0, -0.012])
        self.th = np.array([0.0, 2.0, 0.0])

    def correct_image(self, image_path, artifact_id=None, *, profile=None, extra_image_paths=None):
        self.calls.append((Path(image_path).name, [Path(p).name for p in extra_image_paths or []], profile.name))
        est = {"ok": True, "status": "accepted", "camera_center_in_golden_m": self.c.tolist(),
               "rotvec_deg": self.th.tolist(), "sigma_center_mm": [0.3] * 3, "sigma_rot_deg": [0.02] * 3}
        self.c, self.th = self.c * 0.4, self.th * 0.4
        return {"accepted": True, "status": "accepted", "deviation": {"delta_x": 0.0}, "cita_pose": est}


def test_cita_arm_alignment_session_end_to_end(tmp_path):
    rec, pose = Recorder(), StubPose()
    svc = InspectionService(settings=fake_settings(tmp_path), pose_service=pose, model_service=MockModelService(),
                            command_service=CommandService(), mqtt_bridge=rec, profile=load_profile("pbl_legacy"))
    svc.__dict__["_cita_trial_logger"] = None
    from app.experiments.trial_log import TrialLogger
    svc.__dict__["_cita_trial_logger"] = TrialLogger("E2", root=tmp_path / "logs", run_id="srv")
    png = (SYN / "current_00.png").read_bytes()
    wf = {"auto_alignment_loop": True, "cita_arm": "C-D", "cita_trial_id": "T1"}
    meta = json.dumps({"device_id": "dev01", "artifact_id": "fx", "calibration_data": {"workflow": wf}})
    statuses = []
    for i in range(40):
        resp = asyncio.run(svc.handle_upload(UploadFile(io.BytesIO(png), filename=f"u{i}.png"), meta))
        statuses.append(resp["correction_dispatch"]["status"])
        if statuses[-1] in ("complete", "failed"):
            break
    assert statuses[-1] == "complete", statuses
    # M5: three static frames per observation, extra frames requested with keep_pose (no homing).
    assert all(len(extra) == 2 and prof == "C-D" for _, extra, prof in pose.calls)
    captures = [p for p in rec.published if p["action"] == "capture"]
    assert captures and all(p["keep_pose"] is True and p["workflow"]["cita_arm"] == "C-D" for p in captures)
    moves = [p for p in rec.published if p["action"] == "move"]
    assert moves and all(p["motion_profile"]["float_angles"] is True for p in moves)
    assert rec.published[-1]["action"] == "alignment_complete"
    meta_latest = svc._command_service.get_latest_capture_metadata("dev01")
    assert meta_latest["cita_arm"] == "C-D" and meta_latest["alignment_status"] == "complete"
    assert meta_latest["cita_summary"]["success_est"] is True
    rows = [json.loads(line) for line in (tmp_path / "logs" / "E2" / "srv.jsonl").read_text().splitlines()]
    assert rows[-1]["record_type"] == "summary" and rows[-1]["status"] == "success"
    assert "dev01:fx" not in svc._cita_sessions


def test_legacy_alignment_unaffected_without_cita_arm(tmp_path):
    svc = InspectionService(settings=fake_settings(tmp_path), pose_service=None, model_service=MockModelService(),
                            command_service=CommandService(), mqtt_bridge=None, profile=load_profile("pbl_legacy"))
    assert svc._cita_alignment_profile({"auto_alignment_loop": True}) is None
    assert svc._cita_alignment_profile({"cita_arm": "C-A"}).name == "C-A"
    assert InspectionService._profile_needs_cita_loop(load_profile("proposed"))
    assert not InspectionService._profile_needs_cita_loop(load_profile("M3"))
