#!/usr/bin/env python3
"""Record the behaviour of the ORIGINAL code (tag ``cita-baseline``).

Run on a checkout of ``cita-baseline`` (or any later commit to compare):

    python tests/fixtures/make_synthetic_fixtures.py
    python tests/fixtures/capture_expected_baseline.py [--out DIR]

It drives the public service entry points exactly as the server does:
* ``PoseService.initialize_golden`` + ``PoseService.correct_image``;
* ``InspectionService._analyze_against_reference`` with a mock YOLO;
* ``InspectionService.handle_upload`` alignment loop (motor commands only).

The JSON files in ``tests/fixtures/expected_baseline/`` were produced by this
script at ``cita-baseline`` and are the reference of the ``pbl_legacy``
regression tests.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import platform
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = HERE.parents[1]
sys.path.insert(0, str(SERVER))
sys.path.insert(0, str(HERE.parent / "cita"))
sys.path.insert(0, str(HERE.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mocks import MockModelService, fake_settings  # noqa: E402

FIXTURE = HERE / "synthetic"
ARTIFACT = "fixture01"

# Alignment-loop scenario: deviations returned by a stub pose service.
DEVIATIONS = [
    {"delta_x": 0.012, "delta_z": -0.030, "delta_pan": 1.4, "delta_tilt": -0.6,
     "within_trans_tolerance": False, "within_rot_tolerance": False, "within_tolerance": False},
    {"delta_x": -0.004, "delta_z": 0.008, "delta_pan": 2.6, "delta_tilt": 0.9,
     "within_trans_tolerance": False, "within_rot_tolerance": False, "within_tolerance": False},
    {"delta_x": 0.0011, "delta_z": 0.0021, "delta_pan": -2.4, "delta_tilt": 0.2,
     "within_trans_tolerance": True, "within_rot_tolerance": False, "within_tolerance": False},
    {"delta_x": 0.031, "delta_z": 0.0005, "delta_pan": 0.3, "delta_tilt": -0.1,
     "within_trans_tolerance": False, "within_rot_tolerance": True, "within_tolerance": False},
    {"delta_x": -0.0123456, "delta_z": 0.0300004, "delta_pan": -3.333, "delta_tilt": 2.75,
     "within_trans_tolerance": False, "within_rot_tolerance": False, "within_tolerance": False},
    {"delta_x": 0.001, "delta_z": 0.001, "delta_pan": 0.1, "delta_tilt": 0.1,
     "within_trans_tolerance": True, "within_rot_tolerance": True, "within_tolerance": True},
]


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def strip_volatile(result: dict) -> dict:
    out = dict(result)
    out.pop("elapsed_ms", None)
    return out


def capture_pose(root: Path) -> dict:
    from app.services.pose_service import PoseService
    settings = fake_settings(root, artifact_camera_params=FIXTURE / "calibration.yaml")
    service = PoseService(settings)
    init = service.initialize_golden(FIXTURE / "golden_left.png", FIXTURE / "golden_right.png",
                                     artifact_id=ARTIFACT)
    record = {"initialize": {k: init[k] for k in ("points_3d", "points_2d", "stereo")},
              "initialize_diamond": {k: init["diamond"][k] for k in ("rvec", "tvec", "corners", "rms_px")},
              "corrections": {}}
    for name in ("current_00", "current_01", "current_02"):
        res = service.correct_image(FIXTURE / f"{name}.png", artifact_id=ARTIFACT)
        record["corrections"][name] = strip_volatile(res)
    return jsonable(record)


def capture_inspection(root: Path) -> dict:
    from app.services.inspection_service import InspectionService
    settings = fake_settings(root)
    service = InspectionService(settings=settings, pose_service=None, model_service=MockModelService(),
                                command_service=None, mqtt_bridge=None)
    out = {}
    for name in ("inspect_unchanged", "inspect_damaged"):
        res = service._analyze_against_reference(current_path=FIXTURE / f"{name}.png",
                                                 reference_path=FIXTURE / "golden_left.png",
                                                 artifact_id=ARTIFACT, ts_ms=0)
        res = dict(res)
        for key in ("heatmap_path", "aligned_image_path"):
            if res.get(key):
                res[key] = Path(res[key]).name
        if res.get("detections_json"):
            det = json.loads(res["detections_json"])
            for key in ("annotated_path", "aligned_path"):
                if det.get(key):
                    det[key] = Path(det[key]).name
            res["detections_json"] = det
        out[name] = res
    return jsonable(out)


class _Recorder:
    def __init__(self):
        self.published = []

    def publish_command(self, device_id, payload):
        self.published.append(json.loads(json.dumps(payload)))
        return True, f"cmd/{device_id}"


class _StubPose:
    def __init__(self, deviations):
        self._devs = list(deviations)

    def correct_image(self, image_path, artifact_id=None):
        return {"deviation": dict(self._devs.pop(0)), "accepted": True}


def capture_alignment(root: Path) -> dict:
    from starlette.datastructures import UploadFile
    from app.services.command_service import CommandService
    from app.services.inspection_service import InspectionService
    settings = fake_settings(root)
    recorder = _Recorder()
    service = InspectionService(settings=settings, pose_service=_StubPose(DEVIATIONS),
                                model_service=MockModelService(), command_service=CommandService(),
                                mqtt_bridge=recorder)
    png = (FIXTURE / "current_00.png").read_bytes()
    meta = json.dumps({"device_id": "dev01", "artifact_id": ARTIFACT,
                       "calibration_data": {"workflow": {"auto_alignment_loop": True}}})
    steps = []
    for i in range(len(DEVIATIONS)):
        before = len(recorder.published)
        resp = asyncio.run(service.handle_upload(UploadFile(io.BytesIO(png), filename=f"u{i}.png"), meta))
        cmds = recorder.published[before:]
        for cmd in cmds:
            cmd.pop("task_id", None)
            if isinstance(cmd.get("deviation"), dict):
                cmd["deviation"].get("motor_command", {})
        dispatch = resp.get("correction_dispatch") or {}
        steps.append({"commands": cmds,
                      "dispatch_status": dispatch.get("status"),
                      "dispatch_phase": dispatch.get("alignment_phase"),
                      "motor_command": (resp.get("pose_result") or {}).get("motor_command")})
    return jsonable({"deviations": DEVIATIONS, "steps": steps})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(HERE / "expected_baseline"))
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__,
           "platform": platform.platform()}
    try:
        import skimage
        env["skimage"] = skimage.__version__
    except ImportError:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        results = {"pose": capture_pose(tmp / "pose"),
                   "inspection": capture_inspection(tmp / "inspection"),
                   "alignment": capture_alignment(tmp / "alignment")}
    for name, data in results.items():
        data = {"captured_with": env, **data}
        (out / f"{name}.json").write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
        print(f"wrote {out / (name + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
