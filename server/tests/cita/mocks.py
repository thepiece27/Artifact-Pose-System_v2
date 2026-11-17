"""Deterministic stand-ins for services that need hardware or model weights.

``MockModelService`` mimics the public surface of ``ModelService`` used by the
inspection pipeline (``detect_image``, ``detect_crops_batch``). Its outputs are
pure functions of the input pixels, so regression fixtures are reproducible.
The repository ships no ``.pt`` weights, so tests never load YOLO.
"""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np


class MockModelService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def detect_image(self, name: str, image_bytes: bytes):
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        self.calls.append(("detect_image", 1))
        conf = round(0.2 + 0.5 * float(img.mean()) / 255.0, 4)
        return [{"detections": [{"class_id": 0, "class_name": "mock_full", "confidence": conf,
                                 "bbox_xyxy": [w * 0.25, h * 0.25, w * 0.5, h * 0.5]}]}]

    def detect_crops_batch(self, name: str, crop_list: list, conf: float = 0.15, sub_batch: int = 4):
        self.calls.append(("detect_crops_batch", len(crop_list)))
        out = []
        for crop, ox, oy, spec, r_ssim in crop_list:
            h, w = crop.shape[:2]
            score = round(0.15 + 0.6 * float(np.std(crop)) / 128.0, 4)
            if score < conf:
                continue
            x1, y1, x2, y2 = int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)
            out.append({"bbox_xyxy": [ox + x1, oy + y1, ox + x2, oy + y2],
                        "confidence": score,
                        "class_name": "mock_" + ("dark" if crop.mean() < 128 else "light"),
                        "from_crop": spec, "region_ssim": round(float(r_ssim), 4)})
        return out


def fake_settings(root: Path, **overrides) -> SimpleNamespace:
    """The subset of ``Settings`` read by InspectionService / PoseService."""
    root = Path(root)
    values = dict(
        data_dir=root, uploads_dir=root / "uploads", logs_dir=root / "logs",
        default_ai_model_name="default", max_upload_bytes=50_000_000, max_image_pixels=40_000_000,
        max_metadata_bytes=65536, trans_tolerance_mm=25.0, rot_tolerance_deg=2.0,
        steps_per_mm=800.0, sign_move_x=-1, sign_move_z=-1, sign_rotate_pan=-1,
        sign_rotate_tilt=-1, max_alignment_iterations=20, alignment_timeout_sec=300,
        run_pose_on_upload=True, run_ai_on_upload=False, auto_dispatch_pose_command=True,
        run_ai_on_aligned_image=False, artifact_camera_params=root / "calibration.yaml",
        artifact_camera_params_dir=root, artifact_lens_position=None,
        artifact_golden_pose=root / "golden_pose.yaml",
    )
    values.update(overrides)
    return SimpleNamespace(**values)
