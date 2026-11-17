"""Crop <-> reference coordinate round trip and AIInputPackage schema."""
import json

import cv2
import numpy as np
import pytest

from app.modules.change_detection.ai_input import (SCHEMA, build_package, map_box_to_ref, map_detections_to_ref,
                                                   pad_box, run_two_phase_yolo)
from mocks import MockModelService


def homography():
    return np.array([[0.98, 0.03, 12.0], [-0.02, 1.01, -7.0], [1e-5, -2e-5, 1.0]])


@pytest.mark.parametrize("source", ["aligned", "original_inverse"])
def test_crop_roundtrip_within_one_pixel(tmp_path, source):
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)
    H = homography()
    cur = cv2.warpPerspective(ref, np.linalg.inv(H), (960, 540))
    aligned = cv2.warpPerspective(cur, H, (960, 540))
    regions = [{"region_id": 0, "bbox_ref_xyxy": [400, 200, 460, 240], "area_px": 2400.0, "score": 0.9}]
    pkg, arrays = build_package(pipeline="P3", regions=regions, ref_shape=ref.shape, aligned=aligned, current=cur,
                                H_ref_from_cur=H, crop_source=source, out_dir=tmp_path)
    saved = json.loads((tmp_path / "ai_input_package.json").read_text())
    assert saved["schema"] == SCHEMA and saved["regions"][0]["crops"]["tight"]["file"] == "region000_tight.png"
    crop = pkg["regions"][0]["crops"]["tight"]
    # A detection on a known feature inside the crop maps back to the reference frame within 1 px.
    ref_box = [410.0, 210.0, 440.0, 230.0]
    if source == "aligned":
        src_box = ref_box
    else:
        pts = cv2.perspectiveTransform(np.float64([[ref_box[:2]], [ref_box[2:]]]), np.linalg.inv(H)).reshape(-1, 2)
        src_box = [pts[0, 0], pts[0, 1], pts[1, 0], pts[1, 1]]
    back = map_box_to_ref(src_box, crop["src"], H)
    if source == "aligned":
        assert back == ref_box
    else:
        # Corner-wise round trip (projective box -> bbox may grow by < 1 px for small tilts).
        assert np.max(np.abs(np.asarray(back) - np.asarray(ref_box))) <= 1.0
    sx1, sy1, sx2, sy2 = crop["bbox_src_xyxy"]
    assert arrays[(0, "tight")].shape[:2] == (sy2 - sy1, sx2 - sx1)
    dets = map_detections_to_ref([{"bbox_xyxy": src_box, "confidence": 0.5, "class_name": "x"}], crop, H)
    assert dets[0]["bbox_src_xyxy"] == list(src_box)


def test_pad_box_matches_legacy_formula():
    # cita-baseline _pad_crop on a 50x20 contour at (100, 80) in a 960x540 image.
    assert pad_box(100, 80, 50, 20, 0.30, 40, 960, 540) == (60, 25, 190, 155)
    assert pad_box(0, 0, 10, 10, 0.70, 120, 960, 540) == (0, 0, 130, 130)


def test_two_phase_yolo_boxes_in_reference_frame():
    rng = np.random.default_rng(1)
    ref = rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)
    regions = [{"region_id": 0, "bbox_ref_xyxy": [300, 200, 360, 250], "area_px": 3000.0, "score": 0.5}]
    pkg, arrays = build_package(pipeline="P3", regions=regions, ref_shape=ref.shape, aligned=ref, current=ref,
                                H_ref_from_cur=np.eye(3))
    dets = run_two_phase_yolo(pkg, arrays, model_service=MockModelService(), model_name="m", ssim_source=ref,
                              reference=ref)
    assert dets
    for d in dets:
        x1, y1, x2, y2 = d["bbox_ref_xyxy"]
        assert 0 <= x1 < x2 <= 960 and 0 <= y1 < y2 <= 540
        # Mock detector boxes sit in the middle of the crop that surrounds the region.
        assert x1 <= 360 and x2 >= 300 and y1 <= 250 and y2 >= 200
