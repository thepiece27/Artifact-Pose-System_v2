"""R4: P3 is scored on SSIM regions, P3v on SSIM regions confirmed by YOLO."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "server" / "tools" / "experiments"))
import eval_change  # noqa: E402
import synth  # noqa: E402
from app.experiments.profiles import load_profile  # noqa: E402
from app.modules.change_detection.pipelines import run_pipeline, verify_regions  # noqa: E402
from mocks import MockModelService  # noqa: E402


class NoDetectModel(MockModelService):
    def detect_crops_batch(self, name, crop_list, conf=0.15, sub_batch=4):
        return []

    def detect_image(self, name, image_bytes):
        return [{"detections": []}]


def region(i, box, score):
    return {"region_id": i, "bbox_ref_xyxy": box, "score": score, "area_px": 100.0}


def det(box, conf=0.7, cls="scratch"):
    return {"class_name": cls, "confidence": conf, "bbox_xyxy": box, "bbox_ref_xyxy": box}


def test_verify_regions_rule():
    regs = [region(0, [100, 100, 140, 130], 0.9), region(1, [300, 300, 320, 320], 0.5)]
    v = verify_regions(regs, [det([110, 105, 135, 125], 0.4), det([115, 108, 130, 120], 0.8, "stain")])
    assert [r["region_id"] for r in v] == [0]
    assert v[0]["yolo_confidence_max"] == 0.8 and v[0]["yolo_class"] == "stain" and v[0]["n_detections"] == 2
    # Centre-inside rule with low IoU: a big detection box centred in the region.
    assert verify_regions([region(2, [300, 300, 320, 320], 0.5)], [det([250, 250, 370, 370])])
    assert verify_regions(regs, []) == []


def rec(pipeline, pair_id, regions, verified, dets, score, score_v):
    return {"arm": pipeline, "status": "ok", "input": {"pair_id": pair_id, "offset_mm": 1.0, "offset_deg": 0.2},
            "output": {"regions": regions, "detections": dets, "image_score": score,
                       "verified_regions": verified, "image_score_verified": score_v}}


def test_p3v_rows_in_evaluation():
    gts = {"U": {"changes": []},
           "C": {"changes": [{"class": "scratch", "bbox_ref_xyxy": [200, 200, 240, 220], "size_mm": 9.0}]}}
    spurious = region(0, [50, 50, 90, 80], 0.6)                   # SSIM false region, YOLO silent
    true_reg = region(0, [198, 199, 242, 223], 0.9)
    rows = [rec("P3", "U", [spurious], [], [], 0.6, 0.0),
            rec("P3", "C", [true_reg], verify_regions([true_reg], [det([205, 202, 235, 218])]),
                [det([205, 202, 235, 218])], 0.9, 0.9)]
    res = eval_change.evaluate(rows, gts, n_boot=100)
    assert set(res["pipelines"]) == {"P3", "P3v"}
    p3, p3v = res["summary"]["P3"], res["summary"]["P3v"]
    assert p3["false_alarm_rate_unchanged"] == 1.0 and p3v["false_alarm_rate_unchanged"] == 0.0
    assert p3v["recall"] == 1.0 and p3v["precision"] == 1.0
    assert p3v["has_classes"] and p3v["definition"].startswith("SSIM regions confirmed")
    assert p3v["class_macro_f1"] == 1.0


@pytest.fixture(scope="module")
def damaged_pair():
    scene = synth.make_scene(9)
    ref = scene.render(scene.T_gw(), noise_sigma=1.0, supersample=3)
    tex = scene.damaged_texture(synth.DEFAULT_DAMAGES)
    cur = scene.render(scene.pose_from_motion([0.002, 0, 0.002], [0.2, 0.1, 0]), noise_sigma=1.0,
                       supersample=3, texture=tex)
    roi = {"polygon_ref_px": scene.painting_polygon_image(scene.T_gw()), "size_mm": [297, 210]}
    return ref, cur, roi


def test_pipeline_outputs_verified_regions(damaged_pair):
    ref, cur, roi = damaged_pair
    cfg = load_profile("P3", overrides={"inspection": {"threshold": {"tau": 0.3}}}).inspection
    cv2.setRNGSeed(0)
    silent = run_pipeline("P3", reference=ref, current=cur, cfg=cfg, roi=roi, model_service=NoDetectModel())
    assert silent["regions"] and silent["verified_regions"] == [] and silent["image_score_verified"] == 0.0
    cv2.setRNGSeed(0)
    seen = run_pipeline("P3", reference=ref, current=cur, cfg=cfg, roi=roi, model_service=MockModelService())
    ids = {r["region_id"] for r in seen["regions"]}
    assert seen["verified_regions"] and {r["region_id"] for r in seen["verified_regions"]} <= ids
    assert seen["image_score_verified"] == max(r["score"] for r in seen["verified_regions"])
    # P2 has no YOLO stage at all.
    cfg2 = load_profile("P2", overrides={"inspection": {"threshold": {"tau": 0.3}}}).inspection
    assert "verified_regions" not in run_pipeline("P2", reference=ref, current=cur, cfg=cfg2, roi=roi,
                                                  model_service=MockModelService())
