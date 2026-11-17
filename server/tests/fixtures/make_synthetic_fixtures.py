#!/usr/bin/env python3
"""Generate the small synthetic regression fixture (Phase 0 baseline).

Writes ``tests/fixtures/synthetic/`` : stereo golden pair, three perturbed
views, and an inspection pair set (reference, unchanged, damaged), plus
``truth.json`` and ``calibration.yaml``. Deterministic for a given seed.

    python tests/fixtures/make_synthetic_fixtures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import synth  # noqa: E402

SEED = 20271
POSE_VIEWS = {
    "current_00": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    "current_01": ([0.010, 0.0, 0.0], [0.0, 0.0, 0.0]),
    "current_02": ([0.020, 0.0, -0.020], [0.0, 2.0, 0.0]),
}


def main(out: Path = HERE / "synthetic") -> Path:
    out.mkdir(parents=True, exist_ok=True)
    scene = synth.make_scene(SEED)
    rng = np.random.default_rng(SEED)
    synth.write_calibration(out / "calibration.yaml", scene)
    truth = {"seed": SEED, "views": {}}

    def save(name, T, **kw):
        img = scene.render(T, rng=rng, **kw)
        cv2.imwrite(str(out / f"{name}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        dT = T @ np.linalg.inv(scene.T_gw())
        truth["views"][name] = {"c_m": (-dT[:3, :3].T @ dT[:3, 3]).tolist(),
                                "rotvec_deg": np.rad2deg(cv2.Rodrigues(dT[:3, :3])[0].ravel()).tolist()}

    save("golden_left", scene.T_gw(), noise_sigma=1.0)
    save("golden_right", scene.pose_from_motion([0.100, 0.0, 0.0]), noise_sigma=1.0)
    for name, (c, rot) in POSE_VIEWS.items():
        save(name, scene.pose_from_motion(c, rot), noise_sigma=1.0)
    # Inspection: small residual misalignment + photometric change.
    T_u = scene.pose_from_motion([0.002, -0.001, 0.003], [0.2, -0.3, 0.1])
    save("inspect_unchanged", T_u, noise_sigma=1.5, gain=1.04, offset=-3)
    tex = scene.damaged_texture(synth.DEFAULT_DAMAGES)
    T_d = scene.pose_from_motion([-0.002, 0.001, -0.002], [-0.2, 0.25, -0.1])
    save("inspect_damaged", T_d, noise_sigma=1.5, gain=0.96, offset=4, texture=tex)
    truth["damages_ref_xyxy"] = [{"class": d.cls, "size_mm": d.size_mm,
                                  "bbox_ref_xyxy": scene.damage_bbox_image(scene.T_gw(), d)}
                                 for d in synth.DEFAULT_DAMAGES]
    truth["roi_polygon_ref_px"] = scene.painting_polygon_image(scene.T_gw()).tolist()
    truth["board_polygon_ref_px"] = scene.board_polygon_image(scene.T_gw()).tolist()
    truth["px_per_mm"] = scene.px_per_mm_golden()
    (out / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(main())
