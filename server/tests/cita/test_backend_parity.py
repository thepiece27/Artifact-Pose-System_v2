"""g2o (native) vs scipy backend parity (< 0.1 mm, < 0.01 deg) on synthetic data.

Skipped when the native module is not built (CI, Windows dev machines).
"""
from dataclasses import replace

import numpy as np
import pytest

import synth
from app.modules.artifact_pose import common
from app.modules.artifact_pose.geometry import DEFAULT_QUALITY, project, refine_pose
from app.modules.artifact_pose.motion import as_T, camera_center_in_golden, relative_motion, rotation_error_deg

pytestmark = pytest.mark.skipif(not common.HAS_CPP, reason=f"native pose_solver_cpp unavailable: {common.CPP_IMPORT_ERROR}")


def test_g2o_matches_scipy():
    rng = np.random.default_rng(3)
    K, D = synth.scaled_intrinsics(1920), np.zeros(5)
    h = 0.02
    d3 = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    o3 = np.c_[rng.uniform(-0.3, 0.3, 150), rng.uniform(-0.2, 0.2, 150), np.zeros(150)]
    T_gw = synth.make_T(np.diag([1.0, -1.0, -1.0]), [0, 0, 0.65])
    T = synth.make_T(synth.rotvec_deg_to_R([0.3, -1.0, 0.2]), [0.01, -0.005, 0.0]) @ T_gw
    import cv2
    r, t = cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]
    d2 = project(d3, r, t, K, D) + rng.normal(0, 0.2, (4, 2))
    o2 = project(o3, r, t, K, D) + rng.normal(0, 0.8, (150, 2))
    q = replace(DEFAULT_QUALITY, diamond_sigma_px=0.2, orb_sigma_px=0.8)
    a = refine_pose(r, t, d3, d2, o3, o2, K, D, q, native=None)
    b = refine_pose(r, t, d3, d2, o3, o2, K, D, q, native=common.pose_solver_cpp)
    Ta, Tb = as_T(a["rvec"], a["tvec"]), as_T(b["rvec"], b["tvec"])
    dc = camera_center_in_golden(relative_motion(Ta, Tb))
    assert np.linalg.norm(dc) * 1000 < 0.1
    assert rotation_error_deg(Ta[:3, :3], Tb[:3, :3]) < 0.01
