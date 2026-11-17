import numpy as np
import pytest

import synth
from app.modules.artifact_pose.motion import camera_center_in_golden, rotvec
from app.modules.ground_truth.reference_board import (BoardSpec, board_mask, detect_reference_board,
                                                      relative_motion_gt)


@pytest.fixture(scope="module")
def scene():
    return synth.make_scene(11, width=960)


def test_board_pose_and_independent_relative_motion(scene):
    spec = BoardSpec()
    g = detect_reference_board(scene.render(scene.T_gw(), noise_sigma=1.0, supersample=3), scene.K, scene.D, spec)
    assert g["ok"], g["reason"]
    assert g["n_corners"] >= spec.min_corners and g["reproj_rms_px"] < 0.3
    c_true, rot_true = np.array([0.015, 0.0, -0.010]), np.array([0.0, 1.5, 0.0])
    cur = detect_reference_board(scene.render(scene.pose_from_motion(c_true, rot_true), noise_sigma=1.0,
                                              supersample=3), scene.K, scene.D, spec)
    dT = relative_motion_gt(g, cur)
    assert np.linalg.norm(camera_center_in_golden(dT) - c_true) * 1000 < 3.0
    assert np.linalg.norm(np.rad2deg(rotvec(dT[:3, :3])) - rot_true) < 0.25


def test_board_mask_covers_board(scene):
    img = scene.render(scene.T_gw())
    det = detect_reference_board(img, scene.K, scene.D)
    mask = board_mask(img.shape, det["hull_polygon_px"], 10)
    poly = scene.board_polygon_image(scene.T_gw(), margin_mm=-3.0)
    for x, y in np.rint(poly).astype(int):
        assert mask[y, x] == 0
    assert mask[5, 5] == 255 and mask.dtype == np.uint8
    assert (board_mask(img.shape, None) == 255).all()


def test_missing_board_and_id_collision(scene):
    blank = np.full((540, 960, 3), 128, np.uint8)
    det = detect_reference_board(blank, scene.K, scene.D)
    assert not det["ok"] and det["reason"] == "no_charuco_corners"
    with pytest.raises(ValueError, match="overlap"):
        BoardSpec(dictionary="DICT_4X4_50", ids_offset=0).validate()
    with pytest.raises(ValueError, match="has"):
        BoardSpec(dictionary="DICT_4X4_50", ids_offset=41).validate()
    BoardSpec(dictionary="DICT_4X4_100", ids_offset=50).validate()
