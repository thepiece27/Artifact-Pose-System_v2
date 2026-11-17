import numpy as np
import pytest

from app.modules.artifact_pose.motion import (as_T, average_poses, camera_center_in_golden, invert,
                                              pose_errors, relative_motion, rotation_error_deg, rotvec,
                                              se3_left_perturb)
from synth import make_T, rotvec_deg_to_R


def random_T(rng):
    return make_T(rotvec_deg_to_R(rng.uniform(-20, 20, 3)), rng.uniform(-0.5, 0.5, 3))


def test_appendix_a1_definitions():
    rng = np.random.default_rng(0)
    T_gw = random_T(rng)
    c_true = np.array([0.012, -0.004, 0.020])
    dR = rotvec_deg_to_R([0.5, -2.0, 0.3])
    dT_true = make_T(dR, -dR @ c_true)
    T_cw = dT_true @ T_gw
    dT = relative_motion(T_gw, T_cw)
    assert np.allclose(dT, dT_true)
    assert np.allclose(camera_center_in_golden(dT), c_true)
    # c equals the current camera centre expressed in the golden camera frame.
    C_world = invert(T_cw)[:3, 3]
    assert np.allclose(T_gw[:3, :3] @ C_world + T_gw[:3, 3], c_true)
    assert np.allclose(np.rad2deg(rotvec(dT[:3, :3])), [0.5, -2.0, 0.3])


def test_relative_motion_independent_of_world_frame():
    rng = np.random.default_rng(1)
    T_gw, T_cw, W = random_T(rng), random_T(rng), random_T(rng)
    # Re-expressing the world (board frame anywhere on the plane) leaves dT unchanged.
    assert np.allclose(relative_motion(T_gw @ W, T_cw @ W), relative_motion(T_gw, T_cw))


def test_pose_errors_a2():
    dR = rotvec_deg_to_R([0, 1.0, 0])
    gt = make_T(np.eye(3), [0, 0, 0])
    est = make_T(dR, -dR @ np.array([0.003, 0.0, -0.004]))
    e = pose_errors(est, gt)
    assert e["e_t_mm"] == pytest.approx(5.0)
    assert e["e_R_deg"] == pytest.approx(1.0)
    assert e["c_err_mm"] == pytest.approx([3.0, 0.0, -4.0])
    assert rotation_error_deg(np.eye(3), np.eye(3)) == 0.0


def test_left_perturbation_and_average():
    rng = np.random.default_rng(2)
    T = random_T(rng)
    assert np.allclose(se3_left_perturb(T, np.zeros(6)), T)
    xi = np.r_[0.01, -0.02, 0.005, 0.001, 0.0, -0.002]
    Tp = se3_left_perturb(T, xi)
    assert np.allclose(Tp[:3, 3] - rotvec_deg_to_R(np.rad2deg(xi[:3])) @ T[:3, 3], xi[3:])
    poses = [se3_left_perturb(T, s * xi) for s in (-1, 1)]
    assert np.allclose(average_poses(poses), T, atol=1e-4)
    assert np.allclose(as_T(rotvec(T[:3, :3]), T[:3, 3]), T)
