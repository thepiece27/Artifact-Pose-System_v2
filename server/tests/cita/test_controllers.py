import json

import numpy as np
import pytest

from app.experiments.profiles import load_profile
from app.modules.artifact_pose.alignment_loop import AlignmentSession, LoopConfig, build_session
from app.modules.artifact_pose.control import (ActuatorCommand, ControlState, FixedMappingController,
                                               JacobianController, make_controller, stop_check)


def state(c_mm, th_deg, sigma=None, it=0, tol=(10.0, 1.0)):
    c, th = np.asarray(c_mm, float), np.asarray(th_deg, float)
    return ControlState(iteration=it, c_mm=c, theta_deg=th,
                        sigma_c_mm=None if sigma is None else np.full(3, sigma),
                        sigma_theta_deg=None if sigma is None else np.full(3, sigma / 10),
                        within_trans=np.linalg.norm(c) <= tol[0], within_rot=np.linalg.norm(th) <= tol[1])


def linear_plant(J_true):
    """e(q) = e0 + J q on the scaled error; returns c, theta for a q."""
    def measure(e0, q, scale):
        e = e0 + J_true @ q
        return e[:3] * scale[:3], e[3:] * scale[3:]
    return measure


def test_actuator_command_payload_conversion():
    cmd = ActuatorCommand(dq=np.array([1.25, -0.5, 0.3, -0.2]))
    p = cmd.to_move_payload(800.0)
    assert p == {"x_steps": 1000, "x_dir": 1, "z_steps": 400, "z_dir": -1, "yaw_delta": 0.3, "pitch_delta": -0.2}
    legacy = ActuatorCommand(dq=np.zeros(4), legacy_fields={"x_steps": 5, "alignment_phase": "translation"})
    assert legacy.to_move_payload(800.0) == {"x_steps": 5, "alignment_phase": "translation"}


def test_make_controller_falls_back_to_settings(tmp_path):
    from types import SimpleNamespace
    settings = SimpleNamespace(steps_per_mm=640.0, sign_move_x=1, sign_move_z=-1, sign_rotate_pan=1, sign_rotate_tilt=-1)
    ctl = make_controller(load_profile("pbl_legacy"), settings)
    assert isinstance(ctl, FixedMappingController)
    assert (ctl.steps_per_mm, ctl.sign_x, ctl.sign_pan) == (640.0, 1, 1)
    ctl = make_controller(load_profile("C-A", overrides={"control": {"signs": {"x": 1, "z": 1, "pan": 1, "tilt": 1}}}), settings)
    assert ctl.sign_z == 1
    jac = make_controller(load_profile("C-C"), settings, device_id="d1", artifact_id="a/1", server_root=tmp_path)
    assert isinstance(jac, JacobianController) and jac.store_path.name == "jacobian_d1_a-1.json"


@pytest.mark.parametrize("scheme,moves", [("chained_forward", 4), ("central", 8)])
def test_jacobian_probe_recovers_linear_plant(tmp_path, scheme, moves):
    rng = np.random.default_rng(1)
    J_true = rng.normal(0, 1, (6, 4)) + np.vstack([np.eye(4) * 3, np.zeros((2, 4))])
    prof = load_profile("C-C", overrides={"control": {"probe": {"scheme": scheme}}})
    ctl = JacobianController(prof.control, store_path=tmp_path / "J.json")
    scale = ctl.scale
    e0 = rng.normal(0, 5, 6)
    plant = linear_plant(J_true)
    q = np.zeros(4)
    n_probe = 0
    for it in range(30):
        c, th = plant(e0, q, scale)
        cmd = ctl.step(state(c, th, it=it))
        if cmd.kind == "probe":
            n_probe += 1
        else:
            break
        q = q + cmd.dq
    assert n_probe == moves
    assert np.allclose(ctl.J, J_true, atol=1e-9)
    saved = json.loads((tmp_path / "J.json").read_text())
    assert saved["schema"] == "cita.jacobian.v1" and np.allclose(saved["J"], J_true)
    # Re-use within TTL, re-probe when expired.
    fresh = JacobianController(prof.control, store_path=tmp_path / "J.json")
    fresh.reset({"reuse_jacobian": True})
    assert fresh.mode == "control" and fresh.J_source == "stored"
    old = JacobianController(prof.control, store_path=tmp_path / "J.json", clock=lambda: 1e12)
    old.reset({"reuse_jacobian": True})
    assert old.mode == "probe"


def test_jacobian_control_converges_and_clamps(tmp_path):
    rng = np.random.default_rng(2)
    J_true = np.vstack([np.diag([1.0, 1.0, 5.0, 5.0]), rng.normal(0, 0.1, (2, 4))])
    prof = load_profile("C-C", overrides={"control": {"max_step": {"x_mm": 4, "z_mm": 4, "yaw_deg": 1, "pitch_deg": 1}}})
    ctl = JacobianController(prof.control, store_path=None)
    ctl.J, ctl.J_probe, ctl.mode = J_true.copy(), J_true.copy(), "control"
    e0 = np.array([30.0, -20.0, 0.0, 8.0, -6.0, 0.0])
    q = np.zeros(4)
    norms = []
    for it in range(25):
        c, th = linear_plant(J_true)(e0, q, ctl.scale)
        cmd = ctl.step(state(c, th, it=it))
        assert np.all(np.abs(cmd.dq) <= np.array([4, 4, 1, 1]) + 1e-12)
        assert "uncontrollable_residual" in cmd.meta
        q = q + cmd.dq
        norms.append(np.linalg.norm(e0 + J_true @ q))
    # Best reachable error = component of e0 outside the range of J (rows c_y-like).
    optimum = np.linalg.norm(e0 - J_true @ np.linalg.lstsq(J_true, e0, rcond=None)[0])
    assert norms[-1] < optimum + 0.02 * np.linalg.norm(e0)


def test_broyden_update_moves_towards_true_jacobian():
    J_true = np.vstack([np.diag([1.0, 1.0, 3.0, 3.0]), np.zeros((2, 4))])
    prof = load_profile("C-C", overrides={"control": {"broyden": {"enabled": True, "beta": 1.0, "min_step": 0.0,
                                                                  "noise_k_sigma": 0.0}}})
    ctl = JacobianController(prof.control)
    ctl.J = J_true * 1.5
    ctl.J_probe, ctl.mode = ctl.J.copy(), "control"
    e0 = np.array([10.0, 5.0, 0.0, 2.0, 1.0, 0.0])
    q = np.zeros(4)
    errs = []
    for it in range(6):
        c, th = linear_plant(J_true)(e0, q, ctl.scale)
        q = q + ctl.step(state(c, th, sigma=0.01, it=it)).dq
        errs.append(np.linalg.norm(ctl.J - J_true))
    assert errs[-1] < np.linalg.norm(J_true * 0.5)


def test_stop_rules():
    s = state([3, 0, 4], [0.5, 0, 0], sigma=1.0)
    assert stop_check(s, tol_t_mm=5.0, tol_R_deg=1.0)
    assert not stop_check(s, tol_t_mm=5.0, tol_R_deg=1.0, uncertainty_aware=True, k_sigma=2.0)
    assert not stop_check(ControlState(iteration=0), tol_t_mm=5, tol_R_deg=1)


def test_session_no_improvement_and_max_iterations():
    ctl = FixedMappingController(steps_per_mm=800, sign_x=1, sign_z=-1, sign_pan=-1, sign_tilt=1)
    sess = AlignmentSession(ctl, LoopConfig(max_iterations=10, tol_t_mm=1, tol_R_deg=0.1, no_improvement_iters=2))
    est = {"ok": True, "camera_center_in_golden_m": [0.02, 0, 0], "rotvec_deg": [0, 1, 0],
           "deviation": {"delta_x": -0.02, "delta_z": 0.0, "delta_pan": 1.0, "delta_tilt": 0.0}}
    kinds = [sess.on_observation(est).kind for _ in range(3)]
    # 2nd and 3rd observations do not improve -> stop after 2 consecutive.
    assert kinds == ["move", "move", "fail"]
    assert sess.finished.reason == "no_improvement"
    sess2 = AlignmentSession(ctl, LoopConfig(max_iterations=2, tol_t_mm=1, tol_R_deg=0.1, no_improvement_iters=99))
    for k in range(5):
        est = dict(est, camera_center_in_golden_m=[0.02 / (k + 1), 0, 0])
        d = sess2.on_observation(est)
    assert sess2.finished.reason == "max_iterations" and sess2.iteration == 2
    bad = AlignmentSession(ctl, LoopConfig(max_pose_failures=2))
    assert bad.on_observation({"ok": False}).kind == "retry"
    assert bad.on_observation({"ok": False}).kind == "fail"


def test_session_state_roundtrip(tmp_path):
    prof = load_profile("C-D")
    sess = build_session(prof, arm="C-D", server_root=tmp_path, reuse_jacobian=False)
    est = {"ok": True, "camera_center_in_golden_m": [0.03, 0, 0.0], "rotvec_deg": [0, 2.5, 0],
           "sigma_center_mm": [0.3] * 3, "sigma_rot_deg": [0.02] * 3}
    d = sess.on_observation(est)
    assert d.kind == "move" and d.payload["motion_profile"]["float_angles"] is True
    data = json.loads(json.dumps(sess.to_state()))
    again = AlignmentSession.from_state(data, build_session(prof, server_root=tmp_path).controller)
    assert again.controller.probe_step == sess.controller.probe_step == 1
    assert again.n_probe_moves == 1 and again.iteration == 0
