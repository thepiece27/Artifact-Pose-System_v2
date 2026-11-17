"""I2 device behaviour with fake PCA9685/GPIO (never real hardware)."""
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
AGENT = ROOT / "embed" / "device_agent"
TOOLS = ROOT / "embed" / "tools"
for p in (str(AGENT), str(TOOLS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from adapters.hardware_controller import HardwareController  # noqa: E402
from adapters.motion_profile import MotionProfile, unidirectional_waypoints  # noqa: E402
import capture_session  # noqa: E402


class RecordingKit(capture_session.FakeKit):
    def __init__(self):
        super().__init__()
        self.writes = []
        outer = self

        class Servo:
            def __init__(self, ch):
                self.ch, self._angle = ch, None

            @property
            def angle(self):
                return self._angle

            @angle.setter
            def angle(self, value):
                self._angle = value
                outer.writes.append((self.ch, value))
        self.servo = [Servo(i) for i in range(16)]


def make_hw():
    kit = RecordingKit()
    hw = HardwareController(kit=kit, gpio=capture_session.FakeGPIO(), sleep=lambda s: None)
    kit.writes.clear()
    return hw, kit


def test_profile_priority_command_over_env_over_defaults():
    assert MotionProfile.resolve(None, env={}).is_legacy
    env = {"CITA_MOTION_SETTLE_S": "0.8", "CITA_MOTION_FLOAT_ANGLES": "1"}
    p = MotionProfile.resolve(None, env=env)
    assert p.settle_s == 0.8 and p.float_angles
    p = MotionProfile.resolve({"settle_s": 1.5, "float_angles": False}, env=env)
    assert p.settle_s == 1.5 and not p.float_angles
    capped = MotionProfile.resolve({"settle_s": 999, "backlash_deg": 50, "backlash_steps": 10**6}, env={})
    assert capped.settle_s == 10.0 and capped.backlash_deg == 5.0 and capped.backlash_steps == 8000
    assert MotionProfile.resolve({"settle_s": "bad", "unknown": 1}, env={}).settle_s == 0.0


def test_unidirectional_waypoints():
    assert unidirectional_waypoints(110, 112, enabled=True, approach_sign=1, backlash=2) == [112]
    assert unidirectional_waypoints(110, 108, enabled=True, approach_sign=1, backlash=2) == [106, 108]
    assert unidirectional_waypoints(110, 108, enabled=False, approach_sign=1, backlash=2) == [108]
    assert unidirectional_waypoints(1, 0.5, enabled=True, approach_sign=1, backlash=2, lower=0) == [0, 0.5]
    assert unidirectional_waypoints(110, 112, enabled=True, approach_sign=-1, backlash=2) == [114, 112]


def test_legacy_set_pan_tilt_rounds_to_int():
    hw, kit = make_hw()
    hw.set_pan_tilt(110.4, 35.6)
    assert kit.writes == [(0, 110), (1, 36)]
    assert isinstance(kit.writes[0][1], int)
    assert hw.current_yaw == 110.0 and hw.current_pitch == 36.0


def test_float_angles_and_clamp():
    hw, kit = make_hw()
    hw.set_pan_tilt(110.37, 35.61, float_angles=True)
    assert kit.writes == [(0, 110.37), (1, 35.61)]
    hw.set_pan_tilt(250.0, -5.0, float_angles=True)
    assert hw.current_yaw == 180 and hw.current_pitch == 0


def test_unidirectional_approach_sequence_and_release():
    hw, kit = make_hw()
    prof = MotionProfile(float_angles=True, unidirectional=True, backlash_deg=2.0, approach_sign=1)
    wps = hw.approach_pan_tilt(108.5, 36.0, prof)
    # yaw goes down -> overshoot to 106.5 then 108.5; pitch goes up -> direct (pre = target).
    assert wps == [(106.5, 36.0), (108.5, 36.0)]
    assert kit.writes == [(0, 106.5), (1, 36.0), (0, 108.5), (1, 36.0)]
    kit.writes.clear()
    assert hw.approach_pan_tilt(109.0, 36.0, prof) == [(109.0, 36.0)]
    hw.release_servos()
    assert kit.writes[-2:] == [(0, None), (1, None)]
    # Legacy profile: exactly one rounded command.
    kit.writes.clear()
    assert hw.approach_pan_tilt(100.4, 30.0, MotionProfile()) == [(100.4, 30.0)]
    assert kit.writes == [(0, 100), (1, 30)]


def _fake_main_app(monkeypatch):
    if "api_client" not in sys.modules:
        try:
            import api_client  # noqa: F401
        except ImportError:
            stub = types.ModuleType("api_client")
            stub.APIClient = stub.APIConfig = object
            sys.modules["api_client"] = stub
    from runtime import main_app
    app = object.__new__(main_app.MainApp)
    app.config = main_app.AppConfig()
    app.hardware, kit = make_hw()
    captured = []
    app._handle_capture = lambda cmd: captured.append(cmd)
    sleeps = []
    monkeypatch.setattr(main_app.time, "sleep", lambda s: sleeps.append(s))
    return main_app, app, kit, captured, sleeps


def test_compound_move_legacy_unchanged(monkeypatch):
    main_app, app, kit, captured, sleeps = _fake_main_app(monkeypatch)
    monkeypatch.delenv("CITA_MOTION_SETTLE_S", raising=False)
    info = app._handle_compound_move({"action": "move", "x_steps": 0, "z_steps": 0, "yaw_delta": -1.4,
                                      "pitch_delta": 0.6, "alignment_phase": "rotation"})
    assert kit.writes == [(0, 109), (1, 36)]              # rounded as before
    assert sleeps == [] and captured == []
    assert info["motion_profile"] == MotionProfile().to_dict()
    assert info["commanded"]["yaw_deg"] == 109.0


def test_compound_move_with_i2_profile(monkeypatch):
    main_app, app, kit, captured, sleeps = _fake_main_app(monkeypatch)
    prof = {"float_angles": True, "settle_s": 1.2, "unidirectional": True, "backlash_deg": 2.0,
            "servo_release_after_move": True}
    info = app._handle_compound_move({"action": "move", "yaw_delta": -1.4, "pitch_delta": 0.6,
                                      "alignment_phase": "rotation", "motion_profile": prof,
                                      "workflow": {"auto_alignment_loop": True}})
    assert kit.writes[:2] == [(0, 106.6), (1, 35.6)]      # overshoot on yaw, pitch approached from below
    assert kit.writes[2:4] == [(0, 108.6), (1, 35.6)]
    assert kit.writes[-2:] == [(0, None), (1, None)]       # PWM released after settling
    assert 1.2 in sleeps
    assert info["commanded"]["yaw_deg"] == pytest.approx(108.6)
    assert info["motion_profile"]["settle_s"] == 1.2
    assert captured and captured[0]["_internal_after_move"] is True


def test_slider_unidirectional_overshoot(monkeypatch):
    main_app, app, kit, captured, sleeps = _fake_main_app(monkeypatch)
    moves = []
    app._move_slider_x_with_profile = lambda s, d: moves.append((s, d))
    prof = MotionProfile(unidirectional=True, backlash_steps=400, approach_sign=1)
    app._move_slider_axis_i2("x", 1000, -1, prof)
    app._move_slider_axis_i2("x", 1000, 1, prof)
    app._move_slider_axis_i2("x", 1000, -1, MotionProfile())
    assert moves == [(1400, -1), (400, 1), (1000, 1), (1000, -1)]


def test_capture_keep_pose_skips_homing(monkeypatch):
    main_app, app, kit, captured, sleeps = _fake_main_app(monkeypatch)
    app._handle_capture = main_app.MainApp._handle_capture.__get__(app)
    homed = []
    app.hardware.go_home = lambda: homed.append(True)

    class Cam:
        def capture_simple(self, **kw):
            return None
    app.camera = Cam()
    app._handle_capture({"action": "capture", "keep_pose": True})
    assert homed == []
    app._handle_capture({"action": "capture"})
    assert homed == [True]


def test_capture_session_dry_run_manifests(tmp_path):
    from app.experiments.session import load_session
    args = ["--out", str(tmp_path), "--exp", "E1", "--session-id", "G", "--dry-run",
            "--unidirectional", "--backlash-deg", "1", "--float-angles",
            "grid", "--x", "-5,0,5", "--z", "0", "--pan", "-1,0", "--frames", "2", "--golden-frames", "2"]
    assert capture_session.main(args) == 0
    s = load_session(tmp_path / "E1" / "G", verify=True)
    assert len(s.golden_frames()) == 2 and len(s.evaluation_frames()) == 12
    targets = {json.dumps(f.get("target"), sort_keys=True) for f in s.evaluation_frames()}
    assert len(targets) == 6
    m = s.manifest
    assert m["motion_profile"]["unidirectional"] and m["dry_run"] and m["grid"]["seed"] == 2027
    f = s.evaluation_frames()[0]
    assert {"odometry", "t_capture", "settle_s", "camera_metadata", "sha256"} <= set(f.data)
    with pytest.raises(FileExistsError):
        capture_session.main(args)


def test_capture_session_refuses_real_hardware_without_flag(tmp_path):
    assert capture_session.main(["--out", str(tmp_path), "--exp", "E0.2", "burst", "--n", "1"]) == 2
    assert not (tmp_path / "E0.2").exists()
