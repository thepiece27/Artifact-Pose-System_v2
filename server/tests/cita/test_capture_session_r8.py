"""R8: capture_session pre-flight (envelope / duration / disk), confirmation,
AWB handling and the pan-only servo release. Dry-run / fakes only."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT / "embed" / "device_agent"), str(ROOT / "embed" / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import capture_session  # noqa: E402
from adapters.motion_profile import MotionProfile  # noqa: E402


def _args(argv):
    return capture_session.build_parser().parse_args(capture_session._join_negative_values(list(argv)))


def test_plan_envelope_includes_stereo_baseline_and_backlash_overshoot(tmp_path):
    args = _args(["--out", str(tmp_path), "--exp", "E1", "--dry-run", "--unidirectional", "--backlash-steps",
                  "800", "--backlash-deg", "1", "grid", "--x", "-20,0,20", "--z", "0", "--pan", "-2,0,2",
                  "--frames", "2", "--golden-frames", "3", "--with-stereo", "--baseline-mm", "100", "--return-home"])
    prof = capture_session.motion_profile_from_args(args)
    plan = capture_session.plan_session(args, prof)
    env = plan["envelope"]
    # Stereo baseline is the largest +X excursion.
    assert env["x_mm"][1] == pytest.approx(100.0)
    # Approach sign +1: going to -20 mm overshoots by backlash_steps / steps_per_mm below the target.
    b_mm = 800 / args.steps_per_mm
    assert env["x_mm"][0] == pytest.approx(-20.0 - b_mm)
    assert env["pan_deg"][0] == pytest.approx(-3.0) and env["pan_deg"][1] == pytest.approx(2.0)
    assert plan["n_frames"] == 3 + 1 + 9 * 2
    assert plan["n_targets"] == 1 + 2 + 9 + 1
    assert plan["estimated_duration_s"] > 0 and plan["slider_steps_total"] > 0
    assert plan["estimated_mb"] == pytest.approx(0.3 * plan["n_frames"], abs=0.05)


def test_plan_duration_and_size_scale_with_pulse_delay_and_format(tmp_path):
    base = ["--out", str(tmp_path), "--exp", "E0.2", "--i-know-hardware-moves"]
    slow = _args(base + ["--pulse-delay-us", "500", "stereo", "--baseline-mm", "100"])
    fast = _args(base + ["--pulse-delay-us", "250", "stereo", "--baseline-mm", "100"])
    p_slow = capture_session.plan_session(slow, MotionProfile())
    p_fast = capture_session.plan_session(fast, MotionProfile())
    steps = p_slow["slider_steps_total"]
    assert steps == int(round(200 * slow.steps_per_mm))           # out 100 mm and back
    assert p_slow["estimated_duration_s"] - p_fast["estimated_duration_s"] == pytest.approx(
        steps * 2 * 250e-6, abs=0.2)
    # Real (non dry-run) 4K PNG estimate.
    assert p_slow["frame_mb_estimate"] == pytest.approx(25.0)
    jpg = _args(base + ["--format", "jpg", "burst", "--n", "10"])
    assert capture_session.plan_session(jpg, MotionProfile())["estimated_mb"] == pytest.approx(50.0)


def test_check_disk(tmp_path):
    ok = capture_session.check_disk(tmp_path / "s", 1.0)
    assert ok["ok"] and ok["factor"] == 1.2 and (tmp_path / "s").is_dir()
    assert not capture_session.check_disk(tmp_path, 1e12)["ok"]


def test_refuses_when_disk_too_small(tmp_path, monkeypatch):
    monkeypatch.setattr(capture_session, "FRAME_MB", {"png": 1e12, "jpg": 1e12})
    monkeypatch.setattr(capture_session, "HardwareController",
                        lambda *a, **k: pytest.fail("hardware must not be constructed"))
    rc = capture_session.main(["--out", str(tmp_path), "--exp", "E0.2", "--i-know-hardware-moves",
                               "--awb", "1.5,1.6", "--confirm-envelope", "burst", "--n", "2"])
    assert rc == 5


def test_refuses_without_envelope_confirmation(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt="": "no")
    monkeypatch.setattr(capture_session, "HardwareController",
                        lambda *a, **k: pytest.fail("hardware must not be constructed"))
    rc = capture_session.main(["--out", str(tmp_path), "--exp", "E0.2", "--i-know-hardware-moves",
                               "--awb", "1.5,1.6", "grid", "--x", "-10,0,10", "--z", "0", "--pan", "0",
                               "--with-stereo"])
    assert rc == 3
    out = capsys.readouterr().out
    assert "[PLAN]" in out and "+100.0" in out
    assert not (tmp_path / "E0.2").exists()


def test_eof_on_confirmation_is_a_refusal(tmp_path, monkeypatch):
    def eof(prompt=""):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    monkeypatch.setattr(capture_session, "HardwareController",
                        lambda *a, **k: pytest.fail("hardware must not be constructed"))
    assert capture_session.main(["--out", str(tmp_path), "--exp", "E0.2", "--i-know-hardware-moves",
                                 "--awb", "1.5,1.6", "burst", "--n", "1"]) == 3


def test_real_capture_requires_awb(tmp_path, monkeypatch):
    monkeypatch.setattr(capture_session, "HardwareController",
                        lambda *a, **k: pytest.fail("hardware must not be constructed"))
    assert capture_session.main(["--out", str(tmp_path), "--exp", "E0.2", "--i-know-hardware-moves",
                                 "--confirm-envelope", "burst", "--n", "1"]) == 4


def test_awb_subcommand_with_fake_camera_and_awb_from(tmp_path):
    out = tmp_path / "awb.json"
    assert capture_session.main(["--out", str(tmp_path), "--exp", "E1", "--dry-run", "--lighting", "L1",
                                 "awb", "--seconds", "3", "--awb-out", str(out)]) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["schema"] == "cita.awb.v1" and rec["colour_gains"] == [1.85, 1.62]
    assert rec["lighting"] == "L1" and rec["auto_seconds"] == 3.0 and rec["fake"]
    # Default output location.
    assert capture_session.main(["--out", str(tmp_path / "o"), "--exp", "E1", "--dry-run", "awb"]) == 0
    assert (tmp_path / "o" / "awb_gains.json").is_file()
    # The measured gains are used (and recorded) by a capture session.
    assert capture_session.main(["--out", str(tmp_path), "--exp", "E0.2", "--session-id", "B", "--dry-run",
                                 "--awb-from", str(out), "burst", "--n", "2"]) == 0
    m = json.loads((tmp_path / "E0.2" / "B" / "manifest.json").read_text(encoding="utf-8"))
    assert m["camera"]["colour_gains"] == [1.85, 1.62]
    assert m["camera"]["awb"]["source"] == "awb_from" and m["camera"]["awb"]["lux"] == 320.0
    assert m["plan"]["n_frames"] == 2 and m["disk_check"]["ok"]
    assert m["disk_check_first_frame"]["ok"] and m["disk_check_first_frame"]["frame_mb_measured"] > 0
    assert m["pulse_delay_us"] == 500


def test_dry_run_default_awb_still_works(tmp_path):
    assert capture_session.main(["--out", str(tmp_path), "--exp", "E0.2", "--session-id", "D", "--dry-run",
                                 "burst", "--n", "1"]) == 0
    m = json.loads((tmp_path / "E0.2" / "D" / "manifest.json").read_text(encoding="utf-8"))
    assert m["camera"]["colour_gains"] == [1.2, 1.5] and m["camera"]["awb"]["source"] == "dry_run_default"


def test_pulse_delay_reaches_the_slider_driver(tmp_path, monkeypatch):
    seen = {}
    real = capture_session.HardwareController

    def spy(*a, **k):
        seen["pulse_delay"] = k["slider_config"].pulse_delay
        return real(*a, **k)
    monkeypatch.setattr(capture_session, "HardwareController", spy)
    assert capture_session.main(["--out", str(tmp_path), "--exp", "E0.2", "--dry-run", "--pulse-delay-us", "300",
                                 "burst", "--n", "1"]) == 0
    assert seen["pulse_delay"] == pytest.approx(300e-6)


def test_servo_release_never_touches_tilt(tmp_path):
    writes = []

    class Servo:
        def __init__(self, ch):
            self.ch = ch

        @property
        def angle(self):
            return None

        @angle.setter
        def angle(self, value):
            writes.append((self.ch, value))

    kit = capture_session.FakeKit()
    kit.servo = [Servo(i) for i in range(16)]
    hw = capture_session.HardwareController(kit=kit, gpio=capture_session.FakeGPIO(), sleep=lambda s: None)
    rig = capture_session.Rig(hw, MotionProfile(servo_release_after_move=True), steps_per_mm=80.0,
                              max_travel_mm=120.0, max_servo_offset_deg=15.0, dry_run=True)
    writes.clear()
    rig.settle(0.0)
    assert writes == [(hw.servo_config.pan_channel, None)]
    assert all(ch != hw.servo_config.tilt_channel or v is not None for ch, v in writes)
