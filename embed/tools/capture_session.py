#!/usr/bin/env python3
"""CITA capture sessions on the Raspberry Pi (no server needed).

Drives ``HardwareController`` and a manual-exposure camera directly and writes
``<out>/<exp>/<session_id>/`` = original images + ``manifest.json``
(schema ``cita.session.v1``). Nothing is ever overwritten.

Sub-commands
------------
burst   E0.2/E0.5  static, N frames
settle  E0.4 t_s   move one axis, then capture continuously with timestamps
repeat  E0.4       go-and-return around the start point, capture after settle
grid    E1         visit a grid of offsets around golden (seeded order)
stereo  M2 golden  left, slider X +baseline, right, back

Safety: ``--dry-run`` uses fake servo/GPIO/camera drivers (synthetic grey
images). Without ``--dry-run`` you must also pass ``--i-know-hardware-moves``.
All servo/slider limits of HardwareController and the travel limits below are
enforced. Copy sessions to the server with e.g.::

    rsync -av --ignore-existing data/sessions/ user@server:artifact/server/data/sessions/
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
AGENT = HERE.parent / "device_agent"
sys.path.insert(0, str(AGENT))

from adapters.hardware_controller import HardwareController, SliderConfig  # noqa: E402
from adapters.motion_profile import MotionProfile  # noqa: E402

SCHEMA = "cita.session.v1"
STEPS_PER_MM = 800.0


# ---------------------------------------------------------------------------
# Fake drivers (dry-run, tests)
# ---------------------------------------------------------------------------

class _FakeServo:
    def __init__(self) -> None:
        self.angle: float | None = None


class FakeKit:
    def __init__(self) -> None:
        self.servo = [_FakeServo() for _ in range(16)]


class FakeGPIO:
    BCM, OUT, HIGH, LOW = "BCM", "OUT", 1, 0

    def __init__(self) -> None:
        self.pulses = 0

    def setmode(self, mode: Any) -> None:
        pass

    def setup(self, pins: Any, mode: Any) -> None:
        pass

    def output(self, pin: int, value: int) -> None:
        if value == self.HIGH:
            self.pulses += 1

    def cleanup(self) -> None:
        pass


@dataclass
class Outcome:
    image_path: Path
    metadata: dict


class FakeCamera:
    """Synthetic grey images whose content shifts with the (fake) rig state,
    so the E0 analysis tools produce meaningful numbers on dry-run data."""

    def __init__(self, output_dir: Path, hw: HardwareController, *, width: int = 640, height: int = 360,
                 seed: int = 0, image_format: str = "png", noise: float = 2.0) -> None:
        import numpy as np
        self.np = np
        self.output_dir = Path(output_dir)
        self.hw = hw
        self.width, self.height, self.noise = width, height, noise
        self.image_format = image_format
        rng = np.random.default_rng(seed)
        base = rng.random((height // 8 + 4, width // 8 + 4)) * 255
        self.texture = np.kron(base, np.ones((16, 16)))  # 2x oversized, blocky texture
        self.rng = rng
        self.t_last_move = time.monotonic()

    def __enter__(self) -> "FakeCamera":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def requested_controls(self) -> dict:
        return {"fake": True}

    def capture(self, basename: str) -> Outcome:
        np = self.np
        px_per_mm, px_per_deg = 2.0, 20.0
        dx = (self.hw.current_x_steps / STEPS_PER_MM) * px_per_mm + (self.hw.current_yaw - self.hw.HOME_YAW) * px_per_deg
        dy = (self.hw.current_z_steps / STEPS_PER_MM) * px_per_mm + (self.hw.current_pitch - self.hw.HOME_PITCH) * px_per_deg
        # Decaying vibration after a move (E0.4 settle analysis has something to find).
        age = time.monotonic() - self.t_last_move
        wobble = 3.0 * np.exp(-age / 0.3) * np.sin(age * 40.0)
        ox = int(round(self.width / 2 + dx + wobble))
        oy = int(round(self.height / 2 + dy))
        H, W = self.texture.shape
        ox = min(max(ox, 0), W - self.width)
        oy = min(max(oy, 0), H - self.height)
        img = self.texture[oy:oy + self.height, ox:ox + self.width]
        img = np.clip(img + self.rng.normal(0, self.noise, img.shape), 0, 255).astype(np.uint8)
        path = self.output_dir / f"{basename}.{self.image_format}"
        if path.exists():
            raise FileExistsError(path)
        _write_gray(path, img)
        return Outcome(path, {"fake": True, "SensorTimestamp": time.monotonic_ns()})


def _write_gray(path: Path, img) -> None:
    try:
        import cv2
        if not cv2.imwrite(str(path), img):
            raise OSError(f"cv2.imwrite failed for {path}")
    except ImportError:  # pragma: no cover - Pi without OpenCV: binary PGM
        path = path.with_suffix(".pgm")
        with open(path, "wb") as f:
            f.write(f"P5 {img.shape[1]} {img.shape[0]} 255\n".encode("ascii"))
            f.write(img.tobytes())


# ---------------------------------------------------------------------------
# Rig driver: absolute targets relative to the session start ("golden")
# ---------------------------------------------------------------------------

class Rig:
    def __init__(self, hw: HardwareController, profile: MotionProfile, *, steps_per_mm: float,
                 max_travel_mm: float, max_servo_offset_deg: float, dry_run: bool) -> None:
        self.hw, self.profile = hw, profile
        self.steps_per_mm = steps_per_mm
        self.max_travel_steps = int(max_travel_mm * steps_per_mm)
        self.max_servo_offset_deg = max_servo_offset_deg
        self.dry_run = dry_run
        self.home_x, self.home_z = hw.current_x_steps, hw.current_z_steps
        self.home_yaw, self.home_pitch = float(hw.HOME_YAW), float(hw.HOME_PITCH)
        self.on_move = None

    def _slider(self, axis: str, target_steps: int) -> None:
        current = self.hw.current_x_steps if axis == "x" else self.hw.current_z_steps
        delta = target_steps - current
        if delta == 0:
            return
        direction = 1 if delta > 0 else -1
        move = self.hw.move_slider_x if axis == "x" else self.hw.move_slider_z
        backlash = self.profile.backlash_steps
        if self.profile.unidirectional and backlash > 0 and direction * self.profile.approach_sign < 0:
            move(abs(delta) + backlash, direction)
            move(backlash, -direction)
        else:
            move(abs(delta), direction)

    def goto(self, *, x_mm: float = 0.0, z_mm: float = 0.0, pan_deg: float = 0.0, tilt_deg: float = 0.0) -> dict:
        for name, v, lim in (("x_mm", x_mm, self.max_travel_steps / self.steps_per_mm),
                             ("z_mm", z_mm, self.max_travel_steps / self.steps_per_mm),
                             ("pan_deg", pan_deg, self.max_servo_offset_deg),
                             ("tilt_deg", tilt_deg, self.max_servo_offset_deg)):
            if abs(v) > lim:
                raise ValueError(f"{name}={v} exceeds safety limit {lim}")
        self._slider("x", self.home_x + int(round(x_mm * self.steps_per_mm)))
        self._slider("z", self.home_z + int(round(z_mm * self.steps_per_mm)))
        waypoints = self.hw.approach_pan_tilt(self.home_yaw + pan_deg, self.home_pitch + tilt_deg, self.profile)
        if self.on_move:
            self.on_move()
        return {"servo_waypoints": [list(w) for w in waypoints]}

    def settle(self, seconds: float | None = None) -> float:
        s = self.profile.settle_s if seconds is None else seconds
        if s > 0:
            time.sleep(s)
        if self.profile.servo_release_after_move:
            self.hw.release_servos()
        return s

    def odometry(self) -> dict:
        return {"x_steps": int(self.hw.current_x_steps - self.home_x),
                "z_steps": int(self.hw.current_z_steps - self.home_z),
                "yaw_deg": float(self.hw.current_yaw), "pitch_deg": float(self.hw.current_pitch)}


# ---------------------------------------------------------------------------
# Session writer
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(HERE), capture_output=True,
                             text=True, timeout=5, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


class SessionWriter:
    def __init__(self, root: Path, exp: str, session_id: str, header: dict) -> None:
        self.dir = Path(root) / exp / session_id
        if self.dir.exists():
            raise FileExistsError(f"Session directory already exists: {self.dir}")
        self.dir.mkdir(parents=True)
        self.manifest = {"schema": SCHEMA, "session_id": session_id, "experiment": exp, **header,
                         "golden": {"frame_ids": [], "stereo_right_frame_id": None}, "frames": []}
        self.index = 0
        self.first_frame_hook = None
        self.flush()

    def flush(self) -> None:
        tmp = self.dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(self.manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.dir / "manifest.json")

    def add(self, camera, *, pose_label: str, rep: int, rig: Rig, extra: dict, role: str = "grid") -> dict:
        frame_id = f"f{self.index:05d}_{pose_label}_{rep}"
        t0 = _now()
        outcome = camera.capture(frame_id)
        if self.index == 0 and self.first_frame_hook is not None:
            self.first_frame_hook(outcome.image_path)
        frame = {"frame_id": frame_id, "file": outcome.image_path.name, "sha256": _sha256(outcome.image_path),
                 "role": role, "rep": rep, "t_capture": t0, "t_mono_s": time.monotonic(),
                 "odometry": rig.odometry(), "camera_metadata": outcome.metadata, **extra}
        self.manifest["frames"].append(frame)
        self.index += 1
        if role == "golden":
            self.manifest["golden"]["frame_ids"].append(frame_id)
        if role == "stereo_right":
            self.manifest["golden"]["stereo_right_frame_id"] = frame_id
        self.flush()
        return frame


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _floats(text: str) -> list[float]:
    return [float(v) for v in text.split(",") if v.strip() != ""]


def cmd_burst(args, rig: Rig, cam, writer: SessionWriter) -> None:
    rig.settle()
    for rep in range(args.n):
        writer.add(cam, pose_label="b", rep=rep, rig=rig, role="golden" if rep == 0 and args.first_is_golden else "burst",
                   extra={"pose_index": 0, "target": {"x_mm": 0, "z_mm": 0, "pan_deg": 0, "tilt_deg": 0},
                          "settle_s": rig.profile.settle_s})
        if args.interval_s > 0:
            time.sleep(args.interval_s)


def _axis_target(axis: str, value: float) -> dict:
    key = {"x": "x_mm", "z": "z_mm", "pan": "pan_deg", "tilt": "tilt_deg"}[axis]
    target = {"x_mm": 0.0, "z_mm": 0.0, "pan_deg": 0.0, "tilt_deg": 0.0}
    target[key] = value
    return target


def cmd_settle(args, rig: Rig, cam, writer: SessionWriter) -> None:
    for trial in range(args.trials):
        sign = 1 if trial % 2 == 0 else -1
        target = _axis_target(args.axis, sign * args.delta if args.alternate else args.delta)
        if not args.alternate and trial > 0:
            rig.goto()                      # back to start, untimed
            rig.settle(args.rest_s)
        writer.add(cam, pose_label=f"s{trial:03d}pre", rep=0, rig=rig, role="settle_ref",
                   extra={"pose_index": trial, "trial": trial, "t_since_move_s": None, "target": None})
        t_move = time.monotonic()
        rig.goto(**target)
        t_done = time.monotonic()
        for k in range(args.n_frames):
            frame = writer.add(cam, pose_label=f"s{trial:03d}", rep=k, rig=rig, role="settle",
                               extra={"pose_index": trial, "trial": trial, "target": target,
                                      "t_move_start_mono_s": t_move, "t_move_end_mono_s": t_done})
            frame["t_since_move_s"] = frame["t_mono_s"] - t_done
            writer.flush()
            if args.interval_s > 0:
                time.sleep(args.interval_s)
        rig.settle(args.rest_s)


def cmd_repeat(args, rig: Rig, cam, writer: SessionWriter) -> None:
    levels = _floats(args.levels)
    sides = (1, -1) if args.both_sides else (1,)
    pose_index = 0
    rig.goto()
    rig.settle()
    writer.add(cam, pose_label="g", rep=0, rig=rig, role="golden",
               extra={"pose_index": -1, "target": _axis_target(args.axis, 0.0), "settle_s": rig.profile.settle_s})
    for level in levels:
        for side in sides:
            for rep in range(args.reps):
                excursion = _axis_target(args.axis, side * level)
                rig.goto(**excursion)
                rig.settle(min(0.3, rig.profile.settle_s))
                rig.goto()                  # return to the start point
                rig.settle()
                writer.add(cam, pose_label=f"r{pose_index:03d}", rep=rep, rig=rig, role="repeat",
                           extra={"pose_index": pose_index, "axis": args.axis, "level": level,
                                  "approach_side": side, "target": _axis_target(args.axis, 0.0),
                                  "excursion": excursion, "settle_s": rig.profile.settle_s})
            pose_index += 1


def cmd_grid(args, rig: Rig, cam, writer: SessionWriter) -> None:
    combos = list(itertools.product(_floats(args.x), _floats(args.z), _floats(args.pan), _floats(args.tilt)))
    order = list(range(len(combos)))
    random.Random(args.seed).shuffle(order)
    writer.manifest["grid"] = {"x_mm": _floats(args.x), "z_mm": _floats(args.z), "pan_deg": _floats(args.pan),
                               "tilt_deg": _floats(args.tilt), "order": order, "seed": args.seed}
    rig.goto()
    rig.settle()
    for rep in range(args.golden_frames):
        writer.add(cam, pose_label="g", rep=rep, rig=rig, role="golden",
                   extra={"pose_index": -1, "target": _axis_target("x", 0.0), "settle_s": rig.profile.settle_s})
    if args.with_stereo:
        # Golden stereo pair in the same session (needed by replay_pose / M2).
        rig.goto(x_mm=args.baseline_mm)
        rig.settle()
        writer.add(cam, pose_label="g", rep=args.golden_frames, rig=rig, role="stereo_right",
                   extra={"pose_index": -1, "target": _axis_target("x", args.baseline_mm),
                          "commanded_baseline_mm": args.baseline_mm, "settle_s": rig.profile.settle_s})
        rig.goto()
        rig.settle()
    for pose_index in order:
        x, z, pan, tilt = combos[pose_index]
        target = {"x_mm": x, "z_mm": z, "pan_deg": pan, "tilt_deg": tilt}
        info = rig.goto(**target)
        rig.settle()
        for rep in range(args.frames):
            writer.add(cam, pose_label=f"p{pose_index:03d}", rep=rep, rig=rig, role="grid",
                       extra={"pose_index": pose_index, "target": target, "settle_s": rig.profile.settle_s,
                              "offset_mm": None, "offset_deg": None, "stage": None, **info})
    if args.return_home:
        rig.goto()


def cmd_stereo(args, rig: Rig, cam, writer: SessionWriter) -> None:
    rig.goto()
    rig.settle()
    writer.add(cam, pose_label="g", rep=0, rig=rig, role="golden",
               extra={"pose_index": -1, "target": _axis_target("x", 0.0), "settle_s": rig.profile.settle_s})
    rig.goto(x_mm=args.baseline_mm)
    rig.settle()
    writer.add(cam, pose_label="g", rep=1, rig=rig, role="stereo_right",
               extra={"pose_index": -1, "target": _axis_target("x", args.baseline_mm),
                      "commanded_baseline_mm": args.baseline_mm, "settle_s": rig.profile.settle_s})
    rig.goto()


COMMANDS = {"burst": cmd_burst, "settle": cmd_settle, "repeat": cmd_repeat, "grid": cmd_grid, "stereo": cmd_stereo}


# ---------------------------------------------------------------------------
# Pre-flight (R8): motion envelope, duration, disk space, explicit confirmation
# ---------------------------------------------------------------------------

FRAME_MB = {"png": 25.0, "jpg": 5.0}      # 4K estimates; re-measured on the first frame
CAPTURE_S = 1.5                          # rough per-frame capture + save time on the Pi
SERVO_MOVE_S = 0.5


def planned_targets(args) -> tuple[list[dict], int]:
    """Targets in visiting order (relative to the start position) and frame count."""
    def t(x=0.0, z=0.0, pan=0.0, tilt=0.0):
        return {"x_mm": x, "z_mm": z, "pan_deg": pan, "tilt_deg": tilt}
    cmd = args.command
    if cmd == "burst":
        return [t()], args.n
    if cmd == "settle":
        out = []
        for trial in range(args.trials):
            sign = 1 if trial % 2 == 0 else -1
            out.append(_axis_target(args.axis, sign * args.delta if args.alternate else args.delta))
            if not args.alternate:
                out.append(t())
        return out, args.trials * (args.n_frames + 1)
    if cmd == "repeat":
        out = [t()]
        sides = (1, -1) if args.both_sides else (1,)
        for level in _floats(args.levels):
            for side in sides:
                for _ in range(args.reps):
                    out += [_axis_target(args.axis, side * level), t()]
        return out, 1 + len(_floats(args.levels)) * len(sides) * args.reps
    if cmd == "grid":
        combos = list(itertools.product(_floats(args.x), _floats(args.z), _floats(args.pan), _floats(args.tilt)))
        out = [t()]
        if args.with_stereo:
            out += [t(x=args.baseline_mm), t()]
        out += [t(*c) for c in combos]
        if args.return_home:
            out.append(t())
        return out, args.golden_frames + (1 if args.with_stereo else 0) + len(combos) * args.frames
    if cmd == "stereo":
        return [t(), t(x=args.baseline_mm), t()], 2
    return [], 0


def plan_session(args, profile: MotionProfile) -> dict:
    """Motion envelope (incl. backlash overshoot and stereo baseline), duration
    and disk estimate, BEFORE any hardware is touched."""
    targets, n_frames = planned_targets(args)
    pulse_s = args.pulse_delay_us / 1e6
    b_mm = profile.backlash_steps / args.steps_per_mm if profile.unidirectional else 0.0
    pos = {"x_mm": 0.0, "z_mm": 0.0}
    extent = {k: [0.0, 0.0] for k in ("x_mm", "z_mm", "pan_deg", "tilt_deg")}
    steps = 0.0
    for tgt in targets:
        for axis in ("x_mm", "z_mm"):
            delta = tgt[axis] - pos[axis]
            if delta == 0:
                continue
            direction = 1 if delta > 0 else -1
            far = tgt[axis]
            travel = abs(delta)
            if b_mm > 0 and direction * profile.approach_sign < 0:
                far = tgt[axis] - profile.approach_sign * b_mm
                travel += 2 * b_mm
            extent[axis] = [min(extent[axis][0], far), max(extent[axis][1], far)]
            steps += travel * args.steps_per_mm
            pos[axis] = tgt[axis]
        for axis in ("pan_deg", "tilt_deg"):
            v = tgt[axis]
            if profile.unidirectional and profile.backlash_deg > 0:
                v_lo = v - profile.approach_sign * profile.backlash_deg
                extent[axis] = [min(extent[axis][0], v, v_lo), max(extent[axis][1], v, v_lo)]
            else:
                extent[axis] = [min(extent[axis][0], v), max(extent[axis][1], v)]
    settle = profile.settle_s * len(targets)
    interval = getattr(args, "interval_s", 0.0) * n_frames + getattr(args, "rest_s", 0.0) * getattr(args, "trials", 0)
    duration = steps * 2 * pulse_s + SERVO_MOVE_S * len(targets) + settle + CAPTURE_S * n_frames + interval
    frame_mb = 0.3 if args.dry_run else FRAME_MB[args.format] * (args.width * args.height) / (3840 * 2160)
    return {"n_targets": len(targets), "n_frames": n_frames, "envelope": extent,
            "slider_steps_total": int(round(steps)), "pulse_delay_us": args.pulse_delay_us,
            "estimated_duration_s": round(duration, 1), "frame_mb_estimate": round(frame_mb, 2),
            "estimated_mb": round(frame_mb * n_frames, 1)}


def check_disk(directory: Path, needed_mb: float, factor: float = 1.2) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    free_mb = shutil.disk_usage(directory).free / 1e6
    ok = free_mb >= factor * needed_mb
    return {"free_mb": round(free_mb, 1), "needed_mb": round(needed_mb, 1), "factor": factor, "ok": ok}


def print_plan(plan: dict, disk: dict) -> None:
    env = plan["envelope"]
    print("[PLAN] Vung di chuyen so voi vi tri luc khoi dong (gom overshoot bu ro va baseline stereo):")
    print(f"[PLAN]   X {env['x_mm'][0]:+.1f} .. {env['x_mm'][1]:+.1f} mm | Z {env['z_mm'][0]:+.1f} .. {env['z_mm'][1]:+.1f} mm"
          f" | pan {env['pan_deg'][0]:+.2f} .. {env['pan_deg'][1]:+.2f} deg | tilt {env['tilt_deg'][0]:+.2f} .. "
          f"{env['tilt_deg'][1]:+.2f} deg")
    print(f"[PLAN]   {plan['n_targets']} vi tri, {plan['n_frames']} khung, {plan['slider_steps_total']} buoc slider"
          f" (pulse {plan['pulse_delay_us']} us) ~ {plan['estimated_duration_s'] / 60:.1f} phut")
    print(f"[PLAN]   Dung luong uoc tinh {plan['estimated_mb']:.0f} MB (trong: {disk['free_mb']:.0f} MB, can "
          f">= {disk['factor']} x)")


def confirm_envelope(args) -> bool:
    if args.confirm_envelope:
        return True
    try:
        answer = input("Go 'yes' de dong co phan cung voi vung di chuyen tren: ")
    except EOFError:
        return False
    return answer.strip().lower() == "yes"


# ---------------------------------------------------------------------------
# AWB measurement (R8)
# ---------------------------------------------------------------------------

def load_awb(args) -> tuple[list[float], dict]:
    """ColourGains from --awb-from (measured) or --awb; defaults only in dry-run."""
    if args.awb_from:
        data = json.loads(Path(args.awb_from).read_text(encoding="utf-8"))
        gains = [float(v) for v in data["colour_gains"]]
        return gains, {"source": "awb_from", "file": str(args.awb_from), **{k: data.get(k) for k in
                                                                             ("lux", "colour_temperature", "t_measured")}}
    if args.awb:
        return [float(v) for v in args.awb.split(",")], {"source": "cli"}
    if args.dry_run:
        return [1.2, 1.5], {"source": "dry_run_default"}
    raise SystemExit("Real capture needs --awb r,b or --awb-from awb_gains.json (run the 'awb' sub-command "
                     "under the session lighting first)")


def cmd_awb(args) -> int:
    if args.dry_run:
        meas = {"colour_gains": [1.85, 1.62], "lux": 320.0, "colour_temperature": 4100.0, "fake": True}
    else:  # pragma: no cover - real camera only
        from adapters.camera_manager import measure_awb
        meas = measure_awb(seconds=args.seconds, lens_position=args.lens_position)
    out = Path(args.awb_out or (Path(args.out) / "awb_gains.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema": "cita.awb.v1", **meas, "auto_seconds": args.seconds, "lighting": args.lighting,
              "t_measured": _now(), "tool_version": _git_commit()}
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"AWB: ColourGains={meas['colour_gains']} Lux={meas.get('lux')} CT={meas.get('colour_temperature')} -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(AGENT / "data" / "sessions"))
    p.add_argument("--exp", required=True, help="E0.2, E0.4, E1, E3 ...")
    p.add_argument("--session-id", default=None)
    p.add_argument("--painting-id", default="P000")
    p.add_argument("--lighting", default="L0")
    p.add_argument("--distance-cm", type=float, default=None)
    p.add_argument("--split", default="na", choices=("na", "train", "val", "test"))
    p.add_argument("--dry-run", action="store_true", help="fake servo/GPIO/camera (default for safety)")
    p.add_argument("--i-know-hardware-moves", action="store_true",
                   help="required (together with no --dry-run) to drive the real rig")
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--steps-per-mm", type=float, default=STEPS_PER_MM)
    p.add_argument("--max-travel-mm", type=float, default=120.0)
    p.add_argument("--max-servo-offset-deg", type=float, default=15.0)
    # Motion profile (I2); unspecified = legacy device behaviour + env vars.
    p.add_argument("--settle-s", type=float, default=None)
    p.add_argument("--float-angles", action="store_true")
    p.add_argument("--unidirectional", action="store_true")
    p.add_argument("--approach-sign", type=int, choices=(-1, 1), default=None)
    p.add_argument("--backlash-deg", type=float, default=None)
    p.add_argument("--backlash-steps", type=int, default=None)
    p.add_argument("--servo-release", action="store_true")
    # Camera (manual controls; defaults = calibrated focus).
    p.add_argument("--lens-position", type=float, default=1.5)
    p.add_argument("--exposure-us", type=int, default=20000)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--awb", default=None, help="ColourGains red,blue (dry-run default 1.2,1.5)")
    p.add_argument("--awb-from", default=None, help="awb_gains.json written by the 'awb' sub-command")
    p.add_argument("--confirm-envelope", action="store_true",
                   help="skip the interactive 'yes' after the motion envelope is printed")
    p.add_argument("--pulse-delay-us", type=int, default=500,
                   help="stepper pulse half-period (us); 500 = old behaviour, faster values are tested in E0.4")
    p.add_argument("--format", default="png", choices=("png", "jpg"))
    p.add_argument("--width", type=int, default=3840)
    p.add_argument("--height", type=int, default=2160)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("burst")
    b.add_argument("--n", type=int, default=60)
    b.add_argument("--interval-s", type=float, default=0.0)
    b.add_argument("--first-is-golden", action="store_true")
    s = sub.add_parser("settle")
    s.add_argument("--axis", choices=("x", "z", "pan", "tilt"), required=True)
    s.add_argument("--delta", type=float, required=True, help="mm (x/z) or degrees (pan/tilt)")
    s.add_argument("--n-frames", type=int, default=20)
    s.add_argument("--interval-s", type=float, default=0.0)
    s.add_argument("--trials", type=int, default=10)
    s.add_argument("--alternate", action="store_true", help="alternate +delta / -delta around start")
    s.add_argument("--rest-s", type=float, default=3.0)
    r = sub.add_parser("repeat")
    r.add_argument("--axis", choices=("x", "z", "pan", "tilt"), required=True)
    r.add_argument("--levels", default="10,50", help="excursions, mm (x/z) or degrees (pan/tilt)")
    r.add_argument("--reps", type=int, default=10)
    r.add_argument("--both-sides", action="store_true")
    g = sub.add_parser("grid")
    g.add_argument("--x", default="-40,-20,-10,-5,0,5,10,20,40")
    g.add_argument("--z", default="-20,0,20")
    g.add_argument("--pan", default="-4,0,4")
    g.add_argument("--tilt", default="0")
    g.add_argument("--frames", type=int, default=3)
    g.add_argument("--golden-frames", type=int, default=3)
    g.add_argument("--return-home", action="store_true")
    g.add_argument("--with-stereo", action="store_true", help="also capture the golden stereo-right frame")
    g.add_argument("--baseline-mm", type=float, default=100.0)
    st = sub.add_parser("stereo")
    st.add_argument("--baseline-mm", type=float, default=100.0)
    a = sub.add_parser("awb", help="measure ColourGains once under the session lighting (auto AWB)")
    a.add_argument("--seconds", type=float, default=3.0)
    a.add_argument("--awb-out", default=None, help="default: <out>/awb_gains.json")
    return p


def motion_profile_from_args(args) -> MotionProfile:
    data = {"settle_s": args.settle_s, "approach_sign": args.approach_sign,
            "backlash_deg": args.backlash_deg, "backlash_steps": args.backlash_steps}
    if args.float_angles:
        data["float_angles"] = True
    if args.unidirectional:
        data["unidirectional"] = True
    if args.servo_release:
        data["servo_release_after_move"] = True
    return MotionProfile.resolve({k: v for k, v in data.items() if v is not None})


_LIST_FLAGS = ("--x", "--z", "--pan", "--tilt", "--levels", "--delta")


def _join_negative_values(argv: list[str]) -> list[str]:
    """Allow ``--x -40,-20,0`` (argparse would read -40,... as an option)."""
    out, i = [], 0
    while i < len(argv):
        tok = argv[i]
        if tok in _LIST_FLAGS and i + 1 < len(argv) and argv[i + 1][:1] == "-" and argv[i + 1][1:2].isdigit():
            out.append(f"{tok}={argv[i + 1]}")
            i += 2
            continue
        out.append(tok)
        i += 1
    return out


def main(argv=None) -> int:
    argv = _join_negative_values(list(sys.argv[1:] if argv is None else argv))
    args = build_parser().parse_args(argv)
    if args.command == "awb":
        return cmd_awb(args)
    if not args.dry_run and not args.i_know_hardware_moves:
        print("Refusing to move real hardware without --i-know-hardware-moves (use --dry-run to test).",
              file=sys.stderr)
        return 2
    try:
        colour_gains, awb_info = load_awb(args)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 4
    profile = motion_profile_from_args(args)
    plan = plan_session(args, profile)
    disk = check_disk(Path(args.out), plan["estimated_mb"])
    print_plan(plan, disk)
    if not disk["ok"]:
        print(f"Not enough free disk space: {disk['free_mb']:.0f} MB < {disk['factor']} x {disk['needed_mb']:.0f} MB",
              file=sys.stderr)
        return 5
    if not args.dry_run and not confirm_envelope(args):
        print("Cancelled: motion envelope not confirmed (type 'yes' or pass --confirm-envelope).", file=sys.stderr)
        return 3
    slider_cfg = SliderConfig(pulse_delay=args.pulse_delay_us / 1e6)
    if args.dry_run:
        hw = HardwareController(slider_config=slider_cfg, kit=FakeKit(), gpio=FakeGPIO(), sleep=lambda s: None)
    else:  # pragma: no cover - real rig only
        hw = HardwareController(slider_config=slider_cfg)
    rig = Rig(hw, profile, steps_per_mm=args.steps_per_mm, max_travel_mm=args.max_travel_mm,
              max_servo_offset_deg=args.max_servo_offset_deg, dry_run=args.dry_run)
    session_id = args.session_id or (
        f"{args.exp}_{args.painting_id}_{args.lighting}_{datetime.now().strftime('%Y%m%dT%H%M%S')}")
    header = {"split": args.split, "painting_id": args.painting_id, "lighting": args.lighting,
              "distance_cm": args.distance_cm, "device_id": os.environ.get("DEVICE_ID", "") or "unknown",
              "tool_version": _git_commit(), "tool": "embed/tools/capture_session.py",
              "command": args.command, "argv": argv,
              "dry_run": bool(args.dry_run), "seed": args.seed, "steps_per_mm": args.steps_per_mm,
              "pulse_delay_us": args.pulse_delay_us, "motion_profile": profile.to_dict(),
              "plan": plan, "disk_check": disk,
              "camera": {"width": args.width, "height": args.height, "lens_position": args.lens_position,
                         "exposure_time_us": args.exposure_us, "analogue_gain": args.gain,
                         "colour_gains": colour_gains, "awb": awb_info,
                         "calibration_file": f"camera_params_lens_{args.lens_position:.1f}.yaml",
                         "calibration_sha256": None},
              "t_start": _now()}
    writer = SessionWriter(Path(args.out), args.exp, session_id, header)

    def first_frame_disk_check(path: Path) -> None:
        measured_mb = path.stat().st_size / 1e6
        again = check_disk(writer.dir, measured_mb * max(plan["n_frames"] - 1, 0))
        writer.manifest["disk_check_first_frame"] = {**again, "frame_mb_measured": round(measured_mb, 3)}
        if not again["ok"]:
            raise RuntimeError(f"Not enough disk for the remaining frames ({again['free_mb']:.0f} MB free, "
                               f"{again['needed_mb']:.0f} MB needed x {again['factor']})")
    writer.first_frame_hook = first_frame_disk_check
    if args.dry_run:
        cam = FakeCamera(writer.dir, hw, seed=args.seed, image_format=args.format)
        rig.on_move = lambda: setattr(cam, "t_last_move", time.monotonic())
    else:  # pragma: no cover - real rig only
        from adapters.camera_manager import ManualCameraSession
        cam = ManualCameraSession(writer.dir, width=args.width, height=args.height,
                                  lens_position=args.lens_position, exposure_time_us=args.exposure_us,
                                  analogue_gain=args.gain, colour_gains=colour_gains,
                                  image_format=args.format)
    try:
        with cam:
            COMMANDS[args.command](args, rig, cam, writer)
    finally:
        writer.manifest["t_end"] = _now()
        writer.flush()
        if not args.dry_run:  # pragma: no cover
            hw.cleanup()
    print(f"Session written: {writer.dir} ({len(writer.manifest['frames'])} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
