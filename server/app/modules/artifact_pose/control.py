"""Alignment controllers (E2 arms C-A ... C-D).

Control variables q = [x_mm, z_mm, yaw_deg, pitch_deg]; conversion to stepper
steps / servo degrees happens only at the boundary (``ActuatorCommand.to_move_payload``).

* ``FixedMappingController`` (C-A, C-B): the cita-baseline logic of
  ``inspection_service`` verbatim: Diamond-frame translation correction x
  1000 x STEPS_PER_MM x SIGN, Euler pan/tilt x SIGN, alternating translation /
  rotation phases with smart-skip.
* ``JacobianController`` (C-C, C-D, improvement I3): measures J = de/dq by
  probing (chained forward or central differences), then
  dq = -lambda (J^T W J + mu I)^-1 J^T W e with e = [c / s_t ; theta / s_R],
  optional Broyden updates, per-axis step and travel clamps.

Controllers are pure logic and serialisable (``to_state``/``from_state``) so
the event-driven server can resume them between uploads.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

AXES = ("x_mm", "z_mm", "yaw_deg", "pitch_deg")


@dataclass
class ActuatorCommand:
    dq: np.ndarray                      # [x_mm, z_mm, yaw_deg, pitch_deg]
    kind: str = "control"               # control | probe | legacy
    phase: str | None = None            # translation | rotation | None (all axes)
    meta: dict = field(default_factory=dict)
    legacy_fields: dict | None = None   # exact legacy move fields (fixed mapping)

    def to_move_payload(self, steps_per_mm: float) -> dict:
        if self.legacy_fields is not None:
            return dict(self.legacy_fields)
        raw_x = float(self.dq[0]) * steps_per_mm
        raw_z = float(self.dq[1]) * steps_per_mm
        out = {"x_steps": abs(int(round(raw_x))), "x_dir": 1 if raw_x >= 0 else -1,
               "z_steps": abs(int(round(raw_z))), "z_dir": 1 if raw_z >= 0 else -1,
               "yaw_delta": float(self.dq[2]), "pitch_delta": float(self.dq[3])}
        if self.phase:
            out["alignment_phase"] = self.phase
        return out

    def to_json(self) -> dict:
        return {"dq": [float(v) for v in self.dq], "kind": self.kind, "phase": self.phase,
                "meta": self.meta, "legacy_fields": self.legacy_fields}


@dataclass
class Stop:
    success: bool
    reason: str


@dataclass
class ControlState:
    """What a controller sees at one iteration."""
    iteration: int
    c_mm: np.ndarray | None = None           # camera centre in golden camera frame
    theta_deg: np.ndarray | None = None      # rotvec(dR) in degrees
    sigma_c_mm: np.ndarray | None = None
    sigma_theta_deg: np.ndarray | None = None
    deviation: dict | None = None            # legacy deviation dict (fixed mapping)
    within_trans: bool = False
    within_rot: bool = False

    @classmethod
    def from_estimate(cls, iteration: int, est: Any, tol_t_mm: float, tol_R_deg: float) -> "ControlState":
        """From a PoseEstimate (or its JSON) or a legacy correction result."""
        get = (lambda k: getattr(est, k, None)) if not isinstance(est, dict) else est.get
        c = get("camera_center_in_golden_m")
        th = get("rotvec_deg")
        dev = get("deviation")
        if c is None and isinstance(dev, dict) and dev.get("T_current_camera_desired_camera") is not None:
            dT = np.asarray(dev["T_current_camera_desired_camera"], dtype=float)
            from .motion import camera_center_in_golden, rotvec
            c, th = camera_center_in_golden(dT), np.rad2deg(rotvec(dT[:3, :3]))
        c_mm = None if c is None else np.asarray(c, float) * 1000
        th = None if th is None else np.asarray(th, float)
        sc, st = get("sigma_center_mm"), get("sigma_rot_deg")
        return cls(iteration=iteration, c_mm=c_mm, theta_deg=th,
                   sigma_c_mm=None if sc is None else np.asarray(sc, float),
                   sigma_theta_deg=None if st is None else np.asarray(st, float), deviation=dev,
                   within_trans=c_mm is not None and float(np.linalg.norm(c_mm)) <= tol_t_mm,
                   within_rot=th is not None and float(np.linalg.norm(th)) <= tol_R_deg)


# ---------------------------------------------------------------------------
# C-A / C-B
# ---------------------------------------------------------------------------

class FixedMappingController:
    """Verbatim extraction of inspection_service (cita-baseline, lines 971-1070)."""

    name = "fixed_mapping"

    def __init__(self, *, steps_per_mm: float, sign_x: int, sign_z: int, sign_pan: int, sign_tilt: int) -> None:
        self.steps_per_mm = steps_per_mm
        self.sign_x, self.sign_z, self.sign_pan, self.sign_tilt = sign_x, sign_z, sign_pan, sign_tilt
        self.phase = 0

    def reset(self, context: dict | None = None) -> None:
        self.phase = 0

    def legacy_step(self, deviation: dict, phase: int) -> dict:
        """Returns motor_cmd, the move fields, the phase used and the next phase."""
        raw_move_x = float(deviation.get("delta_x", 0.0)) * 1000.0 * self.steps_per_mm * self.sign_x
        raw_move_z = float(deviation.get("delta_z", 0.0)) * 1000.0 * self.steps_per_mm * self.sign_z
        raw_pan = float(deviation.get("delta_pan", 0.0)) * self.sign_pan
        raw_tilt = float(deviation.get("delta_tilt", 0.0)) * self.sign_tilt
        motor_cmd = {"enabled": True, "move_x": raw_move_x, "move_z": raw_move_z,
                     "rotate_pan": raw_pan, "rotate_tilt": raw_tilt}
        within_trans = bool(deviation.get("within_trans_tolerance", False))
        within_rot = bool(deviation.get("within_rot_tolerance", False))
        skipped = None
        if phase == 0 and within_trans and not within_rot:
            phase, skipped = 1, "translation_within_tolerance"
        elif phase == 1 and within_rot and not within_trans:
            phase, skipped = 0, "rotation_within_tolerance"
        if phase == 0:
            fields = {"x_steps": abs(int(round(raw_move_x))), "x_dir": 1 if raw_move_x >= 0 else -1,
                      "z_steps": abs(int(round(raw_move_z))), "z_dir": 1 if raw_move_z >= 0 else -1,
                      "yaw_delta": 0.0, "pitch_delta": 0.0, "alignment_phase": "translation"}
        else:
            fields = {"x_steps": 0, "x_dir": 1, "z_steps": 0, "z_dir": 1, "yaw_delta": raw_pan,
                      "pitch_delta": raw_tilt, "alignment_phase": "rotation"}
        return {"motor_cmd": motor_cmd, "fields": fields, "phase_used": phase, "next_phase": 1 - phase,
                "smart_skip": skipped, "raw": [raw_move_x, raw_move_z, raw_pan, raw_tilt]}

    def step(self, state: ControlState) -> ActuatorCommand | Stop:
        dev = dict(state.deviation or {})
        if not dev:
            return Stop(False, "no_deviation")
        dev["within_trans_tolerance"] = state.within_trans
        dev["within_rot_tolerance"] = state.within_rot
        out = self.legacy_step(dev, self.phase)
        self.phase = out["next_phase"]
        f = out["fields"]
        dq = np.array([f["x_steps"] * f["x_dir"] / self.steps_per_mm, f["z_steps"] * f["z_dir"] / self.steps_per_mm,
                       f["yaw_delta"], f["pitch_delta"]], dtype=float)
        return ActuatorCommand(dq=dq, kind="legacy", phase=f["alignment_phase"], legacy_fields=f,
                               meta={"motor_cmd": out["motor_cmd"], "smart_skip": out["smart_skip"]})

    def to_state(self) -> dict:
        return {"type": self.name, "phase": self.phase}

    def load_state(self, data: dict) -> None:
        self.phase = int(data.get("phase", 0))


# ---------------------------------------------------------------------------
# C-C / C-D (I3)
# ---------------------------------------------------------------------------

class JacobianController:
    name = "jacobian"

    def __init__(self, cfg, *, store_path: str | Path | None = None, clock=time.time) -> None:
        self.cfg = cfg
        self.store_path = Path(store_path) if store_path else None
        self.clock = clock
        self.delta = np.array([cfg.probe.delta.x_mm, cfg.probe.delta.z_mm,
                               cfg.probe.delta.yaw_deg, cfg.probe.delta.pitch_deg], dtype=float)
        self.max_step = np.array([cfg.max_step.x_mm, cfg.max_step.z_mm, cfg.max_step.yaw_deg,
                                  cfg.max_step.pitch_deg], dtype=float)
        self.travel = np.array([cfg.travel.x_mm, cfg.travel.z_mm, cfg.travel.yaw_deg,
                                cfg.travel.pitch_deg], dtype=float)
        self.scale = np.array([cfg.scale.s_t_mm] * 3 + [cfg.scale.s_R_deg] * 3, dtype=float)
        self.reset()

    # -- state -----------------------------------------------------------------
    def reset(self, context: dict | None = None) -> None:
        self.mode = "probe"
        self.probe_step = 0
        self.J = np.zeros((6, 4))
        self.J_probe = None
        self.e_prev = None
        self.e_base = None
        self.e_plus = None
        self.dq_prev = None
        self.q = np.zeros(4)               # commanded q relative to the loop start
        self.worse_count = 0
        self.err_prev = None
        self.J_source = None
        self.events: list[dict] = []
        if context and context.get("reuse_jacobian", True):
            self._try_load()

    def to_state(self) -> dict:
        arr = lambda a: None if a is None else np.asarray(a).tolist()  # noqa: E731
        return {"type": self.name, "mode": self.mode, "probe_step": self.probe_step, "J": arr(self.J),
                "J_probe": arr(self.J_probe), "e_prev": arr(self.e_prev), "e_base": arr(self.e_base),
                "e_plus": arr(self.e_plus), "dq_prev": arr(self.dq_prev), "q": arr(self.q),
                "worse_count": self.worse_count, "err_prev": self.err_prev, "J_source": self.J_source}

    def load_state(self, data: dict) -> None:
        arr = lambda k: None if data.get(k) is None else np.asarray(data[k], dtype=float)  # noqa: E731
        self.mode, self.probe_step = data["mode"], int(data["probe_step"])
        self.J = arr("J") if data.get("J") is not None else np.zeros((6, 4))
        self.J_probe, self.e_prev, self.e_base, self.e_plus = arr("J_probe"), arr("e_prev"), arr("e_base"), arr("e_plus")
        self.dq_prev = arr("dq_prev")
        self.q = arr("q") if data.get("q") is not None else np.zeros(4)
        self.worse_count, self.err_prev, self.J_source = int(data.get("worse_count", 0)), data.get("err_prev"), data.get("J_source")

    # -- persistence -------------------------------------------------------------
    def _try_load(self) -> None:
        if self.store_path is None or not self.store_path.exists():
            return
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            J = np.asarray(data["J"], dtype=float)
            age = self.clock() - float(data["t_measured"])
            cond = self.condition(J)
            if J.shape == (6, 4) and age <= self.cfg.probe.ttl_s and cond <= self.cfg.probe.max_condition:
                self.J, self.J_probe, self.mode, self.J_source = J, J.copy(), "control", "stored"
                self.events.append({"event": "jacobian_reused", "age_s": age, "condition": cond})
        except (OSError, ValueError, KeyError) as exc:
            self.events.append({"event": "jacobian_load_failed", "error": str(exc)})

    def _save(self, extra: dict) -> None:
        if self.store_path is None:
            return
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": "cita.jacobian.v1", "J": self.J.tolist(), "condition": self.condition(self.J),
                   "t_measured": self.clock(), "delta": self.delta.tolist(), "scheme": self.cfg.probe.scheme,
                   "scale": self.scale.tolist(), "axes_q": list(AXES),
                   "rows_e": ["cx/s_t", "cy/s_t", "cz/s_t", "thx/s_R", "thy/s_R", "thz/s_R"], **extra}
        tmp = self.store_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        tmp.replace(self.store_path)

    @staticmethod
    def condition(J: np.ndarray) -> float:
        s = np.linalg.svd(np.asarray(J, float), compute_uv=False)
        return float(s[0] / max(s[-1], 1e-12)) if s[0] > 0 else float("inf")

    # -- error vector ------------------------------------------------------------
    def error(self, state: ControlState) -> np.ndarray:
        return np.r_[state.c_mm, state.theta_deg] / self.scale

    def weights(self, state: ControlState) -> np.ndarray:
        if state.sigma_c_mm is None or state.sigma_theta_deg is None:
            return np.ones(6)
        s = np.r_[state.sigma_c_mm, state.sigma_theta_deg] / self.scale
        w = 1.0 / np.maximum(s, 1e-6) ** 2
        return w / np.mean(w)

    # -- probing -----------------------------------------------------------------
    def _probe_command(self, dq: np.ndarray, note: str) -> ActuatorCommand:
        self.q = self.q + dq
        self.dq_prev = dq
        return ActuatorCommand(dq=dq, kind="probe", meta={"probe_step": self.probe_step, "note": note})

    def _probe(self, e: np.ndarray, state: ControlState) -> ActuatorCommand | None:
        k = self.probe_step
        if self.cfg.probe.scheme == "chained_forward":
            if k > 0:
                axis = k - 1
                self.J[:, axis] = (e - self.e_prev) / self.delta[axis]
            if k < 4:
                dq = np.zeros(4)
                dq[k] = self.delta[k]
                self.e_prev = e
                self.probe_step += 1
                return self._probe_command(dq, f"+delta axis {AXES[k]}")
        else:  # central: +d_k (measure e+), -2 d_k (measure e-), next axis folds the return
            axis, sub = divmod(k, 2)
            if sub == 1:                       # measured e+ of `axis`
                self.e_plus = e
                dq = np.zeros(4)
                dq[axis] = -2 * self.delta[axis]
                self.probe_step += 1
                return self._probe_command(dq, f"-2 delta axis {AXES[axis]}")
            if k > 0:                          # measured e- of previous axis
                prev = axis - 1
                self.J[:, prev] = (self.e_plus - e) / (2 * self.delta[prev])
            if axis < 4:
                dq = np.zeros(4)
                dq[axis] = self.delta[axis]
                if axis > 0:
                    dq[axis - 1] += self.delta[axis - 1]   # return previous axis to its start
                self.probe_step += 1
                return self._probe_command(dq, f"+delta axis {AXES[axis]}")
        # Probing finished.
        cond = self.condition(self.J)
        self.J_probe = self.J.copy()
        self.mode, self.J_source = "control", "probed"
        self.events.append({"event": "jacobian_probed", "condition": cond, "J": self.J.tolist()})
        self._save({"pose_at_probe": {"c_mm": state.c_mm.tolist(), "theta_deg": state.theta_deg.tolist()},
                    "probe_moves": int(self.probe_step)})
        return None

    # -- control -----------------------------------------------------------------
    def _broyden(self, e: np.ndarray, state: ControlState) -> None:
        b = self.cfg.broyden
        if not b.enabled or self.e_prev is None or self.dq_prev is None:
            return
        de, dq = e - self.e_prev, self.dq_prev
        noise = np.ones(6) if state.sigma_c_mm is None else np.r_[state.sigma_c_mm, state.sigma_theta_deg] / self.scale
        if np.linalg.norm(dq) <= b.min_step or np.linalg.norm(de) <= b.noise_k_sigma * np.linalg.norm(noise):
            return
        self.J = self.J + b.beta * np.outer(de - self.J @ dq, dq) / float(dq @ dq)
        if self.condition(self.J) > self.cfg.probe.max_condition:
            self.J = self.J_probe.copy()
            self.events.append({"event": "broyden_reset_condition"})

    def step(self, state: ControlState) -> ActuatorCommand | Stop:
        if state.c_mm is None or state.theta_deg is None:
            return Stop(False, "no_pose")
        e = self.error(state)
        if self.mode == "probe":
            cmd = self._probe(e, state)
            if cmd is not None:
                return cmd
        err = float(np.linalg.norm(e))
        if self.err_prev is not None and self.mode == "control" and self.dq_prev is not None:
            self._broyden(e, state)
            self.worse_count = self.worse_count + 1 if err > self.err_prev else 0
            if self.worse_count >= 2 and self.cfg.broyden.enabled and self.J_probe is not None:
                self.J = self.J_probe.copy()
                self.worse_count = 0
                self.events.append({"event": "broyden_reset_error_increase"})
        W = self.weights(state)
        JtW = self.J.T * W
        A = JtW @ self.J
        mu = self.cfg.damping_mu if self.cfg.damping_mu is not None else 1e-3 * float(np.trace(A)) / 4.0
        dq = -self.cfg.gain * np.linalg.solve(A + mu * np.eye(4), JtW @ e)
        raw = dq.copy()
        dq = np.clip(dq, -self.max_step, self.max_step)
        q_new = np.clip(self.q + dq, -self.travel, self.travel)
        dq = q_new - self.q
        # Part of the error the 4 actuators cannot reach (e.g. c_y, roll).
        pred = self.J @ np.linalg.lstsq(self.J, e, rcond=None)[0]
        uncontrollable = (e - pred) * self.scale
        self.q = q_new
        self.e_prev, self.dq_prev, self.err_prev = e, dq, err
        return ActuatorCommand(dq=dq, kind="control", meta={
            "e": e.tolist(), "err_norm": err, "dq_unclamped": raw.tolist(), "mu": mu, "W": W.tolist(),
            "J_condition": self.condition(self.J), "J_source": self.J_source,
            "uncontrollable_residual": {"c_mm": uncontrollable[:3].tolist(), "theta_deg": uncontrollable[3:].tolist()}})


def make_controller(profile, settings=None, *, device_id: str = "dev", artifact_id: str = "art",
                    server_root: str | Path | None = None, clock=time.time):
    """Controller of a profile; None fields of ``control`` fall back to Settings."""
    c = profile.control
    if c.controller == "fixed_mapping":
        signs = c.signs
        spm = c.steps_per_mm or getattr(settings, "steps_per_mm", 800.0)
        return FixedMappingController(
            steps_per_mm=spm,
            sign_x=signs.x if signs else getattr(settings, "sign_move_x", -1),
            sign_z=signs.z if signs else getattr(settings, "sign_move_z", -1),
            sign_pan=signs.pan if signs else getattr(settings, "sign_rotate_pan", -1),
            sign_tilt=signs.tilt if signs else getattr(settings, "sign_rotate_tilt", -1))
    root = Path(server_root) if server_root else Path(__file__).resolve().parents[3]
    store = root / c.jacobian_dir / f"jacobian_{_safe(device_id)}_{_safe(artifact_id)}.json"
    return JacobianController(c, store_path=store, clock=clock)


def _safe(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in str(text))[:64] or "x"


def stop_check(state: ControlState, *, tol_t_mm: float, tol_R_deg: float, uncertainty_aware: bool = False,
               k_sigma: float = 2.0) -> bool:
    if state.c_mm is None or state.theta_deg is None:
        return False
    c, th = float(np.linalg.norm(state.c_mm)), float(np.linalg.norm(state.theta_deg))
    if uncertainty_aware and state.sigma_c_mm is not None:
        c += k_sigma * float(np.linalg.norm(state.sigma_c_mm))
        th += k_sigma * float(np.linalg.norm(state.sigma_theta_deg)) if state.sigma_theta_deg is not None else 0.0
    return c <= tol_t_mm and th <= tol_R_deg and math.isfinite(c)
