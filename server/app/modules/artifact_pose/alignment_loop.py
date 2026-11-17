"""Alignment loop logic, independent of MQTT / HTTP (CITA Phase 3).

``AlignmentSession`` is a resumable state machine: feed it one observation
(a PoseEstimate, after k static frames for M5) and it returns a decision:

* ``move``     -> payload for the MQTT ``move`` command (+ motion_profile),
* ``done``     -> converged according to the common stop rule,
* ``fail``     -> max iterations / no improvement / no pose / controller stop.

The same object drives the synchronous ``run_alignment`` loop (virtual rig,
scripted real rig) and the event-driven server (one upload = one call;
state persisted with ``to_state``/``from_state``).

Common stop rule for every arm: ||c|| <= tau_t and ||theta|| <= tau_R
(optionally ||c|| + k sigma_c), or no improvement for N consecutive control
iterations, or N_max control iterations. Probe moves of the Jacobian
controller are counted separately (``n_probe_moves``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .control import (ActuatorCommand, ControlState, FixedMappingController, JacobianController, Stop,
                      make_controller, stop_check)


@dataclass
class LoopConfig:
    max_iterations: int = 10
    tol_t_mm: float = 10.0
    tol_R_deg: float = 1.0
    uncertainty_aware: bool = False
    k_sigma: float = 2.0
    no_improvement_iters: int = 2
    max_pose_failures: int = 3
    steps_per_mm: float = 800.0
    motion_profile: dict = field(default_factory=dict)

    @classmethod
    def from_profile(cls, profile, settings=None) -> "LoopConfig":
        c = profile.control
        return cls(max_iterations=c.max_iterations or getattr(settings, "max_alignment_iterations", 10),
                   tol_t_mm=c.tol_translation_mm or getattr(settings, "trans_tolerance_mm", 10.0),
                   tol_R_deg=c.tol_rotation_deg or getattr(settings, "rot_tolerance_deg", 1.0),
                   uncertainty_aware=c.stop.uncertainty_aware, k_sigma=c.stop.k_sigma,
                   no_improvement_iters=c.stop.no_improvement_iters,
                   steps_per_mm=c.steps_per_mm or getattr(settings, "steps_per_mm", 800.0),
                   motion_profile=profile.motion_payload())


@dataclass
class Decision:
    kind: str                       # move | done | fail
    payload: dict | None = None
    reason: str | None = None
    record: dict = field(default_factory=dict)


def _estimate_json(est: Any) -> dict:
    if est is None:
        return {}
    if isinstance(est, dict):
        return est
    return est.to_json() if hasattr(est, "to_json") else {}


class AlignmentSession:
    def __init__(self, controller, cfg: LoopConfig, *, arm: str = "", meta: dict | None = None) -> None:
        self.controller = controller
        self.cfg = cfg
        self.arm = arm
        self.meta = dict(meta or {})
        self.iteration = 0              # control iterations (moves of kind control/legacy)
        self.n_observations = 0
        self.n_probe_moves = 0
        self.n_moves = 0
        self.pose_failures = 0
        self.best_err = None
        self.no_improve = 0
        self.history: list[dict] = []
        self.t_start = time.time()
        self.finished: Decision | None = None

    # -- persistence -----------------------------------------------------------
    def to_state(self) -> dict:
        return {"arm": self.arm, "meta": self.meta, "iteration": self.iteration,
                "n_observations": self.n_observations, "n_probe_moves": self.n_probe_moves,
                "n_moves": self.n_moves, "pose_failures": self.pose_failures, "best_err": self.best_err,
                "no_improve": self.no_improve, "t_start": self.t_start,
                "controller": self.controller.to_state(), "cfg": self.cfg.__dict__}

    @classmethod
    def from_state(cls, data: dict, controller) -> "AlignmentSession":
        s = cls(controller, LoopConfig(**data["cfg"]), arm=data.get("arm", ""), meta=data.get("meta"))
        for key in ("iteration", "n_observations", "n_probe_moves", "n_moves", "pose_failures", "best_err",
                    "no_improve", "t_start"):
            setattr(s, key, data[key])
        controller.load_state(data["controller"])
        return s

    # -- core ------------------------------------------------------------------
    def _finish(self, kind: str, reason: str, record: dict) -> Decision:
        self.finished = Decision(kind, None, reason, record)
        return self.finished

    def on_observation(self, est: Any, *, extra: dict | None = None) -> Decision:
        if self.finished is not None:
            return self.finished
        self.n_observations += 1
        ej = _estimate_json(est)
        ok = bool(ej.get("ok", ej.get("accepted", False)))
        state = ControlState.from_estimate(self.iteration, est, self.cfg.tol_t_mm, self.cfg.tol_R_deg) if ok else None
        record = {"observation": self.n_observations, "control_iteration": self.iteration, "arm": self.arm,
                  "pose_ok": ok, "pose_status": ej.get("status"),
                  "c_mm": None if state is None or state.c_mm is None else state.c_mm.tolist(),
                  "theta_deg": None if state is None or state.theta_deg is None else state.theta_deg.tolist(),
                  "sigma_c_mm": None if state is None or state.sigma_c_mm is None else state.sigma_c_mm.tolist(),
                  "t_s": time.time() - self.t_start, **(extra or {})}
        if state is not None and state.c_mm is not None:
            record["c_norm_mm"] = float(np.linalg.norm(state.c_mm))
            record["rot_deg"] = float(np.linalg.norm(state.theta_deg))
        self.history.append(record)
        if not ok or state is None or state.c_mm is None:
            self.pose_failures += 1
            if self.pose_failures >= self.cfg.max_pose_failures:
                return self._finish("fail", "pose_unavailable", record)
            return Decision("retry", None, "pose_unavailable", record)
        self.pose_failures = 0
        if stop_check(state, tol_t_mm=self.cfg.tol_t_mm, tol_R_deg=self.cfg.tol_R_deg,
                      uncertainty_aware=self.cfg.uncertainty_aware, k_sigma=self.cfg.k_sigma):
            return self._finish("done", "converged", record)
        probing = isinstance(self.controller, JacobianController) and self.controller.mode == "probe"
        if not probing:
            # Normalised error used for the no-improvement rule (same for every arm).
            err = record["c_norm_mm"] / self.cfg.tol_t_mm + record["rot_deg"] / self.cfg.tol_R_deg
            if self.best_err is None or err < self.best_err - 1e-9:
                self.best_err, self.no_improve = err, 0
            elif self.iteration > 0:
                self.no_improve += 1
                if self.no_improve >= self.cfg.no_improvement_iters:
                    return self._finish("fail", "no_improvement", record)
            if self.iteration >= self.cfg.max_iterations:
                return self._finish("fail", "max_iterations", record)
        cmd = self.controller.step(state)
        if isinstance(cmd, Stop):
            return self._finish("done" if cmd.success else "fail", cmd.reason, record)
        payload = cmd.to_move_payload(self.cfg.steps_per_mm)
        if self.cfg.motion_profile:
            payload["motion_profile"] = dict(self.cfg.motion_profile)
        record["command"] = cmd.to_json()
        record["move_payload"] = payload
        self.n_moves += 1
        if cmd.kind == "probe":
            self.n_probe_moves += 1
        else:
            self.iteration += 1
        return Decision("move", payload, None, record)

    def summary(self) -> dict:
        last = next((h for h in reversed(self.history) if h.get("pose_ok")), {})
        return {"arm": self.arm, "success_est": bool(self.finished and self.finished.kind == "done"),
                "stop_reason": self.finished.reason if self.finished else None,
                "control_iterations": self.iteration, "n_probe_moves": self.n_probe_moves, "n_moves": self.n_moves,
                "n_observations": self.n_observations, "time_s": time.time() - self.t_start,
                "final_c_mm": last.get("c_mm"), "final_c_norm_mm": last.get("c_norm_mm"),
                "final_rot_deg": last.get("rot_deg"), "final_sigma_c_mm": last.get("sigma_c_mm"),
                "controller_events": getattr(self.controller, "events", [])}


class RigPort(Protocol):
    """What the synchronous loop needs from a rig (real or virtual)."""

    def move(self, payload: dict) -> dict: ...

    def observe(self) -> Any: ...


def run_alignment(port: RigPort, session: AlignmentSession, *, logger=None, log_input: dict | None = None,
                  max_observations: int = 100) -> dict:
    """Synchronous loop: observe -> decide -> move, until done/fail."""
    for _ in range(max_observations):
        est = port.observe()
        decision = session.on_observation(est)
        rec = dict(decision.record)
        if logger is not None:
            logger.log(arm=session.arm, input=log_input or {}, output={"iteration": rec},
                       status=decision.kind, error=decision.reason, record_type="iteration")
        if decision.kind in ("done", "fail"):
            break
        if decision.kind == "move":
            ack = port.move(decision.payload)
            rec["ack"] = ack
    else:
        session._finish("fail", "max_observations", {})
    return session.summary()


def build_session(profile, settings=None, *, arm: str | None = None, device_id: str = "dev",
                  artifact_id: str = "art", server_root=None, reuse_jacobian: bool = True,
                  clock=time.time) -> AlignmentSession:
    controller = make_controller(profile, settings, device_id=device_id, artifact_id=artifact_id,
                                 server_root=server_root, clock=clock)
    controller.reset({"reuse_jacobian": reuse_jacobian})
    return AlignmentSession(controller, LoopConfig.from_profile(profile, settings), arm=arm or profile.name)


__all__ = ["ActuatorCommand", "AlignmentSession", "Decision", "FixedMappingController", "JacobianController",
           "LoopConfig", "RigPort", "build_session", "run_alignment"]
