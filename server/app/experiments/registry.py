"""Registry of experiment arms: code -> implementation + default profile.

Implementations are referenced as ``"module:attribute"`` strings and imported
lazily, so the registry can be imported without OpenCV-heavy modules.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

POSE_METHODS = ("M1", "M2", "M3", "M4", "M5")
POSE_ABLATIONS = ("M2-full", "M4-jitter")
CONTROL_ARMS = ("C-A", "C-B", "C-C", "C-D")
PIPELINES = ("P0", "P1", "P2", "P3", "P4")


@dataclass(frozen=True)
class ArmEntry:
    code: str
    kind: str              # pose | control | inspection
    target: str            # "module:attribute"
    default_profile: str   # profile file name in configs/profiles
    description: str

    def resolve(self) -> Callable[..., Any]:
        module, _, attr = self.target.partition(":")
        return getattr(importlib.import_module(module), attr)


_REGISTRY: dict[str, ArmEntry] = {}


def register(code: str, kind: str, target: str, default_profile: str, description: str) -> ArmEntry:
    if code in _REGISTRY:
        raise ValueError(f"Arm {code} already registered")
    entry = ArmEntry(code, kind, target, default_profile, description)
    _REGISTRY[code] = entry
    return entry


def get(code: str) -> ArmEntry:
    try:
        return _REGISTRY[code]
    except KeyError as exc:
        raise KeyError(f"Unknown arm {code!r}; known: {sorted(_REGISTRY)}") from exc


def arms(kind: str | None = None) -> list[ArmEntry]:
    return [e for e in _REGISTRY.values() if kind is None or e.kind == kind]


def load_arm_profile(code: str, **kwargs):
    from .profiles import load_profile
    return load_profile(get(code).default_profile, **kwargs)


_POSE = "app.modules.artifact_pose.methods:estimate_pose"
register("M1", "pose", _POSE, "M1", "Diamond only (IPPE_SQUARE + ambiguity check)")
register("M2", "pose", _POSE, "M2", "Markerless Hung-style ORB/QuadTree + EPnP + pose-only (848x480)")
register("M2-full", "pose", _POSE, "M2-full", "Markerless Hung-style at full resolution")
register("M3", "pose", _POSE, "M3", "Hybrid, PBL weights (Omega_d=1e4 I, Omega_o=I), Huber 2")
register("M4", "pose", _POSE, "M4", "Hybrid, measured-noise weights + covariance")
register("M4-jitter", "pose", _POSE, "M4-jitter", "Ablation: M4 weighted with static-jitter sigmas")
register("M5", "pose", _POSE, "M5", "M4 + stacked fusion of 3 static frames")

_CTRL = "app.modules.artifact_pose.control:make_controller"
register("C-A", "control", _CTRL, "C-A", "Pose M3 + fixed mapping (PBL)")
register("C-B", "control", _CTRL, "C-B", "Pose M5 + I2 + fixed mapping")
register("C-C", "control", _CTRL, "C-C", "Pose M3 + Jacobian controller (I3)")
register("C-D", "control", _CTRL, "C-D", "Pose M5 + I2 + Jacobian controller (proposed)")

_PIPE = "app.modules.change_detection.pipelines:run_pipeline"
register("P0", "inspection", _PIPE, "P0", "PBL: whole-frame SIFT + SSIM + Otsu + YOLO crops (legacy coordinate bug kept)")
register("P1", "inspection", _PIPE, "P1", "P0 + painting ROI mask (+ fixed crop coordinates)")
register("P2", "inspection", _PIPE, "P2", "P1 + ROI homography + ECC + photometric + standard SSIM + fixed threshold")
register("P3", "inspection", _PIPE, "P3", "P2 + two-phase YOLO on AIInputPackage (proposed)")
register("P4", "inspection", _PIPE, "P4", "YOLO only on the current image, no reference")
