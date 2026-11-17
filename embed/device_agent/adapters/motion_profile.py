"""I2 motion profile for the device (software mechanical compensation).

Priority: ``motion_profile`` field of the MQTT command > environment
variables > defaults. Every default reproduces the device behaviour at tag
``cita-baseline`` (integer servo angles, no settle wait, no unidirectional
approach, PWM kept on).

Environment variables (all optional)::

    CITA_MOTION_FLOAT_ANGLES=0|1
    CITA_MOTION_SETTLE_S=1.0
    CITA_MOTION_UNIDIRECTIONAL=0|1
    CITA_MOTION_APPROACH_SIGN=1|-1
    CITA_MOTION_BACKLASH_DEG=2.0
    CITA_MOTION_BACKLASH_STEPS=0
    CITA_MOTION_SERVO_RELEASE=0|1
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Mapping

# Hard safety caps: a command can never exceed these, whatever it requests.
MAX_SETTLE_S = 10.0
MAX_BACKLASH_DEG = 5.0
MAX_BACKLASH_STEPS = 8000  # 10 mm at 800 steps/mm

ENV_KEYS = {
    "float_angles": "CITA_MOTION_FLOAT_ANGLES",
    "settle_s": "CITA_MOTION_SETTLE_S",
    "unidirectional": "CITA_MOTION_UNIDIRECTIONAL",
    "approach_sign": "CITA_MOTION_APPROACH_SIGN",
    "backlash_deg": "CITA_MOTION_BACKLASH_DEG",
    "backlash_steps": "CITA_MOTION_BACKLASH_STEPS",
    "servo_release_after_move": "CITA_MOTION_SERVO_RELEASE",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class MotionProfile:
    float_angles: bool = False
    settle_s: float = 0.0
    unidirectional: bool = False
    approach_sign: int = 1
    backlash_deg: float = 0.0
    backlash_steps: int = 0
    servo_release_after_move: bool = False

    def sanitized(self) -> "MotionProfile":
        sign = 1 if int(self.approach_sign) >= 0 else -1
        return replace(
            self,
            float_angles=bool(self.float_angles),
            settle_s=min(MAX_SETTLE_S, max(0.0, float(self.settle_s))),
            unidirectional=bool(self.unidirectional),
            approach_sign=sign,
            backlash_deg=min(MAX_BACKLASH_DEG, max(0.0, float(self.backlash_deg))),
            backlash_steps=min(MAX_BACKLASH_STEPS, max(0, int(self.backlash_steps))),
            servo_release_after_move=bool(self.servo_release_after_move),
        )

    @staticmethod
    def _coerce(name: str, value: Any) -> Any:
        if name in ("float_angles", "unidirectional", "servo_release_after_move"):
            return _as_bool(value)
        if name in ("settle_s", "backlash_deg"):
            return float(value)
        return int(float(value))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None, base: "MotionProfile | None" = None) -> "MotionProfile":
        base = base or cls()
        if not isinstance(data, Mapping):
            return base
        known = {f.name for f in fields(cls)}
        updates = {}
        for key, value in data.items():
            if key in known and value is not None:
                try:
                    updates[key] = cls._coerce(key, value)
                except (TypeError, ValueError):
                    continue  # ignore malformed field, keep lower-priority value
        return replace(base, **updates).sanitized()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "MotionProfile":
        env = os.environ if env is None else env
        data = {name: env[var] for name, var in ENV_KEYS.items() if env.get(var, "").strip() != ""}
        return cls.from_mapping(data)

    @classmethod
    def resolve(cls, command_profile: Mapping[str, Any] | None,
                env: Mapping[str, str] | None = None) -> "MotionProfile":
        return cls.from_mapping(command_profile, base=cls.from_env(env))

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_legacy(self) -> bool:
        return self == MotionProfile()


def unidirectional_waypoints(current: float, target: float, *, enabled: bool, approach_sign: int,
                             backlash: float, lower: float | None = None,
                             upper: float | None = None) -> list[float]:
    """Positions to command so that ``target`` is always reached moving in
    ``approach_sign`` direction. If the move goes the other way, overshoot by
    ``backlash`` first, then come back. Overshoot is clamped to [lower, upper].
    """
    if not enabled or backlash <= 0 or target == current:
        return [target]
    if (target - current) * approach_sign >= 0:
        return [target]
    pre = target - approach_sign * backlash
    if lower is not None:
        pre = max(lower, pre)
    if upper is not None:
        pre = min(upper, pre)
    if pre == target:
        return [target]
    return [pre, target]
