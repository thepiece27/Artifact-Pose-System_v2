"""Experiment profiles: YAML files -> merged, validated, hashed configuration.

* Files live in ``server/configs/profiles/<name>.yaml``.
* ``extends: <parent>`` merges the parent first (dicts deep-merged, lists and
  scalars replaced), cycles are rejected.
* The merged dict is validated against the frozen dataclasses below (unknown
  keys, wrong types and out-of-range values raise ``ProfileError``).
* ``profile_hash`` = SHA-256 of the canonical JSON (sorted keys) of the fully
  resolved configuration, defaults included.

Defaults of every dataclass reproduce the behaviour of the code at tag
``cita-baseline``; new behaviour is only switched on by profile values.
Units: millimetres/degrees at this boundary (as written by people), converted
to metres/radians inside the algorithms.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

SERVER_ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = SERVER_ROOT / "configs" / "profiles"
DEFAULT_PROFILE = "pbl_legacy"
PROFILE_ENV = "CITA_PROFILE"


class ProfileError(ValueError):
    """Raised for missing, malformed or invalid profiles."""


def _choice(value: Any, name: str, choices: tuple) -> None:
    if value not in choices:
        raise ProfileError(f"{name}={value!r} must be one of {choices}")


def _positive(value: Any, name: str, *, allow_zero: bool = False, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if value is None or (value < 0 if allow_zero else value <= 0):
        raise ProfileError(f"{name}={value!r} must be {'>= 0' if allow_zero else '> 0'}")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PoseSection:
    # legacy = run_correction_step exactly as at cita-baseline.
    method: str = "legacy"
    diamond_sigma_px: float = 0.5
    orb_sigma_px: float = 1.5
    huber_delta: float = 2.0
    huber: bool = True
    backend: str = "auto"
    orb_region: str = "full"
    noise_model: Optional[str] = None
    frames: int = 1
    fusion: str = "stacked"
    covariance: bool = False
    resize_to: Optional[list] = None
    ratio: Optional[float] = None
    min_pnp_inliers: int = 6
    use_board_mask: bool = True

    def __post_init__(self):
        _choice(self.method, "pose.method", ("legacy", "M1", "M2", "M3", "M4", "M5"))
        _choice(self.backend, "pose.backend", ("auto", "python", "g2o"))
        _choice(self.orb_region, "pose.orb_region", ("full", "roi"))
        _choice(self.fusion, "pose.fusion", ("stacked", "average"))
        for name in ("diamond_sigma_px", "orb_sigma_px", "huber_delta"):
            _positive(getattr(self, name), f"pose.{name}")
        if self.frames < 1 or self.frames > 20:
            raise ProfileError("pose.frames must be in [1, 20]")
        if self.method != "M5" and self.frames != 1:
            raise ProfileError("pose.frames > 1 is only meaningful for M5")
        if self.resize_to is not None and (len(self.resize_to) != 2 or min(self.resize_to) <= 0):
            raise ProfileError("pose.resize_to must be [width, height]")
        if self.ratio is not None and not 0 < self.ratio < 1:
            raise ProfileError("pose.ratio must be in (0, 1)")
        if self.min_pnp_inliers < 4:
            raise ProfileError("pose.min_pnp_inliers must be >= 4")


@dataclass(frozen=True)
class GoldenSection:
    # diamond = T_right_left estimated from the Diamond (cita-baseline);
    # commanded = pure translation of the commanded slider baseline.
    stereo_source: str = "diamond"
    baseline_mm: float = 100.0
    build_markerless: bool = False

    def __post_init__(self):
        _choice(self.stereo_source, "golden.stereo_source", ("diamond", "commanded", "commanded_epipolar"))
        _positive(self.baseline_mm, "golden.baseline_mm")


@dataclass(frozen=True)
class ProbeDelta:
    x_mm: float = 5.0
    z_mm: float = 5.0
    yaw_deg: float = 2.0
    pitch_deg: float = 2.0

    def __post_init__(self):
        for f in dataclasses.fields(self):
            _positive(getattr(self, f.name), f"control.probe.delta.{f.name}")


@dataclass(frozen=True)
class ProbeSection:
    scheme: str = "chained_forward"
    delta: ProbeDelta = field(default_factory=ProbeDelta)
    ttl_s: float = 3600.0
    max_condition: float = 1e6

    def __post_init__(self):
        _choice(self.scheme, "control.probe.scheme", ("chained_forward", "central"))
        _positive(self.ttl_s, "control.probe.ttl_s", allow_zero=True)
        _positive(self.max_condition, "control.probe.max_condition")


@dataclass(frozen=True)
class MaxStep:
    x_mm: float = 20.0
    z_mm: float = 20.0
    yaw_deg: float = 5.0
    pitch_deg: float = 5.0

    def __post_init__(self):
        for f in dataclasses.fields(self):
            _positive(getattr(self, f.name), f"control.max_step.{f.name}")


@dataclass(frozen=True)
class TravelLimits:
    x_mm: float = 200.0
    z_mm: float = 200.0
    yaw_deg: float = 45.0
    pitch_deg: float = 45.0

    def __post_init__(self):
        for f in dataclasses.fields(self):
            _positive(getattr(self, f.name), f"control.travel.{f.name}")


@dataclass(frozen=True)
class ErrorScale:
    s_t_mm: float = 1.0
    s_R_deg: float = 0.2

    def __post_init__(self):
        _positive(self.s_t_mm, "control.scale.s_t_mm")
        _positive(self.s_R_deg, "control.scale.s_R_deg")


@dataclass(frozen=True)
class BroydenSection:
    enabled: bool = False
    beta: float = 0.5
    min_step: float = 0.5
    noise_k_sigma: float = 3.0

    def __post_init__(self):
        if not 0 < self.beta <= 1:
            raise ProfileError("control.broyden.beta must be in (0, 1]")


@dataclass(frozen=True)
class StopSection:
    uncertainty_aware: bool = False
    k_sigma: float = 2.0
    no_improvement_iters: int = 2

    def __post_init__(self):
        _positive(self.k_sigma, "control.stop.k_sigma", allow_zero=True)
        if self.no_improvement_iters < 1:
            raise ProfileError("control.stop.no_improvement_iters must be >= 1")


@dataclass(frozen=True)
class Signs:
    x: int = -1
    z: int = -1
    pan: int = -1
    tilt: int = -1

    def __post_init__(self):
        for f in dataclasses.fields(self):
            _choice(getattr(self, f.name), f"control.signs.{f.name}", (-1, 1))


@dataclass(frozen=True)
class ControlSection:
    controller: str = "fixed_mapping"
    # None = use server Settings (MAX_ALIGNMENT_ITERATIONS / TRANS_TOLERANCE_MM /
    # ROT_TOLERANCE_DEG / SIGN_*), i.e. exactly the cita-baseline behaviour.
    max_iterations: Optional[int] = None
    tol_translation_mm: Optional[float] = None
    tol_rotation_deg: Optional[float] = None
    signs: Optional[Signs] = None
    steps_per_mm: Optional[float] = None
    probe: ProbeSection = field(default_factory=ProbeSection)
    gain: float = 0.8
    damping_mu: Optional[float] = None
    max_step: MaxStep = field(default_factory=MaxStep)
    travel: TravelLimits = field(default_factory=TravelLimits)
    scale: ErrorScale = field(default_factory=ErrorScale)
    broyden: BroydenSection = field(default_factory=BroydenSection)
    stop: StopSection = field(default_factory=StopSection)
    jacobian_dir: str = "data/control"

    def __post_init__(self):
        _choice(self.controller, "control.controller", ("fixed_mapping", "jacobian"))
        if self.max_iterations is not None and not 1 <= self.max_iterations <= 100:
            raise ProfileError("control.max_iterations must be in [1, 100]")
        _positive(self.tol_translation_mm, "control.tol_translation_mm", allow_none=True)
        _positive(self.tol_rotation_deg, "control.tol_rotation_deg", allow_none=True)
        _positive(self.steps_per_mm, "control.steps_per_mm", allow_none=True)
        if not 0 < self.gain <= 1.5:
            raise ProfileError("control.gain must be in (0, 1.5]")
        _positive(self.damping_mu, "control.damping_mu", allow_zero=True, allow_none=True)


@dataclass(frozen=True)
class MotionProfile:
    """I2 device motion profile; every default equals the old device behaviour."""
    float_angles: bool = False
    settle_s: float = 0.0
    unidirectional: bool = False
    approach_sign: int = 1
    backlash_deg: float = 0.0
    backlash_steps: int = 0
    servo_release_after_move: bool = False

    def __post_init__(self):
        _positive(self.settle_s, "motion_profile.settle_s", allow_zero=True)
        _positive(self.backlash_deg, "motion_profile.backlash_deg", allow_zero=True)
        _positive(self.backlash_steps, "motion_profile.backlash_steps", allow_zero=True)
        _choice(self.approach_sign, "motion_profile.approach_sign", (-1, 1))
        if self.settle_s > 30 or self.backlash_deg > 10 or self.backlash_steps > 20000:
            raise ProfileError("motion_profile values exceed safety limits")

    def to_payload(self, explicit_keys: tuple[str, ...] | None = None) -> dict:
        data = dataclasses.asdict(self)
        return {k: data[k] for k in (explicit_keys or data)}


@dataclass(frozen=True)
class EccSection:
    enabled: bool = False
    levels: int = 3
    max_iter: int = 100
    eps: float = 1e-6
    max_side_px: int = 2000     # ECC runs at most at this long side (speed at 4K)

    def __post_init__(self):
        if not 1 <= self.levels <= 5:
            raise ProfileError("inspection.ecc.levels must be in [1, 5]")
        if self.max_side_px < 256:
            raise ProfileError("inspection.ecc.max_side_px must be >= 256")


@dataclass(frozen=True)
class SsimSection:
    mode: str = "legacy"
    work_long_side_px: int = 2000
    channels: str = "gray_color"

    def __post_init__(self):
        _choice(self.mode, "inspection.ssim.mode", ("legacy", "standard"))
        _choice(self.channels, "inspection.ssim.channels", ("gray", "gray_color", "lab"))
        if self.work_long_side_px < 256:
            raise ProfileError("inspection.ssim.work_long_side_px must be >= 256")


@dataclass(frozen=True)
class ThresholdSection:
    mode: str = "otsu_legacy"
    file: Optional[str] = None
    tau: Optional[float] = None

    def __post_init__(self):
        _choice(self.mode, "inspection.threshold.mode", ("otsu_legacy", "fixed"))
        if self.tau is not None and not 0 < self.tau < 1:
            raise ProfileError("inspection.threshold.tau must be in (0, 1)")


@dataclass(frozen=True)
class InspectionSection:
    pipeline: str = "P0"
    registration: str = "legacy_whole_frame"
    ecc: EccSection = field(default_factory=EccSection)
    photometric: str = "none"
    ssim: SsimSection = field(default_factory=SsimSection)
    threshold: ThresholdSection = field(default_factory=ThresholdSection)
    min_area_mm2: Optional[float] = None
    crop_source: str = "aligned"
    roi_erode_px: int = 4
    yolo: bool = True

    def __post_init__(self):
        _choice(self.pipeline, "inspection.pipeline", ("P0", "P1", "P2", "P3", "P4"))
        _choice(self.registration, "inspection.registration", ("legacy_whole_frame", "roi_homography"))
        _choice(self.photometric, "inspection.photometric", ("none", "gain_offset", "clahe_lab"))
        _choice(self.crop_source, "inspection.crop_source", ("aligned", "original_inverse"))
        _positive(self.min_area_mm2, "inspection.min_area_mm2", allow_none=True)


@dataclass(frozen=True)
class BoardSection:
    squares_x: int = 5
    squares_y: int = 7
    square_mm: float = 30.0
    marker_mm: float = 22.0
    dictionary: str = "DICT_5X5_100"
    ids_offset: int = 0
    min_corners: int = 12
    mask_dilate_px: int = 25

    def __post_init__(self):
        if self.squares_x < 2 or self.squares_y < 2:
            raise ProfileError("ground_truth.board squares must be >= 2")
        if not 0 < self.marker_mm < self.square_mm:
            raise ProfileError("ground_truth.board marker_mm must be in (0, square_mm)")
        if not self.dictionary.startswith("DICT_"):
            raise ProfileError("ground_truth.board.dictionary must be an OpenCV DICT_* name")
        if self.min_corners < 4:
            raise ProfileError("ground_truth.board.min_corners must be >= 4")


@dataclass(frozen=True)
class GroundTruthSection:
    board: BoardSection = field(default_factory=BoardSection)


@dataclass(frozen=True)
class LoggingSection:
    enabled: bool = True
    root: str = "logs/experiments"


@dataclass(frozen=True)
class Profile:
    name: str = DEFAULT_PROFILE
    description: str = ""
    seed: int = 2027
    pose: PoseSection = field(default_factory=PoseSection)
    golden: GoldenSection = field(default_factory=GoldenSection)
    control: ControlSection = field(default_factory=ControlSection)
    motion_profile: MotionProfile = field(default_factory=MotionProfile)
    # Keys explicitly present in the YAML motion_profile (sent to the device);
    # an empty mapping means "send nothing" = old device behaviour.
    motion_profile_keys: tuple = ()
    inspection: InspectionSection = field(default_factory=InspectionSection)
    ground_truth: GroundTruthSection = field(default_factory=GroundTruthSection)
    logging: LoggingSection = field(default_factory=LoggingSection)
    profile_hash: str = ""
    source_files: tuple = ()

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data["motion_profile_keys"] = list(self.motion_profile_keys)
        data["source_files"] = list(self.source_files)
        return data

    def motion_payload(self) -> dict:
        """Fields to put in MQTT ``motion_profile`` (empty = legacy device)."""
        return self.motion_profile.to_payload(tuple(self.motion_profile_keys)) if self.motion_profile_keys else {}

    def resolve_path(self, value: str | None) -> Path | None:
        if value is None:
            return None
        p = Path(value)
        return p if p.is_absolute() else SERVER_ROOT / p


# ---------------------------------------------------------------------------
# Generic dataclass builder with strict type checking
# ---------------------------------------------------------------------------

def _check_type(value: Any, hint: Any, path: str) -> Any:
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if value is None:
            return None
        return _check_type(value, args[0], path)
    if dataclasses.is_dataclass(hint):
        if not isinstance(value, dict):
            raise ProfileError(f"{path} must be a mapping")
        return _build(hint, value, path)
    if hint is bool:
        if not isinstance(value, bool):
            raise ProfileError(f"{path} must be a boolean, got {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ProfileError(f"{path} must be an integer, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProfileError(f"{path} must be a number, got {value!r}")
        value = float(value)
        if value != value or value in (float("inf"), float("-inf")):
            raise ProfileError(f"{path} must be finite")
        return value
    if hint is str:
        if not isinstance(value, str):
            raise ProfileError(f"{path} must be a string, got {value!r}")
        return value
    if hint is list or origin is list:
        if not isinstance(value, list):
            raise ProfileError(f"{path} must be a list")
        return list(value)
    if hint is tuple or origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ProfileError(f"{path} must be a list")
        return tuple(value)
    return value


def _build(cls, data: dict, path: str):
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ProfileError(f"Unknown key(s) in {path or 'profile'}: {unknown}")
    kwargs = {k: _check_type(v, hints[k], f"{path}.{k}" if path else k) for k, v in data.items()}
    try:
        return cls(**kwargs)
    except ProfileError:
        raise
    except (TypeError, ValueError) as exc:
        raise ProfileError(f"Invalid {path or 'profile'}: {exc}") from exc


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - pyyaml is in requirements
        raise ProfileError("PyYAML is required to read profiles (pip install pyyaml)") from exc
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ProfileError(f"Malformed YAML in {path}: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ProfileError(f"Profile {path} must contain a mapping")
    return data


def _resolve_file(name_or_path: str | Path, profile_dir: Path) -> Path:
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p.resolve()
    candidate = profile_dir / f"{name_or_path}.yaml"
    if candidate.exists():
        return candidate.resolve()
    raise ProfileError(f"Profile not found: {name_or_path} (looked in {profile_dir})")


def load_raw(name_or_path: str | Path, profile_dir: Path = PROFILE_DIR) -> tuple[dict, list[Path]]:
    """Merged raw mapping (``extends`` resolved) and the files that built it."""
    chain: list[Path] = []
    path = _resolve_file(name_or_path, profile_dir)
    data = _read_yaml(path)
    chain.append(path)
    seen = {path}
    merged_layers = [data]
    while "extends" in merged_layers[-1] and merged_layers[-1]["extends"]:
        name = str(merged_layers[-1]["extends"])
        try:  # sibling file first (custom profiles), then the shipped profiles
            parent = _resolve_file(name, chain[-1].parent)
        except ProfileError:
            parent = _resolve_file(name, profile_dir)
        if parent in seen:
            raise ProfileError(f"Cyclic 'extends' involving {parent.name}")
        seen.add(parent)
        chain.append(parent)
        merged_layers.append(_read_yaml(parent))
    merged: dict = {}
    for layer in reversed(merged_layers):
        merged = deep_merge(merged, {k: v for k, v in layer.items() if k != "extends"})
    # The leaf file decides the name.
    merged["name"] = data.get("name", path.stem)
    return merged, chain


def build_profile(raw: dict, source_files: tuple = ()) -> Profile:
    raw = dict(raw)
    motion = raw.get("motion_profile") or {}
    if not isinstance(motion, dict):
        raise ProfileError("motion_profile must be a mapping")
    raw["motion_profile"] = motion
    raw["motion_profile_keys"] = tuple(sorted(motion))
    for key in ("profile_hash", "source_files"):
        if key in raw:
            raise ProfileError(f"'{key}' is computed and must not be set in YAML")
    profile = _build(Profile, raw, "")
    canonical = json.dumps({k: v for k, v in profile.to_dict().items()
                            if k not in ("profile_hash", "source_files")},
                           sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    digest = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    return dataclasses.replace(profile, profile_hash=digest,
                               source_files=tuple(str(p) for p in source_files))


def load_profile(name_or_path: str | Path | None = None, *, profile_dir: Path = PROFILE_DIR,
                 overrides: dict | None = None) -> Profile:
    name_or_path = name_or_path or DEFAULT_PROFILE
    raw, chain = load_raw(name_or_path, profile_dir)
    if overrides:
        raw = deep_merge(raw, overrides)
    return build_profile(raw, tuple(chain))


def legacy_profile() -> Profile:
    """Built-in defaults == behaviour at cita-baseline (no file needed)."""
    return build_profile({"name": DEFAULT_PROFILE, "description": "built-in legacy defaults"})


def active_profile(env: dict | None = None, *, strict: bool = False) -> Profile:
    """Profile selected by ``CITA_PROFILE`` (default ``pbl_legacy``).

    A missing/unreadable profile falls back to the built-in legacy defaults
    (identical behaviour) unless ``strict``.
    """
    env = os.environ if env is None else env
    name = (env.get(PROFILE_ENV) or "").strip() or DEFAULT_PROFILE
    try:
        return load_profile(name)
    except ProfileError:
        if strict or name != DEFAULT_PROFILE:
            raise
        return legacy_profile()
