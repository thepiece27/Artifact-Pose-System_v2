"""Shared helpers for the CITA experiment CLIs (import path, calibration, logs)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SERVER = Path(__file__).resolve().parents[2]
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from app.experiments.profiles import load_profile  # noqa: E402
from app.experiments.session import Session  # noqa: E402
from app.experiments.trial_log import TrialLogger, json_safe, seed_everything, sha256_file  # noqa: E402

DEFAULT_CALIBRATION = SERVER / "data" / "camera_params" / "camera_params_lens_1.5.yaml"


def add_common_args(parser: argparse.ArgumentParser, *, profile_default: str = "pbl_legacy") -> None:
    parser.add_argument("--profile", default=profile_default, help="profile name or YAML path")
    parser.add_argument("--calib", default=None,
                        help="calibration YAML (default: the session's calibration_file if present, "
                             "else data/camera_params/camera_params_lens_1.5.yaml)")
    parser.add_argument("--log-root", default=None, help="override profile logging.root")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=None, help="override profile seed")


def resolve_calibration(session: Session | None, calib: str | None) -> Path:
    if calib:
        return Path(calib)
    if session is not None:
        own = session.calibration_path()
        if own is not None:
            return own
        name = (session.manifest.get("camera") or {}).get("calibration_file")
        if name:
            candidate = SERVER / "data" / "camera_params" / name
            if candidate.exists():
                return candidate
    return DEFAULT_CALIBRATION


def load_calibration(path: Path) -> dict:
    from app.modules.artifact_pose.common import read_calibration
    cal = read_calibration(path)
    cal["path"] = str(path)
    cal["sha256"] = sha256_file(path)
    return cal


def make_logger(exp: str, args, profile) -> TrialLogger:
    root = args.log_root or profile.logging.root
    return TrialLogger(exp, root=root, run_id=args.run_id, profile=profile,
                       seed=args.seed if args.seed is not None else profile.seed, base_dir=SERVER)


def setup(args):
    profile = load_profile(args.profile)
    seed = args.seed if args.seed is not None else profile.seed
    rng = seed_everything(seed)
    return profile, seed, rng


def write_json(path: Path, data) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path
