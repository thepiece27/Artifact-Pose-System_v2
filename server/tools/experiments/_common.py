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


def add_log_args(parser: argparse.ArgumentParser) -> None:
    """--log-root / --run-id for tools that do not take a method profile."""
    parser.add_argument("--log-root", default=None, help="default: logs/experiments")
    parser.add_argument("--run-id", default=None)


def log_tool_run(exp: str, arm: str, *, inputs: list, outputs: dict, log_root: str | None = None,
                 run_id: str | None = None, seed: int | None = None, status: str = "ok") -> Path:
    """One cita.trial.v1 record for evaluation / table / ROI tools (no method profile:
    ``profile`` and ``profile_hash`` are null; the evaluated runs carry their own)."""
    files = [Path(p) for p in inputs]
    log = TrialLogger(exp, root=log_root or "logs/experiments", run_id=run_id, profile=None, seed=seed,
                      base_dir=SERVER)
    log.log(arm=arm, input={"files": [str(p) for p in files],
                            "sha256": [sha256_file(p) if p.is_file() else None for p in files],
                            "argv": sys.argv[1:]},
            output=json_safe(outputs), status=status, record_type="tool")
    return log.path


def expand(patterns: list[str]) -> list[str]:
    import glob
    out = []
    for pattern in patterns:
        out.extend(sorted(glob.glob(pattern)) or [pattern])
    return out


def write_json(path: Path, data) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path
