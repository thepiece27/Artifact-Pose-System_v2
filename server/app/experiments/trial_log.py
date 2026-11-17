"""Append-only JSONL trial logs (schema ``cita.trial.v1``, Appendix B.2).

One line per (input, arm). Files are created exclusively and never
overwritten: ``logs/experiments/<exp>/<run_id>.jsonl``.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

SCHEMA = "cita.trial.v1"
REPO_ROOT = Path(__file__).resolve().parents[3]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_info(repo: str | Path = REPO_ROOT) -> dict:
    """Commit, branch and dirty flag; ``None`` values when git is unavailable."""
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=str(repo), capture_output=True,
                                 text=True, timeout=10, check=True)
            return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(status)}


def env_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__}
    for module, key in (("cv2", "opencv"), ("scipy", "scipy"), ("skimage", "skimage"),
                        ("yaml", "pyyaml")):
        try:
            mod = __import__(module)
            info[key] = getattr(mod, "__version__", None)
        except Exception:  # pragma: no cover - optional dependency
            info[key] = None
    try:
        from app.modules.artifact_pose.common import HAS_CPP
        info["native_pose_solver"] = bool(HAS_CPP)
    except Exception:  # pragma: no cover
        info["native_pose_solver"] = None
    return info


def seed_everything(seed: int) -> np.random.Generator:
    """Seed python, numpy (legacy + Generator) and OpenCV RNGs."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        import cv2
        cv2.setRNGSeed(int(seed) % (2 ** 31))
    except Exception:  # pragma: no cover
        pass
    return np.random.default_rng(seed)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, float) and value in (float("inf"), float("-inf")):
        return None
    if hasattr(value, "to_json"):
        return json_safe(value.to_json())
    return value


def new_run_id(prefix: str = "") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}{stamp}_{uuid.uuid4().hex[:6]}"


class TrialLogger:
    """Exclusive-create, append + flush JSONL writer."""

    def __init__(self, exp: str, *, root: str | Path = "logs/experiments", run_id: str | None = None,
                 profile=None, seed: int | None = None, resume: bool = False,
                 base_dir: str | Path | None = None) -> None:
        root = Path(root)
        if not root.is_absolute():
            root = Path(base_dir or REPO_ROOT / "server") / root
        self.exp = exp
        self.run_id = run_id or new_run_id()
        self.path = root / exp / f"{self.run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not resume:
            raise FileExistsError(f"Refusing to overwrite existing log {self.path}")
        if not self.path.exists():
            # 'x' guarantees we never clobber a file created concurrently.
            with open(self.path, "x", encoding="utf-8"):
                pass
        self.profile_name = getattr(profile, "name", None)
        self.profile_hash = getattr(profile, "profile_hash", None)
        self.seed = seed if seed is not None else getattr(profile, "seed", None)
        git = git_info()
        self.git_commit, self.git_dirty = git["commit"], git["dirty"]
        self.env = env_info()
        self.count = 0

    def write(self, record: dict) -> dict:
        line = json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False, sort_keys=False)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.count += 1
        return record

    def log(self, *, arm: str, input: dict, output: dict | None = None, status: str = "ok",
            error: str | None = None, timing_ms: dict | None = None, t_start: str | None = None,
            t_end: str | None = None, **extra: Any) -> dict:
        record = {"schema": SCHEMA, "run_id": self.run_id, "exp": self.exp, "arm": arm,
                  "profile": self.profile_name, "profile_hash": self.profile_hash,
                  "git_commit": self.git_commit, "git_dirty": self.git_dirty,
                  "seed": self.seed, "env": self.env, "input": input,
                  "output": output or {}, "timing_ms": timing_ms or {}, "status": status,
                  "error": error, "t_start": t_start or utc_now(), "t_end": t_end or utc_now()}
        record.update(extra)
        return self.write(record)

    @contextlib.contextmanager
    def trial(self, *, arm: str, input: dict, **extra: Any) -> Iterator[dict]:
        """``with log.trial(arm=..., input=...) as out: out[...] = ...``.

        Exceptions are logged with ``status="error"`` and re-raised.
        """
        t_start, started = utc_now(), time.perf_counter()
        out: dict = {}
        try:
            yield out
        except Exception as exc:
            self.log(arm=arm, input=input, output=out, status="error",
                     error=f"{type(exc).__name__}: {exc}",
                     timing_ms={"total": (time.perf_counter() - started) * 1000},
                     t_start=t_start, **extra)
            raise
        status = out.pop("_status", "ok")
        self.log(arm=arm, input=input, output=out, status=status, error=out.pop("_error", None),
                 timing_ms={"total": (time.perf_counter() - started) * 1000, **out.pop("_timings", {})},
                 t_start=t_start, **extra)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def describe_runtime() -> dict:
    """Small header used by CLIs: git + env + argv."""
    return {"git": git_info(), "env": env_info(), "argv": sys.argv}
