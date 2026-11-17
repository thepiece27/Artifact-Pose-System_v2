"""Capture sessions (schema ``cita.session.v1``, Appendix B.1).

A session is a directory ``<root>/<exp>/<session_id>/`` holding the original
images (never re-encoded) and ``manifest.json``. This module only reads and
validates; images are never modified.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .trial_log import sha256_file

SESSION_SCHEMA = "cita.session.v1"
REQUIRED_TOP = ("schema", "session_id", "experiment", "frames")
REQUIRED_FRAME = ("frame_id", "file")
SPLITS = ("na", "train", "val", "test")


class SessionError(ValueError):
    pass


@dataclass
class Frame:
    data: dict
    root: Path

    @property
    def frame_id(self) -> str:
        return self.data["frame_id"]

    @property
    def path(self) -> Path:
        return self.root / self.data["file"]

    @property
    def pose_index(self):
        return self.data.get("pose_index")

    @property
    def role(self) -> str:
        return self.data.get("role", "grid")

    def get(self, key, default=None):
        return self.data.get(key, default)

    def load(self, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
        img = cv2.imread(str(self.path), flags)
        if img is None:
            raise SessionError(f"Cannot read {self.path}")
        return img


@dataclass
class Session:
    root: Path
    manifest: dict
    frames: list[Frame] = field(default_factory=list)

    @property
    def session_id(self) -> str:
        return self.manifest["session_id"]

    @property
    def experiment(self) -> str:
        return self.manifest["experiment"]

    @property
    def split(self) -> str:
        return self.manifest.get("split", "na")

    def frame(self, frame_id: str) -> Frame:
        for f in self.frames:
            if f.frame_id == frame_id:
                return f
        raise SessionError(f"Frame {frame_id} not in session {self.session_id}")

    def golden_frames(self) -> list[Frame]:
        ids = (self.manifest.get("golden") or {}).get("frame_ids") or []
        return [self.frame(i) for i in ids]

    def stereo_right(self) -> Frame | None:
        fid = (self.manifest.get("golden") or {}).get("stereo_right_frame_id")
        return self.frame(fid) if fid else None

    def evaluation_frames(self) -> list[Frame]:
        """Frames that are neither golden nor the stereo-right frame."""
        golden = self.manifest.get("golden") or {}
        skip = set(golden.get("frame_ids") or [])
        if golden.get("stereo_right_frame_id"):
            skip.add(golden["stereo_right_frame_id"])
        return [f for f in self.frames if f.frame_id not in skip]

    def groups(self) -> dict:
        """Frames grouped by ``pose_index`` (static repeats of one rig position)."""
        out: dict = {}
        for f in self.evaluation_frames():
            out.setdefault(f.pose_index, []).append(f)
        return out

    def calibration_path(self) -> Path | None:
        name = (self.manifest.get("camera") or {}).get("calibration_file")
        if not name:
            return None
        p = self.root / name
        return p if p.exists() else None

    def iter_frames(self) -> Iterator[Frame]:
        return iter(self.frames)

    def verify_hashes(self) -> list[str]:
        """Return frame_ids whose file is missing or whose SHA-256 differs."""
        bad = []
        for f in self.frames:
            expected = f.get("sha256")
            if not f.path.exists() or (expected and sha256_file(f.path) != expected):
                bad.append(f.frame_id)
        return bad


def validate_manifest(manifest: dict) -> None:
    missing = [k for k in REQUIRED_TOP if k not in manifest]
    if missing:
        raise SessionError(f"Manifest missing keys {missing}")
    if manifest["schema"] != SESSION_SCHEMA:
        raise SessionError(f"Unsupported session schema {manifest['schema']!r}")
    if manifest.get("split", "na") not in SPLITS:
        raise SessionError(f"split must be one of {SPLITS}")
    if not isinstance(manifest["frames"], list):
        raise SessionError("frames must be a list")
    seen = set()
    for i, frame in enumerate(manifest["frames"]):
        miss = [k for k in REQUIRED_FRAME if k not in frame]
        if miss:
            raise SessionError(f"frames[{i}] missing {miss}")
        if frame["frame_id"] in seen:
            raise SessionError(f"Duplicate frame_id {frame['frame_id']}")
        if Path(frame["file"]).name != frame["file"]:
            raise SessionError(f"frames[{i}].file must be a bare file name")
        seen.add(frame["frame_id"])
    golden = manifest.get("golden") or {}
    for fid in list(golden.get("frame_ids") or []) + [golden.get("stereo_right_frame_id")]:
        if fid and fid not in seen:
            raise SessionError(f"golden refers to unknown frame {fid}")


def load_session(path: str | Path, *, verify: bool = False) -> Session:
    root = Path(path)
    manifest_path = root / "manifest.json" if root.is_dir() else root
    root = manifest_path.parent
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SessionError(f"Cannot read manifest {manifest_path}: {exc}") from exc
    validate_manifest(manifest)
    session = Session(root=root, manifest=manifest,
                      frames=[Frame(f, root) for f in manifest["frames"]])
    if verify:
        bad = session.verify_hashes()
        if bad:
            raise SessionError(f"Hash mismatch / missing files: {bad[:10]}")
    return session


def input_record(session: Session, frames: list[Frame]) -> dict:
    """``input`` block of a cita.trial.v1 record."""
    return {"session_id": session.session_id, "frame_ids": [f.frame_id for f in frames],
            "sha256": [f.get("sha256") for f in frames]}
