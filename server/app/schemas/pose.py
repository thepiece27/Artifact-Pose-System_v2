from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class PoseHealthResponse(BaseModel):
    ok: bool
    available: bool
    artifact_pose_root: str
    camera_params_dir: str
    camera_params: str
    configured_lens_position: Optional[float]
    camera_lens_position: Optional[float]
    golden_pose: str
    message: str
    cita_profile: Optional[str] = None
    cita_profile_hash: Optional[str] = None


class PoseCorrectionResponse(BaseModel):
    ok: bool
    result: dict


class PoseInitializeResponse(BaseModel):
    ok: bool
    message: str
    result: dict


class GoldenPoseStatusResponse(BaseModel):
    artifact_id: str
    has_golden_pose: bool
