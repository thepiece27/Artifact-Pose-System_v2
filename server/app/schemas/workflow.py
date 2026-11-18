from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CaptureJobType(str, Enum):
    alignment = "alignment"
    golden_sample = "golden_sample"


class CaptureRequest(BaseModel):
    artifact_id: str = "artifact_demo_001"
    job_type: CaptureJobType = CaptureJobType.alignment
    basename_prefix: str | None = None
    use_latest_metadata: bool = True
    camera_overrides: dict[str, Any] = Field(default_factory=dict)


class StartAlignmentRequest(BaseModel):
    artifact_id: str = "artifact_demo_001"
    use_latest_metadata: bool = True
    camera_overrides: dict[str, Any] = Field(default_factory=dict)
    # CITA experiments (optional; absent = cita-baseline behaviour).
    cita_arm: str | None = Field(default=None, pattern=r"^C-[A-D]$")
    cita_trial_id: str | None = Field(default=None, max_length=80, pattern=r"^[A-Za-z0-9_.:-]+$")
    keep_pose: bool | None = None
    cita_probe_reuse: str | None = Field(default=None, pattern=r"^(ttl|per_block|never)$")
    cita_block_id: str | None = Field(default=None, max_length=160, pattern=r"^[A-Za-z0-9_.:-]+$")


class StartInitializationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str = "artifact_demo_001"
    camera_overrides: dict[str, Any] = Field(default_factory=dict)


class TriggerCommandResponse(BaseModel):
    ok: bool
    device_id: str
    mode: str
    published: bool
    topic: str | None = None
    publish_error: str | None = None
    task_id: str
    queued: int
    payload: dict[str, Any]


class LatestCaptureMetadataResponse(BaseModel):
    ok: bool
    device_id: str
    metadata: dict[str, Any] | None = None
