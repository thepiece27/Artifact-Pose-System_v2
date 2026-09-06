from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.dependencies import get_container, get_current_user, require_device_access
from app.core.database import get_db
from app.models.artifact import Artifact, Image, ImageType
from app.schemas.pose import (
    GoldenPoseStatusResponse,
    PoseCorrectionResponse,
    PoseHealthResponse,
    PoseInitializeResponse,
)
from app.services.state import AppContainer
from app.core.uploads import safe_component, validate_image_bytes

logger = logging.getLogger(__name__)

router = APIRouter()


async def _save_temp_upload(folder: Path, file: UploadFile, container: AppContainer) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    safe_name = safe_component(Path(file.filename or "upload.png").name, field="filename", max_length=120)
    content = await file.read(container.settings.max_upload_bytes + 1)
    validate_image_bytes(content, max_bytes=container.settings.max_upload_bytes,
                         max_pixels=container.settings.max_image_pixels)
    target = folder / f"{int(time.time() * 1000)}_{safe_name}"
    target.write_bytes(content)
    return target


@router.get("/pose/health", response_model=PoseHealthResponse)
def pose_health(
    container: AppContainer = Depends(get_container),
    _=Depends(get_current_user),
) -> PoseHealthResponse:
    return PoseHealthResponse(**container.pose_service.health())


@router.post("/pose/correct", response_model=PoseCorrectionResponse)
async def pose_correct(
    file: UploadFile = File(...),
    container: AppContainer = Depends(get_container),
    _=Depends(get_current_user),
) -> PoseCorrectionResponse:
    temp_dir = container.settings.uploads_dir / "pose"
    image_path = await _save_temp_upload(temp_dir, file, container)

    try:
        result = container.pose_service.correct_image(image_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        image_path.unlink(missing_ok=True)

    return PoseCorrectionResponse(ok=True, result=result)


@router.post("/pose/initialize_golden", response_model=PoseInitializeResponse)
async def initialize_golden(
    left_file: UploadFile = File(...),
    right_file: UploadFile = File(...),
    artifact_id: str | None = Form(None),
    device_id: str | None = Form(None),
    device_code: str | None = Form(None),
    container: AppContainer = Depends(get_container),
    db: Session = Depends(get_db),
    _auth=Depends(require_device_access),
) -> PoseInitializeResponse:
    temp_dir = container.settings.uploads_dir / "pose_init"
    left_path = await _save_temp_upload(temp_dir, left_file, container)
    right_path = await _save_temp_upload(temp_dir, right_file, container)

    try:
        result = container.pose_service.initialize_golden(
            left_path, right_path, artifact_id=artifact_id or None
        )
    except Exception as exc:
        logger.error("[initialize_golden] FAILED: %s", exc, exc_info=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        left_path.unlink(missing_ok=True)
        right_path.unlink(missing_ok=True)

    # Auto-extract artifact_id from filename if Pi didn't send it as form field.
    # Pi saves files as: stereo_left_{artifact_id}_{ts_ns}.png
    if not artifact_id:
        match = re.search(r"stereo_left_([a-f0-9]{6})_", left_path.name)
        if match:
            artifact_id = match.group(1)
            logger.info("[initialize_golden] Auto-extracted artifact_id=%s from filename", artifact_id)

    # Save golden left image as artifact baseline (distinctive name)
    if artifact_id:
        artifact = db.query(Artifact).filter(Artifact.artifact_id == artifact_id).first()
        if artifact:
            ts_ms = int(time.time() * 1000)
            safe_artifact_id = safe_component(artifact_id, field="artifact_id")
            golden_dir = container.settings.uploads_dir / "artifacts" / safe_artifact_id
            golden_dir.mkdir(parents=True, exist_ok=True)
            golden_left_path = golden_dir / f"golden_left_{safe_artifact_id}_{ts_ms}.png"
            golden_left_path.write_bytes(left_path.read_bytes())

            baseline_image = Image(
                artifact_id=artifact_id,
                image_type=ImageType.baseline,
                image_path=str(golden_left_path),
                is_valid=True,
            )
            db.add(baseline_image)
            db.flush()
            artifact.baseline_image_id = baseline_image.image_id
            db.commit()
            logger.info(
                "[initialize_golden] Saved golden left as baseline for artifact=%s: %s",
                artifact_id, golden_left_path.name,
            )
        else:
            logger.warning("[initialize_golden] artifact_id=%s not found in DB, baseline not set", artifact_id)

    return PoseInitializeResponse(
        ok=True,
        message="Golden pose initialized successfully",
        result=result,
    )


@router.get("/pose/golden-pose/{artifact_id}/status", response_model=GoldenPoseStatusResponse)
def golden_pose_status(
    artifact_id: str,
    container: AppContainer = Depends(get_container),
    _=Depends(get_current_user),
) -> GoldenPoseStatusResponse:
    """Kiem tra artifact da co golden pose chua (per-artifact)."""
    has_it = container.pose_service.has_golden_pose(artifact_id)
    return GoldenPoseStatusResponse(artifact_id=artifact_id, has_golden_pose=has_it)
