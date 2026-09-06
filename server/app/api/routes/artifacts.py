from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_container, get_current_user
from app.core.config import get_settings
from app.core.database import get_db
from app.models.artifact import Artifact, Image, ImageComparison, Schedule, InspectionType
from app.schemas.artifact import ArtifactCreate, ArtifactRead, ArtifactUpdate
from app.schemas.inspection_record import InspectionListResponse, InspectionRead
from app.services.state import AppContainer

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/api/v1/artifacts",
    tags=["artifacts"],
    dependencies=[Depends(get_current_user)],
)


def _to_url(path_str: str | None) -> str | None:
    if not path_str:
        return None
    if path_str.startswith("/uploads/"):
        return path_str
    try:
        if "/" not in path_str and "\\" not in path_str:
            return f"/uploads/{path_str}"
        uploads_dir = get_settings().uploads_dir.resolve()
        full = Path(path_str).resolve()
        rel = full.relative_to(uploads_dir).as_posix()
        return f"/uploads/{rel}"
    except (ValueError, OSError):
        # Never expose arbitrary filesystem paths as a media URL. Stored paths
        # must resolve under the private uploads root.
        return path_str if path_str.startswith("http") else None


def _serialize(artifact: Artifact) -> ArtifactRead:
    ref_path = artifact.baseline_image.image_path if artifact.baseline_image else None
    return ArtifactRead(
        id=artifact.artifact_id,
        name=artifact.name,
        description=artifact.description or "",
        location=artifact.location or "",
        status=artifact.status,
        inspection_interval_days=artifact.inspection_interval_days,
        has_image=artifact.baseline_image_id is not None,
        reference_image_path=_to_url(ref_path),
        created_at=artifact.created_at,
        updated_at=artifact.updated_at
    )


def _serialize_comparison(record: ImageComparison) -> InspectionRead:
    prev_path = record.previous_image.image_path if record.previous_image else None
    curr_path = record.current_image.image_path if record.current_image else None
    return InspectionRead(
        id=record.comparison_id,
        artifact_id=record.artifact_id,
        schedule_id=record.schedule_id,
        previous_image_path=_to_url(prev_path),
        current_image_path=_to_url(curr_path),
        heatmap_path=_to_url(record.heatmap_path),
        damage_score=int(record.damage_score),
        ssim_score=record.ssim_score,
        status=record.status.value,
        inspection_type=record.inspection_type.value,
        description=record.description or "",
        detections_json=record.detections_json,
        created_at=record.created_at,
        created_by=record.created_by,
    )


@router.get("", response_model=list[ArtifactRead])
def list_artifacts(
    status: str | None = None,
    db: Session = Depends(get_db),
) -> list[ArtifactRead]:
    query = db.query(Artifact).order_by(Artifact.created_at.desc())
    if status:
        query = query.filter(Artifact.status == status)
    return [_serialize(a) for a in query.all()]


@router.post("", response_model=ArtifactRead, status_code=201)
def create_artifact(
    payload: ArtifactCreate,
    db: Session = Depends(get_db),
) -> ArtifactRead:
    existing = db.query(Artifact).filter(Artifact.name == payload.name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Artifact '{payload.name}' already exists.")

    artifact = Artifact(
        name=payload.name,
        description=payload.description,
        location=payload.location,
        status=payload.status,
        inspection_interval_days=payload.inspection_interval_days,
    )
    db.add(artifact)
    db.flush()

    if payload.scheduled_date:
        schedule = Schedule(
            artifact_id=artifact.artifact_id,
            scheduled_date=payload.scheduled_date,
            scheduled_time=payload.scheduled_time or "09:00",
            notes="Initial schedule."
        )
        db.add(schedule)

    db.commit()
    db.refresh(artifact)
    return _serialize(artifact)


@router.post("/{artifact_id}/inspect-from-device", response_model=InspectionRead)
def inspect_artifact_from_device(
    artifact_id: str,
    device_id: str,
    description: str = "",
    created_by: str = "",
    db: Session = Depends(get_db),
    container: AppContainer = Depends(get_container),
) -> InspectionRead:
    """Run AI inspection on the most recent image captured by the device for this artifact."""
    artifact = db.query(Artifact).filter(Artifact.artifact_id == artifact_id).first()
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    metadata = container.command_service.get_latest_capture_metadata(device_id)
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="No capture metadata for this device. Run alignment first.",
        )

    full_path_str = metadata.get("final_aligned_path") or metadata.get("saved_file_full_path")
    if not full_path_str:
        raise HTTPException(
            status_code=404,
            detail="No saved image path in device metadata. Run alignment first.",
        )

    image_path = Path(full_path_str)
    try:
        image_path = image_path.resolve()
        image_path.relative_to(container.settings.uploads_dir.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Capture path is outside private media storage") from exc
    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Image file not found on server: {image_path.name}",
        )

    try:
        image_bytes = image_path.read_bytes()
        record = container.inspection_service.run_artifact_inspection(
            db=db,
            artifact=artifact,
            image_bytes=image_bytes,
            original_filename=image_path.name,
            description=description or f"Device workflow: {device_id}",
            inspection_type=InspectionType.sudden,
            created_by=created_by or None,
        )
        return _serialize_comparison(record)
    except Exception as e:
        logger.error(f"inspect-from-device failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/{artifact_id}/inspect", response_model=InspectionRead)
async def inspect_artifact(
    artifact_id: str,
    file: UploadFile = File(...),
    description: str = Form(default=""),
    created_by: str = Form(default=""),
    inspection_type: str = Form(default="sudden"),
    schedule_id: str | None = Form(default=None),
    db: Session = Depends(get_db),
    container: AppContainer = Depends(get_container),
) -> InspectionRead:
    artifact = db.query(Artifact).filter(Artifact.artifact_id == artifact_id).first()
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    try:
        image_bytes = await file.read()
        itype = InspectionType.scheduled if inspection_type == "scheduled" else InspectionType.sudden

        record = container.inspection_service.run_artifact_inspection(
            db=db,
            artifact=artifact,
            image_bytes=image_bytes,
            original_filename=file.filename or "upload.jpg",
            description=description,
            inspection_type=itype,
            schedule_id=schedule_id,
            created_by=created_by or None,
        )

        # LOGIC MỚI: Tự động tạo lịch tiếp theo nếu có chu kỳ nhắc lại
        if artifact.inspection_interval_days > 0:
            next_date = datetime.now(timezone.utc) + timedelta(days=artifact.inspection_interval_days)
            new_schedule = Schedule(
                artifact_id=artifact.artifact_id,
                scheduled_date=next_date,
                scheduled_time="09:00",
                notes=f"Tự động tạo: Chu kỳ {artifact.inspection_interval_days} ngày."
            )
            db.add(new_schedule)
            db.commit()

        return _serialize_comparison(record)
    except Exception as e:
        logger.error(f"Inspection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/{artifact_id}", response_model=ArtifactRead)
def get_artifact(artifact_id: str, db: Session = Depends(get_db)) -> ArtifactRead:
    artifact = db.query(Artifact).filter(Artifact.artifact_id == artifact_id).first()
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return _serialize(artifact)


@router.patch("/{artifact_id}", response_model=ArtifactRead)
def update_artifact(
    artifact_id: str,
    payload: ArtifactUpdate,
    db: Session = Depends(get_db),
) -> ArtifactRead:
    artifact = db.query(Artifact).filter(Artifact.artifact_id == artifact_id).first()
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        if key == "id": continue
        setattr(artifact, key, value)

    db.commit()
    db.refresh(artifact)
    return _serialize(artifact)


@router.delete("/{artifact_id}", status_code=204)
def delete_artifact(artifact_id: str, db: Session = Depends(get_db)) -> None:
    artifact = db.query(Artifact).filter(Artifact.artifact_id == artifact_id).first()
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    db.delete(artifact)
    db.commit()


@router.post("/{artifact_id}/reference", response_model=ArtifactRead)
async def upload_reference_image(
    artifact_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    container: AppContainer = Depends(get_container),
) -> ArtifactRead:
    artifact = db.query(Artifact).filter(Artifact.artifact_id == artifact_id).first()
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    new_image_obj = await container.inspection_service.save_reference_image(
        artifact_id=artifact_id,
        file=file,
    )
    
    db.add(new_image_obj)
    db.flush()

    artifact.baseline_image_id = new_image_obj.image_id
    db.commit()
    db.refresh(artifact)
    return _serialize(artifact)


@router.get("/{artifact_id}/inspections", response_model=InspectionListResponse)
def list_artifact_inspections(
    artifact_id: str,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> InspectionListResponse:
    items = (
        db.query(ImageComparison)
        .filter(ImageComparison.artifact_id == artifact_id)
        .order_by(ImageComparison.created_at.desc())
        .limit(limit)
        .all()
    )
    total = db.query(ImageComparison).filter(ImageComparison.artifact_id == artifact_id).count()
    return InspectionListResponse(
        items=[_serialize_comparison(item) for item in items],
        total=total,
    )
