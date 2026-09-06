from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.dependencies import get_container, require_device_access
from app.schemas.inspection import InspectionUploadResponse
from app.services.state import AppContainer

router = APIRouter()


@router.post("/inspections/upload", response_model=InspectionUploadResponse)
async def upload_inspection(
    file: UploadFile = File(...),
    metadata: str = Form(...),
    container: AppContainer = Depends(get_container),
    _auth=Depends(require_device_access),
) -> InspectionUploadResponse:
    try:
        result = await container.inspection_service.handle_upload(file, metadata)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return InspectionUploadResponse(**result)
