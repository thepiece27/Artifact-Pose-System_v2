from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.dependencies import get_container, get_current_user, require_admin
from app.core.uploads import validate_image_bytes
from app.models.user import User
from app.schemas.models import (
    ModelDetectResponse,
    ModelInfo,
    ModelLoadRequest,
    ModelLoadResponse,
    ModelPredictRequest,
    ModelPredictResponse,
)
from app.services.state import AppContainer

router = APIRouter()


@router.get("/models", response_model=list[ModelInfo])
def list_models(
    container: AppContainer = Depends(get_container),
    _: User = Depends(get_current_user),
) -> list[ModelInfo]:
    models = container.model_service.list_models()
    return [
        ModelInfo(
            name=item.name,
            backend=item.backend,
            path=item.path,
            labels=item.labels,
            loaded_at=item.loaded_at,
        )
        for item in models
    ]


@router.post("/models/load", response_model=ModelLoadResponse)
def load_model(
    req: ModelLoadRequest,
    container: AppContainer = Depends(get_container),
    _: User = Depends(require_admin),
) -> ModelLoadResponse:
    try:
        loaded = container.model_service.load_model(
            name=req.name,
            path=req.path,
            backend=req.backend,
            labels=req.labels,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ModelLoadResponse(
        ok=True,
        model=ModelInfo(
            name=loaded.name,
            backend=loaded.backend,
            path=loaded.path,
            labels=loaded.labels,
            loaded_at=loaded.loaded_at,
        ),
    )


@router.delete("/models/{name}")
def unload_model(
    name: str,
    container: AppContainer = Depends(get_container),
    _: User = Depends(require_admin),
) -> dict[str, bool]:
    removed = container.model_service.unload_model(name)
    return {"ok": removed}


@router.post("/models/{name}/predict", response_model=ModelPredictResponse)
def predict(
    name: str,
    req: ModelPredictRequest,
    container: AppContainer = Depends(get_container),
    _: User = Depends(get_current_user),
) -> ModelPredictResponse:
    try:
        output = container.model_service.predict(name, req.input_data)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ModelPredictResponse(
        ok=True,
        model_name=name,
        output=output,
    )


@router.post("/models/{name}/detect", response_model=ModelDetectResponse)
async def detect_image(
    name: str,
    file: UploadFile = File(...),
    container: AppContainer = Depends(get_container),
    _: User = Depends(get_current_user),
) -> ModelDetectResponse:
    image_bytes = await file.read(container.settings.max_upload_bytes + 1)
    try:
        validate_image_bytes(
            image_bytes,
            max_bytes=container.settings.max_upload_bytes,
            max_pixels=container.settings.max_image_pixels,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        output = container.model_service.detect_image(name, image_bytes)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ModelDetectResponse(
        ok=True,
        model_name=name,
        output=output,
    )
