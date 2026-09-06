from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.api.dependencies import get_current_user
from app.core.config import get_settings
from app.models.user import User

router = APIRouter()


@router.get("/uploads/{file_path:path}", response_class=FileResponse)
def get_private_media(file_path: str, _: User = Depends(get_current_user)) -> FileResponse:
    """Serve media only to authenticated users and never outside DATA_DIR."""
    root = get_settings().uploads_dir.resolve()
    candidate = (root / file_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Media not found") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Media not found")
    return FileResponse(candidate)
