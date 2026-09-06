from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np


_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def safe_component(value: str, *, field: str, max_length: int = 96) -> str:
    raw = str(value or "").strip()
    if not raw or raw in {".", ".."}:
        raise ValueError(f"Invalid {field}")
    normalized = _SAFE_COMPONENT.sub("-", raw).strip(".-")[:max_length]
    if not normalized:
        raise ValueError(f"Invalid {field}")
    return normalized


def validate_image_bytes(content: bytes, *, max_bytes: int, max_pixels: int) -> tuple[int, int]:
    if not content:
        raise ValueError("Empty upload")
    if len(content) > max_bytes:
        raise ValueError(f"Upload exceeds {max_bytes} bytes")
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0 or image.ndim not in (2, 3):
        raise ValueError("Upload is not a valid image")
    height, width = int(image.shape[0]), int(image.shape[1])
    if width <= 0 or height <= 0 or width * height > max_pixels:
        raise ValueError("Image exceeds configured pixel limit")
    return width, height


def safe_upload_path(root: Path, *components: str) -> Path:
    path = root.resolve()
    for component in components:
        path /= safe_component(component, field="path component")
    return path
