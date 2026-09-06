from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.uploads import safe_component, safe_upload_path, validate_image_bytes


def test_safe_component_rejects_empty_and_normalizes() -> None:
    with pytest.raises(ValueError):
        safe_component("..", field="artifact_id")
    assert safe_component("../artifact 01", field="artifact_id") == "artifact-01"


def test_safe_upload_path_stays_under_root(tmp_path: Path) -> None:
    path = safe_upload_path(tmp_path, "artifact-01", "image.png")
    assert path.parent == (tmp_path / "artifact-01").resolve()
    escaped = safe_upload_path(tmp_path, "../../outside")
    assert escaped.is_relative_to(tmp_path.resolve())


def test_validate_image_bytes_rejects_invalid_and_accepts_png() -> None:
    with pytest.raises(ValueError):
        validate_image_bytes(b"not an image", max_bytes=1024, max_pixels=10000)

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    assert validate_image_bytes(encoded.tobytes(), max_bytes=1024, max_pixels=10000) == (8, 8)
