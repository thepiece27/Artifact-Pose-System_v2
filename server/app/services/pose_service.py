from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from typing import Any

import cv2
import numpy as np

from app.modules.artifact_pose import common as pose_common
from app.modules.artifact_pose.geometry import DEFAULT_QUALITY
from app.modules.artifact_pose import correction as pose_correction
from app.modules.artifact_pose import initialize as pose_initialize
from app.core.config import Settings
from app.core.uploads import safe_component


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


class PoseService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._module_root = Path(__file__).resolve().parents[1] / "modules" / "artifact_pose"

    def _quality(self):
        return replace(
            DEFAULT_QUALITY,
            translation_tolerance_m=self._settings.trans_tolerance_mm / 1000.0,
            rotation_tolerance_deg=self._settings.rot_tolerance_deg,
        )

    # -- Per-artifact golden pose path -----------------------------------------

    def _golden_pose_path(self, artifact_id: str | None) -> Path:
        if artifact_id and artifact_id.strip():
            # Luu trong uploads/golden_poses/{id}/ de nam trong Docker volume da mount.
            return (
                self._settings.uploads_dir
                / "golden_poses"
                / safe_component(artifact_id, field="artifact_id")
                / "golden_pose.yaml"
            )
        return self._settings.artifact_golden_pose

    def _resolve_golden_pose_path(self, artifact_id: str | None) -> Path | None:
        per_artifact = self._golden_pose_path(artifact_id)
        if per_artifact.exists():
            return per_artifact

        global_path = self._settings.artifact_golden_pose

        old_data_path = (
            self._settings.data_dir / "golden_poses" / safe_component(artifact_id, field="artifact_id") / "golden_pose.yaml"
            if artifact_id and artifact_id.strip() else None
        )
        source = None
        if old_data_path and old_data_path.exists():
            source = old_data_path
        elif artifact_id and artifact_id.strip() and global_path.exists():
            source = global_path

        if source is not None:

            per_artifact.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(str(source), str(per_artifact))
            # Copy descriptors .npy neu co
            desc_src = source.with_name(source.stem + "_descriptors.npy")
            if desc_src.exists():
                shutil.copy2(str(desc_src), str(per_artifact.parent / desc_src.name))
            import logging as _log
            _log.getLogger(__name__).info(
                "[pose] Migrated golden_pose %s -> %s", source, per_artifact
            )
            return per_artifact

        return None

    def has_golden_pose(self, artifact_id: str) -> bool:
        return self._resolve_golden_pose_path(artifact_id) is not None

    # -- Health ----------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        camera_exists = self._settings.artifact_camera_params.exists()
        g2o_status = "enabled" if pose_common.HAS_CPP else "fallback_only"
        lens_position = pose_common.load_camera_lens_position(
            str(self._settings.artifact_camera_params)
        )
        if camera_exists:
            message = f"Integrated Artifact-Pose module ready ({g2o_status})"
        else:
            message = (
                f"Integrated module loaded but camera params missing: "
                f"{self._settings.artifact_camera_params}"
            )
        return {
            "ok": True,
            "available": camera_exists,
            "artifact_pose_root": str(self._module_root),
            "camera_params_dir": str(self._settings.artifact_camera_params_dir),
            "camera_params": str(self._settings.artifact_camera_params),
            "configured_lens_position": self._settings.artifact_lens_position,
            "camera_lens_position": lens_position,
            "golden_pose": str(self._settings.artifact_golden_pose),
            "message": message,
        }

    # -- Pose correction -------------------------------------------------------

    def correct_image(
        self, image_path: Path, artifact_id: str | None = None
    ) -> dict[str, Any]:
        K, D = pose_common.load_camera_params(self._settings.artifact_camera_params)
        if K is None:
            raise RuntimeError(
                f"Camera params not found: {self._settings.artifact_camera_params}"
            )
        golden_pose_path = self._resolve_golden_pose_path(artifact_id)
        if golden_pose_path is None:
            raise RuntimeError(
                f"Golden pose not found for artifact '{artifact_id}'"
            )
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Can not read image: {image_path}")
        golden_pose = pose_common.load_golden_pose(
            golden_pose_path, K=K, D=D,
            image_size=(int(image.shape[1]), int(image.shape[0]))
        )
        result = pose_correction.run_correction_step(
            image, K, D, golden_pose, backend="auto", quality=self._quality()
        )
        result["integrated_module"] = True
        result["g2o_enabled"] = bool(pose_common.HAS_CPP)
        return _to_jsonable(result)

    # -- Golden pose initialization --------------------------------------------

    def initialize_golden(
        self,
        left_image_path: Path,
        right_image_path: Path,
        artifact_id: str | None = None,
    ) -> dict[str, Any]:
        K, D = pose_common.load_camera_params(self._settings.artifact_camera_params)
        if K is None:
            raise RuntimeError(
                f"Camera params not found: {self._settings.artifact_camera_params}"
            )
        left = cv2.imread(str(left_image_path))
        right = cv2.imread(str(right_image_path))
        if left is None or right is None:
            raise RuntimeError("Can not read left/right image for initialization")
        output_path = self._golden_pose_path(artifact_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = pose_initialize.run_initialization(
            left, right, K, D, output=output_path, strategy="hybrid", backend="auto",
            quality=self._quality(),
        )
        if result is None:
            raise RuntimeError("Golden initialization failed")
        return _to_jsonable(result)
