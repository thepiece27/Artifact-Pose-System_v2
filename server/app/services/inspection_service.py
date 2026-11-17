from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from typing import Any
from threading import Lock

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.uploads import safe_component, validate_image_bytes
from app.models.artifact import (
    Artifact, Image, ImageComparison, ImageType, 
    ComparisonStatus, Alert, AlertLevel, InspectionType, Schedule
)
from app.services.command_service import CommandService
from app.services.model_service import ModelService
from app.services.mqtt_bridge import MqttBridge
from app.services.pose_service import PoseService
from app.modules.change_detection import legacy as legacy_inspection
from app.modules.change_detection.legacy import analyze_legacy

logger = logging.getLogger(__name__)

class InspectionService:
    def __init__(
        self,
        settings: Settings,
        pose_service: PoseService,
        model_service: ModelService,
        command_service: CommandService,
        mqtt_bridge: MqttBridge,
        profile: Any = None,
    ) -> None:
        self._settings = settings
        # CITA profile (None / pbl_legacy = cita-baseline behaviour).
        self._profile = profile
        self._pose_service = pose_service
        self._model_service = model_service
        self._command_service = command_service
        self._mqtt_bridge = mqtt_bridge
        self._alignment_counters: dict[str, int] = {}
        self._alignment_start_ts: dict[str, float] = {}
        # Phase per alignment session: 0 = Translation (X,Z steppers first)
        #                              1 = Rotation (Pan,Tilt servos)
        # Always starts at 0 and alternates each dispatched move.
        self._alignment_phase: dict[str, int] = {}
        self._alignment_state_file = settings.data_dir / "alignment_state.json"
        self._alignment_lock = Lock()
        self._load_alignment_state()

    def _load_alignment_state(self) -> None:
        try:
            if not self._alignment_state_file.exists():
                return
            payload = json.loads(self._alignment_state_file.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return
            self._alignment_counters.update({str(k): int(v) for k, v in (payload.get("counters") or {}).items()})
            self._alignment_start_ts.update({str(k): float(v) for k, v in (payload.get("start_ts") or {}).items()})
            self._alignment_phase.update({str(k): int(v) for k, v in (payload.get("phase") or {}).items()})
            cita = payload.get("cita")
            if isinstance(cita, dict):
                self._cita_sessions.update({str(k): v for k, v in cita.items() if isinstance(v, dict)})
        except Exception:
            return

    def _persist_alignment_state(self) -> None:
        with self._alignment_lock:
            try:
                self._alignment_state_file.parent.mkdir(parents=True, exist_ok=True)
                temp = self._alignment_state_file.with_suffix(".tmp")
                temp.write_text(json.dumps({
                    "counters": self._alignment_counters,
                    "start_ts": self._alignment_start_ts,
                    "phase": self._alignment_phase,
                    "cita": self._cita_sessions,
                }, default=str), encoding="utf-8")
                temp.replace(self._alignment_state_file)
            except Exception:
                return

    async def _save_file(
        self, file: UploadFile, artifact_id: str | None = None
    ) -> tuple[Path, int]:
        ts_ms = int(time.time() * 1000)
        safe_name = safe_component(Path(file.filename or "upload.bin").name, field="filename", max_length=120)
        if artifact_id and artifact_id.strip():
            target_dir = self._artifact_uploads_dir / safe_component(artifact_id, field="artifact_id")
        else:
            target_dir = self._settings.uploads_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{ts_ms}_{safe_name}"
        content = await file.read(self._settings.max_upload_bytes + 1)
        if len(content) > self._settings.max_upload_bytes:
            raise ValueError(f"Upload exceeds {self._settings.max_upload_bytes} bytes")
        if not content:
            raise ValueError("Empty upload")
        validate_image_bytes(content, max_bytes=self._settings.max_upload_bytes,
                             max_pixels=self._settings.max_image_pixels)
        target_path.write_bytes(content)
        return target_path, len(content)

    @property
    def _artifact_uploads_dir(self) -> Path:
        return self._settings.uploads_dir / "artifacts"

    async def save_reference_image(self, artifact_id: str, file: UploadFile, operator_id: str | None = None) -> Image:
        target_dir = self._artifact_uploads_dir / safe_component(artifact_id, field="artifact_id")
        target_dir.mkdir(parents=True, exist_ok=True)
        ts_ms = int(time.time() * 1000)
        safe_name = safe_component(Path(file.filename or "reference.jpg").name, field="filename", max_length=120)
        target_path = target_dir / f"reference_{ts_ms}_{safe_name}"
        content = await file.read(self._settings.max_upload_bytes + 1)
        validate_image_bytes(content, max_bytes=self._settings.max_upload_bytes,
                             max_pixels=self._settings.max_image_pixels)
        target_path.write_bytes(content)
        return Image(
            artifact_id=artifact_id,
            operator_id=operator_id,
            image_type=ImageType.baseline,
            image_path=str(target_path),
            is_valid=True
        )

    def run_artifact_inspection(
        self,
        *,
        db: Session,
        artifact: Artifact,
        image_bytes: bytes,
        original_filename: str,
        description: str = "",
        operator_id: str | None = None,
        device_id: str | None = None,
        inspection_type: InspectionType = InspectionType.sudden,
        schedule_id: str | None = None,
        created_by: str | None = None,
    ) -> ImageComparison:
        target_dir = self._artifact_uploads_dir / safe_component(str(artifact.artifact_id), field="artifact_id")
        target_dir.mkdir(parents=True, exist_ok=True)
        ts_ms = int(time.time() * 1000)
        safe_name = safe_component(Path(original_filename).name, field="filename", max_length=120)
        validate_image_bytes(image_bytes, max_bytes=self._settings.max_upload_bytes,
                             max_pixels=self._settings.max_image_pixels)
        current_path = target_dir / f"inspection_{ts_ms}_{safe_name}"
        current_path.write_bytes(image_bytes)

        current_image = Image(
            artifact_id=artifact.artifact_id,
            device_id=device_id,
            operator_id=operator_id,
            image_type=ImageType.inspection,
            image_path=str(current_path),
            is_valid=True
        )
        db.add(current_image)
        db.flush()

        reference_path = None
        previous_image_id = None
        if artifact.baseline_image:
            reference_path = Path(artifact.baseline_image.image_path)
            previous_image_id = artifact.baseline_image.image_id
        
        analysis = self._analyze_against_reference(
            current_path=current_path,
            reference_path=reference_path,
            artifact_id=artifact.artifact_id,
            ts_ms=ts_ms,
        )

        damage_score = 0.0
        all_dets = analysis.get("all_detections") or []
        ssim_val = analysis.get("ssim")
        damage_pct = analysis.get("damage_pct")
        status = self._classify_damage_status(all_dets, ssim_val, damage_pct)
        if reference_path is None:
            # Legacy schema requires previous_image_id, so keep the current
            # image as a compatibility placeholder but explicitly mark this
            # as baseline-missing. It must never be presented as a clean
            # comparison result.
            status = ComparisonStatus.warning
            analysis["analysis_status"] = "baseline_missing"
            analysis["analysis_error"] = "baseline_image_not_configured"
            analysis["detections_json"] = json.dumps({
                "analysis_status": "baseline_missing",
                "analysis_error": "baseline_image_not_configured",
                "detections": all_dets,
            })
        # A resize-only fallback is a degraded comparison, not evidence of a
        # clean artifact.  Preserve damage escalation but downgrade a would-be
        # good result to the conservative warning state.
        if (
            status == ComparisonStatus.good
            and analysis.get("alignment_quality") == "resize_fallback"
        ):
            status = ComparisonStatus.warning
        if analysis.get("analysis_status") == "failed" and status == ComparisonStatus.good:
            status = ComparisonStatus.warning

        comparison = ImageComparison(
            artifact_id=artifact.artifact_id,
            previous_image_id=previous_image_id,
            current_image_id=current_image.image_id,
            schedule_id=schedule_id,
            damage_score=round(damage_pct or 0.0, 2),
            ssim_score=(f"{ssim_val:.4f}" if ssim_val is not None else None),
            heatmap_path=analysis.get("heatmap_path"),
            status=status,
            inspection_type=inspection_type,
            description=description or analysis.get("auto_description", ""),
            detections_json=analysis.get("detections_json"),
            created_by=(created_by or "").strip() or None,
        )
        db.add(comparison)
        db.flush()  # Ensure comparison_id is generated before referencing it in Alert

        if schedule_id:
            sched = db.query(Schedule).filter(Schedule.id == schedule_id).first()
            if sched:
                sched.completed = True

        if status in [ComparisonStatus.warning, ComparisonStatus.damaged]:
            alert_level = AlertLevel.high if status == ComparisonStatus.damaged else AlertLevel.medium
            alert = Alert(
                artifact_id=artifact.artifact_id,
                comparison_id=comparison.comparison_id,
                alert_level=alert_level,
                is_handled=False
            )
            db.add(alert)

        artifact.status = self._merge_artifact_status(artifact.status, status.value)
        db.commit()
        db.refresh(comparison)
        return comparison

    @staticmethod
    def _classify_damage_status(
        detections: list[dict],
        ssim: float | None,
        damage_pct: float | None,
    ) -> ComparisonStatus:
        """Combine SSIM structural similarity with YOLO detections to classify status.

        Priority:
        - If SSIM is available, use it as the primary metric.
        - YOLO HIGH-confidence detections can escalate the status.
        - Without SSIM, fall back to YOLO confidence only.
        """
        # No usable signal is a failed/unknown analysis, never a clean result.
        # Keep the legacy enum contract and surface the detailed failure in
        # detections_json until an explicit UNKNOWN enum is added.
        if ssim is None and not detections:
            return ComparisonStatus.warning

        # Determine SSIM-based grade
        ssim_status: ComparisonStatus | None = None
        if ssim is not None:
            dpct = damage_pct or 0.0
            if ssim >= 0.95 and dpct < 2.0:
                ssim_status = ComparisonStatus.good
            elif ssim >= 0.85 and dpct < 10.0:
                ssim_status = ComparisonStatus.warning
            else:
                ssim_status = ComparisonStatus.damaged

        # Determine YOLO-based grade
        yolo_status: ComparisonStatus | None = None
        if detections:
            max_conf = max(float(d.get("confidence", 0)) for d in detections)
            if max_conf >= 0.65:
                yolo_status = ComparisonStatus.damaged
            elif max_conf >= 0.40:
                yolo_status = ComparisonStatus.warning
            else:
                yolo_status = ComparisonStatus.good

        # Merge: take the worst result from both signals
        _rank = {ComparisonStatus.good: 0, ComparisonStatus.warning: 1, ComparisonStatus.damaged: 2}
        candidates = [s for s in (ssim_status, yolo_status) if s is not None]
        if not candidates:
            # Missing/failed analysis must never become a false negative.
            # Until the schema gains an explicit UNKNOWN enum, warning is the
            # conservative compatible representation.
            return ComparisonStatus.warning
        return max(candidates, key=lambda s: _rank[s])

    @staticmethod
    def _merge_artifact_status(current: str, new_status: str) -> str:
        priority = {"good": 0, "archived": 0, "need_check": 1, "maintenance": 1, "warning": 2, "damaged": 3}
        cur_p = priority.get(current, 0)
        new_p = priority.get(new_status, 0)
        return new_status if new_p > cur_p else current

    # ── Pipeline tuning — edit here to adjust behaviour ──────────────────────
    _YOLO_CONF        = 0.15   # detection confidence threshold
    _YOLO_SUB_BATCH   = 4      # crops per model.predict() call (OOM guard)
    _MIN_AREA_FACTOR  = 0.0001 # contour area filter: max(500, h*w*factor) px²
    _DAMAGE_PCT_GATE  = 0.5    # skip crop-YOLO if damage_pct < this %
    _OTSU_FALLBACK    = 60     # hard threshold when Otsu < 30
    _CONTOUR_CAP      = 50     # max contours to inspect
    _TIGHT_PAD_FACTOR = 0.30   # tight crop: padding as fraction of contour side
    _TIGHT_MIN_PAD    = 40     # tight crop: minimum padding in px
    _WIDE_PAD_FACTOR  = 0.70   # wide crop: padding as fraction of contour side
    _WIDE_MIN_PAD     = 120    # wide crop: minimum padding in px
    _NMS_IOU_THRESH   = 0.45   # NMS IoU threshold (tight vs wide de-dup)
    _EARLY_EXIT_CONF  = 0.40   # skip wide crop if tight already >= this conf
    # ─────────────────────────────────────────────────────────────────────────

    def _analyze_against_reference(self, *, current_path: Path, reference_path: Path | None, artifact_id: str, ts_ms: int) -> dict[str, Any]:
        """Same signature and result keys as cita-baseline.

        P0 (default, pbl_legacy) runs the original code (moved verbatim to
        ``change_detection.legacy``); P1-P4 run the CITA pipelines selected by
        the active profile and add the optional keys ``pipeline``,
        ``roi_afd_px``, ``ecc_ok``, ``ai_input_package_path``.
        """
        pipeline = getattr(getattr(self._profile, "inspection", None), "pipeline", "P0")
        if pipeline == "P0":
            return analyze_legacy(
                current_path=current_path, reference_path=reference_path, artifact_id=artifact_id,
                ts_ms=ts_ms, model_service=self._model_service,
                model_name=self._settings.default_ai_model_name,
                uploads_artifacts_dir=self._artifact_uploads_dir,
            )
        from app.modules.change_detection.pipelines import run_server_pipeline
        return run_server_pipeline(
            self._profile, current_path=current_path, reference_path=reference_path,
            artifact_id=artifact_id, ts_ms=ts_ms, model_service=self._model_service,
            model_name=self._settings.default_ai_model_name,
            uploads_artifacts_dir=self._artifact_uploads_dir,
            golden_dir=self._settings.uploads_dir / "golden_poses",
        )

    # Thin delegates kept for existing callers (tools, tests).
    _nms_detections = staticmethod(legacy_inspection.nms_detections)
    _sift_align_with_mask = staticmethod(legacy_inspection.sift_align_with_mask)
    _compute_region_ssim = staticmethod(legacy_inspection.compute_region_ssim)

    def _run_detector_on_path(self, path: Path) -> dict[str, Any]:
        """Run the configured detector with an explicit failure contract.

        In particular, a missing/unloaded model is reported as ``failed`` and
        never represented as an empty successful detection list.
        """
        try:
            raw = self._model_service.detect_image(
                self._settings.default_ai_model_name,
                path.read_bytes(),
            )
            detections: list[dict[str, Any]] = []
            for item in raw or []:
                if isinstance(item, dict):
                    values = item.get("detections", [])
                    if isinstance(values, list):
                        detections.extend(values)
            return {
                "status": "ok",
                "detections": detections,
                "model_name": self._settings.default_ai_model_name,
            }
        except Exception as exc:
            logger.warning("[ai] detector failed for %s: %s", path.name, exc)
            return {
                "status": "failed",
                "error_code": f"detector_error:{type(exc).__name__}",
                "detections": [],
                "model_name": self._settings.default_ai_model_name,
            }

    async def handle_upload(self, file: UploadFile, metadata_str: str) -> dict[str, Any]:
        """Save image uploaded by device agent, run pose correction, record latest metadata."""
        import json as _json
        if len(metadata_str.encode("utf-8")) > self._settings.max_metadata_bytes:
            raise ValueError("Metadata exceeds configured limit")
        try:
            meta = _json.loads(metadata_str)
        except Exception as exc:
            raise ValueError(f"Invalid metadata JSON: {exc}") from exc
        if not isinstance(meta, dict):
            raise ValueError("Metadata must be a JSON object")

        device_id = str(meta.get("device_id", ""))
        artifact_id = str(meta.get("artifact_id", ""))
        calibration_data = meta.get("calibration_data", {})

        saved_path, size_bytes = await self._save_file(file, artifact_id=artifact_id or None)

        # Record metadata so the latest capture can be retrieved later
        capture_metadata: dict[str, Any] = {
            "saved_file": saved_path.name,
            "saved_file_full_path": str(saved_path),
            "device_id": device_id,
            "artifact_id": artifact_id,
        }
        if isinstance(calibration_data, dict):
            capture_metadata.update(calibration_data)

        self._command_service.record_latest_capture_metadata(device_id, capture_metadata)

        ai_result: dict[str, Any] | None = None
        if self._settings.run_ai_on_upload:
            ai_result = self._run_detector_on_path(saved_path)
            capture_metadata["ai_status"] = ai_result.get("status")
            self._command_service.record_latest_capture_metadata(device_id, capture_metadata)

        # Attempt pose correction (non-fatal if it fails)
        pose_result: dict[str, Any] | None = None
        correction_dispatch: dict[str, Any] | None = None
        workflow: dict[str, Any] = calibration_data.get("workflow", {}) if isinstance(calibration_data, dict) else {}
        auto_alignment_loop: bool = isinstance(workflow, dict) and bool(workflow.get("auto_alignment_loop", False))
        if auto_alignment_loop and not self._settings.run_pose_on_upload:
            # An alignment loop without pose estimation cannot make a safe
            # correction decision; ingest the frame but do not keep issuing
            # retry captures or motor commands.
            capture_metadata["alignment_status"] = "disabled_pose_flag"
            auto_alignment_loop = False

        # CITA experiment arms (workflow.cita_arm or a non-legacy control
        # profile) use the AlignmentSession state machine; everything else
        # follows the cita-baseline loop below unchanged.
        cita_profile = (self._cita_alignment_profile(workflow)
                        if auto_alignment_loop and device_id else None)
        if cita_profile is not None:
            return self._handle_cita_alignment(
                profile=cita_profile, saved_path=saved_path, size_bytes=size_bytes, device_id=device_id,
                artifact_id=artifact_id, workflow=workflow, calibration_data=calibration_data,
                capture_metadata=capture_metadata, ai_result=ai_result,
            )

        # ── Alignment iteration guard ────────────────────────────────────────
        # Each upload during an active alignment loop counts as one iteration.
        # Stop and notify if the limit is exceeded.
        alignment_key = f"{device_id}:{artifact_id}"
        if auto_alignment_loop and device_id:
            # Persisted sessions use wall-clock seconds so a restart does not
            # invalidate the monotonic epoch from the previous process.
            now_mono = time.time()
            # Start the deadline on the first capture of a session and enforce
            # it before any further pose/motor transition.
            session_start = self._alignment_start_ts.setdefault(alignment_key, now_mono)
            deadline = session_start + float(self._settings.alignment_timeout_sec)
            self._alignment_counters[alignment_key] = self._alignment_counters.get(alignment_key, 0) + 1
            current_iter = self._alignment_counters[alignment_key]
            # Phase 0 (translation) is always the starting phase for a fresh session.
            if alignment_key not in self._alignment_phase:
                self._alignment_phase[alignment_key] = 0
            self._persist_alignment_state()
            max_iter = self._settings.max_alignment_iterations
            capture_metadata["alignment_iteration"] = current_iter
            capture_metadata["alignment_max_iterations"] = max_iter
            self._command_service.record_latest_capture_metadata(device_id, capture_metadata)

            if now_mono >= deadline:
                reason = (
                    f"Alignment timeout after {self._settings.alignment_timeout_sec} seconds"
                )
                logger.warning(
                    "[alignment] Deadline exceeded for device=%s artifact=%s",
                    device_id, artifact_id,
                )
                capture_metadata["alignment_status"] = "timeout"
                capture_metadata["alignment_fail_reason"] = reason
                self._command_service.record_latest_capture_metadata(device_id, capture_metadata)
                self._alignment_counters.pop(alignment_key, None)
                self._alignment_start_ts.pop(alignment_key, None)
                self._alignment_phase.pop(alignment_key, None)
                self._persist_alignment_state()
                timeout_payload: dict[str, Any] = {
                    "action": "alignment_failed",
                    "task_id": self._command_service.build_task_id(),
                    "artifact_id": artifact_id,
                    "device_id": device_id,
                    "reason": reason,
                    "failure_code": "alignment_timeout",
                    "workflow": workflow,
                }
                if self._settings.auto_dispatch_pose_command:
                    self._mqtt_bridge.publish_command(device_id, timeout_payload)
                return {
                    "ok": True,
                    "message": reason,
                    "saved_file": saved_path.name,
                    "size_bytes": size_bytes,
                    "pose_result": None,
                    "correction_dispatch": {"status": "alignment_timeout", "reason": reason},
                    "ai_result": None,
                }

            if current_iter > max_iter:
                reason = (
                    f"Alignment did not converge after {max_iter} iterations. "
                    "Please verify the Diamond ArUco marker is clearly visible and well-lit, "
                    "then retry alignment."
                )
                logger.warning(
                    "[alignment] Max iterations exceeded for device=%s artifact=%s (iter=%d/%d)",
                    device_id, artifact_id, current_iter, max_iter,
                )
                capture_metadata["alignment_status"] = "failed"
                capture_metadata["alignment_fail_reason"] = reason
                self._command_service.record_latest_capture_metadata(device_id, capture_metadata)
                self._alignment_counters.pop(alignment_key, None)
                self._alignment_start_ts.pop(alignment_key, None)
                self._alignment_phase.pop(alignment_key, None)
                self._persist_alignment_state()
                failed_payload: dict[str, Any] = {
                    "action": "alignment_failed",
                    "task_id": self._command_service.build_task_id(),
                    "artifact_id": artifact_id,
                    "device_id": device_id,
                    "reason": reason,
                    "iteration": current_iter,
                    "workflow": workflow,
                }
                if self._settings.auto_dispatch_pose_command:
                    self._mqtt_bridge.publish_command(device_id, failed_payload)
                return {
                    "ok": True,
                    "message": f"Alignment stopped: {reason}",
                    "saved_file": saved_path.name,
                    "size_bytes": size_bytes,
                    "pose_result": None,
                    "correction_dispatch": {"status": "alignment_failed", "reason": reason},
                    "ai_result": None,
                }
        else:
            current_iter = 0

        try:
            if self._settings.run_pose_on_upload:
                pose_result = self._pose_service.correct_image(
                    saved_path, artifact_id=artifact_id or None
                )
            else:
                capture_metadata["pose_status"] = "skipped_by_config"
                self._command_service.record_latest_capture_metadata(device_id, capture_metadata)
            deviation = pose_result.get("deviation") if pose_result else None

            # Update metadata with pose deviation so Flutter can poll it live
            if deviation:
                capture_metadata["pose_deviation"] = deviation
                self._command_service.record_latest_capture_metadata(device_id, capture_metadata)

            if deviation and not deviation.get("within_tolerance", True):
                # ── Interleaved phase correction ────────────────────────────────
                # Phase 0 → Translation only (X, Z steppers).  Always runs first.
                # Phase 1 → Rotation only   (Pan, Tilt servos). Runs after steppers.
                # Alternates: 0 → 1 → 0 → 1 … until both axes are within tolerance.
                # Smart-skip: if current phase's axis is already converged, jump ahead.
                # Fixed mapping (C-A logic) lives in control.FixedMappingController;
                # the computation below is identical to cita-baseline.
                fixed_step = self._fixed_mapping_controller().legacy_step(
                    deviation, self._alignment_phase.get(alignment_key, 0)
                )
                raw_move_x, raw_move_z, raw_pan, raw_tilt = fixed_step["raw"]
                motor_cmd = fixed_step["motor_cmd"]
                if pose_result is not None:
                    pose_result["motor_command"] = motor_cmd
                deviation["motor_command"] = motor_cmd

                if device_id:

                    # Resolve active phase; default to 0 (translation) if somehow missing.
                    current_phase = self._alignment_phase.get(alignment_key, 0)

                    # Smart-skip: if this phase's axis is already within tolerance,
                    # advance immediately to the other phase (avoids a wasted round-trip).
                    if fixed_step["smart_skip"] is not None:
                        current_phase = fixed_step["phase_used"]
                        self._alignment_phase[alignment_key] = current_phase
                        self._persist_alignment_state()
                        logger.info(
                            "[alignment] %s within tolerance → skip to %s phase "
                            "(device=%s, iter=%d)",
                            "Trans" if current_phase == 1 else "Rot",
                            "ROTATION" if current_phase == 1 else "TRANSLATION",
                            device_id, current_iter,
                        )

                    fields = fixed_step["fields"]
                    if current_phase == 0:
                        # ── Phase 0: Translation — X, Z steppers only ──────────
                        mc_payload: dict[str, Any] = {
                            "action": "move",
                            "task_id": self._command_service.build_task_id(),
                            "artifact_id": artifact_id,
                            "x_steps": fields["x_steps"],
                            "x_dir":   fields["x_dir"],
                            "z_steps": fields["z_steps"],
                            "z_dir":   fields["z_dir"],
                            "yaw_delta":   0.0,   # servos held during translation
                            "pitch_delta": 0.0,
                            "alignment_phase": "translation",
                            "alignment_iteration": current_iter,
                            "workflow": workflow,
                        }
                        logger.info(
                            "[alignment] Phase 0 TRANSLATION | device=%s iter=%d/%d "
                            "x=%+.0f steps (%s)  z=%+.0f steps (%s)  [pan/tilt held at 0]",
                            device_id, current_iter, self._settings.max_alignment_iterations,
                            raw_move_x, "+" if raw_move_x >= 0 else "-",
                            raw_move_z, "+" if raw_move_z >= 0 else "-",
                        )
                    else:
                        # ── Phase 1: Rotation — Pan, Tilt servos only ──────────
                        mc_payload = {
                            "action": "move",
                            "task_id": self._command_service.build_task_id(),
                            "artifact_id": artifact_id,
                            "x_steps": 0,   # steppers held during rotation
                            "x_dir":   1,
                            "z_steps": 0,
                            "z_dir":   1,
                            "yaw_delta":   raw_pan,
                            "pitch_delta": raw_tilt,
                            "alignment_phase": "rotation",
                            "alignment_iteration": current_iter,
                            "workflow": workflow,
                        }
                        logger.info(
                            "[alignment] Phase 1 ROTATION    | device=%s iter=%d/%d "
                            "pan=%+.3f°  tilt=%+.3f°  [steppers held at 0]",
                            device_id, current_iter, self._settings.max_alignment_iterations,
                            raw_pan, raw_tilt,
                        )

                    # Advance to the next phase so the following iteration uses the other axis.
                    self._alignment_phase[alignment_key] = 1 - current_phase
                    self._persist_alignment_state()

                    if self._settings.auto_dispatch_pose_command:
                        published, result_info = self._mqtt_bridge.publish_command(device_id, mc_payload)
                    else:
                        published, result_info = False, {"reason": "auto_dispatch_disabled"}
                    correction_dispatch = {
                        "status": "published" if published else (
                            "disabled" if not self._settings.auto_dispatch_pose_command else "failed"
                        ),
                        "info": result_info,
                        "alignment_iteration": current_iter,
                        "alignment_phase": "translation" if current_phase == 0 else "rotation",
                    }
                    if auto_alignment_loop and not published:
                        capture_metadata["alignment_status"] = "dispatch_failed"
                        capture_metadata["alignment_fail_reason"] = str(result_info)
                        self._command_service.record_latest_capture_metadata(device_id, capture_metadata)
                        self._alignment_counters.pop(alignment_key, None)
                        self._alignment_start_ts.pop(alignment_key, None)
                        self._alignment_phase.pop(alignment_key, None)
                        self._persist_alignment_state()
                    if auto_alignment_loop and published:
                        phase_label = "trans" if current_phase == 0 else "rot"
                        capture_metadata["alignment_status"] = f"correcting_{phase_label}"
                        capture_metadata["alignment_phase"] = "translation" if current_phase == 0 else "rotation"
                        self._command_service.record_latest_capture_metadata(device_id, capture_metadata)
            elif auto_alignment_loop and device_id:
                if deviation is None:
                    # Diamond/marker not detected — retry capture (still within iteration budget)
                    logger.warning(
                        "[alignment] No Diamond ArUco detected for device=%s (iter=%d/%d), retrying",
                        device_id, current_iter, self._settings.max_alignment_iterations,
                    )
                    capture_metadata["alignment_status"] = "no_diamond"
                    self._command_service.record_latest_capture_metadata(device_id, capture_metadata)
                    retry_payload: dict[str, Any] = {
                        "action": "capture",
                        "task_id": self._command_service.build_task_id(),
                        "artifact_id": artifact_id,
                        "capture_job": calibration_data.get("capture_job", "alignment") if isinstance(calibration_data, dict) else "alignment",
                        "basename": f"align_retry_{artifact_id}_{int(time.time() * 1000)}",
                        "workflow": workflow,
                    }
                    if self._settings.auto_dispatch_pose_command:
                        published, result_info = self._mqtt_bridge.publish_command(device_id, retry_payload)
                    else:
                        published, result_info = False, {"reason": "auto_dispatch_disabled"}
                    correction_dispatch = {
                        "status": "retry_capture_published" if published else "retry_capture_failed",
                        "info": result_info,
                    }
                else:
                    # within_tolerance=True — alignment complete, save final aligned image
                    logger.info(
                        "[alignment] Pose within tolerance for device=%s artifact=%s (iter=%d)",
                        device_id, artifact_id, current_iter,
                    )
                    self._alignment_counters.pop(alignment_key, None)
                    self._alignment_start_ts.pop(alignment_key, None)
                    self._alignment_phase.pop(alignment_key, None)
                    self._persist_alignment_state()

                    # Copy last captured image to distinctive final_aligned filename
                    if artifact_id:
                        ts_final = int(time.time() * 1000)
                        final_dir = self._artifact_uploads_dir / safe_component(artifact_id, field="artifact_id")
                        final_dir.mkdir(parents=True, exist_ok=True)
                        final_path = final_dir / f"final_aligned_{artifact_id}_{ts_final}.png"
                        final_path.write_bytes(saved_path.read_bytes())
                        capture_metadata["final_aligned_path"] = str(final_path)
                        logger.info("[alignment] Saved final aligned image: %s", final_path.name)

                        if self._settings.run_ai_on_aligned_image:
                            ai_result = self._run_detector_on_path(final_path)
                            capture_metadata["ai_status"] = ai_result.get("status")

                    capture_metadata["alignment_status"] = "complete"
                    capture_metadata["alignment_total_iterations"] = current_iter
                    self._command_service.record_latest_capture_metadata(device_id, capture_metadata)

                    complete_payload: dict[str, Any] = {
                        "action": "alignment_complete",
                        "task_id": self._command_service.build_task_id(),
                        "artifact_id": artifact_id,
                        "device_id": device_id,
                        "deviation": deviation,
                        "total_iterations": current_iter,
                        "workflow": workflow,
                    }
                    if self._settings.auto_dispatch_pose_command:
                        published, result_info = self._mqtt_bridge.publish_command(device_id, complete_payload)
                    else:
                        published, result_info = False, {"reason": "auto_dispatch_disabled"}
                    correction_dispatch = {
                        "status": "alignment_complete_published" if published else "alignment_complete_failed",
                        "info": result_info,
                    }
        except Exception as exc:
            logger.warning("Pose correction skipped for device=%s: %s", device_id, exc)
            if auto_alignment_loop and device_id:
                reason = str(exc)
                capture_metadata["alignment_status"] = "failed"
                capture_metadata["alignment_fail_reason"] = reason
                self._command_service.record_latest_capture_metadata(device_id, capture_metadata)
                self._alignment_counters.pop(alignment_key, None)
                self._alignment_start_ts.pop(alignment_key, None)
                self._alignment_phase.pop(alignment_key, None)
                self._persist_alignment_state()
                # Notify device that alignment failed so it stops immediately
                exc_failed_payload: dict[str, Any] = {
                    "action": "alignment_failed",
                    "task_id": self._command_service.build_task_id(),
                    "artifact_id": artifact_id,
                    "device_id": device_id,
                    "reason": reason,
                    "workflow": workflow,
                }
                if self._settings.auto_dispatch_pose_command:
                    self._mqtt_bridge.publish_command(device_id, exc_failed_payload)

        return {
            "ok": True,
            "message": "Uploaded successfully",
            "saved_file": saved_path.name,
            "size_bytes": size_bytes,
            "pose_result": pose_result,
            "correction_dispatch": correction_dispatch,
            "ai_result": ai_result,
        }

    def reset_alignment_counter(self, device_id: str, artifact_id: str) -> None:
        alignment_key = f"{device_id}:{artifact_id}"
        self._alignment_counters.pop(alignment_key, None)
        self._alignment_start_ts.pop(alignment_key, None)
        self._alignment_phase.pop(alignment_key, None)
        self._cita_sessions.pop(alignment_key, None)
        self._persist_alignment_state()

    # ── CITA experiment alignment (Phase 3) ────────────────────────────────────

    def _fixed_mapping_controller(self):
        from app.modules.artifact_pose.control import FixedMappingController
        s = self._settings
        return FixedMappingController(steps_per_mm=s.steps_per_mm, sign_x=s.sign_move_x, sign_z=s.sign_move_z,
                                      sign_pan=s.sign_rotate_pan, sign_tilt=s.sign_rotate_tilt)

    @staticmethod
    def _profile_needs_cita_loop(profile: Any) -> bool:
        if profile is None:
            return False
        c = profile.control
        return bool(c.controller != "fixed_mapping" or c.max_iterations is not None
                    or c.tol_translation_mm is not None or c.tol_rotation_deg is not None
                    or c.signs is not None or profile.motion_payload() or profile.pose.frames > 1)

    def _cita_alignment_profile(self, workflow: Any) -> Any:
        arm = workflow.get("cita_arm") if isinstance(workflow, dict) else None
        if arm:
            from app.experiments.registry import load_arm_profile
            cache = self.__dict__.setdefault("_cita_profile_cache", {})
            if arm not in cache:
                cache[arm] = load_arm_profile(str(arm))
            return cache[arm]
        return self._profile if self._profile_needs_cita_loop(self._profile) else None

    @property
    def _cita_sessions(self) -> dict:
        return self.__dict__.setdefault("_cita_sessions_store", {})

    def _cita_logger(self):
        log = self.__dict__.get("_cita_trial_logger")
        if log is None:
            from datetime import datetime, timezone
            from app.experiments.trial_log import TrialLogger
            root = getattr(self._profile, "logging", None)
            run_id = "server_" + datetime.now(timezone.utc).strftime("%Y%m%d")
            log = TrialLogger("E2", root=root.root if root else "logs/experiments", run_id=run_id,
                              profile=self._profile, resume=True)
            self.__dict__["_cita_trial_logger"] = log
        return log

    def _handle_cita_alignment(self, *, profile: Any, saved_path: Path, size_bytes: int, device_id: str,
                               artifact_id: str, workflow: dict, calibration_data: Any,
                               capture_metadata: dict, ai_result: Any) -> dict[str, Any]:
        from app.modules.artifact_pose.alignment_loop import AlignmentSession, build_session
        from app.modules.artifact_pose.control import make_controller

        key = f"{device_id}:{artifact_id}"
        trial_id = str(workflow.get("cita_trial_id") or "")
        entry = self._cita_sessions.get(key)
        if entry is None or entry.get("trial_id") != trial_id or entry.get("arm") != profile.name:
            session = build_session(profile, self._settings, arm=profile.name, device_id=device_id,
                                    artifact_id=artifact_id)
            entry = {"trial_id": trial_id, "arm": profile.name, "frames": [], "t_start": time.time()}
        else:
            controller = make_controller(profile, self._settings, device_id=device_id, artifact_id=artifact_id)
            session = AlignmentSession.from_state(entry["session"], controller)
        entry["frames"].append(str(saved_path))
        capture_metadata.update({"cita_arm": profile.name, "cita_trial_id": trial_id})

        def publish(payload: dict) -> tuple[bool, Any]:
            if not self._settings.auto_dispatch_pose_command:
                return False, {"reason": "auto_dispatch_disabled"}
            return self._mqtt_bridge.publish_command(device_id, payload)

        def capture_payload(prefix: str) -> dict:
            return {"action": "capture", "task_id": self._command_service.build_task_id(),
                    "artifact_id": artifact_id, "capture_job": "alignment", "keep_pose": True,
                    "basename": f"{prefix}_{artifact_id}_{int(time.time() * 1000)}", "workflow": workflow}

        def save_entry(done: bool) -> None:
            if done:
                self._cita_sessions.pop(key, None)
            else:
                entry["session"] = session.to_state()
                self._cita_sessions[key] = entry
            self._persist_alignment_state()

        def respond(status: str, info: Any = None, pose_result: Any = None) -> dict[str, Any]:
            capture_metadata["alignment_status"] = status
            self._command_service.record_latest_capture_metadata(device_id, capture_metadata)
            return {"ok": True, "message": "Uploaded successfully", "saved_file": saved_path.name,
                    "size_bytes": size_bytes, "pose_result": pose_result,
                    "correction_dispatch": {"status": status, "info": info, "cita_arm": profile.name,
                                            "alignment_iteration": session.iteration},
                    "ai_result": ai_result}

        if time.time() - entry["t_start"] >= float(self._settings.alignment_timeout_sec):
            session._finish("fail", "timeout", {})
        else:
            frames_needed = int(profile.pose.frames)
            if len(entry["frames"]) < frames_needed:
                # M5: more static frames at the same pose (no homing).
                ok, info = publish(capture_payload("align_frame"))
                save_entry(False)
                return respond("collecting_frames" if ok else "dispatch_failed", info)
            frames = [Path(p) for p in entry["frames"][:frames_needed]]
            entry["frames"] = []
            try:
                pose_result = self._pose_service.correct_image(frames[0], artifact_id=artifact_id or None,
                                                               profile=profile, extra_image_paths=frames[1:])
            except Exception as exc:  # pose failure counts as an unusable observation
                logger.warning("[cita-alignment] pose failed: %s", exc)
                pose_result = {"accepted": False, "status": f"error:{type(exc).__name__}"}
            observation = pose_result.get("cita_pose") or pose_result
            if pose_result.get("deviation"):
                capture_metadata["pose_deviation"] = pose_result["deviation"]
            decision = session.on_observation(observation, extra={"frames": [p.name for p in frames]})
            try:
                self._cita_logger().log(arm=profile.name, input={"device_id": device_id, "artifact_id": artifact_id,
                                                                 "trial_id": trial_id, "frames": [str(p) for p in frames]},
                                        output={"iteration": decision.record}, status=decision.kind,
                                        error=decision.reason, record_type="iteration")
            except Exception as log_exc:  # logging must never break the device loop
                logger.warning("[cita-alignment] log failed: %s", log_exc)
            if decision.kind == "move":
                payload = {"action": "move", "task_id": self._command_service.build_task_id(),
                           "artifact_id": artifact_id, **decision.payload,
                           "alignment_iteration": session.iteration, "workflow": workflow}
                ok, info = publish(payload)
                if not ok:
                    session._finish("fail", "dispatch_failed", {})
                else:
                    save_entry(False)
                    return respond(f"cita_{decision.record.get('command', {}).get('kind', 'move')}", info, pose_result)
            elif decision.kind == "retry":
                ok, info = publish(capture_payload("align_retry"))
                save_entry(False)
                return respond("no_pose_retry" if ok else "dispatch_failed", info, pose_result)

        summary = session.summary()
        done = session.finished is not None and session.finished.kind == "done"
        if done and artifact_id:
            final_dir = self._artifact_uploads_dir / safe_component(artifact_id, field="artifact_id")
            final_dir.mkdir(parents=True, exist_ok=True)
            final_path = final_dir / f"final_aligned_{artifact_id}_{int(time.time() * 1000)}.png"
            final_path.write_bytes(saved_path.read_bytes())
            capture_metadata["final_aligned_path"] = str(final_path)
            summary["final_aligned_path"] = str(final_path)
        capture_metadata.update(alignment_total_iterations=session.iteration, cita_summary=summary)
        try:
            self._cita_logger().log(arm=profile.name, input={"device_id": device_id, "artifact_id": artifact_id,
                                                             "trial_id": trial_id},
                                    output=summary, status="success" if done else "failed",
                                    error=None if done else summary.get("stop_reason"), record_type="summary")
        except Exception as log_exc:
            logger.warning("[cita-alignment] log failed: %s", log_exc)
        end_payload = {"action": "alignment_complete" if done else "alignment_failed",
                       "task_id": self._command_service.build_task_id(), "artifact_id": artifact_id,
                       "device_id": device_id, "total_iterations": session.iteration,
                       "reason": summary.get("stop_reason"), "workflow": workflow}
        ok, info = publish(end_payload)
        save_entry(True)
        return respond("complete" if done else "failed", info)
