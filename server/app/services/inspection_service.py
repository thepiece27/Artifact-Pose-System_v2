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

logger = logging.getLogger(__name__)

class InspectionService:
    def __init__(
        self,
        settings: Settings,
        pose_service: PoseService,
        model_service: ModelService,
        command_service: CommandService,
        mqtt_bridge: MqttBridge,
    ) -> None:
        self._settings = settings
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
                }), encoding="utf-8")
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
        result: dict[str, Any] = {
            "ssim": None,
            "ssim_gray": None,
            "ssim_color": None,
            "damage_pct": None,
            "heatmap_path": None,
            "auto_description": "Analysis performed.",
            "detections_json": None,
            "all_detections": [],
            "analysis_status": "pending",
            "analysis_error": None,
            "alignment_quality": "not_run",
            "sift_inliers": 0,
            "valid_mask_ratio": None,
        }

        try:
            import cv2
            import numpy as np
            current_img = cv2.imread(str(current_path))
            if current_img is None:
                result["analysis_status"] = "failed"
                result["analysis_error"] = "current_image_decode_failed"
                result["auto_description"] = "Error: Cannot read inspection image."
                return result
        except Exception as load_exc:
            result["analysis_status"] = "failed"
            result["analysis_error"] = f"image_load_error:{type(load_exc).__name__}"
            result["auto_description"] = f"Image load error: {load_exc}"
            return result

        all_dets: list[dict] = []
        aligned_img: Any = None
        aligned_url: str | None = None
        annotated = current_img.copy()

        if reference_path is not None and reference_path.exists():
            try:
                import numpy as np
                from skimage.metrics import structural_similarity
                reference_img = cv2.imread(str(reference_path))
                if reference_img is not None:
                    h, w = reference_img.shape[:2]
                    cur_resized = (
                        cv2.resize(current_img, (w, h))
                        if current_img.shape[:2] != (h, w)
                        else current_img.copy()
                    )

                    # SIFT: used only for diff map.  YOLO runs on yolo_source (no warp artifacts).
                    aligned_img, valid_mask, sift_inliers = self._sift_align_with_mask(cur_resized, reference_img)
                    result["sift_inliers"] = int(sift_inliers)
                    result["alignment_quality"] = (
                        "sift_homography"
                        if sift_inliers >= 15 and valid_mask is not None
                        else "resize_fallback"
                    )
                    if valid_mask is not None:
                        result["valid_mask_ratio"] = round(
                            float(cv2.countNonZero(valid_mask)) / max(float(h * w), 1.0), 6
                        )
                    ssim_source = aligned_img if aligned_img is not None else cur_resized
                    yolo_source = cur_resized

                    # ── Multi-channel SSIM (gray 60% + color 40%) ──────────────────
                    gray_src = cv2.cvtColor(ssim_source, cv2.COLOR_BGR2GRAY)
                    gray_ref = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
                    score_gray, diff_gray = structural_similarity(
                        gray_ref, gray_src, full=True, win_size=7
                    )
                    score_channels: list[float] = []
                    diff_color = np.zeros_like(gray_src, dtype=np.float64)
                    for c in range(3):
                        sc, dc = structural_similarity(
                            reference_img[:, :, c], ssim_source[:, :, c], full=True, win_size=7
                        )
                        score_channels.append(sc)
                        diff_color += (1.0 - dc)
                    diff_color /= 3.0
                    score_color  = float(np.mean(score_channels))
                    diff_combined = np.maximum(1.0 - diff_gray, diff_color)
                    diff_uint8    = (diff_combined * 255).astype(np.uint8)
                    ssim_score    = 0.6 * float(score_gray) + 0.4 * score_color
                    result["ssim"]       = ssim_score
                    result["ssim_gray"]  = float(score_gray)
                    result["ssim_color"] = score_color

                    if valid_mask is not None:
                        diff_uint8 = cv2.bitwise_and(diff_uint8, valid_mask)

                    # ── Heatmap ────────────────────────────────────────────────────
                    heatmap = cv2.applyColorMap(diff_uint8, cv2.COLORMAP_JET)
                    if valid_mask is not None:
                        heatmap[cv2.bitwise_not(valid_mask) > 0] = [128, 128, 128]
                    heatmap_overlay = cv2.addWeighted(ssim_source, 0.6, heatmap, 0.4, 0)

                    # ── Damage mask → contours ─────────────────────────────────────
                    blurred = cv2.GaussianBlur(diff_uint8, (5, 5), 0)
                    otsu_thresh, damage_mask = cv2.threshold(
                        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    if otsu_thresh < 30:
                        _, damage_mask = cv2.threshold(
                            blurred, self._OTSU_FALLBACK, 255, cv2.THRESH_BINARY
                        )
                    kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    kernel_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_OPEN,  kernel_s, iterations=2)
                    damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_CLOSE, kernel_b, iterations=2)
                    contours, _ = cv2.findContours(
                        damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    min_area = max(500, (h * w) * self._MIN_AREA_FACTOR)
                    big_contours = sorted(
                        [c for c in contours if cv2.contourArea(c) > min_area],
                        key=cv2.contourArea, reverse=True,
                    )[: self._CONTOUR_CAP]
                    cv2.drawContours(heatmap_overlay, big_contours, -1, (0, 0, 255), 2)

                    valid_area  = int(cv2.countNonZero(valid_mask)) if valid_mask is not None else (h * w)
                    damage_area = sum(cv2.contourArea(c) for c in big_contours)
                    damage_pct  = (damage_area / max(valid_area, 1)) * 100.0
                    result["damage_pct"] = round(damage_pct, 2)
                    result["analysis_status"] = "ok"

                    # ── Save heatmap ───────────────────────────────────────────────
                    safe_artifact_id = safe_component(artifact_id, field="artifact_id")
                    _out_dir = self._artifact_uploads_dir / safe_artifact_id
                    _out_dir.mkdir(parents=True, exist_ok=True)
                    heatmap_fname = f"heatmap_{safe_artifact_id}_{ts_ms}.jpg"
                    cv2.imwrite(str(_out_dir / heatmap_fname), heatmap_overlay)
                    result["heatmap_path"] = str(_out_dir / heatmap_fname)

                    # ── Save aligned image ─────────────────────────────────────────
                    if aligned_img is not None:
                        aligned_fname = f"aligned_{safe_artifact_id}_{ts_ms}.jpg"
                        cv2.imwrite(str(_out_dir / aligned_fname), aligned_img)
                        aligned_url = f"/uploads/artifacts/{safe_artifact_id}/{aligned_fname}"
                        result["aligned_image_path"] = aligned_url

                    # ── YOLO: hybrid pipeline on yolo_source ───────────────────────
                    annotated = yolo_source.copy()

                    def _pad_crop(bx: int, by: int, bw: int, bh: int,
                                  pad_factor: float, min_pad: int) -> tuple[int, int, int, int]:
                        _side = max(bw, bh)
                        _pad  = max(min_pad, int(_side * pad_factor))
                        _half = (_side // 2) + _pad
                        _cx, _cy = bx + bw // 2, by + bh // 2
                        return (
                            max(0, _cx - _half), max(0, _cy - _half),
                            min(w, _cx + _half), min(h, _cy + _half),
                        )

                    if damage_pct < self._DAMAGE_PCT_GATE or not big_contours:
                        # Fallback: full-image YOLO when damage level is very low
                        _yolo_raw = self._model_service.detect_image(
                            self._settings.default_ai_model_name,
                            cv2.imencode(".jpg", yolo_source)[1].tobytes(),
                        )
                        for _res in (_yolo_raw or []):
                            for _det in _res.get("detections", []):
                                _conf = float(_det.get("confidence", 0))
                                _name = str(_det.get("class_name", "unknown"))
                                x1, y1, x2, y2 = [int(v) for v in _det["bbox_xyxy"]]
                                _r_ssim = self._compute_region_ssim(
                                    ssim_source, reference_img, x1, y1, x2, y2, padding=0
                                )
                                all_dets.append({
                                    "class_name":  _name,
                                    "confidence":  round(_conf, 4),
                                    "bbox_xyxy":   [x1, y1, x2, y2],
                                    "from_crop":   "full",
                                    "region_ssim": round(float(_r_ssim), 4),
                                })
                    else:
                        # Phase A: tight crops batch
                        tight_crop_list: list = []
                        for _c in big_contours:
                            bx, by, bw, bh = cv2.boundingRect(_c)
                            x1c, y1c, x2c, y2c = _pad_crop(
                                bx, by, bw, bh, self._TIGHT_PAD_FACTOR, self._TIGHT_MIN_PAD
                            )
                            if x2c - x1c < 40 or y2c - y1c < 40:
                                continue
                            _r_ssim = self._compute_region_ssim(
                                ssim_source, reference_img, bx, by, bx + bw, by + bh, padding=0
                            )
                            tight_crop_list.append(
                                (yolo_source[y1c:y2c, x1c:x2c], x1c, y1c, "tight", float(_r_ssim))
                            )

                        tight_dets = (
                            self._model_service.detect_crops_batch(
                                self._settings.default_ai_model_name,
                                tight_crop_list,
                                conf=self._YOLO_CONF,
                                sub_batch=self._YOLO_SUB_BATCH,
                            ) if tight_crop_list else []
                        )

                        # Phase B: wide crops for contours without high-conf tight det
                        wide_crop_list: list = []
                        for _c in big_contours:
                            bx, by, bw, bh = cv2.boundingRect(_c)
                            _cx_c = bx + bw / 2
                            _cy_c = by + bh / 2
                            max_conf = max(
                                (
                                    d["confidence"]
                                    for d in tight_dets
                                    if abs((d["bbox_xyxy"][0] + d["bbox_xyxy"][2]) / 2 - _cx_c)
                                    < bw + self._TIGHT_MIN_PAD
                                    and abs((d["bbox_xyxy"][1] + d["bbox_xyxy"][3]) / 2 - _cy_c)
                                    < bh + self._TIGHT_MIN_PAD
                                ),
                                default=0.0,
                            )
                            if max_conf >= self._EARLY_EXIT_CONF:
                                continue
                            x1c, y1c, x2c, y2c = _pad_crop(
                                bx, by, bw, bh, self._WIDE_PAD_FACTOR, self._WIDE_MIN_PAD
                            )
                            if x2c - x1c < 40 or y2c - y1c < 40:
                                continue
                            _r_ssim = self._compute_region_ssim(
                                ssim_source, reference_img, bx, by, bx + bw, by + bh, padding=0
                            )
                            wide_crop_list.append(
                                (yolo_source[y1c:y2c, x1c:x2c], x1c, y1c, "wide", float(_r_ssim))
                            )

                        wide_dets = (
                            self._model_service.detect_crops_batch(
                                self._settings.default_ai_model_name,
                                wide_crop_list,
                                conf=self._YOLO_CONF,
                                sub_batch=self._YOLO_SUB_BATCH,
                            ) if wide_crop_list else []
                        )

                        # NMS across tight + wide
                        combined = self._nms_detections(tight_dets + wide_dets, self._NMS_IOU_THRESH)
                        for det in combined:
                            all_dets.append({
                                "class_name":  det["class_name"],
                                "confidence":  det["confidence"],
                                "bbox_xyxy":   det["bbox_xyxy"],
                                "from_crop":   det.get("from_crop", ""),
                                "region_ssim": det.get("region_ssim", 0.0),
                            })

            except Exception as exc:
                logger.error(f"[analyze] pipeline error: {exc}", exc_info=True)
                result["analysis_status"] = "failed"
                result["analysis_error"] = f"pipeline_error:{type(exc).__name__}"
        else:
            result["auto_description"] = "No reference image — AI detection only."
            try:
                _yolo_raw = self._model_service.detect_image(
                    self._settings.default_ai_model_name,
                    current_path.read_bytes(),
                )
                for _res in (_yolo_raw or []):
                    for _det in _res.get("detections", []):
                        x1, y1, x2, y2 = [int(v) for v in _det["bbox_xyxy"]]
                        all_dets.append({
                            "class_name":  str(_det.get("class_name", "unknown")),
                            "confidence":  round(float(_det.get("confidence", 0)), 4),
                            "bbox_xyxy":   [x1, y1, x2, y2],
                            "from_crop":   "full",
                            "region_ssim": None,
                        })
            except Exception as exc:
                logger.warning(f"[analyze] YOLO error (no reference): {exc}")
                result["analysis_status"] = "failed"
                result["analysis_error"] = f"detector_error:{type(exc).__name__}"

        if result["analysis_status"] == "pending":
            result["analysis_status"] = (
                "ok" if result.get("ssim") is not None or all_dets else "failed"
            )
            if result["analysis_status"] == "failed" and result["analysis_error"] is None:
                result["analysis_error"] = "no_analysis_signal"

        # ── Annotate bboxes on annotated image (no crop outlines) ─────────────
        try:
            _SEVERITY_COLOR = {
                "HIGH":   (0,   0,   255),
                "MEDIUM": (0,   128, 255),
                "LOW":    (0,   200, 128),
            }
            for det in all_dets:
                x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
                _conf = float(det.get("confidence", 0))
                _name = str(det.get("class_name", "unknown"))
                _r_ssim_val = det.get("region_ssim")
                _severity = "HIGH" if _conf >= 0.65 else "MEDIUM" if _conf >= 0.40 else "LOW"
                _color = _SEVERITY_COLOR[_severity]
                _label = f"{_name} {_conf * 100:.0f}%"
                if _r_ssim_val is not None:
                    _label += f" (SSIM {float(_r_ssim_val) * 100:.0f}%)"
                _thickness = 3 if _severity == "HIGH" else 2
                cv2.rectangle(annotated, (x1, y1), (x2, y2), _color, _thickness)
                _fs = 0.50
                (_tw, _th), _bl = cv2.getTextSize(_label, cv2.FONT_HERSHEY_SIMPLEX, _fs, 1)
                _yt = max(0, y1 - _th - _bl - 4)
                cv2.rectangle(annotated, (x1, _yt), (x1 + _tw + 4, y1), _color, -1)
                cv2.putText(
                    annotated, _label, (x1 + 2, y1 - _bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, _fs, (255, 255, 255), 1, cv2.LINE_AA,
                )
        except Exception as draw_exc:
            logger.warning(f"[analyze] annotation error: {draw_exc}")

        # ── Save annotated image ───────────────────────────────────────────────
        annotated_url: str | None = None
        try:
            safe_artifact_id = safe_component(artifact_id, field="artifact_id")
            _out_dir = self._artifact_uploads_dir / safe_artifact_id
            _out_dir.mkdir(parents=True, exist_ok=True)
            detect_fname = f"detect_{safe_artifact_id}_{ts_ms}.jpg"
            cv2.imwrite(str(_out_dir / detect_fname), annotated)
            annotated_url = f"/uploads/artifacts/{safe_artifact_id}/{detect_fname}"
        except Exception as save_exc:
            logger.warning(f"[analyze] save detect image error: {save_exc}")

        image_size: list[int] | None = None
        try:
            ah, aw = annotated.shape[:2]
            image_size = [int(aw), int(ah)]
        except Exception:
            image_size = None

        result["all_detections"] = all_dets
        result["detections_json"] = json.dumps({
            "annotated_path": annotated_url,
            "aligned_path":   aligned_url,
            "image_size":     image_size,
            "all_detections": all_dets,
            "ssim_summary": {
                "ssim":       result.get("ssim"),
                "ssim_gray":  result.get("ssim_gray"),
                "ssim_color": result.get("ssim_color"),
                "damage_pct": result.get("damage_pct"),
            } if result.get("ssim") is not None else None,
            "analysis_status": result.get("analysis_status"),
            "analysis_error": result.get("analysis_error"),
            "alignment_quality": result.get("alignment_quality"),
            "sift_inliers": result.get("sift_inliers", 0),
            "valid_mask_ratio": result.get("valid_mask_ratio"),
        })

        det_count = len(all_dets)
        ssim_str = f" | SSIM {result['ssim'] * 100:.1f}%" if result.get("ssim") is not None else ""
        result["auto_description"] = (
            f"{det_count} region(s) detected{ssim_str}." if det_count
            else f"No damage detected{ssim_str}."
        )
        logger.info("[analyze] status=%s %d detection(s), SSIM=%.4f for artifact=%s",
                    result.get("analysis_status"), det_count,
                    result.get("ssim") or 0.0, artifact_id)
        return result

    @staticmethod
    def _nms_detections(dets: list[dict], iou_threshold: float = 0.45) -> list[dict]:
        """Class-aware NMS for detections from tight/wide crops.

        Overlap between different defect classes is not a duplicate.  The old
        class-agnostic suppression could silently remove a real finding.
        """
        if not dets:
            return []
        keep: list[dict] = []
        groups: dict[str, list[dict]] = {}
        for det in dets:
            groups.setdefault(str(det.get("class_name", "unknown")), []).append(det)
        for group in groups.values():
            dets_sorted = sorted(
                group, key=lambda d: float(d.get("confidence", 0)), reverse=True
            )
            while dets_sorted:
                d = dets_sorted.pop(0)
                keep.append(d)
                ax1, ay1, ax2, ay2 = [float(v) for v in d["bbox_xyxy"]]
                area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
                remaining: list[dict] = []
                for e in dets_sorted:
                    bx1, by1, bx2, by2 = [float(v) for v in e["bbox_xyxy"]]
                    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
                    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                    iou = inter / max(area_a + area_b - inter, 1e-9)
                    if iou <= iou_threshold:
                        remaining.append(e)
                dets_sorted = remaining
        return keep

    @staticmethod
    def _sift_align_with_mask(img: Any, reference: Any) -> tuple[Any, Any, int]:
        """SIFT + Homography alignment. Returns (aligned, valid_mask, inlier_count)."""
        try:
            import cv2
            import numpy as np
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

            sift = cv2.SIFT_create(nfeatures=3000)
            kp1, des1 = sift.detectAndCompute(gray_img, None)
            kp2, des2 = sift.detectAndCompute(gray_ref, None)

            if des1 is None or des2 is None or len(kp1) < 15 or len(kp2) < 15:
                resized = cv2.resize(img, (reference.shape[1], reference.shape[0]))
                return resized, None, 0

            flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
            matches = flann.knnMatch(des1, des2, k=2)
            good = [m for pair in matches if len(pair) == 2
                    for m, n in [pair] if m.distance < 0.65 * n.distance]

            if len(good) < 15:
                resized = cv2.resize(img, (reference.shape[1], reference.shape[0]))
                return resized, None, len(good)

            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
            if H is None:
                resized = cv2.resize(img, (reference.shape[1], reference.shape[0]))
                return resized, None, len(good)

            inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
            h, w = reference.shape[:2]
            aligned = cv2.warpPerspective(img, H, (w, h))

            # Build valid pixel mask (avoid black warped borders)
            ones = np.ones(img.shape[:2], dtype=np.uint8) * 255
            valid_mask = cv2.warpPerspective(ones, H, (w, h))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            valid_mask = cv2.erode(valid_mask, kernel, iterations=2)

            return aligned, valid_mask, inliers
        except Exception:
            return None, None, 0

    @staticmethod
    def _compute_region_ssim(img: Any, reference: Any, x1: int, y1: int, x2: int, y2: int, padding: int = 20) -> float:
        """SSIM for a YOLO bounding-box region (with padding). Returns score 0-1."""
        try:
            from skimage.metrics import structural_similarity
            h, w = img.shape[:2]
            px1, py1 = max(0, x1 - padding), max(0, y1 - padding)
            px2, py2 = min(w, x2 + padding), min(h, y2 + padding)
            crop_img = img[py1:py2, px1:px2]
            crop_ref = reference[py1:py2, px1:px2]
            if crop_img.shape[0] < 16 or crop_img.shape[1] < 16:
                return 1.0
            import cv2
            gray_i = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(crop_ref, cv2.COLOR_BGR2GRAY)
            win = min(7, min(gray_i.shape) - 1)
            if win % 2 == 0:
                win -= 1
            if win < 3:
                return 1.0
            score, _ = structural_similarity(gray_r, gray_i, full=True, win_size=win)
            return float(score)
        except Exception:
            return 1.0

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
        ai_result: dict[str, Any] | None = None

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
                motor_cmd = pose_result.get("motor_command")
                if motor_cmd and motor_cmd.get("enabled", True) and device_id:
                    raw_move_x = float(motor_cmd.get("move_x",     0))
                    raw_move_z = float(motor_cmd.get("move_z",     0))
                    raw_pan    = float(motor_cmd.get("rotate_pan",  0))
                    raw_tilt   = float(motor_cmd.get("rotate_tilt", 0))

                    within_trans = bool(deviation.get("within_trans_tolerance", False))
                    within_rot   = bool(deviation.get("within_rot_tolerance",   False))

                    # Resolve active phase; default to 0 (translation) if somehow missing.
                    current_phase = self._alignment_phase.get(alignment_key, 0)

                    # Smart-skip: if this phase's axis is already within tolerance,
                    # advance immediately to the other phase (avoids a wasted round-trip).
                    if current_phase == 0 and within_trans and not within_rot:
                        current_phase = 1
                        self._alignment_phase[alignment_key] = 1
                        self._persist_alignment_state()
                        logger.info(
                            "[alignment] Trans within tolerance → skip to ROTATION phase "
                            "(device=%s, iter=%d)",
                            device_id, current_iter,
                        )
                    elif current_phase == 1 and within_rot and not within_trans:
                        current_phase = 0
                        self._alignment_phase[alignment_key] = 0
                        self._persist_alignment_state()
                        logger.info(
                            "[alignment] Rot within tolerance → skip to TRANSLATION phase "
                            "(device=%s, iter=%d)",
                            device_id, current_iter,
                        )

                    if current_phase == 0:
                        # ── Phase 0: Translation — X, Z steppers only ──────────
                        mc_payload: dict[str, Any] = {
                            "action": "move",
                            "task_id": self._command_service.build_task_id(),
                            "artifact_id": artifact_id,
                            "x_steps": abs(int(round(raw_move_x))),
                            "x_dir":   1 if raw_move_x >= 0 else -1,
                            "z_steps": abs(int(round(raw_move_z))),
                            "z_dir":   1 if raw_move_z >= 0 else -1,
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
        self._persist_alignment_state()
