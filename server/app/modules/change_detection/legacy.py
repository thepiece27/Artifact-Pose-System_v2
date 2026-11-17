"""P0 = the inspection pipeline of tag ``cita-baseline`` (PBL).

Moved verbatim from ``InspectionService`` (``_analyze_against_reference`` and
its static helpers); behaviour is unchanged, including the known coordinate
bug (YOLO crops are cut from the UNALIGNED current image at contour
coordinates of the reference frame). ``InspectionService`` keeps thin
delegates so existing callers/tools still work. Regression-tested against
``tests/fixtures/expected_baseline/inspection.json``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.core.uploads import safe_component

logger = logging.getLogger("app.services.inspection_service")

# ── Pipeline tuning — edit here to adjust behaviour ──────────────────────
YOLO_CONF        = 0.15   # detection confidence threshold
YOLO_SUB_BATCH   = 4      # crops per model.predict() call (OOM guard)
MIN_AREA_FACTOR  = 0.0001 # contour area filter: max(500, h*w*factor) px²
DAMAGE_PCT_GATE  = 0.5    # skip crop-YOLO if damage_pct < this %
OTSU_FALLBACK    = 60     # hard threshold when Otsu < 30
CONTOUR_CAP      = 50     # max contours to inspect
TIGHT_PAD_FACTOR = 0.30   # tight crop: padding as fraction of contour side
TIGHT_MIN_PAD    = 40     # tight crop: minimum padding in px
WIDE_PAD_FACTOR  = 0.70   # wide crop: padding as fraction of contour side
WIDE_MIN_PAD     = 120    # wide crop: minimum padding in px
NMS_IOU_THRESH   = 0.45   # NMS IoU threshold (tight vs wide de-dup)
EARLY_EXIT_CONF  = 0.40   # skip wide crop if tight already >= this conf
# ─────────────────────────────────────────────────────────────────────────


def analyze_legacy(*, current_path: Path, reference_path: Path | None, artifact_id: str, ts_ms: int,
                   model_service: Any, model_name: str, uploads_artifacts_dir: Path) -> dict[str, Any]:
    """cita-baseline ``InspectionService._analyze_against_reference`` (P0), moved verbatim.

    Only instance attribute references were replaced by parameters / module
    constants. Addition (CITA): ``result['cita_regions']`` lists the contour
    boxes in the reference frame so P0 can be evaluated at region level.
    """
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
                aligned_img, valid_mask, sift_inliers = sift_align_with_mask(cur_resized, reference_img)
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
                        blurred, OTSU_FALLBACK, 255, cv2.THRESH_BINARY
                    )
                kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                kernel_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_OPEN,  kernel_s, iterations=2)
                damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_CLOSE, kernel_b, iterations=2)
                contours, _ = cv2.findContours(
                    damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                min_area = max(500, (h * w) * MIN_AREA_FACTOR)
                big_contours = sorted(
                    [c for c in contours if cv2.contourArea(c) > min_area],
                    key=cv2.contourArea, reverse=True,
                )[: CONTOUR_CAP]
                cv2.drawContours(heatmap_overlay, big_contours, -1, (0, 0, 255), 2)
                result["cita_regions"] = [
                    {"bbox_ref_xyxy": [int(bx), int(by), int(bx + bw), int(by + bh)], "area_px": float(cv2.contourArea(c))}
                    for c in big_contours for bx, by, bw, bh in [cv2.boundingRect(c)]
                ]

                valid_area  = int(cv2.countNonZero(valid_mask)) if valid_mask is not None else (h * w)
                damage_area = sum(cv2.contourArea(c) for c in big_contours)
                damage_pct  = (damage_area / max(valid_area, 1)) * 100.0
                result["damage_pct"] = round(damage_pct, 2)
                result["analysis_status"] = "ok"

                # ── Save heatmap ───────────────────────────────────────────────
                safe_artifact_id = safe_component(artifact_id, field="artifact_id")
                _out_dir = uploads_artifacts_dir / safe_artifact_id
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

                if damage_pct < DAMAGE_PCT_GATE or not big_contours:
                    # Fallback: full-image YOLO when damage level is very low
                    _yolo_raw = model_service.detect_image(
                        model_name,
                        cv2.imencode(".jpg", yolo_source)[1].tobytes(),
                    )
                    for _res in (_yolo_raw or []):
                        for _det in _res.get("detections", []):
                            _conf = float(_det.get("confidence", 0))
                            _name = str(_det.get("class_name", "unknown"))
                            x1, y1, x2, y2 = [int(v) for v in _det["bbox_xyxy"]]
                            _r_ssim = compute_region_ssim(
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
                            bx, by, bw, bh, TIGHT_PAD_FACTOR, TIGHT_MIN_PAD
                        )
                        if x2c - x1c < 40 or y2c - y1c < 40:
                            continue
                        _r_ssim = compute_region_ssim(
                            ssim_source, reference_img, bx, by, bx + bw, by + bh, padding=0
                        )
                        tight_crop_list.append(
                            (yolo_source[y1c:y2c, x1c:x2c], x1c, y1c, "tight", float(_r_ssim))
                        )

                    tight_dets = (
                        model_service.detect_crops_batch(
                            model_name,
                            tight_crop_list,
                            conf=YOLO_CONF,
                            sub_batch=YOLO_SUB_BATCH,
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
                                < bw + TIGHT_MIN_PAD
                                and abs((d["bbox_xyxy"][1] + d["bbox_xyxy"][3]) / 2 - _cy_c)
                                < bh + TIGHT_MIN_PAD
                            ),
                            default=0.0,
                        )
                        if max_conf >= EARLY_EXIT_CONF:
                            continue
                        x1c, y1c, x2c, y2c = _pad_crop(
                            bx, by, bw, bh, WIDE_PAD_FACTOR, WIDE_MIN_PAD
                        )
                        if x2c - x1c < 40 or y2c - y1c < 40:
                            continue
                        _r_ssim = compute_region_ssim(
                            ssim_source, reference_img, bx, by, bx + bw, by + bh, padding=0
                        )
                        wide_crop_list.append(
                            (yolo_source[y1c:y2c, x1c:x2c], x1c, y1c, "wide", float(_r_ssim))
                        )

                    wide_dets = (
                        model_service.detect_crops_batch(
                            model_name,
                            wide_crop_list,
                            conf=YOLO_CONF,
                            sub_batch=YOLO_SUB_BATCH,
                        ) if wide_crop_list else []
                    )

                    # NMS across tight + wide
                    combined = nms_detections(tight_dets + wide_dets, NMS_IOU_THRESH)
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
            _yolo_raw = model_service.detect_image(
                model_name,
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
        _out_dir = uploads_artifacts_dir / safe_artifact_id
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


def nms_detections(dets: list[dict], iou_threshold: float = 0.45) -> list[dict]:
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

def sift_align_with_mask(img: Any, reference: Any) -> tuple[Any, Any, int]:
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

def compute_region_ssim(img: Any, reference: Any, x1: int, y1: int, x2: int, y2: int, padding: int = 20) -> float:
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
