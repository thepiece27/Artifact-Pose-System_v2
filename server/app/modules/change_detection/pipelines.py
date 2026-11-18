"""Inspection pipelines P0-P4 assembled from the blocks of this package.

P0  cita-baseline code (``legacy.analyze_legacy``), coordinate bug kept.
P1  P0 + painting ROI on the difference map/contours; YOLO crops from
    ``ai_input`` (aligned frame, i.e. coordinate fix).
P2  P1 + ROI homography + ECC + photometric normalisation + standard SSIM +
    fixed threshold (validation-chosen) + min area in mm^2. No YOLO.
P3  P2 + the existing two-phase YOLO on the AIInputPackage (proposed).
P4  YOLO on the current image only (no reference). Its boxes are mapped to
    the reference frame with an evaluation-only ROI homography.

Unified output: ``regions`` (class-agnostic, reference frame), ``detections``
(classes, reference frame), ``image_score`` (threshold-free), ``ssim``,
registration diagnostics, AIInputPackage path.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import legacy
from .ai_input import build_package, map_box_to_ref, run_two_phase_yolo
from .photometric import normalize
from .regions import extract_regions, image_score, load_thresholds, min_area_px, threshold_map
from .registration import legacy_whole_frame, roi_homography
from .roi import auto_detect_quad, load_roi, px_per_mm as roi_px_per_mm, roi_mask
from .ssim_map import legacy_ssim, standard_ssim


def _full_frame_polygon(shape) -> np.ndarray:
    h, w = shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)


def compare(code: str, reference: np.ndarray, current: np.ndarray, cfg, roi_polygon) -> dict:
    """Registration + photometric + SSIM difference map (no threshold)."""
    if current.shape[:2] != reference.shape[:2]:
        current = cv2.resize(current, (reference.shape[1], reference.shape[0]))
    if code == "P1":
        reg = legacy_whole_frame(current, reference, roi_polygon)
        aligned = reg.aligned if reg.aligned is not None else current
        ss = legacy_ssim(reference, aligned)
        valid = reg.valid_mask if reg.valid_mask is not None else np.full(reference.shape[:2], 255, np.uint8)
        valid = cv2.bitwise_and(valid, roi_mask(reference.shape, roi_polygon, cfg.roi_erode_px))
        d = np.where(valid > 0, ss["d"], 0.0)
        diff = cv2.bitwise_and(ss["diff_uint8"], valid)
        inside = valid > 0
        ssim_roi = float(1.0 - d[inside].mean()) if inside.any() else None
        return {"reg": reg, "aligned": aligned, "valid": valid, "d": d, "diff_uint8": diff, "ssim": ss["ssim"],
                "ssim_roi": ssim_roi, "ssim_gray": ss["ssim_gray"], "ssim_color": ss["ssim_color"],
                "photometric": {"mode": "none"}}
    if code in ("P2", "P3"):
        if cfg.registration == "roi_homography":
            reg = roi_homography(current, reference, roi_polygon, ecc=cfg.ecc.enabled, levels=cfg.ecc.levels,
                                 max_iter=cfg.ecc.max_iter, eps=cfg.ecc.eps, erode_px=cfg.roi_erode_px,
                                 max_side_px=cfg.ecc.max_side_px)
        else:
            reg = legacy_whole_frame(current, reference, roi_polygon)
        aligned = reg.aligned if reg.aligned is not None else current
        valid = reg.valid_mask
        if valid is None:
            valid = roi_mask(reference.shape, roi_polygon, cfg.roi_erode_px)
        else:
            valid = cv2.bitwise_and(valid, roi_mask(reference.shape, roi_polygon, cfg.roi_erode_px))
        cur_n, ref_n, photo = normalize(aligned, reference, valid, cfg.photometric)
        if cfg.ssim.mode == "standard":
            ss = standard_ssim(ref_n, cur_n, valid, work_long_side_px=cfg.ssim.work_long_side_px,
                               channels=cfg.ssim.channels)
            out = {"ssim": ss["ssim"], "ssim_roi": ss["ssim"], "ssim_gray": None, "ssim_color": None}
        else:
            ss = legacy_ssim(ref_n, cur_n)
            ss["d"] = np.where(valid > 0, ss["d"], 0.0)
            ss["diff_uint8"] = cv2.bitwise_and(ss["diff_uint8"], valid)
            out = {"ssim": ss["ssim"], "ssim_roi": float(1 - ss["d"][valid > 0].mean()) if (valid > 0).any() else None,
                   "ssim_gray": ss["ssim_gray"], "ssim_color": ss["ssim_color"]}
        return {"reg": reg, "aligned": aligned, "aligned_normalized": cur_n, "valid": valid, "d": ss["d"],
                "diff_uint8": ss["diff_uint8"], "photometric": photo, **out}
    raise ValueError(f"compare() is not defined for {code}")


def _tau(cfg, thresholds: dict | None) -> tuple[float | None, str | None]:
    if cfg.threshold.mode != "fixed":
        return None, None
    if cfg.threshold.tau is not None:
        return float(cfg.threshold.tau), "profile"
    if thresholds is None:
        raise ValueError("fixed threshold: give inspection.threshold.tau or a thresholds.yaml (calibrate_thresholds.py)")
    return float(thresholds["tau"]), f"{Path(thresholds.get('_path', 'thresholds.yaml')).name}#{thresholds.get('_sha256', '')[:12]}"


def _box_match(det_box, region_box, iou_thr: float = 0.25) -> bool:
    """IoU >= iou_thr OR the detection centre lies inside the region box."""
    ax1, ay1, ax2, ay2 = (float(v) for v in det_box)
    bx1, by1, bx2, by2 = (float(v) for v in region_box)
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    cx, cy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    return (union > 0 and inter / union >= iou_thr) or (bx1 <= cx <= bx2 and by1 <= cy <= by2)


def verify_regions(regions: list[dict], detections: list[dict], iou_thr: float = 0.25) -> list[dict]:
    """SSIM regions with >= 1 matching YOLO detection (reference frame), each
    carrying ``yolo_confidence_max`` / ``yolo_class`` of its best detection."""
    out = []
    for r in regions:
        matched = [d for d in detections
                   if _box_match(d.get("bbox_ref_xyxy", d["bbox_xyxy"]), r["bbox_ref_xyxy"], iou_thr)]
        if matched:
            best = max(matched, key=lambda d: float(d["confidence"]))
            out.append({**r, "yolo_confidence_max": float(best["confidence"]), "yolo_class": best["class_name"],
                        "n_detections": len(matched)})
    return out


def _strip(regions: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "contour"} for r in regions]


def _full_image_yolo(model_service, model_name, image) -> list[dict]:
    raw = model_service.detect_image(model_name, cv2.imencode(".jpg", image)[1].tobytes())
    out = []
    for res in raw or []:
        for det in res.get("detections", []):
            out.append({"class_name": str(det.get("class_name", "unknown")),
                        "confidence": round(float(det.get("confidence", 0)), 4),
                        "bbox_xyxy": [int(v) for v in det["bbox_xyxy"]], "from_crop": "full"})
    return out


def run_pipeline(code: str, *, reference: np.ndarray, current: np.ndarray, cfg, roi: dict | None = None,
                 px_per_mm: float | None = None, thresholds: dict | None = None, model_service=None,
                 model_name: str = "default", out_dir: Path | None = None, reference_path: Path | None = None,
                 current_path: Path | None = None, artifact_id: str = "replay", ts_ms: int = 0) -> dict:
    roi_polygon = roi["polygon_ref_px"] if roi else _full_frame_polygon(reference.shape)
    ppm = px_per_mm if px_per_mm is not None else roi_px_per_mm(roi)
    out: dict[str, Any] = {"pipeline": code, "regions": [], "detections": [], "image_score": None, "ssim": None,
                           "roi_afd_px": None, "ecc_ok": None, "ai_input_package_path": None, "status": "ok"}
    if code == "P0":
        if reference_path is None or current_path is None:
            raise ValueError("P0 runs the legacy code on files: give reference_path and current_path")
        tmp = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="p0_"))
        res = legacy.analyze_legacy(current_path=Path(current_path), reference_path=Path(reference_path),
                                    artifact_id=artifact_id, ts_ms=ts_ms, model_service=model_service,
                                    model_name=model_name, uploads_artifacts_dir=tmp)
        regions = [{"region_id": i, **r, "area_mm2": r["area_px"] / ppm ** 2 if ppm else None,
                    "score": r["area_px"]} for i, r in enumerate(res.get("cita_regions") or [])]
        out.update(regions=regions, ssim=res.get("ssim"), legacy=res,
                   image_score=None if res.get("ssim") is None else 1.0 - float(res["ssim"]),
                   detections=[dict(d, bbox_ref_xyxy=d["bbox_xyxy"]) for d in res.get("all_detections") or []],
                   status=res.get("analysis_status"))
        return out
    if code == "P4":
        if model_service is None:
            raise ValueError("P4 needs a model service")
        dets = _full_image_yolo(model_service, model_name, current)
        reg = roi_homography(current, reference, roi_polygon, ecc=False)
        H = reg.H_ref_from_cur
        out["registration_eval_only"] = reg.to_json()
        for d in dets:
            d["bbox_ref_xyxy"] = map_box_to_ref(d["bbox_xyxy"], "original", H) if H is not None else list(map(float, d["bbox_xyxy"]))
        out["detections"] = dets
        out["regions"] = [{"region_id": i, "bbox_ref_xyxy": d["bbox_ref_xyxy"], "score": d["confidence"],
                           "class_name": d["class_name"]} for i, d in enumerate(dets)]
        out["image_score"] = max((d["confidence"] for d in dets), default=0.0)
        return out
    comp = compare(code, reference, current, cfg, roi_polygon)
    reg, valid, d = comp["reg"], comp["valid"], comp["d"]
    out.update(ssim=comp["ssim_roi"], ssim_global=comp["ssim"], ssim_gray=comp.get("ssim_gray"),
               ssim_color=comp.get("ssim_color"), registration=reg.to_json(), roi_afd_px=reg.roi_afd_px,
               ecc_ok=reg.ecc_ok, photometric=comp["photometric"], alignment_quality=reg.quality,
               sift_inliers=reg.n_inliers)
    mode = "otsu_legacy" if code == "P1" else cfg.threshold.mode
    tau, tau_src = (None, None) if code == "P1" else _tau(cfg, thresholds)
    mask, thr_info = threshold_map(d, comp["diff_uint8"], mode, tau)
    mask = cv2.bitwise_and(mask, valid)
    min_area = min_area_px(reference.shape, min_area_mm2=None if code == "P1" else cfg.min_area_mm2, px_per_mm=ppm)
    regions = extract_regions(mask, d, min_area=min_area, px_per_mm=ppm)
    valid_area = max(int(cv2.countNonZero(valid)), 1)
    damage_pct = sum(r["area_px"] for r in regions) / valid_area * 100.0
    out.update(regions=_strip(regions), image_score=image_score(d, valid), damage_pct=round(damage_pct, 2),
               threshold={**thr_info, "source": tau_src, "min_area_px": min_area},
               valid_mask_ratio=round(valid_area / float(valid.size), 6), aligned=comp["aligned"], d=d, valid=valid)
    if code in ("P1", "P3") and model_service is not None and cfg.yolo:
        pkg_dir = Path(out_dir) / "ai_input" if out_dir else None
        package, arrays = build_package(
            pipeline=code, regions=out["regions"], ref_shape=reference.shape, aligned=comp["aligned"],
            current=current, H_ref_from_cur=reg.H_ref_from_cur, crop_source=cfg.crop_source, out_dir=pkg_dir,
            reference_image=str(reference_path) if reference_path else None,
            current_image=str(current_path) if current_path else None, roi_polygon=roi_polygon, px_per_mm=ppm,
            ssim_roi=comp["ssim_roi"], threshold=out["threshold"])
        out["ai_input_package"] = package
        out["ai_input_package_path"] = str(pkg_dir / "ai_input_package.json") if pkg_dir else None
        if out["regions"] and (code == "P3" or damage_pct >= legacy.DAMAGE_PCT_GATE):
            out["detections"] = run_two_phase_yolo(package, arrays, model_service=model_service,
                                                   model_name=model_name, ssim_source=comp["aligned"],
                                                   reference=reference)
        elif code == "P1":
            # cita-baseline fallback (full-image YOLO when damage is tiny), on the ALIGNED image.
            dets = _full_image_yolo(model_service, model_name, comp["aligned"])
            out["detections"] = [dict(x, bbox_ref_xyxy=list(map(float, x["bbox_xyxy"]))) for x in dets]
        # R4: SSIM regions confirmed by at least one YOLO detection (scoring only;
        # YOLO itself is untouched). Evaluated as the virtual pipeline "P3v".
        out["verified_regions"] = verify_regions(out["regions"], out["detections"])
        out["image_score_verified"] = max((r["score"] for r in out["verified_regions"]), default=0.0)
    return out


# ---------------------------------------------------------------------------
# Server adapter (InspectionService._analyze_against_reference for P1-P4)
# ---------------------------------------------------------------------------

def resolve_roi(reference: np.ndarray, golden_dir: Path | None, artifact_id: str) -> dict:
    if golden_dir is not None:
        from app.core.uploads import safe_component
        p = Path(golden_dir) / safe_component(artifact_id, field="artifact_id") / "roi.json"
        if p.exists():
            return load_roi(p)
    quad = auto_detect_quad(reference)
    if quad is not None:
        return {"polygon_ref_px": quad, "size_mm": None, "source": "auto_quad"}
    return {"polygon_ref_px": _full_frame_polygon(reference.shape), "size_mm": None, "source": "full_frame"}


def run_server_pipeline(profile, *, current_path: Path, reference_path: Path | None, artifact_id: str, ts_ms: int,
                        model_service, model_name: str, uploads_artifacts_dir: Path,
                        golden_dir: Path | None = None) -> dict[str, Any]:
    """Same keys as the legacy analyzer + pipeline/roi_afd_px/ecc_ok/ai_input_package_path."""
    from app.core.uploads import safe_component
    cfg = profile.inspection
    code = cfg.pipeline
    result: dict[str, Any] = {"ssim": None, "ssim_gray": None, "ssim_color": None, "damage_pct": None,
                              "heatmap_path": None, "auto_description": "Analysis performed.",
                              "detections_json": None, "all_detections": [], "analysis_status": "pending",
                              "analysis_error": None, "alignment_quality": "not_run", "sift_inliers": 0,
                              "valid_mask_ratio": None, "pipeline": code, "roi_afd_px": None, "ecc_ok": None,
                              "ai_input_package_path": None}
    current = cv2.imread(str(current_path))
    reference = cv2.imread(str(reference_path)) if reference_path is not None and Path(reference_path).exists() else None
    if current is None:
        result.update(analysis_status="failed", analysis_error="current_image_decode_failed")
        return result
    if reference is None and code != "P4":
        # Same contract as cita-baseline: no reference -> detector only.
        return legacy.analyze_legacy(current_path=current_path, reference_path=None, artifact_id=artifact_id,
                                     ts_ms=ts_ms, model_service=model_service, model_name=model_name,
                                     uploads_artifacts_dir=uploads_artifacts_dir)
    safe_id = safe_component(artifact_id, field="artifact_id")
    out_dir = uploads_artifacts_dir / safe_id
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        roi = resolve_roi(reference if reference is not None else current, golden_dir, artifact_id)
        thresholds = None
        if cfg.threshold.mode == "fixed" and cfg.threshold.tau is None and cfg.threshold.file:
            thresholds = load_thresholds(profile.resolve_path(cfg.threshold.file))
        res = run_pipeline(code, reference=reference if reference is not None else current, current=current, cfg=cfg,
                           roi=roi, thresholds=thresholds, model_service=model_service, model_name=model_name,
                           out_dir=out_dir / f"cita_{code}_{ts_ms}", reference_path=reference_path,
                           current_path=current_path, artifact_id=artifact_id, ts_ms=ts_ms)
    except Exception as exc:
        legacy.logger.error(f"[analyze:{code}] pipeline error: {exc}", exc_info=True)
        result.update(analysis_status="failed", analysis_error=f"pipeline_error:{type(exc).__name__}")
        return result
    dets = [{"class_name": d["class_name"], "confidence": d["confidence"],
             "bbox_xyxy": [int(round(v)) for v in d.get("bbox_ref_xyxy", d["bbox_xyxy"])],
             "from_crop": d.get("from_crop", ""), "region_ssim": d.get("region_ssim")} for d in res["detections"]]
    heat_url = aligned_url = None
    if res.get("d") is not None:
        heat = cv2.applyColorMap((res["d"] * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat[res["valid"] == 0] = [128, 128, 128]
        overlay = cv2.addWeighted(res["aligned"], 0.6, heat, 0.4, 0)
        cv2.polylines(overlay, [np.rint(roi["polygon_ref_px"]).astype(np.int32)], True, (255, 255, 0), 2)
        for r in res["regions"]:
            x1, y1, x2, y2 = r["bbox_ref_xyxy"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        heat_path = out_dir / f"heatmap_{safe_id}_{ts_ms}.jpg"
        cv2.imwrite(str(heat_path), overlay)
        aligned_path = out_dir / f"aligned_{safe_id}_{ts_ms}.jpg"
        cv2.imwrite(str(aligned_path), res["aligned"])
        heat_url, aligned_url = str(heat_path), f"/uploads/artifacts/{safe_id}/{aligned_path.name}"
    result.update(ssim=res.get("ssim"), ssim_gray=res.get("ssim_gray"), ssim_color=res.get("ssim_color"),
                  damage_pct=res.get("damage_pct"), heatmap_path=heat_url, all_detections=dets,
                  analysis_status="ok", alignment_quality=res.get("alignment_quality", code),
                  sift_inliers=res.get("sift_inliers", 0), valid_mask_ratio=res.get("valid_mask_ratio"),
                  roi_afd_px=res.get("roi_afd_px"), ecc_ok=res.get("ecc_ok"),
                  ai_input_package_path=res.get("ai_input_package_path"), roi_source=roi.get("source"))
    if aligned_url:
        result["aligned_image_path"] = aligned_url
    result["detections_json"] = json.dumps({
        "annotated_path": None, "aligned_path": aligned_url, "image_size": [int(reference.shape[1]), int(reference.shape[0])]
        if reference is not None else None, "all_detections": dets,
        "ssim_summary": {"ssim": result["ssim"], "ssim_gray": result["ssim_gray"], "ssim_color": result["ssim_color"],
                         "damage_pct": result["damage_pct"]} if result["ssim"] is not None else None,
        "analysis_status": "ok", "analysis_error": None, "alignment_quality": result["alignment_quality"],
        "sift_inliers": result["sift_inliers"], "valid_mask_ratio": result["valid_mask_ratio"],
        "pipeline": code, "regions": res["regions"], "roi_afd_px": result["roi_afd_px"], "ecc_ok": result["ecc_ok"],
        "ai_input_package_path": result["ai_input_package_path"]}, default=float)
    n = len(dets)
    ssim_str = f" | SSIM {result['ssim'] * 100:.1f}%" if result.get("ssim") is not None else ""
    result["auto_description"] = (f"{n} region(s) detected{ssim_str}." if n else
                                  f"{len(res['regions'])} change region(s){ssim_str}." if res["regions"] else
                                  f"No damage detected{ssim_str}.")
    return result
