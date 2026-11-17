"""AIInputPackage (schema ``cita.ai_input.v1``, Appendix B.3) and YOLO glue.

The package lists the SSIM regions (reference frame) and, for each, a tight
and a wide crop (padding as cita-baseline: 30 % / >= 40 px and 70 % / >= 120 px)
cut from:

* ``aligned`` (default): the full-resolution current image warped into the
  reference frame, i.e. the SAME frame as the SSIM map (fixes the legacy bug);
* ``original_inverse``: the original current image; the padded reference box
  is mapped with H^-1 and its bounding box is cropped.

Each crop carries ``bbox_src_xyxy`` + ``src`` so YOLO boxes can be mapped
back (``map_detections_to_ref``). ``run_two_phase_yolo`` reproduces the
cita-baseline tight/wide phases, early exit (>= 0.40) and class-aware NMS
exactly; the model call (``ModelService.detect_crops_batch``) is unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import legacy

SCHEMA = "cita.ai_input.v1"


def pad_box(bx: int, by: int, bw: int, bh: int, pad_factor: float, min_pad: int, w: int, h: int) -> tuple:
    """cita-baseline ``_pad_crop`` (square box around the contour)."""
    side = max(bw, bh)
    pad = max(min_pad, int(side * pad_factor))
    half = (side // 2) + pad
    cx, cy = bx + bw // 2, by + bh // 2
    return max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half)


def _transform_box(box, H, w, h) -> list[int]:
    x1, y1, x2, y2 = box
    pts = np.float64([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
    t = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return [int(max(0, np.floor(t[:, 0].min()))), int(max(0, np.floor(t[:, 1].min()))),
            int(min(w, np.ceil(t[:, 0].max()))), int(min(h, np.ceil(t[:, 1].max())))]


def build_package(*, pipeline: str, regions: list[dict], ref_shape, aligned: np.ndarray, current: np.ndarray,
                  H_ref_from_cur: np.ndarray | None, crop_source: str = "aligned", out_dir: Path | None = None,
                  reference_image: str | None = None, current_image: str | None = None,
                  aligned_image: str | None = None, roi_polygon=None, px_per_mm: float | None = None,
                  ssim_roi: float | None = None, threshold: dict | None = None) -> tuple[dict, dict]:
    """Returns (package_json, crop_arrays) where crop_arrays[(region_id, kind)] = image."""
    h, w = ref_shape[:2]
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    if crop_source == "original_inverse" and H_ref_from_cur is None:
        crop_source = "aligned"
    H_inv = None if H_ref_from_cur is None else np.linalg.inv(H_ref_from_cur)
    ch, cw = current.shape[:2]
    arrays, out_regions = {}, []
    for r in regions:
        x1, y1, x2, y2 = r["bbox_ref_xyxy"]
        bx, by, bw, bh = x1, y1, x2 - x1, y2 - y1
        crops = {}
        for kind, factor, min_pad in (("tight", legacy.TIGHT_PAD_FACTOR, legacy.TIGHT_MIN_PAD),
                                      ("wide", legacy.WIDE_PAD_FACTOR, legacy.WIDE_MIN_PAD)):
            ref_box = list(pad_box(bx, by, bw, bh, factor, min_pad, w, h))
            if crop_source == "aligned":
                src_box, img, src = ref_box, aligned, "aligned"
            else:
                src_box, img, src = _transform_box(ref_box, H_inv, cw, ch), current, "original"
            sx1, sy1, sx2, sy2 = src_box
            crop = img[sy1:sy2, sx1:sx2]
            entry = {"src": src, "bbox_src_xyxy": src_box, "bbox_ref_xyxy": ref_box, "file": None,
                     "size_ok": bool(sx2 - sx1 >= 40 and sy2 - sy1 >= 40)}
            if out_dir is not None and crop.size:
                out_dir.mkdir(parents=True, exist_ok=True)
                f = out_dir / f"region{r['region_id']:03d}_{kind}.png"
                cv2.imwrite(str(f), crop)
                entry["file"] = f.name
            arrays[(r["region_id"], kind)] = crop
            crops[kind] = entry
        out_regions.append({k: r[k] for k in ("region_id", "bbox_ref_xyxy", "area_px", "area_mm2", "mean_d",
                                              "max_d", "score") if k in r} | {"crops": crops})
    package = {"schema": SCHEMA, "pipeline": pipeline, "reference_image": reference_image,
               "current_image": current_image, "aligned_image": aligned_image,
               "H_ref_from_cur": None if H_ref_from_cur is None else np.asarray(H_ref_from_cur).tolist(),
               "roi_polygon_ref_px": None if roi_polygon is None else np.asarray(roi_polygon).tolist(),
               "px_per_mm": px_per_mm, "ssim_roi": ssim_roi, "threshold": threshold, "crop_source": crop_source,
               "regions": out_regions}
    if out_dir is not None:
        (out_dir / "ai_input_package.json").write_text(json.dumps(package, indent=1), encoding="utf-8")
    return package, arrays


def map_box_to_ref(box_src, src: str, H_ref_from_cur) -> list[float]:
    """A box in source-image coordinates -> reference frame (aligned = identity)."""
    if src == "aligned" or H_ref_from_cur is None:
        return [float(v) for v in box_src]
    x1, y1, x2, y2 = box_src
    pts = np.float64([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
    t = cv2.perspectiveTransform(pts, np.asarray(H_ref_from_cur, float)).reshape(-1, 2)
    return [float(t[:, 0].min()), float(t[:, 1].min()), float(t[:, 0].max()), float(t[:, 1].max())]


def map_detections_to_ref(dets: list[dict], crop_meta: dict, H_ref_from_cur=None) -> list[dict]:
    """Detections whose ``bbox_xyxy`` is in full SOURCE-image coordinates
    (ModelService adds the crop origin) -> reference frame."""
    out = []
    for d in dets:
        dd = dict(d)
        dd["bbox_src_xyxy"] = list(d["bbox_xyxy"])
        dd["bbox_ref_xyxy"] = map_box_to_ref(d["bbox_xyxy"], crop_meta.get("src", "aligned"), H_ref_from_cur)
        out.append(dd)
    return out


def run_two_phase_yolo(package: dict, arrays: dict, *, model_service, model_name: str, ssim_source: np.ndarray,
                       reference: np.ndarray) -> list[dict]:
    """cita-baseline Phase A (tight) / Phase B (wide) / NMS on package crops."""
    H = package.get("H_ref_from_cur")
    tight_list, wide_candidates = [], []
    regions = package["regions"]
    for r in regions:
        x1, y1, x2, y2 = r["bbox_ref_xyxy"]
        c = r["crops"]["tight"]
        if not c["size_ok"]:
            continue
        r_ssim = legacy.compute_region_ssim(ssim_source, reference, x1, y1, x2, y2, padding=0)
        sx1, sy1 = c["bbox_src_xyxy"][:2]
        tight_list.append((arrays[(r["region_id"], "tight")], sx1, sy1, "tight", float(r_ssim), r["region_id"]))
    tight = model_service.detect_crops_batch(model_name, [t[:5] for t in tight_list], conf=legacy.YOLO_CONF,
                                             sub_batch=legacy.YOLO_SUB_BATCH) if tight_list else []
    src_of = {"tight": regions[0]["crops"]["tight"]["src"] if regions else "aligned"}
    tight_ref = map_detections_to_ref(tight, {"src": src_of["tight"]}, H)
    wide_list = []
    for r in regions:
        x1, y1, x2, y2 = r["bbox_ref_xyxy"]
        bw, bh = x2 - x1, y2 - y1
        cx, cy = x1 + bw / 2, y1 + bh / 2
        max_conf = max((d["confidence"] for d in tight_ref
                        if abs((d["bbox_ref_xyxy"][0] + d["bbox_ref_xyxy"][2]) / 2 - cx) < bw + legacy.TIGHT_MIN_PAD
                        and abs((d["bbox_ref_xyxy"][1] + d["bbox_ref_xyxy"][3]) / 2 - cy) < bh + legacy.TIGHT_MIN_PAD),
                       default=0.0)
        if max_conf >= legacy.EARLY_EXIT_CONF:
            continue
        c = r["crops"]["wide"]
        if not c["size_ok"]:
            continue
        r_ssim = legacy.compute_region_ssim(ssim_source, reference, x1, y1, x2, y2, padding=0)
        sx1, sy1 = c["bbox_src_xyxy"][:2]
        wide_list.append((arrays[(r["region_id"], "wide")], sx1, sy1, "wide", float(r_ssim)))
    wide = model_service.detect_crops_batch(model_name, wide_list, conf=legacy.YOLO_CONF,
                                            sub_batch=legacy.YOLO_SUB_BATCH) if wide_list else []
    wide_ref = map_detections_to_ref(wide, {"src": regions[0]["crops"]["wide"]["src"] if regions else "aligned"}, H)
    # NMS in the reference frame (class-aware, IoU 0.45) as cita-baseline.
    combined = legacy.nms_detections([dict(d, bbox_xyxy=d["bbox_ref_xyxy"]) for d in tight_ref + wide_ref],
                                     legacy.NMS_IOU_THRESH)
    return [{"class_name": d["class_name"], "confidence": d["confidence"],
             "bbox_xyxy": [int(round(v)) for v in d["bbox_ref_xyxy"]], "bbox_ref_xyxy": d["bbox_ref_xyxy"],
             "bbox_src_xyxy": d["bbox_src_xyxy"], "from_crop": d.get("from_crop", ""),
             "region_ssim": d.get("region_ssim", 0.0)} for d in combined]
