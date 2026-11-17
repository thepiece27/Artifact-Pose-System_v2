#!/usr/bin/env python3
"""Create ``roi.json`` (painting ROI in the reference image frame).

    python tools/experiments/define_roi.py --image golden_left.png --auto --size-mm 297,210 --out roi.json
    python tools/experiments/define_roi.py --image golden_left.png --polygon "812,400;2990,410;2985,1950;805,1940" --out roi.json
    python tools/experiments/define_roi.py --image golden_left.png --diamond-rect -98.5,-105,198.5,105 --calib cal.yaml --out roi.json

``--auto`` finds the largest convex quadrilateral (A4 sheet on the formex);
``--diamond-rect`` projects a rectangle given in Diamond-plane millimetres.
Server use: copy to ``uploads/golden_poses/<artifact_id>/roi.json``.
A preview PNG is written next to the JSON.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402,F401

from app.modules.change_detection.roi import auto_detect_quad, roi_from_diamond, save_roi  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--auto", action="store_true")
    g.add_argument("--polygon", help="x,y;x,y;... in reference pixels")
    g.add_argument("--diamond-rect", help="x0,y0,x1,y1 in Diamond-plane mm (Y up)")
    p.add_argument("--size-mm", default=None, help="physical painting size w,h in mm (enables px_per_mm)")
    p.add_argument("--calib", default=None)
    args = p.parse_args(argv)
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Cannot read {args.image}")
    if args.auto:
        poly, source = auto_detect_quad(img), "auto_quad"
        if poly is None:
            raise SystemExit("No quadrilateral found; use --polygon")
    elif args.polygon:
        poly = np.array([[float(v) for v in pt.split(",")] for pt in args.polygon.split(";")])
        source = "manual"
    else:
        from app.modules.artifact_pose.common import detect_diamond, read_calibration
        cal = read_calibration(args.calib or _common.DEFAULT_CALIBRATION)
        d = detect_diamond(img, cal["K"], cal["D"])
        if d is None:
            raise SystemExit("Diamond not detected in the reference image")
        poly = roi_from_diamond(cal["K"], cal["D"], d["rvec"], d["tvec"], [float(v) for v in args.diamond_rect.split(",")])
        source = "diamond_projection"
    size = [float(v) for v in args.size_mm.split(",")] if args.size_mm else None
    out = save_roi(args.out, poly, size_mm=size, source=source, extra={"image": Path(args.image).name})
    preview = img.copy()
    cv2.polylines(preview, [np.rint(poly).astype(np.int32)], True, (0, 255, 255), 3)
    cv2.imwrite(str(out.with_suffix(".png")), preview)
    print(f"ROI ({source}) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
