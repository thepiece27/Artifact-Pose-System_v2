#!/usr/bin/env python3
"""E3 offline replay: run inspection pipelines P0-P4 on (golden, current) pairs.

    python tools/experiments/replay_change.py --pairs data/E3/test_P101 --pipelines P0,P1,P2,P3,P4 \
        --thresholds configs/thresholds.yaml --model /path/to/model.pt   # or --mock-yolo (tests only)

One JSONL record per (pair, pipeline) in ``logs/experiments/E3/<run_id>.jsonl``
and the AIInputPackage of every pair in ``--out-dir``. OpenCV's RNG (used by
the legacy FLANN matcher) is reseeded before every run, so results do not
depend on the processing order.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SERVER, add_common_args, make_logger, setup  # noqa: E402
from _pairs import PairSet  # noqa: E402

from app.experiments import registry  # noqa: E402
from app.experiments.profiles import load_profile  # noqa: E402
from app.experiments.trial_log import sha256_file  # noqa: E402
from app.modules.change_detection.pipelines import run_pipeline  # noqa: E402
from app.modules.change_detection.regions import load_thresholds  # noqa: E402


def load_model_service(model: str | None, mock: bool):
    if mock:
        sys.path.insert(0, str(SERVER / "tests" / "cita"))
        from mocks import MockModelService
        return MockModelService(), "mock"
    if not model:
        return None, None
    from app.services.model_service import ModelService
    path = Path(model).resolve()
    settings = SimpleNamespace(model_dir=path.parent, default_ai_model_path="", default_ai_model_name="default",
                               max_model_bytes=2 * 1024 ** 3)
    svc = ModelService(settings)
    svc.load_model("default", str(path), backend="yolo")
    return svc, sha256_file(path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", nargs="+", required=True)
    p.add_argument("--pipelines", default="P0,P1,P2,P3,P4")
    p.add_argument("--thresholds", default=None, help="thresholds.yaml (default: each profile's file)")
    p.add_argument("--roi", default=None, help="roi.json overriding the pair-set ROI")
    p.add_argument("--model", default=None, help="YOLO .pt weights provided by the team")
    p.add_argument("--mock-yolo", action="store_true", help="deterministic mock detector (tests only)")
    p.add_argument("--out-dir", default="results/E3")
    add_common_args(p)
    args = p.parse_args(argv)
    base, seed, _ = setup(args)
    model_service, model_hash = load_model_service(args.model, args.mock_yolo)
    log = make_logger("E3", args, base)
    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = SERVER / out_root
    codes = [c.strip() for c in args.pipelines.split(",") if c.strip()]
    for code in codes:
        prof = load_profile(registry.get(code).default_profile)
        cfg = prof.inspection
        thresholds = None
        if cfg.threshold.mode == "fixed" and cfg.threshold.tau is None:
            thresholds = load_thresholds(args.thresholds or prof.resolve_path(cfg.threshold.file))
        if code in ("P0", "P1", "P3", "P4") and model_service is None:
            print(f"[{code}] no YOLO model (--model or --mock-yolo): detections will be empty", file=sys.stderr)
        for path in args.pairs:
            ps = PairSet(Path(path))
            roi = ps.roi(args.roi)
            for q in ps.pairs():
                inp = {"pairs": ps.session_id, "pair_id": q["pair_id"], "split": q["split"],
                       "golden_image": str(q["golden_path"]), "current_image": str(q["current_path"]),
                       "sha256": [sha256_file(q["golden_path"]), sha256_file(q["current_path"])],
                       "offset_mm": q.get("offset_mm"), "offset_deg": q.get("offset_deg")}
                with log.trial(arm=code, input=inp, profile_name=prof.name, profile_hash_arm=prof.profile_hash,
                               model_sha256=model_hash) as out:
                    cv2.setRNGSeed(seed)
                    ref, cur = cv2.imread(str(q["golden_path"])), cv2.imread(str(q["current_path"]))
                    res = run_pipeline(code, reference=ref, current=cur, cfg=cfg, roi=roi,
                                       px_per_mm=ps.manifest.get("px_per_mm"), thresholds=thresholds,
                                       model_service=model_service, out_dir=out_root / log.run_id / code / q["pair_id"],
                                       reference_path=q["golden_path"], current_path=q["current_path"],
                                       artifact_id=q["pair_id"])
                    for key in ("pipeline", "regions", "detections", "image_score", "verified_regions",
                                "image_score_verified", "ssim", "roi_afd_px", "ecc_ok",
                                "ai_input_package_path", "damage_pct", "threshold", "alignment_quality",
                                "sift_inliers", "registration", "photometric", "registration_eval_only"):
                        if key in res:
                            out[key] = res[key]
                    out["_status"] = "ok" if res.get("status") in (None, "ok") else res["status"]
    print(f"E3 replay written: {log.path} ({log.count} records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
