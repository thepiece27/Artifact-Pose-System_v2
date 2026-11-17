"""E3 pair sets: ``pairs.json`` (schema ``cita.pairs.v1``) + per-pair GT
(``cita.change_gt.v1``, Appendix B.4).

    {"schema": "cita.pairs.v1", "session_id": "...", "split": "val" | "test",
     "roi": {"polygon_ref_px": [[x, y], ...], "size_mm": [w, h]} | null,
     "px_per_mm": 4.5 | null,
     "pairs": [{"pair_id": "...", "golden_image": "golden.png", "current_image": "cur_000.png",
                "gt": "gt_000.json", "offset_mm": 0.0, "offset_deg": 0.0}]}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PAIRS_SCHEMA = "cita.pairs.v1"


class PairSet:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        path = self.root / "pairs.json" if self.root.is_dir() else self.root
        self.root = path.parent
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != PAIRS_SCHEMA:
            raise ValueError(f"{path}: schema must be {PAIRS_SCHEMA}")

    @property
    def split(self) -> str:
        return self.manifest.get("split", "na")

    @property
    def session_id(self) -> str:
        return self.manifest.get("session_id", self.root.name)

    def roi(self, override: str | None = None) -> dict | None:
        if override:
            from app.modules.change_detection.roi import load_roi
            return load_roi(override)
        r = self.manifest.get("roi")
        if not r:
            p = self.root / "roi.json"
            if p.exists():
                from app.modules.change_detection.roi import load_roi
                return load_roi(p)
            return None
        return {**r, "polygon_ref_px": np.asarray(r["polygon_ref_px"], float)}

    def pairs(self) -> list[dict]:
        out = []
        for p in self.manifest["pairs"]:
            gt = json.loads((self.root / p["gt"]).read_text(encoding="utf-8")) if p.get("gt") else None
            out.append({**p, "golden_path": self.root / p["golden_image"],
                        "current_path": self.root / p["current_image"], "gt_data": gt,
                        "split": (gt or {}).get("split", self.split)})
        return out
