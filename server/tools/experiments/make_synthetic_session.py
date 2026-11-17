#!/usr/bin/env python3
"""Generate synthetic sessions for smoke tests / dry runs (never real data).

    python tools/experiments/make_synthetic_session.py pose   --root /tmp/cita --session-id E1_SYN
    python tools/experiments/make_synthetic_session.py burst  --root /tmp/cita --session-id E02_SYN --frames 10
    python tools/experiments/make_synthetic_session.py change --root /tmp/cita --split val

Output layout matches ``capture_session.py`` (manifest ``cita.session.v1``);
change pairs use ``pairs.json`` (``cita.pairs.v1``) + GT ``cita.change_gt.v1``.
A ``truth.json`` with the true motion is written next to pose manifests.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SERVER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SERVER / "tests"))
import synth  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=("pose", "burst", "change"))
    p.add_argument("--root", required=True)
    p.add_argument("--session-id", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--frames", type=int, default=3, help="frames per pose (pose) / burst length (burst)")
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--unchanged", type=int, default=3)
    p.add_argument("--changed", type=int, default=3)
    args = p.parse_args(argv)
    root = Path(args.root)
    if args.kind == "pose":
        out = synth.make_pose_session(root, seed=args.seed, session_id=args.session_id or "E1_SYNTH",
                                      frames_per_pose=args.frames, width=args.width)
    elif args.kind == "burst":
        out = synth.make_pose_session(root, seed=args.seed, session_id=args.session_id or "E02_SYNTH",
                                      offsets=[{"x_mm": 0.0, "z_mm": 0.0, "pan_deg": 0.0, "tilt_deg": 0.0}],
                                      frames_per_pose=args.frames, width=args.width)
    else:
        out = synth.make_change_pairs(root, seed=args.seed, split=args.split, n_unchanged=args.unchanged,
                                      n_changed=args.changed, session_id=args.session_id, width=args.width)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
