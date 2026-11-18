#!/usr/bin/env python3
"""E2 relocalisation trials for arms C-A ... C-D (virtual or real rig).

    # virtual rig, two mount conditions (rail aligned / painting yawed 25 deg):
    python tools/experiments/run_reloc_trials.py --virtual --mount aligned --paintings P001,P002 \
        --n-perturb 15 --arms C-A,C-B,C-C,C-D --seed 2027 [--noise-from configs/noise_model.yaml]
    python tools/experiments/run_reloc_trials.py --virtual --mount yawed25 ...

    # real rig through the running server (operator login, MQTT via the server):
    python tools/experiments/run_reloc_trials.py --mount-condition yawed25 --painting-yaw-deg 25 \
        --paintings P001 --artifact-id <id> --device <code> --api http://127.0.0.1:8000 \
        --username operator --password-env CITA_API_PASSWORD --n-perturb 15 --arms C-A,C-B,C-C,C-D \
        --seed 2027 --i-know-hardware-moves

Perturbations are uniform in |dX| [5,40] mm, |dZ| [5,30] mm, |dpan| [1,6] deg,
|dtilt| [1,5] deg with random signs. For each (painting, perturbation) block
the arm order is shuffled (seeded). Every trial: back to golden -> perturb ->
run the arm -> log every iteration and a summary (``logs/experiments/E2``).

Jacobian re-use (R3): every arm has its own J file; ``--probe-reuse per_block``
(default) probes once per (painting, arm) block of THIS run and re-uses J for
the rest of the block, ``never`` probes every trial, ``ttl`` keeps the server
behaviour. Every record carries mount_condition, painting_yaw_deg,
probe_reuse and block_id. Success is judged offline by ``eval_reloc.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SERVER, add_common_args, make_logger, setup  # noqa: E402

from app.experiments import registry  # noqa: E402
from app.experiments.profiles import load_profile  # noqa: E402
from app.modules.artifact_pose.alignment_loop import build_session, run_alignment  # noqa: E402
from app.modules.sim.virtual_rig import (RigParams, VirtualPort, VirtualRig, noise_from_file,  # noqa: E402
                                         nominal_fixed_signs)

RANGES = {"x_mm": (5, 40), "z_mm": (5, 30), "pan_deg": (1, 6), "tilt_deg": (1, 5)}


def sample_perturbation(rng: np.random.Generator) -> dict:
    return {k: float(rng.choice([-1, 1]) * rng.uniform(lo, hi)) for k, (lo, hi) in RANGES.items()}


def arm_profile(arm: str, overrides: dict | None):
    return load_profile(registry.get(arm).default_profile, overrides=overrides or None)


def block_id(run_id: str, painting: str, arm: str) -> str:
    return f"{run_id}:{painting}:{arm}"


def condition_record(args) -> dict:
    return {"mount_condition": args.mount_condition, "painting_yaw_deg": args.painting_yaw_deg,
            "probe_reuse": args.probe_reuse}


# ---------------------------------------------------------------------------
# Virtual rig
# ---------------------------------------------------------------------------

def virtual_params(args) -> tuple[RigParams, dict]:
    extra = json.loads(args.rig_params) if args.rig_params else {}
    noise_info = None
    if args.noise_from:
        noise_info = noise_from_file(args.noise_from)
        extra.setdefault("noise_c_mm", noise_info["noise_c_mm"])
        extra.setdefault("noise_rot_deg", noise_info["noise_rot_deg"])
    if args.pwm_step_deg is not None:
        extra["pwm_step_deg"] = args.pwm_step_deg
    params = RigParams.preset(args.mount, **extra)
    return params, {"mount_preset": args.mount, "noise_from": noise_info, "rig_params": params.__dict__}


def run_virtual(args, base, log, rng) -> list[dict]:
    params, rig_info = virtual_params(args)
    signs = nominal_fixed_signs(params)
    jac_dir = Path(args.jacobian_dir or tempfile.mkdtemp(prefix="cita_jacobian_"))
    results = []
    paintings = [p.strip() for p in args.paintings.split(",") if p.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    overrides = json.loads(args.overrides) if args.overrides else {}
    for p_index, painting in enumerate(paintings):
        for k in range(args.n_perturb):
            pert = sample_perturbation(rng)
            order = list(arms)
            rng.shuffle(order)
            for arm in order:
                ov = json.loads(json.dumps(overrides))
                prof = arm_profile(arm, None)
                if prof.control.controller == "fixed_mapping" and args.fixed_signs == "nominal":
                    ov.setdefault("control", {})["signs"] = signs
                ov.setdefault("control", {})["jacobian_dir"] = str(jac_dir)
                prof = arm_profile(arm, ov)
                rig = VirtualRig(params, seed=args.seed * 1000 + p_index * 100 + k)
                rig.perturb(**pert)
                start_true = rig.true_errors()
                bid = block_id(log.run_id, painting, arm)
                session = build_session(prof, arm=arm, device_id="virtual", artifact_id=painting,
                                        server_root=jac_dir.parent, probe_reuse=args.probe_reuse, block_id=bid)
                session.meta.update(painting=painting, perturbation_index=k)
                inp = {"painting_id": painting, "perturbation_index": k, "perturbation": pert, "mode": "virtual",
                       "block_id": bid, **condition_record(args), **rig_info}
                summary = run_alignment(VirtualPort(rig), session, logger=log, log_input=inp)
                summary.update(true_final=rig.true_errors(), true_start=start_true, profile_hash=prof.profile_hash,
                               tol_t_mm=session.cfg.tol_t_mm, tol_R_deg=session.cfg.tol_R_deg,
                               total_moves=summary["control_iterations"] + summary["n_probe_moves"])
                log.log(arm=arm, input=inp, output=summary, status="success" if summary["success_est"] else "failed",
                        error=summary["stop_reason"], record_type="summary")
                results.append({"arm": arm, **summary})
    return results


# ---------------------------------------------------------------------------
# Real rig (through the server API)
# ---------------------------------------------------------------------------

class Api:
    def __init__(self, base: str, token: str | None, timeout: float = 15.0) -> None:
        self.base, self.token, self.timeout = base.rstrip("/"), token, timeout

    def call(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")

    @classmethod
    def login(cls, base: str, username: str, password: str) -> "Api":
        tok = cls(base, None).call("POST", "/api/v1/auth/login", {"username": username, "password": password})["access_token"]
        return cls(base, tok)


def wait_ack(api: Api, device: str, task_id: str, timeout_s: float) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        acks = api.call("GET", f"/api/v1/devices/{device}/acks?limit=50")
        for item in reversed(acks.get("acks", acks.get("items", [])) or []):
            payload = item.get("payload") or {}
            if payload.get("task_id") == task_id:
                return payload
        time.sleep(1.0)
    raise TimeoutError(f"No ack for {task_id} within {timeout_s}s")


def retire_stale_jacobians(control_dir: Path, device: str, artifact: str) -> list[str]:
    """Rename every J file of this device/artifact to ``*.stale`` (a new run never
    re-uses a J measured in a previous run)."""
    from app.modules.artifact_pose.control import _safe
    renamed = []
    if not control_dir.is_dir():
        return renamed
    for f in sorted(control_dir.glob(f"jacobian_{_safe(device)}_{_safe(artifact)}_*.json")):
        target = f.with_name(f.name + f".{int(time.time())}.stale")
        f.rename(target)
        renamed.append(f"{f.name} -> {target.name}")
    return renamed


def run_real(args, base, log, rng) -> list[dict]:  # pragma: no cover - needs the rig
    if not args.i_know_hardware_moves:
        raise SystemExit("Real-rig trials move hardware: add --i-know-hardware-moves (or use --virtual)")
    if not args.mount_condition:
        raise SystemExit("Real mode needs --mount-condition (e.g. aligned / yawed25) and --painting-yaw-deg")
    password = os.environ.get(args.password_env or "", "")
    if not (args.username and password and args.device and args.artifact_id):
        raise SystemExit("Real mode needs --device, --artifact-id, --username and --password-env")
    control_dir = Path(args.server_data_dir) / "control"
    renamed = retire_stale_jacobians(control_dir, args.device, args.artifact_id)
    log.log(arm="jacobian_cleanup", input={"control_dir": str(control_dir), "device": args.device,
                                           "artifact_id": args.artifact_id}, output={"renamed": renamed},
            record_type="setup")
    api = Api.login(args.api, args.username, password)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    painting = args.paintings.split(",")[0].strip()
    results = []
    for k in range(args.n_perturb):
        pert = sample_perturbation(rng)
        order = list(arms)
        rng.shuffle(order)
        for arm in order:
            trial_id = f"{log.run_id}_{painting}_{k:03d}_{arm}"
            bid = block_id(log.run_id, painting, arm)
            inp = {"painting_id": painting, "perturbation_index": k, "perturbation": pert, "mode": "real",
                   "trial_id": trial_id, "device": args.device, "artifact_id": args.artifact_id, "block_id": bid,
                   **condition_record(args)}
            t0 = time.time()
            try:
                r = api.call("POST", f"/api/v1/devices/{args.device}/queue_move", {"action": "reset"})
                wait_ack(api, args.device, r["task_id"], args.ack_timeout)
                move = {"action": "move", "x_steps": abs(int(round(pert["x_mm"] * args.steps_per_mm))),
                        "x_dir": 1 if pert["x_mm"] >= 0 else -1,
                        "z_steps": abs(int(round(pert["z_mm"] * args.steps_per_mm))),
                        "z_dir": 1 if pert["z_mm"] >= 0 else -1, "yaw_delta": pert["pan_deg"],
                        "pitch_delta": pert["tilt_deg"], "artifact_id": args.artifact_id,
                        "motion_profile": {"float_angles": True, "settle_s": 1.0}}
                r = api.call("POST", f"/api/v1/devices/{args.device}/queue_move", move)
                perturb_ack = wait_ack(api, args.device, r["task_id"], args.ack_timeout)
                api.call("POST", f"/workflows/{args.device}/start-alignment",
                         {"artifact_id": args.artifact_id, "cita_arm": arm, "cita_trial_id": trial_id,
                          "keep_pose": True, "use_latest_metadata": False,
                          "cita_probe_reuse": args.probe_reuse, "cita_block_id": bid})
                meta = None
                while time.time() - t0 < args.trial_timeout:
                    m = (api.call("GET", f"/workflows/{args.device}/latest-capture-metadata") or {}).get("metadata") or {}
                    if m.get("cita_trial_id") == trial_id and m.get("alignment_status") in ("complete", "failed"):
                        meta = m
                        break
                    time.sleep(2.0)
                summary = dict((meta or {}).get("cita_summary") or {})
                summary.update(success_est=bool(meta and meta["alignment_status"] == "complete"),
                               final_aligned_path=(meta or {}).get("final_aligned_path"),
                               perturb_ack=perturb_ack, wall_time_s=time.time() - t0,
                               stop_reason=summary.get("stop_reason") or ("trial_timeout" if meta is None else None),
                               total_moves=int(summary.get("control_iterations") or 0) + int(summary.get("n_probe_moves") or 0))
                status = "success" if summary["success_est"] else "failed"
            except (urllib.error.URLError, TimeoutError, KeyError) as exc:
                summary, status = {"success_est": False, "stop_reason": f"error:{exc}"}, "error"
            log.log(arm=arm, input=inp, output=summary, status=status, error=summary.get("stop_reason"),
                    record_type="summary")
            results.append({"arm": arm, **summary})
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paintings", default="P001")
    p.add_argument("--n-perturb", type=int, default=15)
    p.add_argument("--arms", default="C-A,C-B,C-C,C-D")
    p.add_argument("--virtual", action="store_true")
    p.add_argument("--mount", choices=("aligned", "yawed25"), default="aligned",
                   help="virtual mount preset (slider misalignment 3 / 25 deg)")
    p.add_argument("--mount-condition", default=None,
                   help="label of the mount condition (virtual default: --mount; REQUIRED on the real rig)")
    p.add_argument("--painting-yaw-deg", type=float, default=None,
                   help="yaw of the painting stand w.r.t. the rail (virtual default: the preset value)")
    p.add_argument("--probe-reuse", choices=("per_block", "never", "ttl"), default="per_block")
    p.add_argument("--rig-params", default=None, help="JSON RigParams overrides (virtual)")
    p.add_argument("--noise-from", default=None,
                   help="virtual pose noise from a noise_model.yaml (VCE) or an E1 eval_pose JSON (M5)")
    p.add_argument("--pwm-step-deg", type=float, default=None, help="virtual PWM quantisation (real PCA9685: 0.586)")
    p.add_argument("--fixed-signs", choices=("nominal", "profile"), default="nominal",
                   help="virtual: give C-A/C-B the signs that are correct for the ideal rig")
    p.add_argument("--overrides", default=None, help="JSON profile overrides applied to every arm")
    p.add_argument("--jacobian-dir", default=None)
    p.add_argument("--server-data-dir", default=str(SERVER / "data"),
                   help="real mode: server DATA_DIR (stale J files in <dir>/control are renamed)")
    p.add_argument("--api", default="http://127.0.0.1:8000")
    p.add_argument("--device", default=None)
    p.add_argument("--artifact-id", default=None)
    p.add_argument("--username", default=None)
    p.add_argument("--password-env", default="CITA_API_PASSWORD")
    p.add_argument("--steps-per-mm", type=float, default=800.0)
    p.add_argument("--ack-timeout", type=float, default=60.0)
    p.add_argument("--trial-timeout", type=float, default=600.0)
    p.add_argument("--i-know-hardware-moves", action="store_true")
    add_common_args(p)
    args = p.parse_args(argv)
    base, seed, rng = setup(args)
    args.seed = seed            # resolved (CLI or profile seed); used for rig seeds
    if args.virtual:
        args.mount_condition = args.mount_condition or args.mount
        if args.painting_yaw_deg is None:
            args.painting_yaw_deg = RigParams.preset(args.mount).slider_misalign_deg
    log = make_logger("E2", args, base)
    results = run_virtual(args, base, log, rng) if args.virtual else run_real(args, base, log, rng)
    by_arm: dict = {}
    for r in results:
        by_arm.setdefault(r["arm"], []).append(r)
    for arm, rs in sorted(by_arm.items()):
        moves = [r.get("total_moves", 0) for r in rs]
        print(f"[{args.mount_condition}] {arm}: est. success {np.mean([r.get('success_est', False) for r in rs]):.2f} "
              f"median total moves {np.median(moves) if moves else None} "
              f"probes {sum(1 for r in rs if r.get('n_probe_moves'))}/{len(rs)}")
    print(f"E2 log: {log.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
