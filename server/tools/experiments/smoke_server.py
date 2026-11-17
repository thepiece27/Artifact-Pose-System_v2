#!/usr/bin/env python3
"""In-process server smoke test for a CITA profile (stabilisation step 4).

    python tools/experiments/smoke_server.py --profile pbl_legacy
    python tools/experiments/smoke_server.py --profile proposed

Starts the FastAPI app with a throw-away SQLite DB, data dir and model dir,
the synthetic fixture calibration, no MQTT broker and a MOCK YOLO; registers a
user, then calls GET /pose/health, initialises a golden from the fixture
stereo pair, corrects one image, uploads a reference and an inspection image.
Every value read from server/.env is overridden by explicit throw-away values
for the keys used here; nothing secret is printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

SERVER = Path(__file__).resolve().parents[2]
FIX = SERVER / "tests" / "fixtures" / "synthetic"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", default="pbl_legacy")
    args = p.parse_args(argv)
    tmp = Path(tempfile.mkdtemp(prefix="cita_smoke_"))
    profile = args.profile
    if profile == "proposed":
        # Test-only noise model / threshold (real ones come from E0 / validation).
        (tmp / "noise_model.yaml").write_text("schema: cita.noise_model.v1\ndiamond_sigma_px: 0.15\norb_sigma_px: 0.8\n")
        (tmp / "smoke_proposed.yaml").write_text(
            "name: smoke_proposed\nextends: proposed\n"
            f"pose: {{noise_model: '{(tmp / 'noise_model.yaml').as_posix()}'}}\n"
            "inspection: {threshold: {mode: fixed, tau: 0.3}}\n")
        profile = str(tmp / "smoke_proposed.yaml")
    os.environ.update({
        "CITA_PROFILE": profile, "AUTH_DATABASE_URL": f"sqlite:///{(tmp / 'smoke.db').as_posix()}",
        "DATA_DIR": str(tmp / "data"), "MODEL_DIR": str(tmp / "model"), "ENVIRONMENT": "development",
        "ARTIFACT_CAMERA_PARAMS": str(FIX / "calibration.yaml"), "MQTT_HOST": "127.0.0.1", "MQTT_PORT": "1",
        "AUTH_SECRET_KEY": "smoke-test-secret-key-not-used-anywhere-else-0123456789",
        "ALLOW_PUBLIC_REGISTRATION": "true", "RUN_AI_ON_UPLOAD": "false", "AUTO_DISPATCH_POSE_COMMAND": "false",
        "DEVICE_API_KEY": "smoke-device-key", "MQTT_USERNAME": "", "MQTT_PASSWORD": "",
    })
    sys.path.insert(0, str(SERVER))
    sys.path.insert(0, str(SERVER / "tests" / "cita"))
    import socket
    import threading
    import time
    import urllib.error
    import urllib.request
    import uuid

    import uvicorn
    from mocks import MockModelService
    import app.main as main_mod

    main_mod.container.model_service = MockModelService()
    main_mod.container.inspection_service._model_service = main_mod.container.model_service
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(main_mod.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"

    def call(method, path, *, token=None, json_body=None, fields=None, files=None):
        headers, data = {}, None
        if json_body is not None:
            data, headers["Content-Type"] = json.dumps(json_body).encode(), "application/json"
        elif files is not None or fields is not None:
            boundary = uuid.uuid4().hex
            crlf = "\r\n"
            parts = []
            for k, v in (fields or {}).items():
                parts.append(f'--{boundary}{crlf}Content-Disposition: form-data; name="{k}"{crlf}{crlf}{v}{crlf}'.encode())
            for k, (name, path_) in (files or {}).items():
                head = (f'--{boundary}{crlf}Content-Disposition: form-data; name="{k}"; filename="{name}"{crlf}'
                        f"Content-Type: image/png{crlf}{crlf}")
                parts.append(head.encode() + Path(path_).read_bytes() + crlf.encode())
            data = b"".join(parts) + f"--{boundary}--{crlf}".encode()
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.status, json.loads(resp.read() or b"null")
        except urllib.error.HTTPError as exc:
            raw = exc.read() or b"null"
            try:
                return exc.code, json.loads(raw)
            except ValueError:
                return exc.code, raw.decode("utf-8", "replace")[:500]

    for _ in range(100):
        try:
            if call("GET", "/health")[0] == 200:
                break
        except OSError:
            time.sleep(0.2)
    results = {}
    try:
        st, _ = call("POST", "/api/v1/auth/register", json_body={"username": "smoke_user", "password": "smoke-pass-123"})
        assert st in (201, 400), st
        st, body = call("POST", "/api/v1/auth/login", json_body={"username": "smoke_user", "password": "smoke-pass-123"})
        assert st == 200, body
        tok = body["access_token"]
        st, body = call("GET", "/pose/health", token=tok)
        assert st == 200, body
        results["pose_health"] = {k: body.get(k) for k in ("ok", "available", "cita_profile")}
        st, art = call("POST", "/api/v1/artifacts", token=tok, json_body={"name": "smoke artifact"})
        assert st == 201, art
        aid = art["id"]
        st, body = call("POST", "/pose/initialize_golden", token=tok, fields={"artifact_id": aid},
                        files={"left_file": ("l.png", FIX / "golden_left.png"), "right_file": ("r.png", FIX / "golden_right.png")})
        results["initialize_golden"] = st
        assert st == 200, body
        st, body = call("POST", "/pose/correct", token=tok, files={"file": ("c.png", FIX / "current_01.png")})
        results["pose_correct"] = st
        # /pose/correct uses the global golden (no artifact id): 400 = clean "golden not found".
        assert st in (200, 400), body
        st, body = call("POST", f"/api/v1/artifacts/{aid}/reference", token=tok, files={"file": ("ref.png", FIX / "golden_left.png")})
        assert st == 200, body
        st, insp = call("POST", f"/api/v1/artifacts/{aid}/inspect", token=tok, files={"file": ("cur.png", FIX / "inspect_damaged.png")})
        assert st == 200, insp
        det = insp.get("detections_json")
        det = json.loads(det) if isinstance(det, str) else (det or {})
        results["inspect"] = {"status": insp.get("status"), "ssim": insp.get("ssim_score"),
                              "analysis_status": det.get("analysis_status"), "pipeline": det.get("pipeline", "P0")}
        assert det.get("analysis_status") == "ok", det
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    print(json.dumps({"profile": args.profile, **results}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
