from __future__ import annotations

import copy
import json
import time
import uuid
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Any


class CommandService:
    def __init__(self, ack_history_limit: int = 200, state_file: Path | None = None) -> None:
        self._queue_by_device: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._status_by_device: dict[str, dict[str, Any]] = {}
        self._ack_history_by_device: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._latest_capture_metadata_by_device: dict[str, dict[str, Any]] = {}
        self._ack_history_limit = max(1, ack_history_limit)
        self._lock = Lock()
        self._state_file = state_file
        self._load_state()

    def _load_state(self) -> None:
        if self._state_file is None or not self._state_file.exists():
            return
        try:
            data = json.loads(self._state_file.read_text(encoding="utf-8"))
            if isinstance(data.get("queue"), dict):
                self._queue_by_device.update({str(k): list(v) for k, v in data["queue"].items() if isinstance(v, list)})
            if isinstance(data.get("status"), dict):
                self._status_by_device.update(data["status"])
            if isinstance(data.get("acks"), dict):
                for key, value in data["acks"].items():
                    if isinstance(value, list):
                        self._ack_history_by_device[str(key)] = value[-self._ack_history_limit:]
            if isinstance(data.get("latest_capture"), dict):
                self._latest_capture_metadata_by_device.update(data["latest_capture"])
        except Exception:
            # A corrupt operational cache must not prevent API startup.
            return

    def _persist_locked(self) -> None:
        if self._state_file is None:
            return
        payload = {
            "queue": dict(self._queue_by_device),
            "status": self._status_by_device,
            "acks": dict(self._ack_history_by_device),
            "latest_capture": self._latest_capture_metadata_by_device,
        }
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            temp = self._state_file.with_suffix(self._state_file.suffix + ".tmp")
            temp.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
            temp.replace(self._state_file)
        except Exception:
            return

    @staticmethod
    def build_task_id() -> str:
        return f"task-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"

    def queue_command(self, device_id: str, command: dict[str, Any]) -> int:
        with self._lock:
            queue = self._queue_by_device.setdefault(device_id, [])
            queue.append(copy.deepcopy(command))
            self._persist_locked()
            return len(queue)

    def pop_next_command(self, device_id: str) -> dict[str, Any]:
        with self._lock:
            queue = self._queue_by_device.setdefault(device_id, [])
            if queue:
                value = queue.pop(0)
                self._persist_locked()
                return value

        return {
            "action": "noop",
            "task_id": None,
            "direction": "none",
            "angle": 0.0,
            "step": 0,
        }

    def record_status(self, device_id: str, topic: str, payload: Any) -> None:
        with self._lock:
            self._status_by_device[device_id] = {
                "topic": topic,
                "payload": payload,
                "received_ts_ms": int(time.time() * 1000),
            }
            self._persist_locked()

    def get_status(self, device_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._status_by_device.get(device_id)
            return copy.deepcopy(value) if value is not None else None

    def record_ack(self, device_id: str, topic: str, payload: Any) -> None:
        ack_record = {
            "topic": topic,
            "payload": payload,
            "received_ts_ms": int(time.time() * 1000),
        }

        with self._lock:
            history = self._ack_history_by_device.setdefault(device_id, [])
            history.append(ack_record)
            if len(history) > self._ack_history_limit:
                del history[0 : len(history) - self._ack_history_limit]
            self._persist_locked()

    def get_acks(self, device_id: str, limit: int) -> list[dict[str, Any]]:
        safe_limit = max(1, limit)
        with self._lock:
            history = self._ack_history_by_device.get(device_id, [])
            return copy.deepcopy(history[-safe_limit:])

    def record_latest_capture_metadata(self, device_id: str, metadata: dict[str, Any]) -> None:
        if not device_id:
            return
        with self._lock:
            self._latest_capture_metadata_by_device[device_id] = copy.deepcopy(metadata)
            self._persist_locked()

    def get_latest_capture_metadata(self, device_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._latest_capture_metadata_by_device.get(device_id)
            return copy.deepcopy(value) if value is not None else None
