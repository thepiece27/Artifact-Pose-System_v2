from __future__ import annotations

import json
import os
import re
import time
import uuid
from threading import Lock
from typing import Any

from app.core.config import Settings
from app.services.command_service import CommandService

try:
    import paho.mqtt.client as mqtt_client
except Exception:
    mqtt_client = None


class MqttBridge:
    def __init__(self, settings: Settings, command_service: CommandService) -> None:
        self._settings = settings
        self._command_service = command_service
        self._client: Any = None
        self._lock = Lock()
        self._state: dict[str, Any] = {
            "available": mqtt_client is not None,
            "connected": False,
            "last_error": None,
            "last_connect_ts_ms": None,
        }

    def start(self) -> None:
        if mqtt_client is None:
            with self._lock:
                self._state["last_error"] = "paho-mqtt_not_installed"
            return

        client_id = f"iot-artifact-server-{uuid.uuid4().hex[:8]}"
        self._client = mqtt_client.Client(client_id=client_id)

        if self._settings.mqtt_username:
            self._client.username_pw_set(
                username=self._settings.mqtt_username,
                password=self._settings.mqtt_password or None,
            )

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        try:
            self._client.reconnect_delay_set(min_delay=1, max_delay=30)
        except Exception:
            pass

        try:
            # connect_async lets Paho's network loop handle broker cold-start
            # and transient network failures without blocking API startup.
            self._client.connect_async(
                self._settings.mqtt_host,
                self._settings.mqtt_port,
                self._settings.mqtt_keepalive_sec,
            )
            self._client.loop_start()
        except Exception as exc:
            with self._lock:
                self._state["connected"] = False
                self._state["last_error"] = f"connect_exception={exc}"

    def stop(self) -> None:
        if self._client is None:
            return

        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            pass

    def state(self) -> dict[str, Any]:
        with self._lock:
            state_copy = dict(self._state)

        return {
            "broker": {
                "host": self._settings.mqtt_host,
                "port": self._settings.mqtt_port,
                "keepalive_sec": self._settings.mqtt_keepalive_sec,
                "qos": self._settings.mqtt_qos,
            },
            "topics": {
                "cmd": self._settings.mqtt_cmd_topic_template,
                "ack": self._settings.mqtt_ack_topic_template,
                "status": self._settings.mqtt_status_topic_template,
            },
            "state": state_copy,
        }

    def publish_command(self, device_id: str, payload: dict[str, Any]) -> tuple[bool, str]:
        if self._client is None:
            return False, "mqtt_not_available"

        with self._lock:
            if not self._state.get("connected"):
                return False, "mqtt_not_connected"

        if not self._valid_device_id(device_id):
            return False, "invalid_device_id"
        if not isinstance(payload, dict) or not str(payload.get("task_id", "")).strip():
            return False, "command_task_id_required"

        topic = self._topic(self._settings.mqtt_cmd_topic_template, device_id)

        try:
            info = self._client.publish(
                topic,
                payload=json.dumps(payload),
                qos=self._settings.mqtt_qos,
                retain=False,
            )
        except Exception as exc:
            with self._lock:
                self._state["last_error"] = f"publish_exception={exc}"
            return False, str(exc)

        if getattr(info, "rc", 1) != 0:
            error = f"publish_failed_rc={info.rc}"
            with self._lock:
                self._state["last_error"] = error
            return False, error

        if self._settings.mqtt_qos > 0:
            try:
                # Paho versions differ: wait_for_publish() may return None.
                # The authoritative delivery flag is is_published().
                info.wait_for_publish(timeout=self._settings.mqtt_publish_timeout_sec)
                if not bool(getattr(info, "is_published", lambda: False)()):
                    error = "publish_ack_timeout"
                    with self._lock:
                        self._state["last_error"] = error
                    return False, error
            except Exception as exc:
                return False, f"publish_ack_error={exc}"

        self._log_event("publish_cmd", topic, payload)
        return True, topic

    def _on_connect(self, client: Any, userdata: Any, flags: Any, rc: int) -> None:
        with self._lock:
            self._state["connected"] = rc == 0
            self._state["last_connect_ts_ms"] = int(time.time() * 1000)
            self._state["last_error"] = None if rc == 0 else f"connect_failed_rc={rc}"

        if rc != 0:
            return

        ack_wildcard = self._topic_wildcard(self._settings.mqtt_ack_topic_template)
        status_wildcard = self._topic_wildcard(self._settings.mqtt_status_topic_template)
        client.subscribe(ack_wildcard, qos=self._settings.mqtt_qos)
        client.subscribe(status_wildcard, qos=self._settings.mqtt_qos)

    def _on_disconnect(self, client: Any, userdata: Any, rc: int) -> None:
        with self._lock:
            self._state["connected"] = False
            if rc != 0:
                self._state["last_error"] = f"disconnect_rc={rc}"

    def _on_message(self, client: Any, userdata: Any, msg: Any) -> None:
        topic = str(getattr(msg, "topic", ""))
        raw_payload = getattr(msg, "payload", b"")

        try:
            decoded = raw_payload.decode("utf-8")
            payload: Any = json.loads(decoded)
        except Exception:
            payload = {
                "raw": raw_payload.decode("utf-8", errors="replace"),
                "parse_error": True,
            }

        if not isinstance(payload, dict):
            self._log_event("reject", topic, {"reason": "payload_not_object"})
            return

        topic_device = self._extract_device_id(topic, self._settings.mqtt_status_topic_template)
        if topic_device is None:
            topic_device = self._extract_device_id(topic, self._settings.mqtt_ack_topic_template)
        payload_device = str(payload.get("device_id", "")).strip()
        if topic_device and payload_device != topic_device:
            self._log_event("reject", topic, {"reason": "device_id_topic_mismatch"})
            return

        if self._topic_matches(topic, self._settings.mqtt_status_topic_template):
            device_id = self._extract_device_id(topic, self._settings.mqtt_status_topic_template)
            if device_id:
                self._command_service.record_status(device_id, topic, payload)
                self._log_event("status", topic, payload)
            return

        if self._topic_matches(topic, self._settings.mqtt_ack_topic_template):
            device_id = self._extract_device_id(topic, self._settings.mqtt_ack_topic_template)
            if device_id:
                self._command_service.record_ack(device_id, topic, payload)
                self._log_event("ack", topic, payload)

    def _log_event(self, event_type: str, topic: str, payload: Any) -> None:
        record = {
            "timestamp_ms": int(time.time() * 1000),
            "event_type": event_type,
            "topic": topic,
            "payload": payload,
        }
        log_file = self._settings.mqtt_event_log_file
        try:
            max_bytes = int(os.getenv("MQTT_EVENT_LOG_MAX_BYTES", str(10 * 1024 * 1024)))
            if log_file.exists() and log_file.stat().st_size >= max(1024, max_bytes):
                rotated = log_file.with_suffix(log_file.suffix + ".1")
                rotated.unlink(missing_ok=True)
                log_file.replace(rotated)
        except Exception:
            pass
        with log_file.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _topic(template: str, device_id: str) -> str:
        return template.format(device_id=device_id)

    @staticmethod
    def _valid_device_id(device_id: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", str(device_id or "")))

    @staticmethod
    def _topic_wildcard(template: str) -> str:
        if "{device_id}" in template:
            return template.replace("{device_id}", "+")
        normalized = template.strip("/")
        return "+" if not normalized else f"{normalized}/+"

    @staticmethod
    def _topic_matches(topic: str, template: str) -> bool:
        topic_parts = topic.strip("/").split("/")
        template_parts = template.strip("/").split("/")
        if len(topic_parts) != len(template_parts):
            return False

        for topic_part, template_part in zip(topic_parts, template_parts):
            if template_part == "{device_id}":
                if not topic_part:
                    return False
                continue
            if topic_part != template_part:
                return False

        return True

    @staticmethod
    def _extract_device_id(topic: str, template: str) -> str | None:
        topic_parts = topic.strip("/").split("/")
        template_parts = template.strip("/").split("/")
        if len(topic_parts) != len(template_parts):
            return None

        for topic_part, template_part in zip(topic_parts, template_parts):
            if template_part == "{device_id}":
                return topic_part

        return None
