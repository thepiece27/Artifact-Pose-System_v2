"""API client cho giao tiep voi FastAPI server."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from requests import Response
from requests.exceptions import RequestException


@dataclass
class APIConfig:
	"""Cau hinh ket noi toi FastAPI server."""

	base_url: str
	device_id: str
	device_api_key: str = ""
	device_enrollment_key: str = ""
	timeout_sec: int = 10
	# Timeout rieng cho upload anh 4K + xu ly pose tren server.
	# Pose correction (ORB + G2O) mat 10-20s, golden init mat 60-120s.
	upload_inspection_timeout_sec: int = 60
	upload_stereo_timeout_sec: int = 120


class APIClient:
	"""Client dong goi cac API can thiet cho IoT node."""

	def __init__(self, config: APIConfig) -> None:
		self.config = config

	def _url(self, path: str) -> str:
		return f"{self.config.base_url.rstrip('/')}{path}"

	def _device_headers(self) -> Dict[str, str]:
		return {"X-Device-Key": self.config.device_api_key} if self.config.device_api_key else {}

	def _enrollment_headers(self) -> Dict[str, str]:
		return {"X-Enrollment-Key": self.config.device_enrollment_key} if self.config.device_enrollment_key else {}

	def receive_move_command(self) -> Optional[Dict[str, Any]]:
		"""Nhan lenh dieu khien tu endpoint POST /devices/{id}/move.

		Ghi chu:
		- O day IoT node dong vai tro client, chu dong goi API de nhan lenh.
		- Server du kien tra ve JSON chua action, direction, angle/step.
		"""
		endpoint = f"/api/v1/devices/{self.config.device_id}/move"

		try:
			response = requests.post(
				self._url(endpoint),
				json={"device_id": self.config.device_id},
				headers=self._device_headers(),
				timeout=self.config.timeout_sec,
			)
			response.raise_for_status()
			payload = response.json()
			if not isinstance(payload, dict):
				return None
			return payload
		except (RequestException, ValueError) as exc:
			print(f"[API] Loi nhan lenh move: {exc}")
			return None

	def get_device_id(
		self,
		machine_hash: str,
		preferred_device_id: Optional[str] = None,
	) -> Optional[str]:
		"""Dang ky/lay device_id duy nhat tu server.

		Endpoint: POST /devices/get_device_id
		Payload: {"machine_hash": "...", "preferred_device_id": "..."}
		"""
		endpoint = "/api/v1/devices/get_device_id"
		payload: Dict[str, Any] = {"machine_hash": machine_hash}
		if preferred_device_id:
			payload["preferred_device_id"] = preferred_device_id

		try:
			response = requests.post(
				self._url(endpoint),
				json=payload,
				headers=self._enrollment_headers(),
				timeout=self.config.timeout_sec,
			)
			response.raise_for_status()
			body = response.json()
			if not isinstance(body, dict):
				return None
			device_id = body.get("device_id")
			if not isinstance(device_id, str) or not device_id.strip():
				return None
			return device_id.strip()
		except (RequestException, ValueError) as exc:
			print(f"[API] Loi get_device_id: {exc}")
			return None

	def upload_inspection(
		self,
		image_path: Path,
		device_id: str,
		artifact_id: str,
		calibration_data: Dict[str, float],
	) -> bool:
		"""Upload anh va metadata qua multipart/form-data.

		metadata duoc gui dang JSON string trong field metadata.
		"""
		endpoint = "/inspections/upload"

		if not image_path.exists():
			print(f"[API] Khong tim thay file anh: {image_path}")
			return False

		metadata = {
			"device_id": device_id,
			"artifact_id": artifact_id,
			"calibration_data": calibration_data,
		}

		try:
			with image_path.open("rb") as image_file:
				files = {
					"file": (image_path.name, image_file, "image/png"),
					"metadata": (None, json.dumps(metadata), "application/json"),
				}
				response = requests.post(
					self._url(endpoint),
					files=files,
					headers=self._device_headers(),
					timeout=self.config.upload_inspection_timeout_sec,
				)
				response.raise_for_status()
			print("[API] Upload inspection thanh cong")
			return True
		except RequestException as exc:
			print(f"[API] Loi upload inspection: {exc}")
			return False

	@staticmethod
	def debug_response(response: Response) -> str:
		"""Tra ve thong tin response de debug khi can."""
		return f"status={response.status_code}, body={response.text[:200]}"

	def upload_stereo_pair(
		self,
		left_path: Path,
		right_path: Path,
		artifact_id: Optional[str] = None,
	) -> Optional[Dict[str, Any]]:
		"""Upload cap anh stereo (left, right) len /pose/initialize_golden."""
		endpoint = "/pose/initialize_golden"

		for label, path in [("left", left_path), ("right", right_path)]:
			if not path.exists():
				print(f"[API] Khong tim thay file {label}: {path}")
				return None

		try:
			with left_path.open("rb") as lf, right_path.open("rb") as rf:
				files = {
					"left_file": (left_path.name, lf, "image/png"),
					"right_file": (right_path.name, rf, "image/png"),
				}
				data: Dict[str, Any] = {}
				if self.config.device_id:
					data["device_id"] = self.config.device_id
					data["device_code"] = self.config.device_id
				if artifact_id:
					data["artifact_id"] = artifact_id
				response = requests.post(
					self._url(endpoint),
					files=files,
					data=data,
					headers=self._device_headers(),
					timeout=self.config.upload_stereo_timeout_sec,
				)
				if not response.ok:
					print(f"[API] Loi upload stereo pair: {response.status_code} - {response.text[:500]}")
					return None
			body = response.json()
			print(f"[API] Upload stereo pair thanh cong: {body.get('message', '')}")
			return body if isinstance(body, dict) else None
		except RequestException as exc:
			print(f"[API] Loi upload stereo pair (network): {exc}")
			return None
