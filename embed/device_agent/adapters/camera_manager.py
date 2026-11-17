"""Quan ly camera Pi Camera Module V3 voi che do thong so thu cong."""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

# Resolution 4K khop voi camera_params calibration.
DEFAULT_CAPTURE_WIDTH = 3840
DEFAULT_CAPTURE_HEIGHT = 2160


@dataclass
class CaptureOutcome:
    image_path: Path
    metadata: dict[str, Any]


class CameraManager:
    """Dieu khien viec chup anh 4K voi cau hinh AF manual, lens co dinh."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture_simple(
        self,
        basename: str,
        lens_position: float = 1.5,
        stabilize_sec: float = 1.5,
        capture_width: int = DEFAULT_CAPTURE_WIDTH,
        capture_height: int = DEFAULT_CAPTURE_HEIGHT,
    ) -> Optional[CaptureOutcome]:
        """Chup anh 4K don gian: manual AF, lens co dinh, AE/AWB tu dong.

        Day la mode chinh cho workflow pose hien tai.
        """
        image_path = self.output_dir / f"{basename}.png"

        try:
            picamera2_module = importlib.import_module("picamera2")
            libcamera_module = importlib.import_module("libcamera")
            Picamera2 = getattr(picamera2_module, "Picamera2")
            controls = getattr(libcamera_module, "controls")
        except Exception as exc:
            print(f"[CAM] Khong the import Picamera2/libcamera: {exc}")
            return None

        picam2 = None
        started = False
        try:
            picam2 = Picamera2()
            config = picam2.create_still_configuration(
                main={"size": (capture_width, capture_height), "format": "RGB888"},
            )
            picam2.configure(config)
            picam2.start()
            started = True

            time.sleep(max(0.0, stabilize_sec))

            picam2.set_controls({
                "AfMode": controls.AfModeEnum.Manual,
                "LensPosition": float(lens_position),
            })
            time.sleep(0.5)

            request = picam2.capture_request()
            request.save("main", str(image_path))
            try:
                capture_meta = request.get_metadata() or {}
            except Exception:
                capture_meta = {}
            request.release()

            runtime_metadata = self._extract_runtime_metadata(capture_meta)
            runtime_metadata["applied_lens_position"] = float(lens_position)
            runtime_metadata["capture_width"] = capture_width
            runtime_metadata["capture_height"] = capture_height

            print(f"[CAM] Da chup anh 4K: {image_path}")
            return CaptureOutcome(image_path=image_path, metadata=runtime_metadata)
        except Exception as exc:
            print(f"[CAM] Loi chup anh: {exc}")
            return None
        finally:
            if picam2 is not None:
                try:
                    if started:
                        picam2.stop()
                except Exception:
                    pass
                try:
                    picam2.close()
                except Exception:
                    pass

    def capture_high_quality(
        self,
        basename: str,
        autofocus_mode: str = "manual",
        lens_position: float = 2.5,
        awbgains: Sequence[float] = (1.2, 1.5),
        gain: float = 1.0,
        shutter: int = 20000,
        pre_set_controls_delay_sec: float = 1.5,
        pre_capture_request_delay_sec: float = 2.0,
    ) -> Optional[Path]:
        outcome = self.capture_high_quality_with_metadata(
            basename=basename,
            autofocus_mode=autofocus_mode,
            lens_position=lens_position,
            awbgains=awbgains,
            gain=gain,
            shutter=shutter,
            pre_set_controls_delay_sec=pre_set_controls_delay_sec,
            pre_capture_request_delay_sec=pre_capture_request_delay_sec,
        )
        if outcome is None:
            return None
        return outcome.image_path

    def capture_high_quality_with_metadata(
        self,
        basename: str,
        autofocus_mode: str = "manual",
        lens_position: float = 2.5,
        awbgains: Sequence[float] = (1.2, 1.5),
        gain: float = 1.0,
        shutter: int = 20000,
        pre_set_controls_delay_sec: float = 1.5,
        pre_capture_request_delay_sec: float = 2.0,
        autofocus_probe_sec: float = 0.25,
    ) -> Optional[CaptureOutcome]:
        """Chup anh chat luong cao 12MP va tra metadata runtime camera."""
        image_path = self.output_dir / f"{basename}.png"

        try:
            picamera2_module = importlib.import_module("picamera2")
            libcamera_module = importlib.import_module("libcamera")
            Picamera2 = getattr(picamera2_module, "Picamera2")
            controls = getattr(libcamera_module, "controls")
        except Exception as exc:
            print(f"[CAM] Khong the import Picamera2/libcamera: {exc}")
            return None

        picam2 = None
        started = False
        try:
            picam2 = Picamera2()
            config = picam2.create_still_configuration(
                main={"size": picam2.sensor_resolution, "format": "RGB888"},
                raw={"size": picam2.sensor_resolution},
                controls={"AwbEnable": False},
            )
            picam2.configure(config)
            picam2.start()
            started = True

            time.sleep(max(0.0, pre_set_controls_delay_sec))

            runtime_metadata: dict[str, Any] = {}
            af_lens_position = self._probe_autofocus_lens(
                picam2,
                controls,
                autofocus_probe_sec,
            )
            if af_lens_position is not None:
                runtime_metadata["autofocus_lens_position"] = float(af_lens_position)

            af_manual = controls.AfModeEnum.Manual
            if autofocus_mode.lower() != "manual":
                print("[CAM] Canh bao: se van ep AF manual theo yeu cau")

            effective_lens_position = (
                float(af_lens_position) if af_lens_position is not None else float(lens_position)
            )

            picam2.set_controls(
                {
                    "AfMode": af_manual,
                    "LensPosition": effective_lens_position,
                    "AeEnable": False,
                    "AwbEnable": False,
                    "ColourGains": tuple(float(x) for x in awbgains),
                    "AnalogueGain": float(gain),
                    "ExposureTime": int(shutter),
                }
            )

            time.sleep(max(0.0, pre_capture_request_delay_sec))

            request = picam2.capture_request()
            request.save("main", str(image_path))
            try:
                capture_meta = request.get_metadata() or {}
            except Exception:
                capture_meta = {}
            request.release()

            runtime_metadata.update(self._extract_runtime_metadata(capture_meta))
            runtime_metadata["applied_lens_position"] = effective_lens_position
            runtime_metadata["configured_lens_position"] = float(lens_position)
            runtime_metadata["awb_gains"] = [float(awbgains[0]), float(awbgains[1])]
            runtime_metadata["analogue_gain"] = float(gain)
            runtime_metadata["exposure_time"] = int(shutter)

            print(f"[CAM] Da chup anh: {image_path}")
            return CaptureOutcome(image_path=image_path, metadata=runtime_metadata)
        except Exception as exc:
            print(f"[CAM] Loi chup anh: {exc}")
            return None
        finally:
            if picam2 is not None:
                try:
                    if started:
                        picam2.stop()
                except Exception:
                    pass
                try:
                    picam2.close()
                except Exception:
                    pass

    def manual_session(self, **kwargs: Any) -> "ManualCameraSession":
        """Persistent camera with fixed focus/exposure/white balance (CITA)."""
        return ManualCameraSession(self.output_dir, **kwargs)

    @staticmethod
    def _probe_autofocus_lens(
        picam2: Any,
        controls: Any,
        autofocus_probe_sec: float,
    ) -> float | None:
        try:
            picam2.set_controls({"AfMode": controls.AfModeEnum.Auto})
            try:
                picam2.set_controls({"AfTrigger": controls.AfTriggerEnum.Start})
            except Exception:
                pass
            time.sleep(max(0.0, autofocus_probe_sec))
            af_meta = picam2.capture_metadata() or {}
            lens = af_meta.get("LensPosition")
            if lens is None:
                lens = af_meta.get("LensPositionRaw")
            if lens is None:
                return None
            return float(lens)
        except Exception:
            return None

    @staticmethod
    def _extract_runtime_metadata(raw: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if not isinstance(raw, dict):
            return result

        for key in ("LensPosition", "AnalogueGain", "ExposureTime", "SensorTimestamp"):
            if key in raw:
                result[key.lower()] = raw.get(key)

        if "ColourGains" in raw:
            value = raw.get("ColourGains")
            if isinstance(value, (list, tuple)) and len(value) == 2:
                result["awb_gains_runtime"] = [float(value[0]), float(value[1])]

        return result


class ManualCameraSession:
    """Camera kept open across frames with every automatic control disabled.

    Used by ``embed/tools/capture_session.py`` (E0/E1/E3 data): manual focus at
    a fixed ``LensPosition`` (default 1.5, the calibrated value), AE off, AWB
    off with fixed ``ColourGains``, fixed ``ExposureTime``/``AnalogueGain``.
    No autofocus probe is ever run (it would move the lens).
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        width: int = DEFAULT_CAPTURE_WIDTH,
        height: int = DEFAULT_CAPTURE_HEIGHT,
        lens_position: float = 1.5,
        exposure_time_us: int = 20000,
        analogue_gain: float = 1.0,
        colour_gains: Sequence[float] = (1.2, 1.5),
        warmup_s: float = 2.0,
        image_format: str = "png",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.width, self.height = int(width), int(height)
        self.lens_position = float(lens_position)
        self.exposure_time_us = int(exposure_time_us)
        self.analogue_gain = float(analogue_gain)
        self.colour_gains = (float(colour_gains[0]), float(colour_gains[1]))
        self.warmup_s = float(warmup_s)
        self.image_format = image_format.lower().lstrip(".")
        self._picam2: Any = None

    def __enter__(self) -> "ManualCameraSession":
        picamera2_module = importlib.import_module("picamera2")
        libcamera_module = importlib.import_module("libcamera")
        controls = getattr(libcamera_module, "controls")
        self._picam2 = getattr(picamera2_module, "Picamera2")()
        config = self._picam2.create_still_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"AwbEnable": False, "AeEnable": False},
        )
        self._picam2.configure(config)
        self._picam2.start()
        self._picam2.set_controls({
            "AfMode": controls.AfModeEnum.Manual,
            "LensPosition": self.lens_position,
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": self.exposure_time_us,
            "AnalogueGain": self.analogue_gain,
            "ColourGains": self.colour_gains,
        })
        time.sleep(max(0.0, self.warmup_s))
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._picam2 is not None:
            try:
                self._picam2.stop()
            finally:
                self._picam2.close()
                self._picam2 = None

    def requested_controls(self) -> dict[str, Any]:
        return {"LensPosition": self.lens_position, "ExposureTime": self.exposure_time_us,
                "AnalogueGain": self.analogue_gain, "ColourGains": list(self.colour_gains),
                "AeEnable": False, "AwbEnable": False, "AfMode": "Manual"}

    def capture(self, basename: str) -> CaptureOutcome:
        if self._picam2 is None:
            raise RuntimeError("ManualCameraSession is not open (use 'with')")
        image_path = self.output_dir / f"{basename}.{self.image_format}"
        if image_path.exists():
            raise FileExistsError(f"Refusing to overwrite {image_path}")
        request = self._picam2.capture_request()
        try:
            request.save("main", str(image_path))
            raw = request.get_metadata() or {}
        finally:
            request.release()
        meta: dict[str, Any] = {}
        for key in ("LensPosition", "ExposureTime", "AnalogueGain", "ColourGains",
                    "SensorTimestamp", "Lux", "ColourTemperature", "FocusFoM"):
            if key in raw:
                value = raw[key]
                meta[key] = list(value) if isinstance(value, (list, tuple)) else value
        meta["requested"] = self.requested_controls()
        return CaptureOutcome(image_path=image_path, metadata=meta)
