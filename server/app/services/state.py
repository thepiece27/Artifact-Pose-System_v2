from __future__ import annotations

from pathlib import Path

from app.core.config import Settings
from app.services.command_service import CommandService
from app.services.device_registry import DeviceRegistry
from app.services.inspection_service import InspectionService
from app.services.model_service import ModelService
from app.services.mqtt_bridge import MqttBridge
from app.services.pose_service import PoseService


class AppContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device_registry = DeviceRegistry(settings.data_dir / "device_registry.json")
        self.command_service = CommandService(
            ack_history_limit=settings.ack_history_limit,
            state_file=settings.data_dir / "command_state.json",
        )
        self.mqtt_bridge = MqttBridge(settings, self.command_service)
        self.model_service = ModelService(settings)
        self.pose_service = PoseService(settings)
        self.inspection_service = InspectionService(
            settings=settings,
            pose_service=self.pose_service,
            model_service=self.model_service,
            command_service=self.command_service,
            mqtt_bridge=self.mqtt_bridge,
        )

    def startup(self) -> None:
        self.mqtt_bridge.start()
        self._auto_load_default_model()

    def _auto_load_default_model(self) -> None:
        """Auto-load the default YOLO .pt model from model_dir at startup."""
        model_path_str = self.settings.default_ai_model_path.strip()
        model_name = self.settings.default_ai_model_name

        if model_path_str:
            candidate = Path(model_path_str)
            if not candidate.is_absolute():
                candidate = self.settings.model_dir / candidate
        else:
            # Scan model_dir for first *.pt file
            pt_files = sorted(self.settings.model_dir.glob("*.pt"))
            if not pt_files:
                print(f"[STARTUP] No *.pt model found in {self.settings.model_dir}, AI detection unavailable")
                return
            candidate = pt_files[0]

        try:
            self.model_service.load_model(
                name=model_name,
                path=str(candidate),
                backend="auto",
            )
            print(f"[STARTUP] Auto-loaded AI model '{model_name}' from {candidate}")
        except Exception as exc:
            print(f"[STARTUP] Failed to auto-load AI model '{model_name}' from {candidate}: {exc}")

    def shutdown(self) -> None:
        self.mqtt_bridge.stop()
