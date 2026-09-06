from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _load_dotenv_file(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    if raw.strip() == "":
        return default
    return raw


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float | None = None) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _sign_int(name: str, default: int) -> int:
    """Return +1 or -1 from env var. Any negative value → -1, else +1."""
    raw = os.getenv(name)
    if raw is None:
        return 1 if default >= 0 else -1
    try:
        return -1 if int(raw) < 0 else 1
    except ValueError:
        return 1 if default >= 0 else -1


def _csv(value: str) -> list[str]:
    if not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


SERVER_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = SERVER_ROOT.parent
_load_dotenv_file(SERVER_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_version: str
    app_host: str
    app_port: int
    cors_allow_origins: list[str]
    environment: str
    device_api_key: str
    device_enrollment_key: str
    max_upload_bytes: int
    max_image_pixels: int
    max_metadata_bytes: int
    max_move_steps: int
    max_move_angle_deg: float
    trans_tolerance_mm: float
    rot_tolerance_deg: float
    steps_per_mm: float
    allow_public_registration: bool
    require_model_for_readiness: bool
    max_model_bytes: int

    data_dir: Path
    model_dir: Path
    run_pose_on_upload: bool
    run_ai_on_upload: bool
    auto_dispatch_pose_command: bool

    mqtt_host: str
    mqtt_port: int
    mqtt_keepalive_sec: int
    mqtt_username: str
    mqtt_password: str
    mqtt_qos: int
    mqtt_publish_timeout_sec: float
    mqtt_cmd_topic_template: str
    mqtt_ack_topic_template: str
    mqtt_status_topic_template: str

    artifact_pose_root: Path
    artifact_camera_params_dir: Path
    artifact_camera_params: Path
    artifact_lens_position: float | None
    artifact_golden_pose: Path

    ack_history_limit: int

    max_alignment_iterations: int
    alignment_timeout_sec: int

    sign_move_x: int
    sign_move_z: int
    sign_rotate_pan: int
    sign_rotate_tilt: int

    # AI model auto-load
    run_ai_on_aligned_image: bool
    default_ai_model_name: str
    default_ai_model_path: str  # empty = scan model_dir for first *.pt

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def inspections_log_file(self) -> Path:
        return self.logs_dir / "inspections_log.jsonl"

    @property
    def mqtt_event_log_file(self) -> Path:
        return self.logs_dir / "mqtt_events.jsonl"


def ensure_directories(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)


def validate_runtime_security(settings: Settings) -> None:
    """Fail fast on unsafe production defaults instead of starting silently."""
    if settings.environment != "production":
        return
    weak = {
        "",
        "change_me",
        "change_me_auth_secret",
        "change_me_long_auth_secret",
        "change_me_long_device_api_key",
        "change_me_long_enrollment_key",
        "change_me_long_mqtt_password",
        "123456",
        "artifact123",
    }
    auth_secret = os.getenv("AUTH_SECRET_KEY", "").strip().lower()
    admin_password = os.getenv("ADMIN_PASSWORD", "").strip().lower()
    postgres_password = os.getenv("POSTGRES_PASSWORD", "").strip().lower()
    if auth_secret in weak or len(auth_secret) < 32:
        raise RuntimeError("AUTH_SECRET_KEY must be a unique secret of at least 32 characters in production")
    if admin_password in weak or len(admin_password) < 12:
        raise RuntimeError("ADMIN_PASSWORD must be at least 12 characters in production")
    if postgres_password in weak or len(postgres_password) < 16:
        raise RuntimeError("POSTGRES_PASSWORD must contain at least 16 characters in production")
    if settings.device_api_key.strip().lower() in weak:
        raise RuntimeError("DEVICE_API_KEY must be configured in production")
    if settings.device_enrollment_key.strip().lower() in weak:
        raise RuntimeError("DEVICE_ENROLLMENT_KEY must be configured in production")
    if len(settings.mqtt_password) < 16:
        raise RuntimeError("MQTT_PASSWORD must contain at least 16 characters in production")
    if not settings.mqtt_username.strip():
        raise RuntimeError("MQTT_USERNAME must be configured in production")
    if settings.cors_allow_origins == ["*"]:
        raise RuntimeError("CORS_ALLOW_ORIGINS must not be '*' in production")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    artifact_pose_root_raw = _env_str(
        "ARTIFACT_POSE_ROOT",
        str(SERVER_ROOT / "app" / "modules" / "artifact_pose"),
    )
    artifact_pose_root = Path(artifact_pose_root_raw)

    camera_params_default = SERVER_ROOT / "data" / "camera_params.yaml"
    camera_params_dir_default = SERVER_ROOT / "data" / "camera_params"
    golden_pose_default = SERVER_ROOT / "data" / "golden_pose.yaml"

    artifact_camera_params_dir = Path(
        _env_str("ARTIFACT_CAMERA_PARAMS_DIR", str(camera_params_dir_default))
    )
    artifact_lens_position = _env_float("ARTIFACT_LENS_POSITION", None)

    if artifact_lens_position is not None:
        artifact_lens_position = round(round(artifact_lens_position * 10.0) / 10.0, 1)

    explicit_camera_params = os.getenv("ARTIFACT_CAMERA_PARAMS")
    if explicit_camera_params is not None and explicit_camera_params.strip() != "":
        artifact_camera_params = Path(explicit_camera_params)
    elif artifact_lens_position is not None:
        artifact_camera_params = (
            artifact_camera_params_dir / f"camera_params_lens_{artifact_lens_position:.1f}.yaml"
        )
    else:
        artifact_camera_params = camera_params_default

    origins_raw = _env_str("CORS_ALLOW_ORIGINS", "*")
    origins = _csv(origins_raw)
    if not origins:
        origins = ["*"]

    data_dir_raw = _env_str("DATA_DIR", str(SERVER_ROOT / "data"))
    model_dir_raw = _env_str("MODEL_DIR", str(WORKSPACE_ROOT / "model"))

    return Settings(
        app_name=_env_str("APP_NAME", "IoT Artifact Server"),
        app_version=_env_str("APP_VERSION", "1.0.0"),
        app_host=_env_str("APP_HOST", "0.0.0.0"),
        app_port=_env_int("APP_PORT", 8000),
        cors_allow_origins=origins,
        environment=_env_str("ENVIRONMENT", "development").lower(),
        device_api_key=_env_str("DEVICE_API_KEY", ""),
        device_enrollment_key=_env_str("DEVICE_ENROLLMENT_KEY", ""),
        max_upload_bytes=max(1_048_576, _env_int("MAX_UPLOAD_BYTES", 25 * 1024 * 1024)),
        max_image_pixels=max(1_000_000, _env_int("MAX_IMAGE_PIXELS", 40_000_000)),
        max_metadata_bytes=max(1024, _env_int("MAX_METADATA_BYTES", 64 * 1024)),
        max_move_steps=max(1, _env_int("MAX_MOVE_STEPS", 100_000)),
        max_move_angle_deg=max(1.0, _env_float("MAX_MOVE_ANGLE_DEG", 180.0) or 180.0),
        trans_tolerance_mm=max(0.1, _env_float("TRANS_TOLERANCE_MM", 25.0) or 25.0),
        rot_tolerance_deg=max(0.01, _env_float("ROT_TOLERANCE_DEG", 2.0) or 2.0),
        steps_per_mm=max(1.0, _env_float("STEPS_PER_MM", 800.0) or 800.0),
        allow_public_registration=_env_bool(
            "ALLOW_PUBLIC_REGISTRATION",
            _env_str("ENVIRONMENT", "development").lower() != "production",
        ),
        require_model_for_readiness=_env_bool("REQUIRE_MODEL_FOR_READINESS", True),
        max_model_bytes=max(1, _env_int("MAX_MODEL_BYTES", 2 * 1024 * 1024 * 1024)),
        data_dir=Path(data_dir_raw),
        model_dir=Path(model_dir_raw),
        run_pose_on_upload=_env_bool("RUN_POSE_ON_UPLOAD", True),
        run_ai_on_upload=_env_bool("RUN_AI_ON_UPLOAD", False),
        auto_dispatch_pose_command=_env_bool("AUTO_DISPATCH_POSE_COMMAND", True),
        mqtt_host=_env_str("MQTT_HOST", "127.0.0.1"),
        mqtt_port=_env_int("MQTT_PORT", 1883),
        mqtt_keepalive_sec=_env_int("MQTT_KEEPALIVE_SEC", 60),
        mqtt_username=_env_str("MQTT_USERNAME", ""),
        mqtt_password=_env_str("MQTT_PASSWORD", ""),
        mqtt_qos=max(0, min(2, _env_int("MQTT_QOS", 1))),
        mqtt_publish_timeout_sec=max(1.0, _env_float("MQTT_PUBLISH_TIMEOUT_SEC", 5.0) or 5.0),
        mqtt_cmd_topic_template=_env_str("MQTT_CMD_TOPIC_TEMPLATE", "cmd/{device_id}"),
        mqtt_ack_topic_template=_env_str("MQTT_ACK_TOPIC_TEMPLATE", "ack/{device_id}"),
        mqtt_status_topic_template=_env_str(
            "MQTT_STATUS_TOPIC_TEMPLATE",
            "status/{device_id}",
        ),
        artifact_pose_root=artifact_pose_root,
        artifact_camera_params_dir=artifact_camera_params_dir,
        artifact_camera_params=artifact_camera_params,
        artifact_lens_position=artifact_lens_position,
        artifact_golden_pose=Path(_env_str("ARTIFACT_GOLDEN_POSE", str(golden_pose_default))),
        ack_history_limit=max(1, _env_int("ACK_HISTORY_LIMIT", 200)),
        max_alignment_iterations=max(1, _env_int("MAX_ALIGNMENT_ITERATIONS", 20)),
        alignment_timeout_sec=max(10, _env_int("ALIGNMENT_TIMEOUT_SEC", 300)),
        sign_move_x=_sign_int("SIGN_MOVE_X", -1),
        sign_move_z=_sign_int("SIGN_MOVE_Z", -1),
        sign_rotate_pan=_sign_int("SIGN_ROTATE_PAN", -1),
        sign_rotate_tilt=_sign_int("SIGN_ROTATE_TILT", -1),
        run_ai_on_aligned_image=_env_bool("RUN_AI_ON_ALIGNED_IMAGE", True),
        default_ai_model_name=_env_str("DEFAULT_AI_MODEL_NAME", "default"),
        default_ai_model_path=_env_str("DEFAULT_AI_MODEL_PATH", ""),
    )
