from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_container, get_current_user, require_device_access, require_device_enrollment
from app.core.database import get_db
from app.models.iot_device import IotDevice, DeviceStatus
from app.models.user import User
from app.schemas.devices import (
    DeviceAcksResponse,
    DeviceIdRequest,
    DeviceIdResponse,
    DeviceStatusResponse,
    DeviceSummary,
    MoveCommand,
    MoveCommandRequest,
    QueueMoveResponse,
)
from app.services.state import AppContainer

router = APIRouter(prefix="/api/v1/devices", tags=["devices"])


def _upsert_device_record(db: Session, device_code: str, *, touch: bool) -> tuple[IotDevice, bool]:
    now = datetime.now(timezone.utc)
    device = db.query(IotDevice).filter(IotDevice.device_code == device_code).first()
    if device is None:
        device = IotDevice(
            device_code=device_code,
            description="Registered Raspberry Pi device",
            status=DeviceStatus.offline,
            last_active_at=now if touch else None,
        )
        db.add(device)
        return device, True

    changed = False
    if touch:
        device.last_active_at = now
        changed = True
    if not device.description:
        device.description = "Registered Raspberry Pi device"
        changed = True
    return device, changed


def _sync_registry_devices_to_db(db: Session, container: AppContainer) -> None:
    changed = False
    for item in container.device_registry.list_all():
        device_code = item.get("device_id")
        if not device_code:
            continue
        _, did_change = _upsert_device_record(db, device_code, touch=False)
        changed = changed or did_change
    if changed:
        db.commit()


# --------- DB-backed inventory ---------

@router.get("", response_model=List[DeviceSummary])
def list_devices(
    db: Session = Depends(get_db),
    container: AppContainer = Depends(get_container),
    _: User = Depends(get_current_user),
) -> list[DeviceSummary]:
    _sync_registry_devices_to_db(db, container)
    devices = db.query(IotDevice).order_by(IotDevice.created_at.desc()).all()
    now_ms = int(time.time() * 1000)
    _ONLINE_WINDOW_MS = 120_000  # 2 minutes — device heartbeat threshold
    result = []
    for d in devices:
        # Check real-time MQTT heartbeat status (in-memory, from status/{device_id} topic)
        rt = container.command_service.get_status(d.device_code)
        is_online = False
        if rt:
            received_ts = rt.get("received_ts_ms") or 0
            payload = rt.get("payload") or {}
            if isinstance(payload, dict) and payload.get("status") == "online":
                if (now_ms - received_ts) < _ONLINE_WINDOW_MS:
                    is_online = True
        result.append(
            DeviceSummary(
                device_id=d.device_id,
                device_code=d.device_code,
                machine_hash=d.device_code,
                status={
                    "db_status": "online" if is_online else d.status.value,
                    "description": d.description or "",
                    "last_active_at": d.last_active_at.isoformat() if d.last_active_at else None,
                },
            )
        )
    return result


# --------- IoT operational endpoints (used by Raspberry Pi clients) ---------
# Path params below are MQTT device_code values. The legacy response/request
# field name `device_id` is kept for the current Pi agent contract.

@router.post("/get_device_id", response_model=DeviceIdResponse)
def get_device_id(
    req: DeviceIdRequest,
    container: AppContainer = Depends(get_container),
    db: Session = Depends(get_db),
    _auth: None = Depends(require_device_enrollment),
) -> DeviceIdResponse:
    try:
        device_code = container.device_registry.allocate_device_id(
            machine_hash=req.machine_hash,
            preferred_device_id=req.preferred_device_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    device, _ = _upsert_device_record(db, device_code, touch=True)
    db.commit()
    db.refresh(device)

    return DeviceIdResponse(
        ok=True,
        device_id=device_code,
        device_code=device_code,
        db_device_id=device.device_id,
        machine_hash=req.machine_hash,
    )


@router.post("/{device_code}/queue_move", response_model=QueueMoveResponse)
def queue_move(
    device_code: str,
    cmd: MoveCommand,
    container: AppContainer = Depends(get_container),
    _auth: User | None = Depends(require_device_access),
) -> QueueMoveResponse:
    if not container.mqtt_bridge._valid_device_id(device_code):
        raise HTTPException(status_code=400, detail="Invalid device code")
    payload = cmd.dict()
    if not payload.get("task_id"):
        payload["task_id"] = container.command_service.build_task_id()

    published, publish_result = container.mqtt_bridge.publish_command(device_code, payload)
    queued = 0
    if not published:
        # The current edge agent consumes MQTT only. Reporting an in-memory
        # queue as success would silently lose commands; fail explicitly until
        # a durable authenticated polling worker is deployed.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"MQTT command delivery unavailable: {publish_result}",
            headers={"Retry-After": "3"},
        )

    return QueueMoveResponse(
        ok=True,
        mode="mqtt",
        published=published,
        topic=publish_result if published else None,
        publish_error=None if published else publish_result,
        task_id=str(payload["task_id"]),
        queued=queued,
    )


@router.post("/{device_code}/move", response_model=MoveCommand)
def poll_move_command(
    device_code: str,
    req: MoveCommandRequest | None = None,
    container: AppContainer = Depends(get_container),
    _auth: User | None = Depends(require_device_access),
) -> MoveCommand:
    if not container.mqtt_bridge._valid_device_id(device_code):
        raise HTTPException(status_code=400, detail="Invalid device code")
    if req is not None and req.device_id != device_code:
        raise HTTPException(status_code=400, detail="device_id mismatch")
    payload = container.command_service.pop_next_command(device_code)
    return MoveCommand(**payload)


@router.get("/{device_code}/status", response_model=DeviceStatusResponse)
def device_status(
    device_code: str,
    container: AppContainer = Depends(get_container),
    _auth: User | None = Depends(require_device_access),
) -> DeviceStatusResponse:
    if not container.mqtt_bridge._valid_device_id(device_code):
        raise HTTPException(status_code=400, detail="Invalid device code")
    return DeviceStatusResponse(
        ok=True,
        device_id=device_code,
        status=container.command_service.get_status(device_code),
    )


@router.get("/{device_code}/acks", response_model=DeviceAcksResponse)
def device_acks(
    device_code: str,
    limit: int = Query(default=20, ge=1, le=200),
    container: AppContainer = Depends(get_container),
    _auth: User | None = Depends(require_device_access),
) -> DeviceAcksResponse:
    if not container.mqtt_bridge._valid_device_id(device_code):
        raise HTTPException(status_code=400, detail="Invalid device code")
    history = container.command_service.get_acks(device_code, limit=limit)
    return DeviceAcksResponse(
        ok=True,
        device_id=device_code,
        count=len(history),
        acks=history,
    )
