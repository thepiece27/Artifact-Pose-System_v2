from fastapi import APIRouter

from app.api.routes import (
    artifacts,
    auth,
    devices,
    health,
    inspections,
    models,
    media,
    pose,
    schedules,
    users,
    workflows,
)

router = APIRouter()
router.include_router(auth.router, tags=["auth"])
router.include_router(users.router, tags=["users"])
router.include_router(health.router, tags=["health"])
router.include_router(artifacts.router, tags=["artifacts"])
router.include_router(schedules.router, tags=["schedules"])
router.include_router(devices.router, tags=["devices"])
router.include_router(inspections.router, tags=["inspections"])
router.include_router(models.router, tags=["models"])
router.include_router(media.router, tags=["media"])
router.include_router(pose.router, tags=["pose"])
router.include_router(workflows.router, tags=["workflows"])
