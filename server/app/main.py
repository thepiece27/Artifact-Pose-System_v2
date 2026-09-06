from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.database import init_auth_database
from app.core.config import ensure_directories, get_settings, validate_runtime_security
from app.services.state import AppContainer

settings = get_settings()
validate_runtime_security(settings)
ensure_directories(settings)
container = AppContainer(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_auth_database()
    container.startup()
    try:
        yield
    finally:
        container.shutdown()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)
app.state.container = container

allow_credentials = settings.cors_allow_origins != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
