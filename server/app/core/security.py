from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

# Load the same local configuration source before deriving JWT settings. This
# module is imported by API dependencies before app.main calls get_settings().
from app.core.config import SERVER_ROOT, _load_dotenv_file

_load_dotenv_file(SERVER_ROOT / ".env")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "")
if not SECRET_KEY:
    raise RuntimeError("AUTH_SECRET_KEY must be configured before importing the API")
ALGORITHM = os.getenv("AUTH_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("AUTH_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_minutes: int | None = None) -> str:
    payload = data.copy()
    ttl = expires_minutes if expires_minutes is not None else ACCESS_TOKEN_EXPIRE_MINUTES
    expire_at = datetime.now(timezone.utc) + timedelta(minutes=ttl)
    payload.update({"exp": expire_at})
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and verify a JWT. Raises JWTError on failure."""
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
