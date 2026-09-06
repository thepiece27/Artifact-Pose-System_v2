from __future__ import annotations

import hmac
from fastapi import Depends, HTTPException, Request, Header, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import ALGORITHM, SECRET_KEY
from app.models.user import User
from app.services.state import AppContainer

_bearer = HTTPBearer(auto_error=False)


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


def _user_from_token(token: str, db: Session) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        ) from exc

    username = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject",
        )
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer exists",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is inactive",
        )
    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
        )
    return _user_from_token(credentials.credentials, db)


def get_optional_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User | None:
    if credentials is None or not credentials.credentials:
        return None
    return _user_from_token(credentials.credentials, db)


def require_admin(current: User = Depends(get_current_user)) -> User:
    if current.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return current


def require_device_access(
    request: Request,
    x_device_key: str | None = Header(default=None),
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User | None:
    """Authorize device transport with a secret key or an operator JWT.

    Device endpoints must not be anonymously reachable. The key is compared
    in constant time and is never logged or persisted.
    """
    settings = request.app.state.container.settings
    expected = settings.device_api_key.strip()
    if expected and x_device_key and hmac.compare_digest(x_device_key, expected):
        return None
    if credentials is not None and credentials.credentials:
        return _user_from_token(credentials.credentials, db)
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Device authentication required")


def require_device_enrollment(
    request: Request,
    x_enrollment_key: str | None = Header(default=None),
) -> None:
    expected = request.app.state.container.settings.device_enrollment_key.strip()
    if not expected or not x_enrollment_key or not hmac.compare_digest(x_enrollment_key, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Device enrollment is not authorized")
