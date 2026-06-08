"""Authentication: password hashing, server-side sessions, and FastAPI deps.

Sessions are stored server-side in the database; the browser only holds an
opaque, httponly cookie token. Password hashing uses stdlib pbkdf2_hmac (no
extra dependencies).
"""

import hashlib
import secrets

from fastapi import Request

SESSION_COOKIE = "tk_session"
_ALGO = "pbkdf2_sha256"
_ROUNDS = 200_000


class NotAuthenticated(Exception):
    """Raised when a request has no valid session."""


class NotAuthorized(Exception):
    """Raised when an authenticated user lacks the required role."""


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ROUNDS)
    return f"{_ALGO}${_ROUNDS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, rounds_s, salt_hex, hash_hex = stored.split("$")
        if algo != _ALGO:
            return False
        rounds = int(rounds_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, AttributeError):
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return secrets.compare_digest(dk, expected)


async def get_current_user(request: Request) -> dict | None:
    """Resolve the logged-in user from the session cookie, or None."""
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    db = request.app.state.db
    return await db.get_user_by_session(token)


async def require_user(request: Request) -> dict:
    """FastAPI dependency: require an authenticated user."""
    user = await get_current_user(request)
    if user is None:
        raise NotAuthenticated()
    return user


async def require_admin(request: Request) -> dict:
    """FastAPI dependency: require an authenticated admin user."""
    user = await require_user(request)
    if not user.get("is_admin"):
        raise NotAuthorized()
    return user
