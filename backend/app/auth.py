"""Login with bearer tokens, and admin-only user management."""
import re
import sqlite3
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .db import get_db
from .security import (
    DUMMY_HASH, LoginThrottle, hash_password, hash_token, new_session_token,
    password_problem, verify_password,
)

USERNAME_RE = re.compile(r"^[A-Za-z0-9._-]{3,32}$")

Role = Literal["admin", "user"]
throttle = LoginThrottle()
router = APIRouter(prefix="/api")
Db = Annotated[sqlite3.Connection, Depends(get_db)]
# Clients send "Authorization: Bearer <token>"; this also adds the Authorize button to /docs
bearer = HTTPBearer(auto_error=False)
Credentials = Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]


class User(BaseModel):
    id: int
    username: str
    full_name: str
    role: Role
    is_active: bool
    created_at: str
    last_login_at: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_at: str  # UTC, "YYYY-MM-DD HH:MM:SS"
    user: User


class LoginRequest(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=256)
    role: Role = "user"  # which login tab the person used


class NewUser(BaseModel):
    username: str
    full_name: str = Field("", max_length=100)
    password: str
    role: Role = "user"


class UserUpdate(BaseModel):
    is_active: bool | None = None
    role: Role | None = None
    password: str | None = None


class PasswordChange(BaseModel):
    current_password: str = Field(max_length=256)
    new_password: str


def row_to_user(row: sqlite3.Row) -> User:
    return User(id=row["id"], username=row["username"], full_name=row["full_name"], role=row["role"],
                is_active=bool(row["is_active"]), created_at=row["created_at"], last_login_at=row["last_login_at"])


def current_user(creds: Credentials, db: Db) -> User:
    if creds:
        row = db.execute(
            """SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id
               WHERE s.token_hash = ? AND s.expires_at > datetime('now') AND u.is_active = 1""",
            (hash_token(creds.credentials),),
        ).fetchone()
        if row:
            return row_to_user(row)
    raise HTTPException(401, "Please sign in.", headers={"WWW-Authenticate": "Bearer"})


def require_admin(user: Annotated[User, Depends(current_user)]) -> User:
    if user.role != "admin":
        raise HTTPException(403, "Administrator access required.")
    return user


CurrentUser = Annotated[User, Depends(current_user)]
Admin = Annotated[User, Depends(require_admin)]


# ---- authentication ---------------------------------------------------------

@router.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest, request: Request, db: Db):
    key = f"{body.username.lower()}|{request.client.host if request.client else ''}"
    wait = throttle.retry_after(key)
    if wait:
        raise HTTPException(429, f"Too many failed attempts. Try again in {wait // 60 + 1} minutes.")

    row = db.execute("SELECT * FROM users WHERE username = ?", (body.username.strip(),)).fetchone()
    valid = verify_password(body.password, row["password_hash"] if row else DUMMY_HASH)
    if not row or not valid:
        throttle.failure(key)
        raise HTTPException(401, "Incorrect username or password.")
    if not row["is_active"]:
        raise HTTPException(403, "This account is disabled. Contact your administrator.")
    if body.role == "admin" and row["role"] != "admin":
        raise HTTPException(403, "This account is not an administrator. Use the User login tab.")
    throttle.success(key)

    token, token_hash, expires = new_session_token()
    db.execute("DELETE FROM sessions WHERE expires_at <= datetime('now')")
    db.execute("INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)", (token_hash, row["id"], expires))
    db.execute("UPDATE users SET last_login_at = datetime('now') WHERE id = ?", (row["id"],))
    return TokenResponse(access_token=token, expires_at=expires, user=row_to_user(row))


@router.post("/auth/logout", status_code=204)
def logout(creds: Credentials, db: Db):
    """Revoke the token used for this request."""
    if creds:
        db.execute("DELETE FROM sessions WHERE token_hash = ?", (hash_token(creds.credentials),))


@router.get("/auth/me", response_model=User)
def me(user: CurrentUser):
    return user


@router.post("/auth/change-password", status_code=204)
def change_password(body: PasswordChange, user: CurrentUser, creds: Credentials, db: Db):
    row = db.execute("SELECT password_hash FROM users WHERE id = ?", (user.id,)).fetchone()
    if not verify_password(body.current_password, row["password_hash"]):
        raise HTTPException(400, "Current password is incorrect.")
    if problem := password_problem(body.new_password):
        raise HTTPException(400, problem)
    db.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(body.new_password), user.id))
    # Revoke every other token for this account
    keep = hash_token(creds.credentials)
    db.execute("DELETE FROM sessions WHERE user_id = ? AND token_hash != ?", (user.id, keep))


# ---- user management (admin only) -------------------------------------------

@router.get("/users", response_model=list[User])
def list_users(_: Admin, db: Db):
    return [row_to_user(r) for r in db.execute("SELECT * FROM users ORDER BY role, username")]


@router.post("/users", response_model=User, status_code=201)
def create_user(body: NewUser, admin: Admin, db: Db):
    username = body.username.strip()
    if not USERNAME_RE.match(username):
        raise HTTPException(400, "Username must be 3-32 characters: letters, numbers, dot, dash or underscore.")
    if problem := password_problem(body.password):
        raise HTTPException(400, problem)
    try:
        cur = db.execute(
            "INSERT INTO users (username, full_name, password_hash, role, created_by) VALUES (?, ?, ?, ?, ?)",
            (username, body.full_name.strip(), hash_password(body.password), body.role, admin.id),
        )
    except sqlite3.IntegrityError:
        raise HTTPException(409, f"Username '{username}' is already taken.")
    return row_to_user(db.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone())


def _get_user_row(db: sqlite3.Connection, user_id: int) -> sqlite3.Row:
    row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        raise HTTPException(404, "User not found.")
    return row


def _other_active_admins(db: sqlite3.Connection, user_id: int) -> int:
    return db.execute("SELECT COUNT(*) FROM users WHERE role = 'admin' AND is_active = 1 AND id != ?",
                      (user_id,)).fetchone()[0]


@router.patch("/users/{user_id}", response_model=User)
def update_user(user_id: int, body: UserUpdate, admin: Admin, db: Db):
    row = _get_user_row(db, user_id)
    losing_admin = row["role"] == "admin" and (body.role == "user" or body.is_active is False)
    if losing_admin and _other_active_admins(db, user_id) == 0:
        raise HTTPException(400, "There must always be at least one active administrator.")
    if user_id == admin.id and (body.is_active is False or body.role == "user"):
        raise HTTPException(400, "You can't disable or demote your own account.")

    if body.password is not None:
        if problem := password_problem(body.password):
            raise HTTPException(400, problem)
        db.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(body.password), user_id))
    if body.role is not None:
        db.execute("UPDATE users SET role = ? WHERE id = ?", (body.role, user_id))
    if body.is_active is not None:
        db.execute("UPDATE users SET is_active = ? WHERE id = ?", (int(body.is_active), user_id))
    if body.is_active is False or body.password is not None:
        db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))  # force re-login
    return row_to_user(_get_user_row(db, user_id))


@router.delete("/users/{user_id}", status_code=204)
def delete_user(user_id: int, admin: Admin, db: Db):
    row = _get_user_row(db, user_id)
    if user_id == admin.id:
        raise HTTPException(400, "You can't delete your own account.")
    if row["role"] == "admin" and _other_active_admins(db, user_id) == 0:
        raise HTTPException(400, "There must always be at least one active administrator.")
    db.execute("DELETE FROM users WHERE id = ?", (user_id,))
