"""Password hashing, session tokens and login throttling."""
import hashlib
import hmac
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone

# scrypt parameters (RFC 7914 interactive-login strength); stored in each hash so they can change later
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**14, 8, 1
SESSION_HOURS = 12


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, digest = stored.split("$")
        got = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(got.hex(), digest)
    except ValueError:
        return False


# Verified against when the username doesn't exist, so response time doesn't reveal valid usernames
DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def new_session_token() -> tuple[str, str, str]:
    """Return (bearer token for the client, sha256 stored in the DB, expiry timestamp)."""
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(hours=SESSION_HOURS)
    return token, hash_token(token), expires.strftime("%Y-%m-%d %H:%M:%S")


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def password_problem(password: str) -> str | None:
    if len(password) < 10:
        return "Password must be at least 10 characters."
    if len(password) > 256:
        return "Password is too long."
    if password.isalpha() or password.isdigit():
        return "Password must mix letters with numbers or symbols."
    return None


class LoginThrottle:
    """Lock a username+IP pair out for a while after repeated failed logins (in-memory)."""

    def __init__(self, max_failures=5, window_s=900, lockout_s=900):
        self.max_failures, self.window_s, self.lockout_s = max_failures, window_s, lockout_s
        self._failures: dict[str, list[float]] = {}
        self._locked_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def retry_after(self, key: str) -> int:
        with self._lock:
            return max(0, int(self._locked_until.get(key, 0) - time.time()))

    def failure(self, key: str) -> None:
        now = time.time()
        with self._lock:
            recent = [t for t in self._failures.get(key, []) if now - t < self.window_s] + [now]
            self._failures[key] = recent
            if len(recent) >= self.max_failures:
                self._locked_until[key] = now + self.lockout_s
                self._failures.pop(key, None)

    def success(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)
            self._locked_until.pop(key, None)
