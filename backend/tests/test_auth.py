import pytest
from fastapi.testclient import TestClient

from backend.tests.conftest import make_admin

ADMIN_USER, ADMIN_PASS = "root-admin", "root-pass-2026"


@pytest.fixture()
def app():
    from backend.app import auth
    from backend.app.main import app

    make_admin(ADMIN_USER, ADMIN_PASS)
    auth.throttle.__init__()  # reset lockouts between tests
    return app


def login(c, username, password, role="user"):
    """Log in and, on success, make the client send the bearer token on later requests."""
    r = c.post("/api/auth/login", json={"username": username, "password": password, "role": role})
    if r.status_code == 200:
        c.headers["Authorization"] = f"Bearer {r.json()['access_token']}"
    return r


def test_admin_login_returns_bearer_token(app):
    with TestClient(app) as c:
        r = login(c, ADMIN_USER, ADMIN_PASS, "admin")
        body = r.json()
        assert r.status_code == 200 and body["token_type"] == "bearer" and body["user"]["role"] == "admin"
        assert len(body["access_token"]) >= 40
        assert "set-cookie" not in r.headers and "password" not in r.text
        assert c.get("/api/auth/me").json()["username"] == ADMIN_USER


def test_missing_or_bad_token_is_rejected(app):
    with TestClient(app) as c:
        r = c.get("/api/auth/me")
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
        assert c.get("/api/auth/me", headers={"Authorization": "Bearer not-a-real-token"}).status_code == 401
        assert c.get("/api/auth/me", headers={"Authorization": "Basic abc"}).status_code == 401


def test_tokens_are_independent(app):
    with TestClient(app) as a, TestClient(app) as b:
        login(a, ADMIN_USER, ADMIN_PASS, "admin")
        login(b, ADMIN_USER, ADMIN_PASS, "admin")
        assert a.headers["Authorization"] != b.headers["Authorization"]
        a.post("/api/auth/logout")
        assert a.get("/api/auth/me").status_code == 401
        assert b.get("/api/auth/me").status_code == 200  # logging out one device leaves others signed in


def test_wrong_password_and_unknown_user_look_the_same(app):
    with TestClient(app) as c:
        a = login(c, ADMIN_USER, "wrong-password-1")
        b = login(c, "nobody-here", "wrong-password-1")
        assert a.status_code == b.status_code == 401
        assert a.json() == b.json()


def test_admin_creates_user_who_can_log_in_but_not_manage(app):
    with TestClient(app) as admin:
        assert login(admin, ADMIN_USER, ADMIN_PASS, "admin").status_code == 200
        r = admin.post("/api/users", json={"username": "dr.meera", "full_name": "Dr Meera", "password": "slide-reader-77"})
        assert r.status_code == 201 and r.json()["role"] == "user"
        assert admin.post("/api/users", json={"username": "DR.MEERA", "password": "slide-reader-77"}).status_code == 409

    with TestClient(app) as user:
        assert login(user, "dr.meera", "slide-reader-77").status_code == 200
        assert user.get("/api/users").status_code == 403
        assert user.post("/api/users", json={"username": "x-user", "password": "abcdefgh123"}).status_code == 403
        # The admin tab rejects non-admin accounts
        assert login(user, "dr.meera", "slide-reader-77", role="admin").status_code == 403


def test_password_rules(app):
    with TestClient(app) as admin:
        login(admin, ADMIN_USER, ADMIN_PASS, "admin")
        for pw in ["short1", "onlyletterslong", "12345678901"]:
            assert admin.post("/api/users", json={"username": "weak-user", "password": pw}).status_code == 400
        assert admin.post("/api/users", json={"username": "a", "password": "good-pass-123"}).status_code == 400


def test_disabling_user_ends_their_session(app):
    with TestClient(app) as admin, TestClient(app) as user:
        login(admin, ADMIN_USER, ADMIN_PASS, "admin")
        uid = admin.post("/api/users", json={"username": "temp-tech", "password": "temp-pass-123"}).json()["id"]
        login(user, "temp-tech", "temp-pass-123")
        assert user.get("/api/auth/me").status_code == 200
        assert admin.patch(f"/api/users/{uid}", json={"is_active": False}).json()["is_active"] is False
        assert user.get("/api/auth/me").status_code == 401
        assert login(user, "temp-tech", "temp-pass-123").status_code == 403
        assert admin.delete(f"/api/users/{uid}").status_code == 204


def test_cannot_remove_last_admin_or_self(app):
    with TestClient(app) as admin:
        login(admin, ADMIN_USER, ADMIN_PASS, "admin")
        me = admin.get("/api/auth/me").json()["id"]
        assert admin.patch(f"/api/users/{me}", json={"is_active": False}).status_code == 400
        assert admin.delete(f"/api/users/{me}").status_code == 400


def test_logout_invalidates_session(app):
    with TestClient(app) as c:
        login(c, ADMIN_USER, ADMIN_PASS, "admin")
        assert c.post("/api/auth/logout").status_code == 204
        assert c.get("/api/auth/me").status_code == 401  # the same token no longer works


def test_change_password(app):
    with TestClient(app) as admin:
        login(admin, ADMIN_USER, ADMIN_PASS, "admin")
        admin.post("/api/users", json={"username": "pw-changer", "password": "first-pass-123"})
    with TestClient(app) as c:
        login(c, "pw-changer", "first-pass-123")
        assert c.post("/api/auth/change-password", json={"current_password": "nope-nope-1", "new_password": "second-pass-456"}).status_code == 400
        assert c.post("/api/auth/change-password", json={"current_password": "first-pass-123", "new_password": "second-pass-456"}).status_code == 204
        assert login(c, "pw-changer", "second-pass-456").status_code == 200


def test_lockout_after_repeated_failures(app):
    with TestClient(app) as c:
        for _ in range(5):
            assert login(c, ADMIN_USER, "bad-guess-000").status_code == 401
        r = login(c, ADMIN_USER, ADMIN_PASS, "admin")  # even the right password is refused while locked
        assert r.status_code == 429
