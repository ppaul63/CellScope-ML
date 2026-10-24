import pytest


@pytest.fixture(autouse=True, scope="session")
def isolated_db(tmp_path_factory):
    """Every test session gets its own throwaway user database."""
    mp = pytest.MonkeyPatch()
    mp.setenv("CELLSCOPE_DB", str(tmp_path_factory.mktemp("db") / "test.db"))
    yield
    mp.undo()


def make_admin(username="admin", password="admin-pass-123"):
    from backend.app.db import connect, init_db
    from backend.app.security import hash_password

    init_db()
    conn = connect()
    conn.execute("INSERT OR IGNORE INTO users (username, password_hash, role) VALUES (?, ?, 'admin')",
                 (username, hash_password(password)))
    conn.commit()
    conn.close()
    return username, password
