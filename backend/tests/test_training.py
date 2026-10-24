import json
import os

import pytest
from fastapi.testclient import TestClient

from backend.tests.conftest import make_admin


def write_run(root, model_id, run_id, **over):
    run = {
        "run_id": run_id, "model_id": model_id, "kind": "scratch", "status": "completed",
        "pid": os.getpid(), "host": "elsewhere", "started_at": "2026-10-24T08:00:00Z",
        "updated_at": "2026-10-24T08:10:00Z", "finished_at": "2026-10-24T08:10:00Z",
        "epoch": 2, "batch": 0, "epochs_total": 4, "batches_per_epoch": 10, "best": {"epoch": 2, "val_macro_f1": 0.8},
        "history": [{"epoch": 1, "train_loss": 1.2, "seconds": 30}, {"epoch": 2, "train_loss": 0.9, "seconds": 50}],
        **over,
    }
    d = root / "training" / model_id
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{run_id}.json").write_text(json.dumps(run))
    return run


@pytest.fixture()
def env(monkeypatch, tmp_path):
    from backend.app.main import app

    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    with TestClient(app) as c:
        username, password = make_admin()
        token = c.post("/api/auth/login", json={"username": username, "password": password}).json()["access_token"]
        c.headers["Authorization"] = f"Bearer {token}"
        yield c, tmp_path


def test_lists_runs_running_first_then_newest(env):
    from backend.app.training import utc_now_str

    c, root = env
    now = utc_now_str()
    write_run(root, "cervical-cytology", "old", started_at="2026-10-20T08:00:00Z")
    write_run(root, "cervical-cytology", "new", started_at="2026-10-23T08:00:00Z")
    import socket
    write_run(root, "cervical-cytology", "live", status="running", host=socket.gethostname(), pid=os.getpid(),
              started_at="2026-10-01T08:00:00Z", updated_at=now, finished_at=None, batch=5)
    runs = c.get("/api/training/runs").json()
    assert [r["run_id"] for r in runs] == ["live", "new", "old"]
    live = runs[0]
    assert "history" not in live and live["model_name"] == "Cervical Cytology"
    assert live["progress"] == pytest.approx((2 + 0.5) / 4)
    assert live["avg_epoch_seconds"] == 40 and live["eta_seconds"] == 60


def test_run_detail_has_history(env):
    c, root = env
    write_run(root, "cervical-cytology", "r1")
    run = c.get("/api/training/runs/cervical-cytology/r1").json()
    assert len(run["history"]) == 2 and run["progress"] == 0.5
    assert c.get("/api/training/runs/cervical-cytology/missing").status_code == 404
    assert c.get("/api/training/runs/..%2F..%2Fetc/passwd").status_code == 404


def test_dead_process_is_marked_interrupted(env):
    import socket

    c, root = env
    write_run(root, "cervical-cytology", "crashed", status="running", host=socket.gethostname(),
              pid=999_999, finished_at=None)
    run = c.get("/api/training/runs/cervical-cytology/crashed").json()
    assert run["status"] == "interrupted"


def test_training_is_admin_only(env):
    c, _ = env
    c.post("/api/users", json={"username": "plain-user", "password": "plain-pass-123"})
    with TestClient(c.app) as u:
        token = u.post("/api/auth/login", json={"username": "plain-user", "password": "plain-pass-123"}).json()["access_token"]
        assert u.get("/api/training/runs", headers={"Authorization": f"Bearer {token}"}).status_code == 403
        assert u.get("/api/training/runs").status_code == 401
