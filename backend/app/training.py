"""Read-only view of training runs written by ml/progress.py (admin only)."""
import json
import os
import re
import socket
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from .auth import Admin

router = APIRouter(prefix="/api/training")
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
STALE_AFTER_S = 30 * 60  # a run that hasn't reported for this long is assumed dead


def _training_dir(request: Request) -> Path:
    return request.app.state.registry.artifacts_root / "training"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _seconds_since(ts: str) -> float:
    return (datetime.now(timezone.utc) - datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)).total_seconds()


def _enrich(run: dict, model_names: dict[str, str]) -> dict:
    """Add derived fields: real status for crashed runs, progress fraction, timing and ETA."""
    if run.get("status") == "running":
        same_host = run.get("host") == socket.gethostname()
        if (same_host and not _pid_alive(run.get("pid", -1))) or _seconds_since(run["updated_at"]) > STALE_AFTER_S:
            run["status"] = "interrupted"
            run["message"] = "The training process stopped without reporting a result."

    total, done = run.get("epochs_total") or 0, run.get("epoch", 0)
    per_epoch = run.get("batches_per_epoch") or 1
    partial = run.get("batch", 0) / per_epoch if run["status"] == "running" else 0
    run["progress"] = min(1.0, (done + partial) / total) if total else 0.0

    # Finished runs: start -> finish. Running: start -> now. Interrupted: start -> last report.
    end = run.get("finished_at") or (None if run["status"] == "running" else run.get("updated_at"))
    end_ago = _seconds_since(end) if end else 0
    run["elapsed_seconds"] = max(0, round(_seconds_since(run["started_at"]) - end_ago))
    secs = [h["seconds"] for h in run.get("history", []) if "seconds" in h]
    run["avg_epoch_seconds"] = round(sum(secs) / len(secs), 1) if secs else None
    run["eta_seconds"] = (round(run["avg_epoch_seconds"] * (total - done - partial))
                          if run["status"] == "running" and run["avg_epoch_seconds"] else None)
    run["model_name"] = model_names.get(run.get("model_id"), run.get("model_id"))
    if run.get("history"):
        run["latest"] = run["history"][-1]
    return run


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


@router.get("/runs")
def list_runs(_: Admin, request: Request, model_id: str | None = None):
    base = _training_dir(request)
    names = {m: s["name"] for m, s in request.app.state.registry.specs.items()}
    pattern = f"{model_id}/*.json" if model_id and SAFE_ID.match(model_id) else "*/*.json"
    runs = []
    for path in base.glob(pattern):
        run = _load(path)
        if run:
            run = _enrich(run, names)
            run.pop("history", None)  # the list stays small; fetch one run for its curves
            runs.append(run)
    # Running runs first, then the rest newest first
    running = [r for r in runs if r["status"] == "running"]
    rest = sorted((r for r in runs if r["status"] != "running"), key=lambda r: r["started_at"], reverse=True)
    return running + rest


@router.get("/runs/{model_id}/{run_id}")
def get_run(model_id: str, run_id: str, _: Admin, request: Request):
    if not (SAFE_ID.match(model_id) and SAFE_ID.match(run_id)):
        raise HTTPException(404, "Training run not found.")
    run = _load(_training_dir(request) / model_id / f"{run_id}.json")
    if not run:
        raise HTTPException(404, "Training run not found.")
    names = {m: s["name"] for m, s in request.app.state.registry.specs.items()}
    return _enrich(run, names)
