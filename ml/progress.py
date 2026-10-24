"""Live training progress, written to artifacts/training/<model_id>/<run_id>.json.

The web portal's Training page reads these files, so a run can be watched while it trains.
Files are replaced atomically so a reader never sees a half-written JSON document.
"""
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RunTracker:
    def __init__(self, artifacts_dir: Path, model_id: str, kind: str, **info):
        self.run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}"
        self.path = artifacts_dir / "training" / model_id / f"{self.run_id}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_write = 0.0
        self.state = {
            "run_id": self.run_id, "model_id": model_id, "kind": kind, "status": "running",
            "pid": os.getpid(), "host": socket.gethostname(),
            "started_at": utcnow(), "updated_at": utcnow(), "finished_at": None,
            "epoch": 0, "batch": 0, "message": None, "best": None, "history": [],
            **info,  # arch, device, epochs_total, batches_per_epoch, train_size, val_size, ...
        }
        self._write()

    def _write(self) -> None:
        self.state["updated_at"] = utcnow()
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=2))
        os.replace(tmp, self.path)
        self._last_write = time.time()

    def batch(self, epoch: int, batch: int, every_s: float = 2.0) -> None:
        """Record progress inside an epoch; throttled so it doesn't slow training down."""
        self.state["batch"] = batch
        self.state["current_epoch"] = epoch
        if time.time() - self._last_write >= every_s:
            self._write()

    def epoch(self, metrics: dict, best: dict | None, epoch_seconds: float) -> None:
        self.state["history"].append({**metrics, "seconds": round(epoch_seconds, 1)})
        self.state["epoch"] = metrics["epoch"]
        self.state["batch"] = 0
        self.state["best"] = best
        self._write()

    def finish(self, status: str, message: str | None = None, **extra) -> None:
        self.state.update(status=status, message=message, finished_at=utcnow(), **extra)
        self._write()
