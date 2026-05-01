from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from backend.config import JOB_LOGS_DIR, PROJECT_ROOT
from backend.services import storage

# ---- Status constants ----
QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"

TYPE_SINGLE = "single"
TYPE_SWEEP = "sweep"

_POLL_INTERVAL_SEC = 1.0


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class JobRow:
    id: int
    type: str
    status: str
    config_yaml: str | None
    exp_dir: str | None
    stdout_path: str | None
    pid: int | None
    error: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None

    @classmethod
    def from_row(cls, row) -> "JobRow":
        return cls(
            id=row["id"],
            type=row["type"],
            status=row["status"],
            config_yaml=row["config_yaml"],
            exp_dir=row["exp_dir"],
            stdout_path=row["stdout_path"],
            pid=row["pid"],
            error=row["error"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )


# ---- DB helpers ----

def _insert_job(type_: str, config_yaml: str) -> int:
    with storage.connection() as conn:
        cur = conn.execute(
            "INSERT INTO jobs(type, status, config_yaml, created_at) VALUES (?, ?, ?, ?)",
            (type_, QUEUED, config_yaml, _iso_now()),
        )
        return int(cur.lastrowid)


def _fetch_job(job_id: int) -> JobRow | None:
    with storage.connection() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return JobRow.from_row(row) if row else None


def _next_queued() -> JobRow | None:
    with storage.connection() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status = ? ORDER BY id ASC LIMIT 1", (QUEUED,)
        ).fetchone()
    return JobRow.from_row(row) if row else None


def list_jobs(limit: int = 50) -> list[JobRow]:
    with storage.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [JobRow.from_row(r) for r in rows]


def list_active() -> list[JobRow]:
    with storage.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE status IN (?, ?) ORDER BY id ASC", (QUEUED, RUNNING)
        ).fetchall()
    return [JobRow.from_row(r) for r in rows]


def list_sweep_children(parent_job_id: int) -> list[dict[str, Any]]:
    """For sweep jobs, read the batch log JSON that run_experiments.py writes."""
    child_dir = JOB_LOGS_DIR / f"sweep_{parent_job_id}"
    if not child_dir.is_dir():
        return []
    # batch_TS.json - latest one wins (usually only one per job)
    candidates = sorted(child_dir.glob("batch_*.json"))
    if not candidates:
        return []
    try:
        return json.loads(candidates[-1].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


# ---- Sanitizing job startup state on cold boot ----

def reset_running_on_boot() -> None:
    """Mark previously running jobs as interrupted after a process restart."""
    with storage.connection() as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, error = ?, finished_at = ? "
            "WHERE status = ?",
            (FAILED, "Job interrupted (webapp restart)", _iso_now(), RUNNING),
        )


# ---- Public enqueue API ----

def enqueue_single(cfg: dict[str, Any]) -> int:
    text = yaml.safe_dump(cfg, sort_keys=False)
    job_id = _insert_job(TYPE_SINGLE, text)
    _manager.wake()
    return job_id


def enqueue_sweep(yaml_text: str) -> int:
    # Validate that it parses and is a list or {experiments: [...]}
    data = yaml.safe_load(yaml_text)
    if isinstance(data, dict):
        exps = data.get("experiments")
    elif isinstance(data, list):
        exps = data
    else:
        raise ValueError("Sweep YAML must be a list or {'experiments': [...]}.")
    if not isinstance(exps, list) or not exps:
        raise ValueError("Sweep must contain at least one experiment entry.")
    job_id = _insert_job(TYPE_SWEEP, yaml_text)
    _manager.wake()
    return job_id


def cancel(job_id: int) -> bool:
    return _manager.cancel(job_id)


# ---- Worker thread ----

class _JobManager:
    def __init__(self) -> None:
        self._wake = threading.Event()
        self._current_pid: int | None = None
        self._current_job_id: int | None = None
        self._thread = threading.Thread(target=self._loop, name="sisr-job-worker", daemon=True)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        JOB_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        reset_running_on_boot()
        self._thread.start()

    def wake(self) -> None:
        self._wake.set()

    def cancel(self, job_id: int) -> bool:
        job = _fetch_job(job_id)
        if job is None:
            return False
        if job.status == QUEUED:
            with storage.connection() as conn:
                conn.execute(
                    "UPDATE jobs SET status = ?, finished_at = ?, error = ? WHERE id = ? AND status = ?",
                    (CANCELLED, _iso_now(), "Cancelled before start", job_id, QUEUED),
                )
            return True
        if job.status == RUNNING and self._current_job_id == job_id and self._current_pid:
            try:
                os.killpg(os.getpgid(self._current_pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(self._current_pid, signal.SIGTERM)
                except ProcessLookupError:
                    return False
            return True
        return False

    def _loop(self) -> None:
        while True:
            job = _next_queued()
            if job is None:
                self._wake.wait(timeout=2.0)
                self._wake.clear()
                continue
            try:
                self._execute(job)
            except Exception as exc:  # noqa: BLE001
                with storage.connection() as conn:
                    conn.execute(
                        "UPDATE jobs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                        (FAILED, f"Worker error: {exc!r}", _iso_now(), job.id),
                    )

    def _execute(self, job: JobRow) -> None:
        log_path = JOB_LOGS_DIR / f"job_{job.id}.log"
        cmd, extra_cleanup = self._build_command(job)

        with storage.connection() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, started_at = ?, stdout_path = ? WHERE id = ?",
                (RUNNING, _iso_now(), str(log_path), job.id),
            )

        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        logf = log_path.open("w", encoding="utf-8", buffering=1)
        try:
            proc = subprocess.Popen(  # noqa: S603
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._current_pid = proc.pid
            self._current_job_id = job.id

            with storage.connection() as conn:
                conn.execute("UPDATE jobs SET pid = ? WHERE id = ?", (proc.pid, job.id))

            # Poll until completion; also discover exp_dir for single jobs.
            discovered_exp_dir = False
            while True:
                ret = proc.poll()
                if ret is not None:
                    break
                if not discovered_exp_dir and job.type == TYPE_SINGLE:
                    exp_dir = _discover_exp_dir_for_single(job)
                    if exp_dir is not None:
                        with storage.connection() as conn:
                            conn.execute("UPDATE jobs SET exp_dir = ? WHERE id = ?", (exp_dir, job.id))
                        discovered_exp_dir = True
                time.sleep(_POLL_INTERVAL_SEC)

            # Final status
            if ret == 0:
                status = SUCCEEDED
                err = None
            elif ret < 0 and -ret in (signal.SIGTERM, signal.SIGKILL):
                status = CANCELLED
                err = f"Process terminated (signal {-ret})"
            else:
                status = FAILED
                err = f"Process exited with code {ret}"

            with storage.connection() as conn:
                conn.execute(
                    "UPDATE jobs SET status = ?, finished_at = ?, error = ?, pid = NULL WHERE id = ?",
                    (status, _iso_now(), err, job.id),
                )
        finally:
            self._current_pid = None
            self._current_job_id = None
            try:
                logf.close()
            except Exception:  # noqa: BLE001
                pass
            for path in extra_cleanup:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass

    def _build_command(self, job: JobRow) -> tuple[list[str], list[Path]]:
        tmp_dir = JOB_LOGS_DIR / f"job_{job.id}_inputs"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        if job.type == TYPE_SINGLE:
            yaml_path = tmp_dir / "config.yaml"
            yaml_path.write_text(job.config_yaml or "", encoding="utf-8")
            return (
                [sys.executable, "-u", "train.py", "--config", str(yaml_path)],
                [yaml_path],
            )
        elif job.type == TYPE_SWEEP:
            yaml_path = tmp_dir / "sweep.yaml"
            yaml_path.write_text(job.config_yaml or "", encoding="utf-8")
            log_dir = JOB_LOGS_DIR / f"sweep_{job.id}"
            log_dir.mkdir(parents=True, exist_ok=True)
            return (
                [
                    sys.executable, "-u", "run_experiments.py",
                    str(yaml_path), "--log_dir", str(log_dir),
                ],
                [yaml_path],
            )
        else:
            raise ValueError(f"Unknown job type: {job.type}")


def _discover_exp_dir_for_single(job: JobRow) -> str | None:
    """Infer which runs/<arch>/<dataset>/exp_* dir the training process is writing to."""
    if not job.config_yaml:
        return None
    try:
        cfg = yaml.safe_load(job.config_yaml) or {}
    except yaml.YAMLError:
        return None
    model_name = cfg.get("model_name")
    dataset_name = cfg.get("dataset_name")
    if not model_name or not dataset_name:
        return None

    parent = PROJECT_ROOT / "runs" / model_name / dataset_name
    if not parent.is_dir():
        return None

    started = datetime.fromisoformat(job.started_at) if job.started_at else None
    best: tuple[float, str] | None = None
    for child in parent.iterdir():
        if not (child.is_dir() and child.name.startswith("exp_")):
            continue
        try:
            mtime = child.stat().st_mtime
        except OSError:
            continue
        if started is not None and mtime < started.timestamp() - 5:
            continue
        if best is None or mtime > best[0]:
            best = (mtime, f"{model_name}/{dataset_name}/{child.name}")
    return best[1] if best else None


def read_log_tail(log_path: str | Path, max_bytes: int = 8192) -> str:
    path = Path(log_path)
    if not path.is_file():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(-max_bytes, os.SEEK_END)
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


_manager = _JobManager()


def start_worker() -> None:
    _manager.start()


def get_job(job_id: int) -> JobRow | None:
    return _fetch_job(job_id)
