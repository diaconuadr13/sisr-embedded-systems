from __future__ import annotations

from pathlib import Path

WEBAPP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WEBAPP_DIR.parent

RUNS_DIR = PROJECT_ROOT / "runs"
DATA_VAL_DIR = PROJECT_ROOT / "data" / "val" / "DIV2K_valid_HR"
DATA_TRAIN_DIR = PROJECT_ROOT / "data" / "train" / "DIV2K_train_HR"

WEBAPP_DATA_DIR = WEBAPP_DIR / "data"
DB_PATH = WEBAPP_DATA_DIR / "webapp.db"
UPLOADS_DIR = WEBAPP_DATA_DIR / "uploads"
JOB_LOGS_DIR = WEBAPP_DATA_DIR / "job_logs"

TEMPLATES_DIR = WEBAPP_DIR / "templates"
STATIC_DIR = WEBAPP_DIR / "static"
SCHEMA_SQL_PATH = WEBAPP_DIR / "backend" / "db" / "schema.sql"
