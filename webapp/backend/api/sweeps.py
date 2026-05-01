from __future__ import annotations

from typing import Any

import yaml
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from backend.config import PROJECT_ROOT, TEMPLATES_DIR
from backend.services import job_manager, metrics_service

DATA_ROOT = PROJECT_ROOT / "data"


def _scan_dirs(sub: str) -> list[str]:
    p = DATA_ROOT / sub
    if not p.is_dir():
        return []
    return sorted(
        str((p / d).relative_to(PROJECT_ROOT))
        for d in p.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

CONFIGS_DIR = PROJECT_ROOT / "configs"

DEFAULT_TEMPLATE = """# Each row overrides train.py defaults. model_name must be unique.
experiments:
  - model_name: "ESPCN_x2_demo"
    arch: "ESPCN"
    scale: 2
    epochs: 3
    lr: 0.001
  - model_name: "FSRCNN_x2_demo"
    arch: "FSRCNN"
    scale: 2
    epochs: 3
    lr: 0.001
"""

DEFAULT_ROWS = [
    {
        "model_name": "ESPCN_x2_demo",
        "arch": "ESPCN",
        "dataset_name": "DIV2K",
        "scale": 2,
        "epochs": 3,
        "lr": 1e-3,
    },
    {
        "model_name": "FSRCNN_x2_demo",
        "arch": "FSRCNN",
        "dataset_name": "DIV2K",
        "scale": 2,
        "epochs": 3,
        "lr": 1e-3,
    },
]


def _extract_experiments(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        exps = data.get("experiments")
    elif isinstance(data, list):
        exps = data
    else:
        exps = None
    if not isinstance(exps, list):
        return []
    return [e for e in exps if isinstance(e, dict)]


def _normalize_row(exp: dict[str, Any], index: int) -> dict[str, Any]:
    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    return {
        "model_name": str(exp.get("model_name") or f"run_{index}"),
        "arch": str(exp.get("arch") or "ESPCN"),
        "dataset_name": str(exp.get("dataset_name") or "DIV2K"),
        "hr_dir": str(exp.get("hr_dir") or ""),
        "scale": _as_int(exp.get("scale"), 2),
        "epochs": _as_int(exp.get("epochs"), 3),
        "lr": _as_float(exp.get("lr"), 1e-3),
        "grayscale": bool(exp.get("grayscale", False)),
    }


def _normalize_rows(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [_normalize_row(exp, idx) for idx, exp in enumerate(experiments, start=1)]
    return rows or list(DEFAULT_ROWS)


def _list_config_files() -> list[str]:
    if not CONFIGS_DIR.is_dir():
        return []
    return sorted(p.name for p in CONFIGS_DIR.glob("*.yaml"))


@router.get("/sweeps", response_class=HTMLResponse)
def list_sweeps(request: Request):
    jobs = [j for j in job_manager.list_jobs(limit=100) if j.type == job_manager.TYPE_SWEEP]
    children_by_job = {j.id: job_manager.list_sweep_children(j.id) for j in jobs}
    return templates.TemplateResponse(
        request=request,
        name="sweeps/list.html",
        context={"jobs": jobs, "children_by_job": children_by_job, "nav": "sweeps"},
    )


@router.get("/sweeps/new", response_class=HTMLResponse)
def new_sweep(request: Request):
    try:
        archs = metrics_service.list_architectures()
    except Exception:
        archs = []
    return templates.TemplateResponse(
        request=request,
        name="sweeps/new.html",
        context={
            "archs": archs,
            "existing": _list_config_files(),
            "template": DEFAULT_TEMPLATE,
            "default_rows": list(DEFAULT_ROWS),
            "train_dirs": _scan_dirs("train"),
            "val_dirs": _scan_dirs("val"),
            "nav": "sweeps",
        },
    )


@router.get("/sweeps/load/{name}", response_class=HTMLResponse)
def load_existing_yaml(name: str):
    if "/" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid name")
    path = CONFIGS_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Config not found")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@router.get("/sweeps/load/{name}/rows")
def load_existing_rows(name: str):
    if "/" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid name")
    path = CONFIGS_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Config not found")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML in {name}: {exc}") from exc
    experiments = _extract_experiments(data)
    if not experiments:
        raise HTTPException(status_code=400, detail="Config has no experiments list")
    return {"rows": _normalize_rows(experiments)}


@router.get("/sweeps/{job_id}", response_class=HTMLResponse)
def sweep_detail(request: Request, job_id: int):
    job = job_manager.get_job(job_id)
    if job is None or job.type != job_manager.TYPE_SWEEP:
        raise HTTPException(status_code=404, detail=f"Sweep job {job_id} not found")
    return templates.TemplateResponse(
        request=request,
        name="sweeps/detail.html",
        context={"job": job, "nav": "sweeps"},
    )


@router.post("/sweeps")
def submit_sweep(yaml_text: str = Form(...)):
    try:
        job_id = job_manager.enqueue_sweep(yaml_text)
    except (yaml.YAMLError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid sweep YAML: {exc}") from exc
    return RedirectResponse(f"/sweeps/{job_id}", status_code=303)
