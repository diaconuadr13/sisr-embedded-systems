from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from backend.config import PROJECT_ROOT, TEMPLATES_DIR
from backend.services import job_manager, run_scanner

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _enrich_children(children: list[dict]) -> list[dict]:
    """Add best_psnr and best_ssim to completed sweep children by reading their exp_dir."""
    enriched = []
    for child in children:
        exp_dir = child.get("exp_dir")
        if exp_dir:
            rel = exp_dir.replace("runs/", "", 1)
            detail = run_scanner.get_run_detail(rel)
            child = dict(child)
            child["best_psnr"] = detail.best_psnr if detail else None
            child["best_ssim"] = detail.best_ssim if detail else None
        enriched.append(child)
    return enriched


def _sweep_planned(job: job_manager.JobRow) -> list[dict]:
    """Return the ordered experiment list from the sweep YAML."""
    if not job.config_yaml:
        return []
    try:
        data = yaml.safe_load(job.config_yaml)
    except yaml.YAMLError:
        return []
    exps = data.get("experiments", data) if isinstance(data, dict) else data
    if not isinstance(exps, list):
        return []
    return [
        {
            "index": i + 1,
            "model_name": e.get("model_name", f"run_{i+1}"),
            "arch": e.get("arch", "?"),
            "scale": e.get("scale", 2),
            "epochs": e.get("epochs", "?"),
            "dataset_name": e.get("dataset_name", "?"),
        }
        for i, e in enumerate(exps)
        if isinstance(e, dict)
    ]


def _sweep_active_metrics(job: job_manager.JobRow, children: list[dict]) -> dict | None:
    """Find the currently running experiment and return its live metrics."""
    if not job.config_yaml or job.status != "running":
        return None
    try:
        data = yaml.safe_load(job.config_yaml)
    except yaml.YAMLError:
        return None
    exps = data.get("experiments", data) if isinstance(data, dict) else data
    if not isinstance(exps, list):
        return None

    done_names = {c["config"].get("model_name") for c in children}
    started_ts = datetime.fromisoformat(job.started_at).timestamp() if job.started_at else 0

    for exp in exps:
        if not isinstance(exp, dict):
            continue
        model_name = exp.get("model_name")
        if model_name in done_names:
            continue
        dataset_name = exp.get("dataset_name", "")
        parent = PROJECT_ROOT / "runs" / model_name / dataset_name
        if not parent.is_dir():
            continue
        # Find the exp_ subdir created after job started
        for candidate in sorted(parent.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not (candidate.is_dir() and candidate.name.startswith("exp_")):
                continue
            if candidate.stat().st_mtime < started_ts - 5:
                break
            detail = run_scanner.get_run_detail(
                f"{model_name}/{dataset_name}/{candidate.name}"
            )
            if detail is None:
                continue
            last = detail.metrics[-1] if detail.metrics else None
            return {
                "model_name": model_name,
                "arch": exp.get("arch", "?"),
                "epoch": last.epoch if last else 0,
                "total_epochs": detail.total_epochs,
                "val_psnr": last.val_psnr if last else None,
                "val_ssim": last.val_ssim if last else None,
            }
    return None


@router.get("/jobs", response_class=HTMLResponse)
def list_jobs_page(request: Request):
    jobs = job_manager.list_jobs(limit=100)
    return templates.TemplateResponse(
        request=request,
        name="jobs/list.html",
        context={"jobs": jobs, "nav": "jobs"},
    )


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_detail(request: Request, job_id: int):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return templates.TemplateResponse(
        request=request,
        name="jobs/detail.html",
        context={"job": job, "nav": "jobs"},
    )


@router.get("/jobs/{job_id}/progress", response_class=HTMLResponse)
def job_progress(request: Request, job_id: int):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    live = {
        "epoch": None,
        "total_epochs": None,
        "val_psnr": None,
        "val_ssim": None,
        "metrics": [],
        "children": [],
    }

    if job.type == job_manager.TYPE_SINGLE and job.exp_dir:
        detail = run_scanner.get_run_detail(job.exp_dir)
        if detail is not None:
            live["total_epochs"] = detail.total_epochs
            if detail.metrics:
                last = detail.metrics[-1]
                live["epoch"] = last.epoch
                live["val_psnr"] = last.val_psnr
                live["val_ssim"] = last.val_ssim
                live["metrics"] = [m.model_dump() for m in detail.metrics]
    elif job.type == job_manager.TYPE_SWEEP:
        children = job_manager.list_sweep_children(job.id)
        live["children"] = _enrich_children(children)
        live["planned"] = _sweep_planned(job)
        live["active"] = _sweep_active_metrics(job, live["children"])

    stdout_tail = job_manager.read_log_tail(job.stdout_path) if job.stdout_path else ""

    return templates.TemplateResponse(
        request=request,
        name="partials/job_progress.html",
        context={
            "job": job,
            "live": live,
            "stdout_tail": stdout_tail,
        },
    )


@router.post("/jobs/{job_id}/cancel")
def job_cancel(job_id: int):
    ok = job_manager.cancel(job_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Cannot cancel this job")
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)
