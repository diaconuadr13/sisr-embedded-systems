from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from backend.config import TEMPLATES_DIR
from backend.services import run_scanner, storage

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _run_id(model_name: str, dataset: str, exp_name: str) -> str:
    return f"{model_name}/{dataset}/{exp_name}"


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    runs = run_scanner.list_runs()
    tags_by_run = storage.get_tags_bulk([r.id for r in runs])
    for r in runs:
        r.tags = tags_by_run.get(r.id, [])

    best_per_arch: dict[str, object] = {}
    for r in runs:
        if r.best_psnr is None:
            continue
        current = best_per_arch.get(r.arch)
        if current is None or (current.best_psnr or -1) < r.best_psnr:
            best_per_arch[r.arch] = r
    leaderboard = sorted(best_per_arch.values(), key=lambda r: (r.best_psnr or 0), reverse=True)

    favorites = [r for r in runs if "favorite" in (r.tags or [])]
    favorites.sort(key=lambda r: (r.best_psnr or -1), reverse=True)

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "runs": runs,
            "total_runs": len(runs),
            "leaderboard": leaderboard,
            "favorites": favorites,
            "nav": "dashboard",
        },
    )


@router.get("/runs", response_class=HTMLResponse)
def list_runs(
    request: Request,
    arch: str | None = Query(default=None),
    dataset: str | None = Query(default=None),
    scale: int | None = Query(default=None),
    tag: str | None = Query(default=None),
    q: str | None = Query(default=None),
):
    runs = run_scanner.list_runs()
    tags_by_run = storage.get_tags_bulk([r.id for r in runs])
    for r in runs:
        r.tags = tags_by_run.get(r.id, [])

    if arch:
        runs = [r for r in runs if r.arch == arch]
    if dataset:
        runs = [r for r in runs if r.dataset == dataset]
    if scale is not None:
        runs = [r for r in runs if r.scale == scale]
    if tag:
        runs = [r for r in runs if tag in r.tags]
    if q:
        ql = q.lower()
        runs = [r for r in runs if ql in r.id.lower() or ql in r.model_name.lower()]

    all_runs = run_scanner.list_runs()
    archs = sorted({r.arch for r in all_runs})
    datasets = sorted({r.dataset for r in all_runs})
    scales = sorted({r.scale for r in all_runs})
    all_the_tags = storage.all_tags()

    return templates.TemplateResponse(
        request=request,
        name="runs/list.html",
        context={
            "runs": runs,
            "archs": archs,
            "datasets": datasets,
            "scales": scales,
            "all_tags": all_the_tags,
            "filters": {"arch": arch, "dataset": dataset, "scale": scale, "tag": tag, "q": q},
            "nav": "runs",
        },
    )


@router.get("/runs/{model_name}/{dataset}/{exp_name}", response_class=HTMLResponse)
def run_detail(request: Request, model_name: str, dataset: str, exp_name: str):
    run_id = _run_id(model_name, dataset, exp_name)
    detail = run_scanner.get_run_detail(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    detail.tags = storage.get_tags(run_id)
    detail.note = storage.get_note(run_id)

    epochs = [m.epoch for m in detail.metrics]
    chart_data = {
        "epochs": epochs,
        "train_loss": [m.train_loss for m in detail.metrics],
        "val_loss": [m.val_loss for m in detail.metrics],
        "val_psnr": [m.val_psnr for m in detail.metrics],
        "val_ssim": [m.val_ssim for m in detail.metrics],
        "lr": [m.lr for m in detail.metrics],
    }

    return templates.TemplateResponse(
        request=request,
        name="runs/detail.html",
        context={
            "run": detail,
            "chart_data": chart_data,
            "nav": "runs",
        },
    )


@router.get("/runs/{model_name}/{dataset}/{exp_name}/visuals/{filename}")
def run_visual(model_name: str, dataset: str, exp_name: str, filename: str):
    run_id = _run_id(model_name, dataset, exp_name)
    # Simple filename safety: block anything that is not a flat filename
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    from backend.config import RUNS_DIR

    path = RUNS_DIR / run_id / "visuals" / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@router.get("/runs/{model_name}/{dataset}/{exp_name}/checkpoint/{which}")
def run_checkpoint(model_name: str, dataset: str, exp_name: str, which: str):
    run_id = _run_id(model_name, dataset, exp_name)
    path = run_scanner.checkpoint_path(run_id, which)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Checkpoint '{which}' not found")
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")
