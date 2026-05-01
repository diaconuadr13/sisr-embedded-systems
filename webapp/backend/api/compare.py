from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from backend.config import TEMPLATES_DIR
from backend.services import run_scanner, storage

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/compare", response_class=HTMLResponse)
def compare(request: Request, runs: list[str] = Query(default=[])):
    details = []
    for run_id in runs:
        d = run_scanner.get_run_detail(run_id)
        if d is not None:
            d.tags = storage.get_tags(run_id)
            details.append(d)

    series = []
    for d in details:
        series.append(
            {
                "id": d.id,
                "label": d.model_name + f" ({d.arch} x{d.scale})",
                "epochs": [m.epoch for m in d.metrics],
                "train_loss": [m.train_loss for m in d.metrics],
                "val_loss": [m.val_loss for m in d.metrics],
                "val_psnr": [m.val_psnr for m in d.metrics],
                "val_ssim": [m.val_ssim for m in d.metrics],
            }
        )

    return templates.TemplateResponse(
        request=request,
        name="compare.html",
        context={
            "details": details,
            "series": series,
            "nav": "runs",
        },
    )
