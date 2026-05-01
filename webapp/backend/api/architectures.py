from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from backend.config import TEMPLATES_DIR
from backend.services import metrics_service

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/architectures", response_class=HTMLResponse)
def architectures(request: Request):
    arch_names = metrics_service.list_architectures()
    scales = (2, 3, 4)

    rows = []
    for name in arch_names:
        per_scale = {}
        for s in scales:
            try:
                stats = metrics_service.arch_stats(name, s)
            except Exception as e:
                stats = {"params": None, "flops_m": None, "error": str(e)}
            per_scale[s] = stats
        rows.append(
            {
                "name": name,
                "description": metrics_service.architecture_description(name),
                "per_scale": per_scale,
            }
        )

    return templates.TemplateResponse(
        request=request,
        name="architectures.html",
        context={
            "rows": rows,
            "scales": scales,
            "nav": "architectures",
        },
    )
