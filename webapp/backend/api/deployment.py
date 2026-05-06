from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from backend.config import TEMPLATES_DIR
from backend.services import deployment_metrics_service

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/deployment", response_class=HTMLResponse)
def deployment_metrics_page(request: Request):
    reports = deployment_metrics_service.list_reports()
    return templates.TemplateResponse(
        request=request,
        name="deployment/list.html",
        context={
            "reports": reports,
            "reports_dir": deployment_metrics_service.DEPLOYMENT_REPORTS_DIR,
            "nav": "deployment",
        },
    )
