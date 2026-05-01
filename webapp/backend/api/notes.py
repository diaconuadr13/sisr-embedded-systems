from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from backend.config import TEMPLATES_DIR
from backend.services import storage

router = APIRouter(prefix="/api")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _run_id(model_name: str, dataset: str, exp_name: str) -> str:
    return f"{model_name}/{dataset}/{exp_name}"


@router.post("/notes/{model_name}/{dataset}/{exp_name}", response_class=HTMLResponse)
def save_note(
    request: Request,
    model_name: str,
    dataset: str,
    exp_name: str,
    body: str = Form(default=""),
):
    run_id = _run_id(model_name, dataset, exp_name)
    storage.upsert_note(run_id, body)
    return templates.TemplateResponse(
        request=request,
        name="partials/note_saved.html",
        context={"body": storage.get_note(run_id) or ""},
    )


@router.post("/tags/{model_name}/{dataset}/{exp_name}", response_class=HTMLResponse)
def add_tag(
    request: Request,
    model_name: str,
    dataset: str,
    exp_name: str,
    tag: str = Form(...),
):
    run_id = _run_id(model_name, dataset, exp_name)
    storage.add_tag(run_id, tag)
    return templates.TemplateResponse(
        request=request,
        name="partials/tags.html",
        context={"run": {"id": run_id, "model_name": model_name, "dataset": dataset, "exp_name": exp_name},
                 "tags": storage.get_tags(run_id)},
    )


@router.delete("/tags/{model_name}/{dataset}/{exp_name}/{tag}", response_class=HTMLResponse)
def remove_tag(
    request: Request,
    model_name: str,
    dataset: str,
    exp_name: str,
    tag: str,
):
    run_id = _run_id(model_name, dataset, exp_name)
    storage.remove_tag(run_id, tag)
    return templates.TemplateResponse(
        request=request,
        name="partials/tags.html",
        context={"run": {"id": run_id, "model_name": model_name, "dataset": dataset, "exp_name": exp_name},
                 "tags": storage.get_tags(run_id)},
    )
