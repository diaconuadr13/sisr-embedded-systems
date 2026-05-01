from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict

from backend.config import PROJECT_ROOT, TEMPLATES_DIR
from backend.services import job_manager, metrics_service

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# Mirrors train.DEFAULT_CONFIG — we don't import train to avoid loading PyTorch at webapp startup.
DEFAULT_FORM_VALUES = {
    "model_name": "",
    "arch": "ESPCN",
    "dataset_name": "DIV2K",
    "scale": 2,
    "batch_size": 64,
    "epochs": 100,
    "lr": 1e-3,
    "num_workers": 4,
    "device": "auto",
    "amp": True,
    "grayscale": False,
    "hr_dir": "data/train/DIV2K_train_HR",
    "val_dir": "data/val/DIV2K_valid_HR",
}

DEVICE_CHOICES = ["auto", "cuda", "cpu", "mps"]


def _scan_dirs(root: str) -> list[str]:
    p = PROJECT_ROOT / root
    if not p.is_dir():
        return []
    return sorted(
        str((p / d).relative_to(PROJECT_ROOT))
        for d in p.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


@router.get("/experiments/new", response_class=HTMLResponse)
def new_experiment(request: Request):
    try:
        archs = metrics_service.list_architectures()
    except Exception:
        archs = [DEFAULT_FORM_VALUES["arch"]]
    # Suggest a fresh model_name using today's timestamp as default
    suggested = f"{DEFAULT_FORM_VALUES['arch']}_x{DEFAULT_FORM_VALUES['scale']}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    values = dict(DEFAULT_FORM_VALUES, model_name=suggested)
    return templates.TemplateResponse(
        request=request,
        name="experiments/new.html",
        context={
            "values": values,
            "archs": archs,
            "devices": DEVICE_CHOICES,
            "train_dirs": _scan_dirs("data/train"),
            "val_dirs": _scan_dirs("data/val"),
            "nav": "new",
        },
    )


class ExperimentForm(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    arch: str
    dataset_name: str = "DIV2K"
    scale: int = 2
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-3
    num_workers: int = 4
    device: str = "auto"
    amp: str = "false"
    grayscale: str = "false"
    hr_dir: str = DEFAULT_FORM_VALUES["hr_dir"]
    val_dir: str = DEFAULT_FORM_VALUES["val_dir"]


@router.post("/experiments")
def submit_experiment(data: Annotated[ExperimentForm, Form()]):
    amp_bool       = data.amp.lower() in {"true", "on", "1", "yes"}
    grayscale_bool = data.grayscale.lower() in {"true", "on", "1", "yes"}
    cfg = {
        "hr_dir": data.hr_dir,
        "val_dir": data.val_dir,
        "model_name": data.model_name.strip() or f"{data.arch}_x{data.scale}",
        "arch": data.arch,
        "dataset_name": data.dataset_name.strip() or "DIV2K",
        "scale": int(data.scale),
        "batch_size": int(data.batch_size),
        "epochs": int(data.epochs),
        "lr": float(data.lr),
        "num_workers": int(data.num_workers),
        "device": data.device,
        "amp": amp_bool,
        "grayscale": grayscale_bool,
    }
    job_id = job_manager.enqueue_single(cfg)
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)
