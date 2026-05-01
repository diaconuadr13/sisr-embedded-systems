from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class MetricRow(BaseModel):
    epoch: int
    train_loss: float
    val_loss: float
    val_psnr: float
    val_ssim: float
    lr: float


class RunSummary(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    model_name: str
    arch: str
    dataset: str
    scale: int
    exp_name: str
    created_at: datetime | None
    total_epochs: int
    completed_epochs: int
    best_epoch: int | None
    best_psnr: float | None
    best_ssim: float | None
    final_train_loss: float | None
    final_val_loss: float | None
    has_best_checkpoint: bool
    has_last_checkpoint: bool
    tags: list[str] = []


class RunDetail(RunSummary):
    config: dict[str, Any]
    metrics: list[MetricRow]
    sample_epochs: list[int]
    note: str | None = None


class ArchitectureInfo(BaseModel):
    name: str
    params: dict[int, int]
    flops_m: dict[int, float]
    description: str
