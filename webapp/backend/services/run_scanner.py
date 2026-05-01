from __future__ import annotations

import csv
import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.config import RUNS_DIR
from backend.schemas import MetricRow, RunDetail, RunSummary

EXP_NAME_RE = re.compile(r"exp_(\d{8})_(\d{6})")
VISUAL_EPOCH_RE = re.compile(r"epoch_(\d{4})_sample_\d+\.png$")


@dataclass
class _CacheEntry:
    detail: RunDetail
    config_mtime: float
    csv_mtime: float
    visuals_mtime: float


_cache: dict[str, _CacheEntry] = {}
_cache_lock = threading.Lock()


def _exp_dirs(runs_dir: Path = RUNS_DIR):
    if not runs_dir.exists():
        return
    for model_dir in sorted(runs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for dataset_dir in sorted(model_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for exp_dir in sorted(dataset_dir.iterdir()):
                if exp_dir.is_dir() and exp_dir.name.startswith("exp_"):
                    yield model_dir.name, dataset_dir.name, exp_dir


def _parse_created_at(exp_name: str) -> datetime | None:
    m = EXP_NAME_RE.search(exp_name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _read_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _read_metrics(csv_path: Path) -> list[MetricRow]:
    if not csv_path.exists():
        return []
    rows: list[MetricRow] = []
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                try:
                    rows.append(
                        MetricRow(
                            epoch=int(raw["Epoch"]),
                            train_loss=float(raw["Train_Loss"]),
                            val_loss=float(raw["Val_Loss"]),
                            val_psnr=float(raw["Val_PSNR"]),
                            val_ssim=float(raw["Val_SSIM"]),
                            lr=float(raw["LR"]),
                        )
                    )
                except (KeyError, ValueError):
                    continue
    except OSError:
        return []
    return rows


def _sample_epochs(visuals_dir: Path) -> list[int]:
    if not visuals_dir.exists():
        return []
    epochs: set[int] = set()
    for png in visuals_dir.iterdir():
        m = VISUAL_EPOCH_RE.search(png.name)
        if m:
            epochs.add(int(m.group(1)))
    return sorted(epochs)


def _build_detail(run_id: str, exp_dir: Path, model_name: str, dataset: str) -> RunDetail:
    config_path = exp_dir / "config.json"
    csv_path = exp_dir / "training_log.csv"
    visuals_dir = exp_dir / "visuals"

    cfg = _read_config(config_path)
    metrics = _read_metrics(csv_path)
    samples = _sample_epochs(visuals_dir)

    best_epoch: int | None = None
    best_psnr: float | None = None
    best_ssim: float | None = None
    for m in metrics:
        if best_psnr is None or m.val_psnr > best_psnr:
            best_psnr = m.val_psnr
            best_ssim = m.val_ssim
            best_epoch = m.epoch

    final = metrics[-1] if metrics else None

    arch = str(cfg.get("arch", "unknown"))
    scale = int(cfg.get("scale", 0) or 0)
    total_epochs = int(cfg.get("epochs", 0) or 0)

    return RunDetail(
        id=run_id,
        model_name=model_name,
        arch=arch,
        dataset=dataset,
        scale=scale,
        exp_name=exp_dir.name,
        created_at=_parse_created_at(exp_dir.name),
        total_epochs=total_epochs,
        completed_epochs=len(metrics),
        best_epoch=best_epoch,
        best_psnr=best_psnr,
        best_ssim=best_ssim,
        final_train_loss=final.train_loss if final else None,
        final_val_loss=final.val_loss if final else None,
        has_best_checkpoint=(exp_dir / "best_model.pth").exists(),
        has_last_checkpoint=(exp_dir / "last_model.pth").exists(),
        config=cfg,
        metrics=metrics,
        sample_epochs=samples,
    )


def get_run_detail(run_id: str) -> RunDetail | None:
    exp_dir = RUNS_DIR / run_id
    if not exp_dir.is_dir():
        return None

    parts = run_id.strip("/").split("/")
    if len(parts) != 3:
        return None
    model_name, dataset, _exp_name = parts

    config_mtime = _safe_mtime(exp_dir / "config.json")
    csv_mtime = _safe_mtime(exp_dir / "training_log.csv")
    visuals_mtime = _safe_mtime(exp_dir / "visuals")

    with _cache_lock:
        cached = _cache.get(run_id)
        if (
            cached
            and cached.config_mtime == config_mtime
            and cached.csv_mtime == csv_mtime
            and cached.visuals_mtime == visuals_mtime
        ):
            return cached.detail

    detail = _build_detail(run_id, exp_dir, model_name, dataset)
    with _cache_lock:
        _cache[run_id] = _CacheEntry(
            detail=detail,
            config_mtime=config_mtime,
            csv_mtime=csv_mtime,
            visuals_mtime=visuals_mtime,
        )
    return detail


def list_runs() -> list[RunSummary]:
    summaries: list[RunSummary] = []
    for model_name, dataset, exp_dir in _exp_dirs():
        run_id = f"{model_name}/{dataset}/{exp_dir.name}"
        detail = get_run_detail(run_id)
        if detail is None:
            continue
        summaries.append(RunSummary(**detail.model_dump(exclude={"config", "metrics", "sample_epochs", "note"})))
    summaries.sort(key=lambda r: (r.created_at or datetime.min), reverse=True)
    return summaries


def invalidate_cache(run_id: str | None = None) -> None:
    with _cache_lock:
        if run_id is None:
            _cache.clear()
        else:
            _cache.pop(run_id, None)


def sample_image_path(run_id: str, epoch: int, sample: int) -> Path | None:
    exp_dir = RUNS_DIR / run_id
    path = exp_dir / "visuals" / f"epoch_{epoch:04d}_sample_{sample}.png"
    return path if path.exists() else None


def checkpoint_path(run_id: str, which: str) -> Path | None:
    if which not in {"best", "last"}:
        return None
    exp_dir = RUNS_DIR / run_id
    path = exp_dir / f"{which}_model.pth"
    return path if path.exists() else None
