from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.config import PROJECT_ROOT

DEPLOYMENT_REPORTS_DIR = PROJECT_ROOT / "reports" / "deployment_metrics"


@dataclass
class DeploymentReport:
    filename: str
    path: Path
    arch: str
    scale: int | None
    input_tile: list[int] | None
    output_tile: list[int] | None
    target: str | None
    inference_ms: float | None
    macs: int | None
    mops: float | None
    mops_per_watt: float | None
    energy_per_inference_mj: float | None
    psnr_gain: float | None
    ssim_gain: float | None
    warnings: list[str]
    raw: dict[str, Any]


def _as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _as_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None


def _as_tile(value: Any) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    if not all(isinstance(v, int) for v in value):
        return None
    return value


def _from_json(path: Path) -> DeploymentReport | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None

    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    model = data.get("model") if isinstance(data.get("model"), dict) else {}
    runtime = data.get("runtime") if isinstance(data.get("runtime"), dict) else {}
    compute = data.get("compute") if isinstance(data.get("compute"), dict) else {}
    power = data.get("power") if isinstance(data.get("power"), dict) else {}
    quality = data.get("quality") if isinstance(data.get("quality"), dict) else {}
    warnings = metadata.get("warnings") if isinstance(metadata.get("warnings"), list) else []

    return DeploymentReport(
        filename=path.name,
        path=path,
        arch=str(model.get("arch", "unknown")),
        scale=_as_int(model.get("scale")),
        input_tile=_as_tile(model.get("input_tile")),
        output_tile=_as_tile(model.get("output_tile")),
        target=str(runtime["target"]) if runtime.get("target") is not None else None,
        inference_ms=_as_float(runtime.get("inference_ms")),
        macs=_as_int(compute.get("macs")),
        mops=_as_float(compute.get("mops")),
        mops_per_watt=_as_float(power.get("mops_per_watt")),
        energy_per_inference_mj=_as_float(power.get("energy_per_inference_mj")),
        psnr_gain=_as_float(quality.get("psnr_gain")),
        ssim_gain=_as_float(quality.get("ssim_gain")),
        warnings=[str(w) for w in warnings],
        raw=data,
    )


def list_reports() -> list[DeploymentReport]:
    if not DEPLOYMENT_REPORTS_DIR.is_dir():
        return []

    reports: list[DeploymentReport] = []
    for path in sorted(DEPLOYMENT_REPORTS_DIR.glob("*.json")):
        report = _from_json(path)
        if report is not None:
            reports.append(report)
    return reports
