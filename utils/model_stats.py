from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

OPS_CONVENTION = "1 MAC = 2 operations"


def count_trainable_params(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def calculate_file_kb(path: str | Path | None) -> float | None:
    if path is None:
        return None

    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None
    return file_path.stat().st_size / 1024.0


def calculate_tile_metrics(tile_h: int, tile_w: int, scale: int) -> dict[str, Any]:
    if tile_h <= 0:
        raise ValueError("tile_h must be > 0")
    if tile_w <= 0:
        raise ValueError("tile_w must be > 0")
    if scale <= 0:
        raise ValueError("scale must be > 0")
    input_tile = [tile_h, tile_w]
    output_tile = [tile_h * scale, tile_w * scale]
    return {
        "input_tile": input_tile,
        "output_tile": output_tile,
        "output_pixels": output_tile[0] * output_tile[1],
    }


def _parse_value(key: str, raw: str, line_number: int) -> Any:
    """Parse a board-log value and include context in any numeric parsing error."""
    try:
        if key == "tile":
            parts = raw.lower().split("x")
            if len(parts) != 2:
                raise ValueError("invalid tile format")
            h, w = parts
            return [int(h), int(w)]
        if key in {"scale", "free_heap_before", "free_heap_after", "tensor_arena_bytes"}:
            return int(raw)
        if key in {"inference_ms", "sample_ms"}:
            return float(raw)
        return raw
    except ValueError as exc:
        raise ValueError(f"line {line_number}: invalid value for key '{key}': {raw!r}") from exc


def calculate_compute_metrics(
    macs: int | None,
    inference_ms: float | None,
    output_pixels: int,
) -> dict[str, float | int | None]:
    if macs is not None and macs < 0:
        raise ValueError("macs must be >= 0")
    if output_pixels <= 0:
        raise ValueError("output_pixels must be > 0")
    if inference_ms is not None and inference_ms <= 0:
        raise ValueError("inference_ms must be > 0")

    ops: int | None = None if macs is None else macs * 2
    metrics = {
        "macs": macs,
        "ops": ops,
        "mops": None,
        "output_pixels_per_second": None,
        "ops_convention": OPS_CONVENTION,
    }

    if inference_ms is not None and inference_ms > 0:
        inference_seconds = inference_ms / 1000.0
        if ops is not None:
            metrics["mops"] = ops / inference_seconds / 1_000_000.0
        if output_pixels > 0:
            metrics["output_pixels_per_second"] = output_pixels / inference_seconds

    return metrics


def calculate_power_metrics(
    *,
    voltage_v: float,
    idle_current_ma: float,
    inference_current_ma: float,
    inference_ms: float,
    output_pixels: int,
    mops: float | None,
    ops: int | None,
    psnr_gain: float | None,
) -> dict[str, float | None]:
    if voltage_v <= 0:
        raise ValueError("voltage_v must be > 0")
    if idle_current_ma < 0:
        raise ValueError("idle_current_ma must be >= 0")
    if inference_current_ma <= 0:
        raise ValueError("inference_current_ma must be > 0")
    if inference_ms <= 0:
        raise ValueError("inference_ms must be > 0")
    if output_pixels <= 0:
        raise ValueError("output_pixels must be > 0")
    if ops is not None and ops <= 0:
        raise ValueError("ops must be > 0")

    inference_seconds = inference_ms / 1000.0
    idle_power_mw = voltage_v * idle_current_ma
    inference_power_mw = voltage_v * inference_current_ma
    energy_per_inference_mj = inference_power_mw * inference_seconds
    power_w = inference_power_mw / 1000.0

    return {
        "voltage_v": voltage_v,
        "idle_current_ma": idle_current_ma,
        "inference_current_ma": inference_current_ma,
        "idle_power_mw": idle_power_mw,
        "inference_power_mw": inference_power_mw,
        "energy_per_inference_mj": energy_per_inference_mj,
        "energy_per_output_pixel_uj": energy_per_inference_mj * 1000.0 / output_pixels,
        "mops_per_watt": None if mops is None else mops / power_w,
        "energy_per_operation_nj": None if ops is None else energy_per_inference_mj * 1_000_000.0 / ops,
        "psnr_gain_per_mj": None if psnr_gain is None else psnr_gain / energy_per_inference_mj,
    }


def calculate_runtime_summary(samples_ms: list[float]) -> dict[str, float] | None:
    if not samples_ms:
        return None
    return {
        "mean_ms": statistics.fmean(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "std_ms": statistics.pstdev(samples_ms),
    }


def parse_board_log(path: str | Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    samples: list[float] = []

    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue

        key, raw_value = text.split("=", maxsplit=1)
        key = key.strip()
        value = _parse_value(key, raw_value.strip(), line_number)
        if key == "sample_ms":
            samples.append(float(value))
        else:
            result[key] = value

    if samples:
        result["samples_ms"] = samples
        result["sample_summary"] = calculate_runtime_summary(samples)

    return result


def _resolve_inference_ms_for_report(
    runtime: dict[str, Any] | None,
) -> float | None:
    """Return explicit inference_ms when available, otherwise sample summary mean_ms."""
    if runtime is None:
        return None

    inference_ms = runtime.get("inference_ms")
    if isinstance(inference_ms, (int, float)):
        return float(inference_ms)

    sample_summary = runtime.get("sample_summary")
    mean_ms = None if not isinstance(sample_summary, dict) else sample_summary.get("mean_ms")
    if isinstance(mean_ms, (int, float)):
        return float(mean_ms)
    return None


def _build_runtime_warnings(
    runtime: dict[str, Any] | None,
    *,
    tile_h: int,
    tile_w: int,
    scale: int,
) -> list[str]:
    warnings: list[str] = []
    if runtime is None:
        return warnings

    runtime_tile = runtime.get("tile")
    if isinstance(runtime_tile, list) and len(runtime_tile) == 2:
        if runtime_tile != [tile_h, tile_w]:
            warnings.append(
                f"Board log tile {runtime_tile[0]}x{runtime_tile[1]} does not match CLI tile {tile_h}x{tile_w}."
            )

    runtime_scale = runtime.get("scale")
    if runtime_scale is not None and runtime_scale != scale:
        warnings.append(
            f"Board log scale {runtime_scale} does not match resolved scale {scale}."
        )

    return warnings


def build_deployment_report(
    *,
    arch: str,
    scale: int,
    tile_h: int,
    tile_w: int,
    params: int,
    checkpoint_path: str | Path,
    tflite_float32_path: str | Path | None,
    tflite_int8_path: str | Path | None,
    c_header_path: str | Path | None,
    quality: dict[str, Any] | None,
    runtime: dict[str, Any] | None,
    macs: int | None,
    power_inputs: dict[str, float] | None,
) -> dict[str, Any]:
    tile = calculate_tile_metrics(tile_h=tile_h, tile_w=tile_w, scale=scale)
    warnings = _build_runtime_warnings(runtime, tile_h=tile_h, tile_w=tile_w, scale=scale)
    inference_ms = _resolve_inference_ms_for_report(runtime)

    compute = calculate_compute_metrics(
        macs=macs,
        inference_ms=inference_ms,
        output_pixels=tile["output_pixels"],
    )

    report: dict[str, Any] = {
        "metadata": {
            "ops_convention": OPS_CONVENTION,
            "warnings": warnings,
        },
        "model": {
            "arch": arch,
            "scale": scale,
            "input_tile": tile["input_tile"],
            "output_tile": tile["output_tile"],
            "output_pixels": tile["output_pixels"],
            "params": params,
        },
        "artifacts": {
            "checkpoint_kb": calculate_file_kb(checkpoint_path),
            "tflite_float32_kb": calculate_file_kb(tflite_float32_path),
            "tflite_int8_kb": calculate_file_kb(tflite_int8_path),
            "c_header_kb": calculate_file_kb(c_header_path),
        },
        "compute": compute,
    }

    if quality is not None:
        report["quality"] = quality
    if runtime is not None:
        report["runtime"] = runtime
    if power_inputs is not None:
        if inference_ms is None:
            raise ValueError("power metrics require runtime inference_ms")
        report["power"] = calculate_power_metrics(
            voltage_v=power_inputs["voltage_v"],
            idle_current_ma=power_inputs["idle_current_ma"],
            inference_current_ma=power_inputs["inference_current_ma"],
            inference_ms=float(inference_ms),
            output_pixels=tile["output_pixels"],
            mops=compute.get("mops"),
            ops=compute.get("ops"),
            psnr_gain=None if quality is None else quality.get("psnr_gain"),
        )

    return report
