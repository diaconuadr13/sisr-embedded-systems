"""Inference speed benchmarking — mirrors evaluate_pc.py.

Runs a fixed-size synthetic LR tensor through the checkpoint's model for a
configurable number of warmup + measurement iterations, and caches the
result per (run_id, checkpoint, device, amp) in SQLite.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services import run_scanner, storage


# Fixed input resolution (LR). The model is fully convolutional so output is
# deterministic at scale * (H, W). 180x180 is a common pattern-matching size
# close to evaluate_pc's patch_size while staying fast on CPU too.
BENCH_LR_HEIGHT = 180
BENCH_LR_WIDTH = 180
BENCH_WARMUP_ITERS = 5
BENCH_MEASURE_ITERS = 50


@dataclass
class BenchmarkResult:
    run_id: str
    checkpoint: str
    device: str
    amp: bool
    avg_ms: float
    fps: float
    params: int
    created_at: str


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_torch():
    import torch

    return torch


def _count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def run_benchmark(
    run_id: str,
    checkpoint_which: str = "best",
    device_str: str = "auto",
    amp: bool = False,
    force: bool = False,
) -> BenchmarkResult:
    """Profile inference speed. Caches results per (run, ckpt, device, amp)."""
    from backend.services import inference_service  # defers torch import

    ckpt_path = run_scanner.checkpoint_path(run_id, checkpoint_which)
    if ckpt_path is None:
        raise FileNotFoundError(f"Checkpoint '{checkpoint_which}' not found for run '{run_id}'.")

    # Cache hit
    if not force:
        existing = _get_cached(run_id, checkpoint_which, device_str, amp)
        if existing is not None:
            return existing

    torch = _load_torch()
    model, arch, scale, device = inference_service._load_model(ckpt_path, device_str)  # noqa: SLF001
    amp_active = amp and device.type == "cuda"

    params = _count_params(model)

    lr = torch.rand(1, 3, BENCH_LR_HEIGHT, BENCH_LR_WIDTH, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(BENCH_WARMUP_ITERS):
            if amp_active:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = model(lr)
            else:
                _ = model(lr)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # Measure
    total = 0.0
    with torch.no_grad():
        for _ in range(BENCH_MEASURE_ITERS):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            if amp_active:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = model(lr)
            else:
                _ = model(lr)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total += time.perf_counter() - t0

    avg_ms = (total / BENCH_MEASURE_ITERS) * 1000.0
    fps = (1.0 / (total / BENCH_MEASURE_ITERS)) if total > 0 else float("inf")

    result = BenchmarkResult(
        run_id=run_id,
        checkpoint=checkpoint_which,
        device=str(device),
        amp=amp_active,
        avg_ms=float(avg_ms),
        fps=float(fps),
        params=params,
        created_at=_iso_now(),
    )
    _store(result)
    return result


def _store(r: BenchmarkResult) -> None:
    with storage.connection() as conn:
        conn.execute(
            "REPLACE INTO benchmark_cache(run_id, checkpoint, device, amp, avg_ms, fps, params, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (r.run_id, r.checkpoint, r.device, int(r.amp), r.avg_ms, r.fps, r.params, r.created_at),
        )


def _row_to_result(row) -> BenchmarkResult:
    return BenchmarkResult(
        run_id=row["run_id"],
        checkpoint=row["checkpoint"],
        device=row["device"],
        amp=bool(row["amp"]),
        avg_ms=float(row["avg_ms"]),
        fps=float(row["fps"]),
        params=int(row["params"]),
        created_at=row["created_at"],
    )


def _get_cached(run_id: str, checkpoint: str, device_str: str, amp: bool) -> BenchmarkResult | None:
    # Device can be "auto" at request-time. Match the canonical torch device string
    # (e.g. "cuda:0") by first resolving it, since that's what's stored.
    from utils.device import resolve_device

    try:
        canonical = str(resolve_device(device_str))
    except RuntimeError:
        canonical = device_str
    with storage.connection() as conn:
        row = conn.execute(
            "SELECT * FROM benchmark_cache WHERE run_id=? AND checkpoint=? AND device=? AND amp=?",
            (run_id, checkpoint, canonical, int(amp and canonical.startswith("cuda"))),
        ).fetchone()
    return _row_to_result(row) if row else None


def list_all() -> list[BenchmarkResult]:
    with storage.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM benchmark_cache ORDER BY created_at DESC"
        ).fetchall()
    return [_row_to_result(r) for r in rows]


def list_for_run(run_id: str) -> list[BenchmarkResult]:
    with storage.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM benchmark_cache WHERE run_id=? ORDER BY created_at DESC",
            (run_id,),
        ).fetchall()
    return [_row_to_result(r) for r in rows]
