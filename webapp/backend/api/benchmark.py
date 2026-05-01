from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from backend.config import TEMPLATES_DIR
from backend.services import benchmark_service, run_scanner

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _runs_with_checkpoints():
    summaries = run_scanner.list_runs()
    rows = []
    for s in summaries:
        has_best = run_scanner.checkpoint_path(s.id, "best") is not None
        has_last = run_scanner.checkpoint_path(s.id, "last") is not None
        if has_best or has_last:
            rows.append({
                "run_id": s.id,
                "model_name": s.model_name,
                "arch": s.arch,
                "scale": s.scale,
                "has_best": has_best,
                "has_last": has_last,
            })
    return rows


@router.get("/benchmark", response_class=HTMLResponse)
def benchmark_page(request: Request):
    cached = benchmark_service.list_all()
    runs = _runs_with_checkpoints()
    return templates.TemplateResponse(
        request=request,
        name="benchmark.html",
        context={
            "cached": cached,
            "runs": runs,
            "lr_hw": (benchmark_service.BENCH_LR_HEIGHT, benchmark_service.BENCH_LR_WIDTH),
            "warmup": benchmark_service.BENCH_WARMUP_ITERS,
            "measure": benchmark_service.BENCH_MEASURE_ITERS,
            "nav": "benchmark",
        },
    )


@router.post("/benchmark/run")
def run_benchmark_endpoint(
    run_id: str = Form(...),
    checkpoint: str = Form("best"),
    device: str = Form("auto"),
    amp: str = Form("false"),
    force: str = Form("false"),
):
    if checkpoint not in {"best", "last"}:
        raise HTTPException(status_code=400, detail="checkpoint must be 'best' or 'last'")
    try:
        benchmark_service.run_benchmark(
            run_id=run_id,
            checkpoint_which=checkpoint,
            device_str=device,
            amp=amp.lower() in {"true", "on", "1", "yes"},
            force=force.lower() in {"true", "on", "1", "yes"},
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {exc!r}") from exc
    return RedirectResponse("/benchmark", status_code=303)
