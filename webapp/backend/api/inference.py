from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from backend.config import TEMPLATES_DIR
from backend.services import inference_service, run_scanner

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

_MAX_UPLOAD_BYTES = 16 * 1024 * 1024  # 16 MB


def _runs_with_checkpoints():
    summaries = run_scanner.list_runs()
    result = []
    for s in summaries:
        has_best = run_scanner.checkpoint_path(s.id, "best") is not None
        has_last = run_scanner.checkpoint_path(s.id, "last") is not None
        if has_best or has_last:
            result.append({
                "run_id": s.id,
                "model_name": s.model_name,
                "arch": s.arch,
                "scale": s.scale,
                "dataset": s.dataset,
                "best_psnr": s.best_psnr,
                "has_best": has_best,
                "has_last": has_last,
            })
    return result


@router.get("/inference", response_class=HTMLResponse)
def inference_page(request: Request, run_id: str | None = None):
    runs = _runs_with_checkpoints()
    preselect = run_id if any(r["run_id"] == run_id for r in runs) else None
    return templates.TemplateResponse(
        request=request,
        name="inference.html",
        context={
            "runs": runs,
            "preselect": preselect,
            "nav": "inference",
        },
    )


@router.post("/inference/run", response_class=HTMLResponse)
async def run_inference_endpoint(
    request: Request,
    run_id: str = Form(...),
    checkpoint: str = Form("best"),
    upload_kind: str = Form("hr"),
    device: str = Form("auto"),
    amp: str = Form("false"),
    image: UploadFile = File(...),
):
    if checkpoint not in {"best", "last"}:
        raise HTTPException(status_code=400, detail="checkpoint must be 'best' or 'last'")
    if upload_kind not in {"lr", "hr"}:
        raise HTTPException(status_code=400, detail="upload_kind must be 'lr' or 'hr'")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds {_MAX_UPLOAD_BYTES // (1024*1024)} MB limit.")

    try:
        result = inference_service.run_inference(
            run_id=run_id,
            checkpoint_which=checkpoint,
            image_bytes=data,
            upload_kind=upload_kind,
            device_str=device,
            use_amp=amp.lower() in {"true", "on", "1", "yes"},
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc!r}") from exc

    return templates.TemplateResponse(
        request=request,
        name="partials/inference_result.html",
        context={"r": result, "run_id": run_id, "checkpoint": checkpoint},
    )


@router.get("/inference/files/{uid}/{filename}")
def serve_upload(uid: str, filename: str):
    path = inference_service.read_upload_file(uid, filename)
    if path is None:
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)
