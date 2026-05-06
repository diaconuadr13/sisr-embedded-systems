from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.config import RUNS_DIR, STATIC_DIR, WEBAPP_DATA_DIR
from backend.api import architectures, benchmark, compare, deployment, experiments, export, inference, jobs, notes, runs, sweeps
from backend.services import job_manager, storage


def create_app() -> FastAPI:
    WEBAPP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    # Warm up SQLite schema
    with storage.connection() as _:
        pass

    # Start the background job worker (reset crashed runs, spawn worker thread)
    job_manager.start_worker()

    app = FastAPI(title="SISR Thesis Manager", docs_url="/api/docs", redoc_url=None)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse("/dashboard", status_code=307)

    app.include_router(runs.router)
    app.include_router(compare.router)
    app.include_router(deployment.router)
    app.include_router(architectures.router)
    app.include_router(notes.router)
    app.include_router(experiments.router)
    app.include_router(sweeps.router)
    app.include_router(jobs.router)
    app.include_router(inference.router)
    app.include_router(benchmark.router)
    app.include_router(export.router)

    return app


app = create_app()
