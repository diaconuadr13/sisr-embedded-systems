# SISR Thesis Manager - Web App

A FastAPI + HTMX web app for browsing, comparing, launching, and demoing experiments from the `sisr-embedded-systems` project. The app reads the existing `runs/` tree as source of truth, stores app-only state in SQLite, and keeps the training code untouched.

## Quick start

From the `webapp/` directory:

```bash
source ../.disertatie/bin/activate
pip install -r requirements.txt
./run.sh
```

The server binds to `127.0.0.1:8000` by default. You can override the runtime settings with environment variables:

- `HOST` - default `127.0.0.1`
- `PORT` - default `8000`
- `RELOAD` - default `1`; set to `0` to disable auto reload

## Feature map

### Browse and compare

- **Dashboard** - `/dashboard`: total run count, favorites pinned by the `favorite` tag, best run per architecture, and recent runs.
- **Runs browser** - `/runs`: filter by arch, dataset, scale, tag, and search text, then multi-select runs for comparison.
- **Run detail** - `/runs/{arch}/{dataset}/{exp}`: config, metrics curves, sample gallery, checkpoint links, notes, and tags.
- **Compare** - `/compare?runs=...`: overlaid training curves and summary metrics for selected runs.
- **LaTeX export** - `/export/latex?runs=...`: downloads a zip containing `table.tex` and `curves.pdf` for thesis use.
- **Architectures** - `/architectures`: parameter counts and FLOPs for the registered models at scales x2, x3, and x4.

### Training control

- **New experiment** - `/experiments/new`: launches a single training job through `train.py`.
- **Sweeps** - `/sweeps/new`: uploads or composes a sweep config and launches `run_experiments.py`.
- **Jobs** - `/jobs` and `/jobs/{id}`: queued, running, and finished jobs with polling, live progress, and cancel support.

### Inference and benchmarking

- **Inference** - `/inference`: upload an image, pick a checkpoint, and compare LR, bicubic, SR, and optional HR output.
- **Benchmark** - `/benchmark`: runs FPS/latency profiling and caches results in SQLite.

## Data sources

- `runs/` - read-only experiment source of truth.
- `webapp/data/webapp.db` - SQLite for notes, tags, jobs, and benchmark cache.
- `webapp/data/uploads/` - temporary inference uploads and generated files.
- `webapp/data/job_logs/` - stdout/stderr captures for launched jobs.

## Repository layout

```text
webapp/
├── backend/        FastAPI app, route modules, services, and SQLite schema
├── templates/      Jinja2 pages and HTMX fragments
├── static/         CSS and small client-side helpers
├── data/           SQLite DB, uploads, and job logs
├── requirements.txt
├── run.sh
└── README.md
```

## Notes

- The dashboard favorites section is driven by the `favorite` tag.
- The export zip is designed to be `pdflatex` friendly: `table.tex` is `booktabs`-based and `curves.pdf` is vector output.
- The app is intended for local use only; it binds to localhost and does not implement auth.
