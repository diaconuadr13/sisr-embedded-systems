# VSR Experiments

This project separates VSR model code from reproducible experiment infrastructure: dataset download/normalization, staged training configs, cross-dataset evaluation, manifests, and report artifacts.

## Datasets

Small evaluation sets:

- Vid4: classic VSR benchmark with 4 short clips.
- UDM10: 10 HD clips, useful for checking behavior on larger frames.
- SPMCS: 30 clips with 31 frames each, useful for fine-detail evaluation.

Large/optional:

- REDS4: four REDS validation clips commonly used by BasicSR-style VSR work. REDS is large, so it is not included in `all-small`.
- Vimeo90K: training/validation dataset. Use the existing `download_data.py` path instead of duplicating the downloader.

Normalized evaluation layout:

```text
data/vsr/Vid4/calendar/frame_000.png
data/vsr/Vid4/calendar/frame_001.png
data/vsr/UDM10/000/frame_000.png
```

Download commands:

```bash
python tools/download_vsr_datasets.py --dataset vid4 --root data/vsr
python tools/download_vsr_datasets.py --dataset udm10 --root data/vsr
python tools/download_vsr_datasets.py --dataset spmcs --root data/vsr
python tools/download_vsr_datasets.py --dataset all-small --root data/vsr
python tools/download_vsr_datasets.py --dataset reds4 --root data/vsr --confirm-large-download
```

Tested source status on 2026-05-26:

- Vid4: Google Drive works through the built-in Drive fallback; the PaddleGAN mirror also worked in this environment.
- UDM10: the PaddleGAN mirror returned 404, but Google Drive worked through the built-in Drive fallback.
- SPMCS: the official TinyURL resolves to the original archive host, but that host returned 403 from this environment. Use manual download if you have access to a working mirror.
- REDS4: intentionally not auto-started without `--confirm-large-download`; Google Drive may hit quota, so the downloader also tries the official Hugging Face `train_sharp.zip` mirror and normalizes clips `000`, `011`, `015`, `020`.

If a Google Drive or TinyURL link expires, download manually from the cited project page, extract it, then normalize:

```bash
python tools/download_vsr_datasets.py --dataset spmcs --root data/vsr --source-dir /path/to/SPMCS
python tools/download_vsr_datasets.py --dataset reds4 --root data/vsr --source-dir /path/to/REDS/train_sharp
```

Vimeo90K training data:

```bash
python download_data.py --vimeo90k_kaggle --vimeo90k_raw_root data/raw/vimeo90k_kaggle
```

## Stages

Run in this order:

```bash
python tools/run_vsr_experiments.py --stage smoke
python tools/run_vsr_experiments.py --stage pilot
python tools/run_vsr_experiments.py --stage ablation --confirm-large-run
python tools/run_vsr_experiments.py --stage final --confirm-large-run
python tools/run_vsr_experiments.py --stage eval-cross-dataset --checkpoint runs/.../best_model.pth --arch VSRBasic --num-frames 5
```

Use `--dry-run` before starting long jobs:

```bash
python tools/run_vsr_experiments.py --stage ablation --dry-run
```

Stage definitions:

- smoke: 1 epoch, 128 train samples, 32 val samples. Pipeline sanity check.
- pilot: 5 epochs, 2000 train samples, 200 val samples. Fast model/frame-count comparison.
- ablation: 25 epochs, 8000 train samples, 500 val samples. VSRBasic 3/5/7 frames and VSRPlusPlus 5/7 frames.
- final: 50 epochs, 16000 train samples, 1000 val samples. Only the selected VSRBasic/VSRPlusPlus candidates.

Do not run 100 epochs for every model by default. VSR experiments multiply cost by architecture, frame count, resolution, and evaluation dataset. The staged budgets catch broken runs early, identify weak candidates cheaply, and reserve GPU time for the most promising configurations.

## Evaluation

Example:

```bash
python evaluate_vsr.py \
  --checkpoint runs/.../best_model.pth \
  --video-root data/vsr/Vid4 \
  --dataset-name Vid4 \
  --scale 2 \
  --num-frames 5 \
  --arch VSRBasic \
  --output-dir reports/vsr_eval/vid4_vsrbasic
```

Outputs:

- `reports/vsr_eval/<run>/summary.json`
- `reports/vsr_eval/<run>/metrics.csv`
- `reports/vsr_eval/<run>/samples/*.png`

Tracked metrics:

- PSNR
- SSIM
- mean inference time per frame
- estimated FPS
- trainable parameters
- estimated MACs/FLOPs from convolution/linear layers
- temporal consistency error

## Dissertation Exports

Export these artifacts when runs are complete:

- `reports/vsr_experiment_manifest.csv`: experiment inventory and checkpoint paths.
- `reports/vsr_eval/*/summary.json`: table-ready aggregate metrics.
- `reports/vsr_eval/*/metrics.csv`: per-clip/per-frame data for plots.
- `reports/vsr_eval/*/samples/*.png`: LR/SR/HR visual comparisons.
- `runs/vsr/*/*/training_log.csv`: learning curves for selected runs.
