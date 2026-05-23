# Embedded SISR – Single Image Super-Resolution for Embedded Systems

**Master's thesis** – Analysis of image super-resolution algorithms for embedded systems.

Lightweight CNN architectures (ESPCN, FSRCNN, ESPCN-Light) trained and benchmarked for on-device image upscaling, targeting quality–latency trade-offs on resource-constrained hardware.

## Project Structure

```
├── train.py                 # Training entrypoint (single run or config file)
├── run_experiments.py        # Sweep orchestrator (YAML-driven, fault-tolerant)
├── evaluate_pc.py            # Inference speed profiling
├── plot_metrics.py           # Loss / PSNR / SSIM curve plotting
├── download_data.py          # DIV2K dataset downloader
├── models/
│   ├── __init__.py           # Model factory (get_model / MODEL_REGISTRY)
│   ├── espcn.py              # ESPCN (Shi et al., 2016)
│   ├── espcn_light.py        # ESPCN-Light (halved channels, embedded target)
│   ├── espcn_micro.py        # ESPCN-Micro (microcontroller grayscale target)
│   └── fsrcnn.py             # FSRCNN (Dong et al., 2016)
├── utils/
│   ├── dataset.py            # SISRDataset (HR patch extraction + LR downsampling)
│   └── metrics.py            # PSNR / SSIM computation
└── configs/
    └── sweep_example.yaml    # Example architecture comparison sweep
```

## Quick Start

```bash
# 1. Install dependencies
# Install a CUDA-enabled PyTorch build if you want GPU training/inference.
pip install -r requirements.txt

# 2. Download DIV2K dataset
python download_data.py

# 3. Train a single model
python train.py --arch ESPCN --scale 2 --epochs 100 --device cuda --amp true

# 4. Run an architecture comparison sweep
python run_experiments.py configs/sweep_example.yaml

# 5. Plot results
python plot_metrics.py --exp_dir runs/<model_name>/<dataset>/exp_YYYYMMDD_HHMMSS

# 6. Profile inference speed
python evaluate_pc.py --weights runs/.../best_model.pth --val_dir data/val/DIV2K_valid_HR --device cuda --amp

# Smallest grayscale microcontroller-targeted run
python train.py --config configs/train_espcn_micro_gray_x2.yaml

# Native 32x24 infrared thermal run: LR 16x12 -> HR 32x24 grayscale
python train.py --config configs/train_espcn_micro_thermal_gray_x2.yaml

# Train all supported models on the native thermal dataset
python run_experiments.py configs/sweep_thermal_gray_x2.yaml

# Compare completed thermal runs by validation PSNR/SSIM
python tools/compare_training_runs.py --dataset InfraredThermal32x24 \
  --output reports/thermal_gray_x2_comparison.csv
```

## Supported Architectures

| Architecture | Params (×2) | Description |
|---|---|---|
| `ESPCN` | 26,796 | Sub-pixel convolution (Shi et al., CVPR 2016) |
| `ESPCN_Light` | 8,796 | Halved channel widths and ReLU activations for edge deployment |
| `ESPCN_Micro` | <2,000 | Tiny grayscale sub-pixel CNN for microcontroller-class tests |
| `FSRCNN` | 24,683 | Deconvolution-based (Dong et al., ECCV 2016) |

## Sweep Configuration

Experiments are defined in YAML. Each entry overrides default config values:

```yaml
experiments:
  - model_name: "ESPCN_x2"
    arch: "ESPCN"
    scale: 2
    lr: 0.001
    epochs: 100

  - model_name: "FSRCNN_x2"
    arch: "FSRCNN"
    scale: 2
    lr: 0.001
    epochs: 100
```

The orchestrator runs experiments sequentially with full error isolation and VRAM cleanup between runs.

GPU settings can also be placed directly in a sweep entry:

```yaml
experiments:
  - model_name: "ESPCN_x2_cuda"
    arch: "ESPCN"
    scale: 2
    device: "cuda"
    amp: true
```

## Evaluation Metrics

- **PSNR** – Peak Signal-to-Noise Ratio (dB)
- **SSIM** – Structural Similarity Index
- **Inference time** (ms/image) and **FPS**
- **Parameter count** and memory footprint


## PC-only Video Super-Resolution

This workflow trains, evaluates, and runs lightweight PyTorch video SR on a computer. Embedded boards, TFLite export, deployment metrics, and power metrics are intentionally out of scope for this phase.

Dataset layout:

```text
data/video/train/clip_000/frame_000.png
data/video/train/clip_000/frame_001.png
data/video/train/clip_001/frame_000.png
data/video/val/clip_000/frame_000.png
```

A flat directory of frames is also supported and is treated as one clip.

Create a toy dataset for smoke tests:

```bash
python tools/create_toy_video_dataset.py --output data/video_toy
```

Train a small three-frame VideoESPCN model:

```bash
python train_video.py \
  --config configs/train_video_espcn_x2_3f.yaml \
  --hr_video_dir data/video_toy/train \
  --val_video_dir data/video_toy/val \
  --epochs 2 \
  --samples_per_epoch 64 \
  --batch_size 4 \
  --device auto
```

Evaluate a trained checkpoint:

```bash
python evaluate_video_pc.py \
  --weights runs/.../best_model.pth \
  --video_dir data/video_toy/val \
  --device auto \
  --output_json runs/.../video_eval.json
```

Run inference on a video file:

```bash
python video_infer.py \
  --weights runs/.../best_model.pth \
  --input input.mp4 \
  --output output_sr.mp4 \
  --device auto
```

Run the frame-by-frame baseline mode. For video checkpoints this repeats the center frame across the temporal window, so the model gets no neighboring-frame information:

```bash
python video_infer.py \
  --weights runs/.../best_model.pth \
  --input input.mp4 \
  --output output_sisr_per_frame.mp4 \
  --frame_by_frame
```

Run the pytest smoke tests:

```bash
python -m pytest tests
```

Known limitations:

- This first VSR model does not use optical flow or deformable alignment.
- Temporal windows are concatenated as channels.
- It may improve stability/details, but can still fail on large motion.
- Board/TFLite deployment is intentionally out of scope for this phase.

## Deployment Metrics

Deployment reports combine checkpoint metadata, tile dimensions, optional validation-patch quality metrics, optional ESP32 runtime logs, and optional power measurements.

```bash
python tools/collect_deployment_metrics.py \
  --checkpoint runs/ESPCN_Light_gray_x2/Flickr2K/exp_YYYYMMDD_HHMMSS/best_model.pth \
  --tile 8 8 \
  --val-dir data/val/DIV2K_valid_HR \
  --board-log esp32_benchmark.log \
  --voltage 3.3 \
  --idle-current-ma 48 \
  --inference-current-ma 92 \
  --profile-macs \
  --output runs/.../deployment_metrics_8x8.json
```

The board log can stay simple:

```text
target=esp32
tile=8x8
scale=2
inference_ms=142.7
free_heap_before=180224
free_heap_after=96320
sample_ms=141.9
sample_ms=142.8
sample_ms=143.1
```

Raspberry Pi Pico/RP2040 sketches are available under `pico_firmware/` for
`EDSR_Tiny` INT8 and `SRCNN` INT8. Their serial output uses the same board-log
format with `target=pico` and Pico SRAM fields:

```text
target=pico
tile=8x8
scale=2
tensor_arena_bytes=49152
free_sram_before=180224
free_sram_after=130560
inference_ms=42.5
sample_ms=42.0
sample_ms=43.0
```

Power-derived fields such as energy per inference, energy per output pixel, and MOPS/W are calculated on the PC after measurement.

For native thermal x2 deployment/export, use `--tile 12 16`; the tile arguments
are LR height/width, so the model runs `12x16 -> 24x32`.
