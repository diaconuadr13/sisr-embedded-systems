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
├── test_sweep_dryrun.py      # Verification: 3-arch sequential dry-run
├── models/
│   ├── __init__.py           # Model factory (get_model / MODEL_REGISTRY)
│   ├── espcn.py              # ESPCN (Shi et al., 2016)
│   ├── espcn_light.py        # ESPCN-Light (halved channels, embedded target)
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
pip install -r requirements.txt

# 2. Download DIV2K dataset
python download_data.py

# 3. Train a single model
python train.py --arch ESPCN --scale 2 --epochs 100

# 4. Run an architecture comparison sweep
python run_experiments.py configs/sweep_example.yaml

# 5. Plot results
python plot_metrics.py --exp_dir runs/<model_name>/<dataset>/exp_YYYYMMDD_HHMMSS

# 6. Profile inference speed
python evaluate_pc.py --weights runs/.../best_model.pth --val_dir data/val/DIV2K_valid_HR --scale 2
```

## Supported Architectures

| Architecture | Params (×2) | Description |
|---|---|---|
| `ESPCN` | 26,796 | Sub-pixel convolution (Shi et al., CVPR 2016) |
| `ESPCN_Light` | 8,796 | Halved channel widths for edge deployment |
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

## Evaluation Metrics

- **PSNR** – Peak Signal-to-Noise Ratio (dB)
- **SSIM** – Structural Similarity Index
- **Inference time** (ms/image) and **FPS**
- **Parameter count** and memory footprint
