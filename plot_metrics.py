import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training metrics.")
    parser.add_argument("--exp_dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    log_path = exp_dir / "training_log.csv"
    df = pd.read_csv(log_path)

    plt.figure(figsize=(8, 5))
    plt.plot(df["Epoch"], df["Train_Loss"], label="Train_Loss")
    plt.plot(df["Epoch"], df["Val_Loss"], label="Val_Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(exp_dir / "loss_curve.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df["Epoch"], df["Val_PSNR"])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val_PSNR")
    axes[0].grid(True)

    axes[1].plot(df["Epoch"], df["Val_SSIM"])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val_SSIM")
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(exp_dir / "quality_curve.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
