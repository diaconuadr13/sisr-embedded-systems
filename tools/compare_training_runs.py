#!/usr/bin/env python3
"""Summarize completed training runs for one dataset.

Example:
    python tools/compare_training_runs.py --dataset InfraredThermal32x24
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SISR training runs by validation metrics.")
    parser.add_argument("--runs-root", default="runs", help="Root directory containing training runs.")
    parser.add_argument("--dataset", default="InfraredThermal32x24", help="Dataset name under runs/<model>/<dataset>.")
    parser.add_argument("--output", default=None, help="Optional CSV output path for the summary table.")
    return parser.parse_args()


def read_config(exp_dir: Path) -> Dict[str, Any]:
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def to_float(row: Dict[str, str], key: str) -> float:
    return float(row.get(key, "nan"))


def summarize_run(exp_dir: Path) -> Optional[Dict[str, Any]]:
    log_path = exp_dir / "training_log.csv"
    if not log_path.exists():
        return None

    with log_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    config = read_config(exp_dir)
    best_psnr = max(rows, key=lambda row: to_float(row, "Val_PSNR"))
    best_ssim = max(rows, key=lambda row: to_float(row, "Val_SSIM"))
    final = rows[-1]

    return {
        "model_name": config.get("model_name") or final.get("Model") or exp_dir.parents[1].name,
        "arch": config.get("arch", ""),
        "dataset": config.get("dataset_name") or final.get("Dataset", ""),
        "exp_dir": str(exp_dir),
        "epochs": int(float(final["Epoch"])),
        "best_psnr_epoch": int(float(best_psnr["Epoch"])),
        "best_psnr": to_float(best_psnr, "Val_PSNR"),
        "ssim_at_best_psnr": to_float(best_psnr, "Val_SSIM"),
        "best_ssim_epoch": int(float(best_ssim["Epoch"])),
        "best_ssim": to_float(best_ssim, "Val_SSIM"),
        "final_psnr": to_float(final, "Val_PSNR"),
        "final_ssim": to_float(final, "Val_SSIM"),
    }


def find_runs(runs_root: Path, dataset: str) -> List[Path]:
    if not runs_root.exists():
        return []
    return sorted(path.parent for path in runs_root.glob(f"*/{dataset}/exp_*/training_log.csv"))


def print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No completed runs found.")
        return

    columns = [
        ("model_name", "model", 30),
        ("arch", "arch", 12),
        ("best_psnr", "best_psnr", 10),
        ("ssim_at_best_psnr", "ssim@best", 10),
        ("best_psnr_epoch", "epoch", 5),
        ("final_psnr", "final_psnr", 10),
        ("final_ssim", "final_ssim", 10),
    ]

    header = "  ".join(label.ljust(width) for _key, label, width in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        values = []
        for key, _label, width in columns:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.4f}".ljust(width))
            else:
                values.append(str(value).ljust(width))
        print("  ".join(values))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "arch",
        "dataset",
        "epochs",
        "best_psnr_epoch",
        "best_psnr",
        "ssim_at_best_psnr",
        "best_ssim_epoch",
        "best_ssim",
        "final_psnr",
        "final_ssim",
        "exp_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = [
        summary
        for exp_dir in find_runs(Path(args.runs_root), args.dataset)
        if (summary := summarize_run(exp_dir)) is not None
    ]
    rows.sort(key=lambda row: row["best_psnr"], reverse=True)
    print_table(rows)
    if args.output is not None:
        write_csv(Path(args.output), rows)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
