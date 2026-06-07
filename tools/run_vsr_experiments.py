from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml


STAGE_CONFIGS: Dict[str, List[str]] = {
    "smoke": [
        "configs/vsr/smoke_video_espcn_x2_3f.yaml",
        "configs/vsr/smoke_vsrbasic_x2_5f.yaml",
    ],
    "pilot": [
        "configs/vsr/pilot_video_espcn_x2_3f.yaml",
        "configs/vsr/pilot_vsrbasic_x2_3f.yaml",
        "configs/vsr/pilot_vsrbasic_x2_5f.yaml",
        "configs/vsr/pilot_vsrbasic_x2_7f.yaml",
    ],
    "ablation": [
        "configs/vsr/ablation_vsrbasic_x2_3f.yaml",
        "configs/vsr/ablation_vsrbasic_x2_5f.yaml",
        "configs/vsr/ablation_vsrbasic_x2_7f.yaml",
        "configs/vsr/ablation_vsrplusplus_x2_5f.yaml",
        "configs/vsr/ablation_vsrplusplus_x2_7f.yaml",
    ],
    "final": [
        "configs/vsr/final_best_vsrbasic_x2.yaml",
        "configs/vsr/final_best_vsrplusplus_x2.yaml",
    ],
}

EVAL_DATASETS = {
    "Vid4": "data/vsr/Vid4",
    "UDM10": "data/vsr/UDM10",
    "SPMCS": "data/vsr/SPMCS",
    "REDS4": "data/vsr/REDS4",
}

MANIFEST_FIELDS = [
    "experiment_id",
    "model",
    "dataset",
    "num_frames",
    "epochs",
    "samples_per_epoch",
    "checkpoint_path",
    "status",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run reproducible VSR experiment stages.")
    p.add_argument("--stage", required=True, choices=["smoke", "pilot", "ablation", "final", "eval-cross-dataset"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--confirm-large-run", action="store_true")
    p.add_argument("--force", action="store_true", help="Run even if a best_model.pth already exists")
    p.add_argument("--manifest", default="reports/vsr_experiment_manifest.csv")
    p.add_argument("--checkpoint", default="", help="Checkpoint for --stage eval-cross-dataset")
    p.add_argument("--arch", default="VSRBasic")
    p.add_argument("--num-frames", type=int, default=5)
    p.add_argument("--scale", type=int, default=2)
    return p.parse_args()


def read_config(path: Path) -> Dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def find_existing_checkpoint(cfg: Dict) -> Path | None:
    run_group = str(cfg.get("run_group", "")).strip()
    base = Path("runs")
    if run_group:
        base = base / run_group
    base = base / str(cfg["model_name"]) / str(cfg.get("dataset_name", "Vimeo90K"))
    checkpoints = sorted(base.glob("exp_*/best_model.pth"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return checkpoints[0] if checkpoints else None


def write_manifest_row(manifest_path: Path, row: Dict[str, str]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    existing: List[Dict[str, str]] = []
    if manifest_path.exists():
        with manifest_path.open("r", newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    existing = [item for item in existing if item.get("experiment_id") != row["experiment_id"]]
    existing.append(row)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(existing)


def manifest_from_config(experiment_id: str, cfg: Dict, checkpoint: str, status: str) -> Dict[str, str]:
    return {
        "experiment_id": experiment_id,
        "model": str(cfg.get("arch", cfg.get("model_name", ""))),
        "dataset": str(cfg.get("dataset_name", "")),
        "num_frames": str(cfg.get("num_frames", "")),
        "epochs": str(cfg.get("epochs", "")),
        "samples_per_epoch": str(cfg.get("samples_per_epoch", "")),
        "checkpoint_path": checkpoint,
        "status": status,
    }


def run_training_stage(args: argparse.Namespace) -> None:
    if args.stage in {"ablation", "final"} and not args.confirm_large_run and not args.dry_run:
        raise SystemExit(f"{args.stage} can be expensive. Re-run with --confirm-large-run.")

    manifest_path = Path(args.manifest)
    for config_name in STAGE_CONFIGS[args.stage]:
        config_path = Path(config_name)
        cfg = read_config(config_path)
        experiment_id = config_path.stem
        existing = find_existing_checkpoint(cfg)
        if existing and not args.force:
            print(f"[skip] {experiment_id}: existing checkpoint {existing}")
            write_manifest_row(manifest_path, manifest_from_config(experiment_id, cfg, str(existing), "completed_existing"))
            continue

        cmd = [sys.executable, "train.py", "--config", str(config_path)]
        print("[run]", " ".join(cmd))
        if args.dry_run:
            write_manifest_row(manifest_path, manifest_from_config(experiment_id, cfg, str(existing or ""), "dry_run"))
            continue

        status = "completed"
        checkpoint = ""
        try:
            subprocess.run(cmd, check=True)
            checkpoint_path = find_existing_checkpoint(cfg)
            checkpoint = str(checkpoint_path) if checkpoint_path else ""
            if not checkpoint:
                status = "completed_no_checkpoint"
        except subprocess.CalledProcessError as exc:
            status = f"failed:{exc.returncode}"
        write_manifest_row(manifest_path, manifest_from_config(experiment_id, cfg, checkpoint, status))
        if status.startswith("failed"):
            raise SystemExit(status)


def latest_checkpoint() -> Path | None:
    checkpoints = sorted(Path("runs").glob("**/best_model.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0] if checkpoints else None


def run_cross_dataset_eval(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint()
    if checkpoint is None:
        raise SystemExit("No checkpoint supplied and no runs/**/best_model.pth found.")

    for dataset_name, root in EVAL_DATASETS.items():
        root_path = Path(root)
        if not root_path.is_dir():
            print(f"[skip] {dataset_name}: missing {root}")
            continue
        out_dir = Path("reports/vsr_eval") / f"{dataset_name.lower()}_{checkpoint.parent.parent.name.lower()}"
        cmd = [
            sys.executable,
            "evaluate_vsr.py",
            "--checkpoint",
            str(checkpoint),
            "--video-root",
            str(root_path),
            "--dataset-name",
            dataset_name,
            "--scale",
            str(args.scale),
            "--num-frames",
            str(args.num_frames),
            "--arch",
            args.arch,
            "--output-dir",
            str(out_dir),
        ]
        print("[eval]", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    if args.stage == "eval-cross-dataset":
        run_cross_dataset_eval(args)
    else:
        run_training_stage(args)


if __name__ == "__main__":
    main()
