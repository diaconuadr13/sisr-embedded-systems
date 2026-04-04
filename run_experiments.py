"""
Orchestrator for sequential SISR experiments.
Reads a YAML sweep file and runs each config through train.train().
Errors in one experiment are logged but never kill the batch.
"""
import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

from train import DEFAULT_CONFIG, cleanup_vram, train


def load_sweep(path: str) -> List[Dict[str, Any]]:
    """Load a YAML sweep file → list of per-experiment config overrides."""
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if isinstance(data, dict) and "experiments" in data:
        return data["experiments"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Sweep file must contain a list or {{'experiments': [...]}}.")


def merge_config(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Defaults + per-experiment overrides."""
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(overrides)
    return cfg


def run_sweep(sweep_path: str, log_dir: str = "sweep_logs") -> None:
    experiments = load_sweep(sweep_path)
    log_root = Path(log_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_log = log_root / f"batch_{ts}.json"

    results: List[Dict[str, Any]] = []
    total = len(experiments)
    print(f"[sweep] {total} experiment(s) queued from {sweep_path}\n")

    for i, overrides in enumerate(experiments, 1):
        cfg = merge_config(overrides)
        tag = f"[{i}/{total}] {cfg['model_name']} scale={cfg['scale']} lr={cfg['lr']}"
        print(f"\n{'='*60}\n{tag}\n{'='*60}")

        entry: Dict[str, Any] = {"index": i, "config": cfg, "status": "pending"}
        try:
            exp_dir = train(cfg)
            entry["status"] = "success"
            entry["exp_dir"] = str(exp_dir)
            print(f"[sweep] ✓ {tag} → {exp_dir}")
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = str(exc)
            entry["traceback"] = traceback.format_exc()
            print(f"[sweep] ✗ {tag} → {exc}", file=sys.stderr)
        finally:
            # Guarantee VRAM is clean before next experiment
            cleanup_vram()

        results.append(entry)
        # Incremental write so partial results survive a hard crash
        batch_log.write_text(json.dumps(results, indent=2), encoding="utf-8")

    passed = sum(1 for r in results if r["status"] == "success")
    failed = total - passed
    print(f"\n[sweep] Done: {passed}/{total} succeeded, {failed} failed.")
    print(f"[sweep] Batch log: {batch_log}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run a sweep of SISR experiments.")
    p.add_argument("sweep", type=str, help="Path to YAML sweep configuration file.")
    p.add_argument("--log_dir", type=str, default="sweep_logs")
    args = p.parse_args()
    run_sweep(args.sweep, args.log_dir)


if __name__ == "__main__":
    main()
