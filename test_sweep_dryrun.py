"""
Dry-run: 3 sequential 1-epoch experiments, one per architecture.
Asserts: unique dirs, artifacts present, configs isolated, VRAM clean, no crashes.
"""
import json
import shutil
import sys
from pathlib import Path

import torch

from train import DEFAULT_CONFIG, cleanup_vram, train


def vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def run_dryrun() -> None:
    configs = [
        {**DEFAULT_CONFIG, "model_name": "dryrun_ESPCN",       "arch": "ESPCN",       "scale": 2, "lr": 1e-3, "epochs": 1},
        {**DEFAULT_CONFIG, "model_name": "dryrun_ESPCN_Light",  "arch": "ESPCN_Light", "scale": 2, "lr": 1e-3, "epochs": 1},
        {**DEFAULT_CONFIG, "model_name": "dryrun_FSRCNN",       "arch": "FSRCNN",      "scale": 2, "lr": 1e-3, "epochs": 1},
    ]

    exp_dirs: list[Path] = []
    vram_baseline = vram_mb()
    errors: list[str] = []

    for i, cfg in enumerate(configs, 1):
        print(f"\n--- Dry-run {i}/3: {cfg['arch']} ({cfg['model_name']}) ---")
        try:
            exp_dir = train(cfg)
            exp_dirs.append(exp_dir)
        except Exception as exc:
            errors.append(f"Experiment {i} ({cfg['arch']}) crashed: {exc}")
            cleanup_vram()

    # --- Assertions ---
    print("\n=== Verification ===")
    ok = True

    # 1. Unique directories
    if len(exp_dirs) != 3:
        print(f"FAIL: expected 3 exp_dirs, got {len(exp_dirs)}")
        ok = False
    elif len(set(exp_dirs)) != 3:
        print(f"FAIL: duplicate exp_dirs: {exp_dirs}")
        ok = False
    else:
        print("PASS: 3 unique experiment directories.")

    # 2. Required artifacts
    for d in exp_dirs:
        for artifact in ["config.json", "training_log.csv", "best_model.pth"]:
            if not (d / artifact).exists():
                print(f"FAIL: missing {artifact} in {d}")
                ok = False
    if ok:
        print("PASS: all artifacts present.")

    # 3. Config isolation — each config.json has distinct arch + model_name
    archs_on_disk = []
    for d in exp_dirs:
        c = json.loads((d / "config.json").read_text(encoding="utf-8"))
        archs_on_disk.append(c["arch"])
    if len(set(archs_on_disk)) != 3:
        print(f"FAIL: arch collision: {archs_on_disk}")
        ok = False
    else:
        print(f"PASS: configs isolated (archs={archs_on_disk}).")

    # 4. VRAM released
    leaked = vram_mb() - vram_baseline
    if leaked > 5.0:
        print(f"FAIL: VRAM leak {leaked:.1f} MB.")
        ok = False
    else:
        print(f"PASS: VRAM clean (delta={leaked:.1f} MB).")

    # 5. No errors
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        ok = False
    else:
        print("PASS: no crashes.")

    # Cleanup
    for d in exp_dirs:
        shutil.rmtree(d, ignore_errors=True)
    for name in ["dryrun_ESPCN", "dryrun_ESPCN_Light", "dryrun_FSRCNN"]:
        p = Path("runs") / name
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)

    print(f"\n{'='*40}")
    print(f"OVERALL: {'ALL PASSED' if ok else 'FAILURES DETECTED'}")
    print(f"{'='*40}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    run_dryrun()
