"""Aggregate PC VSR benchmark + cross-dataset summaries into one report."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RV = ROOT / "reports" / "real_video"
VE = ROOT / "reports" / "vsr_eval"


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        return None


def table1():
    # (label, summary file) for slide-29 ms/cadru PC column
    rows = [
        ("VideoESPCN 3f", "DJI_videoespcn_x2_3f_raw.summary.json"),
        ("VSRBasic 3f", "DJI_vsrbasic_x2_3f_raw.summary.json"),
        ("VSRBasic 7f", "DJI_vsrbasic_x2_7f_raw.summary.json"),
        ("VSRPlusPlus 7f", "DJI_vsrplusplus_x2_7f_raw.summary.json"),
    ]
    print("\n## Tabel slide 29 - inferenta video PC (ms/cadru, x2 640x512->1280x1024)\n")
    print("| Model | Frames | PC ms/cadru | PC FPS |")
    print("|---|---|---|---|")
    for label, fn in rows:
        s = load(RV / fn)
        if s is None:
            print(f"| {label} | - | (lipseste) | - |")
            continue
        print(f"| {label} | {s['num_frames']} | {s['avg_inference_ms_per_frame']:.2f} | {s['effective_wall_fps']:.2f} |")


def table2():
    models = ["videoespcn_3f", "vsrbasic_3f", "vsrbasic_7f", "vsrplusplus_7f"]
    datasets = ["vid4", "udm10", "reds4"]
    print("\n## Tabela 3 - generalizare cross-dataset (PC)\n")
    print("| Model | Dataset | Cadre | PSNR [dB] | SSIM | FPS (PC) |")
    print("|---|---|---|---|---|---|")
    for mdl in models:
        for ds in datasets:
            s = load(VE / f"{ds}_{mdl}" / "summary.json")
            if s is None:
                continue
            print(f"| {mdl} | {s['dataset']} | {s['frames_evaluated']} | "
                  f"{s['psnr']:.3f} | {s['ssim']:.3f} | {s['estimated_fps']:.2f} |")


if __name__ == "__main__":
    table1()
    table2()
