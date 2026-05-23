import argparse
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable


FLIR_IISR_URL = "https://huggingface.co/datasets/bigmamu6/FLIR-IISR/resolve/main/FLIR-IISR.zip?download=true"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FLIR-IISR paired LR/HR data from Hugging Face.")
    parser.add_argument("--raw_dir", type=str, default="data/raw/flir_iisr")
    parser.add_argument("--output", type=str, default="data/flir_iisr")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--lr_dir", type=str, default=None, help="Optional extracted LR directory override.")
    parser.add_argument("--hr_dir", type=str, default=None, help="Optional extracted HR directory override.")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("symlink", "copy", "hardlink"), default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def download_with_progress(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"[flir] Found existing archive: {destination}")
        return

    def report(block_count: int, block_size: int, total_size: int) -> None:
        downloaded = block_count * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100.0, 100.0)
            print(f"\r[flir] Downloading {downloaded / 1024**2:.1f}/{total_size / 1024**2:.1f} MiB ({pct:.1f}%)", end="", flush=True)
        else:
            print(f"\r[flir] Downloading {downloaded / 1024**2:.1f} MiB", end="", flush=True)

    print(f"[flir] Downloading FLIR-IISR.zip -> {destination}")
    urllib.request.urlretrieve(url, destination, reporthook=report)
    print()


def extract_zip(archive: Path, raw_dir: Path) -> None:
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")
    marker = raw_dir / ".flir_iisr_extracted"
    if marker.exists():
        print(f"[flir] Found extraction marker: {marker}")
        return
    print(f"[flir] Extracting {archive} into {raw_dir}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_dir)
    marker.write_text("ok\n", encoding="utf-8")


def image_count(directory: Path) -> int:
    return sum(1 for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def discover_pair_dirs(raw_dir: Path, lr_override: str | None, hr_override: str | None) -> tuple[Path, Path]:
    if lr_override and hr_override:
        return Path(lr_override), Path(hr_override)

    dirs = [p for p in raw_dir.rglob("*") if p.is_dir()]
    scored = []
    for directory in dirs:
        count = image_count(directory)
        if count == 0:
            continue
        name = directory.as_posix().lower()
        lr_score = count + (5000 if any(token in name for token in ("/lr", "low", "lq", "degraded")) else 0)
        if "lr_4x" in name or "x4" in name or "4x" in name:
            lr_score += 10000
        hr_score = count + (5000 if any(token in name for token in ("/hr", "high", "gt", "target", "ground")) else 0)
        scored.append((directory, count, lr_score, hr_score))

    if not scored:
        raise FileNotFoundError(f"No image directories found under {raw_dir}")

    lr_dir = Path(lr_override) if lr_override else max(scored, key=lambda item: item[2])[0]
    if hr_override:
        hr_dir = Path(hr_override)
    else:
        hr_item = max((item for item in scored if item[0] != lr_dir), key=lambda item: item[3], default=None)
        if hr_item is None:
            raise FileNotFoundError("Could not infer an HR directory. Pass --hr_dir explicitly.")
        hr_dir = hr_item[0]

    print("[flir] Candidate image directories:")
    for directory, count, _, _ in sorted(scored, key=lambda item: item[1], reverse=True)[:12]:
        print(f"  {count:5d} images  {directory}")
    print(f"[flir] Using LR: {lr_dir}")
    print(f"[flir] Using HR: {hr_dir}")
    return lr_dir, hr_dir


def list_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def match_pairs(lr_dir: Path, hr_dir: Path) -> list[tuple[Path, Path]]:
    lr_paths = list_images(lr_dir)
    hr_paths = list_images(hr_dir)
    hr_by_rel = {p.relative_to(hr_dir).with_suffix("").as_posix(): p for p in hr_paths}
    pairs = []
    for lr_path in lr_paths:
        key = lr_path.relative_to(lr_dir).with_suffix("").as_posix()
        if key in hr_by_rel:
            pairs.append((lr_path, hr_by_rel[key]))
    if pairs:
        return pairs

    hr_by_stem = {p.stem: p for p in hr_paths}
    pairs = [(lr_path, hr_by_stem[lr_path.stem]) for lr_path in lr_paths if lr_path.stem in hr_by_stem]
    if pairs:
        return pairs

    if len(lr_paths) == len(hr_paths):
        return list(zip(lr_paths, hr_paths))
    raise RuntimeError(f"Could not match LR/HR pairs ({len(lr_paths)} LR, {len(hr_paths)} HR). Pass --lr_dir/--hr_dir if detection picked the wrong folders.")


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        dst.hardlink_to(src)
    else:
        dst.symlink_to(src.resolve())


def write_split(split: str, pairs: Iterable[tuple[Path, Path]], output_dir: Path, mode: str) -> int:
    count = 0
    for count, (lr_path, hr_path) in enumerate(pairs, start=1):
        name = f"{count - 1:06d}{hr_path.suffix.lower()}"
        link_or_copy(lr_path, output_dir / split / "LR" / name, mode)
        link_or_copy(hr_path, output_dir / split / "HR" / name, mode)
    return count


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output)
    archive = raw_dir / "FLIR-IISR.zip"

    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if args.download:
        download_with_progress(FLIR_IISR_URL, archive)
    if args.extract:
        extract_zip(archive, raw_dir)

    lr_dir, hr_dir = discover_pair_dirs(raw_dir, args.lr_dir, args.hr_dir)
    pairs = match_pairs(lr_dir, hr_dir)
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    val_count = max(1, int(round(len(pairs) * args.val_fraction)))
    val_pairs = sorted(pairs[:val_count], key=lambda item: item[0].as_posix())
    train_pairs = sorted(pairs[val_count:], key=lambda item: item[0].as_posix())

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_count = write_split("train", train_pairs, output_dir, args.mode)
    val_count = write_split("val", val_pairs, output_dir, args.mode)

    print(f"[flir] output: {output_dir}")
    print(f"[flir] train pairs: {train_count}")
    print(f"[flir] val pairs: {val_count}")
    print("[flir] Train with configs/train_flir_iisr_espcn_micro_rgb_x4.yaml")


if __name__ == "__main__":
    main()
