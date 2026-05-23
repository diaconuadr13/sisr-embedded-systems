import argparse
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable


VIMEO90K_SEPTUPLET_URL = "http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Vimeo-90K septuplet clips for VideoSISRDataset.")
    parser.add_argument("--raw_dir", type=str, default="data/raw/vimeo90k")
    parser.add_argument("--output", type=str, default="data/video_vimeo90k")
    parser.add_argument("--download", action="store_true", help="Download vimeo_septuplet.zip if it is missing.")
    parser.add_argument("--extract", action="store_true", help="Extract vimeo_septuplet.zip if sequences are missing.")
    parser.add_argument("--num_train_clips", type=int, default=None, help="Optional cap for quick experiments.")
    parser.add_argument("--num_val_clips", type=int, default=1000, help="Validation clips to prepare when available.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("symlink", "copy", "hardlink"), default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"[vimeo90k] Found existing archive: {destination}")
        return
    print(f"[vimeo90k] Downloading {url} -> {destination}")
    urllib.request.urlretrieve(url, destination)


def extract_zip(archive: Path, raw_dir: Path) -> None:
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")
    print(f"[vimeo90k] Extracting {archive} into {raw_dir}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_dir)


def find_sequences_dir(raw_dir: Path) -> Path:
    candidates = [
        raw_dir / "vimeo_septuplet" / "sequences",
        raw_dir / "sequences",
        raw_dir / "vimeo90k" / "GT",
        raw_dir / "GT",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    found = [p for p in raw_dir.rglob("sequences") if p.is_dir()]
    if found:
        return found[0]
    raise FileNotFoundError(
        "Could not find Vimeo-90K sequences directory. Expected something like "
        "data/raw/vimeo90k/vimeo_septuplet/sequences."
    )


def find_list_file(raw_dir: Path, name: str) -> Path | None:
    candidates = [raw_dir / name, raw_dir / "vimeo_septuplet" / name]
    candidates.extend(raw_dir.rglob(name))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_clip_list(list_path: Path | None, sequences_dir: Path) -> list[str]:
    if list_path is not None:
        return [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    clips = []
    for directory in sorted(p for p in sequences_dir.rglob("*") if p.is_dir()):
        if list(directory.glob("im*.png")):
            clips.append(str(directory.relative_to(sequences_dir)))
    return clips


def select_clips(clips: list[str], limit: int | None, seed: int) -> list[str]:
    if limit is None or limit >= len(clips):
        return list(clips)
    rng = random.Random(seed)
    selected = list(clips)
    rng.shuffle(selected)
    return sorted(selected[:limit])


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        dst.hardlink_to(src)
    else:
        dst.symlink_to(src.resolve())


def prepare_split(split: str, clips: Iterable[str], sequences_dir: Path, output_dir: Path, mode: str) -> int:
    count = 0
    for count, clip_rel in enumerate(clips, start=1):
        src_dir = sequences_dir / clip_rel
        frame_paths = sorted(src_dir.glob("im*.png"))
        if not frame_paths:
            frame_paths = sorted(p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if not frame_paths:
            continue
        dst_dir = output_dir / split / f"clip_{count - 1:06d}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx, src in enumerate(frame_paths):
            dst = dst_dir / f"frame_{frame_idx:03d}{src.suffix.lower()}"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            link_or_copy(src, dst, mode)
    return count


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output)
    archive = raw_dir / "vimeo_septuplet.zip"

    if args.download:
        download_file(VIMEO90K_SEPTUPLET_URL, archive)
    if args.extract:
        extract_zip(archive, raw_dir)

    sequences_dir = find_sequences_dir(raw_dir)
    train_list = read_clip_list(find_list_file(raw_dir, "sep_trainlist.txt"), sequences_dir)
    test_list_file = find_list_file(raw_dir, "sep_testlist.txt") or find_list_file(raw_dir, "sep_validlist.txt")
    val_list = read_clip_list(test_list_file, sequences_dir)

    if not val_list and train_list:
        rng = random.Random(args.seed)
        shuffled = list(train_list)
        rng.shuffle(shuffled)
        val_count = min(args.num_val_clips, max(1, len(shuffled) // 20))
        val_list = sorted(shuffled[:val_count])
        train_list = sorted(shuffled[val_count:])

    train_clips = select_clips(train_list, args.num_train_clips, args.seed)
    val_clips = select_clips(val_list, args.num_val_clips, args.seed + 1)

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_count = prepare_split("train", train_clips, sequences_dir, output_dir, args.mode)
    val_count = prepare_split("val", val_clips, sequences_dir, output_dir, args.mode)

    print(f"[vimeo90k] sequences: {sequences_dir}")
    print(f"[vimeo90k] output: {output_dir}")
    print(f"[vimeo90k] train clips: {train_count}")
    print(f"[vimeo90k] val clips: {val_count}")
    print("[vimeo90k] Use --mode copy if your filesystem does not support symlinks.")


if __name__ == "__main__":
    main()
