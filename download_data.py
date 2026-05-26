import argparse
import os
import random
import shlex
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple



DIV2K_TRAIN_HR_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
DIV2K_VALID_HR_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
FLICKR2K_URL       = "https://huggingface.co/datasets/yangtao9009/Flickr2K/resolve/main/Flickr2K.zip"
INFRARED_THERMAL_URL = "https://zenodo.org/records/5574233/files/Infrared%20thermal%20dataset.zip"
DEFAULT_KAGGLE_VIMEO90K_DATASET = "wangsally/vimeo-90k-9"


ClipEntry = Tuple[str, Path]


def download(url: str, dest: str) -> None:
    print(f"Downloading {os.path.basename(dest)} from {url}")
    subprocess.run(
        ["curl", "-L", "-C", "-", "--progress-bar", "-o", dest, url],
        check=True,
    )


def extract(archive: str, target_dir: str) -> None:
    print(f"Extracting {os.path.basename(archive)} -> {target_dir}")
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(target_dir)
    elif archive.endswith(".tar") or archive.endswith(".tar.gz") or archive.endswith(".tgz"):
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(target_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive}")
    os.remove(archive)
    print(f"Removed archive: {archive}")


def download_dataset(url: str, archive_path: str, marker_dir: str, label: str) -> None:
    if os.path.isdir(marker_dir):
        print(f"{label} already present at {marker_dir}, skipping.")
        return
    download(url, archive_path)
    extract(archive_path, os.path.dirname(archive_path))
    print(f"{label} ready.")


def ensure_kaggle_available() -> None:
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI not found. Install it with:\n"
            "  pip install kaggle"
        )

    has_token_file = (Path.home() / ".kaggle" / "kaggle.json").is_file()
    has_env_creds = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    if not has_token_file and not has_env_creds:
        raise RuntimeError(
            "Kaggle credentials not found.\n"
            "Create an API token from Kaggle account settings and place kaggle.json at ~/.kaggle/kaggle.json,\n"
            "or set KAGGLE_USERNAME and KAGGLE_KEY."
        )


def _archive_suffix(path: Path) -> str:
    suffixes = "".join(path.suffixes).lower()
    return suffixes


def _contains_extracted_files(path: Path) -> bool:
    if not path.exists():
        return False
    archive_suffixes = {".zip", ".tar", ".tar.gz", ".tgz"}
    for child in path.iterdir():
        if child.name.startswith("."):
            continue
        if child.is_dir():
            return True
        if _archive_suffix(child) not in archive_suffixes:
            return True
    return False


def download_kaggle_dataset(dataset: str, output_dir: Path, unzip: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if _contains_extracted_files(output_dir):
        print(f"Kaggle dataset already appears extracted at {output_dir}, skipping download.")
        return

    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(output_dir)]
    if unzip:
        cmd.append("--unzip")

    print(f"Downloading Kaggle dataset: {dataset}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Kaggle dataset download failed.\n"
            f"Command: {shlex.join(cmd)}\n"
            f"Exit code: {exc.returncode}"
        ) from exc


def _remove_kaggle_archives(output_dir: Path) -> None:
    archive_suffixes = {".zip", ".tar", ".tar.gz", ".tgz"}
    for path in output_dir.iterdir():
        if path.is_file() and _archive_suffix(path) in archive_suffixes:
            path.unlink()
            print(f"Removed archive: {path}")


def _small_directory_tree(root: Path, max_depth: int = 3, max_entries: int = 80) -> str:
    if not root.exists():
        return f"{root} does not exist"

    lines = [str(root)]
    entries_seen = 0

    def walk(path: Path, depth: int, prefix: str) -> None:
        nonlocal entries_seen
        if depth >= max_depth or entries_seen >= max_entries:
            return
        try:
            children = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return
        for child in children:
            if entries_seen >= max_entries:
                lines.append(f"{prefix}...")
                return
            lines.append(f"{prefix}{child.name}/" if child.is_dir() else f"{prefix}{child.name}")
            entries_seen += 1
            if child.is_dir():
                walk(child, depth + 1, prefix + "  ")

    walk(root, 0, "  ")
    return "\n".join(lines)


def _clip_frame_paths(clip_dir: Path) -> List[Path]:
    frames = sorted(
        clip_dir.glob("im*.png"),
        key=lambda p: (len(p.stem), p.stem.lower(), p.name.lower()),
    )
    if not frames:
        frames = sorted(clip_dir.glob("*.png"), key=lambda p: p.name.lower())
    return frames


def _has_vimeo_clip_children(sequences_dir: Path) -> bool:
    for clip_dir in sequences_dir.rglob("*"):
        if clip_dir.is_dir() and len(_clip_frame_paths(clip_dir)) >= 3:
            return True
    return False


def _find_sequence_dirs(raw_root: Path) -> List[Path]:
    sequence_dirs: List[Path] = []
    for current_root, dirnames, _filenames in os.walk(raw_root):
        if "sequences" in dirnames:
            sequence_dirs.append(Path(current_root) / "sequences")
            dirnames.remove("sequences")
    return sequence_dirs


def find_vimeo90k_root(raw_root: Path) -> Path:
    if not raw_root.exists():
        raise RuntimeError(f"Raw Vimeo-90K root does not exist: {raw_root}")

    candidates: List[Tuple[int, Path]] = []
    for sequences_dir in _find_sequence_dirs(raw_root):
        if not sequences_dir.is_dir() or not _has_vimeo_clip_children(sequences_dir):
            continue
        parent = sequences_dir.parent
        score = 0
        if (parent / "sep_trainlist.txt").is_file():
            score += 10
        if (parent / "sep_testlist.txt").is_file():
            score += 10
        if "vimeo" in parent.name.lower():
            score += 2
        candidates.append((score, parent))

    if candidates:
        candidates.sort(key=lambda item: (-item[0], len(item[1].parts), str(item[1])))
        return candidates[0][1]

    raise RuntimeError(
        "Could not detect a valid Vimeo-90K structure under the Kaggle extract.\n"
        "Expected a directory named 'sequences' containing clip folders with PNG frames.\n"
        "Directory tree:\n"
        f"{_small_directory_tree(raw_root)}"
    )


def _read_split_file(split_file: Path, sequences_dir: Path) -> List[ClipEntry]:
    clips: List[ClipEntry] = []
    for line in split_file.read_text().splitlines():
        rel = line.strip()
        if not rel:
            continue
        rel = rel.split()[0].replace("\\", "/")
        parts = Path(rel)
        clip_dir = sequences_dir / parts
        if parts.parts and parts.parts[0] == "sequences":
            clip_dir = sequences_dir.parent / parts
            rel = str(Path(*parts.parts[1:]))
        if clip_dir.is_dir() and len(_clip_frame_paths(clip_dir)) >= 3:
            clips.append((rel.replace("\\", "/"), clip_dir))
    return clips


def _scan_vimeo_clips(sequences_dir: Path) -> List[ClipEntry]:
    clips: List[ClipEntry] = []
    for clip_dir in sequences_dir.rglob("*"):
        if clip_dir.is_dir() and len(_clip_frame_paths(clip_dir)) >= 3:
            rel = clip_dir.relative_to(sequences_dir).as_posix()
            clips.append((rel, clip_dir))
    clips.sort(key=lambda item: item[0])
    return clips


def _split_scanned_clips(clips: List[ClipEntry]) -> Tuple[List[ClipEntry], List[ClipEntry]]:
    shuffled = list(clips)
    random.Random(42).shuffle(shuffled)
    split_idx = int(len(shuffled) * 0.9)
    if len(shuffled) > 1:
        split_idx = min(max(split_idx, 1), len(shuffled) - 1)
    train_clips = sorted(shuffled[:split_idx], key=lambda item: item[0])
    val_clips = sorted(shuffled[split_idx:], key=lambda item: item[0])
    return train_clips, val_clips


def _load_vimeo_splits(vimeo_root: Path) -> Tuple[List[ClipEntry], List[ClipEntry]]:
    sequences_dir = vimeo_root / "sequences"
    train_split = vimeo_root / "sep_trainlist.txt"
    val_split = vimeo_root / "sep_testlist.txt"

    if train_split.is_file() and val_split.is_file():
        train_clips = _read_split_file(train_split, sequences_dir)
        val_clips = _read_split_file(val_split, sequences_dir)
        if train_clips and val_clips:
            return train_clips, val_clips
        print("Split files were found, but no valid clips were resolved. Falling back to scanned split.")

    clips = _scan_vimeo_clips(sequences_dir)
    if not clips:
        raise RuntimeError(f"No Vimeo-90K clip folders with PNG frames found under {sequences_dir}")
    return _split_scanned_clips(clips)


def _apply_clip_limit(
    train_clips: List[ClipEntry],
    val_clips: List[ClipEntry],
    limit_clips: Optional[int],
) -> Tuple[List[ClipEntry], List[ClipEntry]]:
    if limit_clips is None:
        return train_clips, val_clips
    if limit_clips < 2:
        raise ValueError("--limit_clips must be at least 2 so both train and val splits can be validated.")

    total = len(train_clips) + len(val_clips)
    if total <= limit_clips:
        return train_clips, val_clips

    if not train_clips or not val_clips:
        return train_clips[:limit_clips], val_clips[: max(0, limit_clips - len(train_clips))]

    val_ratio = len(val_clips) / total
    val_limit = max(1, int(round(limit_clips * val_ratio)))
    train_limit = max(1, limit_clips - val_limit)
    if train_limit + val_limit > limit_clips:
        train_limit = limit_clips - val_limit
    return train_clips[:train_limit], val_clips[:val_limit]


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required to prepare Vimeo-90K LR frames. Install it with:\n"
            "  pip install opencv-python"
        ) from exc
    return cv2


def _link_or_copy(src: Path, dest: Path, copy_video_data: bool) -> None:
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if copy_video_data:
        shutil.copy2(src, dest)
        return
    try:
        dest.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dest)


def _prepare_clip(
    src_clip_dir: Path,
    out_clip_dir: Path,
    scale: int,
    max_frames_per_clip: Optional[int],
    copy_video_data: bool,
) -> int:
    cv2 = _import_cv2()
    src_frames = _clip_frame_paths(src_clip_dir)
    if max_frames_per_clip is not None:
        src_frames = src_frames[:max_frames_per_clip]
    if len(src_frames) < 3:
        raise RuntimeError(f"Clip {src_clip_dir} has fewer than 3 selected frames.")

    if out_clip_dir.exists():
        shutil.rmtree(out_clip_dir)
    hr_dir = out_clip_dir / "HR"
    lr_dir = out_clip_dir / f"LR_x{scale}"
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, src_frame in enumerate(src_frames):
        img = cv2.imread(str(src_frame), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Could not read Vimeo-90K frame: {src_frame}")

        height, width = img.shape[:2]
        crop_h = height - (height % scale)
        crop_w = width - (width % scale)
        if crop_h <= 0 or crop_w <= 0:
            raise RuntimeError(f"Frame {src_frame} is too small for scale x{scale}: {width}x{height}")

        out_name = f"frame_{frame_idx:03d}.png"
        hr_dest = hr_dir / out_name
        lr_dest = lr_dir / out_name

        if crop_h == height and crop_w == width:
            hr_img = img
            _link_or_copy(src_frame, hr_dest, copy_video_data)
        else:
            hr_img = img[:crop_h, :crop_w]
            if not cv2.imwrite(str(hr_dest), hr_img):
                raise RuntimeError(f"Could not write cropped HR frame: {hr_dest}")

        lr_img = cv2.resize(hr_img, (crop_w // scale, crop_h // scale), interpolation=cv2.INTER_CUBIC)
        if not cv2.imwrite(str(lr_dest), lr_img):
            raise RuntimeError(f"Could not write LR frame: {lr_dest}")

    return len(src_frames)


def _prepared_clip_dirs(split_dir: Path, scale: int) -> List[Path]:
    if not split_dir.is_dir():
        return []
    lr_name = f"LR_x{scale}"
    return sorted(
        [
            path
            for path in split_dir.iterdir()
            if path.is_dir() and (path / "HR").is_dir() and (path / lr_name).is_dir()
        ],
        key=lambda p: p.name,
    )


def _dataset_already_prepared(output_root: Path, scale: int) -> bool:
    return bool(_prepared_clip_dirs(output_root / "train", scale)) and bool(
        _prepared_clip_dirs(output_root / "val", scale)
    )


def validate_prepared_vimeo90k(output_root: Path, scale: int) -> Dict[str, int]:
    cv2 = _import_cv2()
    train_dir = output_root / "train"
    val_dir = output_root / "val"
    if not output_root.is_dir():
        raise RuntimeError(f"Prepared dataset root does not exist: {output_root}")
    if not train_dir.is_dir():
        raise RuntimeError(f"Prepared dataset train directory does not exist: {train_dir}")
    if not val_dir.is_dir():
        raise RuntimeError(f"Prepared dataset val directory does not exist: {val_dir}")

    summary: Dict[str, int] = {"train_clips": 0, "val_clips": 0, "train_frames": 0, "val_frames": 0}
    for split_name, split_dir in (("train", train_dir), ("val", val_dir)):
        clips = _prepared_clip_dirs(split_dir, scale)
        if not clips:
            raise RuntimeError(f"Prepared dataset has no {split_name} clips with LR_x{scale}.")
        for clip_dir in clips:
            hr_dir = clip_dir / "HR"
            lr_dir = clip_dir / f"LR_x{scale}"
            hr_frames = sorted(hr_dir.glob("frame_*.png"))
            lr_frames = sorted(lr_dir.glob("frame_*.png"))
            if len(hr_frames) < 3:
                raise RuntimeError(f"Prepared clip {clip_dir} has fewer than 3 HR frames.")
            if len(hr_frames) != len(lr_frames):
                raise RuntimeError(f"Prepared clip {clip_dir} has mismatched HR/LR frame counts.")
            for hr_frame, lr_frame in zip(hr_frames[:1], lr_frames[:1]):
                hr_img = cv2.imread(str(hr_frame), cv2.IMREAD_UNCHANGED)
                lr_img = cv2.imread(str(lr_frame), cv2.IMREAD_UNCHANGED)
                if hr_img is None:
                    raise RuntimeError(f"Could not read prepared HR frame: {hr_frame}")
                if lr_img is None:
                    raise RuntimeError(f"Could not read prepared LR frame: {lr_frame}")
                hr_h, hr_w = hr_img.shape[:2]
                lr_h, lr_w = lr_img.shape[:2]
                if hr_h != lr_h * scale or hr_w != lr_w * scale:
                    raise RuntimeError(
                        f"Prepared clip {clip_dir} has invalid x{scale} dimensions: "
                        f"HR {hr_w}x{hr_h}, LR {lr_w}x{lr_h}"
                    )
            summary[f"{split_name}_clips"] += 1
            summary[f"{split_name}_frames"] += len(hr_frames)
    return summary


def _print_vimeo_summary(output_root: Path, scale: int, summary: Dict[str, int]) -> None:
    print("Kaggle Vimeo-90K VSR dataset prepared:")
    print(f"  root: {output_root}")
    print(f"  scale: x{scale}")
    print(f"  train clips: {summary['train_clips']}")
    print(f"  val clips: {summary['val_clips']}")
    print(f"  train frames: {summary['train_frames']}")
    print(f"  val frames: {summary['val_frames']}")


def prepare_vimeo90k_vsr(
    vimeo_root: Path,
    output_root: Path,
    scale: int,
    limit_clips: Optional[int],
    max_frames_per_clip: Optional[int],
    copy_video_data: bool,
    force_prepare: bool,
) -> Dict[str, int]:
    if scale < 2:
        raise ValueError("--scale must be 2 or greater.")
    if max_frames_per_clip is not None and max_frames_per_clip < 3:
        raise ValueError("--max_frames_per_clip must be at least 3.")

    if output_root.exists() and force_prepare:
        print(f"Removing existing prepared Vimeo-90K output: {output_root}")
        shutil.rmtree(output_root)
    if output_root.exists() and _dataset_already_prepared(output_root, scale) and not force_prepare:
        print(f"Vimeo-90K VSR dataset already prepared for x{scale} at {output_root}, validating.")
        return validate_prepared_vimeo90k(output_root, scale)
    if output_root.exists():
        print(f"Rebuilding prepared Vimeo-90K output: {output_root}")
        shutil.rmtree(output_root)

    train_clips, val_clips = _load_vimeo_splits(vimeo_root)
    train_clips, val_clips = _apply_clip_limit(train_clips, val_clips, limit_clips)
    if not train_clips or not val_clips:
        raise RuntimeError("Vimeo-90K preparation requires at least one train clip and one val clip.")

    output_root.mkdir(parents=True, exist_ok=True)
    prepared_frames = {"train": 0, "val": 0}
    for split_name, clips in (("train", train_clips), ("val", val_clips)):
        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for clip_idx, (_rel, src_clip_dir) in enumerate(clips):
            out_clip_dir = split_dir / f"clip_{clip_idx:06d}"
            prepared_frames[split_name] += _prepare_clip(
                src_clip_dir=src_clip_dir,
                out_clip_dir=out_clip_dir,
                scale=scale,
                max_frames_per_clip=max_frames_per_clip,
                copy_video_data=copy_video_data,
            )

    summary = validate_prepared_vimeo90k(output_root, scale)
    if summary["train_frames"] < prepared_frames["train"] or summary["val_frames"] < prepared_frames["val"]:
        raise RuntimeError("Prepared Vimeo-90K validation found fewer frames than expected.")
    return summary


def handle_vimeo90k_kaggle(args: argparse.Namespace) -> None:
    raw_root = Path(args.vimeo90k_raw_root)
    output_root = Path(args.vimeo90k_root)

    ensure_kaggle_available()
    download_kaggle_dataset(args.kaggle_dataset, raw_root, unzip=True)
    if not args.keep_archives:
        _remove_kaggle_archives(raw_root)

    vimeo_root = find_vimeo90k_root(raw_root)
    print(f"Detected Vimeo-90K root: {vimeo_root}")
    summary = prepare_vimeo90k_vsr(
        vimeo_root=vimeo_root,
        output_root=output_root,
        scale=args.scale,
        limit_clips=args.limit_clips,
        max_frames_per_clip=args.max_frames_per_clip,
        copy_video_data=args.copy_video_data,
        force_prepare=args.force_prepare,
    )
    _print_vimeo_summary(output_root, args.scale, summary)


def main() -> None:
    p = argparse.ArgumentParser(description="Download SISR training datasets.")
    p.add_argument("--div2k",   action="store_true", default=False, help="Download DIV2K")
    p.add_argument("--flickr2k", action="store_true", default=False, help="Download Flickr2K (~12 GB)")
    p.add_argument("--infrared_thermal", action="store_true", default=False, help="Download Infrared Thermal dataset")
    p.add_argument("--vimeo90k_kaggle", action="store_true", default=False, help="Download and prepare Vimeo-90K from Kaggle")
    p.add_argument("--kaggle_dataset", default=DEFAULT_KAGGLE_VIMEO90K_DATASET, help="Kaggle dataset slug for Vimeo-90K")
    p.add_argument("--vimeo90k_root", default="data/vimeo90k_vsr", help="Prepared Vimeo-90K VSR output root")
    p.add_argument("--vimeo90k_raw_root", default="data/raw/vimeo90k_kaggle", help="Raw Kaggle Vimeo-90K extract root")
    p.add_argument("--scale", type=int, default=2, help="LR downsampling scale for Vimeo-90K")
    p.add_argument("--limit_clips", type=int, default=None, help="Prepare only the first N total Vimeo-90K clips")
    p.add_argument("--max_frames_per_clip", type=int, default=None, help="Prepare only the first N frames per Vimeo-90K clip")
    p.add_argument("--copy_video_data", action="store_true", help="Copy HR video frames instead of symlinking when possible")
    p.add_argument("--force_prepare", action="store_true", help="Rebuild the prepared Vimeo-90K output root")
    p.add_argument("--keep_archives", action="store_true", help="Keep Kaggle archive files after extraction")
    args = p.parse_args()

    # Default: download existing image datasets if nothing specified.
    if not args.div2k and not args.flickr2k and not args.infrared_thermal and not args.vimeo90k_kaggle:
        args.div2k = True
        args.flickr2k = True
        args.infrared_thermal = True

    train_dir = os.path.join("data", "train")
    val_dir   = os.path.join("data", "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir,   exist_ok=True)

    if args.div2k:
        download_dataset(
            DIV2K_TRAIN_HR_URL,
            os.path.join(train_dir, "DIV2K_train_HR.zip"),
            os.path.join(train_dir, "DIV2K_train_HR"),
            "DIV2K train",
        )
        download_dataset(
            DIV2K_VALID_HR_URL,
            os.path.join(val_dir, "DIV2K_valid_HR.zip"),
            os.path.join(val_dir, "DIV2K_valid_HR"),
            "DIV2K valid",
        )

    if args.flickr2k:
        download_dataset(
            FLICKR2K_URL,
            os.path.join(train_dir, "Flickr2K.zip"),
            os.path.join(train_dir, "Flickr2K"),
            "Flickr2K",
        )

    if args.infrared_thermal:
        download_dataset(
            INFRARED_THERMAL_URL,
            os.path.join(train_dir, "Infrared_thermal.zip"),
            os.path.join(train_dir, "Infrared_thermal"),
            "Infrared Thermal",
        )

    if args.vimeo90k_kaggle:
        handle_vimeo90k_kaggle(args)

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
