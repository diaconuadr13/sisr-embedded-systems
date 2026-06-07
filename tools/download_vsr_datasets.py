from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import requests


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetSpec:
    canonical_name: str
    aliases: tuple[str, ...]
    expected_clips: int
    min_frames_per_clip: int
    min_size: tuple[int, int]
    size_label: str
    urls: tuple[str, ...] = ()
    google_drive_ids: tuple[str, ...] = ()
    preferred_roots: tuple[str, ...] = ("GT", "truth", "HR", "sharp", "val_sharp", "train_sharp")
    manual_note: str = ""
    large: bool = False
    selected_clips: tuple[str, ...] = ()


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "vid4": DatasetSpec(
        canonical_name="Vid4",
        aliases=("vid4",),
        expected_clips=4,
        min_frames_per_clip=20,
        min_size=(240, 240),
        size_label="small",
        urls=("https://paddlegan.bj.bcebos.com/datasets/Vid4.zip",),
        google_drive_ids=("1ZuvNNLgR85TV_whJoHM7uVb-XW1y70DW",),
        manual_note="Vid4 is also linked from MMEditing/MMagic dataset docs.",
    ),
    "udm10": DatasetSpec(
        canonical_name="UDM10",
        aliases=("udm10", "udm"),
        expected_clips=10,
        min_frames_per_clip=20,
        min_size=(360, 640),
        size_label="small/medium",
        urls=("https://paddlegan.bj.bcebos.com/datasets/udm10.tar",),
        google_drive_ids=("1G4V4KZZhhfzUlqHiSBBuWyqLyIOvOs0W",),
        manual_note="UDM10 should contain a GT directory with 10 HD sequences.",
    ),
    "spmcs": DatasetSpec(
        canonical_name="SPMCS",
        aliases=("spmcs", "spmc"),
        expected_clips=30,
        min_frames_per_clip=31,
        min_size=(120, 160),
        size_label="small",
        urls=("https://tinyurl.com/y426dcn9",),
        preferred_roots=("truth", "GT", "HR"),
        manual_note="SPMCS is published from https://github.com/jiangsutx/SPMC_VideoSR; if the tinyurl is stale, download the testing set manually and pass --source-dir.",
    ),
    "reds4": DatasetSpec(
        canonical_name="REDS4",
        aliases=("reds4", "reds"),
        expected_clips=4,
        min_frames_per_clip=100,
        min_size=(360, 640),
        size_label="large",
        urls=("https://huggingface.co/datasets/snah/REDS/resolve/main/train_sharp.zip",),
        google_drive_ids=("1YLksKtMhd2mWyVSkvhDaDLWSc1qYNCz-",),
        manual_note="REDS4 uses clips 000, 011, 015, 020 from REDS train_sharp. Download REDS from https://seungjunnah.github.io/Datasets/reds.html or BasicSR docs, then pass --source-dir.",
        large=True,
        selected_clips=("000", "011", "015", "020"),
    ),
}


def spec_for(dataset: str) -> DatasetSpec:
    key = dataset.lower()
    for spec in DATASET_SPECS.values():
        if key in spec.aliases:
            return spec
    raise ValueError(f"Unknown dataset '{dataset}'. Expected one of: vid4, udm10, spmcs, reds4, all-small")


def run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def download_with_gdown(file_id: str, dest: Path) -> bool:
    if shutil.which("gdown") is None:
        print("[download] gdown is not installed. Install with: pip install gdown")
        return False
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        run(["gdown", "--fuzzy", url, "-O", str(dest)])
        return dest.exists() and dest.stat().st_size > 0
    except subprocess.CalledProcessError:
        return False


def _google_drive_confirm_params(html: str) -> Dict[str, str]:
    params: Dict[str, str] = {"confirm": "t"}
    uuid_match = re.search(r'name="uuid"\s+value="([^"]+)"', html)
    if uuid_match:
        params["uuid"] = uuid_match.group(1)
    return params


def download_google_drive_file(file_id: str, dest: Path) -> bool:
    """Download a public Google Drive file without requiring gdown."""
    session = requests.Session()
    base_params = {"export": "download", "id": file_id}
    try:
        response = session.get(
            "https://drive.google.com/uc",
            params=base_params,
            stream=True,
            timeout=30,
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            confirm_params = {**base_params, **_google_drive_confirm_params(response.text)}
            response.close()
            response = session.get(
                "https://drive.usercontent.google.com/download",
                params=confirm_params,
                stream=True,
                timeout=30,
            )
            response.raise_for_status()

        first_chunk = True
        total = int(response.headers.get("content-length") or 0)
        downloaded = 0
        with dest.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                if first_chunk and chunk.lstrip().startswith(b"<!DOCTYPE html"):
                    title_match = re.search(rb"<title>([^<]+)</title>", chunk[:4096], flags=re.IGNORECASE)
                    if title_match:
                        title = title_match.group(1).decode("utf-8", errors="replace")
                        print(f"[download] Google Drive returned HTML instead of an archive: {title}")
                    try:
                        dest.unlink()
                    except FileNotFoundError:
                        pass
                    return False
                first_chunk = False
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100.0 / total
                    print(f"\r[download] Google Drive {pct:5.1f}%", end="", flush=True)
        if total:
            print()
        return dest.exists() and dest.stat().st_size > 0
    except requests.RequestException as exc:
        print(f"[download] Google Drive direct download failed: {exc}")
        return False


def download_with_curl(url: str, dest: Path) -> bool:
    if shutil.which("curl") is None:
        return False
    try:
        run(["curl", "-L", "-C", "-", "--fail", "--progress-bar", "-o", str(dest), url])
        return dest.exists() and dest.stat().st_size > 0
    except subprocess.CalledProcessError:
        return False


def extract_archive(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(archive.suffixes).lower()
    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(target)
        return
    if suffixes.endswith(".tar") or suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(target)
        return
    raise ValueError(f"Unsupported archive format: {archive}")


def image_paths(directory: Path) -> List[Path]:
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def natural_frame_name(index: int) -> str:
    return f"frame_{index:03d}.png"


def preferred_score(path: Path, spec: DatasetSpec) -> int:
    parts = {part.lower() for part in path.parts}
    score = sum(10 for root in spec.preferred_roots if root.lower() in parts)
    if spec.selected_clips and path.name in spec.selected_clips:
        score += 20
    if any(part.lower().startswith(("bi", "bd", "lr", "x2", "x3", "x4")) for part in path.parts):
        score -= 8
    return score


def find_clip_dirs(source_root: Path, spec: DatasetSpec) -> List[Path]:
    candidates = [p for p in source_root.rglob("*") if p.is_dir() and len(image_paths(p)) >= 1]
    if spec.selected_clips:
        candidates = [p for p in candidates if p.name in spec.selected_clips]
    candidates.sort(key=lambda p: (-preferred_score(p, spec), len(p.parts), str(p)))

    selected: List[Path] = []
    used_names = set()
    for candidate in candidates:
        name = candidate.name
        if name in used_names:
            continue
        selected.append(candidate)
        used_names.add(name)
        if len(selected) >= spec.expected_clips:
            break
    return sorted(selected, key=lambda p: p.name)


def normalize_dataset(source_root: Path, output_root: Path, spec: DatasetSpec, force: bool = False) -> Path:
    dataset_root = output_root / spec.canonical_name
    if dataset_root.exists() and force:
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    clip_dirs = find_clip_dirs(source_root, spec)
    if not clip_dirs:
        raise RuntimeError(f"No clip directories with image frames found under {source_root}")

    for clip_dir in clip_dirs:
        frames = image_paths(clip_dir)
        out_clip = dataset_root / clip_dir.name
        if out_clip.exists():
            shutil.rmtree(out_clip)
        out_clip.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(frames):
            dest = out_clip / natural_frame_name(idx)
            img = cv2.imread(str(frame), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Could not read frame: {frame}")
            if not cv2.imwrite(str(dest), img):
                raise RuntimeError(f"Could not write normalized frame: {dest}")
    validate_normalized_dataset(dataset_root, spec)
    return dataset_root


def validate_normalized_dataset(dataset_root: Path, spec: DatasetSpec) -> Dict[str, int]:
    if not dataset_root.is_dir():
        raise RuntimeError(f"Dataset root does not exist: {dataset_root}")
    clips = sorted(p for p in dataset_root.iterdir() if p.is_dir())
    if len(clips) < min(spec.expected_clips, 1):
        raise RuntimeError(f"{spec.canonical_name}: expected clips, found {len(clips)}")

    total_frames = 0
    min_w, min_h = spec.min_size
    for clip in clips:
        frames = image_paths(clip)
        if len(frames) < spec.min_frames_per_clip:
            raise RuntimeError(
                f"{spec.canonical_name}/{clip.name}: expected at least {spec.min_frames_per_clip} frames, found {len(frames)}"
            )
        expected_names = [natural_frame_name(i) for i in range(len(frames))]
        if [p.name for p in frames] != expected_names:
            raise RuntimeError(f"{spec.canonical_name}/{clip.name}: frames are not normalized as frame_000.png, ...")
        for frame in frames[:1]:
            img = cv2.imread(str(frame), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Could not read normalized frame: {frame}")
            h, w = img.shape[:2]
            if w < min_w or h < min_h:
                raise RuntimeError(
                    f"{spec.canonical_name}/{clip.name}: frame too small ({w}x{h}); expected at least {min_w}x{min_h}"
                )
        total_frames += len(frames)

    return {"clips": len(clips), "frames": total_frames}


def download_to_temp(spec: DatasetSpec, temp_dir: Path) -> Optional[Path]:
    def try_google_drive() -> Optional[Path]:
        archive = temp_dir / f"{spec.canonical_name}.zip"
        for file_id in spec.google_drive_ids:
            print(f"[download] Trying Google Drive for {spec.canonical_name}: {file_id}")
            if download_google_drive_file(file_id, archive):
                return archive
            if download_with_gdown(file_id, archive):
                return archive
        return None

    def try_urls() -> Optional[Path]:
        for index, url in enumerate(spec.urls, start=1):
            suffix = Path(url.split("?")[0]).suffix or ".zip"
            archive = temp_dir / f"{spec.canonical_name}_{index}{suffix}"
            print(f"[download] Trying URL for {spec.canonical_name}: {url}")
            if download_with_curl(url, archive):
                return archive
        return None

    if spec.large and spec.urls:
        return try_urls() or try_google_drive()
    return try_google_drive() or try_urls()


def print_manual_instructions(spec: DatasetSpec, root: Path) -> None:
    print(f"[manual] Could not download {spec.canonical_name} automatically.")
    print(f"[manual] {spec.manual_note}")
    print("[manual] After downloading/extracting, run:")
    print(f"  python tools/download_vsr_datasets.py --dataset {spec.aliases[0]} --root {root} --source-dir /path/to/extracted/{spec.canonical_name}")


def prepare_dataset(spec: DatasetSpec, root: Path, source_dir: Optional[Path], force: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if source_dir is not None:
        out = normalize_dataset(source_dir, root, spec, force=force)
        summary = validate_normalized_dataset(out, spec)
        print(f"[ok] {spec.canonical_name}: {summary['clips']} clips, {summary['frames']} frames -> {out}")
        return

    with tempfile.TemporaryDirectory(prefix=f"{spec.canonical_name.lower()}_") as tmp:
        temp_dir = Path(tmp)
        archive = download_to_temp(spec, temp_dir)
        if archive is None:
            print_manual_instructions(spec, root)
            return
        extracted = temp_dir / "extracted"
        try:
            extract_archive(archive, extracted)
        except (ValueError, zipfile.BadZipFile, tarfile.TarError):
            print_manual_instructions(spec, root)
            return
        out = normalize_dataset(extracted, root, spec, force=force)
        summary = validate_normalized_dataset(out, spec)
        print(f"[ok] {spec.canonical_name}: {summary['clips']} clips, {summary['frames']} frames -> {out}")


def iter_requested(dataset: str) -> Iterable[DatasetSpec]:
    if dataset == "all-small":
        for key in ("vid4", "udm10", "spmcs"):
            yield DATASET_SPECS[key]
        return
    yield spec_for(dataset)


def main() -> None:
    p = argparse.ArgumentParser(description="Download and normalize VSR evaluation datasets.")
    p.add_argument("--dataset", required=True, choices=["vid4", "udm10", "spmcs", "reds4", "all-small"])
    p.add_argument("--root", default="data/vsr")
    p.add_argument("--source-dir", default=None, help="Use an already downloaded/extracted dataset instead of network download")
    p.add_argument("--confirm-large-download", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    root = Path(args.root)
    source_dir = Path(args.source_dir) if args.source_dir else None
    for spec in iter_requested(args.dataset):
        if spec.large and not args.confirm_large_download and source_dir is None:
            print(f"[large] {spec.canonical_name} is a large dataset ({spec.size_label}).")
            print("[large] Re-run with --confirm-large-download, or download manually and pass --source-dir.")
            print_manual_instructions(spec, root)
            continue
        prepare_dataset(spec, root, source_dir, force=args.force)

    print("[info] Vimeo90K training data is handled by the existing flow:")
    print("  python download_data.py --vimeo90k_kaggle --vimeo90k_raw_root data/raw/vimeo90k_kaggle")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
