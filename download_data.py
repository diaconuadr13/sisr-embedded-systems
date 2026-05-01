import argparse
import os
import subprocess
import tarfile
import zipfile


DIV2K_TRAIN_HR_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
DIV2K_VALID_HR_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
FLICKR2K_URL       = "https://huggingface.co/datasets/yangtao9009/Flickr2K/resolve/main/Flickr2K.zip"


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


def main() -> None:
    p = argparse.ArgumentParser(description="Download SISR training datasets.")
    p.add_argument("--div2k",   action="store_true", default=False, help="Download DIV2K")
    p.add_argument("--flickr2k", action="store_true", default=False, help="Download Flickr2K (~12 GB)")
    args = p.parse_args()

    # Default: download both if nothing specified
    if not args.div2k and not args.flickr2k:
        args.div2k = True
        args.flickr2k = True

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

    print("Done.")


if __name__ == "__main__":
    main()
