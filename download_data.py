import os
import urllib.request
import zipfile


DIV2K_TRAIN_HR_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
DIV2K_VALID_HR_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"


def download_zip(url: str, zip_path: str) -> None:
    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        percent = min(100.0, downloaded * 100.0 / total_size)
        print(f"Downloading {os.path.basename(zip_path)}: {percent:6.2f}%", end="\r", flush=True)

    print(f"Starting download: {url}")
    urllib.request.urlretrieve(url, zip_path, reporthook=reporthook)
    print(f"Downloaded: {zip_path}" + " " * 20)


def extract_zip(zip_path: str, target_dir: str) -> None:
    print(f"Extracting {zip_path} -> {target_dir}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
    os.remove(zip_path)
    print(f"Removed archive: {zip_path}")


def main() -> None:
    train_dir = os.path.join("data", "train")
    val_dir = os.path.join("data", "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_zip_path = os.path.join(train_dir, "DIV2K_train_HR.zip")
    val_zip_path = os.path.join(val_dir, "DIV2K_valid_HR.zip")

    download_zip(DIV2K_TRAIN_HR_URL, train_zip_path)
    extract_zip(train_zip_path, train_dir)

    download_zip(DIV2K_VALID_HR_URL, val_zip_path)
    extract_zip(val_zip_path, val_dir)

    print("DIV2K download and extraction complete.")


if __name__ == "__main__":
    main()
