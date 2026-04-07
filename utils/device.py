import torch


def resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA-capable GPU is available.")

    mps_backend = getattr(torch.backends, "mps", None)
    if device.type == "mps" and (mps_backend is None or not mps_backend.is_available()):
        raise RuntimeError("MPS was requested, but it is not available in this PyTorch build.")

    return device


def configure_runtime(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
