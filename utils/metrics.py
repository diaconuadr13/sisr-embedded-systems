import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    img1 = torch.clamp(img1.float(), 0.0, 1.0)
    img2 = torch.clamp(img2.float(), 0.0, 1.0)
    mse = torch.mean((img1 - img2) ** 2)
    if mse.item() == 0.0:
        return 100.0  # finite sentinel to avoid inf contaminating averages
    psnr = 10.0 * torch.log10(1.0 / mse)
    return float(psnr.item())


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    img1_np = img1.detach().cpu().permute(1, 2, 0).numpy()
    img2_np = img2.detach().cpu().permute(1, 2, 0).numpy()
    img1_np = np.clip(img1_np, 0.0, 1.0)
    img2_np = np.clip(img2_np, 0.0, 1.0)

    # Tiny HR/LR tiles can be narrower than skimage's default SSIM window; for very small
    # crops, compare tensors directly so the metric remains deterministic and bounded.
    if img1_np.shape[-1] == 1:
        img1_np = img1_np[..., 0]
        img2_np = img2_np[..., 0]
        channel_axis = None
    else:
        channel_axis = -1

    h, w = img1_np.shape[:2]
    if min(h, w) < 3:
        return 1.0 if np.array_equal(img1_np, img2_np) else 0.0

    max_win = min(7, h, w)
    win_size = max_win if max_win % 2 == 1 else max_win - 1

    return float(
        ssim(
            img1_np,
            img2_np,
            channel_axis=channel_axis,
            win_size=win_size,
            data_range=1.0,
        )
    )
