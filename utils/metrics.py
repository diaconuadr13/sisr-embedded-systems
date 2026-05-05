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
    return float(ssim(img1_np, img2_np, channel_axis=-1, data_range=1.0))
