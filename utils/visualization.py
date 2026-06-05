"""diffusion sample 的影像繪圖與儲存工具。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _as_grid_tensor(images: torch.Tensor, mode: str = "L", nrow: int | None = None) -> torch.Tensor:
    """將一個 batch 的影像轉成 torchvision grid tensor。"""

    try:
        from torchvision.utils import make_grid
    except ImportError as exc:
        raise ImportError("Image grid utilities require the optional 'torchvision' package.") from exc

    images = images.detach().cpu()
    if images.dtype == torch.uint8:
        images = images.float() / 255.0
    images = images.float().clamp(0.0, 1.0)

    if mode == "L" and images.ndim == 4:
        batch_size, channels, height, width = images.shape
        images = images.reshape(batch_size * channels, 1, height, width)
    elif images.ndim == 3:
        images = images[:, None]
    elif images.ndim == 4 and images.shape[1] not in (1, 3):
        images = images[:, :3]

    return make_grid(images, nrow=nrow or max(1, int(images.shape[0] ** 0.5)))


def grid_to_numpy(images: torch.Tensor, mode: str = "L", nrow: int | None = None) -> np.ndarray:
    """回傳 HWC uint8 numpy grid，供 logging 或外部上傳使用。"""

    grid = _as_grid_tensor(images, mode=mode, nrow=nrow)
    array = (grid.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    if mode == "L":
        return array[:, :, 0]
    return array


def save_images(
    images: torch.Tensor,
    path: str | Path,
    mode: str = "L",
    nrow: int | None = None,
) -> Path:
    """將一個 batch 的影像存成單張 grid，格式相容原版 ANDi。"""

    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = grid_to_numpy(images, mode=mode, nrow=nrow)
    image_mode = "L" if mode == "L" else None
    Image.fromarray(array, mode=image_mode).save(path)
    return path


def upload_images(images: torch.Tensor, mode: str = "L", nrow: int | None = None) -> np.ndarray:
    """回傳適合 wandb.Image 或類似 logger 使用的 numpy image grid。"""

    return grid_to_numpy(images, mode=mode, nrow=nrow)


def plot_images(images: torch.Tensor, mode: str = "L", nrow: int | None = None) -> None:
    """使用 matplotlib 顯示一個 batch 的影像。"""

    from matplotlib import pyplot as plt

    array = grid_to_numpy(images, mode=mode, nrow=nrow)
    plt.figure(figsize=(12, 12))
    if mode == "L":
        plt.imshow(array, cmap="gray")
    else:
        plt.imshow(array)
    plt.axis("off")
    plt.show()
