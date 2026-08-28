"""Minimal image IO. NumPy + Pillow only so this box can run it."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


def load_rgb(path: Union[str, Path]) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float64) / 255.0


def to_gray(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb.astype(np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def gaussian_blur(gray: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    tmp = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, gray)
    return np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), 0, tmp)
