#!/usr/bin/env python3
"""
PRNU / sensor-noise baseline (Fridrich / Goljan PCE).
Reference: polimi-ispl/prnu-python. Wavelet residual + Wiener + PCE.
Not a scam stamp. DRUNet not included.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

try:
    import cv2
    from cv2 import DFT_COMPLEX_OUTPUT, DFT_REAL_OUTPUT, DFT_SCALE
    import pywt
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install: numpy opencv-python-headless pywavelets scipy"
    ) from exc


def _to_float_gray(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.float64:
        image = image.astype(np.float64)
    if image.max() > 1.0:
        image = image / 255.0
    if image.ndim == 3:
        image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    return image.astype(np.float64)


def _wiener_dft(residual: np.ndarray, noise_var: float = 4.0) -> np.ndarray:
    h, w = residual.shape
    pad_h = h + (h % 2)
    pad_w = w + (w % 2)
    padded = np.zeros((pad_h, pad_w), dtype=np.float64)
    padded[:h, :w] = residual
    dft = cv2.dft(padded, flags=DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    mag = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    noise_var = max(noise_var, 1e-6)
    gain = mag / (mag + noise_var)
    dft_shift[:, :, 0] *= gain
    dft_shift[:, :, 1] *= gain
    restored = cv2.idft(np.fft.ifftshift(dft_shift), flags=DFT_SCALE | DFT_REAL_OUTPUT)
    return restored[:h, :w]


def _mad_sigma(coeff: np.ndarray) -> float:
    med = np.median(coeff)
    mad = np.median(np.abs(coeff - med))
    return mad / 0.6745


def _wavelet_denoise(image: np.ndarray, wavelet: str = "db8", level: int = 4) -> np.ndarray:
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    detail_levels = coeffs[1:]
    denoised_details = []
    for cH, cV, cD in detail_levels:
        sigma = _mad_sigma(np.concatenate([cH.ravel(), cV.ravel(), cD.ravel()]))
        threshold = sigma * math.sqrt(2 * math.log(image.size)) if sigma > 0 else 0.0
        denoised_details.append((
            pywt.threshold(cH, threshold, mode="soft"),
            pywt.threshold(cV, threshold, mode="soft"),
            pywt.threshold(cD, threshold, mode="soft"),
        ))
    out = pywt.waverec2((coeffs[0], *denoised_details), wavelet)
    return out[: image.shape[0], : image.shape[1]]


def extract_noise_residual(
    image: np.ndarray,
    denoiser: str = "wavelet",
    wiener: bool = True,
    zero_mean: bool = True,
) -> np.ndarray:
    gray = _to_float_gray(image)
    if denoiser != "wavelet":
        raise ValueError(f"Unknown denoiser: {denoiser}")
    residual = gray - _wavelet_denoise(gray)
    if zero_mean:
        residual = residual - residual.mean()
    if wiener:
        residual = _wiener_dft(residual)
    return residual


def build_camera_fingerprint(flat_images: List[np.ndarray]) -> np.ndarray:
    if len(flat_images) < 10:
        print(f"[warn] Only {len(flat_images)} flat images. ~50 recommended.")
    residuals = [extract_noise_residual(img) for img in flat_images]
    h = min(r.shape[0] for r in residuals)
    w = min(r.shape[1] for r in residuals)
    residuals = [r[:h, :w] for r in residuals]
    fingerprint = np.mean(residuals, axis=0)
    fingerprint = fingerprint - fingerprint.mean()
    return _wiener_dft(fingerprint)


def _normalized_cross_correlation(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    q = query - query.mean()
    r = reference - reference.mean()
    q_norm = np.sqrt((q ** 2).sum())
    r_norm = np.sqrt((r ** 2).sum())
    if q_norm < 1e-12 or r_norm < 1e-12:
        return np.zeros_like(query)
    corr = np.fft.ifft2(np.fft.fft2(q) * np.conj(np.fft.fft2(r))).real
    return corr / (q_norm * r_norm)


def match_pce(query_image: np.ndarray, fingerprint: np.ndarray, peak_region: int = 11) -> dict:
    residual = extract_noise_residual(query_image)
    h = min(residual.shape[0], fingerprint.shape[0])
    w = min(residual.shape[1], fingerprint.shape[1])
    residual = residual[:h, :w]
    fingerprint = fingerprint[:h, :w]
    corr_plane = _normalized_cross_correlation(residual, fingerprint)
    peak_idx = np.unravel_index(np.argmax(np.abs(corr_plane)), corr_plane.shape)
    peak_val = corr_plane[peak_idx]
    energy_mask = np.ones_like(corr_plane, dtype=bool)
    ph, pw = peak_idx
    r0 = max(0, ph - peak_region // 2)
    r1 = min(corr_plane.shape[0], ph + peak_region // 2 + 1)
    c0 = max(0, pw - peak_region // 2)
    c1 = min(corr_plane.shape[1], pw + peak_region // 2 + 1)
    energy_mask[r0:r1, c0:c1] = False
    energy = np.mean(corr_plane[energy_mask] ** 2)
    pce = (peak_val ** 2) / energy if energy > 1e-12 else 0.0
    threshold = 60.0
    return {
        "pce": float(pce),
        "peak_correlation": float(peak_val),
        "peak_location": (int(ph), int(pw)),
        "matches_camera": bool(pce > threshold),
        "threshold": threshold,
    }


def local_noise_variance_map(image: np.ndarray, block: int = 32) -> np.ndarray:
    gray = _to_float_gray(image)
    h, w = gray.shape
    out = np.zeros_like(gray)
    for i in range(0, h - block, block):
        for j in range(0, w - block, block):
            block_img = gray[i : i + block, j : j + block]
            out[i : i + block, j : j + block] = cv2.Laplacian(block_img, cv2.CV_64F).var()
    if out.max() > 0:
        out = out / out.max()
    return out


def analyze_image(image: np.ndarray, fingerprint: Optional[np.ndarray] = None) -> dict:
    residual = extract_noise_residual(image)
    noise_var_map = local_noise_variance_map(image)
    result = {
        "residual_std": float(residual.std()),
        "noise_variance_map": noise_var_map,
        "noise_variance_inconsistency": float(noise_var_map.std()),
    }
    if fingerprint is not None:
        result["camera_match"] = match_pce(image, fingerprint)
    return result
