"""Capture integrity.

Catches the case that keeps walking into this thread: a photo of a monitor.
This is a physical claim about the *file in hand*, not about the person on screen.
"""

from __future__ import annotations

import numpy as np

from ..image import to_gray
from ..report import LaneFinding


def _radial_fft_peaks(gray: np.ndarray) -> dict:
    h, w = gray.shape
    y0, x0 = h // 8, w // 8
    crop = gray[y0 : h - y0, x0 : w - x0]
    if crop.size < 64:
        crop = gray
    windowed = crop - crop.mean()
    spec = np.abs(np.fft.fftshift(np.fft.fft2(windowed)))
    spec = spec / (spec.max() + 1e-12)
    cy, cx = spec.shape[0] // 2, spec.shape[1] // 2
    spec[cy - 3 : cy + 4, cx - 3 : cx + 4] = 0
    peak_idx = np.unravel_index(np.argmax(spec), spec.shape)
    peak = float(spec[peak_idx])
    dy = (peak_idx[0] - cy) / max(cy, 1)
    dx = (peak_idx[1] - cx) / max(cx, 1)
    radius = float(np.hypot(dy, dx))
    axis = float(np.concatenate([spec[cy, :], spec[:, cx]]).mean())
    off = float(spec.mean())
    ratio = axis / (off + 1e-12)
    return {
        "fft_peak": peak,
        "axis_to_mean": ratio,
        "peak_radius": radius,
        "peak_dx": float(dx),
        "peak_dy": float(dy),
    }


def analyze_capture(rgb: np.ndarray) -> LaneFinding:
    gray = to_gray(rgb)
    stats = _radial_fft_peaks(gray)
    far_grid = stats["fft_peak"] >= 0.85 and stats["peak_radius"] >= 0.35
    if far_grid:
        finding = "second_generation"
        mechanism = (
            "Dominant far-from-DC spectral peak "
            f"(r={stats['peak_radius']:.2f}). Consistent with a sampled display lattice."
        )
        confidence = min(0.85, 0.5 + 0.3 * stats["fft_peak"])
    else:
        finding = "insufficient"
        mechanism = (
            "No dominant far-from-DC lattice isolated. "
            "Does not prove first-generation; only fails to prove recapture."
        )
        confidence = 0.35
    return LaneFinding(
        lane="capture",
        finding=finding,
        confidence=float(confidence),
        mechanism=mechanism,
        metrics=stats,
    )
