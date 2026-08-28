"""Geometry lane — foundation first, then hooks."""

from __future__ import annotations

import numpy as np

from ..image import to_gray
from ..report import LaneFinding


def analyze_geometry(rgb: np.ndarray) -> LaneFinding:
    gray = to_gray(rgb)
    h, w = gray.shape
    qh, qw = h // 2, w // 2
    quads = {
        "tl": float(gray[:qh, :qw].mean()),
        "tr": float(gray[:qh, qw:].mean()),
        "bl": float(gray[qh:, :qw].mean()),
        "br": float(gray[qh:, qw:].mean()),
    }
    brightest = max(quads, key=quads.get)
    darkest = min(quads, key=quads.get)
    spread = quads[brightest] - quads[darkest]
    gy = np.abs(np.diff(gray, axis=0)).mean()
    gx = np.abs(np.diff(gray, axis=1)).mean()
    metrics = {
        "quad_means": quads,
        "brightest_quad": brightest,
        "darkest_quad": darkest,
        "illumination_spread": spread,
        "grad_x": float(gx),
        "grad_y": float(gy),
        "aspect": float(w / max(h, 1)),
    }
    if spread < 0.02:
        finding = "insufficient"
        mechanism = (
            "Near-flat illumination across quadrants. "
            "Cannot reconstruct a key light or a room from this file."
        )
        confidence = 0.4
    else:
        finding = "consistent"
        mechanism = (
            f"Key appears toward {brightest} with falloff toward {darkest} "
            f"(spread {spread:.3f}). This is a foundation sketch only — "
            "catchlights and contact shadows still have to agree with it."
        )
        confidence = min(0.55, 0.25 + spread)
    return LaneFinding(
        lane="geometry_foundation",
        finding=finding,
        confidence=float(confidence),
        mechanism=mechanism,
        metrics=metrics,
    )
