"""Sensor lane — pluggable residual, PCE later."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from ..image import gaussian_blur, to_gray
from ..report import LaneFinding


class ResidualBackend(Protocol):
    name: str

    def residual(self, gray: np.ndarray) -> np.ndarray: ...


@dataclass
class GaussianResidual:
    name: str = "gaussian_highpass"
    sigma: float = 1.2

    def residual(self, gray: np.ndarray) -> np.ndarray:
        out = gray - gaussian_blur(gray, sigma=self.sigma)
        return out - out.mean()


@dataclass
class DrunetResidual:
    name: str = "drunet"
    weights_path: Optional[str] = None

    def residual(self, gray: np.ndarray) -> np.ndarray:
        raise RuntimeError(
            "DRUNet residual is not loaded. Place KAIR weights on the GPU box "
            "and construct DrunetResidual(weights_path=...)."
        )


def analyze_sensor(
    rgb: np.ndarray,
    backend: Optional[ResidualBackend] = None,
    fingerprint: Optional[np.ndarray] = None,
) -> LaneFinding:
    backend = backend or GaussianResidual()
    gray = to_gray(rgb)
    try:
        residual = backend.residual(gray)
    except Exception as exc:
        return LaneFinding(
            lane="sensor",
            finding="insufficient",
            confidence=0.0,
            mechanism=f"Residual backend {backend.name} failed: {exc}",
            metrics={"backend": backend.name},
        )

    metrics = {
        "backend": backend.name,
        "residual_std": float(residual.std()),
        "residual_mean": float(residual.mean()),
        "has_fingerprint": fingerprint is not None,
    }
    if fingerprint is None:
        return LaneFinding(
            lane="sensor",
            finding="insufficient",
            confidence=0.3,
            mechanism=(
                f"{backend.name} residual extracted (std={metrics['residual_std']:.5f}). "
                "No camera fingerprint on file, so PCE is not computed. "
                "Absence of a match is not evidence of generation."
            ),
            metrics=metrics,
        )
    h = min(residual.shape[0], fingerprint.shape[0])
    w = min(residual.shape[1], fingerprint.shape[1])
    a = residual[:h, :w] - residual[:h, :w].mean()
    b = fingerprint[:h, :w] - fingerprint[:h, :w].mean()
    denom = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()) + 1e-12)
    corr = float((a * b).sum() / denom)
    metrics["centered_corr"] = corr
    return LaneFinding(
        lane="sensor",
        finding="consistent" if corr > 0.15 else "conflict",
        confidence=min(0.7, abs(corr)),
        mechanism="Centered correlation against a supplied fingerprint. Not PCE yet.",
        metrics=metrics,
    )
