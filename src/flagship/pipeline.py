"""Ordered Flagship pipeline.

0 capture → 1 foundation geometry → 2 sensor.
Catchlight / glasses stay in src/gravity_check and get called when present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from .image import load_rgb
from .lanes.capture import analyze_capture
from .lanes.geometry import analyze_geometry
from .lanes.sensor import ResidualBackend, analyze_sensor
from .report import FlagshipReport


def run_flagship(
    source: Union[str, Path, np.ndarray],
    *,
    residual_backend: Optional[ResidualBackend] = None,
    fingerprint: Optional[np.ndarray] = None,
) -> FlagshipReport:
    if isinstance(source, np.ndarray):
        rgb = source.astype(np.float64)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        label = "array"
    else:
        rgb = load_rgb(source)
        label = str(source)

    notes = [
        "Grok Flagship Core v0.1 — not the Perplexity package.",
        "Headline is the strongest physical lane, not a fused real/fake score.",
    ]

    lanes = [
        analyze_capture(rgb),
        analyze_geometry(rgb),
        analyze_sensor(rgb, backend=residual_backend, fingerprint=fingerprint),
    ]

    if lanes[0].finding == "second_generation":
        notes.append(
            "Capture lane fired second_generation. Geometry of the *depicted* room "
            "is not trusted until a first-generation file exists."
        )

    return FlagshipReport(source=label, lanes=lanes, notes=notes)
