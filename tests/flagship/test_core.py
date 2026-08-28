#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from flagship.lanes.sensor import DrunetResidual, analyze_sensor
from flagship.pipeline import run_flagship


def synthetic_gradient():
    y = np.linspace(0.15, 0.85, 128)
    x = np.linspace(0.2, 0.7, 128)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    g = 0.35 + 0.4 * xx + 0.15 * yy
    return np.clip(np.stack([g, g * 0.95, g * 0.85], axis=-1), 0, 1)


def synthetic_lcd():
    h, w = 160, 160
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    grid = 0.55 + 0.45 * (((xx % 4) == 0) | ((yy % 4) == 0)).astype(np.float64)
    return np.stack([grid, grid, grid], axis=-1)


def test_core():
    g = run_flagship(synthetic_gradient())
    assert g.headline != "second_generation"
    lcd = run_flagship(synthetic_lcd())
    assert lcd.headline == "second_generation"
    hole = analyze_sensor(synthetic_gradient(), backend=DrunetResidual())
    assert hole.finding == "insufficient"
    print("PASS", json.dumps({"gradient": g.headline, "lcd": lcd.headline}))


if __name__ == "__main__":
    test_core()
