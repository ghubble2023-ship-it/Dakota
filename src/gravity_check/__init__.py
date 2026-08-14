"""Gravity Check - Physics-first image analysis package."""

from .spatial_measurement import spatial_report, estimate_relative_scale, check_depth_ordering
from .shadow_direction import analyze_shadow_consistency

__all__ = [
    "spatial_report",
    "estimate_relative_scale",
    "check_depth_ordering",
    "analyze_shadow_consistency",
]
