"""Gravity Check - Physics-first image analysis package."""

from .spatial_measurement import (
    spatial_report,
    estimate_relative_scale,
    check_depth_ordering,
    estimate_vanishing_point,
    check_perspective_consistency,
)
from .shadow_direction import analyze_shadow_consistency, shadow_vector_from_points
from .engine import run_gravity_check

__all__ = [
    "spatial_report",
    "estimate_relative_scale",
    "check_depth_ordering",
    "estimate_vanishing_point",
    "check_perspective_consistency",
    "analyze_shadow_consistency",
    "shadow_vector_from_points",
    "run_gravity_check",
]
