"""Gravity Check - Physics-first image analysis package."""

from .spatial_measurement import (
    spatial_report,
    estimate_relative_scale,
    check_depth_ordering,
    estimate_vanishing_point,
    check_perspective_consistency,
)
from .shadow_direction import analyze_shadow_consistency, shadow_vector_from_points
from .lighting_geometry import analyze_lighting_geometry
from .reflections import analyze_reflections
from .engine import run_gravity_check
from .measurement import (
    ImageMeasurements,
    ObjectMeasurement,
    ShadowMeasurement,
    PerspectiveLine,
    measure_shadow_from_points,
    measure_object_from_bbox,
    feed_to_engine,
)
from .scoring import Evidence, score_from_evidence, build_score

__all__ = [
    "spatial_report",
    "estimate_relative_scale",
    "check_depth_ordering",
    "estimate_vanishing_point",
    "check_perspective_consistency",
    "analyze_shadow_consistency",
    "shadow_vector_from_points",
    "analyze_lighting_geometry",
    "analyze_reflections",
    "run_gravity_check",
    "ImageMeasurements",
    "ObjectMeasurement",
    "ShadowMeasurement",
    "PerspectiveLine",
    "measure_shadow_from_points",
    "measure_object_from_bbox",
    "feed_to_engine",
    "Evidence",
    "score_from_evidence",
    "build_score",
]
