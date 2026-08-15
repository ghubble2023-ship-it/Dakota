"""
Geometric feature vector for Gravity Check.

Pure classical measurements → fixed-length vector.
Used by optional learned/adversarial scoring head.
Every dimension is explainable.
"""

from typing import Dict, Any, List, Optional
import numpy as np


FEATURE_NAMES = [
    "n_objects",
    "n_shadows",
    "mean_object_conf",
    "mean_shadow_conf",
    "mean_height_px_norm",
    "height_std_norm",
    "depth_order_conflict",
    "shadow_inconsistency",
    "lighting_inconsistency",
    "reflection_inconsistency",
    "glasses_artifact",
    "edge_bleeding_artifact",
    "mean_bright_side_angle_abs",
    "brightness_std",
    "observed_reflections",
    "expected_reflections",
    "reflection_ratio",
    "primary_frame",
    "ghost_frame",
    "lens_refl_obs",
    "edge_bleed_score",
    "mean_sharpness_norm",
    "base_score",
]


def _flag(flags: List[str], name: str) -> float:
    return 1.0 if name in flags else 0.0


def bundle_and_report_to_vector(
    bundle_notes: Optional[Dict[str, Any]] = None,
    report: Optional[Dict[str, Any]] = None,
    engine_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    report = report or {}
    kw = engine_kwargs or {}
    flags = report.get("flags") or []
    scoring = report.get("scoring") or {}

    heights = kw.get("object_heights_px") or []
    img_h = float(kw.get("image_height") or 1.0)
    n_obj = float(len(heights))
    n_sh = float(len(kw.get("shadow_vectors") or []))

    mean_h = float(np.mean(heights) / img_h) if heights else 0.0
    std_h = float(np.std(heights) / img_h) if len(heights) > 1 else 0.0

    mean_obj_conf = 0.5
    mean_sh_conf = 0.5

    bright_angles = kw.get("object_bright_side_angles") or []
    brightness = kw.get("object_brightness") or []
    mean_angle_abs = float(np.mean(np.abs(bright_angles))) if bright_angles else 0.0
    bright_std = float(np.std(brightness)) if len(brightness) > 1 else 0.0

    obs_r = float(kw.get("observed_reflections") or 0)
    exp_r = float(kw.get("expected_reflections") or 0)
    refl_ratio = obs_r / exp_r if exp_r > 0 else (1.0 if obs_r == 0 else 2.0)

    primary = 1.0 if kw.get("primary_frame_detected") else 0.0
    ghost = 1.0 if kw.get("secondary_ghost_detected") else 0.0
    lens_obs = float(kw.get("observed_lens_reflections") or 0)

    bleed = float(kw.get("edge_bleeding_severity") or 0.0)
    sharp = kw.get("sharpness_values") or []
    mean_sharp = float(np.mean(sharp) / 255.0) if sharp else 0.0

    base = float(report.get("score") if report.get("score") is not None else scoring.get("score") or 0.5)

    vec = np.array([
        n_obj, n_sh, mean_obj_conf, mean_sh_conf, mean_h, std_h,
        _flag(flags, "depth_ordering_conflict"),
        _flag(flags, "shadow_inconsistency"),
        _flag(flags, "lighting_inconsistency"),
        _flag(flags, "reflection_inconsistency"),
        _flag(flags, "glasses_artifact"),
        _flag(flags, "edge_bleeding_artifact"),
        mean_angle_abs / 180.0,
        bright_std / 255.0 if bright_std else 0.0,
        obs_r, exp_r, refl_ratio, primary, ghost, lens_obs,
        bleed, mean_sharp, base,
    ], dtype=np.float64)
    assert len(vec) == len(FEATURE_NAMES)
    return vec


def vector_to_dict(vec: np.ndarray) -> Dict[str, float]:
    return {n: float(v) for n, v in zip(FEATURE_NAMES, vec)}
