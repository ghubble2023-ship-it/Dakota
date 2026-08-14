"""
Gravity Check Engine

Orchestrates modules in order and produces a unified report
with weighted evidence scoring.
"""

from typing import List, Tuple, Optional, Dict, Any
from .spatial_measurement import spatial_report
from .shadow_direction import analyze_shadow_consistency
from .lighting_geometry import analyze_lighting_geometry
from .reflections import analyze_reflections
from .scoring import build_score


def run_gravity_check(
    # --- Spatial inputs ---
    object_heights_px: List[float],
    object_bottoms_y: List[float],
    image_height: float,
    assumed_real_heights_m: Optional[List[float]] = None,
    perspective_lines: Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = None,

    # --- Shadow inputs ---
    shadow_vectors: Optional[List[Tuple[float, float]]] = None,
    shadow_lengths: Optional[List[float]] = None,

    # --- Lighting inputs ---
    object_bright_side_angles: Optional[List[float]] = None,
    object_distances_proxy: Optional[List[float]] = None,
    object_brightness: Optional[List[float]] = None,

    # --- Reflection inputs ---
    expected_reflections: int = 0,
    observed_reflections: int = 0,
    reflection_object_centers: Optional[List[Tuple[float, float]]] = None,
    reflection_centers: Optional[List[Tuple[float, float]]] = None,
    mirror_line_y: Optional[float] = None,
    reflection_heights_px: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Run the full Gravity Check pipeline.
    """

    report: Dict[str, Any] = {
        "engine": "Gravity Check",
        "version": "0.5.0",
        "modules_run": [],
        "flags": [],
        "modules": {}
    }

    # 1. SPATIAL MEASUREMENT
    spatial = spatial_report(
        object_heights_px=object_heights_px,
        object_bottoms_y=object_bottoms_y,
        image_height=image_height,
        assumed_real_heights_m=assumed_real_heights_m,
        perspective_lines=perspective_lines
    )
    report["modules"]["spatial_measurement"] = spatial
    report["modules_run"].append("spatial_measurement")

    if spatial.get("depth_ordering", {}).get("consistent") is False:
        report["flags"].append("depth_ordering_conflict")

    # 2. SHADOW DIRECTION
    light_angle = None
    if shadow_vectors:
        shadow = analyze_shadow_consistency(
            shadow_vectors=shadow_vectors,
            shadow_lengths=shadow_lengths
        )
        report["modules"]["shadow_direction"] = shadow
        report["modules_run"].append("shadow_direction")

        if not shadow.get("consistent", True):
            report["flags"].append("shadow_inconsistency")

        if shadow.get("average_angle") is not None:
            light_angle = (shadow["average_angle"] + 180) % 360

    # 3. LIGHTING GEOMETRY
    has_lighting_data = (
        object_bright_side_angles is not None
        or (object_distances_proxy is not None and object_brightness is not None)
    )
    if has_lighting_data:
        lighting = analyze_lighting_geometry(
            light_angle=light_angle,
            object_bright_side_angles=object_bright_side_angles,
            object_distances_proxy=object_distances_proxy,
            object_brightness=object_brightness,
        )
        report["modules"]["lighting_geometry"] = lighting
        report["modules_run"].append("lighting_geometry")

        if lighting.get("consistent") is False:
            report["flags"].append("lighting_inconsistency")

    # 4. REFLECTIONS
    has_reflection_data = (
        expected_reflections > 0
        or observed_reflections > 0
        or reflection_centers is not None
        or reflection_heights_px is not None
    )
    if has_reflection_data:
        reflections = analyze_reflections(
            expected_reflections=expected_reflections,
            observed_reflections=observed_reflections,
            object_centers=reflection_object_centers,
            reflection_centers=reflection_centers,
            mirror_line_y=mirror_line_y,
            object_heights_px=object_heights_px if reflection_heights_px else None,
            reflection_heights_px=reflection_heights_px,
        )
        report["modules"]["reflections"] = reflections
        report["modules_run"].append("reflections")

        if reflections.get("consistent") is False:
            report["flags"].append("reflection_inconsistency")

    # WEIGHTED SCORING
    scoring = build_score(report)
    report["scoring"] = scoring

    score = scoring["score"]
    interpretation = scoring["interpretation"]

    if score >= 0.60:
        summary = (
            f"Gravity Check: mostly consistent. "
            f"Score {score:.2f}. {interpretation} "
            f"Modules: {', '.join(report['modules_run'])}."
        )
    elif score >= 0.45:
        summary = (
            f"Gravity Check: inconclusive. "
            f"Score {score:.2f}. {interpretation} "
            f"Modules: {', '.join(report['modules_run'])}."
        )
    else:
        summary = (
            f"Gravity Check: inconsistencies detected. "
            f"Score {score:.2f}. {interpretation} "
            f"Flags: {', '.join(report['flags']) if report['flags'] else 'none'}."
        )

    report["summary"] = summary
    report["score"] = score
    return report
