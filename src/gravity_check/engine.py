"""
Gravity Check Engine

Orchestrates all geometric modules and produces a unified
weighted-evidence report.

Core modules (v0.7.0):
1. Spatial Measurement
2. Shadow Direction
3. Lighting Geometry
4. Reflections
5. Glasses & Frame Artifacts
6. Edge & Bleeding
"""

from typing import List, Tuple, Optional, Dict, Any
from .spatial_measurement import spatial_report
from .shadow_direction import analyze_shadow_consistency
from .lighting_geometry import analyze_lighting_geometry
from .reflections import analyze_reflections
from .glasses_artifacts import analyze_glasses_artifacts
from .edge_bleeding import analyze_edge_bleeding
from .scoring import build_score


def run_gravity_check(
    # Spatial
    object_heights_px: List[float],
    object_bottoms_y: List[float],
    image_height: float,
    assumed_real_heights_m: Optional[List[float]] = None,
    perspective_lines: Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = None,

    # Shadow
    shadow_vectors: Optional[List[Tuple[float, float]]] = None,
    shadow_lengths: Optional[List[float]] = None,

    # Lighting
    object_bright_side_angles: Optional[List[float]] = None,
    object_distances_proxy: Optional[List[float]] = None,
    object_brightness: Optional[List[float]] = None,

    # Reflections
    expected_reflections: int = 0,
    observed_reflections: int = 0,
    reflection_object_centers: Optional[List[Tuple[float, float]]] = None,
    reflection_centers: Optional[List[Tuple[float, float]]] = None,
    mirror_line_y: Optional[float] = None,
    reflection_heights_px: Optional[List[float]] = None,

    # Glasses
    primary_frame_detected: bool = False,
    secondary_ghost_detected: bool = False,
    ghost_offset_px: Optional[float] = None,
    frame_hair_overlap: bool = False,
    frame_skin_overlap: bool = False,
    bleeding_severity: float = 0.0,
    expected_lens_reflections: int = 0,
    observed_lens_reflections: int = 0,
    reflections_consistent_with_light: Optional[bool] = None,

    # Edge & Bleeding
    edge_bleeding_detected: bool = False,
    edge_bleeding_severity: float = 0.0,
    edge_affected_regions: Optional[List[str]] = None,
    sharpness_values: Optional[List[float]] = None,
    sharpness_depth_proxies: Optional[List[float]] = None,
    fringing_detected: bool = False,
    fringing_severity: float = 0.0,
    fringing_has_optical_explanation: bool = False,
) -> Dict[str, Any]:

    report: Dict[str, Any] = {
        "engine": "Gravity Check",
        "version": "0.7.0",
        "modules_run": [],
        "flags": [],
        "modules": {}
    }

    # 1. Spatial
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

    # 2. Shadow
    light_angle = None
    if shadow_vectors:
        shadow = analyze_shadow_consistency(shadow_vectors, shadow_lengths)
        report["modules"]["shadow_direction"] = shadow
        report["modules_run"].append("shadow_direction")
        if not shadow.get("consistent", True):
            report["flags"].append("shadow_inconsistency")
        if shadow.get("average_angle") is not None:
            light_angle = (shadow["average_angle"] + 180) % 360

    # 3. Lighting
    if object_bright_side_angles is not None or (
        object_distances_proxy is not None and object_brightness is not None
    ):
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

    # 4. Reflections
    if (expected_reflections > 0 or observed_reflections > 0
            or reflection_centers is not None or reflection_heights_px is not None):
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

    # 5. Glasses
    if (primary_frame_detected or secondary_ghost_detected
            or frame_hair_overlap or frame_skin_overlap
            or observed_lens_reflections > 0 or expected_lens_reflections > 0):
        glasses = analyze_glasses_artifacts(
            primary_frame_detected=primary_frame_detected,
            secondary_ghost_detected=secondary_ghost_detected,
            ghost_offset_px=ghost_offset_px,
            frame_hair_overlap=frame_hair_overlap,
            frame_skin_overlap=frame_skin_overlap,
            bleeding_severity=bleeding_severity,
            expected_lens_reflections=expected_lens_reflections,
            observed_lens_reflections=observed_lens_reflections,
            reflections_consistent_with_light=reflections_consistent_with_light,
        )
        report["modules"]["glasses_artifacts"] = glasses
        report["modules_run"].append("glasses_artifacts")
        if glasses.get("consistent") is False:
            report["flags"].append("glasses_artifact")

    # 6. Edge & Bleeding
    if (edge_bleeding_detected or sharpness_values is not None
            or fringing_detected):
        edges = analyze_edge_bleeding(
            bleeding_detected=edge_bleeding_detected,
            bleeding_severity=edge_bleeding_severity,
            affected_regions=edge_affected_regions,
            sharpness_values=sharpness_values,
            depth_proxies=sharpness_depth_proxies,
            fringing_detected=fringing_detected,
            fringing_severity=fringing_severity,
            has_optical_explanation=fringing_has_optical_explanation,
        )
        report["modules"]["edge_bleeding"] = edges
        report["modules_run"].append("edge_bleeding")
        if edges.get("consistent") is False:
            report["flags"].append("edge_bleeding_artifact")

    # Scoring
    scoring = build_score(report)
    report["scoring"] = scoring
    score = scoring["score"]
    interpretation = scoring["interpretation"]

    if score >= 0.60:
        summary = (f"Gravity Check: mostly consistent. Score {score:.2f}. "
                   f"{interpretation} Modules: {', '.join(report['modules_run'])}.")
    elif score >= 0.45:
        summary = (f"Gravity Check: inconclusive. Score {score:.2f}. "
                   f"{interpretation} Modules: {', '.join(report['modules_run'])}.")
    else:
        summary = (f"Gravity Check: inconsistencies detected. Score {score:.2f}. "
                   f"{interpretation} Flags: {', '.join(report['flags']) if report['flags'] else 'none'}.")

    report["summary"] = summary
    report["score"] = score
    return report
