"""
Gravity Check Engine

This is the orchestrator. It runs the modules in the correct order
and produces a single unified report.

Current status:
- Spatial Measurement  → implemented
- Shadow Direction     → implemented (with circular statistics)
- Future modules       → Lighting, Reflections, Glasses, Edges (placeholders)

The engine does NOT yet extract measurements from raw pixels.
It expects structured measurements to be supplied.
That extraction layer is the next major piece of work.
"""

from typing import List, Tuple, Optional, Dict, Any
from .spatial_measurement import spatial_report
from .shadow_direction import analyze_shadow_consistency


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

    # --- Future module inputs (placeholders) ---
    # lighting_data=None,
    # reflection_data=None,
    # glasses_data=None,
    # edge_data=None,
) -> Dict[str, Any]:
    """
    Run the full Gravity Check pipeline in the correct order.

    Returns a unified report with per-module results and an overall assessment.
    """

    report: Dict[str, Any] = {
        "engine": "Gravity Check",
        "version": "0.2.0",
        "modules_run": [],
        "overall_consistent": True,
        "overall_confidence": 1.0,
        "flags": [],
        "modules": {}
    }

    # -------------------------------------------------
    # 1. SPATIAL MEASUREMENT (required first)
    # -------------------------------------------------
    spatial = spatial_report(
        object_heights_px=object_heights_px,
        object_bottoms_y=object_bottoms_y,
        image_height=image_height,
        assumed_real_heights_m=assumed_real_heights_m,
        perspective_lines=perspective_lines
    )
    report["modules"]["spatial_measurement"] = spatial
    report["modules_run"].append("spatial_measurement")

    # Soft impact on overall score
    if spatial.get("depth_ordering", {}).get("consistent") is False:
        report["overall_consistent"] = False
        report["flags"].append("depth_ordering_conflict")
        report["overall_confidence"] *= 0.7

    # -------------------------------------------------
    # 2. SHADOW DIRECTION
    # -------------------------------------------------
    if shadow_vectors:
        shadow = analyze_shadow_consistency(
            shadow_vectors=shadow_vectors,
            shadow_lengths=shadow_lengths
        )
        report["modules"]["shadow_direction"] = shadow
        report["modules_run"].append("shadow_direction")

        if not shadow.get("consistent", True):
            report["overall_consistent"] = False
            report["flags"].append("shadow_inconsistency")

        # Blend confidence
        shadow_conf = shadow.get("confidence", 0.5)
        report["overall_confidence"] *= (0.4 + 0.6 * shadow_conf)

    # -------------------------------------------------
    # 3–7. FUTURE MODULES (placeholders)
    # -------------------------------------------------
    # report["modules"]["lighting_geometry"] = ...
    # report["modules"]["reflections"] = ...
    # report["modules"]["glasses_artifacts"] = ...
    # report["modules"]["edge_bleeding"] = ...

    # -------------------------------------------------
    # Final summary
    # -------------------------------------------------
    report["overall_confidence"] = round(max(0.0, min(1.0, report["overall_confidence"])), 3)

    if report["overall_consistent"]:
        summary = (
            f"Gravity Check passed. "
            f"Modules run: {', '.join(report['modules_run'])}. "
            f"Overall confidence: {report['overall_confidence']:.2f}."
        )
    else:
        summary = (
            f"Gravity Check flagged issues. "
            f"Flags: {', '.join(report['flags'])}. "
            f"Overall confidence: {report['overall_confidence']:.2f}."
        )

    report["summary"] = summary
    return report


if __name__ == "__main__":
    # Minimal demo
    demo = run_gravity_check(
        object_heights_px=[420.0, 280.0],
        object_bottoms_y=[310.0, 520.0],
        image_height=720.0,
        shadow_vectors=[(0.9, -0.4), (0.85, -0.5), (0.88, -0.45)]
    )
    print(demo["summary"])
    print("Modules run:", demo["modules_run"])
