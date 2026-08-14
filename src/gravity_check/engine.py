"""
Gravity Check Engine

Orchestrates modules in order and produces a unified report
with weighted evidence scoring.
"""

from typing import List, Tuple, Optional, Dict, Any
from .spatial_measurement import spatial_report
from .shadow_direction import analyze_shadow_consistency
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
) -> Dict[str, Any]:
    """
    Run the full Gravity Check pipeline.

    Returns a unified report with per-module results and a weighted score.
    """

    report: Dict[str, Any] = {
        "engine": "Gravity Check",
        "version": "0.3.0",
        "modules_run": [],
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

    if spatial.get("depth_ordering", {}).get("consistent") is False:
        report["flags"].append("depth_ordering_conflict")

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
            report["flags"].append("shadow_inconsistency")

    # -------------------------------------------------
    # WEIGHTED EVIDENCE SCORING
    # -------------------------------------------------
    scoring = build_score(report)
    report["scoring"] = scoring

    # -------------------------------------------------
    # Final summary
    # -------------------------------------------------
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


if __name__ == "__main__":
    demo = run_gravity_check(
        object_heights_px=[420.0, 280.0],
        object_bottoms_y=[310.0, 520.0],
        image_height=720.0,
        shadow_vectors=[(0.9, -0.4), (0.85, -0.5), (0.88, -0.45)]
    )
    print(demo["summary"])
    print("Score:", demo["score"])
