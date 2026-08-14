"""
Gravity Check - Reflections Module

Goal:
Check whether reflections in the image are geometrically plausible.

This module is physics-first. It does not look at raw pixels.
It reasons about measured reflection relationships:

1. Does the reflection appear roughly where geometry says it should?
2. Is the reflected content consistent in orientation and scale?
3. Are there impossible reflections (wrong side, wrong size, missing when expected)?

Common AI failure modes this targets:
- Reflections that don't match the object
- Reflections on the wrong side of a surface
- Reflections with incorrect scale or perspective
- Completely missing reflections on clearly reflective surfaces
"""

from typing import List, Tuple, Optional, Dict, Any
import math


def check_reflection_presence(
    expected_reflections: int,
    observed_reflections: int
) -> Dict[str, Any]:
    """
    Simple presence check.

    If a surface is clearly reflective and an object is in position
    to cast a reflection, we expect to see one.
    """
    if expected_reflections == 0:
        return {
            "consistent": True,
            "confidence": 0.4,
            "explanation": "No reflections were expected in this scene."
        }

    if observed_reflections >= expected_reflections:
        return {
            "consistent": True,
            "confidence": 0.65,
            "explanation": (
                f"Observed {observed_reflections} reflection(s), "
                f"expected at least {expected_reflections}."
            )
        }

    return {
        "consistent": False,
        "confidence": 0.75,
        "explanation": (
            f"Missing reflections: expected {expected_reflections}, "
            f"observed only {observed_reflections}. "
            "Reflective surfaces without expected reflections are suspicious."
        )
    }


def check_reflection_alignment(
    object_centers: List[Tuple[float, float]],
    reflection_centers: List[Tuple[float, float]],
    mirror_line_y: Optional[float] = None,
    tolerance_px: float = 40.0
) -> Dict[str, Any]:
    """
    Check whether reflections sit in roughly the correct place
    relative to their objects across a horizontal mirror line
    (e.g. water, glossy floor, table surface).

    For a horizontal mirror at mirror_line_y:
    reflection_y should be approximately 2 * mirror_line_y - object_y
    """
    if not object_centers or not reflection_centers:
        return {
            "consistent": None,
            "confidence": 0.0,
            "explanation": "Need both object and reflection centers to check alignment."
        }

    if mirror_line_y is None:
        return {
            "consistent": None,
            "confidence": 0.0,
            "explanation": "No mirror line provided; cannot verify reflection placement."
        }

    # Pair each object with the nearest reflection (simple greedy matching)
    used = set()
    deviations = []

    for ox, oy in object_centers:
        expected_ry = 2 * mirror_line_y - oy
        best_dist = float("inf")
        best_idx = None

        for i, (rx, ry) in enumerate(reflection_centers):
            if i in used:
                continue
            dist = math.hypot(rx - ox, ry - expected_ry)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is not None:
            used.add(best_idx)
            deviations.append(best_dist)

    if not deviations:
        return {
            "consistent": False,
            "confidence": 0.6,
            "explanation": "Could not pair objects with reflections."
        }

    avg_dev = sum(deviations) / len(deviations)
    max_dev = max(deviations)
    consistent = max_dev <= tolerance_px
    confidence = max(0.0, 1.0 - (avg_dev / (tolerance_px * 2.5)))

    if consistent:
        explanation = (
            f"Reflections are reasonably aligned with their objects. "
            f"Average deviation {avg_dev:.1f}px (max {max_dev:.1f}px)."
        )
    else:
        explanation = (
            f"Reflection placement is inconsistent. "
            f"Average deviation {avg_dev:.1f}px (max {max_dev:.1f}px) "
            f"exceeds {tolerance_px}px tolerance."
        )

    return {
        "consistent": consistent,
        "confidence": round(confidence, 3),
        "average_deviation_px": round(avg_dev, 1),
        "max_deviation_px": round(max_dev, 1),
        "explanation": explanation
    }


def check_reflection_scale(
    object_heights_px: List[float],
    reflection_heights_px: List[float],
    scale_tolerance: float = 0.35
) -> Dict[str, Any]:
    """
    Reflections should generally be similar in scale to the objects
    (or slightly smaller depending on viewing angle).
    Extreme scale mismatches are suspicious.
    """
    if len(object_heights_px) != len(reflection_heights_px) or not object_heights_px:
        return {
            "consistent": None,
            "confidence": 0.0,
            "explanation": "Need matching object and reflection height lists."
        }

    ratios = []
    for oh, rh in zip(object_heights_px, reflection_heights_px):
        if oh <= 0:
            continue
        ratios.append(rh / oh)

    if not ratios:
        return {
            "consistent": None,
            "confidence": 0.0,
            "explanation": "No valid height ratios could be computed."
        }

    avg_ratio = sum(ratios) / len(ratios)
    # Ideal ratio is near 1.0 (or a bit less). Large deviations are bad.
    deviation = abs(avg_ratio - 1.0)
    consistent = deviation <= scale_tolerance
    confidence = max(0.0, 1.0 - (deviation / 0.8))

    if consistent:
        explanation = (
            f"Reflection scale is plausible (average ratio {avg_ratio:.2f})."
        )
    else:
        explanation = (
            f"Reflection scale is inconsistent (average ratio {avg_ratio:.2f}). "
            "Reflections that are much larger or smaller than their objects "
            "are a common generation artifact."
        )

    return {
        "consistent": consistent,
        "confidence": round(confidence, 3),
        "average_scale_ratio": round(avg_ratio, 3),
        "explanation": explanation
    }


def analyze_reflections(
    expected_reflections: int = 0,
    observed_reflections: int = 0,
    object_centers: Optional[List[Tuple[float, float]]] = None,
    reflection_centers: Optional[List[Tuple[float, float]]] = None,
    mirror_line_y: Optional[float] = None,
    object_heights_px: Optional[List[float]] = None,
    reflection_heights_px: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Full reflections pass. Runs whichever sub-checks have data.
    """
    results: Dict[str, Any] = {
        "module": "reflections",
        "presence_check": None,
        "alignment_check": None,
        "scale_check": None,
        "consistent": None,
        "confidence": 0.0,
        "explanation": ""
    }

    evidence = []  # (score 0/1, confidence, severity)

    # Presence
    presence = check_reflection_presence(expected_reflections, observed_reflections)
    results["presence_check"] = presence
    if presence["consistent"] is not None:
        evidence.append((
            1.0 if presence["consistent"] else 0.0,
            presence["confidence"],
            0.60
        ))

    # Alignment
    if object_centers and reflection_centers:
        alignment = check_reflection_alignment(
            object_centers, reflection_centers, mirror_line_y
        )
        results["alignment_check"] = alignment
        if alignment["consistent"] is not None:
            evidence.append((
                1.0 if alignment["consistent"] else 0.0,
                alignment["confidence"],
                0.75
            ))

    # Scale
    if object_heights_px and reflection_heights_px:
        scale = check_reflection_scale(object_heights_px, reflection_heights_px)
        results["scale_check"] = scale
        if scale["consistent"] is not None:
            evidence.append((
                1.0 if scale["consistent"] else 0.0,
                scale["confidence"],
                0.55
            ))

    if not evidence:
        results["explanation"] = "Insufficient reflection measurements to run checks."
        results["confidence"] = 0.0
        results["consistent"] = None
        return results

    total_weight = sum(conf * sev for _, conf, sev in evidence)
    if total_weight > 0:
        weighted = sum(score * conf * sev for score, conf, sev in evidence) / total_weight
    else:
        weighted = 0.5

    results["consistent"] = weighted >= 0.55
    results["confidence"] = round(min(1.0, total_weight / 1.3), 3)

    if results["consistent"]:
        results["explanation"] = (
            "Reflections are geometrically plausible "
            f"(confidence {results['confidence']:.2f})."
        )
    else:
        results["explanation"] = (
            "Reflection geometry shows inconsistencies "
            f"(confidence {results['confidence']:.2f})."
        )

    return results
