"""
Gravity Check - Lighting Geometry Module

Goal:
Check whether the brightness behavior in the scene is consistent
with the light direction established by the Shadow module.

This module is physics-first. It does not look at raw pixels.
It reasons about measured brightness relationships and light direction.

Core checks:
1. Do the brighter sides of objects face the expected light direction?
2. Is there a plausible brightness falloff with distance?
3. Are there strong contradictions (bright side facing away from the light,
   or flat lighting that ignores the claimed light direction)?
"""

from typing import List, Tuple, Optional, Dict, Any
import math


def _angle_difference(a: float, b: float) -> float:
    """Smallest difference between two angles in degrees."""
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def _vector_to_angle(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dy, dx))


def check_bright_side_alignment(
    light_angle: float,
    object_bright_side_angles: List[float],
    tolerance_degrees: float = 50.0
) -> Dict[str, Any]:
    """
    Check whether the bright sides of objects roughly face the light.

    Parameters
    ----------
    light_angle : float
        Direction the light is coming FROM (degrees), usually from shadow analysis.
        Convention: angle of the vector pointing toward the light source.

    object_bright_side_angles : list of float
        For each object, the direction of its brighter side
        (angle of the outward normal of the bright face).

    tolerance_degrees : float
        How far the bright side is allowed to deviate from the light direction.
    """
    if not object_bright_side_angles:
        return {
            "consistent": None,
            "confidence": 0.0,
            "explanation": "No bright-side measurements provided.",
            "deviations": []
        }

    deviations = [_angle_difference(light_angle, a) for a in object_bright_side_angles]
    max_dev = max(deviations)
    avg_dev = sum(deviations) / len(deviations)

    # Bright side should generally face the light (low deviation)
    consistent = max_dev <= tolerance_degrees

    # Confidence drops as average deviation grows
    confidence = max(0.0, 1.0 - (avg_dev / 90.0))
    confidence = round(confidence, 3)

    if consistent:
        explanation = (
            f"Bright sides are reasonably aligned with the light direction. "
            f"Average deviation {avg_dev:.1f}° (max {max_dev:.1f}°)."
        )
    else:
        explanation = (
            f"Bright sides conflict with the claimed light direction. "
            f"Average deviation {avg_dev:.1f}° (max {max_dev:.1f}° exceeds "
            f"{tolerance_degrees}° tolerance)."
        )

    return {
        "consistent": consistent,
        "confidence": confidence,
        "average_deviation": avg_dev,
        "max_deviation": max_dev,
        "deviations": deviations,
        "explanation": explanation
    }


def check_brightness_falloff(
    object_distances_proxy: List[float],
    object_brightness: List[float],
    falloff_tolerance: float = 0.45
) -> Dict[str, Any]:
    """
    Soft check for inverse-square-style falloff.

    object_distances_proxy : larger value = farther from light (or from camera if light is camera-relative)
    object_brightness      : relative brightness measurements (higher = brighter)

    We only look for strong contradictions (far objects brighter than near ones
    by a large margin). Mild variation is allowed.
    """
    if len(object_distances_proxy) != len(object_brightness) or len(object_brightness) < 2:
        return {
            "consistent": None,
            "confidence": 0.0,
            "explanation": "Need at least two objects with distance and brightness values."
        }

    # Pairwise: farther object should not be dramatically brighter
    violations = 0
    total_pairs = 0

    for i in range(len(object_brightness)):
        for j in range(i + 1, len(object_brightness)):
            total_pairs += 1
            dist_i = object_distances_proxy[i]
            dist_j = object_distances_proxy[j]
            bright_i = object_brightness[i]
            bright_j = object_brightness[j]

            if dist_i <= 0 or dist_j <= 0:
                continue

            # If j is significantly farther than i, it should not be much brighter
            if dist_j > dist_i * 1.4 and bright_j > bright_i * (1.0 + falloff_tolerance):
                violations += 1
            elif dist_i > dist_j * 1.4 and bright_i > bright_j * (1.0 + falloff_tolerance):
                violations += 1

    if total_pairs == 0:
        return {
            "consistent": None,
            "confidence": 0.0,
            "explanation": "Could not form valid distance/brightness pairs."
        }

    violation_ratio = violations / total_pairs
    consistent = violation_ratio <= 0.25
    confidence = max(0.0, 1.0 - violation_ratio)

    if consistent:
        explanation = (
            f"Brightness falloff is plausible. "
            f"Violation ratio {violation_ratio:.2f}."
        )
    else:
        explanation = (
            f"Brightness falloff shows contradictions. "
            f"Farther objects are frequently brighter than nearer ones "
            f"(violation ratio {violation_ratio:.2f})."
        )

    return {
        "consistent": consistent,
        "confidence": round(confidence, 3),
        "violation_ratio": round(violation_ratio, 3),
        "explanation": explanation
    }


def analyze_lighting_geometry(
    light_angle: Optional[float] = None,
    object_bright_side_angles: Optional[List[float]] = None,
    object_distances_proxy: Optional[List[float]] = None,
    object_brightness: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Full lighting geometry pass.

    All inputs are optional. The module runs whatever checks it has data for
    and returns an overall assessment.
    """
    results = {
        "module": "lighting_geometry",
        "bright_side_check": None,
        "falloff_check": None,
        "consistent": None,
        "confidence": 0.0,
        "explanation": ""
    }

    evidence_scores = []

    # Bright-side alignment
    if light_angle is not None and object_bright_side_angles:
        bright = check_bright_side_alignment(light_angle, object_bright_side_angles)
        results["bright_side_check"] = bright
        if bright["consistent"] is not None:
            evidence_scores.append(
                (1.0 if bright["consistent"] else 0.0, bright["confidence"], 0.75)
            )

    # Falloff
    if object_distances_proxy and object_brightness:
        falloff = check_brightness_falloff(object_distances_proxy, object_brightness)
        results["falloff_check"] = falloff
        if falloff["consistent"] is not None:
            evidence_scores.append(
                (1.0 if falloff["consistent"] else 0.0, falloff["confidence"], 0.55)
            )

    if not evidence_scores:
        results["explanation"] = "Insufficient lighting measurements to run checks."
        results["confidence"] = 0.0
        results["consistent"] = None
        return results

    # Weighted combination of the sub-checks
    total_weight = sum(conf * sev for _, conf, sev in evidence_scores)
    if total_weight > 0:
        weighted = sum(score * conf * sev for score, conf, sev in evidence_scores) / total_weight
    else:
        weighted = 0.5

    results["consistent"] = weighted >= 0.55
    results["confidence"] = round(min(1.0, total_weight / 1.2), 3)

    if results["consistent"]:
        results["explanation"] = (
            "Lighting geometry is broadly consistent with the claimed light direction "
            f"and distance relationships (confidence {results['confidence']:.2f})."
        )
    else:
        results["explanation"] = (
            "Lighting geometry shows conflicts with expected light direction "
            f"or falloff (confidence {results['confidence']:.2f})."
        )

    return results
