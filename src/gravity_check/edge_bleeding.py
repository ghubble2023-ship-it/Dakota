"""
Gravity Check - Edge & Bleeding Module

Goal:
Detect unnatural edge behavior that frequently appears in AI-generated images:

1. Edge bleeding / haloing (color or structure leaking across boundaries)
2. Overly soft or overly hard edges that don't match the optical conditions
3. Inconsistent edge sharpness across similar depths
4. Color fringing that has no optical explanation

This module is measurement-driven and explainable.
It expects structured observations rather than raw pixel access.
"""

from typing import List, Optional, Dict, Any


def check_edge_bleeding(
    bleeding_detected: bool,
    bleeding_severity: float = 0.0,
    affected_regions: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Check for color or structure bleeding across object boundaries.

    bleeding_severity: 0.0 (none) → 1.0 (severe)
    affected_regions: e.g. ["hair_forehead", "shoulder_background", "glasses_skin"]
    """
    if not bleeding_detected or bleeding_severity < 0.15:
        return {
            "consistent": True,
            "confidence": 0.60,
            "explanation": "No significant edge bleeding detected."
        }

    regions = ", ".join(affected_regions) if affected_regions else "unspecified regions"

    if bleeding_severity < 0.40:
        return {
            "consistent": True,
            "confidence": 0.45,
            "explanation": (
                f"Mild edge bleeding noted in {regions} "
                f"(severity {bleeding_severity:.2f}). "
                "May be compression or minor anti-aliasing."
            )
        }

    return {
        "consistent": False,
        "confidence": min(0.90, 0.55 + bleeding_severity * 0.35),
        "explanation": (
            f"Significant edge bleeding detected in {regions} "
            f"(severity {bleeding_severity:.2f}). "
            "Clean optical boundaries should not leak structure or color this way."
        )
    }


def check_edge_sharpness_consistency(
    sharpness_values: List[float],
    depth_proxies: Optional[List[float]] = None,
    max_allowed_variation: float = 0.45
) -> Dict[str, Any]:
    """
    Objects at similar depths should have roughly similar edge sharpness
    under the same optical conditions.

    sharpness_values: higher = sharper
    depth_proxies: optional relative depth (higher = farther)
    """
    if len(sharpness_values) < 2:
        return {
            "consistent": None,
            "confidence": 0.0,
            "explanation": "Need at least two sharpness measurements."
        }

    mean_sharp = sum(sharpness_values) / len(sharpness_values)
    if mean_sharp <= 0:
        return {
            "consistent": None,
            "confidence": 0.0,
            "explanation": "Invalid sharpness values."
        }

    # Coefficient of variation
    variance = sum((s - mean_sharp) ** 2 for s in sharpness_values) / len(sharpness_values)
    std = variance ** 0.5
    cv = std / mean_sharp

    consistent = cv <= max_allowed_variation
    confidence = max(0.0, 1.0 - (cv / 0.9))

    if consistent:
        explanation = (
            f"Edge sharpness is reasonably consistent across measured regions "
            f"(CV {cv:.2f})."
        )
    else:
        explanation = (
            f"Edge sharpness varies more than expected (CV {cv:.2f}). "
            "Large sharpness differences at similar depths are often "
            "a generation artifact."
        )

    return {
        "consistent": consistent,
        "confidence": round(confidence, 3),
        "coefficient_of_variation": round(cv, 3),
        "explanation": explanation
    }


def check_color_fringing(
    fringing_detected: bool,
    fringing_severity: float = 0.0,
    has_optical_explanation: bool = False
) -> Dict[str, Any]:
    """
    Chromatic aberration / color fringing check.

    Real lenses can produce fringing, but it should be consistent
    with optical centers and usually mild. Random or heavy fringing
    with no optical explanation is suspicious.
    """
    if not fringing_detected or fringing_severity < 0.20:
        return {
            "consistent": True,
            "confidence": 0.55,
            "explanation": "No significant color fringing detected."
        }

    if has_optical_explanation and fringing_severity < 0.55:
        return {
            "consistent": True,
            "confidence": 0.50,
            "explanation": (
                f"Color fringing present (severity {fringing_severity:.2f}) "
                "but has a plausible optical explanation."
            )
        }

    return {
        "consistent": False,
        "confidence": min(0.85, 0.50 + fringing_severity * 0.4),
        "explanation": (
            f"Color fringing detected (severity {fringing_severity:.2f}) "
            "without clear optical justification. "
            "This pattern is common in generated imagery."
        )
    }


def analyze_edge_bleeding(
    bleeding_detected: bool = False,
    bleeding_severity: float = 0.0,
    affected_regions: Optional[List[str]] = None,
    sharpness_values: Optional[List[float]] = None,
    depth_proxies: Optional[List[float]] = None,
    fringing_detected: bool = False,
    fringing_severity: float = 0.0,
    has_optical_explanation: bool = False,
) -> Dict[str, Any]:
    """
    Full edge & bleeding pass.
    """
    results: Dict[str, Any] = {
        "module": "edge_bleeding",
        "bleeding_check": None,
        "sharpness_check": None,
        "fringing_check": None,
        "consistent": None,
        "confidence": 0.0,
        "explanation": ""
    }

    evidence = []

    # Bleeding
    bleeding = check_edge_bleeding(
        bleeding_detected, bleeding_severity, affected_regions
    )
    results["bleeding_check"] = bleeding
    if bleeding["consistent"] is not None:
        evidence.append((
            1.0 if bleeding["consistent"] else 0.0,
            bleeding["confidence"],
            0.75
        ))

    # Sharpness consistency
    if sharpness_values:
        sharp = check_edge_sharpness_consistency(sharpness_values, depth_proxies)
        results["sharpness_check"] = sharp
        if sharp["consistent"] is not None:
            evidence.append((
                1.0 if sharp["consistent"] else 0.0,
                sharp["confidence"],
                0.55
            ))

    # Color fringing
    fringe = check_color_fringing(
        fringing_detected, fringing_severity, has_optical_explanation
    )
    results["fringing_check"] = fringe
    if fringe["consistent"] is not None:
        evidence.append((
            1.0 if fringe["consistent"] else 0.0,
            fringe["confidence"],
            0.60
        ))

    if not evidence:
        results["explanation"] = "No edge/bleeding measurements provided."
        results["confidence"] = 0.0
        results["consistent"] = None
        return results

    total_weight = sum(conf * sev for _, conf, sev in evidence)
    if total_weight > 0:
        weighted = sum(s * conf * sev for s, conf, sev in evidence) / total_weight
    else:
        weighted = 0.5

    results["consistent"] = weighted >= 0.55
    results["confidence"] = round(min(1.0, total_weight / 1.3), 3)

    if results["consistent"]:
        results["explanation"] = (
            "Edge behavior is optically plausible "
            f"(confidence {results['confidence']:.2f})."
        )
    else:
        results["explanation"] = (
            "Edge or bleeding artifacts detected "
            f"(confidence {results['confidence']:.2f})."
        )

    return results
