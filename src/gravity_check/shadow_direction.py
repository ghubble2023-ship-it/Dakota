"""
Gravity Check - Shadow Direction Module

Goal:
Determine whether shadows in an image are consistent with a single light source
using robust circular statistics.

This module is pure geometry. It does not look at pixels.
It only reasons about the direction (and optionally length) of shadows
that have already been measured.
"""

from typing import List, Tuple, Optional, Dict
import math


def _vector_to_angle(dx: float, dy: float) -> float:
    """Convert a direction vector to an angle in degrees."""
    return math.degrees(math.atan2(dy, dx))


def _angular_difference(a: float, b: float) -> float:
    """Smallest absolute difference between two angles (handles wrap-around)."""
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def _circular_mean(angles: List[float], weights: Optional[List[float]] = None) -> float:
    """Circular mean of angles (degrees). Supports optional weights."""
    if weights is None:
        weights = [1.0] * len(angles)

    sin_sum = 0.0
    cos_sum = 0.0
    for angle, weight in zip(angles, weights):
        rad = math.radians(angle)
        sin_sum += weight * math.sin(rad)
        cos_sum += weight * math.cos(rad)

    return math.degrees(math.atan2(sin_sum, cos_sum))


def _circular_stats(angles: List[float], weights: Optional[List[float]] = None) -> Dict:
    """
    Compute robust circular statistics.

    Returns:
        mean_angle     : circular mean (degrees)
        R              : mean resultant length (0 to 1)
        kappa          : approximate concentration parameter
        angular_std    : circular standard deviation (degrees)
    """
    n = len(angles)
    if n == 0:
        return {"mean_angle": None, "R": 0.0, "kappa": 0.0, "angular_std": None}

    if weights is None:
        weights = [1.0] * n

    total_weight = sum(weights)
    if total_weight <= 0:
        total_weight = 1.0

    sin_sum = 0.0
    cos_sum = 0.0
    for angle, w in zip(angles, weights):
        rad = math.radians(angle)
        sin_sum += w * math.sin(rad)
        cos_sum += w * math.cos(rad)

    # Mean resultant length R (0 = uniform, 1 = perfectly concentrated)
    R = math.sqrt(sin_sum**2 + cos_sum**2) / total_weight
    R = max(0.0, min(1.0, R))

    mean_angle = math.degrees(math.atan2(sin_sum, cos_sum))

    # Approximate concentration kappa (Banerjee / Mardia approximation)
    if R < 0.53:
        kappa = 2 * R + R**3 + (5/6) * R**5
    elif R < 0.85:
        kappa = -0.4 + 1.39 * R + 0.43 / (1 - R)
    else:
        kappa = 1 / (R**3 - 4 * R**2 + 3 * R) if (R**3 - 4 * R**2 + 3 * R) != 0 else 100.0

    # Circular standard deviation in degrees
    if R > 0.0:
        angular_std = math.degrees(math.sqrt(-2.0 * math.log(R)))
    else:
        angular_std = 180.0

    return {
        "mean_angle": mean_angle,
        "R": R,
        "kappa": kappa,
        "angular_std": angular_std
    }


def analyze_shadow_consistency(
    shadow_vectors: List[Tuple[float, float]],
    tolerance_degrees: float = 25.0,
    shadow_lengths: Optional[List[float]] = None,
    length_tolerance_ratio: float = 0.55,
    min_confidence: float = 0.55
) -> Dict:
    """
    Analyze shadow direction consistency using robust circular statistics.

    Parameters
    ----------
    shadow_vectors : list of (dx, dy)
        Direction vectors of detected shadows.

    tolerance_degrees : float
        Soft angular tolerance (still reported for transparency).

    shadow_lengths : list of float, optional
        Lengths used for weighting and length-consistency check.

    length_tolerance_ratio : float
        Minimum allowed (shortest / longest) length ratio.

    min_confidence : float
        Minimum confidence (0-1) required to declare consistency.

    Returns
    -------
    dict containing:
        consistent, confidence, average_angle, max_deviation,
        angular_std, R, kappa, length_consistent, explanation, details
    """
    if not shadow_vectors:
        return {
            "consistent": False,
            "confidence": 0.0,
            "average_angle": None,
            "max_deviation": None,
            "angular_std": None,
            "R": 0.0,
            "kappa": 0.0,
            "deviations": [],
            "length_consistent": None,
            "explanation": "No shadow vectors provided.",
            "details": {}
        }

    angles = [_vector_to_angle(dx, dy) for dx, dy in shadow_vectors]

    # Single observation
    if len(shadow_vectors) == 1:
        return {
            "consistent": True,
            "confidence": 0.35,
            "average_angle": angles[0],
            "max_deviation": 0.0,
            "angular_std": 0.0,
            "R": 1.0,
            "kappa": 100.0,
            "deviations": [0.0],
            "length_consistent": None,
            "explanation": (
                "Only one shadow detected. "
                "Direction consistency cannot be fully verified."
            ),
            "details": {"shadow_count": 1}
        }

    # Weights from lengths
    weights = None
    if shadow_lengths and len(shadow_lengths) == len(shadow_vectors):
        weights = [max(l, 0.01) for l in shadow_lengths]

    stats = _circular_stats(angles, weights)
    mean_angle = stats["mean_angle"]
    R = stats["R"]
    kappa = stats["kappa"]
    angular_std = stats["angular_std"]

    deviations = [_angular_difference(a, mean_angle) for a in angles]
    max_deviation = max(deviations)

    # Confidence score (0 → 1)
    # Driven primarily by concentration R and secondarily by angular spread
    conf_from_R = R ** 1.4
    conf_from_std = max(0.0, 1.0 - (angular_std / 90.0))
    confidence = 0.65 * conf_from_R + 0.35 * conf_from_std
    confidence = max(0.0, min(1.0, confidence))

    # Length consistency
    length_consistent = None
    length_note = ""
    if shadow_lengths and len(shadow_lengths) == len(shadow_vectors):
        min_len = min(shadow_lengths)
        max_len = max(shadow_lengths)
        if max_len > 0:
            ratio = min_len / max_len
            length_consistent = ratio >= length_tolerance_ratio
            if not length_consistent:
                length_note = (
                    f" Shadow lengths also vary strongly "
                    f"(shortest is only {ratio:.0%} of longest)."
                )
                confidence *= 0.7  # penalize confidence

    direction_ok = max_deviation <= tolerance_degrees and confidence >= min_confidence
    overall_consistent = direction_ok and (length_consistent is not False)

    if overall_consistent:
        explanation = (
            f"Shadows are consistent with a single light source. "
            f"Average direction: {mean_angle:.1f}°. "
            f"Concentration R={R:.3f}, κ≈{kappa:.1f}, "
            f"angular std={angular_std:.1f}°, confidence={confidence:.2f}."
        )
    else:
        explanation = (
            f"Shadows are inconsistent with a single light source. "
            f"Average direction: {mean_angle:.1f}°. "
            f"Max deviation={max_deviation:.1f}°, "
            f"R={R:.3f}, angular std={angular_std:.1f}°, "
            f"confidence={confidence:.2f}."
        )
        explanation += length_note

    return {
        "consistent": overall_consistent,
        "confidence": round(confidence, 3),
        "average_angle": mean_angle,
        "max_deviation": max_deviation,
        "angular_std": angular_std,
        "R": round(R, 4),
        "kappa": round(kappa, 2),
        "deviations": deviations,
        "length_consistent": length_consistent,
        "explanation": explanation,
        "details": {
            "shadow_count": len(shadow_vectors),
            "tolerance_degrees": tolerance_degrees,
            "min_confidence": min_confidence,
            "angles": angles,
            "weights_used": weights is not None
        }
    }


def shadow_vector_from_points(
    start: Tuple[float, float],
    end: Tuple[float, float]
) -> Tuple[float, float]:
    """Create a direction vector from two points (contact → tip of shadow)."""
    return (end[0] - start[0], end[1] - start[1])


if __name__ == "__main__":
    test_consistent = [(0.9, -0.4), (0.85, -0.5), (0.88, -0.45)]
    print("Consistent:", analyze_shadow_consistency(test_consistent)["explanation"])

    test_inconsistent = [(0.9, -0.4), (-0.7, 0.6), (0.1, 0.9)]
    print("Inconsistent:", analyze_shadow_consistency(test_inconsistent)["explanation"])
