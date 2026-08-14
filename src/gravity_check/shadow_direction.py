"""
Gravity Check - Shadow Direction Module

Goal:
Determine whether shadows in an image are consistent with a single light source.

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
    """
    Compute the circular mean of a list of angles.
    Optional weights let longer / more reliable shadows count more.
    """
    if weights is None:
        weights = [1.0] * len(angles)

    sin_sum = 0.0
    cos_sum = 0.0
    for angle, weight in zip(angles, weights):
        rad = math.radians(angle)
        sin_sum += weight * math.sin(rad)
        cos_sum += weight * math.cos(rad)

    return math.degrees(math.atan2(sin_sum, cos_sum))


def analyze_shadow_consistency(
    shadow_vectors: List[Tuple[float, float]],
    tolerance_degrees: float = 25.0,
    shadow_lengths: Optional[List[float]] = None,
    length_tolerance_ratio: float = 0.55
) -> Dict:
    """
    Analyze a list of shadow direction vectors for consistency with a single light source.

    Parameters
    ----------
    shadow_vectors : list of (dx, dy)
        Direction vectors of detected shadows.
        Example: [(0.8, -0.6), (0.75, -0.65)]

    tolerance_degrees : float
        Maximum allowed angular deviation from the average direction.
        Default 25° is intentional — real soft shadows and surface tilt create variation,
        but large conflicts are strong evidence of generation error.

    shadow_lengths : list of float, optional
        Length of each shadow in the same order as the vectors.
        When provided, longer shadows receive higher weight in the average
        and a simple length-consistency check is also performed.

    length_tolerance_ratio : float
        If lengths are supplied, the shortest shadow should not be shorter than
        this ratio of the longest shadow (default 0.55).
        Extreme length differences under the same light can be suspicious.

    Returns
    -------
    dict with keys:
        consistent          : bool
        average_angle       : float or None
        max_deviation       : float or None
        deviations          : list of float
        length_consistent   : bool or None
        explanation         : str
        details             : dict (extra diagnostics)
    """
    if not shadow_vectors:
        return {
            "consistent": False,
            "average_angle": None,
            "max_deviation": None,
            "deviations": [],
            "length_consistent": None,
            "explanation": "No shadow vectors provided.",
            "details": {}
        }

    # --- Convert vectors to angles ---
    angles = [_vector_to_angle(dx, dy) for dx, dy in shadow_vectors]

    # --- Single shadow case ---
    if len(shadow_vectors) == 1:
        return {
            "consistent": True,
            "average_angle": angles[0],
            "max_deviation": 0.0,
            "deviations": [0.0],
            "length_consistent": None,
            "explanation": (
                "Only one shadow detected. "
                "Direction consistency cannot be fully verified with a single observation."
            ),
            "details": {"shadow_count": 1}
        }

    # --- Weighted circular mean ---
    weights = None
    if shadow_lengths and len(shadow_lengths) == len(shadow_vectors):
        # Longer shadows are usually more reliable
        weights = [max(l, 0.01) for l in shadow_lengths]

    mean_angle = _circular_mean(angles, weights)

    # --- Angular deviations ---
    deviations = [_angular_difference(a, mean_angle) for a in angles]
    max_deviation = max(deviations)
    direction_consistent = max_deviation <= tolerance_degrees

    # --- Optional length consistency ---
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

    # --- Final decision ---
    overall_consistent = direction_consistent and (length_consistent is not False)

    if overall_consistent:
        explanation = (
            f"Shadows are consistent with a single light source. "
            f"Average direction: {mean_angle:.1f}°. "
            f"Maximum angular deviation: {max_deviation:.1f}° "
            f"(within {tolerance_degrees}° tolerance)."
        )
    else:
        explanation = (
            f"Shadows are inconsistent. "
            f"Average direction: {mean_angle:.1f}°. "
            f"Maximum angular deviation: {max_deviation:.1f}° "
            f"exceeds the {tolerance_degrees}° tolerance. "
            f"This suggests multiple or conflicting light sources."
        )
        explanation += length_note

    return {
        "consistent": overall_consistent,
        "average_angle": mean_angle,
        "max_deviation": max_deviation,
        "deviations": deviations,
        "length_consistent": length_consistent,
        "explanation": explanation,
        "details": {
            "shadow_count": len(shadow_vectors),
            "tolerance_degrees": tolerance_degrees,
            "angles": angles,
            "weights_used": weights is not None
        }
    }


def shadow_vector_from_points(
    start: Tuple[float, float],
    end: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Helper: create a direction vector from two points
    (object contact point → tip of the shadow).
    """
    return (end[0] - start[0], end[1] - start[1])


if __name__ == "__main__":
    # Test 1 — consistent shadows
    test_consistent = [(0.9, -0.4), (0.85, -0.5), (0.88, -0.45)]
    result1 = analyze_shadow_consistency(test_consistent)
    print("Test 1 (should be consistent):")
    print(result1["explanation"])
    print()

    # Test 2 — inconsistent shadows
    test_inconsistent = [(0.9, -0.4), (-0.7, 0.6), (0.1, 0.9)]
    result2 = analyze_shadow_consistency(test_inconsistent)
    print("Test 2 (should be inconsistent):")
    print(result2["explanation"])
    print()

    # Test 3 — with lengths
    vectors = [(0.9, -0.4), (0.85, -0.5)]
    lengths = [120.0, 40.0]  # large length difference
    result3 = analyze_shadow_consistency(vectors, shadow_lengths=lengths)
    print("Test 3 (direction ok, length suspicious):")
    print(result3["explanation"])
