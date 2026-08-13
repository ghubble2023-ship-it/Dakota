"""
Gravity Check - Shadow Direction Module

First detection module.
Goal: Determine whether shadows in an image are consistent with a single light source.
"""

from typing import List, Tuple
import math


def analyze_shadow_consistency(
    shadow_vectors: List[Tuple[float, float]],
    tolerance_degrees: float = 25.0
) -> dict:
    """
    Analyze a list of shadow direction vectors for consistency.

    Parameters
    ----------
    shadow_vectors : list of (dx, dy)
        Direction vectors of detected shadows.
        Example: [(0.8, -0.6), (0.75, -0.65)]
    tolerance_degrees : float
        Maximum allowed angular difference between shadows.

    Returns
    -------
    dict
        {
            "consistent": bool,
            "average_angle": float or None,
            "max_deviation": float or None,
            "explanation": str
        }
    """
    if not shadow_vectors:
        return {
            "consistent": False,
            "average_angle": None,
            "max_deviation": None,
            "explanation": "No shadow vectors provided."
        }

    if len(shadow_vectors) == 1:
        angle = math.degrees(math.atan2(shadow_vectors[0][1], shadow_vectors[0][0]))
        return {
            "consistent": True,
            "average_angle": angle,
            "max_deviation": 0.0,
            "explanation": "Only one shadow detected. Consistency cannot be fully verified."
        }

    # Convert vectors to angles in degrees
    angles = []
    for dx, dy in shadow_vectors:
        angle = math.degrees(math.atan2(dy, dx))
        angles.append(angle)

    # Calculate circular mean for angles
    sin_sum = sum(math.sin(math.radians(a)) for a in angles)
    cos_sum = sum(math.cos(math.radians(a)) for a in angles)
    mean_angle = math.degrees(math.atan2(sin_sum, cos_sum))

    # Calculate maximum deviation from mean
    deviations = []
    for a in angles:
        diff = abs(a - mean_angle)
        # Handle angle wrap-around
        if diff > 180:
            diff = 360 - diff
        deviations.append(diff)

    max_deviation = max(deviations)
    consistent = max_deviation <= tolerance_degrees

    if consistent:
        explanation = (
            f"Shadows are consistent. "
            f"Average direction: {mean_angle:.1f}°. "
            f"Maximum deviation: {max_deviation:.1f}° "
            f"(within {tolerance_degrees}° tolerance)."
        )
    else:
        explanation = (
            f"Shadows are inconsistent. "
            f"Average direction: {mean_angle:.1f}°. "
            f"Maximum deviation: {max_deviation:.1f}° "
            f"exceeds {tolerance_degrees}° tolerance. "
            f"This suggests multiple or conflicting light sources."
        )

    return {
        "consistent": consistent,
        "average_angle": mean_angle,
        "max_deviation": max_deviation,
        "explanation": explanation
    }


if __name__ == "__main__":
    # Consistent shadows
    test_consistent = [(0.9, -0.4), (0.85, -0.5), (0.88, -0.45)]
    result1 = analyze_shadow_consistency(test_consistent)
    print("Test 1 (should be consistent):")
    print(result1["explanation"])
    print()

    # Inconsistent shadows
    test_inconsistent = [(0.9, -0.4), (-0.7, 0.6), (0.1, 0.9)]
    result2 = analyze_shadow_consistency(test_inconsistent)
    print("Test 2 (should be inconsistent):")
    print(result2["explanation"])
