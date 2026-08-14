"""
Gravity Check - Spatial Measurement Module

This is the FIRST required step in every Gravity Check inspection.
Before looking at shadows, reflections, glasses, or any other artifact,
we measure the physical space of the image.

Core questions this module answers:
1. Approximate camera-to-subject distance relationships
2. Subject-to-background distance relationships
3. Relative scale of objects that should share the same depth
4. Light direction consistency with the measured geometry
"""

from typing import List, Optional
import math


def estimate_relative_scale(
    object_heights_px: List[float],
    assumed_real_heights_m: Optional[List[float]] = None
) -> dict:
    """
    Compare relative pixel heights of objects that should exist at similar depths.

    Parameters
    ----------
    object_heights_px : list of float
        Measured heights of objects in pixels.
    assumed_real_heights_m : list of float, optional
        Known or estimated real-world heights in meters.
        If provided, returns approximate distance ratios.

    Returns
    -------
    dict
        Analysis of relative scale consistency.
    """
    if len(object_heights_px) < 2:
        return {
            "status": "insufficient_data",
            "explanation": "Need at least two objects to compare relative scale."
        }

    ratios = []
    for i in range(len(object_heights_px)):
        for j in range(i + 1, len(object_heights_px)):
            if object_heights_px[j] == 0:
                continue
            ratio = object_heights_px[i] / object_heights_px[j]
            ratios.append({
                "pair": (i, j),
                "pixel_ratio": ratio
            })

    result = {
        "status": "ok",
        "pixel_ratios": ratios,
        "explanation": (
            "Relative pixel scale calculated. "
            "Large unexpected differences between objects that should be "
            "at similar depth are a strong signal of compositing or generation error."
        )
    }

    if assumed_real_heights_m and len(assumed_real_heights_m) == len(object_heights_px):
        distance_hints = []
        for i, (h_px, h_m) in enumerate(zip(object_heights_px, assumed_real_heights_m)):
            if h_px > 0:
                # Focal length cancels out in ratios; this is relative only
                distance_hints.append({
                    "object_index": i,
                    "relative_distance_proxy": h_m / h_px
                })
        result["relative_distance_proxies"] = distance_hints

    return result


def check_depth_ordering(
    object_bottoms_y: List[float],
    object_heights_px: List[float],
    image_height: float
) -> dict:
    """
    Basic depth ordering check using vertical position and scale.

    In a normal perspective photo:
    - Objects lower in the frame (higher y) are usually closer to camera
    - Closer objects should appear larger if they are the same real size

    This is a soft geometric consistency check, not a full depth map.
    """
    if len(object_bottoms_y) != len(object_heights_px):
        return {
            "consistent": False,
            "explanation": "Mismatched input lengths."
        }

    if len(object_bottoms_y) < 2:
        return {
            "consistent": True,
            "explanation": "Only one object measured. Depth ordering not applicable."
        }

    # Sort by vertical position (bottom of object)
    indexed = sorted(
        enumerate(zip(object_bottoms_y, object_heights_px)),
        key=lambda x: x[1][0]
    )

    issues = []
    for i in range(len(indexed) - 1):
        idx_a, (y_a, h_a) = indexed[i]
        idx_b, (y_b, h_b) = indexed[i + 1]

        # Object A is higher in frame (should be farther)
        # Object B is lower in frame (should be closer)
        # Therefore B should generally be larger if same real-world size
        if h_a > h_b * 1.35:  # significant inversion
            issues.append(
                f"Object {idx_a} is higher in frame but significantly larger "
                f"than object {idx_b} below it. Possible depth inconsistency."
            )

    if issues:
        return {
            "consistent": False,
            "issues": issues,
            "explanation": (
                "Depth ordering shows possible geometric inconsistency. "
                "In real perspective, objects lower in the frame are usually "
                "closer and should appear larger if they share real-world size."
            )
        }

    return {
        "consistent": True,
        "issues": [],
        "explanation": "No strong depth-ordering contradictions detected."
    }


def spatial_report(
    object_heights_px: List[float],
    object_bottoms_y: List[float],
    image_height: float,
    assumed_real_heights_m: Optional[List[float]] = None
) -> dict:
    """
    Full spatial measurement pass.

    This is the entry point that should be called before any other
    Gravity Check module (shadows, reflections, glasses, etc.).
    """
    scale = estimate_relative_scale(object_heights_px, assumed_real_heights_m)
    depth = check_depth_ordering(object_bottoms_y, object_heights_px, image_height)

    return {
        "module": "spatial_measurement",
        "scale_analysis": scale,
        "depth_ordering": depth,
        "summary": (
            "Spatial measurement complete. "
            "This establishes the geometric baseline for all subsequent checks."
        )
    }


if __name__ == "__main__":
    # Example: two people of similar real height
    # One appears much larger and is higher in the frame → suspicious
    heights = [420.0, 280.0]          # pixels
    bottoms = [310.0, 520.0]          # y coordinates
    img_h = 720.0

    report = spatial_report(heights, bottoms, img_h)
    print("Spatial Measurement Report")
    print("==========================")
    print(report["depth_ordering"]["explanation"])
    if report["depth_ordering"]["issues"]:
        for issue in report["depth_ordering"]["issues"]:
            print(" -", issue)
