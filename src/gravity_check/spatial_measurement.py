"""
Gravity Check - Spatial Measurement Module

This is the FIRST required step in every Gravity Check inspection.
Before looking at shadows, reflections, glasses, or any other artifact,
we measure the physical space of the image.

Core questions this module answers:
1. Approximate camera-to-subject distance relationships
2. Subject-to-background distance relationships
3. Relative scale of objects that should share the same depth
4. Basic perspective / vanishing-point consistency (photogrammetry)
"""

from typing import List, Optional, Tuple, Dict
import math


def estimate_relative_scale(
    object_heights_px: List[float],
    assumed_real_heights_m: Optional[List[float]] = None
) -> dict:
    """
    Compare relative pixel heights of objects that should exist at similar depths.
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
            ratios.append({"pair": (i, j), "pixel_ratio": ratio})

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
    """
    if len(object_bottoms_y) != len(object_heights_px):
        return {"consistent": False, "explanation": "Mismatched input lengths."}

    if len(object_bottoms_y) < 2:
        return {
            "consistent": True,
            "explanation": "Only one object measured. Depth ordering not applicable."
        }

    indexed = sorted(
        enumerate(zip(object_bottoms_y, object_heights_px)),
        key=lambda x: x[1][0]
    )

    issues = []
    for i in range(len(indexed) - 1):
        idx_a, (y_a, h_a) = indexed[i]
        idx_b, (y_b, h_b) = indexed[i + 1]

        if h_a > h_b * 1.35:
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


def estimate_vanishing_point(
    lines: List[Tuple[Tuple[float, float], Tuple[float, float]]]
) -> Dict:
    """
    Very simple vanishing-point estimator from a set of line segments.

    Each line is given as ((x1, y1), (x2, y2)).
    Returns the approximate intersection of the lines (vanishing point)
    or signals that the lines are inconsistent.

    This is a first photogrammetry building block.
    """
    if len(lines) < 2:
        return {
            "status": "insufficient_data",
            "vanishing_point": None,
            "explanation": "Need at least two lines to estimate a vanishing point."
        }

    # Convert lines to parametric form and find pairwise intersections
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            (x1, y1), (x2, y2) = lines[i]
            (x3, y3), (x4, y4) = lines[j]

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-8:
                continue  # parallel

            px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
            py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
            intersections.append((px, py))

    if not intersections:
        return {
            "status": "no_intersection",
            "vanishing_point": None,
            "explanation": "Lines appear parallel or nearly parallel; no reliable vanishing point."
        }

    # Average the intersections
    avg_x = sum(p[0] for p in intersections) / len(intersections)
    avg_y = sum(p[1] for p in intersections) / len(intersections)

    # Simple spread check
    spreads = [math.hypot(p[0] - avg_x, p[1] - avg_y) for p in intersections]
    max_spread = max(spreads) if spreads else 0.0

    consistent = max_spread < 80.0  # pixels — soft threshold

    return {
        "status": "ok" if consistent else "inconsistent",
        "vanishing_point": (avg_x, avg_y),
        "max_spread": max_spread,
        "intersection_count": len(intersections),
        "explanation": (
            f"Vanishing point estimated at ({avg_x:.1f}, {avg_y:.1f}). "
            f"Spread of intersections: {max_spread:.1f}px. "
            + ("Consistent." if consistent else "High spread — perspective may be inconsistent.")
        )
    }


def check_perspective_consistency(
    object_bottoms_y: List[float],
    object_heights_px: List[float],
    vanishing_point_y: Optional[float] = None
) -> dict:
    """
    Soft perspective check.

    If a vanishing point Y is known, objects should generally get smaller
    as they approach that vanishing line.
    """
    if vanishing_point_y is None or len(object_bottoms_y) < 2:
        return {
            "status": "skipped",
            "explanation": "No vanishing point supplied or insufficient objects."
        }

    # Objects closer to the vanishing point (in Y) should tend to be smaller
    issues = []
    for i, (y, h) in enumerate(zip(object_bottoms_y, object_heights_px)):
        distance_to_vp = abs(y - vanishing_point_y)
        # This is only a weak heuristic for now
        if distance_to_vp < 40 and h > 300:
            issues.append(
                f"Object {i} is very close to the vanishing line but still large."
            )

    return {
        "status": "ok" if not issues else "warning",
        "issues": issues,
        "explanation": (
            "Perspective consistency check passed."
            if not issues else
            "Possible perspective / scale conflict near vanishing line."
        )
    }


def spatial_report(
    object_heights_px: List[float],
    object_bottoms_y: List[float],
    image_height: float,
    assumed_real_heights_m: Optional[List[float]] = None,
    perspective_lines: Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = None
) -> dict:
    """
    Full spatial measurement pass (entry point).
    """
    scale = estimate_relative_scale(object_heights_px, assumed_real_heights_m)
    depth = check_depth_ordering(object_bottoms_y, object_heights_px, image_height)

    vanishing = None
    perspective = None
    if perspective_lines:
        vanishing = estimate_vanishing_point(perspective_lines)
        vp_y = vanishing["vanishing_point"][1] if vanishing.get("vanishing_point") else None
        perspective = check_perspective_consistency(
            object_bottoms_y, object_heights_px, vp_y
        )

    return {
        "module": "spatial_measurement",
        "scale_analysis": scale,
        "depth_ordering": depth,
        "vanishing_point": vanishing,
        "perspective_check": perspective,
        "summary": (
            "Spatial measurement complete. "
            "This establishes the geometric baseline for all subsequent checks."
        )
    }
