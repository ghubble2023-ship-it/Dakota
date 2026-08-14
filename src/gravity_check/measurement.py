"""
Gravity Check - Measurement Front-End

This module is responsible for turning a raw image into the structured
measurements that the geometric modules need.

Design principles:
- Physics-first and explainable
- Classical methods preferred
- Clean interface so the engine never cares how the numbers were obtained

Current status:
- Interface is defined
- Basic classical helpers are present
- Full automatic extraction from complex photos is still limited
  (soft shadows and cluttered scenes remain hard)
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import math


# ============================================================
# 1. MEASUREMENT INTERFACE (the contract)
# ============================================================

@dataclass
class ObjectMeasurement:
    """One detected object in the image."""
    label: str                          # e.g. "person", "car", "unknown"
    height_px: float                    # vertical size in pixels
    bottom_y: float                     # y-coordinate of the lowest point
    center_x: float                     # horizontal center
    real_height_m: Optional[float] = None  # if known or assumed


@dataclass
class ShadowMeasurement:
    """One measured shadow."""
    start: Tuple[float, float]          # (x, y) contact point with object
    end: Tuple[float, float]            # (x, y) tip of the shadow
    length_px: float = 0.0

    def direction_vector(self) -> Tuple[float, float]:
        return (self.end[0] - self.start[0], self.end[1] - self.start[1])


@dataclass
class PerspectiveLine:
    """A line segment used for vanishing-point estimation."""
    p1: Tuple[float, float]
    p2: Tuple[float, float]


@dataclass
class ImageMeasurements:
    """
    Complete set of measurements extracted from one image.
    This is the only object the engine should need.
    """
    image_width: int
    image_height: int
    objects: List[ObjectMeasurement] = field(default_factory=list)
    shadows: List[ShadowMeasurement] = field(default_factory=list)
    perspective_lines: List[PerspectiveLine] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # Convenience helpers for the engine
    def object_heights_px(self) -> List[float]:
        return [o.height_px for o in self.objects]

    def object_bottoms_y(self) -> List[float]:
        return [o.bottom_y for o in self.objects]

    def assumed_real_heights_m(self) -> Optional[List[float]]:
        heights = [o.real_height_m for o in self.objects]
        if any(h is None for h in heights):
            return None
        return heights  # type: ignore

    def shadow_vectors(self) -> List[Tuple[float, float]]:
        return [s.direction_vector() for s in self.shadows]

    def shadow_lengths(self) -> List[float]:
        return [s.length_px for s in self.shadows]

    def perspective_line_segments(self):
        return [(pl.p1, pl.p2) for pl in self.perspective_lines]


# ============================================================
# 2. CLASSICAL HELPERS (starting point)
# ============================================================

def measure_shadow_from_points(
    start: Tuple[float, float],
    end: Tuple[float, float]
) -> ShadowMeasurement:
    """
    Create a ShadowMeasurement from two manually or automatically
    detected points (contact point → tip of shadow).
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    return ShadowMeasurement(start=start, end=end, length_px=length)


def measure_object_from_bbox(
    x1: float, y1: float, x2: float, y2: float,
    label: str = "unknown",
    real_height_m: Optional[float] = None
) -> ObjectMeasurement:
    """
    Create an ObjectMeasurement from a bounding box.

    This is the most common classical output format
    (from contour detection, simple detectors, or manual labeling).
    """
    height_px = abs(y2 - y1)
    bottom_y = max(y1, y2)
    center_x = (x1 + x2) / 2.0
    return ObjectMeasurement(
        label=label,
        height_px=height_px,
        bottom_y=bottom_y,
        center_x=center_x,
        real_height_m=real_height_m
    )


def estimate_shadow_direction_simple(
    shadow_mask_points: List[Tuple[float, float]]
) -> Optional[Tuple[float, float]]:
    """
    Very simple classical estimator.

    Given a list of points that belong to a shadow region,
    estimate the dominant direction using the vector between
    the centroid and the farthest point.

    This is a starting classical method — not robust on soft shadows.
    """
    if len(shadow_mask_points) < 5:
        return None

    # Centroid
    cx = sum(p[0] for p in shadow_mask_points) / len(shadow_mask_points)
    cy = sum(p[1] for p in shadow_mask_points) / len(shadow_mask_points)

    # Farthest point from centroid
    farthest = max(
        shadow_mask_points,
        key=lambda p: (p[0] - cx)**2 + (p[1] - cy)**2
    )

    dx = farthest[0] - cx
    dy = farthest[1] - cy
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None

    # Normalize
    return (dx / length, dy / length)


# ============================================================
# 3. EXAMPLE USAGE (how the front-end feeds the engine)
# ============================================================

def example_manual_measurement() -> ImageMeasurements:
    """
    Example of how measurements look when they come from
    careful manual or semi-automatic labeling
    (the current most reliable path).
    """
    measurements = ImageMeasurements(
        image_width=1280,
        image_height=720
    )

    # Two people
    measurements.objects.append(
        measure_object_from_bbox(200, 180, 310, 520, label="person", real_height_m=1.75)
    )
    measurements.objects.append(
        measure_object_from_bbox(600, 250, 690, 510, label="person", real_height_m=1.70)
    )

    # Two shadows
    measurements.shadows.append(
        measure_shadow_from_points(start=(255, 520), end=(340, 610))
    )
    measurements.shadows.append(
        measure_shadow_from_points(start=(645, 510), end=(720, 590))
    )

    # Optional perspective lines (e.g. road edges or building lines)
    measurements.perspective_lines.append(
        PerspectiveLine(p1=(100, 600), p2=(500, 420))
    )
    measurements.perspective_lines.append(
        PerspectiveLine(p1=(900, 650), p2=(700, 430))
    )

    measurements.notes.append("Manual / semi-automatic measurements")
    return measurements


def feed_to_engine(measurements: ImageMeasurements):
    """
    Shows the clean hand-off from measurement front-end → engine.
    """
    from .engine import run_gravity_check

    return run_gravity_check(
        object_heights_px=measurements.object_heights_px(),
        object_bottoms_y=measurements.object_bottoms_y(),
        image_height=measurements.image_height,
        assumed_real_heights_m=measurements.assumed_real_heights_m(),
        perspective_lines=measurements.perspective_line_segments(),
        shadow_vectors=measurements.shadow_vectors(),
        shadow_lengths=measurements.shadow_lengths()
    )


if __name__ == "__main__":
    ms = example_manual_measurement()
    print("Objects measured:", len(ms.objects))
    print("Shadows measured:", len(ms.shadows))
    print("Shadow vectors:", ms.shadow_vectors())
