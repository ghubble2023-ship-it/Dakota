"""
Gravity Check - Measurement Front-End (OpenCV)

This layer turns an image into structured measurements
that the geometric modules already understand.

Design rules:
- Classical OpenCV only (no neural nets)
- Every measurement carries a confidence
- Fail soft: return partial results rather than crash
- Output format matches what run_gravity_check() expects

We stay in full control of every extraction step.
"""

from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ExtractedObject:
    bbox: Tuple[int, int, int, int]          # x, y, w, h
    height_px: float
    bottom_y: float
    center: Tuple[float, float]
    confidence: float
    contour_area: float = 0.0


@dataclass
class ExtractedShadow:
    vector: Tuple[float, float]              # dx, dy
    length: float
    origin: Tuple[float, float]
    confidence: float


@dataclass
class ExtractedEdges:
    sharpness_values: List[float] = field(default_factory=list)
    bleeding_score: float = 0.0
    bleeding_regions: List[str] = field(default_factory=list)


@dataclass
class MeasurementBundle:
    """All measurements extracted from one image."""
    image_height: int
    image_width: int
    objects: List[ExtractedObject] = field(default_factory=list)
    shadows: List[ExtractedShadow] = field(default_factory=list)
    edges: Optional[ExtractedEdges] = None
    notes: List[str] = field(default_factory=list)

    def to_engine_kwargs(self) -> Dict[str, Any]:
        """Convert into arguments for run_gravity_check()."""
        kwargs: Dict[str, Any] = {
            "object_heights_px": [o.height_px for o in self.objects],
            "object_bottoms_y": [o.bottom_y for o in self.objects],
            "image_height": float(self.image_height),
        }

        if self.shadows:
            kwargs["shadow_vectors"] = [s.vector for s in self.shadows]
            kwargs["shadow_lengths"] = [s.length for s in self.shadows]

        if self.edges:
            if self.edges.sharpness_values:
                kwargs["sharpness_values"] = self.edges.sharpness_values
            if self.edges.bleeding_score > 0.15:
                kwargs["edge_bleeding_detected"] = True
                kwargs["edge_bleeding_severity"] = self.edges.bleeding_score
                kwargs["edge_affected_regions"] = self.edges.bleeding_regions or ["general"]

        return kwargs


def _largest_contours(mask: np.ndarray, min_area: float = 500.0) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    contours.sort(key=cv2.contourArea, reverse=True)
    return contours


def extract_objects(
    image: np.ndarray,
    max_objects: int = 8,
    min_area_ratio: float = 0.002
) -> List[ExtractedObject]:
    """
    Simple foreground object extraction via edge + contour.
    This is deliberately conservative and explainable.
    """
    h, w = image.shape[:2]
    min_area = h * w * min_area_ratio

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours = _largest_contours(closed, min_area=min_area)
    objects: List[ExtractedObject] = []

    for cnt in contours[:max_objects]:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        # Confidence based on how filled the bbox is
        fill = area / (bw * bh + 1e-6)
        conf = float(np.clip(fill * 1.2, 0.25, 0.95))

        objects.append(ExtractedObject(
            bbox=(x, y, bw, bh),
            height_px=float(bh),
            bottom_y=float(y + bh),
            center=(x + bw / 2.0, y + bh / 2.0),
            confidence=conf,
            contour_area=float(area)
        ))

    return objects


def extract_shadows(
    image: np.ndarray,
    objects: List[ExtractedObject],
    darkness_percentile: float = 18.0
) -> List[ExtractedShadow]:
    """
    Extract dark regions that are plausible shadows near objects.
    Uses simple thresholding + principal axis of the dark blob.
    """
    if not objects:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape

    # Adaptive dark threshold
    thresh_val = np.percentile(gray, darkness_percentile)
    _, dark = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

    # Clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)

    shadows: List[ExtractedShadow] = []

    for obj in objects:
        ox, oy, ow, oh = obj.bbox
        # Search region below and around the object
        y1 = max(0, oy + int(oh * 0.4))
        y2 = min(h, oy + oh + int(oh * 1.2))
        x1 = max(0, ox - int(ow * 0.4))
        x2 = min(w, ox + ow + int(ow * 0.4))

        roi = dark[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        contours = _largest_contours(roi, min_area=80)
        if not contours:
            continue

        cnt = contours[0]
        # Shift contour back to image coordinates
        cnt = cnt + np.array([[[x1, y1]]])

        # Fit principal axis via PCA / moments
        moments = cv2.moments(cnt)
        if moments["m00"] < 1e-4:
            continue

        # Centroid of shadow blob
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        # Object bottom center as origin estimate
        origin_x = obj.center[0]
        origin_y = obj.bottom_y

        dx = cx - origin_x
        dy = cy - origin_y
        length = float(np.hypot(dx, dy))

        if length < 8:
            continue

        # Normalize direction
        vector = (float(dx / length), float(dy / length))

        # Confidence: longer + darker blobs score higher
        area = cv2.contourArea(cnt)
        conf = float(np.clip(0.35 + (length / 200.0) + (area / 5000.0), 0.3, 0.9))

        shadows.append(ExtractedShadow(
            vector=vector,
            length=length,
            origin=(origin_x, origin_y),
            confidence=conf
        ))

    return shadows


def extract_edge_metrics(image: np.ndarray, objects: List[ExtractedObject]) -> ExtractedEdges:
    """
    Compute simple edge sharpness and a rough bleeding indicator.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)

    sharpness_values: List[float] = []
    bleeding_scores: List[float] = []

    for obj in objects:
        x, y, bw, bh = obj.bbox
        # Sample a band just outside the bbox
        pad = 6
        y1, y2 = max(0, y - pad), min(gray.shape[0], y + bh + pad)
        x1, x2 = max(0, x - pad), min(gray.shape[1], x + bw + pad)

        region = magnitude[y1:y2, x1:x2]
        if region.size == 0:
            continue

        # Sharpness proxy: high percentile of gradient magnitude
        sharp = float(np.percentile(region, 85))
        sharpness_values.append(sharp)

        # Very crude bleeding proxy: high gradient spread outside vs inside
        inside = magnitude[y:y+bh, x:x+bw]
        if inside.size > 0:
            outside_mean = float(np.mean(region))
            inside_mean = float(np.mean(inside))
            if inside_mean > 1e-3:
                ratio = outside_mean / inside_mean
                # High outside energy relative to inside can indicate soft bleed
                bleeding_scores.append(float(np.clip((ratio - 0.8) * 0.8, 0.0, 1.0)))

    avg_bleed = float(np.mean(bleeding_scores)) if bleeding_scores else 0.0

    return ExtractedEdges(
        sharpness_values=sharpness_values,
        bleeding_score=avg_bleed,
        bleeding_regions=["object_boundaries"] if avg_bleed > 0.25 else []
    )


def extract_measurements(
    image: np.ndarray,
    max_objects: int = 6
) -> MeasurementBundle:
    """
    Main entry point.

    Parameters
    ----------
    image : BGR or grayscale numpy array (as loaded by cv2.imread)
    """
    if image is None or image.size == 0:
        raise ValueError("Empty image passed to extract_measurements")

    h, w = image.shape[:2]
    bundle = MeasurementBundle(image_height=h, image_width=w)

    # 1. Objects
    objects = extract_objects(image, max_objects=max_objects)
    bundle.objects = objects
    if not objects:
        bundle.notes.append("No objects extracted above area threshold.")

    # 2. Shadows (depends on objects)
    shadows = extract_shadows(image, objects)
    bundle.shadows = shadows
    if objects and not shadows:
        bundle.notes.append("Objects found but no plausible shadows extracted.")

    # 3. Edge metrics
    if objects:
        bundle.edges = extract_edge_metrics(image, objects)

    bundle.notes.append(
        f"Extracted {len(objects)} objects, {len(shadows)} shadows."
    )
    return bundle


def extract_from_path(image_path: str, max_objects: int = 6) -> MeasurementBundle:
    """Convenience loader."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return extract_measurements(image, max_objects=max_objects)
