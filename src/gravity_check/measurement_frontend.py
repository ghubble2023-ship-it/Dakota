"""
Gravity Check - Measurement Front-End (OpenCV)

Optimized contour detection + refined measurement logic.

Design rules:
- Classical OpenCV only
- Every measurement carries confidence
- Fail soft
- Output matches run_gravity_check() inputs
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
    solidity: float = 0.0
    aspect_ratio: float = 0.0


@dataclass
class ExtractedShadow:
    vector: Tuple[float, float]              # unit dx, dy
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
    image_height: int
    image_width: int
    objects: List[ExtractedObject] = field(default_factory=list)
    shadows: List[ExtractedShadow] = field(default_factory=list)
    edges: Optional[ExtractedEdges] = None
    notes: List[str] = field(default_factory=list)

    def to_engine_kwargs(self) -> Dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Contour helpers (optimized)
# ---------------------------------------------------------------------------

def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _adaptive_canny(gray: np.ndarray) -> np.ndarray:
    """
    Auto-threshold Canny based on median intensity.
    More stable across lighting conditions than fixed 50/150.
    """
    v = float(np.median(gray))
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    # Floor so very dark images still produce edges
    lower = max(lower, 20)
    upper = max(upper, lower + 40)
    return cv2.Canny(gray, lower, upper)


def _preprocess_for_contours(gray: np.ndarray) -> np.ndarray:
    """
    Stronger preprocessing pipeline:
    1. Mild denoise
    2. Contrast-limited adaptive histogram equalization (CLAHE)
    3. Adaptive Canny
    4. Morphological close + open to clean edges
    """
    # Denoise while keeping edges
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # Local contrast boost
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    edges = _adaptive_canny(enhanced)

    # Close gaps in object outlines, then open to drop speckles
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)

    return cleaned


def _contour_quality(cnt: np.ndarray, image_area: float) -> Dict[str, float]:
    """
    Compute geometric quality metrics for a contour.
    Used both for filtering and for confidence.
    """
    area = float(cv2.contourArea(cnt))
    x, y, w, h = cv2.boundingRect(cnt)
    bbox_area = float(w * h + 1e-6)

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull) + 1e-6)

    extent = area / bbox_area                    # how filled the bbox is
    solidity = area / hull_area                  # how convex
    aspect = h / (w + 1e-6)                      # height/width
    area_ratio = area / (image_area + 1e-6)

    # Perimeter smoothness (lower = smoother)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    complexity = len(approx)

    return {
        "area": area,
        "extent": extent,
        "solidity": solidity,
        "aspect": aspect,
        "area_ratio": area_ratio,
        "complexity": float(complexity),
        "bbox": (x, y, w, h),
    }


def _is_good_object_contour(q: Dict[str, float], min_area_ratio: float) -> bool:
    """
    Reject contours that are unlikely to be real objects.
    """
    if q["area_ratio"] < min_area_ratio:
        return False
    if q["solidity"] < 0.35:          # too fragmented
        return False
    if q["extent"] < 0.15:            # mostly empty bbox
        return False
    if q["aspect"] < 0.15 or q["aspect"] > 8.0:  # extreme shapes
        return False
    # Reject near-full-frame contours (usually the border)
    if q["area_ratio"] > 0.85:
        return False
    return True


def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter + 1e-6
    return inter / union


def _nms_objects(objects: List[ExtractedObject], iou_thresh: float = 0.45) -> List[ExtractedObject]:
    """Suppress overlapping detections, keep higher confidence."""
    if not objects:
        return []
    objects = sorted(objects, key=lambda o: o.confidence, reverse=True)
    kept: List[ExtractedObject] = []
    for obj in objects:
        if all(_iou(obj.bbox, k.bbox) < iou_thresh for k in kept):
            kept.append(obj)
    return kept


def _object_confidence(q: Dict[str, float]) -> float:
    """
    Confidence from geometric quality, not just fill ratio.
    """
    # Prefer solid, reasonably filled, mid-sized contours
    conf = 0.25
    conf += 0.30 * np.clip(q["solidity"], 0, 1)
    conf += 0.25 * np.clip(q["extent"], 0, 1)
    conf += 0.15 * np.clip(q["area_ratio"] * 8.0, 0, 1)  # reward some size
    # Mild penalty for extreme aspect
    if q["aspect"] < 0.4 or q["aspect"] > 4.0:
        conf *= 0.85
    return float(np.clip(conf, 0.20, 0.95))


# ---------------------------------------------------------------------------
# Object extraction (optimized)
# ---------------------------------------------------------------------------

def extract_objects(
    image: np.ndarray,
    max_objects: int = 8,
    min_area_ratio: float = 0.0015
) -> List[ExtractedObject]:
    h, w = image.shape[:2]
    image_area = float(h * w)
    gray = _to_gray(image)

    edge_mask = _preprocess_for_contours(gray)

    contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[ExtractedObject] = []
    for cnt in contours:
        q = _contour_quality(cnt, image_area)
        if not _is_good_object_contour(q, min_area_ratio):
            continue

        x, y, bw, bh = q["bbox"]
        conf = _object_confidence(q)

        candidates.append(ExtractedObject(
            bbox=(x, y, bw, bh),
            height_px=float(bh),
            bottom_y=float(y + bh),
            center=(x + bw / 2.0, y + bh / 2.0),
            confidence=conf,
            contour_area=q["area"],
            solidity=q["solidity"],
            aspect_ratio=q["aspect"],
        ))

    # Prefer larger + higher confidence, then NMS
    candidates.sort(key=lambda o: (o.confidence * np.sqrt(o.contour_area)), reverse=True)
    kept = _nms_objects(candidates, iou_thresh=0.45)
    return kept[:max_objects]


# ---------------------------------------------------------------------------
# Shadow extraction (refined)
# ---------------------------------------------------------------------------

def extract_shadows(
    image: np.ndarray,
    objects: List[ExtractedObject],
    darkness_percentile: float = 15.0
) -> List[ExtractedShadow]:
    if not objects:
        return []

    gray = _to_gray(image)
    h, w = gray.shape

    # Soft dark mask (percentile + Otsu hybrid tendency)
    thresh_val = float(np.percentile(gray, darkness_percentile))
    _, dark = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)

    shadows: List[ExtractedShadow] = []

    for obj in objects:
        ox, oy, ow, oh = obj.bbox

        # Search mainly below the object (where cast shadows usually fall)
        y1 = max(0, oy + int(oh * 0.35))
        y2 = min(h, oy + oh + int(oh * 1.4))
        x1 = max(0, ox - int(ow * 0.55))
        x2 = min(w, ox + ow + int(ow * 0.55))

        roi = dark[y1:y2, x1:x2]
        if roi.size < 50:
            continue

        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= 60]
        if not contours:
            continue

        # Take the largest dark blob in the search region
        cnt = max(contours, key=cv2.contourArea)
        cnt = cnt + np.array([[[x1, y1]]])  # back to image coords

        # PCA on contour points for principal direction
        pts = cnt.reshape(-1, 2).astype(np.float32)
        if len(pts) < 5:
            continue

        mean, eigenvectors = cv2.PCACompute(pts, mean=None)
        axis = eigenvectors[0]  # principal axis
        centroid = mean[0]

        origin_x = obj.center[0]
        origin_y = obj.bottom_y

        # Direction from object base toward shadow centroid
        dx = float(centroid[0] - origin_x)
        dy = float(centroid[1] - origin_y)
        length = float(np.hypot(dx, dy))
        if length < 10:
            continue

        # Prefer the PCA axis if it roughly agrees with centroid direction
        pca_dx, pca_dy = float(axis[0]), float(axis[1])
        # Flip PCA axis to point away from object if needed
        if (pca_dx * dx + pca_dy * dy) < 0:
            pca_dx, pca_dy = -pca_dx, -pca_dy

        # Blend centroid direction with PCA for stability
        blend_dx = 0.65 * (dx / length) + 0.35 * pca_dx
        blend_dy = 0.65 * (dy / length) + 0.35 * pca_dy
        norm = float(np.hypot(blend_dx, blend_dy) + 1e-8)
        vector = (blend_dx / norm, blend_dy / norm)

        area = float(cv2.contourArea(cnt))
        conf = float(np.clip(
            0.30 + 0.25 * obj.confidence + (length / 250.0) + (area / 8000.0),
            0.25, 0.92
        ))

        shadows.append(ExtractedShadow(
            vector=vector,
            length=length,
            origin=(origin_x, origin_y),
            confidence=conf
        ))

    return shadows


# ---------------------------------------------------------------------------
# Edge metrics (refined)
# ---------------------------------------------------------------------------

def extract_edge_metrics(image: np.ndarray, objects: List[ExtractedObject]) -> ExtractedEdges:
    gray = _to_gray(image)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    sharpness_values: List[float] = []
    bleeding_scores: List[float] = []

    for obj in objects:
        x, y, bw, bh = obj.bbox
        pad = max(4, int(min(bw, bh) * 0.08))

        y1 = max(0, y - pad)
        y2 = min(gray.shape[0], y + bh + pad)
        x1 = max(0, x - pad)
        x2 = min(gray.shape[1], x + bw + pad)

        ring = magnitude[y1:y2, x1:x2]
        inside = magnitude[y:y + bh, x:x + bw]
        if ring.size == 0 or inside.size == 0:
            continue

        # Sharpness: robust high percentile on the border ring
        sharp = float(np.percentile(ring, 90))
        sharpness_values.append(sharp)

        # Bleeding proxy: energy just outside relative to interior edge energy
        # Create a rough "outside only" by subtracting a shrunken inside mean
        outside_mean = float(np.mean(ring))
        inside_mean = float(np.mean(inside))
        if inside_mean > 1e-3:
            ratio = outside_mean / inside_mean
            # Soft edges / halos push ratio up
            bleed = float(np.clip((ratio - 0.75) * 0.9, 0.0, 1.0))
            bleeding_scores.append(bleed)

    avg_bleed = float(np.mean(bleeding_scores)) if bleeding_scores else 0.0

    return ExtractedEdges(
        sharpness_values=sharpness_values,
        bleeding_score=avg_bleed,
        bleeding_regions=["object_boundaries"] if avg_bleed > 0.22 else []
    )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def extract_measurements(
    image: np.ndarray,
    max_objects: int = 6
) -> MeasurementBundle:
    if image is None or image.size == 0:
        raise ValueError("Empty image passed to extract_measurements")

    h, w = image.shape[:2]
    bundle = MeasurementBundle(image_height=h, image_width=w)

    objects = extract_objects(image, max_objects=max_objects)
    bundle.objects = objects
    if not objects:
        bundle.notes.append("No objects passed quality filters.")

    shadows = extract_shadows(image, objects)
    bundle.shadows = shadows
    if objects and not shadows:
        bundle.notes.append("Objects found but no plausible shadows extracted.")

    if objects:
        bundle.edges = extract_edge_metrics(image, objects)

    avg_obj_conf = float(np.mean([o.confidence for o in objects])) if objects else 0.0
    bundle.notes.append(
        f"Extracted {len(objects)} objects (avg conf {avg_obj_conf:.2f}), "
        f"{len(shadows)} shadows."
    )
    return bundle


def extract_from_path(image_path: str, max_objects: int = 6) -> MeasurementBundle:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return extract_measurements(image, max_objects=max_objects)
