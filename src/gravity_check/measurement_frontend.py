"""
Gravity Check - Measurement Front-End (OpenCV)

Classical OpenCV extraction that feeds the engine.
Produces objects, shadows, edge metrics, lighting cues,
basic reflection proxies, and simple glasses-region signals.
"""

from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ExtractedObject:
    bbox: Tuple[int, int, int, int]
    height_px: float
    bottom_y: float
    center: Tuple[float, float]
    confidence: float
    contour_area: float = 0.0
    solidity: float = 0.0
    aspect_ratio: float = 0.0
    bright_side_angle: Optional[float] = None
    mean_brightness: float = 0.0


@dataclass
class ExtractedShadow:
    vector: Tuple[float, float]
    length: float
    origin: Tuple[float, float]
    confidence: float


@dataclass
class ExtractedEdges:
    sharpness_values: List[float] = field(default_factory=list)
    bleeding_score: float = 0.0
    bleeding_regions: List[str] = field(default_factory=list)


@dataclass
class ExtractedLighting:
    object_bright_side_angles: List[float] = field(default_factory=list)
    object_brightness: List[float] = field(default_factory=list)
    object_distances_proxy: List[float] = field(default_factory=list)


@dataclass
class ExtractedReflections:
    observed_reflections: int = 0
    expected_reflections: int = 0
    reflection_centers: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class ExtractedGlasses:
    primary_frame_detected: bool = False
    secondary_ghost_detected: bool = False
    ghost_offset_px: Optional[float] = None
    frame_hair_overlap: bool = False
    frame_skin_overlap: bool = False
    observed_lens_reflections: int = 0
    expected_lens_reflections: int = 0


@dataclass
class MeasurementBundle:
    image_height: int
    image_width: int
    objects: List[ExtractedObject] = field(default_factory=list)
    shadows: List[ExtractedShadow] = field(default_factory=list)
    edges: Optional[ExtractedEdges] = None
    lighting: Optional[ExtractedLighting] = None
    reflections: Optional[ExtractedReflections] = None
    glasses: Optional[ExtractedGlasses] = None
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
        if self.lighting:
            if self.lighting.object_bright_side_angles:
                kwargs["object_bright_side_angles"] = self.lighting.object_bright_side_angles
            if self.lighting.object_brightness:
                kwargs["object_brightness"] = self.lighting.object_brightness
            if self.lighting.object_distances_proxy:
                kwargs["object_distances_proxy"] = self.lighting.object_distances_proxy
        if self.reflections:
            kwargs["observed_reflections"] = self.reflections.observed_reflections
            kwargs["expected_reflections"] = self.reflections.expected_reflections
            if self.reflections.reflection_centers:
                kwargs["reflection_centers"] = self.reflections.reflection_centers
        if self.glasses:
            kwargs["primary_frame_detected"] = self.glasses.primary_frame_detected
            kwargs["secondary_ghost_detected"] = self.glasses.secondary_ghost_detected
            if self.glasses.ghost_offset_px is not None:
                kwargs["ghost_offset_px"] = self.glasses.ghost_offset_px
            kwargs["frame_hair_overlap"] = self.glasses.frame_hair_overlap
            kwargs["frame_skin_overlap"] = self.glasses.frame_skin_overlap
            kwargs["observed_lens_reflections"] = self.glasses.observed_lens_reflections
            kwargs["expected_lens_reflections"] = self.glasses.expected_lens_reflections
        return kwargs


def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _preprocess_for_contours(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    v = np.median(blur)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(blur, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    return edges


def _contour_quality(cnt: np.ndarray, image_area: float) -> Dict[str, float]:
    area = float(cv2.contourArea(cnt))
    x, y, w, h = cv2.boundingRect(cnt)
    bbox_area = float(max(w * h, 1))
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull)) + 1e-6
    solidity = area / hull_area
    extent = area / bbox_area
    aspect = float(w) / float(max(h, 1))
    return {
        "area": area,
        "area_ratio": area / image_area,
        "bbox": (x, y, w, h),
        "solidity": solidity,
        "extent": extent,
        "aspect": aspect,
    }


def _is_good_object_contour(q: Dict[str, float], min_area_ratio: float) -> bool:
    if q["area_ratio"] < min_area_ratio:
        return False
    if q["solidity"] < 0.35:
        return False
    if q["extent"] < 0.20:
        return False
    if q["aspect"] < 0.15 or q["aspect"] > 6.0:
        return False
    return True


def _object_confidence(q: Dict[str, float]) -> float:
    conf = 0.25
    conf += 0.30 * np.clip(q["solidity"], 0, 1)
    conf += 0.25 * np.clip(q["extent"], 0, 1)
    conf += 0.15 * np.clip(q["area_ratio"] * 8.0, 0, 1)
    if q["aspect"] < 0.4 or q["aspect"] > 4.0:
        conf *= 0.85
    return float(np.clip(conf, 0.20, 0.95))


def _nms_objects(objects: List[ExtractedObject], iou_thresh: float = 0.45) -> List[ExtractedObject]:
    if not objects:
        return []
    boxes = []
    for o in objects:
        x, y, w, h = o.bbox
        boxes.append([x, y, x + w, y + h, o.confidence])
    boxes = np.array(boxes, dtype=np.float32)
    x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return [objects[i] for i in keep]


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
    candidates.sort(key=lambda o: (o.confidence * np.sqrt(o.contour_area)), reverse=True)
    kept = _nms_objects(candidates, iou_thresh=0.45)
    return kept[:max_objects]


def extract_shadows(
    image: np.ndarray,
    objects: List[ExtractedObject],
    darkness_percentile: float = 15.0
) -> List[ExtractedShadow]:
    if not objects:
        return []
    gray = _to_gray(image)
    h, w = gray.shape
    thresh_val = float(np.percentile(gray, darkness_percentile))
    _, dark = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)
    shadows: List[ExtractedShadow] = []
    for obj in objects:
        ox, oy, ow, oh = obj.bbox
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
        cnt = max(contours, key=cv2.contourArea)
        cnt = cnt + np.array([[[x1, y1]]])
        pts = cnt.reshape(-1, 2).astype(np.float32)
        if len(pts) < 5:
            continue
        mean, eigenvectors = cv2.PCACompute(pts, mean=None)
        axis = eigenvectors[0]
        centroid = mean[0]
        origin_x = obj.center[0]
        origin_y = obj.bottom_y
        dx = float(centroid[0] - origin_x)
        dy = float(centroid[1] - origin_y)
        length = float(np.hypot(dx, dy))
        if length < 10:
            continue
        pca_dx, pca_dy = float(axis[0]), float(axis[1])
        if (pca_dx * dx + pca_dy * dy) < 0:
            pca_dx, pca_dy = -pca_dx, -pca_dy
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
            vector=vector, length=length,
            origin=(origin_x, origin_y), confidence=conf
        ))
    return shadows


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
        y1, y2 = max(0, y - pad), min(gray.shape[0], y + bh + pad)
        x1, x2 = max(0, x - pad), min(gray.shape[1], x + bw + pad)
        ring = magnitude[y1:y2, x1:x2]
        inside = magnitude[y:y + bh, x:x + bw]
        if ring.size == 0 or inside.size == 0:
            continue
        sharp = float(np.percentile(ring, 90))
        sharpness_values.append(sharp)
        outside_mean = float(np.mean(ring))
        inside_mean = float(np.mean(inside))
        if inside_mean > 1e-3:
            ratio = outside_mean / inside_mean
            bleed = float(np.clip((ratio - 0.75) * 0.9, 0.0, 1.0))
            bleeding_scores.append(bleed)
    avg_bleed = float(np.mean(bleeding_scores)) if bleeding_scores else 0.0
    return ExtractedEdges(
        sharpness_values=sharpness_values,
        bleeding_score=avg_bleed,
        bleeding_regions=["object_boundaries"] if avg_bleed > 0.22 else []
    )


def extract_lighting_cues(
    image: np.ndarray,
    objects: List[ExtractedObject]
) -> ExtractedLighting:
    gray = _to_gray(image)
    angles: List[float] = []
    brightness: List[float] = []
    distances: List[float] = []
    h = float(gray.shape[0])
    for obj in objects:
        x, y, bw, bh = obj.bbox
        roi = gray[y:y + bh, x:x + bw]
        if roi.size < 20:
            angles.append(0.0)
            brightness.append(0.0)
            distances.append(1.0 - (obj.bottom_y / h))
            continue
        mean_b = float(np.mean(roi))
        brightness.append(mean_b)
        obj.mean_brightness = mean_b
        mid = bw // 2
        left = roi[:, :max(mid, 1)]
        right = roi[:, mid:]
        left_m = float(np.mean(left)) if left.size else mean_b
        right_m = float(np.mean(right)) if right.size else mean_b
        mid_h = bh // 2
        top = roi[:max(mid_h, 1), :]
        bot = roi[mid_h:, :]
        top_m = float(np.mean(top)) if top.size else mean_b
        bot_m = float(np.mean(bot)) if bot.size else mean_b
        dx = right_m - left_m
        dy = bot_m - top_m
        angle = float(np.degrees(np.arctan2(dy, dx)))
        angles.append(angle)
        obj.bright_side_angle = angle
        distances.append(float(1.0 - (obj.bottom_y / max(h, 1.0))))
    return ExtractedLighting(
        object_bright_side_angles=angles,
        object_brightness=brightness,
        object_distances_proxy=distances,
    )


def extract_reflection_proxies(
    image: np.ndarray,
    objects: List[ExtractedObject]
) -> ExtractedReflections:
    gray = _to_gray(image)
    thresh = float(np.percentile(gray, 98.5))
    _, bright = cv2.threshold(gray, max(thresh, 220), 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers: List[Tuple[float, float]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 8 or area > 800:
            continue
        m = cv2.moments(cnt)
        if m["m00"] < 1e-3:
            continue
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        centers.append((cx, cy))
    expected = min(len(objects), 3)
    return ExtractedReflections(
        observed_reflections=len(centers),
        expected_reflections=expected,
        reflection_centers=centers[:12],
    )


def extract_glasses_proxies(
    image: np.ndarray,
    objects: List[ExtractedObject]
) -> ExtractedGlasses:
    if not objects:
        return ExtractedGlasses()
    gray = _to_gray(image)
    h, w = gray.shape
    primary = max(objects, key=lambda o: o.confidence * o.height_px)
    x, y, bw, bh = primary.bbox
    y1 = y + int(bh * 0.22)
    y2 = y + int(bh * 0.55)
    x1, x2 = x, x + bw
    y1, y2 = max(0, y1), min(h, y2)
    x1, x2 = max(0, x1), min(w, x2)
    roi = gray[y1:y2, x1:x2]
    if roi.size < 100:
        return ExtractedGlasses()
    edges = cv2.Canny(roi, 40, 120)
    row_energy = np.sum(edges, axis=1).astype(np.float32)
    if row_energy.max() < 1:
        return ExtractedGlasses(primary_frame_detected=False)
    peak = float(row_energy.max())
    strong_rows = np.where(row_energy > 0.55 * peak)[0]
    primary_frame = len(strong_rows) >= 2
    ghost = False
    ghost_offset = None
    if len(strong_rows) >= 2:
        diffs = np.diff(strong_rows)
        close = diffs[(diffs > 1) & (diffs < max(6, roi.shape[0] // 8))]
        if len(close) > 0:
            ghost = True
            ghost_offset = float(np.median(close))
    eye_roi = roi
    thresh = float(np.percentile(eye_roi, 97))
    _, spec = cv2.threshold(eye_roi, max(thresh, 200), 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(spec, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lens_refl = sum(1 for c in cnts if 4 < cv2.contourArea(c) < 120)
    return ExtractedGlasses(
        primary_frame_detected=primary_frame,
        secondary_ghost_detected=ghost,
        ghost_offset_px=ghost_offset,
        frame_hair_overlap=False,
        frame_skin_overlap=False,
        observed_lens_reflections=lens_refl,
        expected_lens_reflections=2 if primary_frame else 0,
    )


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
        bundle.lighting = extract_lighting_cues(image, objects)
        bundle.reflections = extract_reflection_proxies(image, objects)
        bundle.glasses = extract_glasses_proxies(image, objects)
    avg_obj_conf = float(np.mean([o.confidence for o in objects])) if objects else 0.0
    mods = []
    if bundle.lighting:
        mods.append("lighting")
    if bundle.reflections:
        mods.append(f"reflections({bundle.reflections.observed_reflections})")
    if bundle.glasses and bundle.glasses.primary_frame_detected:
        mods.append("glasses")
    bundle.notes.append(
        f"Extracted {len(objects)} objects (avg conf {avg_obj_conf:.2f}), "
        f"{len(shadows)} shadows. Extra: {', '.join(mods) or 'none'}."
    )
    return bundle


def extract_from_path(image_path: str, max_objects: int = 6) -> MeasurementBundle:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return extract_measurements(image, max_objects=max_objects)
