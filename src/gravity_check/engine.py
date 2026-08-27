"""
Gravity Check Engine

Orchestrates geometric modules under the locked mandatory order.

Phase 1 — Foundation (always first):
    spatial + shadows + lighting (+ optional eye-volume note)
Phase 2 — Catchlights (always second, after the room exists):
    protocol checks; UNKNOWN if unmeasured
Phase 3 — Secondary geometry (always third):
    glasses, contact/hair/specular/support/face-body, reflections, edges

A phase is never skipped. Missing measurements = incomplete (consistent=None),
not an automatic fail.
"""

from typing import List, Tuple, Optional, Dict, Any

from .spatial_measurement import spatial_report
from .shadow_direction import analyze_shadow_consistency
from .lighting_geometry import analyze_lighting_geometry
from .catchlights import analyze_catchlights
from .phase3_secondary import analyze_phase3_secondary
from .reflections import analyze_reflections
from .glasses_artifacts import analyze_glasses_artifacts
from .edge_bleeding import analyze_edge_bleeding
from .scoring import build_score


ENGINE_VERSION = "0.8.0"

MANDATORY_ORDER = (
    "phase_1_foundation",
    "phase_2_catchlights",
    "phase_3_secondary",
)


def _incomplete(module: str, phase: int, reason: str) -> Dict[str, Any]:
    return {
        "module": module,
        "phase": phase,
        "consistent": None,
        "confidence": 0.0,
        "status": "incomplete",
        "explanation": reason,
    }


def _phase_status(modules: List[Dict[str, Any]]) -> str:
    if not modules:
        return "incomplete"
    states = []
    for m in modules:
        if m.get("consistent") is False:
            states.append("conflict")
        elif m.get("status") == "incomplete" or m.get("consistent") is None:
            states.append("incomplete")
        else:
            states.append("complete")
    if "conflict" in states:
        return "conflict"
    if all(s == "complete" for s in states):
        return "complete"
    if all(s == "incomplete" for s in states):
        return "incomplete"
    return "partial"


def run_gravity_check(
    object_heights_px: Optional[List[float]] = None,
    object_bottoms_y: Optional[List[float]] = None,
    image_height: float = 0.0,
    assumed_real_heights_m: Optional[List[float]] = None,
    perspective_lines: Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
    eyeball_curvature_ok: Optional[bool] = None,
    camera_to_subject_proxy: Optional[float] = None,
    subject_to_background_proxy: Optional[float] = None,
    shadow_vectors: Optional[List[Tuple[float, float]]] = None,
    shadow_lengths: Optional[List[float]] = None,
    object_bright_side_angles: Optional[List[float]] = None,
    object_distances_proxy: Optional[List[float]] = None,
    object_brightness: Optional[List[float]] = None,
    left_eye_center: Optional[Tuple[float, float]] = None,
    right_eye_center: Optional[Tuple[float, float]] = None,
    left_glint: Optional[Dict[str, float]] = None,
    right_glint: Optional[Dict[str, float]] = None,
    identical_glint_structure: Optional[bool] = None,
    extra_streaks_fight_light: Optional[bool] = None,
    contact_shadow_present: Optional[bool] = None,
    contact_shadow_expected: Optional[bool] = None,
    hair_occlusion_ok: Optional[bool] = None,
    specular_matches_light: Optional[bool] = None,
    support_weight_ok: Optional[bool] = None,
    face_body_light_split: Optional[bool] = None,
    expected_reflections: int = 0,
    observed_reflections: int = 0,
    reflection_object_centers: Optional[List[Tuple[float, float]]] = None,
    reflection_centers: Optional[List[Tuple[float, float]]] = None,
    mirror_line_y: Optional[float] = None,
    reflection_heights_px: Optional[List[float]] = None,
    primary_frame_detected: bool = False,
    secondary_ghost_detected: bool = False,
    ghost_offset_px: Optional[float] = None,
    frame_hair_overlap: bool = False,
    frame_skin_overlap: bool = False,
    bleeding_severity: float = 0.0,
    expected_lens_reflections: int = 0,
    observed_lens_reflections: int = 0,
    reflections_consistent_with_light: Optional[bool] = None,
    edge_bleeding_detected: bool = False,
    edge_bleeding_severity: float = 0.0,
    edge_affected_regions: Optional[List[str]] = None,
    sharpness_values: Optional[List[float]] = None,
    sharpness_depth_proxies: Optional[List[float]] = None,
    fringing_detected: bool = False,
    fringing_severity: float = 0.0,
    fringing_has_optical_explanation: bool = False,
) -> Dict[str, Any]:

    object_heights_px = list(object_heights_px or [])
    object_bottoms_y = list(object_bottoms_y or [])

    report: Dict[str, Any] = {
        "engine": "Gravity Check",
        "version": ENGINE_VERSION,
        "mandatory_order": list(MANDATORY_ORDER),
        "skipped_ahead": False,
        "modules_run": [],
        "flags": [],
        "modules": {},
        "phases": {},
    }

    phase1_mods: List[Dict[str, Any]] = []

    spatial = spatial_report(
        object_heights_px=object_heights_px,
        object_bottoms_y=object_bottoms_y if object_bottoms_y else [0.0] * len(object_heights_px),
        image_height=float(image_height or 0.0),
        assumed_real_heights_m=assumed_real_heights_m,
        perspective_lines=perspective_lines,
    )
    spatial["phase"] = 1
    if not object_heights_px:
        spatial["status"] = "incomplete"
        spatial["summary"] = (
            "Phase 1 spatial ran with no objects. Incomplete room, not a fail."
        )
        if isinstance(spatial.get("depth_ordering"), dict):
            spatial["depth_ordering"]["consistent"] = None
            spatial["depth_ordering"]["explanation"] = (
                "No objects measured. Depth ordering not evaluated."
            )
    else:
        spatial["status"] = "complete"
    spatial["room"] = {
        "eyeball_curvature_ok": eyeball_curvature_ok,
        "camera_to_subject_proxy": camera_to_subject_proxy,
        "subject_to_background_proxy": subject_to_background_proxy,
    }
    report["modules"]["spatial_measurement"] = spatial
    report["modules_run"].append("spatial_measurement")
    phase1_mods.append(spatial)
    if spatial.get("depth_ordering", {}).get("consistent") is False:
        report["flags"].append("depth_ordering_conflict")

    if shadow_vectors:
        shadow = analyze_shadow_consistency(
            shadow_vectors,
            shadow_lengths=shadow_lengths,
        )
        shadow["phase"] = 1
        shadow["status"] = "complete"
        if not shadow.get("consistent", True):
            report["flags"].append("shadow_inconsistency")
    else:
        shadow = _incomplete(
            "shadow_direction",
            1,
            "Phase 1 shadow ran. No shadow vectors. Incomplete, not a fail.",
        )
    report["modules"]["shadow_direction"] = shadow
    report["modules_run"].append("shadow_direction")
    phase1_mods.append(shadow)

    light_angle = None
    if shadow.get("average_angle") is not None:
        light_angle = (shadow["average_angle"] + 180) % 360

    lighting_has_data = (
        object_bright_side_angles is not None
        or (object_distances_proxy is not None and object_brightness is not None)
    )
    if lighting_has_data:
        lighting = analyze_lighting_geometry(
            light_angle=light_angle,
            object_bright_side_angles=object_bright_side_angles,
            object_distances_proxy=object_distances_proxy,
            object_brightness=object_brightness,
        )
        lighting["phase"] = 1
        lighting["status"] = "complete" if lighting.get("consistent") is not None else "incomplete"
        if lighting.get("consistent") is False:
            report["flags"].append("lighting_inconsistency")
    else:
        lighting = _incomplete(
            "lighting_geometry",
            1,
            "Phase 1 lighting ran. No bright-side or falloff measurements. Incomplete, not a fail.",
        )
        lighting["light_angle_from_shadows"] = light_angle
    report["modules"]["lighting_geometry"] = lighting
    report["modules_run"].append("lighting_geometry")
    phase1_mods.append(lighting)

    report["phases"]["phase_1_foundation"] = {
        "order": 1,
        "status": _phase_status(phase1_mods),
        "modules": ["spatial_measurement", "shadow_direction", "lighting_geometry"],
        "light_angle": light_angle,
    }

    catch = analyze_catchlights(
        light_angle=light_angle,
        eyeball_curvature_ok=eyeball_curvature_ok,
        left_eye_center=left_eye_center,
        right_eye_center=right_eye_center,
        left_glint=left_glint,
        right_glint=right_glint,
        identical_structure=identical_glint_structure,
        extra_streaks_fight_light=extra_streaks_fight_light,
    )
    report["modules"]["catchlights"] = catch
    report["modules_run"].append("catchlights")
    if catch.get("consistent") is False:
        report["flags"].append("catchlight_inconsistency")
    report["phases"]["phase_2_catchlights"] = {
        "order": 2,
        "status": catch.get("status", "incomplete"),
        "modules": ["catchlights"],
        "ran_after_foundation": True,
        "measurement_gap_open": True,
    }

    phase3_mods: List[Dict[str, Any]] = []

    secondary = analyze_phase3_secondary(
        contact_shadow_present=contact_shadow_present,
        contact_shadow_expected=contact_shadow_expected,
        hair_occlusion_ok=hair_occlusion_ok,
        specular_matches_light=specular_matches_light,
        support_weight_ok=support_weight_ok,
        face_body_light_split=face_body_light_split,
    )
    report["modules"]["phase3_secondary"] = secondary
    report["modules_run"].append("phase3_secondary")
    phase3_mods.append(secondary)
    if secondary.get("consistent") is False:
        report["flags"].append("phase3_secondary_conflict")

    glasses_has_data = (
        primary_frame_detected or secondary_ghost_detected
        or frame_hair_overlap or frame_skin_overlap
        or observed_lens_reflections > 0 or expected_lens_reflections > 0
        or ghost_offset_px is not None
    )
    if glasses_has_data:
        glasses = analyze_glasses_artifacts(
            primary_frame_detected=primary_frame_detected,
            secondary_ghost_detected=secondary_ghost_detected,
            ghost_offset_px=ghost_offset_px,
            frame_hair_overlap=frame_hair_overlap,
            frame_skin_overlap=frame_skin_overlap,
            bleeding_severity=bleeding_severity,
            expected_lens_reflections=expected_lens_reflections,
            observed_lens_reflections=observed_lens_reflections,
            reflections_consistent_with_light=reflections_consistent_with_light,
        )
        glasses["phase"] = 3
        glasses["status"] = "complete" if glasses.get("consistent") is not None else "incomplete"
    else:
        glasses = _incomplete(
            "glasses_artifacts",
            3,
            "Phase 3 glasses ran. No frame/lens measurements. Incomplete, not a fail.",
        )
    report["modules"]["glasses_artifacts"] = glasses
    report["modules_run"].append("glasses_artifacts")
    phase3_mods.append(glasses)
    if glasses.get("consistent") is False:
        report["flags"].append("glasses_artifact")

    reflections_has_data = (
        expected_reflections > 0 or observed_reflections > 0
        or reflection_centers is not None or reflection_heights_px is not None
        or reflection_object_centers is not None
    )
    if reflections_has_data:
        reflections = analyze_reflections(
            expected_reflections=expected_reflections,
            observed_reflections=observed_reflections,
            object_centers=reflection_object_centers,
            reflection_centers=reflection_centers,
            mirror_line_y=mirror_line_y,
            object_heights_px=object_heights_px if reflection_heights_px else None,
            reflection_heights_px=reflection_heights_px,
        )
        reflections["phase"] = 3
        reflections["status"] = "complete" if reflections.get("consistent") is not None else "incomplete"
    else:
        reflections = _incomplete(
            "reflections",
            3,
            "Phase 3 reflections ran. No reflection measurements. Incomplete, not a fail.",
        )
    report["modules"]["reflections"] = reflections
    report["modules_run"].append("reflections")
    phase3_mods.append(reflections)
    if reflections.get("consistent") is False:
        report["flags"].append("reflection_inconsistency")

    edges_has_data = (
        edge_bleeding_detected or sharpness_values is not None or fringing_detected
    )
    if edges_has_data:
        edges = analyze_edge_bleeding(
            bleeding_detected=edge_bleeding_detected,
            bleeding_severity=edge_bleeding_severity,
            affected_regions=edge_affected_regions,
            sharpness_values=sharpness_values,
            depth_proxies=sharpness_depth_proxies,
            fringing_detected=fringing_detected,
            fringing_severity=fringing_severity,
            has_optical_explanation=fringing_has_optical_explanation,
        )
        edges["phase"] = 3
        edges["status"] = "complete" if edges.get("consistent") is not None else "incomplete"
    else:
        edges = _incomplete(
            "edge_bleeding",
            3,
            "Phase 3 edges ran. No bleeding/sharpness/fringing measurements. Incomplete, not a fail.",
        )
    report["modules"]["edge_bleeding"] = edges
    report["modules_run"].append("edge_bleeding")
    phase3_mods.append(edges)
    if edges.get("consistent") is False:
        report["flags"].append("edge_bleeding_artifact")

    report["phases"]["phase_3_secondary"] = {
        "order": 3,
        "status": _phase_status(phase3_mods),
        "modules": [
            "phase3_secondary",
            "glasses_artifacts",
            "reflections",
            "edge_bleeding",
        ],
        "ran_after_catchlights": True,
    }

    scoring = build_score(report)
    report["scoring"] = scoring
    score = scoring["score"]
    interpretation = scoring["interpretation"]

    p1 = report["phases"]["phase_1_foundation"]["status"]
    p2 = report["phases"]["phase_2_catchlights"]["status"]
    p3 = report["phases"]["phase_3_secondary"]["status"]
    order_note = f"Order locked 1\u21922\u21923 (foundation={p1}, catchlights={p2}, secondary={p3})."

    if score >= 0.60:
        summary = (
            f"Gravity Check: mostly consistent. Score {score:.2f}. "
            f"{interpretation} {order_note}"
        )
    elif score >= 0.45:
        summary = (
            f"Gravity Check: inconclusive. Score {score:.2f}. "
            f"{interpretation} {order_note}"
        )
    else:
        flags = ", ".join(report["flags"]) if report["flags"] else "none"
        summary = (
            f"Gravity Check: inconsistencies detected. Score {score:.2f}. "
            f"{interpretation} Flags: {flags}. {order_note}"
        )

    report["summary"] = summary
    report["score"] = score
    return report
