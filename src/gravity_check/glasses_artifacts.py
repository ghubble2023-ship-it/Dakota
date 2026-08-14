"""
Gravity Check - Glasses & Frame Artifacts Module

Goal:
Detect common geometric and optical failures around eyeglasses
that frequently appear in AI-generated faces.

This module is physics-first and measurement-driven.
It does not look at raw pixels directly. It reasons about
structured observations of the glasses region.

Targeted failure modes (from real observations):
1. Ghost / doubled frame (frame appears slightly offset or duplicated)
2. Frame edges that bleed into hair or skin
3. Inconsistent reflections in the lenses
4. Frame thickness or geometry that breaks perspective
5. Missing or impossible lens refraction cues

These are the kinds of discrepancies the "Eye of Greg" has repeatedly flagged.
"""

from typing import List, Optional, Dict, Any


def check_ghost_frame(
    primary_frame_detected: bool,
    secondary_ghost_detected: bool,
    offset_px: Optional[float] = None,
    offset_tolerance_px: float = 3.0
) -> Dict[str, Any]:
    """
    Detect the classic "ghost frame" artifact:
    a second faint or partial outline of the glasses frame
    sitting a few pixels away from the real frame.
    """
    if not primary_frame_detected:
        return {
            "consistent": None,
            "confidence": 0.0,
            "explanation": "No primary glasses frame was provided for analysis."
        }

    if not secondary_ghost_detected:
        return {
            "consistent": True,
            "confidence": 0.70,
            "explanation": "No secondary/ghost frame detected."
        }

    # Ghost present
    if offset_px is not None and offset_px <= offset_tolerance_px:
        # Very small offset can sometimes be anti-aliasing; still suspicious
        conf = 0.55
        explanation = (
            f"Possible ghost frame detected with very small offset ({offset_px:.1f}px). "
            "May be anti-aliasing, but still worth noting."
        )
        consistent = True  # soft pass
    else:
        conf = 0.85
        explanation = (
            f"Ghost/doubled frame detected"
            + (f" with offset {offset_px:.1f}px" if offset_px is not None else "")
            + ". This is a frequent AI generation artifact."
        )
        consistent = False

    return {
        "consistent": consistent,
        "confidence": conf,
        "offset_px": offset_px,
        "explanation": explanation
    }


def check_frame_bleeding(
    frame_hair_overlap: bool,
    frame_skin_overlap: bool,
    bleeding_severity: float = 0.0
) -> Dict[str, Any]:
    """
    Check for frame edges that incorrectly blend/bleed into hair or skin.

    bleeding_severity: 0.0 (none) → 1.0 (severe)
    """
    if not frame_hair_overlap and not frame_skin_overlap:
        return {
            "consistent": True,
            "confidence": 0.65,
            "explanation": "No frame-to-hair or frame-to-skin bleeding detected."
        }

    if bleeding_severity < 0.25:
        return {
            "consistent": True,
            "confidence": 0.50,
            "explanation": "Minor possible blending at frame edges (low severity)."
        }

    locations = []
    if frame_hair_overlap:
        locations.append("hair")
    if frame_skin_overlap:
        locations.append("skin")

    return {
        "consistent": False,
        "confidence": min(0.90, 0.55 + bleeding_severity * 0.4),
        "explanation": (
            f"Frame bleeding into {' and '.join(locations)} detected "
            f"(severity {bleeding_severity:.2f}). "
            "Clean optical edges should not dissolve into adjacent hair or skin."
        )
    }


def check_lens_reflections(
    expected_lens_reflections: int,
    observed_lens_reflections: int,
    reflections_consistent_with_light: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Basic check on reflections visible inside the lenses.
    """
    if expected_lens_reflections == 0 and observed_lens_reflections == 0:
        return {
            "consistent": True,
            "confidence": 0.40,
            "explanation": "No lens reflections expected or observed."
        }

    if observed_lens_reflections == 0 and expected_lens_reflections > 0:
        return {
            "consistent": False,
            "confidence": 0.60,
            "explanation": (
                "Expected lens reflections are missing. "
                "Glasses under directional light usually show some specular response."
            )
        }

    if reflections_consistent_with_light is False:
        return {
            "consistent": False,
            "confidence": 0.75,
            "explanation": (
                "Lens reflections are present but do not align with the "
                "overall light direction of the scene."
            )
        }

    return {
        "consistent": True,
        "confidence": 0.60,
        "explanation": (
            f"Lens reflections look plausible "
            f"({observed_lens_reflections} observed)."
        )
    }


def analyze_glasses_artifacts(
    primary_frame_detected: bool = False,
    secondary_ghost_detected: bool = False,
    ghost_offset_px: Optional[float] = None,
    frame_hair_overlap: bool = False,
    frame_skin_overlap: bool = False,
    bleeding_severity: float = 0.0,
    expected_lens_reflections: int = 0,
    observed_lens_reflections: int = 0,
    reflections_consistent_with_light: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Full glasses/frame artifact pass.
    """
    results: Dict[str, Any] = {
        "module": "glasses_artifacts",
        "ghost_frame_check": None,
        "bleeding_check": None,
        "lens_reflection_check": None,
        "consistent": None,
        "confidence": 0.0,
        "explanation": ""
    }

    evidence = []  # (score, confidence, severity)

    # Ghost frame
    ghost = check_ghost_frame(
        primary_frame_detected, secondary_ghost_detected, ghost_offset_px
    )
    results["ghost_frame_check"] = ghost
    if ghost["consistent"] is not None:
        evidence.append((
            1.0 if ghost["consistent"] else 0.0,
            ghost["confidence"],
            0.80   # ghost frames are high-value signals
        ))

    # Bleeding
    bleeding = check_frame_bleeding(
        frame_hair_overlap, frame_skin_overlap, bleeding_severity
    )
    results["bleeding_check"] = bleeding
    if bleeding["consistent"] is not None:
        evidence.append((
            1.0 if bleeding["consistent"] else 0.0,
            bleeding["confidence"],
            0.70
        ))

    # Lens reflections
    lens = check_lens_reflections(
        expected_lens_reflections,
        observed_lens_reflections,
        reflections_consistent_with_light
    )
    results["lens_reflection_check"] = lens
    if lens["consistent"] is not None:
        evidence.append((
            1.0 if lens["consistent"] else 0.0,
            lens["confidence"],
            0.50
        ))

    if not evidence:
        results["explanation"] = "No glasses-related measurements provided."
        results["confidence"] = 0.0
        results["consistent"] = None
        return results

    total_weight = sum(conf * sev for _, conf, sev in evidence)
    if total_weight > 0:
        weighted = sum(s * conf * sev for s, conf, sev in evidence) / total_weight
    else:
        weighted = 0.5

    results["consistent"] = weighted >= 0.55
    results["confidence"] = round(min(1.0, total_weight / 1.4), 3)

    if results["consistent"]:
        results["explanation"] = (
            "Glasses/frame region appears geometrically and optically plausible "
            f"(confidence {results['confidence']:.2f})."
        )
    else:
        results["explanation"] = (
            "Glasses/frame artifacts detected (ghosting, bleeding, or "
            f"implausible lens behavior). Confidence {results['confidence']:.2f}."
        )

    return results
