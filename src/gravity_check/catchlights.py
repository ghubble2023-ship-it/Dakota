"""
Gravity Check — Catchlights (mandatory phase 2)

Runs only after foundation (phase 1) has produced a room / light angle.
This module reasons about *measured* catchlights. It does not close the
pixel-measurement gap (detect eye → isolate specular → mask to black →
measure height/width/aspect/edge/structure). That detector is still open.

If no eye/glint measurements are supplied, the phase still runs and
returns consistent=None (incomplete / UNKNOWN). Missing data is not a fail.
"""

from typing import Any, Dict, List, Optional, Tuple
import math


def _angle_difference(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def _aspect(width: float, height: float) -> Optional[float]:
    if width <= 0 or height <= 0:
        return None
    return width / height


def analyze_catchlights(
    light_angle: Optional[float] = None,
    eyeball_curvature_ok: Optional[bool] = None,
    left_eye_center: Optional[Tuple[float, float]] = None,
    right_eye_center: Optional[Tuple[float, float]] = None,
    left_glint: Optional[Dict[str, float]] = None,
    right_glint: Optional[Dict[str, float]] = None,
    identical_structure: Optional[bool] = None,
    extra_streaks_fight_light: Optional[bool] = None,
    direction_tolerance_degrees: float = 45.0,
    twin_aspect_tol: float = 0.08,
    twin_offset_px: float = 1.5,
) -> Dict[str, Any]:
    """
    Protocol checks that can run from measurements.

    Each glint dict may contain: cx, cy, width, height, angle_deg
    (angle of the glint relative to its pupil, toward the light).
    """
    result: Dict[str, Any] = {
        "module": "catchlights",
        "phase": 2,
        "measurement_gap_open": True,
        "eyeball_check": None,
        "offset_check": None,
        "twin_check": None,
        "direction_check": None,
        "streak_check": None,
        "consistent": None,
        "confidence": 0.0,
        "status": "incomplete",
        "explanation": "",
    }

    evidence: List[Tuple[float, float, float]] = []

    if eyeball_curvature_ok is False:
        result["eyeball_check"] = {
            "consistent": False,
            "confidence": 0.80,
            "explanation": (
                "Eyeball reads flat or near-flat. Real catchlights need "
                "spherical volume; painted texture on a disc cannot host them."
            ),
        }
        evidence.append((0.0, 0.80, 0.90))
    elif eyeball_curvature_ok is True:
        result["eyeball_check"] = {
            "consistent": True,
            "confidence": 0.55,
            "explanation": "Eyeball curvature / volume accepted as present.",
        }
        evidence.append((1.0, 0.55, 0.40))

    have_pair = bool(left_glint and right_glint)
    have_one = bool(left_glint or right_glint)

    if have_pair and left_eye_center and right_eye_center:
        lx, ly = left_glint.get("cx"), left_glint.get("cy")
        rx, ry = right_glint.get("cx"), right_glint.get("cy")
        if None not in (lx, ly, rx, ry):
            off_l = (float(lx) - left_eye_center[0], float(ly) - left_eye_center[1])
            off_r = (float(rx) - right_eye_center[0], float(ry) - right_eye_center[1])
            delta = math.hypot(off_l[0] - off_r[0], off_l[1] - off_r[1])
            result["offset_check"] = {
                "inter_glint_offset_delta_px": round(delta, 3),
                "left_offset_px": (round(off_l[0], 3), round(off_l[1], 3)),
                "right_offset_px": (round(off_r[0], 3), round(off_r[1], 3)),
                "consistent": None,
                "confidence": 0.50,
                "explanation": (
                    f"Glint-to-pupil offset delta between eyes is {delta:.2f}px. "
                    "Real pair: small nonzero difference. Zero difference plus "
                    "identical shape is the twin-glint failure."
                ),
            }

    if have_pair:
        la = _aspect(float(left_glint.get("width") or 0), float(left_glint.get("height") or 0))
        ra = _aspect(float(right_glint.get("width") or 0), float(right_glint.get("height") or 0))
        twins = False
        reason = []
        if la is not None and ra is not None and abs(la - ra) <= twin_aspect_tol:
            reason.append(f"aspect almost equal ({la:.3f} vs {ra:.3f})")
        lw, lh = float(left_glint.get("width") or 0), float(left_glint.get("height") or 0)
        rw, rh = float(right_glint.get("width") or 0), float(right_glint.get("height") or 0)
        if lw and rw and lh and rh:
            size_delta = abs(lw - rw) + abs(lh - rh)
            if size_delta <= twin_offset_px:
                reason.append(f"size delta {size_delta:.2f}px")
        if identical_structure is True:
            reason.append("structure marked identical")
            twins = True
        if la is not None and ra is not None and abs(la - ra) <= twin_aspect_tol:
            if lw and rw and abs(lw - rw) + abs(lh - rh) <= twin_offset_px:
                twins = True
        result["twin_check"] = {
            "consistent": (not twins) if (la is not None or identical_structure is not None) else None,
            "confidence": 0.75 if twins else 0.55,
            "explanation": (
                "Identical twin catchlights (same size/aspect/structure). "
                "Common generation copy of one glint."
                if twins else
                ("Catchlights are not identical twins. " + ("; ".join(reason) if reason else "Shape differs."))
            ),
        }
        if result["twin_check"]["consistent"] is not None:
            evidence.append((
                1.0 if result["twin_check"]["consistent"] else 0.0,
                result["twin_check"]["confidence"],
                0.85,
            ))

    glint_angles = []
    for g in (left_glint, right_glint):
        if g and g.get("angle_deg") is not None:
            glint_angles.append(float(g["angle_deg"]))
    if light_angle is not None and glint_angles:
        deviations = [_angle_difference(light_angle, a) for a in glint_angles]
        max_dev = max(deviations)
        ok = max_dev <= direction_tolerance_degrees
        result["direction_check"] = {
            "consistent": ok,
            "confidence": round(max(0.0, 1.0 - (sum(deviations) / len(deviations)) / 90.0), 3),
            "max_deviation_deg": max_dev,
            "deviations_deg": deviations,
            "explanation": (
                f"Catchlight direction vs room light: max deviation {max_dev:.1f}\u00b0 "
                f"(tolerance {direction_tolerance_degrees}\u00b0)."
            ),
        }
        evidence.append((
            1.0 if ok else 0.0,
            result["direction_check"]["confidence"],
            0.80,
        ))
    elif have_one and light_angle is None:
        result["direction_check"] = {
            "consistent": None,
            "confidence": 0.0,
            "explanation": "Glint present but phase-1 light angle missing \u2014 cannot test direction.",
        }

    if extra_streaks_fight_light is True:
        result["streak_check"] = {
            "consistent": False,
            "confidence": 0.70,
            "explanation": "Extra streaks in the eye fight the established light direction.",
        }
        evidence.append((0.0, 0.70, 0.70))
    elif extra_streaks_fight_light is False:
        result["streak_check"] = {
            "consistent": True,
            "confidence": 0.50,
            "explanation": "No extra streaks fighting the light.",
        }
        evidence.append((1.0, 0.50, 0.35))

    if not evidence:
        result["status"] = "incomplete"
        result["consistent"] = None
        result["confidence"] = 0.0
        result["explanation"] = (
            "Phase 2 ran. No usable catchlight measurements. "
            "Status UNKNOWN. Pixel detector (eye \u2192 specular mask \u2192 "
            "height/width/aspect/edge) is still open."
        )
        return result

    total_weight = sum(c * v for _, c, v in evidence)
    weighted = sum(s * c * v for s, c, v in evidence) / total_weight if total_weight else 0.5
    result["consistent"] = weighted >= 0.55
    result["confidence"] = round(min(1.0, total_weight / 1.4), 3)
    result["status"] = "complete"
    if result["consistent"]:
        result["explanation"] = (
            "Catchlights do not fight the room light "
            f"(confidence {result['confidence']:.2f}). "
            "Measurement gap still open for automatic isolation."
        )
    else:
        result["explanation"] = (
            "Catchlights conflict with the room (twins, wrong side, "
            f"or streaks). Confidence {result['confidence']:.2f}."
        )
    return result
