"""
Gravity Check — Phase 3 secondary geometry

Runs after foundation and catchlights. Measurement-driven. Missing
observations are incomplete (consistent=None), not automatic fails.

Covers the locked phase-3 list that is not already a standalone module:
contact shadows, hair/occlusion, specular mismatch, support/weight,
face-vs-body light split.

Glasses, reflections, and edges stay in their own modules and are
invoked by the engine in this same phase.
"""

from typing import Any, Dict, List, Optional, Tuple


def analyze_phase3_secondary(
    contact_shadow_present: Optional[bool] = None,
    contact_shadow_expected: Optional[bool] = None,
    hair_occlusion_ok: Optional[bool] = None,
    specular_matches_light: Optional[bool] = None,
    support_weight_ok: Optional[bool] = None,
    face_body_light_split: Optional[bool] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "module": "phase3_secondary",
        "phase": 3,
        "contact_shadow": None,
        "hair_occlusion": None,
        "specular": None,
        "support_weight": None,
        "face_body_split": None,
        "consistent": None,
        "confidence": 0.0,
        "status": "incomplete",
        "explanation": "",
    }

    evidence: List[Tuple[float, float, float]] = []

    if contact_shadow_expected is False:
        result["contact_shadow"] = {
            "consistent": True,
            "confidence": 0.40,
            "explanation": "No contact shadow expected (subject not planted on a surface).",
        }
        evidence.append((1.0, 0.40, 0.30))
    elif contact_shadow_present is False and contact_shadow_expected is not False:
        result["contact_shadow"] = {
            "consistent": False,
            "confidence": 0.75,
            "explanation": (
                "Body meets a surface but the contact shadow is missing. "
                "Weight should darken the contact line."
            ),
        }
        evidence.append((0.0, 0.75, 0.80))
    elif contact_shadow_present is True:
        result["contact_shadow"] = {
            "consistent": True,
            "confidence": 0.60,
            "explanation": "Contact shadow present where the body meets a surface.",
        }
        evidence.append((1.0, 0.60, 0.70))

    if hair_occlusion_ok is False:
        result["hair_occlusion"] = {
            "consistent": False,
            "confidence": 0.70,
            "explanation": "Hair/occlusion error: strands do not sit in front of or behind the surface they claim.",
        }
        evidence.append((0.0, 0.70, 0.65))
    elif hair_occlusion_ok is True:
        result["hair_occlusion"] = {
            "consistent": True,
            "confidence": 0.50,
            "explanation": "Hair occlusion looks physically ordered.",
        }
        evidence.append((1.0, 0.50, 0.45))

    if specular_matches_light is False:
        result["specular"] = {
            "consistent": False,
            "confidence": 0.70,
            "explanation": "Specular highlights on skin or accessories fight the room light.",
        }
        evidence.append((0.0, 0.70, 0.70))
    elif specular_matches_light is True:
        result["specular"] = {
            "consistent": True,
            "confidence": 0.55,
            "explanation": "Speculars agree with the established light.",
        }
        evidence.append((1.0, 0.55, 0.50))

    if support_weight_ok is False:
        result["support_weight"] = {
            "consistent": False,
            "confidence": 0.80,
            "explanation": "Support/weight failure: the figure does not rest on the surface with real load.",
        }
        evidence.append((0.0, 0.80, 0.85))
    elif support_weight_ok is True:
        result["support_weight"] = {
            "consistent": True,
            "confidence": 0.55,
            "explanation": "Support and weight read as planted.",
        }
        evidence.append((1.0, 0.55, 0.60))

    if face_body_light_split is True:
        result["face_body_split"] = {
            "consistent": False,
            "confidence": 0.75,
            "explanation": "Face and body are lit as if they do not share the same room.",
        }
        evidence.append((0.0, 0.75, 0.80))
    elif face_body_light_split is False:
        result["face_body_split"] = {
            "consistent": True,
            "confidence": 0.55,
            "explanation": "Face and body share one light story.",
        }
        evidence.append((1.0, 0.55, 0.55))

    if not evidence:
        result["status"] = "incomplete"
        result["consistent"] = None
        result["confidence"] = 0.0
        result["explanation"] = (
            "Phase 3 secondary ran. No contact/hair/specular/support/"
            "face-body measurements supplied. Incomplete, not a fail."
        )
        return result

    total_weight = sum(c * v for _, c, v in evidence)
    weighted = sum(s * c * v for s, c, v in evidence) / total_weight if total_weight else 0.5
    result["consistent"] = weighted >= 0.55
    result["confidence"] = round(min(1.0, total_weight / 1.4), 3)
    result["status"] = "complete"
    if result["consistent"]:
        result["explanation"] = (
            "Phase 3 secondary geometry is plausible "
            f"(confidence {result['confidence']:.2f})."
        )
    else:
        result["explanation"] = (
            "Phase 3 secondary geometry failed "
            f"(confidence {result['confidence']:.2f})."
        )
    return result
