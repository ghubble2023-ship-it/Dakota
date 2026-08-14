"""
Gravity Check - Weighted Evidence Scoring
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Evidence:
    module: str
    supports_consistency: bool
    confidence: float
    severity: float
    weight: float = 1.0
    note: str = ""

    @property
    def signed_strength(self) -> float:
        sign = 1.0 if self.supports_consistency else -1.0
        return sign * self.confidence * self.severity * self.weight


def score_from_evidence(evidence_list: List[Evidence]) -> Dict[str, Any]:
    if not evidence_list:
        return {
            "score": 0.5,
            "confidence": 0.0,
            "interpretation": "No evidence available. Score is neutral.",
            "evidence_count": 0,
            "net_strength": 0.0
        }

    net_strength = sum(e.signed_strength for e in evidence_list)
    total_possible = sum(e.confidence * e.severity * e.weight for e in evidence_list)

    if total_possible > 0:
        normalized = net_strength / total_possible
    else:
        normalized = 0.0

    score = max(0.0, min(1.0, 0.5 + 0.5 * normalized))
    score_confidence = min(1.0, total_possible / 2.5)

    if score >= 0.75:
        interpretation = "Strong evidence of physical consistency."
    elif score >= 0.60:
        interpretation = "Mostly consistent. Minor or low-confidence issues only."
    elif score >= 0.45:
        interpretation = "Mixed or weak evidence. Inconclusive."
    elif score >= 0.30:
        interpretation = "Notable inconsistencies detected."
    else:
        interpretation = "Strong evidence of physical inconsistency."

    return {
        "score": round(score, 3),
        "confidence": round(score_confidence, 3),
        "interpretation": interpretation,
        "evidence_count": len(evidence_list),
        "net_strength": round(net_strength, 3),
        "details": [
            {
                "module": e.module,
                "supports_consistency": e.supports_consistency,
                "confidence": e.confidence,
                "severity": e.severity,
                "signed_strength": round(e.signed_strength, 3),
                "note": e.note
            }
            for e in evidence_list
        ]
    }


def _simple_evidence(result: Dict[str, Any], module: str, severity: float) -> Optional[Evidence]:
    if not result or result.get("consistent") is None:
        return None
    return Evidence(
        module=module,
        supports_consistency=bool(result["consistent"]),
        confidence=float(result.get("confidence", 0.5)),
        severity=severity,
        note=result.get("explanation", "")[:120]
    )


def evidence_from_spatial(spatial_result: Dict[str, Any]) -> List[Evidence]:
    items = []
    depth = spatial_result.get("depth_ordering", {})
    if depth and depth.get("consistent") is not None:
        items.append(Evidence(
            module="spatial_depth_ordering",
            supports_consistency=bool(depth["consistent"]),
            confidence=0.70 if depth["consistent"] else 0.75,
            severity=0.55,
            note=depth.get("explanation", "")[:120]
        ))
    vanishing = spatial_result.get("vanishing_point", {})
    if vanishing and vanishing.get("status") in ("ok", "inconsistent"):
        items.append(Evidence(
            module="spatial_vanishing_point",
            supports_consistency=vanishing.get("status") == "ok",
            confidence=0.60,
            severity=0.45,
            note=vanishing.get("explanation", "")[:120]
        ))
    return items


def build_score(report: Dict[str, Any]) -> Dict[str, Any]:
    evidence: List[Evidence] = []
    modules = report.get("modules", {})

    # High-value geometric modules
    for key, severity in [
        ("shadow_direction", 0.85),
        ("lighting_geometry", 0.70),
        ("reflections", 0.65),
        ("glasses_artifacts", 0.80),
    ]:
        ev = _simple_evidence(modules.get(key), key, severity)
        if ev:
            evidence.append(ev)

    spatial = modules.get("spatial_measurement")
    if spatial:
        evidence.extend(evidence_from_spatial(spatial))

    return score_from_evidence(evidence)
