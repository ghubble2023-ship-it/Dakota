"""
Gravity Check - Weighted Evidence Scoring

Replaces a crude plus/minus tally with a proper evidence model:

- Each module contributes evidence, not just a pass/fail
- Evidence has direction (supports real vs supports fake)
- Evidence has strength (confidence)
- Evidence has severity (how serious a failure is)
- Final score is a weighted combination of all available evidence

Score range: 0.0 (strong evidence of inconsistency) → 1.0 (strong evidence of physical consistency)
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Evidence:
    """One piece of evidence from a module."""
    module: str
    supports_consistency: bool          # True = evidence for real/consistent, False = against
    confidence: float                   # 0.0 → 1.0
    severity: float                     # 0.0 → 1.0 (how important this check is)
    weight: float = 1.0                 # optional extra multiplier
    note: str = ""

    @property
    def signed_strength(self) -> float:
        """Positive = supports consistency, negative = opposes it."""
        sign = 1.0 if self.supports_consistency else -1.0
        return sign * self.confidence * self.severity * self.weight


def score_from_evidence(evidence_list: List[Evidence]) -> Dict[str, Any]:
    """
    Combine multiple evidence items into a final score.

    Returns:
        score          : float 0–1
        confidence     : how much total evidence we actually had
        interpretation : plain language
        evidence_count : number of evidence items used
    """
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

    # Normalize net strength into roughly -1 → +1, then map to 0 → 1
    if total_possible > 0:
        normalized = net_strength / total_possible
    else:
        normalized = 0.0

    score = 0.5 + 0.5 * normalized
    score = max(0.0, min(1.0, score))

    # Overall confidence in the score itself (how much evidence we had)
    score_confidence = min(1.0, total_possible / 2.5)  # soft saturation

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


def evidence_from_shadow(shadow_result: Dict[str, Any]) -> Optional[Evidence]:
    """Convert shadow module output into Evidence."""
    if not shadow_result:
        return None

    consistent = shadow_result.get("consistent")
    confidence = float(shadow_result.get("confidence", 0.5))

    if consistent is None:
        return None

    return Evidence(
        module="shadow_direction",
        supports_consistency=bool(consistent),
        confidence=confidence,
        severity=0.85,          # shadow conflicts are high-value evidence
        note=shadow_result.get("explanation", "")[:120]
    )


def evidence_from_spatial(spatial_result: Dict[str, Any]) -> List[Evidence]:
    """Convert spatial module output into one or more Evidence items."""
    items = []

    depth = spatial_result.get("depth_ordering", {})
    if depth:
        consistent = depth.get("consistent")
        if consistent is not None:
            # Depth ordering is useful but usually lower severity than clear shadow conflicts
            items.append(Evidence(
                module="spatial_depth_ordering",
                supports_consistency=bool(consistent),
                confidence=0.70 if consistent else 0.75,
                severity=0.55,
                note=depth.get("explanation", "")[:120]
            ))

    vanishing = spatial_result.get("vanishing_point", {})
    if vanishing and vanishing.get("status") in ("ok", "inconsistent"):
        supports = vanishing.get("status") == "ok"
        items.append(Evidence(
            module="spatial_vanishing_point",
            supports_consistency=supports,
            confidence=0.60,
            severity=0.45,
            note=vanishing.get("explanation", "")[:120]
        ))

    return items


def build_score(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Take a full engine report and attach a weighted score.
    """
    evidence: List[Evidence] = []

    modules = report.get("modules", {})

    # Shadow evidence
    shadow = modules.get("shadow_direction")
    ev = evidence_from_shadow(shadow) if shadow else None
    if ev:
        evidence.append(ev)

    # Spatial evidence
    spatial = modules.get("spatial_measurement")
    if spatial:
        evidence.extend(evidence_from_spatial(spatial))

    # Future modules will add their own evidence converters here

    scoring = score_from_evidence(evidence)
    return scoring
