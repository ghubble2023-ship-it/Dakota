"""signal_schema.py — contract every forensic module returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional

STATUS_LEVELS = [
    "clear",
    "weak_discrepancy",
    "moderate_discrepancy",
    "strong_discrepancy",
    "not_available",
]

STATUS_RANK = {s: i for i, s in enumerate(STATUS_LEVELS)}


@dataclass
class SignalResult:
    signal_name: str
    score: float
    confidence: float
    status: str
    findings: List[str] = field(default_factory=list)
    raw_metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    contradicts: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.status not in STATUS_LEVELS:
            raise ValueError(f"Invalid status '{self.status}'. Must be one of {STATUS_LEVELS}")
        if not 0 <= self.score <= 100:
            raise ValueError(f"score must be 0–100, got {self.score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0–1, got {self.confidence}")

    def is_strong(self) -> bool:
        return self.status == "strong_discrepancy"

    def is_moderate_or_worse(self) -> bool:
        return self.status in ("moderate_discrepancy", "strong_discrepancy")


@dataclass
class AnalysisReport:
    verdict: str
    overall_score: float
    overall_confidence: float
    top_reasons: List[str]
    signal_table: List[Dict]
    caution: str

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "overall_score": round(self.overall_score, 1),
            "overall_confidence": round(self.overall_confidence, 1),
            "top_reasons": self.top_reasons,
            "signal_table": self.signal_table,
            "caution": self.caution,
        }
