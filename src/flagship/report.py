"""Structured report. Language is physical, not legal."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal

Finding = Literal["consistent", "conflict", "insufficient", "second_generation"]


@dataclass
class LaneFinding:
    lane: str
    finding: Finding
    confidence: float
    mechanism: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FlagshipReport:
    source: str
    lanes: List[LaneFinding]
    notes: List[str] = field(default_factory=list)

    @property
    def headline(self) -> str:
        order = {
            "second_generation": 0,
            "conflict": 1,
            "insufficient": 2,
            "consistent": 3,
        }
        if not self.lanes:
            return "insufficient"
        return min(self.lanes, key=lambda x: order[x.finding]).finding

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "headline": self.headline,
            "lanes": [lane.to_dict() for lane in self.lanes],
            "notes": list(self.notes),
        }
