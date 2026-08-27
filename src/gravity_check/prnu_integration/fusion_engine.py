"""Side-lane combiner. NOT the Gravity Check engine. No real/fake stamp."""
from __future__ import annotations
from pathlib import Path
from typing import List
import yaml
from .signal_schema import SignalResult, AnalysisReport, STATUS_RANK

CAUTION_TEXT = (
    "Side-lane fusion only. Gravity Check physics report is primary. "
    "This is not proof of authenticity or manipulation and is not a scam verdict."
)
_DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"

class FusionEngine:
    def __init__(self, config_path: str | None = None):
        path = Path(config_path) if config_path else _DEFAULT_CONFIG
        with open(path) as f:
            self.config = yaml.safe_load(f)
        self.weights = self.config["signal_weights"]
        self.thresholds = self.config["thresholds"]
        self.min_confidence = self.config["min_confidence"]
        self.overrides = self.config.get("overrides", {})

    def analyze(self, signals: List[SignalResult]) -> AnalysisReport:
        available = [s for s in signals if s.status != "not_available"]
        if not available:
            return self._build_report("Inconclusive", 0.0, 0.0,
                ["No forensic signals could be produced from this image."], signals)
        total_weight = sum(self.weights.get(s.signal_name, 0.0) for s in available)
        if total_weight <= 0:
            return self._build_report("Inconclusive", 0.0, 0.0,
                ["Signals present but none have configured weights."], signals)
        weighted_score = sum(s.score * self.weights.get(s.signal_name, 0.0) for s in available) / total_weight
        weighted_conf = sum(s.confidence * self.weights.get(s.signal_name, 0.0) for s in available) / total_weight
        missing_fraction = (len(signals) - len(available)) / max(len(signals), 1)
        weighted_conf *= max(0.4, 1.0 - 0.15 * missing_fraction)
        sm_cfg = self.overrides.get("smoking_gun", {})
        smoking_gun_fired = any(
            s.is_strong() and s.confidence >= sm_cfg.get("min_signal_confidence", 0.70)
            for s in available
        )
        ms_cfg = self.overrides.get("multi_signal_corroboration", {})
        bump = sum(1 for s in available if s.is_moderate_or_worse()) >= ms_cfg.get("min_signals", 2)
        conflict = False
        for s in available:
            if s.contradicts and s.is_moderate_or_worse():
                for other in available:
                    if other.signal_name in s.contradicts and other.is_moderate_or_worse():
                        conflict = True
        if conflict:
            verdict, reasons = "Inconclusive", ["Signals disagree. Read the Gravity Check physics report."]
        elif weighted_conf < (self.min_confidence / 100.0):
            verdict = "Inconclusive"
            reasons = [f"Overall confidence too low ({weighted_conf * 100:.0f}/100)."]
            missing = [s.signal_name for s in signals if s.status == "not_available"]
            if missing:
                reasons.append("Signals unavailable: " + ", ".join(missing))
        else:
            t = self.thresholds
            if weighted_score < t["no_discrepancies"]:
                verdict = "No discrepancies detected"
            elif weighted_score < t["possible_manipulation"]:
                verdict = "Possible manipulation"
            elif weighted_score < t["likely_manipulation"]:
                verdict = "Likely manipulation"
            else:
                verdict = "Strong manipulation indicators"
            ranked = sorted(available, key=lambda s: s.score, reverse=True)
            reasons = []
            for s in ranked[:3]:
                reasons.append(f"[{s.signal_name}] " + (s.findings[0] if s.findings else s.status))
        if smoking_gun_fired and not conflict:
            if verdict in ("No discrepancies detected", "Inconclusive", "Possible manipulation"):
                if weighted_conf >= (self.min_confidence / 100.0):
                    verdict = "Likely manipulation"
        if bump and not conflict:
            if verdict == "Possible manipulation":
                verdict = "Likely manipulation"
            elif verdict == "Likely manipulation":
                verdict = "Strong manipulation indicators"
        return self._build_report(verdict, weighted_score, weighted_conf, reasons, signals)

    def _build_report(self, verdict, score, conf, reasons, signals) -> AnalysisReport:
        table = [{
            "signal": s.signal_name,
            "evidence_score": round(s.score, 1),
            "confidence": round(s.confidence * 100, 1),
            "status": s.status,
        } for s in signals]
        return AnalysisReport(verdict, score, conf * 100, reasons, table, CAUTION_TEXT)
