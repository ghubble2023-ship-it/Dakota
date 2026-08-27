"""
Dual-lane runner: Gravity Check first, PRNU second.

Primary output is the physics report (phases 1→2→3).
PRNU is a side lane. Fusion is optional and never a scam stamp.

Calibration (Flagship 2, 40 low-res GAN faces): raw noise stats were a
WEAK discriminator (best accuracy 62.5%, Cohen's d 0.28). Do not treat
residual_std alone as proof.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .run_image import analyze_image as analyze_gravity
from .prnu_sensor_noise import analyze_image as analyze_prnu
from .prnu_integration.signal_schema import SignalResult
from .prnu_integration.fusion_engine import FusionEngine


def _gc_consistency_to_evidence(score: Optional[float]) -> float:
    if score is None:
        return 0.0
    return float(np.clip((1.0 - float(score)) * 100.0, 0.0, 100.0))


def gravity_to_signals(gc_report: Dict[str, Any]) -> List[SignalResult]:
    flags = gc_report.get("flags") or []
    empty = "extraction_empty" in flags
    score = gc_report.get("score")
    evidence = _gc_consistency_to_evidence(score)
    phases = gc_report.get("phases") or {}

    def phase_status(name: str) -> str:
        phase = phases.get(name) or {}
        if empty or phase.get("status") == "incomplete":
            return "not_available"
        if evidence >= 70:
            return "strong_discrepancy"
        if evidence >= 40:
            return "moderate_discrepancy"
        if evidence >= 15:
            return "weak_discrepancy"
        return "clear"

    findings = list(flags)
    conf = 0.25 if empty else 0.55
    lighting_status = phase_status("foundation")
    catch_status = phase_status("catchlights")
    geom_status = phase_status("secondary_geometry")

    return [
        SignalResult(
            signal_name="lighting_shadow",
            score=evidence if lighting_status != "not_available" else 0.0,
            confidence=conf if lighting_status != "not_available" else 0.0,
            status=lighting_status,
            findings=findings[:2] or ["Foundation lane from Gravity Check."],
            raw_metrics={"gc_consistency_score": score},
        ),
        SignalResult(
            signal_name="catchlight_eye_geometry",
            score=evidence if catch_status != "not_available" else 0.0,
            confidence=0.20 if catch_status != "not_available" else 0.0,
            status=catch_status,
            findings=["Catchlight measurement gap still open unless glints were supplied."],
        ),
        SignalResult(
            signal_name="general_geometry",
            score=evidence if geom_status != "not_available" else 0.0,
            confidence=conf if geom_status != "not_available" else 0.0,
            status=geom_status,
            findings=["Phase-3 secondary geometry from Gravity Check."],
        ),
    ]


def prnu_to_signal(prnu: Dict[str, Any], fingerprint_used: bool) -> SignalResult:
    residual_std = float(prnu.get("residual_std") or 0.0)
    inconsistency = float(prnu.get("noise_variance_inconsistency") or 0.0)
    match = prnu.get("camera_match") or {}
    findings = [
        f"residual_std={residual_std:.6f}",
        f"noise_variance_inconsistency={inconsistency:.6f}",
    ]
    if not fingerprint_used:
        return SignalResult(
            signal_name="sensor_noise_prnu",
            score=0.0,
            confidence=0.25,
            status="not_available",
            findings=[
                "No camera fingerprint supplied. PCE attribution skipped. "
                "Raw residual stats are a weak discriminator (cal report d=0.28)."
            ] + findings,
            raw_metrics={
                "residual_std": residual_std,
                "noise_variance_inconsistency": inconsistency,
            },
        )

    pce = float(match.get("pce") or 0.0)
    matches = bool(match.get("matches_camera"))
    if matches:
        status, score, conf = "clear", 8.0, 0.70
        findings.insert(0, f"PCE={pce:.2f} matched supplied fingerprint.")
    else:
        status = "moderate_discrepancy" if pce < 60 else "weak_discrepancy"
        score = 55.0 if pce < 60 else 25.0
        conf = 0.55
        findings.insert(0, f"PCE={pce:.2f} did not match supplied fingerprint.")
    return SignalResult(
        signal_name="sensor_noise_prnu",
        score=score,
        confidence=conf,
        status=status,
        findings=findings,
        raw_metrics={"pce": pce, "residual_std": residual_std},
    )


def run_dual_lane(
    image_path: str,
    fingerprint: Optional[np.ndarray] = None,
    fuse: bool = True,
) -> Dict[str, Any]:
    gc = analyze_gravity(image_path, use_head=False)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    prnu = analyze_prnu(image, fingerprint=fingerprint)
    prnu_public = {k: v for k, v in prnu.items() if k != "noise_variance_map"}
    if "noise_variance_map" in prnu and hasattr(prnu["noise_variance_map"], "shape"):
        prnu_public["noise_variance_map_shape"] = list(prnu["noise_variance_map"].shape)
    out: Dict[str, Any] = {
        "primary": "gravity_check",
        "gravity_check": gc,
        "prnu": prnu_public,
        "lane_note": (
            "Gravity Check is primary. PRNU is the second lane. "
            "Fusion is a side report, not a scam/real stamp."
        ),
    }
    if fuse:
        signals = gravity_to_signals(gc) + [
            prnu_to_signal(prnu, fingerprint_used=fingerprint is not None)
        ]
        out["side_fusion"] = FusionEngine().analyze(signals).to_dict()
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.gravity_check.run_dual_lane <image_path>")
        sys.exit(1)
    print(json.dumps(run_dual_lane(sys.argv[1]), indent=2, default=str))
