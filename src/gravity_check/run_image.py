"""
Single-image runner: extract → engine → optional adversarial head.
"""

import sys
import json
from pathlib import Path

from .measurement_frontend import extract_from_path
from .engine import run_gravity_check
from .features import bundle_and_report_to_vector, vector_to_dict
from .adversarial_head import AdversarialHead


def _default_head_path() -> Path:
    here = Path(__file__).resolve()
    return here.parents[2] / "docs" / "runs" / "adversarial_head.json"


def analyze_image(image_path: str, use_head: bool = True) -> dict:
    bundle = extract_from_path(image_path)
    kwargs = bundle.to_engine_kwargs()

    if not kwargs.get("object_heights_px"):
        return {
            "error": "No objects extracted",
            "notes": bundle.notes,
            "score": None,
            "adjusted_score": None,
        }

    report = run_gravity_check(**kwargs)
    vec = bundle_and_report_to_vector(report=report, engine_kwargs=kwargs)

    adjusted = report.get("score")
    head_used = False
    if use_head:
        head_path = _default_head_path()
        if head_path.is_file():
            head = AdversarialHead()
            try:
                head.load(str(head_path))
                if head.trained:
                    adjusted = head.predict_score(vec)
                    head_used = True
            except Exception:
                pass

    return {
        "summary": report.get("summary"),
        "score": report.get("score"),
        "adjusted_score": adjusted,
        "head_used": head_used,
        "flags": report.get("flags", []),
        "modules_run": report.get("modules_run", []),
        "extraction_notes": bundle.notes,
        "object_count": len(bundle.objects),
        "shadow_count": len(bundle.shadows),
        "scoring": report.get("scoring"),
        "features": vector_to_dict(vec),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.gravity_check.run_image <image_path>")
        sys.exit(1)
    result = analyze_image(sys.argv[1])
    print(json.dumps(result, indent=2, default=str))
