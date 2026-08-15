"""
Simple runner: image path → OpenCV measurements → Gravity Check engine.

Usage (from repo root):
    python -m src.gravity_check.run_image path/to/photo.jpg
"""

import sys
import json
from .measurement_frontend import extract_from_path
from .engine import run_gravity_check


def analyze_image(image_path: str) -> dict:
    bundle = extract_from_path(image_path)
    kwargs = bundle.to_engine_kwargs()

    # Need at least some spatial data
    if not kwargs.get("object_heights_px"):
        return {
            "error": "No objects extracted",
            "notes": bundle.notes,
            "score": None
        }

    report = run_gravity_check(**kwargs)

    return {
        "summary": report.get("summary"),
        "score": report.get("score"),
        "flags": report.get("flags", []),
        "modules_run": report.get("modules_run", []),
        "extraction_notes": bundle.notes,
        "object_count": len(bundle.objects),
        "shadow_count": len(bundle.shadows),
        "scoring": report.get("scoring"),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.gravity_check.run_image <image_path>")
        sys.exit(1)

    result = analyze_image(sys.argv[1])
    print(json.dumps(result, indent=2))
