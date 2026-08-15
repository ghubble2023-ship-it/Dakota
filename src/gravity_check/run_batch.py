"""
Batch evaluation runner for Gravity Check.

Usage (from repo root):
    python -m src.gravity_check.run_batch /path/to/real_dir /path/to/fake_dir
    python -m src.gravity_check.run_batch --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from .run_image import analyze_image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _list_images(folder: str) -> List[str]:
    p = Path(folder)
    if not p.is_dir():
        return []
    files = []
    for f in sorted(p.iterdir()):
        if f.suffix.lower() in IMAGE_EXTS and f.is_file():
            files.append(str(f))
    return files


def run_batch(
    real_dir: Optional[str] = None,
    fake_dir: Optional[str] = None,
    limit: Optional[int] = None,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []

    pairs = []
    if real_dir:
        for path in _list_images(real_dir):
            pairs.append(("real", path))
    if fake_dir:
        for path in _list_images(fake_dir):
            pairs.append(("fake", path))

    if limit is not None:
        if real_dir and fake_dir:
            real_paths = [p for lab, p in pairs if lab == "real"][:limit]
            fake_paths = [p for lab, p in pairs if lab == "fake"][:limit]
            pairs = [("real", p) for p in real_paths] + [("fake", p) for p in fake_paths]
        else:
            pairs = pairs[:limit]

    for label, path in pairs:
        name = os.path.basename(path)
        try:
            r = analyze_image(path)
            results.append({
                "file": name,
                "path": path,
                "label": label,
                "score": r.get("score"),
                "flags": r.get("flags", []),
                "modules_run": r.get("modules_run", []),
                "object_count": r.get("object_count"),
                "shadow_count": r.get("shadow_count"),
                "summary": r.get("summary"),
                "error": r.get("error"),
            })
            print(f"{label:4s} {name}: score={r.get('score')} "
                  f"objs={r.get('object_count')} shadows={r.get('shadow_count')} "
                  f"flags={r.get('flags')}")
        except Exception as e:
            print(f"{label:4s} {name}: ERROR {type(e).__name__}: {e}")
            results.append({
                "file": name, "path": path, "label": label,
                "score": None, "error": f"{type(e).__name__}: {e}"
            })

    real_scores = [r["score"] for r in results if r.get("label") == "real" and r.get("score") is not None]
    fake_scores = [r["score"] for r in results if r.get("label") == "fake" and r.get("score") is not None]

    def _stats(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {"count": 0, "avg": None, "min": None, "max": None}
        return {
            "count": len(vals),
            "avg": round(sum(vals) / len(vals), 3),
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
        }

    report = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline": "measurement_frontend -> engine -> scoring",
        "real_dir": real_dir,
        "fake_dir": fake_dir,
        "results": results,
        "summary_stats": {
            "real": _stats(real_scores),
            "fake": _stats(fake_scores),
        },
    }

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {out_path}")

    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Gravity Check batch evaluation")
    parser.add_argument("real_dir", nargs="?", default=None, help="Folder of real images")
    parser.add_argument("fake_dir", nargs="?", default=None, help="Folder of fake/AI images")
    parser.add_argument("--limit", type=int, default=None, help="Max images per class")
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args(argv)

    if not args.real_dir and not args.fake_dir:
        root = Path(__file__).resolve().parents[3]
        candidates_real = [
            root / "artifacts" / "samples_real",
            Path("/home/workdir/artifacts/samples_real"),
            Path("docs/samples/real"),
        ]
        candidates_fake = [
            root / "artifacts" / "1strealtest_fake",
            Path("/home/workdir/artifacts/1strealtest_fake"),
            Path("docs/samples/ai_generated"),
        ]
        for c in candidates_real:
            if c.is_dir() and any(c.iterdir()):
                args.real_dir = str(c)
                break
        for c in candidates_fake:
            if c.is_dir() and any(c.iterdir()):
                args.fake_dir = str(c)
                break

    if not args.real_dir and not args.fake_dir:
        print("No image folders found. Pass real_dir and/or fake_dir.")
        return 1

    out = args.out
    if out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out = f"docs/runs/gravity_check_batch_{stamp}.json"

    run_batch(args.real_dir, args.fake_dir, limit=args.limit, out_path=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
