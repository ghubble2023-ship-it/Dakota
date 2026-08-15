# Dakota — Project AI / Gravity Check

Physics-based visual inspection system for detecting AI-generated and manipulated images.

## Core Principle

Every finding must be explainable through real optical or geometric behavior.  
No black-box pattern matching. No unexplained confidence scores.

## Repository Structure

```
Dakota/
├── docs/
│   ├── gravity-check-checklist.md
│   ├── samples/
│   │   ├── real/
│   │   └── ai_generated/
│   └── runs/                        ← automated evaluation outputs
├── src/
│   └── gravity_check/
│       ├── measurement_frontend.py  ← OpenCV contour / shadow extraction
│       ├── engine.py                ← orchestrates all modules
│       ├── scoring.py               ← weighted evidence scoring
│       ├── spatial_measurement.py
│       ├── shadow_direction.py
│       ├── lighting_geometry.py
│       ├── reflections.py
│       ├── glasses_artifacts.py
│       ├── edge_bleeding.py
│       └── run_image.py             ← CLI entry point
├── .github/workflows/
├── requirements.txt
└── README.md
```

## Gravity Check Order

1. **Spatial Measurement** (required first)
2. Shadow Analysis
3. Lighting Geometry
4. Reflections
5. Glasses & Frame Artifacts
6. Edge & Bleeding
7. Weighted evidence scoring

## Current Status (2026-08-14)

- [x] Core principle locked
- [x] Full module set present (spatial, shadow, lighting, reflections, glasses, edge)
- [x] Measurement front-end (OpenCV) implemented
- [x] Engine + scoring pipeline wired and runnable
- [x] Live automated run completed (OpenCV 5.0.0)
- [ ] Clean real vs fake separation (not yet achieved on current samples)
- [ ] Larger labeled evaluation set

### Latest automated run (2026-08-14)

| Label | Avg Score | Range |
|-------|-----------|-------|
| real  | 0.563     | 0.391 – 0.714 |
| fake  | 0.655     | 0.471 – 0.759 |

Dominant flags on both classes: `depth_ordering_conflict`, `shadow_inconsistency`.  
No reliable separation yet. Full results in `docs/runs/gravity_check_run_2026-08-14.json`.

## How to run

```bash
pip install -r requirements.txt
python -m src.gravity_check.run_image path/to/image.jpg
```

## Next Priority

1. Improve contour / object filtering to reduce false depth conflicts on face crops
2. Add more full-body + strong-shadow samples
3. Expand evaluation set and re-measure
