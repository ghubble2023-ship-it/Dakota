# Dakota — Project AI / Gravity Check

Physics-based visual inspection system for detecting AI-generated and manipulated images.

## Core Principle

Every finding must be explainable through real optical or geometric behavior.  
No black-box pattern matching. No unexplained confidence scores.

## What is included

| Module | Role |
|--------|------|
| `measurement_frontend.py` | OpenCV extraction: objects, shadows, edges, lighting cues, reflection proxies, glasses heuristics |
| `engine.py` | Orchestrates all geometric modules |
| `scoring.py` | Weighted evidence → final score |
| `spatial_measurement.py` | Depth ordering, scale, vanishing |
| `shadow_direction.py` | Shadow vector consistency |
| `lighting_geometry.py` | Bright-side vs light direction |
| `reflections.py` | Reflection count / geometry |
| `glasses_artifacts.py` | Frame ghosting, lens reflections |
| `edge_bleeding.py` | Soft edges / halo artifacts |
| `run_image.py` | Single-image CLI |
| `run_batch.py` | Batch evaluation runner |

## Gravity Check Order

1. Spatial Measurement (required first)
2. Shadow Analysis
3. Lighting Geometry
4. Reflections
5. Glasses & Frame Artifacts
6. Edge & Bleeding
7. Weighted evidence scoring

## Current Status (2026-08-14)

- [x] Full module set present and wired
- [x] Measurement front-end extracts lighting, reflections, glasses signals
- [x] Engine + scoring pipeline runnable
- [x] Batch runner (`run_batch.py`)
- [x] Live automated runs with all modules active
- [ ] Clean real vs fake separation (not yet on current samples)
- [ ] Larger labeled evaluation set + better object filters

### Latest full run (all modules live)

| Label | Avg Score |
|-------|-----------|
| real  | 0.734 |
| fake  | 0.923 |

No reliable separation yet. See `docs/runs/` for JSON outputs.

## How to run

```bash
pip install -r requirements.txt

# Single image
python -m src.gravity_check.run_image path/to/image.jpg

# Batch
python -m src.gravity_check.run_batch path/to/real_dir path/to/fake_dir --limit 20 --out docs/runs/batch.json
```

## Next Priority

1. Tighten object filters (reduce false depth conflicts / empty extractions)
2. More full-body + strong-shadow samples
3. Larger evaluation set and re-measure separation
