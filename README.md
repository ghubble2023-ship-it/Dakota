# Dakota — Project AI / Gravity Check

Physics-based visual inspection system for detecting AI-generated and manipulated images.

## Core Principle

Every finding must be explainable through real optical or geometric behavior.  
No black-box pattern matching. No unexplained confidence scores.

## Repository Structure

```
Dakota/
├── docs/
│   ├── gravity-check-checklist.md   ← Master inspection checklist (locked)
│   └── samples/
│       ├── real/                     ← Real photos for testing
│       └── ai_generated/             ← AI-generated photos for testing
├── src/
│   └── gravity_check/
│       ├── __init__.py               ← Package exports
│       ├── spatial_measurement.py    ← Required FIRST step
│       └── shadow_direction.py       ← Shadow consistency module
├── .github/workflows/               ← CI checks
├── requirements.txt
└── README.md
```

## Gravity Check Order

1. **Spatial Measurement** (required first)  
   Depth relationships, relative scale, camera-to-subject, subject-to-background
2. Shadow Analysis  
3. Lighting Geometry  
4. Reflections  
5. Glasses & Frame Artifacts  
6. Edge & Bleeding Issues  
7. Overall Consistency  

## Current Status

- [x] Core principle locked
- [x] Master checklist locked (`docs/gravity-check-checklist.md`)
- [x] Spatial Measurement module implemented and in correct location
- [x] Shadow Direction module implemented
- [x] Package structure cleaned and exports working
- [x] Sample folders prepared
- [ ] Additional detection modules (lighting, reflections, glasses, edges)
- [ ] Evaluation pipeline against labeled dataset

## How to use the modules

```python
from src.gravity_check import spatial_report, analyze_shadow_consistency

# Always run spatial measurement first
report = spatial_report(
    object_heights_px=[420.0, 280.0],
    object_bottoms_y=[310.0, 520.0],
    image_height=720.0
)

# Then shadow analysis
shadow_result = analyze_shadow_consistency([
    (0.9, -0.4),
    (0.85, -0.5)
])
```

## Next Priority

1. Add Lighting Geometry module
2. Add Reflections module
3. Build simple evaluation runner against sample folders
