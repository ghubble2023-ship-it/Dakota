# Dakota — Project AI / Gravity Check

Physics-based visual inspection system for detecting AI-generated and manipulated images.

## Core Principle

Every finding must be explainable through real optical or geometric behavior.
No black-box pattern matching. No unexplained confidence scores.

## Structure
docs/
gravity-check-checklist.md   ← Master inspection checklist
src/
gravity_check/
shadow_direction.py        ← First detection module
samples/
real/                        ← Real photos for testing
ai_generated/                ← AI-generated photos for testing
## Current Status

- Spatial measurement is the required first step
- Shadow direction consistency module is implemented
- Checklist is locked and versioned

## Gravity Check Order

1. Spatial Measurement (depth, camera distance, subject-to-background, light angle)
2. Shadow Analysis
3. Lighting Geometry
4. Reflections
5. Glasses & Frame Artifacts
6. Edge & Bleeding Issues
7. Overall Consistency

## Next

- Formalize spatial measurement helpers
- Add more detection modules
- Build evaluation pipeline against labeled samples
