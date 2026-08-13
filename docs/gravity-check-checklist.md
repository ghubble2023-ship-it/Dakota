# Gravity Check Checklist

Physics-based inspection list for detecting AI-generated or manipulated images.

Every item must be explainable through real optical or geometric behavior.
Nothing is assumed. Every step is written down.

---

## 0. Spatial Measurement (First Step)

This is the foundation. Complete this before any other checks.

- [ ] Estimate the depth of the room or scene
- [ ] Estimate distance from camera to the main subject
- [ ] Estimate distance from the main subject to the background
- [ ] Identify the approximate angle of the primary light source
- [ ] Estimate relative distances between key objects
- [ ] Check whether scale relationships feel consistent across the scene
- [ ] Note any impossible or contradictory depth cues

If the spatial measurements do not make physical sense, the image is already highly suspect.

---

## 1. Shadow Analysis

- [ ] Shadow direction matches a consistent light source
- [ ] Shadow length is realistic for the estimated light angle and distances
- [ ] Softness of shadow edges matches the light type (hard vs soft light)
- [ ] Multiple objects cast shadows that agree with the same light direction
- [ ] No floating or detached shadows

## 2. Lighting Geometry

- [ ] Highlights and specular reflections point to the same light source
- [ ] Brightness falloff across the scene is physically plausible
- [ ] No contradictory lighting on different parts of the same object
- [ ] Ambient light vs direct light balance looks natural

## 3. Reflections

- [ ] Mirror and reflective surfaces show correct perspective
- [ ] Reflected objects match the real scene geometry
- [ ] Reflection intensity matches the material (glass, metal, water, etc.)
- [ ] No missing or impossible reflections

## 4. Glasses & Frame Artifacts

- [ ] Eyeglass frames have clean, single edges
- [ ] No ghosting or double-frame artifacts
- [ ] No faint duplicate frames offset from the real ones
- [ ] Lenses show consistent refraction and reflection behavior

## 5. Edge & Bleeding Issues

- [ ] Object edges are clean and properly anti-aliased
- [ ] No color bleeding into adjacent areas
- [ ] No unnatural halos or glow around subjects
- [ ] Hair and fine details do not show layered or duplicated edges

## 6. Overall Consistency

- [ ] Scale and perspective are coherent across the image
- [ ] No impossible geometry or warped anatomy that contradicts physics
- [ ] Texture detail remains consistent with distance and lighting

---

## Scoring Notes

- Always begin with Section 0 (Spatial Measurement).
- Each failed item must include a short physical explanation.
- Prefer clear optical and geometric evidence over statistical guesses.
- When uncertain, mark for manual review instead of forcing a decision.
