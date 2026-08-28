# Flagship Core — Grok version

This is not the Perplexity runnable package and not Project-A1.

## Goal

Same as the others: a physics-first authenticity instrument that can sit behind a
camera or a drop zone. Different architecture.

## Lanes (mandatory order)

0. **Capture** — is the file in hand a first-generation optical capture, or a photo of a screen?
1. **Foundation geometry** — room, key light, falloff. Catchlights wait.
2. **Sensor** — residual in / fingerprint optional / PCE later.
3. Existing Gravity Check modules (`catchlights`, `glasses`, `shadows`) plug in after 1.

## Language

`consistent` | `conflict` | `insufficient` | `second_generation`

No `real`. No `fake`. No `scam`. Headline is the strongest physical lane, not a blend.

## Residuals

- Default: Gaussian high-pass (runs in this sandbox).
- Wavelet / Wiener: already in `src/gravity_check/prnu_sensor_noise.py`.
- DRUNet: explicit hole (`DrunetResidual`) until KAIR weights land on the GPU box.

## What victory is

Reports that read. Independent test split, not the 700-image train score.
README rewrite waits for that.
