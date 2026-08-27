# Flagship 2 landing (27 Aug 2026)

Scratch from Drive `Flagship 2` landed in Dakota as **lane 2**.

Ignored: `file (3).py`

## What landed
- `src/gravity_check/prnu_sensor_noise.py` — Fridrich/Goljan-style residual + PCE
- `src/gravity_check/prnu_integration/` — signal contract + side fusion
- `src/gravity_check/run_dual_lane.py` — GC first, PRNU second
- `docs/flagship2/calibration_report.md` — honest weak result on 256px GAN faces
- production upgrade plan stays in Drive Flagship 2 (DRUNet/Noiseprint/ESIDE not built)

## Laws kept
- Gravity Check 1→2→3 is primary
- No real/fake stamp, no scammer stamp
- Smooth skin is not a fail
- README.md in repo root was **not** rewritten
- Missing PRNU lowers confidence; it does not raise evidence

## Calibration (do not forget)
Best single-feature accuracy **62.5%**, Cohen's d **0.28**. Raw stats are weak. PCE needs a camera fingerprint.
