# Model Card — NOVA-VAD-frame-v2

## What changed from v1

Targeted goal: close the MCC/accuracy gap with Silero/Pyannote via noise
robustness and precision, without touching clean-audio accuracy or the
benchmark's measurement code. Full methodology, integrity checks, and
honest caveats: `reports/decision_v3.md` — **read that before trusting
these numbers**, especially the "Explicit flags" section.

- **62 features** (v1's 58 + periodicity_strength, estimated_f0_hz,
  spectral_flatness mean/std) — structural/harmonic features chosen from
  acoustic theory, validated against synthetic tones/noise before ever
  touching project audio.
- **4x training volume**: 400 scenes (100 original + 300 new "train2"),
  48,000 windows, vs. v1's 100 scenes / 12,000 windows.
- **Re-tuned hysteresis** on a fresh validation split ('val', 40 scenes,
  never used for v1's tuning), selecting T_on=0.55/T_off=0.45 (up from
  v1's 0.45/0.35) — a real precision/recall trade-off, not a free
  improvement.
- Median-filter post-processing was tested (kernel sizes 1/3/5) and did
  **not** help on validation — reported as a negative result, not dropped
  silently.

## Result on the locked test scenes (single evaluation, see decision_v3.md for the integrity trail)

| Metric | v1 | v2 | Change |
|---|---|---|---|
| Accuracy | 69.88% | 78.99% | +9.1pp |
| Precision | 42.77% | 57.10% | +14.3pp |
| Recall | 68.86% | 56.24% | -12.6pp |
| F1 | 52.77% | 56.67% | +3.9pp |
| MCC | 0.344 | 0.428 | +0.084 |

Now close to SpeechBrain (MCC 0.44); still behind Pyannote (0.53) and
Silero (0.57).

## Known issues, disclosed not hidden

- Clean-audio accuracy moved -1.1pp (90.6% → 89.5%) against an explicit
  "don't touch clean" goal — plausibly within scene-to-scene noise (only
  10 clean test scenes) but not statistically proven so.
- The new noise-robust features ranked 29th/37th/38th/52nd of 62 in RF
  feature importance — real contributors, but not the dominant ones; most
  of the gain is more likely attributable to training volume + retuned
  thresholds.
- The biggest single-condition jump (+23.4pp at 0dB SNR) is most plausibly
  a noise-file-identity effect (see decision_v3.md's diagnostic), not
  evidence of a stable SNR-band-specific improvement — the benchmark has
  only 8 unique noise files per condition, which isn't enough to fully
  separate "better at 0dB" from "better at handling one specific hard
  noise file that happened to land in the 0dB bucket."
- Recall dropped meaningfully; not a drop-in replacement for v1 in any use
  case where missing speech is costlier than a false alarm.
