# Decision v2: NOVA-VAD-frame-v1 — did the fix work?

**Date:** 2026-07-23
**Evidence:** `reports/frame_level_benchmark_v1.md` / `.json`, same 40 locked
test scenes v0 was scored on (byte-identical files, verified by SHA-256
before regenerating additional train scenes — see `scripts/generate_scenes.py`
comments). Same scoring code (`scripts/frame_benchmark.py`) for a fair
before/after comparison.

## What was built

Per `reports/decision_v1.md`'s recommendation (item 3): a frame-level model
trained on true per-frame labels, using a causal reduced feature set, with
tuned hysteresis post-processing.

- **Training data:** 100 new mixed scenes (`data/scenes/train/`), built the
  same way as the test scenes, from the same train-split source files
  NOVA-VAD v0 was originally trained on (leakage-safe — train scenes never
  touch the 50 held-out speech / 50 held-out noise files reserved for test).
- **Features:** 58 causal features per 320ms window (`scripts/frame_features.py`)
  — log energy, ZCR, spectral centroid/bandwidth/rolloff/flux, MFCC
  statistics, small mel representation. No tempo, chroma, or
  harmonic/percussive separation (non-causal, didn't help in v0).
- **Model:** same RF(200)+GBT(100) ensemble family as v0, but trained on
  12,000 real per-frame examples (100ms hops) instead of 500 whole-file
  aggregates.
- **Post-processing:** hysteresis (T_on=0.45, T_off=0.35) plus 150ms min
  speech/gap duration, grid-searched on 20 dev scenes (train-source files,
  never the test scenes).

## Result

| Metric | v0 (broken) | v1 (this) | Silero (best of the 4 baselines) |
|---|---|---|---|
| Frame accuracy | 33.3% | **69.9%** | 85.2% |
| F1 | 20.7% | **52.8%** | 60.1% |
| MCC | -0.28 | **+0.34** | 0.57 |

v1 clears WebRTC (MCC 0.20) but not Silero (0.57), Pyannote (0.53), or
SpeechBrain (0.44). Per-condition breakdown shows *why*: v1 is the best
system of all six on clean audio (90.6% accuracy, edging out even Silero's
88.1%), but degrades faster than the neural baselines as noise increases —
55–64% accuracy at 0dB/-5dB SNR vs. their 81–88%. The causal 58-feature set
appears to carry a real, generalizable speech-detection signal (unlike v0),
but is less noise-robust than the pretrained neural systems, which is a
plausible, unsurprising gap for a small classical-feature ensemble vs.
models pretrained on much larger and more varied corpora.

## Decision

**Keep building on v1, don't stop here, and don't claim parity with Silero/Pyannote.**

1. This confirms the plan's Section 7.8 architecture ladder was the right
   call — the fix targeted the actual root cause (whole-file features asked
   to do a frame-level job) and produced a large, verified improvement, not
   a marginal one.
2. Registered at `models/registry/nova-vad-frame-v1/` with checksums, same
   discipline as v0.
3. **Honest public claim from here forward:** "NOVA-VAD-frame-v1 detects
   speech at 10ms frame resolution with 69.9% accuracy / 0.34 MCC on a
   locked benchmark across clean and noisy (down to -5dB SNR) conditions,
   ahead of WebRTC but behind Silero/Pyannote/SpeechBrain, and best of all
   five compared systems on clean audio specifically." Nothing stronger than
   that is supported by the evidence.
4. **Next iteration candidates**, if noise robustness is the priority:
   noise-augmented training data (train scenes currently only cover 4 fixed
   SNR levels — a wider/randomized SNR distribution during training would
   likely help), or the plan's V2/V3 step (small log-mel neural model)
   which is the architecture the strongest baselines (Silero, Pyannote) also
   use.
5. `src/stream.py` still needs the same fix applied — it's unchanged and
   still uses v0's whole-file feature path. That's the natural next target
   now that v1's approach is validated on offline scenes.

## What this still does not test

Same gaps as `reports/decision_v1.md`: no codec/RTC degradation, no hard
negatives beyond music, no on-device latency/memory measurement, no
non-English audio.
