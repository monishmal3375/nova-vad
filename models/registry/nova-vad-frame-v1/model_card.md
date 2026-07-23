# Model Card — NOVA-VAD-frame-v1

## What changed from v0

v0 (`models/registry/nova-vad-v0.1/`) computed 106 features as global
aggregates over whole files and scored **-0.28 MCC** (worse than random) at
real frame-level speech detection — see `reports/decision_v1.md`. This
version fixes the root cause:

- **Trained directly on frame-level labels**, not whole-file labels repurposed
  for a task they were never designed for. 12,000 causal windows sampled
  from 100 mixed speech+noise scenes (`data/scenes/train/`), each labeled by
  the true (majority) ground truth over its 100ms hop.
- **Causal, reduced 58-feature set** (`scripts/frame_features.py`): log
  energy, ZCR, spectral centroid/bandwidth/rolloff, spectral flux, MFCC
  statistics, a small mel representation. Drops tempo/beat-tracking, chroma,
  and harmonic/percussive separation from v0's set — those need a full clip
  or forward context and didn't transfer to frame-level use.
- **Hysteresis post-processing** (`scripts/frame_vad_v1.py`): T_on=0.45,
  T_off=0.35, 150ms minimum speech/gap duration — tuned by grid search on
  20 DEV scenes only (`scripts/tune_frame_vad_v1.py`), never on the locked
  test scenes.

## Result on the locked test scenes (40 scenes, never used for training or tuning)

| Metric | v0 | v1 | Change |
|---|---|---|---|
| Frame accuracy | 33.3% | **69.9%** | +36.6pp |
| F1 | 20.7% | **52.8%** | +32.1pp |
| MCC | -0.28 | **+0.34** | from worse-than-random to real signal |

Full breakdown, per-condition results, and the other 4 systems for context:
`reports/frame_level_benchmark_v1.md`.

## Honest framing — this is progress, not parity

v1 now beats WebRTC (MCC 0.20) on this benchmark and is genuinely usable as
a real-time-style VAD, unlike v0. It does **not** beat Silero (0.57),
Pyannote (0.53), or SpeechBrain (0.44) — those remain more accurate,
especially as noise increases (v1 drops from 90.6% accuracy on clean audio
to 55–64% at 0dB/-5dB SNR, while Silero and Pyannote stay in the 81–88%
range across all conditions). v1's main strength is the clean condition,
where it's the best of all six systems tested (90.6%).

## What this does not yet include

- No codec/RTC degradation testing (plan Section 7.2 Layer 3).
- No hard negatives beyond music (laughter, coughing, DTMF — Layer 4).
- No streaming-latency measurement (real-time factor, memory) — this was
  scored by running the same causal windowing offline over full scenes, not
  yet wired into `src/stream.py`'s live microphone path.
- Threshold tuning used a single global (T_on, T_off) pair; per-condition or
  adaptive thresholds (like `src/stream.py`'s existing room-calibration idea)
  were not explored here.
