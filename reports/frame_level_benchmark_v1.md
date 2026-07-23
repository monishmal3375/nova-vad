# NOVA-VAD Frame-Level Benchmark v1

Generated from 40 locked test scenes (`data/scenes/test/`), built from files never used to train NOVA-VAD, scored at 10ms frame resolution using each system's native output (see `scripts/frame_vad_adapters.py`).

Scope: clean + 3 noise-level (SNR) conditions on speech mixed into the same file as noise/music. Codec/RTC transmission conditions and non-music hard negatives are **not yet covered** — see plan Section 7.2 Layers 3-4.

## Overall frame-level results (95% CI via cluster bootstrap over scenes)

| Model | Accuracy | 95% CI | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| NOVA-VAD | 33.26% | [27.88, 39.13] | 14.56% | 35.57% | 20.66% | -0.2795 |
| NOVA-VAD-frame-v1 | 69.88% | [61.88, 77.21] | 42.77% | 68.86% | 52.77% | 0.3437 |
| WebRTC VAD | 53.76% | [45.48, 62.67] | 31.48% | 75.89% | 44.5% | 0.1965 |
| Silero VAD | 85.24% | [83.51, 86.88] | 88.39% | 45.57% | 60.13% | 0.565 |
| Pyannote VAD | 83.81% | [79.59, 87.0] | 72.64% | 54.13% | 62.03% | 0.5293 |
| SpeechBrain VAD | 73.53% | [69.56, 76.99] | 47.48% | 78.65% | 59.22% | 0.4424 |

## Per-condition breakdown (accuracy %)

| Model | clean | snr_10db | snr_0db | snr_-5db |
|---|---|---|---|---|
| NOVA-VAD | 26.22% | 33.33% | 36.18% | 37.32% |
| NOVA-VAD-frame-v1 | 90.61% | 69.93% | 55.37% | 63.62% |
| WebRTC VAD | 90.17% | 38.62% | 41.64% | 44.62% |
| Silero VAD | 88.09% | 87.02% | 84.38% | 81.47% |
| Pyannote VAD | 88.39% | 81.84% | 80.64% | 84.38% |
| SpeechBrain VAD | 77.05% | 78.61% | 68.82% | 69.66% |
