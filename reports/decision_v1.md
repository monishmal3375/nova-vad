# Decision: NOVA-VAD v0 ensemble — keep, reduce, or replace?

**Date:** 2026-07-22
**Evidence:** `reports/frame_level_benchmark_v1.md` / `.json` — 40 locked test
scenes, mixed speech+noise in the same file, scored at 10ms frame resolution
against ground truth, using each of the 5 systems through its native
output (see `scripts/frame_vad_adapters.py`).

## Result summary

| Model | Frame accuracy | F1 | MCC |
|---|---|---|---|
| NOVA-VAD (sliding-window mode) | 33.3% | 20.7% | **-0.28** |
| WebRTC VAD | 53.8% | 44.5% | 0.20 |
| SpeechBrain VAD | 73.5% | 59.2% | 0.44 |
| Pyannote VAD | 83.8% | 62.0% | 0.53 |
| Silero VAD | 85.2% | 60.1% | 0.57 |

An MCC of 0 is what an uninformed classifier produces; NOVA-VAD's -0.28 means
its predictions are *worse than uninformed* on this task — actively
anti-correlated with where speech really is. It is last place by a wide
margin on every metric and in every SNR condition, including the clean
condition (26.2% frame accuracy).

## Interpretation

This does not contradict the original 93% whole-file result — that number is
still real and reproducible (`models/registry/nova-vad-v0.1/predictions.csv`).
It answers the specific question the plan raised in Section 7.7: **does the
150-feature ensemble contain a real, generalizable VAD signal, or does it
mainly work because whole clips from Google Speech Commands and MUSAN are
easy to tell apart at the file level?**

The evidence says the latter. The features (MFCC/tempo/chroma/mel statistics
etc.) were computed as global aggregates over entire 1-second isolated-word
clips vs. entire 4-second noise/music clips — very different distributions.
When the same feature extraction is applied to a 1-second sliding window
inside a continuous 12-second scene that mixes speech, silence, and
background noise, the resulting windows don't resemble either training
distribution well, and the classifier's decisions become close to
uninformative — worse than WebRTC's much simpler, purpose-built
frame-level energy/spectral heuristic.

## Decision

**Reduce, don't extend.** Specifically:

1. **Keep** the current RF+GBT ensemble exactly as-is, exclusively for its
   proven task: whole-file speech-vs-noise classification on files
   resembling the training distribution (frozen at
   `models/registry/nova-vad-v0.1/`). Do not market it as a VAD in the
   conventional (frame-level) sense.
2. **Do not** invest further effort in the sliding-window "coarse mode"
   adaptation tested here — the evidence doesn't support it being a viable
   path to a real-time VAD without architectural changes.
3. **If a true frame-level VAD is wanted**, the plan's architecture ladder
   (Section 7.8) is the right next move: a reduced, causal feature set
   designed for short windows (V1), or a small log-mel MLP/TCN trained
   directly on frame-level labels rather than repurposing whole-file
   features (V2/V3) — trained on scenes like the ones generated here, not
   on isolated whole-file clips.
4. **`src/stream.py`'s real-time mode** should be flagged with the same
   caveat — it uses the same whole-file feature extraction over 1-second
   chunks (`extract_features_from_array`), so this finding likely applies
   to it directly. Worth a targeted follow-up test using the same scene
   methodology before trusting its "personal calibration" accuracy claims.

## What this does not test

Codec/RTC transmission degradation, non-music hard negatives (laughter,
coughing, DTMF, overlapping speech), and languages beyond English are not
covered by this pass (plan Section 7.2, Layers 3-4). The ranking among the
four working systems (WebRTC/SpeechBrain/Pyannote/Silero) could shift under
those conditions and hasn't been separately verified here.

---

## Update 2026-07-23: item 3 above has been done — see `reports/decision_v2.md`

A V1 architecture (causal reduced features, trained on true frame labels,
plus hysteresis) was built per the recommendation above. Result: **MCC went
from -0.28 to +0.34** on the same locked test scenes — real, verified
improvement, now ahead of WebRTC. Still behind Silero/Pyannote/SpeechBrain.
Full writeup in `reports/decision_v2.md`; this file is kept as the historical
record of the v0 finding that motivated the change.
