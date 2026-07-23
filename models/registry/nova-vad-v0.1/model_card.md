# Model Card — NOVA-VAD v0.1

## What this model is

A Random Forest (200 trees) + Gradient Boosting (100 estimators) ensemble that
classifies an entire `.wav` file as SPEECH or NO-SPEECH, using **106** handcrafted
acoustic features (MFCCs + deltas, energy/spectral statistics, chroma, mel
statistics, tempo/beat features, harmonic/percussive ratio, silence ratio)
averaged/aggregated across the whole clip.

**Correction:** the README and the earlier engineering plan both describe this
as "150+ features." Direct measurement of `extract_features()`'s output on
2026-07-22 shows the actual vector length is 106 (78 MFCC/delta/delta² +
4 ZCR + 5 RMS + 2 centroid + 2 rolloff + 2 flux + 2 bandwidth + 2 chroma +
4 mel + 1 tempo + 3 harmonic/percussive + 1 silence ratio = 106). See
`feature_schema.json` for the itemized, index-by-index breakdown.

## What this model is not

- **Not a frame-level VAD.** It does not return a speech probability over time
  or precise onset/offset boundaries. It returns one label per whole file.
- **Not a synthetic-speech / AI-voice detector.** It has never seen a TTS,
  voice-conversion, or neural-codec-generated sample. It cannot distinguish a
  human voice from a cloned one — it only distinguishes "speech-like audio"
  from "noise/music-like audio."
- **Not validated on telephone/RTC audio, codecs, or packet loss.** Training
  and test data are all clean, single-channel WAV files at their original
  sample rate (resampled to 16kHz).

## Defensible claim for this version

> On a seeded 100-file holdout drawn from Google Speech Commands (speech) and
> MUSAN (noise/music), after this project's denoising preprocessing, this
> model achieved 93.0% accuracy and 92.63% F1 at whole-file speech-vs-noise
> classification.

## Known risk: dataset-shortcut learning

The positive class (Google Speech Commands) and negative class (MUSAN) come
from different recording pipelines, equipment, and mastering. The model may
be partially learning "which dataset does this file come from" rather than
"is speech present." The 80/20 split is seeded and reproducible, but it is
**not** grouped by source or speaker, so this risk has not been ruled out.
This is the reason a frame-level, mixed-scene benchmark (same file, both
classes, millisecond ground truth) is being built as the next step — see
`reports/` once available.

## Known artifact-registry issues at freeze time

- `neural_scaler.pkl` and `scaler.pkl` (outside this registry, in `models/`)
  are byte-for-byte identical (verified via SHA-256) — one is a duplicate.
- `vad_classifier.pkl` (outside this registry) is an earlier/legacy artifact
  of unclear provenance relative to `nova_vad_rf.pkl` / `nova_vad_gbt.pkl`.
- These are being moved to `models/archive/` with an explanatory README
  rather than deleted.

## Do not use this model for

- Detecting AI-generated or cloned voices (it has no training signal for this).
- Any claim of "beats WebRTC/Silero/Pyannote/SpeechBrain" without also citing
  that the comparison was whole-file classification with a custom 40%-frame
  rule applied to convert the frame-based baselines into file labels — this
  is not standard VAD evaluation.
