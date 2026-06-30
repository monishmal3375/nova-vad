---
license: mit
tags:
  - audio
  - voice-activity-detection
  - vad
  - speech
  - explainable-ai
  - noisy-audio
datasets:
  - danavery/urbansound8K
language:
  - en
---

# NOVA-VAD

NOVA-VAD is a lightweight, explainable Voice Activity Detector for noisy real-world audio.

It predicts `SPEECH` or `NO SPEECH` and returns a confidence score plus the top audio features that influenced the decision.

GitHub: https://github.com/monishmal3375/nova-vad  
X: https://x.com/Nova_vad

## Why It Exists

Voice activity detection is the first quality gate for ASR, diarization, call transcription, robotics, edge audio, and realtime voice agents.

Bad VAD can cause:

- clipped speech
- false speech segments
- wasted compute
- noisy transcripts
- worse realtime agent behavior

NOVA-VAD is built around a practical wedge: noisy-audio performance, lightweight deployment, and explainable decisions.

## Current Benchmark

Tested on 100 held-out noisy-audio files from UrbanSound8K categories such as traffic, sirens, jackhammers, AC units, and construction noise.

These results describe this benchmark setup only.

| Model | Accuracy | Precision | Recall | F1 | Lightweight | Explainable |
|---|---:|---:|---:|---:|---|---|
| WebRTC VAD | 58.0% | 57.69% | 60.0% | 58.82% | yes | no |
| Pyannote VAD | 62.0% | 57.32% | 94.0% | 71.21% | no | no |
| Silero VAD | 87.0% | 86.27% | 88.0% | 87.13% | no | no |
| NOVA-VAD | 93.0% | 97.78% | 88.0% | 92.63% | yes | yes |

## How It Works

```text
raw audio -> denoiser -> 150+ audio features -> ensemble classifier -> prediction + explanation
```

Feature families include:

- MFCCs and deltas
- zero crossing rate
- RMS energy patterns
- spectral flux
- spectral centroid and rolloff
- harmonic/percussive ratio
- tempo/rhythm
- mel spectrogram statistics
- silence ratio

The current model uses a Random Forest + Gradient Boosting ensemble.

## Quick Start

```bash
git clone https://github.com/monishmal3375/nova-vad.git
cd nova-vad
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 download_data.py
python3 -m src.pipeline
```

Explain one prediction:

```bash
python3 -m src.explainer data/clean_speech/speech_001.wav
```

Run benchmark:

```bash
python3 -m src.benchmark
```

## Example Output

```text
NOVA-VAD EXPLANATION
File:        speech_001.wav
Prediction:  SPEECH
Confidence:  93.47%

Why this decision was made:
MFCC Delta 1 std      (10.63%) -> HIGH spectral change rate, dynamic audio like speech
MFCC Delta 2 std      ( 6.14%) -> HIGH acceleration, rapidly changing audio
Silence ratio          ( 5.92%) -> mix of speech and pauses
Spectral centroid std  ( 4.27%) -> shifting frequency center
Mel mean               ( 3.50%) -> normal speech level
```

## Limitations

- The benchmark is scoped to the current noisy-audio test setup.
- More datasets and real production audio are needed.
- Streaming support exists but is still being improved.
- The project is early and packaging is not finished yet.
- Do not use private call recordings or sensitive speech data without permission.

## Contribute

The most useful contributions are hard noisy-audio cases, benchmark results, packaging help, and streaming improvements.

Open an issue or PR on GitHub:

https://github.com/monishmal3375/nova-vad
