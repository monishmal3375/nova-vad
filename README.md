# NOVA-VAD 🎙️

> **Noise-robust, Optimized, eXplainable Voice Activity Detector**

NOVA-VAD is a lightweight, explainable Voice Activity Detector for noisy real-world audio.

It is built for people working on ASR, diarization, call transcription, edge audio, robotics, and realtime voice agents who need to decide when speech is actually present before sending audio downstream.

On the current held-out noisy-audio benchmark, NOVA-VAD reports **93.0% accuracy / 92.63 F1** while staying lightweight and explainable.

**Links**

- Hugging Face: https://huggingface.co/monishmal0204/nova-vad
- X: https://x.com/Nova_vad
- Roadmap: [ROADMAP.md](ROADMAP.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 🏆 Benchmark Results

Tested on 100 held-out files from UrbanSound8K noise categories including traffic, sirens, jackhammers, AC units, and construction noise.

The benchmark is intentionally scoped: these numbers describe this repo's noisy-audio test setup, not a universal claim across every speech domain.

| Model | Accuracy | Precision | Recall | F1 | Lightweight | Explainable |
|---|---|---|---|---|---|---|
| WebRTC VAD | 58.0% | 57.69% | 60.0% | 58.82% | ✅ | ❌ |
| Pyannote VAD | 62.0% | 57.32% | 94.0% | 71.21% | ❌ | ❌ |
| Silero VAD | 87.0% | 86.27% | 88.0% | 87.13% | ❌ | ❌ |
| **NOVA-VAD** | **93.0%** | **97.78%** | **88.0%** | **92.63%** | **✅** | **✅** |

Note: the full benchmark environment installs heavier baseline libraries so the repo can compare against them. The NOVA-VAD classifier itself is a feature-based scikit-learn ensemble.

---

## 🔑 What Makes NOVA-VAD Different

| Feature | WebRTC | Silero | Pyannote | NOVA-VAD |
|---|---|---|---|---|
| Accurate on noisy audio | ❌ | Partial | Partial | ✅ |
| Lightweight core classifier | ✅ | ❌ | ❌ | ✅ |
| Fully open source | ✅ | Partial | ✅ | ✅ |
| Explains every decision | ❌ | ❌ | ❌ | ✅ |
| Retrainable on custom data | ❌ | ❌ | ❌ | ✅ |
| Confidence scores | ❌ | ❌ | ❌ | ✅ |

---

## Who This Is For

- Voice-agent builders who need cleaner speech boundaries before ASR
- Speech researchers testing VAD behavior in noisy environments
- Edge/audio developers who want a lightweight baseline without a GPU
- Open-source contributors interested in explainable audio ML

If you try NOVA-VAD on your own noisy dataset, please open an issue with the result. Hard failure cases are especially useful.

---

## 🧠 How It Works
Raw Audio → Denoiser → 150+ Features → Ensemble Classifier → SPEECH / NO SPEECH + Explanation
### 150+ Features Extracted Per File
- MFCCs + deltas (78 features) — spectral shape and change over time
- Zero Crossing Rate — speech is more consistent than noise
- RMS Energy pattern — speech rises and falls rhythmically
- Spectral Flux — speech transitions smoothly, noise changes randomly
- Harmonic/Percussive ratio — human voice is mostly harmonic
- Tempo/rhythm — speech has syllable rhythm noise does not
- Mel Spectrogram statistics — energy distribution across frequency bands
- Silence ratio — proportion of frames below energy threshold

### Ensemble Model
Random Forest + Gradient Boosting voting together.

### Explainability
Every prediction includes confidence score + top 10 features that drove the decision in plain English.

---

## 🚀 Quick Start

```bash
git clone https://github.com/monishmal3375/nova-vad.git
cd nova-vad
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 download_data.py
python3 -m src.pipeline
```

The first full run downloads data, denoises audio, trains the ensemble, and saves local model files into `models/`.

### Explain a Prediction
```bash
python3 -m src.explainer data/clean_speech/speech_001.wav
```

### Try Your Own Audio
After running the pipeline once so local models are saved:

```bash
python3 -m src.explainer path/to/your_audio.wav
```

If NOVA-VAD gets your clip wrong, open a noisy-audio issue with the expected label, prediction, confidence, and a short description of the noise.

### Run Full Benchmark
```bash
python3 -m src.benchmark
```

### Realtime Streaming
```bash
python3 -m src.stream
```

For better streaming behavior, first run:

```bash
python3 retrain_streaming.py
```

---

## 📊 Example Output
=======================================================

NOVA-VAD EXPLANATION
File:        speech_001.wav

Prediction:  SPEECH

Confidence:  93.47%
Why this decision was made:

MFCC Delta 1 std      (10.63%) → HIGH spectral change rate — dynamic audio like speech
MFCC Delta 2 std      ( 6.14%) → HIGH acceleration — rapidly changing audio, speech-like
Silence ratio          ( 5.92%) → 56% silence — mix of speech and pauses
Spectral centroid std  ( 4.27%) → HIGH variation — shifting frequency center
Mel mean               ( 3.50%) → MODERATE energy — normal speech level

=======================================================
---

## 📁 Project Structure
nova-vad/

├── data/

│   ├── speech/          # raw speech files

│   ├── noise/           # raw noise files

│   ├── clean_speech/    # denoised speech

│   └── clean_noise/     # denoised noise

├── src/

│   ├── denoiser.py      # noise reduction pipeline

│   ├── vad.py           # WebRTC VAD baseline

│   ├── classifier.py    # NOVA-VAD 150+ features + ensemble

│   ├── explainer.py     # explainability layer

│   ├── benchmark.py     # head-to-head comparison

│   └── pipeline.py      # end-to-end runner

├── models/              # saved trained models

├── download_data.py     # automated dataset downloader

└── requirements.txt
---

## 🔬 Why This Matters

**Existing VADs fail in three ways:**

1. They break in noisy environments — WebRTC gets 58% on real-world noise
2. They are black boxes — no explanation of why a decision was made
3. They are too heavy for edge devices — Silero needs PyTorch (200MB+)

NOVA-VAD is designed to push on all three at once: noisy-audio performance, lightweight inference, and explainable decisions.

---

## 🛣️ Roadmap

- [x] Denoiser pipeline
- [x] WebRTC VAD baseline
- [x] 150+ feature MFCC classifier
- [x] Ensemble model (RF + GBT)
- [x] Explainability layer
- [x] Benchmark vs Silero, Pyannote, WebRTC
- [ ] Harden real-time streaming audio support
- [ ] pip install nova-vad packaging
- [ ] Research paper

See [ROADMAP.md](ROADMAP.md) for contributor-friendly tasks.

---

## 🤝 Contributing

NOVA-VAD is early and useful test coverage matters more than polished hype.

Good ways to help:

- Try it on noisy speech from your own project
- Open an issue with false positives or false negatives
- Add benchmark results against another VAD
- Improve packaging so users can `pip install nova-vad`
- Help harden realtime streaming support

Start with [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 👤 Author

**Monish**

---

## 📄 License

MIT License — free to use, modify, and distribute.
