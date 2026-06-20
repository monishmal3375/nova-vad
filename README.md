# NOVA-VAD 🎙️

> **Noise-robust, Optimized, eXplainable Voice Activity Detector**

NOVA-VAD is a lightweight, explainable Voice Activity Detector that outperforms every major open-source alternative on real-world noisy audio — without requiring a GPU or PyTorch.

Built as an open-source contribution to solving a problem that has existed in speech processing for 15+ years: existing VADs are either accurate OR lightweight OR explainable. Never all three.

---

## 🏆 Benchmark Results

Tested on 100 held-out files from UrbanSound8K (traffic, sirens, jackhammers, AC units, construction noise):

| Model | Accuracy | Precision | Recall | F1 | Lightweight | Explainable |
|---|---|---|---|---|---|---|
| WebRTC VAD | 58.0% | 57.69% | 60.0% | 58.82% | ✅ | ❌ |
| Pyannote VAD | 62.0% | 57.32% | 94.0% | 71.21% | ❌ | ❌ |
| Silero VAD | 87.0% | 86.27% | 88.0% | 87.13% | ❌ | ❌ |
| **NOVA-VAD** | **93.0%** | **97.78%** | **88.0%** | **92.63%** | **✅** | **✅** |

---

## 🔑 What Makes NOVA-VAD Different

| Feature | WebRTC | Silero | Pyannote | NOVA-VAD |
|---|---|---|---|---|
| Accurate on noisy audio | ❌ | Partial | Partial | ✅ |
| Lightweight (no PyTorch) | ✅ | ❌ | ❌ | ✅ |
| Fully open source | ✅ | Partial | ✅ | ✅ |
| Explains every decision | ❌ | ❌ | ❌ | ✅ |
| Retrainable on custom data | ❌ | ❌ | ❌ | ✅ |
| Confidence scores | ❌ | ❌ | ❌ | ✅ |

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

### Explain a Prediction
```bash
python3 -m src.explainer data/clean_speech/speech_001.wav
```

### Run Full Benchmark
```bash
python3 -m src.benchmark
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

NOVA-VAD solves all three simultaneously. No existing open-source tool does this.

---

## 🛣️ Roadmap

- [x] Denoiser pipeline
- [x] WebRTC VAD baseline
- [x] 150+ feature MFCC classifier
- [x] Ensemble model (RF + GBT)
- [x] Explainability layer
- [x] Benchmark vs Silero, Pyannote, WebRTC
- [ ] Real-time streaming audio support
- [ ] pip install nova-vad packaging
- [ ] Research paper

---

## 👤 Author

**Monish**

---

## 📄 License

MIT License — free to use, modify, and distribute.
