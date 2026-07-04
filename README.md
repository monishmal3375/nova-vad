# NOVA-VAD 🎙️

> **Noise-robust, Optimized, eXplainable Voice Activity Detector**

NOVA-VAD is a lightweight, explainable Voice Activity Detector for noisy real-world audio.

It is built for people working on ASR, diarization, call transcription, edge audio, robotics, and realtime voice agents who need to decide when speech is actually present before sending audio downstream.

On the current held-out noisy-audio benchmark, NOVA-VAD reports **94.0% accuracy / 94.0 F1** while staying lightweight and explainable.

**Links**

- Hugging Face: https://huggingface.co/monishmal0204/nova-vad
- X: https://x.com/Nova_vad
- Roadmap: [ROADMAP.md](ROADMAP.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 🏆 Benchmark Results

Tested on 100 held-out files sampled from all 10 UrbanSound8K noise categories — air
conditioner, car horn, children playing, dog bark, drilling, engine idling, gun shot,
jackhammer, siren, and street music — stratified so no single category dominates the
test set (previously this benchmark only covered 5 categories: traffic, sirens,
jackhammers, AC units, and construction noise).

The benchmark is intentionally scoped: these numbers describe this repo's noisy-audio test setup, not a universal claim across every speech domain.

| Model | Accuracy | Precision | Recall | F1 | Mean Latency | Model Size | Lightweight | Explainable |
|---|---|---|---|---|---|---|---|---|
| WebRTC VAD | 52.0% | 51.67% | 62.0% | 56.36% | 0.85ms | N/A | ✅ | ❌ |
| Energy Threshold (naive) | 74.0% | 70.69% | 82.0% | 75.93% | 0.29ms | 0B | ✅ | ⚠️ trivial |
| TEN-VAD | 56.0% | 54.84% | 68.0% | 60.71% | 22.94ms | N/A | ✅ | ❌ |
| SpeechBrain VAD | 58.0% | 55.26% | 84.0% | 66.67% | 60.41ms | N/A | ❌ | ❌ |
| Pyannote VAD | 67.0% | 60.76% | 96.0% | 74.42% | 58.87ms | N/A | ❌ | ❌ |
| Silero VAD | 91.0% | 88.68% | 94.0% | 91.26% | 8.52ms | N/A | ❌ | ❌ |
| **NOVA-VAD** | **94.0%** | **94.0%** | **94.0%** | **94.0%** | **62.82ms** | **1.1MB** | **✅** | **✅** |

Picovoice Cobra is wired into the benchmark script but skipped by default — it requires a
commercial AccessKey. Set `PICOVOICE_ACCESS_KEY` (and `pip install pvcobra`) to include it.

Note: the full benchmark environment installs heavier baseline libraries so the repo can
compare against them. The NOVA-VAD classifier itself is a feature-based scikit-learn
ensemble. Every run of `python3 -m src.benchmark` saves this full table, per-category
accuracy, and false positive/negative file lists to `results/` — see
[Run Full Benchmark](#run-full-benchmark) below.

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

Each run saves reproducible artifacts to `results/`:
- `results/benchmark_latest.json` — full metrics (accuracy/precision/recall/F1, mean + p95 latency, model size on disk) for every model compared, plus false positive/negative filenames
- `results/benchmark_<timestamp>.json` — timestamped copy of the same
- `results/false_positives_negatives.txt` — plain-text list of which NOVA-VAD predictions were wrong and what the model predicted vs. the true label

### Realtime Streaming
```bash
python3 -m src.stream
```

For better streaming behavior, first run:

```bash
python3 retrain_streaming.py
```

Streaming picks up a few flags:

```bash
python3 -m src.stream --list-devices     # list available input devices and exit
python3 -m src.stream --device 2         # use a specific input device by index
```

If no `--device` is given and the session is interactive, you'll be prompted to pick a
microphone from the list (press Enter to use the system default).

The displayed SPEECH / NO SPEECH state uses chunk-level hysteresis: a state flip only
takes effect once the new label has majority support over the last few 1s chunks, so
single borderline frames don't cause visible flicker.

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
- [x] Benchmark vs Silero, Pyannote, WebRTC, SpeechBrain, TEN-VAD
- [x] Harden real-time streaming audio support (chunk-level hysteresis + mic device selection)
- [x] Expand noisy-audio benchmark to all 10 UrbanSound8K categories + latency/model-size tracking + FP/FN artifacts
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
