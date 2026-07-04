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
| WebRTC VAD | 44.0% | 46.25% | 74.0% | 56.92% | 0.90ms | N/A | ✅ | ❌ |
| Energy Threshold (naive) | 52.0% | 51.09% | 94.0% | 66.20% | 0.65ms | 0B | ✅ | ⚠️ trivial |
| TEN-VAD | 83.0% | 83.67% | 82.0% | 82.83% | 18.63ms | N/A | ✅ | ❌ |
| SpeechBrain VAD | 89.0% | 91.49% | 86.0% | 88.66% | 58.92ms | N/A | ❌ | ❌ |
| Pyannote VAD | 92.0% | 90.38% | 94.0% | 92.16% | 61.33ms | N/A | ❌ | ❌ |
| Silero VAD | 96.0% | 100.0% | 92.0% | 95.83% | 9.16ms | N/A | ❌ | ❌ |
| **NOVA-VAD** | **94.0%** | **94.0%** | **94.0%** | **94.0%** | **66.12ms** | **895.4KB** | **✅** | **✅** |

Picovoice Cobra is wired into the benchmark script but skipped by default — it requires a
commercial AccessKey. Set `PICOVOICE_ACCESS_KEY` (and `pip install pvcobra`) to include it.

Note: the full benchmark environment installs heavier baseline libraries so the repo can
compare against them. The NOVA-VAD classifier itself is a feature-based scikit-learn
ensemble. Every run of `python3 -m src.benchmark` saves this full table, per-category
accuracy, and false positive/negative file lists to `results/` — see
[Run Full Benchmark](#run-full-benchmark) below.

**Note on Silero:** on this run Silero VAD (96.0%) edges out NOVA-VAD (94.0%) by 2
points. We're leaving that result as-is rather than tuning against it — see
"Benchmark methodology fix" below for why we care more about the test set being honest
than about NOVA-VAD always winning the table.

### ⚠️ Benchmark methodology fix (2026-07-03)

A previous run of this benchmark reported the naive Energy-Threshold baseline
(`run_energy_threshold` in `src/benchmark.py` — a single RMS-energy check, no ML) at
**74.0% accuracy**, beating WebRTC (52.0%), TEN-VAD (56.0%), and SpeechBrain (58.0%).
That result was a bug, not a real finding, and it has been fixed. Root cause:

- Every model in the benchmark was being evaluated on `data/clean_speech` /
  `data/clean_noise` — audio pre-processed by `src/denoiser.py`'s `noisereduce`-based
  denoiser, not on raw audio.
- `denoise_file()` builds each clip's noise profile from **that same clip's own first
  0.5 seconds** (`noise_sample = audio[:sr*0.5]`) and denoises against it. For a
  roughly-stationary noise clip (drilling, siren, AC hum, engine idling — most of
  UrbanSound8K) the first 0.5s is representative of the whole clip, so this profile
  ends up subtracting out most of the clip's own energy. Measured across the noise
  test set, RMS energy dropped by **67% on average** after this step. For speech
  clips, the first 0.5s is usually a quiet lead-in that isn't representative of the
  louder voiced segments that follow, so speech RMS only dropped by **18% on
  average**.
- That asymmetry manufactured an artificial energy gap between the SPEECH and
  NO-SPEECH classes that does not exist in the raw source audio — in the raw
  recordings, noise is actually *louder* than speech on average in this dataset
  (8.85% mean RMS vs. 6.91%). A single RMS threshold only looked strong because the
  denoising step happened to erase most of the noise energy specifically, not because
  volume genuinely separates speech from real-world noise.
- It also didn't match how NOVA-VAD is actually used: `src/explainer.py` and
  `src/stream.py` (the real inference entry points) never run the denoiser — it was
  only ever an offline data-prep step before training/evaluation. Benchmarking against
  denoised audio was measuring every model's performance on a signal condition that
  never occurs at inference time.

**Fix:** `src/benchmark.py` now trains and evaluates every model (NOVA-VAD included)
on raw, undenoised audio (`data/speech`, `data/noise`) instead of
`data/clean_speech`/`data/clean_noise`. On raw audio, the same fixed 0.02 RMS
threshold gets **52.0% accuracy — a coin flip**, which is the honest result for a
volume-only heuristic against real-world environmental noise. All real trained/
heuristic VAD systems (WebRTC, TEN-VAD, SpeechBrain, Pyannote, Silero) now score above
it, as expected. NOVA-VAD's own accuracy is unaffected by this fix (94.0% either way,
since its 150+ features were never as reliant on raw RMS as a single threshold check).
WebRTC's accuracy also changed (52.0% → 44.0%) under the corrected raw-audio setup —
expected, since WebRTC's frame classifier was previously also being fed
artificially-quiet noise and normal-volume speech, an unrealistic signal regime that
doesn't reflect how WebRTC behaves on real audio.

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

1. They break in noisy environments — WebRTC gets 44% on this repo's real-world noise benchmark
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
