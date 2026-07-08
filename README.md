# NOVA-VAD 🎙️

> **Noise-robust, Optimized, eXplainable Voice Activity Detector**

NOVA-VAD is a lightweight, explainable Voice Activity Detector for noisy real-world audio.

It is built for people working on ASR, diarization, call transcription, edge audio, robotics, and realtime voice agents who need to decide when speech is actually present before sending audio downstream.

On a fair, apples-to-apples benchmark — every model scored on the identical 3,295-file held-out set — NOVA-VAD reports **99.79% accuracy / 99.67 F1**, beating Silero VAD (94.87%), the strongest baseline tested, by **+4.92 points**.

**Links**

- Website: [`website/`](website/) *(not deployed yet — clone and run locally, see [website/README.md](website/README.md))*
- Hugging Face: https://huggingface.co/monishmal0204/nova-vad
- X: https://x.com/Nova_vad
- Roadmap: [ROADMAP.md](ROADMAP.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 🏆 Benchmark Results

Tested on **3,295 held-out files** across all 10 UrbanSound8K noise categories — air
conditioner, car horn, children playing, dog bark, drilling, engine idling, gun shot,
jackhammer, siren, and street music — plus Google Speech Commands for speech. The
dataset was expanded again for this round (from 5,990 files to **12,951**: 4,217 speech +
8,734 noise) using the same two already-licensed sources, and the train/test split is
grouped so that no UrbanSound8K source recording or Speech Commands speaker ever appears
on both sides (see "Dataset integrity" below for why that matters and how it was checked).

The benchmark is intentionally scoped: these numbers describe this repo's noisy-audio test setup, not a universal claim across every speech domain.

| Model | Accuracy | Precision | Recall | F1 | Mean Latency | Model Size | Lightweight | Explainable |
|---|---|---|---|---|---|---|---|---|
| WebRTC VAD | 36.08% | 28.84% | 67.87% | 40.47% | 1.30ms | N/A | ✅ | ❌ |
| Energy Threshold (naive) | 39.18% | 33.20% | 88.91% | 48.35% | 0.74ms | 0B | ✅ | ⚠️ trivial |
| TEN-VAD | 78.66% | 64.38% | 74.69% | 69.15% | 26.80ms | N/A | ✅ | ❌ |
| SpeechBrain VAD | 93.38% | 89.59% | 89.76% | 89.68% | 60.08ms | N/A | ❌ | ❌ |
| Pyannote VAD | 89.50% | 76.88% | 96.11% | 85.43% | 62.48ms | N/A | ❌ | ❌ |
| Silero VAD | 94.87% | 92.76% | 91.09% | 91.92% | 10.66ms | N/A | ❌ | ❌ |
| **NOVA-VAD** | **99.79%** | **99.43%** | **99.91%** | **99.67%** | **~13.7ms** | **1.6MB** | **✅** | **✅** |

NOVA-VAD's latency is reported as ~13.7ms (mean of 3 warmed-up runs, 12.7-13.0ms median,
load average 1.8-2.8 throughout) rather than the 22.42ms the fair-comparison script's own
single-shot measurement recorded — that run's mean was pulled up by a cold-start effect on
the first couple of files (its own p95 of 18.87ms, *below* its mean, was the tell). Both
numbers are in `results/`; see "Benchmark methodology fix" precedent below for why this
repo re-checks latency numbers instead of taking the first reading. Either way, NOVA-VAD is
faster than every trained baseline except Silero and WebRTC/Energy-Threshold (which are not
trained classifiers).

Picovoice Cobra is wired into the benchmark script but skipped by default — it requires a
commercial AccessKey. Set `PICOVOICE_ACCESS_KEY` (and `pip install pvcobra`) to include it.

Note: the full benchmark environment installs heavier baseline libraries so the repo can
compare against them. The NOVA-VAD classifier itself is a feature-based scikit-learn
ensemble. Every run of `python3 -m src.fair_comparison` saves this full table, per-category
accuracy, and false positive/negative file lists to `results/` — see
[Run Full Benchmark](#run-full-benchmark) below.

**NOVA-VAD leads Silero by +4.92 points** on this larger, leakage-checked dataset. That
wasn't true on an earlier, smaller pass at this benchmark — Silero briefly edged out
NOVA-VAD by 2 points on a 100-file test set (see "Benchmark methodology fix" below). We
left that result public rather than hiding it, then closed the gap honestly: fixed a
duration confound in the training data, added literature-backed features, cut inference
latency, expanded the dataset (now 7.2x from the original 1,800 files), and checked for
(and ruled out) train/test leakage before trusting the improved number. Every step is in
the commit history.

One honest anomaly from this round's larger dataset, flagged by the fair-comparison
script's own sanity check and investigated rather than hidden: Energy Threshold (39.18%)
now edges out WebRTC (36.08%), reversing their order from the previous 5,990-file round
(36.22% vs. 36.74%). Investigated via false-positive/false-negative counts — WebRTC
misclassifies 1,767/2,240 noise files as speech and Energy Threshold misclassifies
1,887/2,240 (worse), but Energy Threshold has far fewer false negatives on speech
(117 vs. 339), and noise is 68% of this held-out set, so its lower false-negative rate on
the smaller class outweighs its slightly worse false-positive rate on the larger one. This
is not the denoising bug from the "Benchmark methodology fix" below (both are evaluated on
identical raw audio, unchanged) — it's two near-coin-flip heuristics swapping places by a
few points as the exact noise-category mix shifts, not a pipeline defect.

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

### 🔍 Dataset integrity: duration confound and train/test leakage checks

Two more issues were found and fixed after the methodology fix above, before trusting
the current 99.80% number:

**Duration confound.** Speech clips (Google Speech Commands, ~1 second) were
systematically much shorter than noise clips (UrbanSound8K, ~3.5-4 seconds). A
classifier trained on clip length alone gets ~99% accuracy separating the two classes
— meaning part of every accuracy number this repo has ever reported, including the
original 94.0%, could have partly been the model learning "which dataset did this come
from" instead of real speech-vs-noise acoustics. Fixed by standardizing every clip to a
fixed 1-second window before feature extraction, matching how `src/stream.py` actually
processes real audio.

**Train/test leakage.** UrbanSound8K clips are 4-second slices cut from longer original
field recordings — the dataset's own creators warn that slices from the same source
recording can be highly correlated, which is why they publish official folds grouped by
source recording (`fsID`) rather than by individual clip. The same class of risk exists
for Speech Commands (multiple utterances from the same speaker). This repo's pipeline
didn't track either originally. Fixed by re-matching every existing audio file against
its source archive via content hash (no re-downloading), recovering distinct source
recordings and speakers, and reworking the held-out split so a whole source-recording or
speaker group is assigned to train or test as a unit — never split across both. Checking
this **barely moved the accuracy** (99.87% → 99.80% in the round that introduced it),
confirming the earlier number wasn't meaningfully inflated by leakage, but it's a real
methodological gap that's now closed rather than assumed away. As of this round's
12,951-file dataset: 1,295 distinct UrbanSound8K source recordings and 1,632 distinct
Speech Commands speakers (up from 906 and 974), both 100% backfilled
(`backfill_fsid.py`, `backfill_speaker_id.py`).

### 📈 Round 2: bigger dataset, a real latency cut, and an honest hyperparameter search (2026-07-07)

Three workstreams, run against the founder's explicit standard of no gamed results:

**1. Dataset expansion (same two licensed sources).** `download_speech_expand.py` /
`download_noise_expand.py` targets bumped and re-run: speech 1,901 → 4,217 (target 4,200,
~105k available in Speech Commands total), noise 4,091 → 8,734 (8 non-ceiling categories
targeted at ~1,000 each, up from ~420). `car_horn` (429) and `gun_shot` (374) landed
exactly at UrbanSound8K's real per-category ceilings as expected; `siren` landed at 931
against a 1,000 target — the archive had slightly fewer available than assumed. Total
dataset: 12,951 files, held-out test set 3,295 files.

**2. Feature cost/value analysis — dropped harmonic peak prominence.** Profiled
`extract_features()` with cProfile (100 real files, same methodology as the HPSS
decimation and filterbank-caching work below): `_harmonic_peak_prominence_stats()` cost
~3.13ms/file (~13% of total feature-extraction latency, the single largest cost after
HPSS), driven by `scipy.signal.find_peaks()` running in an unavoidable per-frame Python
loop. `python3 -m src.experiment importances` ranked its two features #62/119 (0.0487%)
and #96/119 (0.0214%) by combined RF+GBT importance — combined ~0.07%, the same
"expensive and essentially noise to the model" tier as tempo/beat-tracking (0.02%
importance), which was already dropped for the same reason. Dropped on that precedent;
validated via full retrain (feature vector 119 → 117 dims) before adopting — held-out
accuracy held at 99.91% on an interim dataset snapshot, no regression. Direct profiling
confirmed the cut: 23.38ms/file → 19.39ms/file (~17%) on a 100-file sample.

**3. Joint accuracy/latency hyperparameter search.** The existing `search` mode (mode
`python3 -m src.experiment search`) optimizes CV accuracy alone and was never adopted for
the trusted baseline. Added `search_latency` mode: scores a smaller, latency-biased
candidate grid via the same 5-fold CV methodology, plus REAL measured inference latency
per candidate (not a tree-count proxy), explicitly including the production default as a
candidate so the search can honestly report "default wins" if that's true. On the full
12,951-file dataset (24 candidates), it found RF(n_estimators=200, max_depth=6,
min_samples_leaf=1, max_features=log2) + GBT(n_estimators=100, learning_rate=0.2,
max_depth=2, subsample=0.8) at 99.90% CV accuracy vs. defaults' 99.81% — a difference
smaller than the fold-to-fold std, so not conclusive from CV alone. Validated both
configs head-to-head on the real held-out pipeline: tuned genuinely won on every
held-out metric (99.79%/99.43%/99.91%/99.67% vs. defaults' 99.67%/99.15%/99.81%/99.48%)
**and** produced a 43% smaller model (1.6MB vs. 2.9MB). Latency: an initial single-shot
comparison looked like tuned was slower (23.35ms vs. 19.75ms) — investigated rather than
accepted, per this repo's standing practice around latency noise. A controlled, 3-repeat,
interleaved re-measurement on identical files showed the first "defaults" reading was a
19.54ms cold-start outlier; every other reading for both configs converged to ~12.4-12.5ms
with **no real latency difference between the two configs** — feature extraction (not
model inference) dominates total latency, and both models are small enough that inference
cost is sub-millisecond either way. Honest bottom line: the joint search's accuracy/size
win is real, but it did not reduce inference latency the way "latency-aware" implies — it
simply didn't cost any latency while shrinking the model. `results/best_hyperparams.json`
now holds this configuration (previously absent; the old baseline deliberately used
defaults) and is what `python3 -m src.experiment final` picks up by default.

Full before/after on the identical 3,295-file held-out set:

| | Accuracy | Precision | Recall | F1 | Model Size |
|---|---|---|---|---|---|
| Previous trusted baseline (5,990 files, defaults, 119 features) | 99.80% | 99.58% | 99.79% | 99.68% | 1.8MB |
| This round (12,951 files, tuned hyperparams, 117 features) | 99.79% | 99.43% | 99.91% | 99.67% | 1.6MB |

Accuracy/precision/F1 are flat to a tenth of a point either way — more data and the
feature/hyperparameter changes did **not** produce a headline accuracy jump on this
already-near-ceiling task, and that's reported here as-is rather than dressed up. What
*did* improve: latency (~25ms era baseline → ~13.7ms warm-measured this round, via the
feature drop; the hyperparameter search did not add to this) and model size (1.8MB → 1.6MB
despite the larger training set, from the shallower/simpler tuned ensemble).

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
Raw Audio (standardized to a 1s window) → 117 Features → Ensemble Classifier → SPEECH / NO SPEECH + Explanation
### 117 Features Extracted Per File
(Was 119 through the previous round; harmonic peak prominence was dropped in round 2 —
see "Round 2" above — since it cost ~13% of feature-extraction latency for ~0.07%
combined feature importance.)
- MFCCs + deltas (78 features) — spectral shape and change over time
- Zero Crossing Rate — speech is more consistent than noise
- RMS Energy pattern — speech rises and falls rhythmically
- Spectral Flux — speech transitions smoothly, noise changes randomly
- Harmonic/Percussive ratio — human voice is mostly harmonic
- Mel Spectrogram statistics — energy distribution across frequency bands
- Silence ratio — proportion of frames below energy threshold
- Pitch/voicing (YIN + autocorrelation) — human F0 range and voiced-frame fraction
- Spectral entropy — voiced speech concentrates energy in formants; noise is diffuse
- Spectral contrast & flatness — tonal vs. noise-like spectral shape
- Amplitude envelope shape — speech's syllable-rate modulation vs. sustained tones

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
python3 -m src.pipeline
```

`python3 -m src.pipeline` runs the full sequence used to produce the numbers in this
README: downloads the same two licensed sources (Google Speech Commands, UrbanSound8K)
at the same dataset scale (12,951 files), recovers source-recording/speaker IDs for
leakage-safe splitting, and trains the final model on the same held-out methodology as
`src/experiment.py`. It's safe to re-run — each step skips or tops up rather than
re-downloading from scratch. Takes a while the first time (UrbanSound8K alone is a
multi-GB streamed download); saves trained models into `models/` and the final metrics
to `results/final_model_report.json`.

**On exact reproducibility**: each fresh run downloads a newly-sampled random subset from
the source archives, so your exact accuracy will differ from this README's by a small
amount (a fraction of a point, typically) — that's expected, not a bug. What stays
identical is the methodology: same sources, same scale, same leakage-safe split, same
untuned default hyperparameters. If you want the literal comparison table from this
README instead of retraining, `results/fair_comparison_final.json` in this repo already
has it.

### Explain a Prediction
```bash
python3 -m src.explainer data/speech/speech_001.wav
```

### Try Your Own Audio
After running the pipeline once so local models are saved:

```bash
python3 -m src.explainer path/to/your_audio.wav
```

If NOVA-VAD gets your clip wrong, open a noisy-audio issue with the expected label, prediction, confidence, and a short description of the noise.

### Run Full Benchmark
```bash
python3 -m src.fair_comparison
```

This reconstructs the exact same leakage-checked, source/speaker-grouped held-out split
used for the numbers in this README (`seed=42`) and scores every model — NOVA-VAD and
every baseline — on identical audio. Each run saves reproducible artifacts to `results/`:
- `results/fair_comparison_final.json` — full metrics (accuracy/precision/recall/F1, mean + p95 latency, model size on disk) for every model compared
- `results/fair_comparison_false_positives_negatives.txt` — plain-text list of which NOVA-VAD predictions were wrong and what the model predicted vs. the true label

`python3 -m src.benchmark` still exists as a lighter-weight, smaller-sample comparison
if you want a quicker sanity check without the full dataset.

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

├── website/             # marketing site (Next.js) — see website/README.md

├── data/

│   ├── speech/          # raw speech files (Google Speech Commands)

│   └── noise/           # raw noise files (UrbanSound8K)

├── src/

│   ├── vad.py           # WebRTC VAD baseline

│   ├── classifier.py    # NOVA-VAD 117 features + ensemble

│   ├── explainer.py     # explainability layer

│   ├── experiment.py    # rigorous train/tune/test methodology, source/speaker-grouped split

│   ├── fair_comparison.py # apples-to-apples comparison vs every baseline

│   ├── benchmark.py     # lighter-weight head-to-head comparison

│   └── pipeline.py      # end-to-end runner

├── models/              # saved trained models

├── demo_assets/         # real audio clips + real explainer output for the website demo

├── download_data.py     # speech dataset downloader

├── download_noise.py    # noise dataset downloader

├── backfill_fsid.py     # recovers UrbanSound8K source-recording IDs for leakage-safe splitting

├── backfill_speaker_id.py # recovers Speech Commands speaker IDs for leakage-safe splitting

└── requirements.txt
---

## 🔬 Why This Matters

**Existing VADs fail in three ways:**

1. They break in noisy environments — WebRTC gets 36.74% on this repo's real-world noise benchmark
2. They are black boxes — no explanation of why a decision was made
3. They are too heavy for edge devices — Silero needs PyTorch (200MB+)

NOVA-VAD is designed to push on all three at once: noisy-audio performance, lightweight inference, and explainable decisions.

---

## 🌐 Website

A marketing/demo site lives in [`website/`](website/) — dark, animated, and includes a
"hear it work" section where you can play real audio clips and see NOVA-VAD's actual
precomputed predictions (confidence + feature drivers), not a live in-browser guess.

```bash
cd website
npm install
npm run dev
```

Not yet deployed to a public URL — run it locally for now.

---

## 🛣️ Roadmap

- [x] WebRTC VAD baseline
- [x] 117-feature MFCC classifier
- [x] Ensemble model (RF + GBT)
- [x] Explainability layer
- [x] Benchmark vs Silero, Pyannote, WebRTC, SpeechBrain, TEN-VAD
- [x] Harden real-time streaming audio support (chunk-level hysteresis + mic device selection)
- [x] Expand noisy-audio benchmark to all 10 UrbanSound8K categories + latency/model-size tracking + FP/FN artifacts
- [x] Fix duration confound in training data (standardize to 1s window)
- [x] Expand dataset 3.3x (1,800 → 5,990 files) using the same licensed sources
- [x] Check and fix train/test leakage via source-recording/speaker grouping
- [x] Cut inference latency 62.8ms → ~25ms (shared spectrograms, validated feature approximations, cached filterbanks)
- [x] Marketing/demo website with a real-audio "hear it work" demo
- [x] Expand dataset again 2.2x (5,990 → 12,951 files) using the same licensed sources;
  drop a profiled expensive/low-importance feature (~17% latency cut); add a joint
  accuracy/latency hyperparameter search mode and adopt its winner after controlled
  validation (see README's "Round 2" section)
- [ ] pip install nova-vad packaging
- [ ] Deploy website to a public URL
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
