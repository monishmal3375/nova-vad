# NOVA-VAD 🎙️

> **Noise-robust, Optimized, eXplainable Voice Activity Detector — v0.1 baseline**

NOVA-VAD is an experimental, lightweight, explainable **whole-file speech-vs-noise
classifier**. On one seeded benchmark it matches or exceeds several established
open-source VADs. It is **not yet validated as a general-purpose VAD** — see
[Known Limitations](#-known-limitations-read-this-first) before relying on the
numbers below for anything beyond this project's own test set.

This README was rewritten on 2026-07-22 to correct claims that didn't match the
code. The previous version claimed testing on UrbanSound8K and "150+ features";
neither is accurate (see corrections below). The frozen model, its exact
checksums, per-file predictions, and full feature list live in
[`models/registry/nova-vad-v0.1/`](models/registry/nova-vad-v0.1/).

---

## 🟡 Frame-level benchmark result (update 2026-07-23)

A real frame-level benchmark now exists (`scripts/generate_scenes.py` +
`scripts/frame_benchmark.py`, 40 locked test scenes with millisecond-accurate
ground truth). At the actual VAD task — detecting *where* speech occurs
inside continuous audio, not just whether a whole file is speech — the
original whole-file-trained ensemble ("NOVA-VAD v0") scored **worst of 5
systems, with a negative MCC (-0.28)**, worse than an uninformed guess. That
led to a second model, **NOVA-VAD-frame-v1**, trained directly on true
per-frame labels with a causal feature set — it recovers to 69.9% frame
accuracy, MCC +0.34, ahead of WebRTC but still behind Silero, Pyannote, and
SpeechBrain. A third pass, **NOVA-VAD-frame-v2**, added noise-robust
periodicity/harmonicity features and 4x the noisy training data, tuned
strictly on a held-out validation split (never the locked test set) — it
reaches **78.99% accuracy, MCC +0.43**, closing most of the gap to
SpeechBrain (0.44), though a real precision/recall trade-off and a small
(-1.1pp) clean-audio shift came with it, flagged as open questions.

**Round 2 (same date) found the benchmark itself was part of the problem:**
a leave-one-noise-file-out analysis showed a single noise file could swing
a condition's accuracy by up to **11.9pp** with only 8 unique noise files
per condition — affecting all 7 systems compared, not just NOVA-VAD. Fixed
by expanding to 25 unique noise files/condition (`data/scenes/test_v2/`,
test-only noise pool, zero leakage) — confidence intervals tightened
24-50% for every system. On the now-more-trustworthy benchmark: the
clean-audio "regression" reversed on a larger sample (turned out to be
sampling noise, not a real cost); the precision/recall trade-off got a
full 84-point threshold curve instead of one blind operating point; and a
follow-up attempt to close the noise gap with targeted low-SNR training
data **failed outright** (regressed every condition, -8.7pp at 10dB) and
was not deployed — reported as a negative result, not hidden.
**NOVA-VAD-frame-v2 remains the best model**: 78.88% accuracy, MCC 0.438
on test_v2, still behind Pyannote (0.541) and Silero (0.522).

**Round 3 tested whether NOVA-VAD-v2's low per-scene correlation with
Pyannote/Silero (0.25-0.39, from round 2) was exploitable**: a simple
logistic combination of NOVA-VAD-v2 + Silero + Pyannote frame
probabilities, fit on `val` only, reached **MCC 0.56 on `test_v2`** —
beating Pyannote (0.54) and Silero (0.52) individually. Real result, but
comes with a real cost (running 3 systems instead of 1) that's a
product/licensing decision, not a modeling one — see
[`reports/phase_a_decision.md`](reports/phase_a_decision.md) for the
verdict: **NOVA-VAD-frame-v2 is kept as the standalone model**;
standalone architecture scaling looks exhausted (v3 regressed), but
combination is a real, separately-evaluated open path.
**Round 4 closed part of the codec/hard-negative gap**: simulated G.711
(A-law/mu-law) and Opus codec degradation barely hurt any system tested,
NOVA-VAD-v2 included (MCC -0.025 vs. clean — far smaller than the noise
conditions' impact); a DTMF hard-negative test showed v2 handling it well
(6.67% false-positive rate, better than v1's 17.0%). WebRTC, previously
only tested at one aggressiveness mode, is now tested at all 4 — the
mode used in every earlier comparison table (3) turns out not to be its
best (mode 2 is, by MCC). Phase A's release gate was still not fully
closed after round 4.

**Round 5 pushed to close all three remaining gaps: one closes, two
advance substantially but stay open, reported exactly as such.** (1)
Fair baseline threshold tuning is now done for real — Silero and
SpeechBrain both have tunable thresholds and were tuned on `val`;
Pyannote's onset/offset turned out to **not be tunable at all** for the
powerset model in use, confirmed by the library itself raising an error
when tried, not assumed. Honest finding: Silero's fair-tuned threshold
scored *worse* on `test_v2` than its untuned default (0.5096 vs 0.5218
MCC) — reported plainly, no ranking conclusions changed either way. (2) A
genuine `RTCPeerConnection` transmission path was built in-browser
(real ICE/STUN negotiation confirmed via a real public IP in the
gathered candidates, real Opus encoding, real RTP transport) and scored
against ground truth for 5 scenes — real progress, but explicitly a
pilot scale, same-machine loopback with no adverse network conditions
tested, disclosed as such rather than oversold. Two implementation bugs
were caught and fixed before trusting any recording. (3) Three more
hard-negative categories added (overlapping speech, breathing, hold
music, each with documented source/license) — NOVA-VAD-v2 hit 0% false
positives on both pure hard-negative categories, matching every neural
baseline. Four to five categories (laughter, coughing, crying, singing,
TV) remain untested, explicitly deferred rather than faked with
unreliable synthesis. **Full accounting of what's closed vs. still open:**
[`reports/decision_v7.md`](reports/decision_v7.md) and the round-5 update
in [`reports/phase_a_decision.md`](reports/phase_a_decision.md).

Full results in
[`reports/frame_level_benchmark_v1.md`](reports/frame_level_benchmark_v1.md)
(superseded — see
[`reports/master_comparison_table.md`](reports/master_comparison_table.md)
for the current authoritative table); the decision writeups are
[`reports/decision_v1.md`](reports/decision_v1.md) (why v0 failed),
[`reports/decision_v2.md`](reports/decision_v2.md) (v1's fix),
[`reports/decision_v3.md`](reports/decision_v3.md) (v2's noise-robustness
pass),
[`reports/decision_v4.md`](reports/decision_v4.md) (round 2: benchmark
validity fix, both open flags resolved, v3 negative result),
[`reports/decision_v5.md`](reports/decision_v5.md) (round 3: the
ensembling test),
[`reports/decision_v6.md`](reports/decision_v6.md) (round 4: codec/
hard-negative/WebRTC-mode testing), and
[`reports/decision_v7.md`](reports/decision_v7.md) (round 5: real RTC
transmission, more hard negatives, fair baseline tuning). Evidence
package for v2:
[`reports/evidence_package_index.md`](reports/evidence_package_index.md).
The 93%/92% numbers below are still real, but they measure a much narrower
task (whole-file classification, not frame-level detection — see next
section).

## ⚠️ Known limitations (read this first)

- **This is a whole-file classifier, not a frame-level VAD.** It labels an
  entire audio clip as SPEECH or NO-SPEECH; it does not return timestamps or
  a probability over time. WebRTC, Silero, Pyannote, and SpeechBrain are all
  frame-level systems — the comparison below converts their frame output into
  a file label using a custom "more than 40% of frames are speech" rule,
  which is not how they're meant to be evaluated. See [Fair comparison
  caveat](#fair-comparison-caveat).
- **It has never seen synthetic/AI-generated speech.** It cannot detect voice
  cloning, TTS, or voice conversion — it only separates "speech-like" from
  "noise/music-like" audio.
- **The positive and negative classes come from different source datasets**
  (Google Speech Commands vs. MUSAN), which differ in recording equipment,
  duration, and mastering. The model may be partially learning dataset
  identity rather than pure speech-presence. A grouped, mixed-scene,
  frame-labeled benchmark is in progress to rule this out.
- **Denoising is opt-in, not a default**, as of this version. The denoiser
  uses each clip's first 0.5 seconds as its noise profile — for clips that
  start talking immediately, that can strip real speech energy rather than
  noise. See [Denoising caveat](#denoising-caveat) for a measured example of
  how much this changes the comparison.

---

## 🏆 Benchmark results

Seeded 80/20 split (500 files: 250 Google Speech Commands "speech" + 250
MUSAN noise/music), NOVA-VAD trained on the 400-file training split only,
all five systems scored on the same 100 held-out files never seen during
training. Reproducible via `python3 -m src.benchmark`.

### On raw audio (current default)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| WebRTC VAD | 50.0% | 50.0% | 74.0% | 59.68% |
| **NOVA-VAD** | **92.0%** | 90.38% | **94.0%** | 92.16% |
| Silero VAD | 92.0% | **97.73%** | 86.0% | 91.49% |
| Pyannote VAD | 87.0% | 82.46% | 94.0% | 87.85% |
| **SpeechBrain VAD** | **93.0%** | 93.88% | 92.0% | **92.93%** |

On raw audio, NOVA-VAD does **not** clearly beat the field — it ties Silero
and is slightly behind SpeechBrain.

### On denoised audio (opt-in via `--denoised`, was previously the only path tested)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| WebRTC VAD | 58.0% | 57.69% | 60.0% | 58.82% |
| **NOVA-VAD** | **93.0%** | **97.78%** | 88.0% | **92.63%** |
| Silero VAD | 87.0% | 86.27% | 88.0% | 87.13% |
| Pyannote VAD | 62.0% | 57.32% | 94.0% | 71.21% |
| SpeechBrain VAD | 60.0% | 55.95% | 94.0% | 70.15% |

#### Denoising caveat

These two tables are the same 100 test files, same seed, same NOVA-VAD model
architecture — the only difference is whether the noisereduce preprocessing
step ran first. Denoising specifically helps NOVA-VAD's relative ranking and
specifically hurts Pyannote and SpeechBrain (which lose 25 and 33 accuracy
points respectively without it). That is evidence the original "NOVA-VAD
beats every alternative" claim was partly an artifact of preprocessing
choices, not of NOVA-VAD being categorically better. Full raw-audio run
saved at
[`reports/benchmark_raw_audio_2026-07-22.txt`](reports/benchmark_raw_audio_2026-07-22.txt).

#### Fair comparison caveat

WebRTC, Silero, Pyannote, and SpeechBrain are frame-level VADs being forced
through a whole-file conversion rule they weren't designed for. A benchmark
that scores every system on the same 10ms-resolution ground truth, using
each baseline through its native streaming/segmentation interface, is in
progress — track it in `reports/` once available. Until then, treat this
table as "how these five systems perform at whole-file classification on
this specific dataset," not "which is the better VAD."

---

## 🧠 How it works

```
Raw Audio → [optional Denoiser] → 106 Features → Ensemble Classifier → SPEECH / NO SPEECH + Explanation
```

### 106 features extracted per file

(Previously documented as "150+" — that didn't match the code. Verified
count as of 2026-07-22; full itemized list in
[`models/registry/nova-vad-v0.1/feature_schema.json`](models/registry/nova-vad-v0.1/feature_schema.json).)

- MFCCs + delta + delta² (78 features) — spectral shape and change over time
- Zero Crossing Rate (4) — speech is more consistent than noise
- RMS Energy pattern (5) — speech rises and falls rhythmically
- Spectral Centroid (2), Rolloff (2), Flux (2), Bandwidth (2)
- Chroma (2) — pitch class information
- Mel Spectrogram statistics (4) — energy distribution across frequency bands
- Tempo/rhythm (1) — speech has syllable rhythm noise does not
- Harmonic/Percussive ratio (3) — human voice is mostly harmonic
- Silence ratio (1) — proportion of frames below energy threshold

### Ensemble model

Random Forest (200 trees) + Gradient Boosting (100 estimators), averaging
predicted probabilities.

### Explainability

Every prediction includes a confidence score and the top 10 features ranked
by **global** Random Forest feature importance (`feature_importances_`),
mapped to a plain-English interpretation. Note: global importance describes
which features were useful across the whole trained forest — it is not yet a
per-sample explanation of *this specific file's* prediction. Local
attribution (e.g. TreeSHAP) is the planned next step; `shap` is already a
listed dependency but not yet wired into `src/explainer.py`.

---

## 🚀 Quick start

```bash
git clone https://github.com/monishmal3375/nova-vad.git
cd nova-vad
python3 -m venv venv
source venv/bin/activate

# Full install (everything, same as before this split):
pip install -r requirements.txt

# Or install only what you need:
pip install -r requirements-runtime.txt    # just run NOVA-VAD
pip install -r requirements-benchmark.txt  # + compare against Silero/Pyannote/SpeechBrain
pip install -r requirements-dev.txt        # + pytest for the test suite

python3 download_data.py
python3 -m src.pipeline              # raw audio (default)
python3 -m src.pipeline --denoise    # opt into denoising
```

### Explain a prediction

```bash
python3 -m src.explainer data/speech/speech_001.wav
```

### Run the full benchmark

```bash
python3 -m src.benchmark             # raw audio (default)
python3 -m src.benchmark --denoised  # opt into denoising
```

### Reproduce the frame-level benchmark (NOVA-VAD v0 vs. v1 vs. v2 vs. baselines)

```bash
python3 -m scripts.generate_scenes       # builds train/dev/test mixed scenes (deterministic, seeded)
python3 -m scripts.frame_benchmark       # scores v0 + WebRTC/Silero/Pyannote/SpeechBrain
python3 -m scripts.train_frame_vad       # trains NOVA-VAD-frame-v1 on the train scenes
python3 -m scripts.tune_frame_vad_v1     # tunes hysteresis thresholds on dev scenes only
python3 -m scripts.evaluate_frame_vad_v1 # scores v1 on the locked test scenes, updates reports/

# v2 (noise-robustness pass — see reports/decision_v3.md for the full integrity trail)
python3 -m scripts.generate_val_split    # separate validation split, seed=43, never touches test
python3 -m scripts.generate_train2_split # 300 additional train scenes, seed=44, train-only noise
python3 -m scripts.train_frame_vad_v2    # trains on train+train2, 62 features (58 + noise-robust)
python3 -m scripts.tune_frame_vad_v2     # tunes on val only, never dev or test
python3 -m scripts.evaluate_frame_vad_v2 # single evaluation on the locked test scenes
```

---

## 📊 Example output

```
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
```

---

## 📁 Project structure

```
nova-vad/
├── data/
│   ├── speech/           # raw speech (Google Speech Commands subset)
│   ├── noise/            # raw noise/music (MUSAN subset)
│   ├── clean_speech/     # denoised speech (opt-in, not default)
│   ├── clean_noise/      # denoised noise (opt-in, not default)
│   ├── mic_speech/       # personal mic recordings (gitignored)
│   └── mic_noise/        # personal mic recordings (gitignored)
├── src/
│   ├── denoiser.py       # optional noise reduction preprocessing
│   ├── vad.py            # WebRTC VAD baseline
│   ├── classifier.py     # NOVA-VAD 106-feature extraction + ensemble
│   ├── explainer.py      # explainability layer (global feature importance)
│   ├── benchmark.py      # 5-way comparison vs WebRTC/Silero/Pyannote/SpeechBrain
│   ├── pipeline.py       # end-to-end runner
│   ├── stream.py         # real-time microphone VAD
│   ├── neural_vad.py     # standalone PyTorch MLP experiment (not in main pipeline)
│   └── evaluate.py       # standalone WebRTC-only evaluator
├── models/
│   ├── registry/          # frozen, checksummed model versions (source of truth)
│   └── archive/           # superseded/duplicate artifacts, kept for reference
├── reports/                # benchmark run outputs
├── requirements*.txt       # split by purpose — see Quick Start
├── record_dataset.py       # interactive mic dataset recorder
├── retrain_streaming.py    # retrains stream.py's model on your mic data
└── download_data.py        # automated dataset downloader
```

---

## 🛣️ Roadmap

- [x] Denoiser pipeline (now opt-in, not default)
- [x] WebRTC VAD baseline
- [x] 106-feature MFCC/spectral classifier
- [x] Ensemble model (RF + GBT)
- [x] Global feature-importance explainability layer
- [x] Whole-file benchmark vs Silero, Pyannote, SpeechBrain, WebRTC
- [x] Real-time streaming audio support (mic-calibrated demo)
- [x] Frame-level (10ms resolution) benchmark with mixed speech+noise scenes and grouped splits
- [x] Frame-level model trained on true per-frame labels (NOVA-VAD-frame-v1, MCC -0.28 → +0.34)
- [x] Noise-robustness pass (NOVA-VAD-frame-v2, MCC +0.34 → +0.43; still behind Pyannote/Silero — see reports/decision_v3.md)
- [ ] Benchmark needs more unique noise files per SNR condition (currently 8/condition — not enough to separate SNR effects from noise-file-identity effects, see decision_v3.md's diagnostic)
- [ ] `src/stream.py` real-time path updated to use a frame-level model
- [ ] Per-sample (local) explanation via TreeSHAP
- [ ] Separate anti-spoofing / synthetic-speech detection model
- [ ] pip install nova-vad packaging
- [ ] Research paper

---

## 👤 Author

**Monish**

---

## 📄 License

MIT License — free to use, modify, and distribute.
