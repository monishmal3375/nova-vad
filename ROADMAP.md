# NOVA-VAD Roadmap

NOVA-VAD is focused on becoming a practical, lightweight, explainable VAD for noisy real-world audio.

## Near Term

- Package install: `pip install nova-vad`
- Simple CLI: `nova-vad predict path/to/audio.wav`
- Lightweight inference path that does not require benchmark-only dependencies
- Clear Hugging Face demo/model card
- Reproducible benchmark script with saved result artifacts

## Streaming

- Improve realtime stream calibration
- [x] Add chunk-level smoothing to reduce flicker between `SPEECH` and `NO SPEECH` —
  `src/stream.py` now debounces the displayed label with a rolling-window
  hysteresis (`StateSmoother`): a flip only takes effect once the new label wins
  majority support (3 of the last 5 one-second chunks) instead of flipping on any
  single borderline frame.
- [x] Add microphone device selection — `python3 -m src.stream --list-devices` lists
  input devices, `--device N` selects one, and an interactive picker prompts when
  neither is given.
- Add examples for voice-agent and ASR pre-processing

## Evaluation

- [x] Expand noisy-audio benchmark sets — `download_noise.py` now stratifies
  UrbanSound8K sampling across all 10 categories (previously coverage was whatever
  fell out of random sampling, in practice skewed toward
  traffic/siren/jackhammer/AC/construction). A `data/noise/_category_manifest.csv`
  tracks which category each sampled file came from, and `src/benchmark.py` reports
  a per-category accuracy breakdown.
- [x] Add more baselines where setup is reproducible — added TEN-VAD (open source,
  no API key, popular in real-time voice-agent stacks) and a dependency-free
  "Energy Threshold" naive baseline. Picovoice Cobra is wired in as an opt-in stub
  (set `PICOVOICE_ACCESS_KEY` once a key is available).
- [x] Publish false-positive and false-negative examples —
  `results/false_positives_negatives.txt` and `results/benchmark_latest.json` list
  exactly which files NOVA-VAD got wrong each run, predicted vs. actual label, and
  confidence.
- [x] Track latency and model size alongside accuracy/F1 — every benchmark run
  reports mean/p95 per-file latency and on-disk model size per model in the same
  table and in `results/benchmark_latest.json`.
- [x] Fix a benchmark methodology bug where the naive Energy-Threshold baseline
  (74.0%) beat WebRTC, TEN-VAD, and SpeechBrain — root cause was that every model
  was evaluated on `data/clean_speech`/`data/clean_noise`, output of
  `src/denoiser.py`'s `noisereduce` step, which builds each clip's noise profile
  from that same clip's own first 0.5s. That erased ~67% of noise-clip RMS energy
  on average but only ~18% of speech-clip RMS energy, manufacturing an artificial
  energy gap between classes that doesn't exist in the raw audio (and doesn't match
  how `src/explainer.py`/`src/stream.py` run inference — neither denoises).
  `src/benchmark.py` now trains and evaluates every model on raw `data/speech`/
  `data/noise`. Energy Threshold now correctly scores 52.0% (coin-flip, as expected
  for a volume-only heuristic on real-world noise) and every real VAD system beats
  it. See the README's "Benchmark methodology fix" note for full numbers.
- [x] Fix a duration confound — speech clips (Speech Commands, ~1s) were
  systematically shorter than noise clips (UrbanSound8K, ~3.5-4s), letting a
  classifier partly learn "which dataset did this come from" instead of real
  speech-vs-noise acoustics. `src/experiment.py` now standardizes every clip to a
  fixed 1-second window before feature extraction, matching `src/stream.py`'s real
  inference-time chunking.
- [x] Expand the dataset 3.3x (1,800 → 5,990 files) using the same two
  already-licensed sources, and check for train/test leakage — UrbanSound8K source
  recordings (`fsID`) and Speech Commands speakers are now tracked
  (`backfill_fsid.py`, `backfill_speaker_id.py`) and grouped so a whole source
  recording or speaker is assigned to train or test as a unit, never split across
  both. See the README's "Dataset integrity" section for what this changed (or
  didn't).
- [x] Cut inference latency 62.8ms → ~25ms — shared spectrograms across features
  (one STFT reused instead of recomputed per-feature), a validated 2x-decimated
  HPSS computation (Pearson r=0.99 vs. the original), and cached mel/chroma
  filterbank construction (verified bit-for-bit identical output, not an
  approximation).

## Explainability

- Improve plain-English explanations for the most important features
- Add JSON output for downstream apps
- Add docs explaining when feature importances can be trusted

## Research Writeup

- Problem statement
- Methodology
- Benchmark setup
- Limitations
- Failure cases
- Future work
