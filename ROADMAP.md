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
