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
- Add chunk-level smoothing to reduce flicker between `SPEECH` and `NO SPEECH`
- Add microphone device selection
- Add examples for voice-agent and ASR pre-processing

## Evaluation

- Expand noisy-audio benchmark sets
- Add more baselines where setup is reproducible
- Publish false-positive and false-negative examples
- Track latency and model size alongside accuracy/F1

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
