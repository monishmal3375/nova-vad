# GitHub Issue Drafts

Use these to seed contributor-friendly issues after the launch docs are pushed.

## Good First Issue: Try NOVA-VAD On One Noisy Clip

Title:

```text
Try NOVA-VAD on one noisy speech clip and report the result
```

Body:

```text
We need real-world noisy audio cases.

Pick one short non-private clip, run:

python3 -m src.explainer path/to/your_audio.wav

Then comment with:
- noise type
- expected label
- NOVA-VAD prediction
- confidence
- whether the explanation made sense

Please do not upload private calls or sensitive recordings.
```

Labels: `good first issue`, `noisy-audio`, `benchmark`

## Good First Issue: Add JSON Output To Explainer

Title:

```text
Add JSON output option to src.explainer
```

Body:

```text
NOVA-VAD already returns a structured explanation internally, but the CLI prints human-readable text.

Add a CLI flag:

python3 -m src.explainer audio.wav --json

Expected behavior:
- default output remains unchanged
- --json prints the result dictionary as JSON
- include file, label, confidence, and top_features

This will make NOVA-VAD easier to use in downstream apps.
```

Labels: `good first issue`, `enhancement`, `cli`

## Feature Issue: Package For pip Install

Title:

```text
Package NOVA-VAD for pip install
```

Body:

```text
Goal:

Make installation simpler:

pip install nova-vad

Expected work:
- add pyproject.toml
- expose a CLI command like nova-vad
- separate core inference dependencies from benchmark-only dependencies
- update README quickstart

This is important for adoption because cloning the repo and installing the full benchmark stack is too much friction for first-time users.
```

Labels: `enhancement`, `packaging`, `help wanted`

## Feature Issue: Streaming Smoothing

Title:

```text
Add smoothing to realtime streaming predictions
```

Body:

```text
The streaming path currently predicts chunk by chunk.

Goal:
- reduce flicker between SPEECH and NO SPEECH
- add a short rolling window or hysteresis
- keep latency low
- document the tradeoff

Useful for voice agents, live transcription, and microphone demos.
```

Labels: `enhancement`, `streaming`, `help wanted`

## Benchmark Issue: Add Latency And Model Size Metrics

Title:

```text
Add latency and model size to benchmark output
```

Body:

```text
The current benchmark reports accuracy, precision, recall, F1, and elapsed time.

Add:
- model artifact size
- average inference time per file
- dependency/runtime notes

This makes the lightweight claim easier to evaluate.
```

Labels: `benchmark`, `help wanted`
