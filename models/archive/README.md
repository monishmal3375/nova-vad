# Archived model artifacts

These files are not part of any currently-active code path. They are kept
(not deleted) so nothing is silently lost, per the model-first engineering
plan's "artifact ambiguity" finding. Regenerate any of them from source if
needed — none require external data beyond what's already in `data/`.

| File | What it is | Written by | Status |
|---|---|---|---|
| `vad_classifier.pkl` | Unknown-provenance classifier. No script in the current repo writes this filename. | *(orphan — no current writer)* | Likely output of an earlier, since-rewritten version of `src/classifier.py`. Safe to ignore; not referenced anywhere. |
| `scaler.pkl` | Duplicate of `neural_scaler.pkl` — **verified byte-identical via SHA-256** (`06c3961c...`, see `../CHECKSUMS.txt`). | *(orphan — no current writer)* | Same status as `vad_classifier.pkl`: no current script produces this exact filename either. |
| `neural_scaler.pkl` | `StandardScaler` for the standalone neural-network experiment. | `python3 -m src.neural_vad` | Regeneratable. `src/neural_vad.py` is not called by `pipeline.py`, `benchmark.py`, or `stream.py` — it's an independent MLP (78→32→16→1) experiment, evaluated via its own leave-one-out loop. |
| `vad_neural_net.pt` | PyTorch weights for the same MLP experiment. | `python3 -m src.neural_vad` | Same as above. |

## Actively used models (still in `models/`, not archived)

| File | Used by |
|---|---|
| `nova_vad_rf.pkl`, `nova_vad_gbt.pkl`, `nova_vad_scaler.pkl` | `src/pipeline.py`, `src/benchmark.py`, `src/explainer.py` — the main NOVA-VAD ensemble. Frozen copy also lives in `models/registry/nova-vad-v0.1/`. |
| `stream_rf.pkl`, `stream_gbt.pkl`, `stream_scaler.pkl` | `src/stream.py` — retrained on personal mic recordings via `retrain_streaming.py`. |
| `speechbrain_vad/` | Auto-downloaded pretrained checkpoint, cached by `src/benchmark.py`'s SpeechBrain comparison the first time it runs. Not something anyone trained. |
