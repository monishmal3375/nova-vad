"""
Extracts per-10ms-frame speech probability/decision arrays for NOVA-VAD-v2,
Silero, and Pyannote, for the ensembling test (round 3, Part 1).

- NOVA-VAD-v2: real probabilities (average of RF+GBT predict_proba), already
  computed at 100ms hop resolution — expanded to the 10ms grid.
- Silero: real probabilities, extracted from the underlying scripted model's
  raw forward() output (512-sample/32ms native chunks at 16kHz — bypasses
  get_speech_timestamps()'s internal thresholding, which only returns
  binary segments). Expanded to the 10ms grid.
- Pyannote: binary mask only (via the existing, already-tested
  predict_mask_pyannote adapter). The segmentation-3.0 model's raw output
  is multi-speaker "powerset" logits (7 classes for up to 3 speakers),
  not a direct speech probability — converting that correctly requires
  knowing the exact powerset class mapping, which risks a subtle
  conversion bug if reimplemented from scratch under this round's time
  budget. Using the binary mask is a deliberate, disclosed scope
  decision, not an oversight.

Run: python3 -m scripts.ensemble_frame_probs <scene_dir> <output_json>
"""
import glob
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import joblib
import librosa
import numpy as np
import torch

from scripts.frame_vad_v2 import predict_hop_probs_v2, HOP_S as V2_HOP_S
from scripts.frame_vad_adapters import predict_mask_pyannote

SR = 16000
FRAME_MS = 10
SILERO_CHUNK = 512  # required exact chunk size at 16kHz


def load_scenes(scene_dir):
    scenes = []
    for json_path in sorted(glob.glob(os.path.join(scene_dir, "*.json"))):
        with open(json_path) as f:
            meta = json.load(f)
        wav_path = json_path.replace(".json", ".wav")
        scenes.append((wav_path, meta))
    return scenes


def expand_to_10ms(values, source_hop_ms, n_frames_10ms):
    """Each value in `values` covers source_hop_ms of audio; repeat it to
    fill the corresponding number of 10ms frames."""
    frames_per_value = max(1, round(source_hop_ms / FRAME_MS))
    expanded = []
    for v in values:
        expanded.extend([v] * frames_per_value)
    if len(expanded) < n_frames_10ms:
        expanded.extend([expanded[-1] if expanded else 0.0] * (n_frames_10ms - len(expanded)))
    return expanded[:n_frames_10ms]


def get_silero_frame_probs(wav_path, model, duration_ms):
    audio, sr = librosa.load(wav_path, sr=SR, mono=True)
    n_frames_10ms = duration_ms // FRAME_MS

    probs = []
    cursor = 0
    while cursor + SILERO_CHUNK <= len(audio):
        chunk = torch.from_numpy(audio[cursor:cursor + SILERO_CHUNK].astype(np.float32))
        with torch.no_grad():
            p = model(chunk, SR).item()
        probs.append(p)
        cursor += SILERO_CHUNK

    chunk_ms = SILERO_CHUNK / SR * 1000  # 32ms
    return expand_to_10ms(probs, chunk_ms, n_frames_10ms)


def get_nova_v2_frame_probs(wav_path, rf, gbt, scaler, duration_ms):
    n_frames_10ms = duration_ms // FRAME_MS
    hop_probs = predict_hop_probs_v2(wav_path, rf, gbt, scaler, duration_ms)
    return expand_to_10ms(hop_probs, V2_HOP_S * 1000, n_frames_10ms)


def build(scene_dir, output_path):
    scenes = load_scenes(scene_dir)
    print(f"Loaded {len(scenes)} scenes from {scene_dir}")

    rf2 = joblib.load("models/registry/nova-vad-frame-v2/frame_vad_v2_rf.pkl")
    gbt2 = joblib.load("models/registry/nova-vad-frame-v2/frame_vad_v2_gbt.pkl")
    scaler2 = joblib.load("models/registry/nova-vad-frame-v2/frame_vad_v2_scaler.pkl")

    from silero_vad import load_silero_vad
    silero_model = load_silero_vad()

    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection
    token = os.environ.get("HF_TOKEN")
    pmodel = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=token)
    pyannote_pipeline = VoiceActivityDetection(segmentation=pmodel)
    pyannote_pipeline.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})

    results = []
    for i, (wav_path, meta) in enumerate(scenes):
        nova_probs = get_nova_v2_frame_probs(wav_path, rf2, gbt2, scaler2, meta["duration_ms"])
        silero_probs = get_silero_frame_probs(wav_path, silero_model, meta["duration_ms"])
        pyannote_mask = predict_mask_pyannote(wav_path, pyannote_pipeline, meta["duration_ms"])
        truth = meta["frame_labels_10ms"]

        n = min(len(nova_probs), len(silero_probs), len(pyannote_mask), len(truth))
        results.append({
            "scene_id": meta["scene_id"],
            "condition": meta["condition"],
            "nova_v2_prob": nova_probs[:n],
            "silero_prob": silero_probs[:n],
            "pyannote_mask": [int(x) for x in pyannote_mask[:n]],
            "truth": truth[:n],
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(scenes)} scenes processed")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    scene_dir = sys.argv[1] if len(sys.argv) > 1 else "data/scenes/val"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "reports/ensemble_frame_probs_val.json"
    build(scene_dir, output_path)
