"""
NOVA-VAD-frame-v3 predict_mask() adapter. Identical feature extraction and
hysteresis machinery to v2 (scripts/frame_vad_v2.py, reused not
reimplemented) — v3 changes only the training data (adds targeted 0dB/-5dB
scenes), not the architecture or post-processing logic.
"""
import json
import os

import joblib

from scripts.frame_vad_v2 import predict_hop_probs_v2, _apply_median_filter
from scripts.frame_vad_v1 import apply_hysteresis, FRAME_MS

DEFAULT_PARAMS = {
    "t_on": 0.55,
    "t_off": 0.45,
    "min_speech_frames": 15,
    "min_gap_frames": 15,
    "median_filter_size": 1,
}
PARAMS_PATH = "models/frame_vad_v3_hysteresis.json"


def load_tuned_params():
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            return json.load(f)
    return dict(DEFAULT_PARAMS)


def predict_mask_frame_v3(wav_path, rf, gbt, scaler, duration_ms, params=None):
    if params is None:
        params = load_tuned_params()
    hop_probs = predict_hop_probs_v2(wav_path, rf, gbt, scaler, duration_ms)
    hop_probs = _apply_median_filter(hop_probs, params.get("median_filter_size", 1))
    mask = apply_hysteresis(
        hop_probs, params["t_on"], params["t_off"],
        params["min_speech_frames"], params["min_gap_frames"],
    )
    n_expected = duration_ms // FRAME_MS
    if len(mask) < n_expected:
        mask = mask + [mask[-1] if mask else 0] * (n_expected - len(mask))
    return mask[:n_expected]
