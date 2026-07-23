"""
NOVA-VAD-frame-v2 predict_mask() adapter. Same hysteresis state machine as
v1 (scripts/frame_vad_v1.py, reused directly — not reimplemented, so
behavior stays identical for that part), plus an optional median filter
over the raw hop-probability sequence before thresholding, aimed at
priority 3 from the noise-robustness brief: suppressing short spurious
false-positive bursts (a median filter removes isolated single-hop spikes
that don't reflect a sustained trend, which hysteresis's min-duration
merging catches only after the fact, post-threshold).

median_filter_size and the hysteresis thresholds are tuned on the 'val'
split ONLY (scripts/tune_frame_vad_v2.py) — never on data/scenes/test/.
"""
import json
import os

import librosa
import numpy as np
from scipy.signal import medfilt

from scripts.frame_features_v2 import extract_frame_features_v2
from scripts.frame_vad_v1 import apply_hysteresis, HOP_FRAMES, FRAME_MS

SR = 16000
WINDOW_S = 0.32
HOP_S = 0.10

DEFAULT_PARAMS = {
    "t_on": 0.5,
    "t_off": 0.3,
    "min_speech_frames": 15,
    "min_gap_frames": 15,
    "median_filter_size": 1,  # 1 = no filtering (identity)
}
PARAMS_PATH = "models/frame_vad_v2_hysteresis.json"


def load_tuned_params():
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            return json.load(f)
    return dict(DEFAULT_PARAMS)


def predict_hop_probs_v2(wav_path, rf, gbt, scaler, duration_ms):
    audio, sr = librosa.load(wav_path, sr=SR, mono=True)
    window_samples = int(WINDOW_S * sr)
    hop_samples = int(HOP_S * sr)

    probs = []
    cursor = 0
    while cursor + hop_samples <= len(audio):
        chunk_end = cursor + hop_samples
        window_start = max(0, chunk_end - window_samples)
        window = audio[window_start:chunk_end]
        feats = extract_frame_features_v2(window, sr)
        X_scaled = scaler.transform([feats])
        rf_prob = rf.predict_proba(X_scaled)[0][1]
        gbt_prob = gbt.predict_proba(X_scaled)[0][1]
        probs.append((rf_prob + gbt_prob) / 2)
        cursor = chunk_end

    return probs


def _apply_median_filter(hop_probs, kernel_size):
    if kernel_size <= 1:
        return hop_probs
    if kernel_size % 2 == 0:
        kernel_size += 1  # medfilt requires odd kernel size
    arr = np.array(hop_probs, dtype=np.float64)
    if len(arr) < kernel_size:
        return hop_probs
    filtered = medfilt(arr, kernel_size=kernel_size)
    return filtered.tolist()


def predict_mask_frame_v2(wav_path, rf, gbt, scaler, duration_ms, params=None):
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
