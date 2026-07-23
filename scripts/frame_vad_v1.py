"""
NOVA-VAD-frame-v1 predict_mask() adapter — same interface as the other
five systems in scripts/frame_vad_adapters.py, so it drops straight into
scripts/frame_benchmark.py for an apples-to-apples comparison.

Adds hysteresis post-processing (plan Section 7.6): a probability must stay
above T_on before entering SPEECH and fall below T_off before leaving it,
plus minimum-duration merging, instead of a raw per-hop threshold. This is
tuned on the DEV scenes only (scripts/tune_frame_vad_v1.py) — never on the
locked test scenes.
"""
import json
import os

import librosa
import numpy as np

from scripts.frame_features import extract_frame_features

SR = 16000
WINDOW_S = 0.32
HOP_S = 0.10
FRAME_MS = 10
HOP_FRAMES = int(HOP_S * 1000 // FRAME_MS)  # 10ms sub-frames per hop

DEFAULT_PARAMS = {
    "t_on": 0.5,
    "t_off": 0.3,
    "min_speech_frames": 15,  # 150ms
    "min_gap_frames": 15,     # 150ms
}
PARAMS_PATH = "models/frame_vad_v1_hysteresis.json"


def load_tuned_params():
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            return json.load(f)
    return dict(DEFAULT_PARAMS)


def predict_hop_probs(wav_path, rf, gbt, scaler, duration_ms):
    """Returns one speech-probability float per 100ms hop, causal (no look-ahead)."""
    audio, sr = librosa.load(wav_path, sr=SR, mono=True)
    window_samples = int(WINDOW_S * sr)
    hop_samples = int(HOP_S * sr)

    probs = []
    cursor = 0
    while cursor + hop_samples <= len(audio):
        chunk_end = cursor + hop_samples
        window_start = max(0, chunk_end - window_samples)
        window = audio[window_start:chunk_end]
        feats = extract_frame_features(window, sr)
        X_scaled = scaler.transform([feats])
        rf_prob = rf.predict_proba(X_scaled)[0][1]
        gbt_prob = gbt.predict_proba(X_scaled)[0][1]
        probs.append((rf_prob + gbt_prob) / 2)
        cursor = chunk_end

    return probs


def apply_hysteresis(hop_probs, t_on, t_off, min_speech_frames, min_gap_frames):
    """hop_probs -> binary 10ms-frame mask, via a simple two-threshold state machine
    plus minimum-duration merging."""
    # expand hop-level probabilities to a 10ms-frame probability sequence
    frame_probs = []
    for p in hop_probs:
        frame_probs.extend([p] * HOP_FRAMES)

    mask = [0] * len(frame_probs)
    state = 0
    for i, p in enumerate(frame_probs):
        if state == 0 and p >= t_on:
            state = 1
        elif state == 1 and p <= t_off:
            state = 0
        mask[i] = state

    # fill short gaps between speech regions
    i = 0
    while i < len(mask):
        if mask[i] == 0:
            j = i
            while j < len(mask) and mask[j] == 0:
                j += 1
            gap_len = j - i
            if 0 < i and j < len(mask) and gap_len < min_gap_frames:
                for k in range(i, j):
                    mask[k] = 1
            i = j
        else:
            i += 1

    # drop speech regions shorter than the minimum duration
    i = 0
    while i < len(mask):
        if mask[i] == 1:
            j = i
            while j < len(mask) and mask[j] == 1:
                j += 1
            if (j - i) < min_speech_frames:
                for k in range(i, j):
                    mask[k] = 0
            i = j
        else:
            i += 1

    return mask


def predict_mask_frame_v1(wav_path, rf, gbt, scaler, duration_ms, params=None):
    if params is None:
        params = load_tuned_params()
    hop_probs = predict_hop_probs(wav_path, rf, gbt, scaler, duration_ms)
    mask = apply_hysteresis(
        hop_probs, params["t_on"], params["t_off"],
        params["min_speech_frames"], params["min_gap_frames"],
    )
    n_expected = duration_ms // FRAME_MS
    if len(mask) < n_expected:
        mask = mask + [mask[-1] if mask else 0] * (n_expected - len(mask))
    return mask[:n_expected]
