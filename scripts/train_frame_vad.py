"""
Trains NOVA-VAD-frame-v1: a Random Forest + Gradient Boosting ensemble on
TRUE per-frame labels sampled from the 100 mixed train scenes, using the
causal reduced feature set (scripts/frame_features.py). This directly
targets the failure mode found in reports/decision_v1.md — v0's features
were computed as whole-file aggregates and didn't transfer to localized
frame-level detection (-0.28 MCC, worse than random).

Window: 320ms causal lookback, 100ms hop (label = majority ground-truth
class over the most recent 100ms, given up to 320ms of preceding audio —
no look-ahead, consistent with real-time streaming use).

Run: python3 -m scripts.train_frame_vad
"""
import glob
import json
import os

import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from scripts.frame_features import extract_frame_features

TRAIN_DIR = "data/scenes/train"
SR = 16000
WINDOW_S = 0.32
HOP_S = 0.10

OUT_RF = "models/frame_vad_v1_rf.pkl"
OUT_GBT = "models/frame_vad_v1_gbt.pkl"
OUT_SCALER = "models/frame_vad_v1_scaler.pkl"


def build_training_examples():
    window_samples = int(WINDOW_S * SR)
    hop_samples = int(HOP_S * SR)

    X, y = [], []
    scene_paths = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.json")))
    print(f"Building training examples from {len(scene_paths)} train scenes...")

    for i, json_path in enumerate(scene_paths):
        with open(json_path) as f:
            meta = json.load(f)
        wav_path = json_path.replace(".json", ".wav")
        audio, sr = librosa.load(wav_path, sr=SR, mono=True)
        frame_labels = meta["frame_labels_10ms"]  # 10ms resolution ground truth

        cursor = 0
        while cursor + hop_samples <= len(audio):
            chunk_end = cursor + hop_samples
            window_start = max(0, chunk_end - window_samples)
            window = audio[window_start:chunk_end]

            feats = extract_frame_features(window, sr)

            start_frame_10ms = cursor // (SR // 100)
            end_frame_10ms = min(len(frame_labels), chunk_end // (SR // 100))
            chunk_labels = frame_labels[start_frame_10ms:end_frame_10ms]
            if not chunk_labels:
                cursor = chunk_end
                continue
            label = 1 if sum(chunk_labels) / len(chunk_labels) >= 0.5 else 0

            X.append(feats)
            y.append(label)
            cursor = chunk_end

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(scene_paths)} scenes processed, {len(X)} examples so far")

    X = np.array(X)
    y = np.array(y)
    return X, y


def train():
    X, y = build_training_examples()
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    print(f"\nTotal training examples: {len(y)} ({n_pos} speech, {n_neg} no-speech, "
          f"{n_pos/len(y)*100:.1f}% positive)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training Random Forest + Gradient Boosting...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced")
    gbt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    rf.fit(X_scaled, y)
    gbt.fit(X_scaled, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, OUT_RF)
    joblib.dump(gbt, OUT_GBT)
    joblib.dump(scaler, OUT_SCALER)
    print(f"\nSaved {OUT_RF}, {OUT_GBT}, {OUT_SCALER}")

    # quick in-sample sanity check (NOT a real evaluation — see scripts/frame_benchmark.py
    # run against the locked test scenes for the real number)
    rf_probs = rf.predict_proba(X_scaled)[:, 1]
    gbt_probs = gbt.predict_proba(X_scaled)[:, 1]
    preds = ((rf_probs + gbt_probs) / 2 > 0.5).astype(int)
    train_acc = np.mean(preds == y) * 100
    print(f"In-sample (train) accuracy: {train_acc:.1f}% — sanity check only, not a real metric")


if __name__ == "__main__":
    train()
