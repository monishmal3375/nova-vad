"""
Trains NOVA-VAD-frame-v3: same 62-feature set and architecture as v2, but
adds the 150 targeted train3 scenes (0dB + -5dB only) on top of v2's
train+train2 pool — 100 + 300 + 150 = 550 scenes. No test-set or val-set
contact.

Run: python3 -m scripts.train_frame_vad_v3
"""
import glob
import json
import os

import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from scripts.frame_features_v2 import extract_frame_features_v2

TRAIN_DIRS = ["data/scenes/train", "data/scenes/train2", "data/scenes/train3"]
SR = 16000
WINDOW_S = 0.32
HOP_S = 0.10

OUT_RF = "models/frame_vad_v3_rf.pkl"
OUT_GBT = "models/frame_vad_v3_gbt.pkl"
OUT_SCALER = "models/frame_vad_v3_scaler.pkl"


def build_training_examples():
    window_samples = int(WINDOW_S * SR)
    hop_samples = int(HOP_S * SR)

    X, y = [], []
    scene_paths = []
    for d in TRAIN_DIRS:
        scene_paths.extend(sorted(glob.glob(os.path.join(d, "*.json"))))
    print(f"Building training examples from {len(scene_paths)} train scenes "
          f"({' + '.join(TRAIN_DIRS)})...")

    for i, json_path in enumerate(scene_paths):
        with open(json_path) as f:
            meta = json.load(f)
        wav_path = json_path.replace(".json", ".wav")
        audio, sr = librosa.load(wav_path, sr=SR, mono=True)
        frame_labels = meta["frame_labels_10ms"]

        cursor = 0
        while cursor + hop_samples <= len(audio):
            chunk_end = cursor + hop_samples
            window_start = max(0, chunk_end - window_samples)
            window = audio[window_start:chunk_end]
            feats = extract_frame_features_v2(window, sr)

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

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(scene_paths)} scenes processed, {len(X)} examples so far")

    return np.array(X), np.array(y)


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

    rf_probs = rf.predict_proba(X_scaled)[:, 1]
    gbt_probs = gbt.predict_proba(X_scaled)[:, 1]
    preds = ((rf_probs + gbt_probs) / 2 > 0.5).astype(int)
    train_acc = np.mean(preds == y) * 100
    print(f"In-sample (train) accuracy: {train_acc:.1f}% — sanity check only, not a real metric")


if __name__ == "__main__":
    train()
