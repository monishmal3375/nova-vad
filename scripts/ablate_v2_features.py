"""
Feature ablation for the clean-audio regression diagnosis (Part 2.1): trains
5 variants of NOVA-VAD-frame-v2 on the SAME train+train2 data — the full
62-feature model, and 4 variants each missing exactly one of the round-1
features (periodicity_strength, estimated_f0_hz, flatness_mean,
flatness_std) — then evaluates each on the VAL split's CLEAN-condition
scenes only (never test) to see which feature's removal changes clean
performance, if any.

Feature extraction (the expensive part) runs once and is cached; only the
lightweight classifier training is repeated per variant.

Run: python3 -m scripts.ablate_v2_features
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
from scripts.frame_vad_v1 import apply_hysteresis
from scripts.frame_vad_v2 import load_tuned_params

TRAIN_DIRS = ["data/scenes/train", "data/scenes/train2"]
VAL_DIR = "data/scenes/val"
SR = 16000
WINDOW_S = 0.32
HOP_S = 0.10
CACHE_PATH = "reports/ablation_train_features_cache.npz"

# indices 58-61 of the 62-dim vector, per scripts/frame_features_v2.py
NEW_FEATURE_NAMES = ["periodicity_strength", "estimated_f0_hz", "flatness_mean", "flatness_std"]
NEW_FEATURE_INDICES = [58, 59, 60, 61]


def build_or_load_training_cache():
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached training features from {CACHE_PATH}")
        data = np.load(CACHE_PATH)
        return data["X"], data["y"]

    window_samples = int(WINDOW_S * SR)
    hop_samples = int(HOP_S * SR)
    X, y = [], []
    scene_paths = []
    for d in TRAIN_DIRS:
        scene_paths.extend(sorted(glob.glob(os.path.join(d, "*.json"))))
    print(f"Building (cached) training features from {len(scene_paths)} scenes...")

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

            start_frame = cursor // (SR // 100)
            end_frame = min(len(frame_labels), chunk_end // (SR // 100))
            chunk_labels = frame_labels[start_frame:end_frame]
            if not chunk_labels:
                cursor = chunk_end
                continue
            label = 1 if sum(chunk_labels) / len(chunk_labels) >= 0.5 else 0
            X.append(feats)
            y.append(label)
            cursor = chunk_end

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(scene_paths)} scenes")

    X = np.array(X)
    y = np.array(y)
    os.makedirs("reports", exist_ok=True)
    np.savez(CACHE_PATH, X=X, y=y)
    print(f"Cached to {CACHE_PATH}")
    return X, y


def train_variant(X, y, drop_index=None):
    if drop_index is not None:
        keep = [i for i in range(X.shape[1]) if i != drop_index]
        X = X[:, keep]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced")
    gbt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    rf.fit(X_scaled, y)
    gbt.fit(X_scaled, y)
    return rf, gbt, scaler


def eval_on_val_clean(rf, gbt, scaler, drop_index=None):
    params = load_tuned_params()
    window_samples = int(WINDOW_S * SR)
    hop_samples = int(HOP_S * SR)

    clean_scenes = sorted(glob.glob(os.path.join(VAL_DIR, "*_clean.json")))
    all_pred, all_truth = [], []
    for json_path in clean_scenes:
        with open(json_path) as f:
            meta = json.load(f)
        wav_path = json_path.replace(".json", ".wav")
        audio, sr = librosa.load(wav_path, sr=SR, mono=True)

        hop_probs = []
        cursor = 0
        while cursor + hop_samples <= len(audio):
            chunk_end = cursor + hop_samples
            window_start = max(0, chunk_end - window_samples)
            window = audio[window_start:chunk_end]
            feats = extract_frame_features_v2(window, sr)
            if drop_index is not None:
                keep = [i for i in range(len(feats)) if i != drop_index]
                feats = feats[keep]
            X_scaled = scaler.transform([feats])
            rf_prob = rf.predict_proba(X_scaled)[0][1]
            gbt_prob = gbt.predict_proba(X_scaled)[0][1]
            hop_probs.append((rf_prob + gbt_prob) / 2)
            cursor = chunk_end

        mask = apply_hysteresis(hop_probs, params["t_on"], params["t_off"],
                                 params["min_speech_frames"], params["min_gap_frames"])
        truth = meta["frame_labels_10ms"]
        n = min(len(mask), len(truth))
        all_pred.extend(mask[:n])
        all_truth.extend(truth[:n])

    acc = float(np.mean(np.array(all_pred) == np.array(all_truth))) * 100
    return acc, len(clean_scenes)


def main():
    X, y = build_or_load_training_cache()
    print(f"\nTraining set: {len(y)} examples, {X.shape[1]} features")

    print("\n--- Variant: FULL (all 62 features) ---")
    rf, gbt, scaler = train_variant(X, y, drop_index=None)
    full_acc, n_scenes = eval_on_val_clean(rf, gbt, scaler, drop_index=None)
    print(f"Val-clean accuracy (full model): {full_acc:.2f}% ({n_scenes} clean val scenes)")

    results = {"full": full_acc}
    for name, idx in zip(NEW_FEATURE_NAMES, NEW_FEATURE_INDICES):
        print(f"\n--- Variant: WITHOUT {name} (index {idx}) ---")
        rf_a, gbt_a, scaler_a = train_variant(X, y, drop_index=idx)
        acc, _ = eval_on_val_clean(rf_a, gbt_a, scaler_a, drop_index=idx)
        delta = acc - full_acc
        print(f"Val-clean accuracy (without {name}): {acc:.2f}% (delta vs full: {delta:+.2f}pp)")
        results[f"without_{name}"] = acc

    with open("reports/ablation_v2_clean_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved reports/ablation_v2_clean_results.json")


if __name__ == "__main__":
    main()
