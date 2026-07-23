"""
Grid-searches hysteresis thresholds for NOVA-VAD-frame-v3 on the VAL split
ONLY (data/scenes/val — never test or test_v2). Same objective (frame F1)
as v1/v2 tuning for comparability.

Run: python3 -m scripts.tune_frame_vad_v3
"""
import glob
import json
import os

import joblib

from scripts.frame_vad_v2 import predict_hop_probs_v2
from scripts.frame_vad_v1 import apply_hysteresis

VAL_DIR = "data/scenes/val"
OUT_PATH = "models/frame_vad_v3_hysteresis.json"

T_ON_GRID = [0.35, 0.45, 0.55, 0.65, 0.75]
T_OFF_GRID = [0.15, 0.25, 0.35, 0.45]
MIN_SPEECH_FRAMES = 15
MIN_GAP_FRAMES = 15


def frame_f1_precision_recall(pred, truth):
    n = min(len(pred), len(truth))
    pred, truth = pred[:n], truth[:n]
    tp = sum(1 for p, t in zip(pred, truth) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(pred, truth) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(pred, truth) if p == 0 and t == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1, precision, recall


def tune():
    rf = joblib.load("models/frame_vad_v3_rf.pkl")
    gbt = joblib.load("models/frame_vad_v3_gbt.pkl")
    scaler = joblib.load("models/frame_vad_v3_scaler.pkl")

    val_scenes = []
    for json_path in sorted(glob.glob(os.path.join(VAL_DIR, "*.json"))):
        with open(json_path) as f:
            meta = json.load(f)
        wav_path = json_path.replace(".json", ".wav")
        hop_probs = predict_hop_probs_v2(wav_path, rf, gbt, scaler, meta["duration_ms"])
        val_scenes.append((hop_probs, meta["frame_labels_10ms"]))
    print(f"Computed hop probabilities for {len(val_scenes)} val scenes")

    best = {"f1": -1}
    for t_on in T_ON_GRID:
        for t_off in T_OFF_GRID:
            if t_off >= t_on:
                continue
            all_pred, all_truth = [], []
            for hop_probs, truth in val_scenes:
                mask = apply_hysteresis(hop_probs, t_on, t_off, MIN_SPEECH_FRAMES, MIN_GAP_FRAMES)
                n = min(len(mask), len(truth))
                all_pred.extend(mask[:n])
                all_truth.extend(truth[:n])
            f1, precision, recall = frame_f1_precision_recall(all_pred, all_truth)
            print(f"    T_on={t_on}, T_off={t_off}: val F1={f1*100:.2f}% (P={precision*100:.1f}%, R={recall*100:.1f}%)")
            if f1 > best["f1"]:
                best = {"f1": f1, "precision": precision, "recall": recall, "t_on": t_on, "t_off": t_off}

    params = {
        "t_on": best["t_on"], "t_off": best["t_off"],
        "min_speech_frames": MIN_SPEECH_FRAMES, "min_gap_frames": MIN_GAP_FRAMES,
        "median_filter_size": 1,
        "tuned_on": "data/scenes/val (40 scenes, train-source files, seed=43)",
        "selection_objective": "frame F1 (same objective as v1/v2 tuning, for comparability)",
        "val_f1_pct": round(best["f1"] * 100, 2),
        "val_precision_pct": round(best["precision"] * 100, 2),
        "val_recall_pct": round(best["recall"] * 100, 2),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nBest: T_on={best['t_on']}, T_off={best['t_off']}, val F1={best['f1']*100:.2f}%")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    tune()
