"""
Grid-searches hysteresis thresholds (T_on, T_off) for NOVA-VAD-frame-v1 on
the DEV scenes ONLY (never the locked test scenes — plan Section 7.4/7.6:
"tune each model's public threshold and post-processing only on the
development split"). Saves the winning params to
models/frame_vad_v1_hysteresis.json for scripts/frame_vad_v1.py to load.

Run: python3 -m scripts.tune_frame_vad_v1
"""
import glob
import json
import math
import os

import joblib

from scripts.frame_vad_v1 import predict_hop_probs, apply_hysteresis

DEV_DIR = "data/scenes/dev"
OUT_PATH = "models/frame_vad_v1_hysteresis.json"

T_ON_GRID = [0.35, 0.45, 0.55, 0.65, 0.75]
T_OFF_GRID = [0.15, 0.25, 0.35, 0.45]
MIN_SPEECH_FRAMES = 15  # 150ms, fixed
MIN_GAP_FRAMES = 15     # 150ms, fixed


def frame_f1(pred, truth):
    n = min(len(pred), len(truth))
    pred, truth = pred[:n], truth[:n]
    tp = sum(1 for p, t in zip(pred, truth) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(pred, truth) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(pred, truth) if p == 0 and t == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def tune():
    rf = joblib.load("models/frame_vad_v1_rf.pkl")
    gbt = joblib.load("models/frame_vad_v1_gbt.pkl")
    scaler = joblib.load("models/frame_vad_v1_scaler.pkl")

    dev_scenes = []
    for json_path in sorted(glob.glob(os.path.join(DEV_DIR, "*.json"))):
        with open(json_path) as f:
            meta = json.load(f)
        wav_path = json_path.replace(".json", ".wav")
        print(f"  Computing hop probabilities for {meta['scene_id']}...")
        hop_probs = predict_hop_probs(wav_path, rf, gbt, scaler, meta["duration_ms"])
        dev_scenes.append((hop_probs, meta["frame_labels_10ms"]))

    print(f"\nGrid-searching {len(T_ON_GRID) * len(T_OFF_GRID)} threshold combinations "
          f"on {len(dev_scenes)} dev scenes...")

    best = {"f1": -1, "t_on": None, "t_off": None}
    for t_on in T_ON_GRID:
        for t_off in T_OFF_GRID:
            if t_off >= t_on:
                continue
            all_pred, all_truth = [], []
            for hop_probs, truth in dev_scenes:
                mask = apply_hysteresis(hop_probs, t_on, t_off, MIN_SPEECH_FRAMES, MIN_GAP_FRAMES)
                n = min(len(mask), len(truth))
                all_pred.extend(mask[:n])
                all_truth.extend(truth[:n])

            f1 = frame_f1(all_pred, all_truth)
            print(f"    T_on={t_on}, T_off={t_off}: dev F1={f1*100:.2f}%")
            if f1 > best["f1"]:
                best = {"f1": f1, "t_on": t_on, "t_off": t_off}

    params = {
        "t_on": best["t_on"],
        "t_off": best["t_off"],
        "min_speech_frames": MIN_SPEECH_FRAMES,
        "min_gap_frames": MIN_GAP_FRAMES,
        "tuned_on": "data/scenes/dev (20 scenes, train-source files)",
        "dev_f1_pct": round(best["f1"] * 100, 2),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(params, f, indent=2)

    print(f"\nBest: T_on={best['t_on']}, T_off={best['t_off']}, dev F1={best['f1']*100:.2f}%")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    tune()
