"""
Full precision/recall/F1/MCC sweep across hysteresis thresholds for
NOVA-VAD-frame-v2, on the VAL split ONLY (data/scenes/val — never test).
Round 1 only reported the single F1-optimal operating point; this reports
the entire curve so a threshold can be chosen deliberately for NOVA
Verify's actual use case rather than by a single blind objective.

Run: python3 -m scripts.threshold_sweep_v2
"""
import glob
import json
import math
import os

import joblib

from scripts.frame_vad_v2 import predict_hop_probs_v2
from scripts.frame_vad_v1 import apply_hysteresis

VAL_DIR = "data/scenes/val"
OUT_PATH = "reports/threshold_sweep_v2.json"

T_ON_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
T_OFF_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
MIN_SPEECH_FRAMES = 15
MIN_GAP_FRAMES = 15


def metrics(pred, truth):
    n = min(len(pred), len(truth))
    pred, truth = pred[:n], truth[:n]
    tp = sum(1 for p, t in zip(pred, truth) if p == 1 and t == 1)
    tn = sum(1 for p, t in zip(pred, truth) if p == 0 and t == 0)
    fp = sum(1 for p, t in zip(pred, truth) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(pred, truth) if p == 0 and t == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0
    return accuracy, precision, recall, f1, mcc


def sweep():
    rf = joblib.load("models/frame_vad_v2_rf.pkl")
    gbt = joblib.load("models/frame_vad_v2_gbt.pkl")
    scaler = joblib.load("models/frame_vad_v2_scaler.pkl")

    val_scenes = []
    for json_path in sorted(glob.glob(os.path.join(VAL_DIR, "*.json"))):
        with open(json_path) as f:
            meta = json.load(f)
        wav_path = json_path.replace(".json", ".wav")
        hop_probs = predict_hop_probs_v2(wav_path, rf, gbt, scaler, meta["duration_ms"])
        val_scenes.append((hop_probs, meta["frame_labels_10ms"]))
    print(f"Computed hop probabilities for {len(val_scenes)} val scenes")

    curve = []
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
            acc, prec, rec, f1, mcc = metrics(all_pred, all_truth)
            curve.append({
                "t_on": t_on, "t_off": t_off,
                "accuracy_pct": round(acc * 100, 2),
                "precision_pct": round(prec * 100, 2),
                "recall_pct": round(rec * 100, 2),
                "f1_pct": round(f1 * 100, 2),
                "mcc": round(mcc, 4),
            })

    curve.sort(key=lambda r: -r["f1_pct"])
    with open(OUT_PATH, "w") as f:
        json.dump(curve, f, indent=2)

    print(f"\n{'T_on':<6}{'T_off':<7}{'Acc%':<8}{'Prec%':<8}{'Rec%':<8}{'F1%':<8}{'MCC':<8}")
    for r in curve:
        print(f"{r['t_on']:<6}{r['t_off']:<7}{r['accuracy_pct']:<8}{r['precision_pct']:<8}"
              f"{r['recall_pct']:<8}{r['f1_pct']:<8}{r['mcc']:<8}")
    print(f"\nSaved full {len(curve)}-point curve to {OUT_PATH}")


if __name__ == "__main__":
    sweep()
