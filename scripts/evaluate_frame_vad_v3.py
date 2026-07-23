"""
SINGLE final evaluation of NOVA-VAD-frame-v3 on test_v2 (the expanded,
now-locked test set from Part 1 of round 2 — NOT the original 40-scene
test/, which stays as historical record). Run once, after model+threshold
are frozen from train/val-only work.

Run: python3 -m scripts.evaluate_frame_vad_v3
"""
import glob
import json
import os

import joblib

from scripts.frame_benchmark import confusion, metrics_from_confusion, cluster_bootstrap_ci, per_scene_accuracy
from scripts.frame_vad_v3 import predict_mask_frame_v3, load_tuned_params

TEST_V2_DIR = "data/scenes/test_v2"


def load_test_v2_scenes():
    scenes = []
    for json_path in sorted(glob.glob(os.path.join(TEST_V2_DIR, "*.json"))):
        with open(json_path) as f:
            meta = json.load(f)
        wav_path = json_path.replace(".json", ".wav")
        scenes.append((wav_path, meta))
    return scenes


def main():
    rf = joblib.load("models/frame_vad_v3_rf.pkl")
    gbt = joblib.load("models/frame_vad_v3_gbt.pkl")
    scaler = joblib.load("models/frame_vad_v3_scaler.pkl")
    params = load_tuned_params()
    print(f"Using tuned params (from VAL scenes only): {params}")

    scenes = load_test_v2_scenes()
    print(f"Loaded {len(scenes)} test_v2 scenes "
          f"(FIRST AND ONLY test contact for v3)")

    all_pred, all_truth = [], []
    scene_accuracies = []
    per_scene_out = []
    per_condition = {}
    for wav_path, meta in scenes:
        pred = predict_mask_frame_v3(wav_path, rf, gbt, scaler, meta["duration_ms"], params)
        truth = meta["frame_labels_10ms"]
        n = min(len(pred), len(truth))
        pred, truth = pred[:n], truth[:n]
        all_pred.extend(pred)
        all_truth.extend(truth)
        scene_accuracies.append(per_scene_accuracy(pred, truth))
        per_scene_out.append({
            "scene_id": meta["scene_id"], "condition": meta["condition"],
            "source_noise_file": meta["source_noise_file"],
            "accuracy": round(per_scene_accuracy(pred, truth), 2),
            "pred": pred, "truth": truth,
        })
        cond = meta["condition"]
        per_condition.setdefault(cond, {"pred": [], "truth": []})
        per_condition[cond]["pred"].extend(pred)
        per_condition[cond]["truth"].extend(truth)

    overall = metrics_from_confusion(*confusion(all_pred, all_truth))
    ci_lo, ci_hi = cluster_bootstrap_ci(scene_accuracies)
    overall["accuracy_ci95"] = [ci_lo, ci_hi]
    by_condition = {c: metrics_from_confusion(*confusion(d["pred"], d["truth"])) for c, d in per_condition.items()}

    print(f"\nNOVA-VAD-frame-v3: {overall['accuracy']}% accuracy (95% CI [{ci_lo},{ci_hi}]), "
          f"F1={overall['f1']}%, MCC={overall['mcc']}, precision={overall['precision']}%, recall={overall['recall']}%")
    print("Per-condition:", json.dumps(by_condition, indent=2))

    with open("reports/per_scene_test_v2_v3.json", "w") as f:
        json.dump({"NOVA-VAD-frame-v3": per_scene_out}, f)

    with open("reports/frame_v3_test_v2_result.json", "w") as f:
        json.dump({"overall": overall, "by_condition": by_condition}, f, indent=2)
    print("\nSaved reports/frame_v3_test_v2_result.json and reports/per_scene_test_v2_v3.json")


if __name__ == "__main__":
    main()
