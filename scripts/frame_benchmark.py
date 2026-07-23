"""
Runs all 5 systems (NOVA-VAD, WebRTC, Silero, Pyannote, SpeechBrain) through
the unified predict_mask() interface on the locked test scenes, scores them
at 10ms frame resolution, and produces a report with cluster-bootstrap
confidence intervals (resampling whole scenes, not individual frames, since
adjacent 10ms frames are not independent — plan Section 3.4).

Run: python3 -m scripts.frame_benchmark
"""
import glob
import json
import math
import os
import random
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np

from scripts.frame_vad_adapters import (
    predict_mask_nova, predict_mask_webrtc, predict_mask_silero,
    predict_mask_pyannote, predict_mask_speechbrain,
)

TEST_DIR = "data/scenes/test"
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42


def load_test_scenes():
    scenes = []
    for json_path in sorted(glob.glob(os.path.join(TEST_DIR, "*.json"))):
        with open(json_path) as f:
            meta = json.load(f)
        wav_path = json_path.replace(".json", ".wav")
        scenes.append((wav_path, meta))
    return scenes


def confusion(pred, truth):
    pred = np.array(pred)
    truth = np.array(truth)
    tp = int(np.sum((pred == 1) & (truth == 1)))
    tn = int(np.sum((pred == 0) & (truth == 0)))
    fp = int(np.sum((pred == 1) & (truth == 0)))
    fn = int(np.sum((pred == 0) & (truth == 1)))
    return tp, tn, fp, fn


def metrics_from_confusion(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0
    return {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "mcc": round(mcc, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def per_scene_accuracy(pred, truth):
    pred = np.array(pred)
    truth = np.array(truth)
    return float(np.mean(pred == truth)) * 100


def cluster_bootstrap_ci(scene_accuracies, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    rng = random.Random(seed)
    n = len(scene_accuracies)
    means = []
    for _ in range(n_bootstrap):
        sample = [scene_accuracies[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_bootstrap)]
    hi = means[int(0.975 * n_bootstrap)]
    return round(lo, 2), round(hi, 2)


def run_system(name, predict_fn, scenes):
    print(f"\n  Running {name} on {len(scenes)} test scenes...")
    all_pred, all_truth = [], []
    per_condition = {}
    scene_accuracies = []

    for wav_path, meta in scenes:
        truth = meta["frame_labels_10ms"]
        pred = predict_fn(wav_path, meta)
        n = min(len(pred), len(truth))
        pred, truth = pred[:n], truth[:n]

        all_pred.extend(pred)
        all_truth.extend(truth)
        scene_accuracies.append(per_scene_accuracy(pred, truth))

        cond = meta["condition"]
        per_condition.setdefault(cond, {"pred": [], "truth": []})
        per_condition[cond]["pred"].extend(pred)
        per_condition[cond]["truth"].extend(truth)

    overall = metrics_from_confusion(*confusion(all_pred, all_truth))
    ci_lo, ci_hi = cluster_bootstrap_ci(scene_accuracies)
    overall["accuracy_ci95"] = [ci_lo, ci_hi]

    condition_results = {}
    for cond, d in per_condition.items():
        condition_results[cond] = metrics_from_confusion(*confusion(d["pred"], d["truth"]))

    print(f"    {name}: {overall['accuracy']}% frame accuracy "
          f"(95% CI [{ci_lo}, {ci_hi}]), F1={overall['f1']}%, MCC={overall['mcc']}")

    return {"overall": overall, "by_condition": condition_results}


def main():
    scenes = load_test_scenes()
    if not scenes:
        print("No test scenes found — run `python3 -m scripts.generate_scenes` first.")
        return

    print(f"Loaded {len(scenes)} test scenes from {TEST_DIR}")
    results = {}

    # NOVA-VAD
    rf = joblib.load("models/nova_vad_rf.pkl")
    gbt = joblib.load("models/nova_vad_gbt.pkl")
    scaler = joblib.load("models/nova_vad_scaler.pkl")
    results["NOVA-VAD"] = run_system(
        "NOVA-VAD",
        lambda wav, meta: predict_mask_nova(wav, rf, gbt, scaler, meta["duration_ms"]),
        scenes,
    )

    # WebRTC
    results["WebRTC VAD"] = run_system(
        "WebRTC VAD",
        lambda wav, meta: predict_mask_webrtc(wav, meta["duration_ms"]),
        scenes,
    )

    # Silero
    from silero_vad import load_silero_vad
    silero_model = load_silero_vad()
    results["Silero VAD"] = run_system(
        "Silero VAD",
        lambda wav, meta: predict_mask_silero(wav, silero_model, meta["duration_ms"]),
        scenes,
    )

    # Pyannote
    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection
    token = os.environ.get("HF_TOKEN")
    pmodel = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=token)
    pyannote_pipeline = VoiceActivityDetection(segmentation=pmodel)
    pyannote_pipeline.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})
    results["Pyannote VAD"] = run_system(
        "Pyannote VAD",
        lambda wav, meta: predict_mask_pyannote(wav, pyannote_pipeline, meta["duration_ms"]),
        scenes,
    )

    # SpeechBrain
    from speechbrain.inference.VAD import VAD
    sb_model = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="models/speechbrain_vad")
    results["SpeechBrain VAD"] = run_system(
        "SpeechBrain VAD",
        lambda wav, meta: predict_mask_speechbrain(wav, sb_model, meta["duration_ms"]),
        scenes,
    )

    os.makedirs("reports", exist_ok=True)
    with open("reports/frame_level_benchmark_v1.json", "w") as f:
        json.dump(results, f, indent=2)

    write_report(results, len(scenes))
    print("\nWrote reports/frame_level_benchmark_v1.json and reports/frame_level_benchmark_v1.md")


def write_report(results, n_scenes):
    lines = []
    lines.append("# NOVA-VAD Frame-Level Benchmark v1\n")
    lines.append(f"Generated from {n_scenes} locked test scenes (`data/scenes/test/`), "
                  "built from files never used to train NOVA-VAD, scored at 10ms frame "
                  "resolution using each system's native output "
                  "(see `scripts/frame_vad_adapters.py`).\n")
    lines.append("Scope: clean + 3 noise-level (SNR) conditions on speech mixed into the same "
                  "file as noise/music. Codec/RTC transmission conditions and non-music hard "
                  "negatives are **not yet covered** — see plan Section 7.2 Layers 3-4.\n")

    lines.append("## Overall frame-level results (95% CI via cluster bootstrap over scenes)\n")
    lines.append("| Model | Accuracy | 95% CI | Precision | Recall | F1 | MCC |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, r in results.items():
        o = r["overall"]
        lines.append(
            f"| {name} | {o['accuracy']}% | [{o['accuracy_ci95'][0]}, {o['accuracy_ci95'][1]}] "
            f"| {o['precision']}% | {o['recall']}% | {o['f1']}% | {o['mcc']} |"
        )

    lines.append("\n## Per-condition breakdown (accuracy %)\n")
    conditions = list(next(iter(results.values()))["by_condition"].keys())
    lines.append("| Model | " + " | ".join(conditions) + " |")
    lines.append("|---|" + "---|" * len(conditions))
    for name, r in results.items():
        row = [f"{r['by_condition'][c]['accuracy']}%" for c in conditions]
        lines.append(f"| {name} | " + " | ".join(row) + " |")

    with open("reports/frame_level_benchmark_v1.md", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
