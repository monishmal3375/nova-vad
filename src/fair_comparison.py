"""
Fair, apples-to-apples comparison of NOVA-VAD against every baseline VAD
system on the EXACT SAME held-out test set that produced NOVA-VAD's
documented held-out accuracy (see results/final_model_report.json,
produced by `python3 -m src.experiment final`). The held-out set size is
whatever held_out_split() currently produces from data/speech + data/noise
on disk (it is NOT a fixed constant -- it grows as the dataset is expanded).

Why this file exists (separate from src/benchmark.py):
  benchmark.py carves its own 80/20 split via split_dataset() — a different,
  non-stratified split from a smaller/older dataset snapshot, used to
  produce the older README numbers (Silero 96.0%, Pyannote 92.0%, etc. on
  100 files). That split is NOT the same held-out set used for NOVA-VAD's
  documented held-out number, so comparing across the two is invalid —
  different test sets, different data. This script fixes that by reusing
  src.experiment.held_out_split() (same seed=42, same group-stratification)
  to reconstruct the identical held-out test set, then runs every baseline
  from benchmark.py against THAT set instead.

Methodology, per system:
  - NOVA-VAD: evaluated with its own real training/eval pipeline — audio is
    standardized to a 1-second window before feature extraction (see
    src/experiment.py's WINDOW_SECONDS / _standardize_duration), using the
    saved final model (models/nova_vad_rf.pkl, nova_vad_gbt.pkl,
    nova_vad_scaler.pkl). This windowing is NOT applied to any other model —
    it was specifically an internal fix for NOVA-VAD's own feature
    extraction / training confound (raw speech clips ~1s, raw noise clips
    ~3.5-4s), not a general preprocessing step other systems expect or were
    designed around.
  - Every other baseline (WebRTC, Silero, Pyannote, SpeechBrain, TEN-VAD,
    Energy Threshold): evaluated on RAW, natural-length audio exactly as
    src/benchmark.py already does — these are pretrained/black-box systems,
    not retrained here, and are run through their own native preprocessing.
"""
import os
import sys
import json
import time
import platform
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment import load_or_build_features, held_out_split, _extract_features_windowed
from src.benchmark import (
    run_webrtc, run_silero, run_pyannote, run_speechbrain, run_ten_vad,
    run_energy_threshold, compute_metrics, model_size_bytes, human_size,
    load_noise_category_manifest, build_category_breakdown,
)

RESULTS_DIR = "results"
SPEECH_DIR = "data/speech"
NOISE_DIR = "data/noise"


def get_fair_test_split():
    """Reconstructs the held-out test set using the SAME methodology that
    produced NOVA-VAD's documented held-out number: same seed=42, same
    held_out_split() grouping logic, same feature cache
    (data/_feature_cache.joblib). The exact file count is derived from the
    dataset on disk (via held_out_split's stratified-by-group logic), not
    hardcoded, so this stays correct as the dataset is expanded over time --
    a hardcoded expected count here previously required editing this file
    every time the dataset size changed, which is exactly the kind of stale
    assumption that silently invalidates a "fair" comparison."""
    data = load_or_build_features()
    y, groups, filenames = data["y"], data["groups"], data["filenames"]
    split_keys = data.get("split_keys")
    train_idx, test_idx = held_out_split(y, groups, split_keys=split_keys)

    print(f"  Held-out test set reconstructed via held_out_split(seed=42): "
          f"{len(test_idx)} files out of {len(y)} total in data/.")

    test_speech = sorted(filenames[i] for i in test_idx if y[i] == 1)
    test_noise = sorted(filenames[i] for i in test_idx if y[i] == 0)
    return test_speech, test_noise, test_idx, data


def run_nova_vad_fair(test_speech, test_noise, data, test_idx):
    """
    Runs NOVA-VAD using its actual production methodology: 1-second
    duration-standardized feature extraction (matches src/experiment.py and
    how src/stream.py feeds real-time audio), with the saved final model
    trained on the train/val pool only (disjoint from this held-out test set).
    """
    rf = joblib.load("models/nova_vad_rf.pkl")
    gbt = joblib.load("models/nova_vad_gbt.pkl")
    scaler = joblib.load("models/nova_vad_scaler.pkl")

    filenames = data["filenames"]
    fname_to_idx = {f: i for i, f in enumerate(filenames)}
    X = data["X"]

    speech_dirset = set(os.listdir(SPEECH_DIR))

    results = []
    latencies = []
    start = time.time()

    for f in test_speech:
        idx = fname_to_idx[f]
        feats = X[idx]
        Xs = scaler.transform([feats])
        rf_prob = rf.predict_proba(Xs)[0][1]
        gbt_prob = gbt.predict_proba(Xs)[0][1]
        avg_prob = (rf_prob + gbt_prob) / 2
        pred = 1 if avg_prob > 0.5 else 0
        results.append({"true": 1, "pred": pred, "file": f,
                         "confidence": round(float(avg_prob if pred == 1 else 1 - avg_prob) * 100, 2)})

    for f in test_noise:
        idx = fname_to_idx[f]
        feats = X[idx]
        Xs = scaler.transform([feats])
        rf_prob = rf.predict_proba(Xs)[0][1]
        gbt_prob = gbt.predict_proba(Xs)[0][1]
        avg_prob = (rf_prob + gbt_prob) / 2
        pred = 1 if avg_prob > 0.5 else 0
        results.append({"true": 0, "pred": pred, "file": f,
                         "confidence": round(float(avg_prob if pred == 1 else 1 - avg_prob) * 100, 2)})

    # measure real per-file latency (feature extraction from disk + inference)
    # separately, matching experiment.py's methodology (first 50 test files)
    for fname in ([test_speech[i] for i in range(min(25, len(test_speech)))] +
                  [test_noise[i] for i in range(min(25, len(test_noise)))]):
        path = os.path.join(SPEECH_DIR, fname) if fname in speech_dirset else os.path.join(NOISE_DIR, fname)
        t0 = time.time()
        feats = _extract_features_windowed(path)
        Xs = scaler.transform([feats])
        _ = (rf.predict_proba(Xs)[0][1] + gbt.predict_proba(Xs)[0][1]) / 2
        latencies.append((time.time() - t0) * 1000)

    elapsed = time.time() - start
    model_size = model_size_bytes([
        "models/nova_vad_rf.pkl", "models/nova_vad_gbt.pkl", "models/nova_vad_scaler.pkl"
    ])
    metrics = compute_metrics(results, elapsed, "NOVA-VAD", model_size_bytes_val=model_size)
    metrics["mean_latency_ms"] = round(float(np.mean(latencies)), 2)
    metrics["p95_latency_ms"] = round(float(np.percentile(latencies, 95)), 2)
    return metrics


def print_and_diagnose(results, n_test):
    print("\n" + "=" * 96)
    print("   NOVA-VAD FAIR COMPARISON — identical held-out test set for every model")
    print("=" * 96)
    print(f"  Test files: {n_test} (same split used for NOVA-VAD's documented held-out number)\n")
    print(f"  {'Model':<20} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AvgLatency':>11} {'ModelSize':>10}")
    print("  " + "-" * 90)
    for r in results:
        latency_str = f"{r['mean_latency_ms']}ms" if r['mean_latency_ms'] is not None else "N/A"
        print(f"  {r['name']:<20} {r['accuracy']:>8}% {r['precision']:>9}% {r['recall']:>7}% {r['f1']:>7}% {latency_str:>11} {r['model_size_human']:>10}")
    print("=" * 96)

    nova = next(r for r in results if r["name"] == "NOVA-VAD")
    for other in results:
        if other["name"] == "NOVA-VAD":
            continue
        diff = round(nova["accuracy"] - other["accuracy"], 2)
        sign = "+" if diff >= 0 else ""
        print(f"  NOVA-VAD vs {other['name']:<18} {sign}{diff}%")

    # sanity check: flag any baseline that looks anomalous (e.g. naive
    # Energy Threshold beating trained/pretrained models, which previously
    # happened due to a preprocessing bug — see benchmark.py's STEP 1 comment)
    print("\n  Sanity checks:")
    energy = next((r for r in results if r["name"] == "Energy Threshold"), None)
    trained = [r for r in results if r["name"] not in ("Energy Threshold",)]
    if energy and trained:
        beats = [r["name"] for r in trained if energy["accuracy"] > r["accuracy"]]
        if beats:
            print(f"    WARNING: naive Energy Threshold ({energy['accuracy']}%) beats: {beats} — investigate before trusting.")
        else:
            print(f"    OK: naive Energy Threshold ({energy['accuracy']}%) does not beat any trained/pretrained model.")
    print("=" * 96)


def save_artifacts(results, n_test, test_speech, test_noise, category_breakdowns=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    summary = {
        "timestamp": timestamp,
        "description": (
            "Fair, apples-to-apples comparison: every model evaluated on the "
            "IDENTICAL held-out test set that produced NOVA-VAD's documented "
            "held-out accuracy (results/final_model_report.json). "
            "Test set reconstructed via src.experiment.held_out_split(seed=42), "
            "stratified by group (speech / UrbanSound8K noise category). "
            "NOVA-VAD is evaluated with its real training methodology "
            "(1-second duration-standardized features); every other baseline "
            "is evaluated on raw natural-length audio via its own native "
            "preprocessing, exactly as src/benchmark.py already does."
        ),
        "n_test_files": n_test,
        "n_test_speech": len(test_speech),
        "n_test_noise": len(test_noise),
        "held_out_split_seed": 42,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "models": [],
    }

    for r in results:
        entry = {k: v for k, v in r.items() if k not in ("false_positives", "false_negatives")}
        entry["false_positive_files"] = [f["file"] for f in r.get("false_positives", [])]
        entry["false_negative_files"] = [f["file"] for f in r.get("false_negatives", [])]
        summary["models"].append(entry)

    if category_breakdowns:
        summary["nova_vad_noise_category_breakdown"] = category_breakdowns

    path = os.path.join(RESULTS_DIR, "fair_comparison_final.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved fair comparison: {path}")

    # FP/FN diagnostic report for NOVA-VAD, same style as benchmark.py's
    nova = next((r for r in results if r["name"] == "NOVA-VAD"), None)
    if nova:
        fpfn_path = os.path.join(RESULTS_DIR, "fair_comparison_false_positives_negatives.txt")
        with open(fpfn_path, "w") as f:
            f.write("NOVA-VAD false positive / false negative report (FAIR comparison)\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Test set size: {n_test} (identical split used for the documented held-out number)\n\n")
            f.write(f"FALSE POSITIVES ({len(nova['false_positives'])}) — noise misclassified as SPEECH\n")
            f.write("-" * 60 + "\n")
            for item in nova["false_positives"]:
                f.write(f"  {item['file']:<30} predicted=SPEECH actual=NO SPEECH confidence={item.get('confidence','N/A')}\n")
            f.write(f"\nFALSE NEGATIVES ({len(nova['false_negatives'])}) — speech misclassified as NO SPEECH\n")
            f.write("-" * 60 + "\n")
            for item in nova["false_negatives"]:
                f.write(f"  {item['file']:<30} predicted=NO SPEECH actual=SPEECH confidence={item.get('confidence','N/A')}\n")
        print(f"  Saved FP/FN report: {fpfn_path}")

    return path


def main():
    print("=" * 70)
    print("  NOVA-VAD FAIR COMPARISON (same held-out test set as final_model_report.json)")
    print("=" * 70)

    test_speech, test_noise, test_idx, data = get_fair_test_split()
    n_test = len(test_speech) + len(test_noise)
    print(f"\nReconstructed held-out test set: {n_test} files "
          f"({len(test_speech)} speech, {len(test_noise)} noise)")

    print("\nRunning NOVA-VAD (real methodology: 1s windowed features, saved final model)...")
    nova_r = run_nova_vad_fair(test_speech, test_noise, data, test_idx)
    print(f"  Done — {nova_r['accuracy']}%")

    print("\nRunning WebRTC VAD...")
    webrtc_r = run_webrtc(test_speech, test_noise, SPEECH_DIR, NOISE_DIR)
    print(f"  Done — {webrtc_r['accuracy']}%")

    print("\nRunning Energy Threshold (naive baseline)...")
    energy_r = run_energy_threshold(test_speech, test_noise, SPEECH_DIR, NOISE_DIR)
    print(f"  Done — {energy_r['accuracy']}%")

    print("\nRunning Silero VAD...")
    silero_r = run_silero(test_speech, test_noise, SPEECH_DIR, NOISE_DIR)
    print(f"  Done — {silero_r['accuracy']}%")

    print("\nRunning Pyannote VAD...")
    pyannote_r = run_pyannote(test_speech, test_noise, SPEECH_DIR, NOISE_DIR)
    print(f"  Done — {pyannote_r['accuracy']}%")

    print("\nRunning SpeechBrain VAD...")
    speechbrain_r = run_speechbrain(test_speech, test_noise, SPEECH_DIR, NOISE_DIR)
    print(f"  Done — {speechbrain_r['accuracy']}%")

    print("\nRunning TEN-VAD...")
    ten_vad_r = run_ten_vad(test_speech, test_noise, SPEECH_DIR, NOISE_DIR)
    print(f"  Done — {ten_vad_r['accuracy']}%")

    all_results = [nova_r, webrtc_r, energy_r, silero_r, pyannote_r, speechbrain_r, ten_vad_r]
    print_and_diagnose(all_results, n_test)

    category_map = load_noise_category_manifest(NOISE_DIR)
    category_breakdowns = {}
    if category_map:
        nova_fp_files = {f["file"] for f in nova_r["false_positives"]}
        noise_results = [{"true": 0, "pred": 1 if f in nova_fp_files else 0, "file": f} for f in test_noise]
        category_breakdowns = build_category_breakdown(noise_results, category_map)
        print("\n  NOVA-VAD accuracy by noise category (fair held-out test set):")
        for cat, stats in sorted(category_breakdowns.items()):
            print(f"    {cat:<20} {stats['correct']}/{stats['total']} ({stats['accuracy']}%)")

    save_artifacts(all_results, n_test, test_speech, test_noise, category_breakdowns)


if __name__ == "__main__":
    main()
