"""
Evaluates NOVA-VAD-frame-v1 on the LOCKED test scenes (never touched during
training or threshold tuning), using the exact same scoring code
(scripts/frame_benchmark.py's confusion/metrics/bootstrap functions) that
produced v0's -0.28 MCC result, so the two are directly comparable. Merges
the result into reports/frame_level_benchmark_v1.json /.md alongside the
five systems already there.

Run: python3 -m scripts.evaluate_frame_vad_v1
"""
import json
import os

import joblib

from scripts.frame_benchmark import load_test_scenes, run_system, write_report
from scripts.frame_vad_v1 import predict_mask_frame_v1, load_tuned_params

REPORT_JSON = "reports/frame_level_benchmark_v1.json"
REPORT_MD = "reports/frame_level_benchmark_v1.md"


def main():
    rf = joblib.load("models/frame_vad_v1_rf.pkl")
    gbt = joblib.load("models/frame_vad_v1_gbt.pkl")
    scaler = joblib.load("models/frame_vad_v1_scaler.pkl")
    params = load_tuned_params()
    print(f"Using tuned hysteresis params (from dev scenes): {params}")

    scenes = load_test_scenes()
    print(f"Loaded {len(scenes)} locked test scenes from data/scenes/test")

    result = run_system(
        "NOVA-VAD-frame-v1",
        lambda wav, meta: predict_mask_frame_v1(wav, rf, gbt, scaler, meta["duration_ms"], params),
        scenes,
    )

    with open(REPORT_JSON) as f:
        all_results = json.load(f)

    # keep NOVA-VAD (v0) as-is for comparison; add v1 alongside it
    ordered = {}
    for name, r in all_results.items():
        ordered[name] = r
        if name == "NOVA-VAD":
            ordered["NOVA-VAD-frame-v1"] = result
    if "NOVA-VAD-frame-v1" not in ordered:
        ordered["NOVA-VAD-frame-v1"] = result

    with open(REPORT_JSON, "w") as f:
        json.dump(ordered, f, indent=2)

    write_report(ordered, len(scenes))
    print(f"\nUpdated {REPORT_JSON} and {REPORT_MD} with NOVA-VAD-frame-v1")


if __name__ == "__main__":
    main()
