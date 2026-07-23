"""
SINGLE final evaluation of NOVA-VAD-frame-v2 on the locked test scenes.
This script must only be run once model+threshold are frozen from val-only
tuning — running it repeatedly and reporting the best result would violate
the no-cherry-picking constraint. See reports/decision_v3.md for the run
log (this was run exactly once; if it's ever rerun for a genuinely new
model iteration, both results get reported, not just the better one).

Run: python3 -m scripts.evaluate_frame_vad_v2
"""
import json

import joblib

from scripts.frame_benchmark import load_test_scenes, run_system, write_report
from scripts.frame_vad_v2 import predict_mask_frame_v2, load_tuned_params

REPORT_JSON = "reports/frame_level_benchmark_v1.json"
REPORT_MD = "reports/frame_level_benchmark_v1.md"


def main():
    rf = joblib.load("models/frame_vad_v2_rf.pkl")
    gbt = joblib.load("models/frame_vad_v2_gbt.pkl")
    scaler = joblib.load("models/frame_vad_v2_scaler.pkl")
    params = load_tuned_params()
    print(f"Using tuned params (from VAL scenes only, never test): {params}")

    scenes = load_test_scenes()
    print(f"Loaded {len(scenes)} locked test scenes from data/scenes/test "
          f"(FIRST AND ONLY contact with test data for v2)")

    result = run_system(
        "NOVA-VAD-frame-v2",
        lambda wav, meta: predict_mask_frame_v2(wav, rf, gbt, scaler, meta["duration_ms"], params),
        scenes,
    )

    with open(REPORT_JSON) as f:
        all_results = json.load(f)

    ordered = {}
    for name, r in all_results.items():
        ordered[name] = r
        if name == "NOVA-VAD-frame-v1":
            ordered["NOVA-VAD-frame-v2"] = result
    if "NOVA-VAD-frame-v2" not in ordered:
        ordered["NOVA-VAD-frame-v2"] = result

    with open(REPORT_JSON, "w") as f:
        json.dump(ordered, f, indent=2)

    write_report(ordered, len(scenes))
    print(f"\nUpdated {REPORT_JSON} and {REPORT_MD} with NOVA-VAD-frame-v2")


if __name__ == "__main__":
    main()
