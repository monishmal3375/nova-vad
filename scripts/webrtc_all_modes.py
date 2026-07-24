"""
Runs WebRTC VAD at all 4 supported aggressiveness modes (0-3) on test_v2,
per plan Section 7.4: "Run WebRTC in all supported aggressiveness modes and
report the selected mode plus all modes." Previously only mode 3 (most
aggressive) had ever been tested in this project.

Uses the same scoring code as every other system (scripts/frame_benchmark.py),
imported not reimplemented.

Run: python3 -m scripts.webrtc_all_modes
"""
import json

from scripts.frame_benchmark import load_test_scenes, run_system
from scripts.frame_vad_adapters import predict_mask_webrtc

TEST_DIR = "data/scenes/test_v2"


def main():
    import scripts.frame_benchmark as fb
    fb.TEST_DIR = TEST_DIR
    scenes = load_test_scenes()
    print(f"Loaded {len(scenes)} scenes from {TEST_DIR}")

    results = {}
    for mode in [0, 1, 2, 3]:
        name = f"WebRTC VAD (aggressiveness={mode})"
        results[name] = run_system(
            name,
            lambda wav, meta, m=mode: predict_mask_webrtc(wav, meta["duration_ms"], aggressiveness=m),
            scenes,
        )

    best_mode = max(results.items(), key=lambda kv: kv[1]["overall"]["mcc"])
    print(f"\nBest mode by MCC: {best_mode[0]} (MCC={best_mode[1]['overall']['mcc']})")

    with open("reports/webrtc_all_modes_test_v2.json", "w") as f:
        json.dump({"results": results, "best_mode": best_mode[0]}, f, indent=2)
    print("Saved to reports/webrtc_all_modes_test_v2.json")


if __name__ == "__main__":
    main()
