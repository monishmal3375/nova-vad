"""
Scores all 7 systems on the DTMF hard-negative scenes. Ground truth is
"no speech anywhere" — the metric that matters is false-positive rate
(how often each system wrongly says "speech" on pure DTMF tones), not
accuracy/MCC in the usual sense (MCC/F1 are degenerate when one class
never occurs in the ground truth — reported anyway for completeness, but
false-positive rate is the honest, meaningful number here).

Run: python3 -m scripts.score_hardneg_dtmf
"""
import json

from scripts.compute_per_scene_results import run_all_systems

SCENE_DIR = "data/scenes/test_v2_hardneg"
OUT_PATH = "reports/per_scene_test_v2_hardneg.json"


def main():
    run_all_systems(SCENE_DIR, OUT_PATH)

    with open(OUT_PATH) as f:
        results = json.load(f)

    print(f"\n{'System':<25}{'False-positive rate (10ms frames)':<40}")
    summary = {}
    for name, scenes in results.items():
        total_frames = 0
        false_positive_frames = 0
        for s in scenes:
            pred = s["pred"]
            truth = s["truth"]  # all zeros by construction
            total_frames += len(truth)
            false_positive_frames += sum(1 for p, t in zip(pred, truth) if p == 1 and t == 0)
        fp_rate = false_positive_frames / total_frames * 100 if total_frames else 0.0
        summary[name] = {"false_positive_rate_pct": round(fp_rate, 2), "total_frames": total_frames}
        print(f"{name:<25}{fp_rate:<40.2f}")

    with open("reports/hardneg_dtmf_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to reports/hardneg_dtmf_summary.json")


if __name__ == "__main__":
    main()
