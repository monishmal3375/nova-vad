"""
Leave-one-noise-file-out (LONO) swing analysis: for each SNR condition and
each system, quantify how much a single noise file's scenes can move the
condition-level accuracy. This directly answers whether round 1's
per-condition breakdown (8 unique noise files/condition, 10 scenes/condition)
was statistically reliable, for ALL systems, not just NOVA-VAD.

For condition C with noise files {f1..fk} and full-condition accuracy A:
  swing(fi) = A - accuracy(C excluding all scenes using fi)
A large |swing(fi)| means that one file's scenes are pulling the aggregate
away from what the condition "really" looks like without it.

Run: python3 -m scripts.lono_analysis <per_scene_results.json>
"""
import json
import sys
from collections import defaultdict

import numpy as np


def condition_accuracy(scenes):
    correct = 0
    total = 0
    for s in scenes:
        pred = np.array(s["pred"])
        truth = np.array(s["truth"])
        n = min(len(pred), len(truth))
        correct += int(np.sum(pred[:n] == truth[:n]))
        total += n
    return correct / total * 100 if total else 0.0


def lono_for_system(scene_results):
    by_condition = defaultdict(list)
    for s in scene_results:
        by_condition[s["condition"]].append(s)

    report = {}
    for cond, scenes in by_condition.items():
        full_acc = condition_accuracy(scenes)
        noise_files = sorted(set(s["source_noise_file"] for s in scenes))
        swings = {}
        for f in noise_files:
            without_f = [s for s in scenes if s["source_noise_file"] != f]
            if not without_f:
                continue
            acc_without = condition_accuracy(without_f)
            swings[f] = round(full_acc - acc_without, 2)
        max_swing = max(swings.values(), key=abs) if swings else 0.0
        report[cond] = {
            "full_condition_accuracy": round(full_acc, 2),
            "n_unique_noise_files": len(noise_files),
            "n_scenes": len(scenes),
            "swings_by_file": swings,
            "max_abs_swing": round(max(abs(v) for v in swings.values()), 2) if swings else 0.0,
            "mean_abs_swing": round(np.mean([abs(v) for v in swings.values()]), 2) if swings else 0.0,
        }
    return report


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "reports/per_scene_test_original.json"
    with open(path) as f:
        all_results = json.load(f)

    full_report = {}
    print(f"{'System':<22}{'Condition':<12}{'#files':<8}{'FullAcc':<10}{'MaxSwing':<10}{'MeanSwing':<10}")
    print("-" * 72)
    for system_name, scene_results in all_results.items():
        lono = lono_for_system(scene_results)
        full_report[system_name] = lono
        for cond, d in lono.items():
            print(f"{system_name:<22}{cond:<12}{d['n_unique_noise_files']:<8}"
                  f"{d['full_condition_accuracy']:<10}{d['max_abs_swing']:<10}{d['mean_abs_swing']:<10}")

    out_path = path.replace(".json", "_lono.json")
    with open(out_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nFull LONO report saved to {out_path}")


if __name__ == "__main__":
    main()
