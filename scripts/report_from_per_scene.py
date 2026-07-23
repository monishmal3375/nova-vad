"""
Builds the full official metrics table (accuracy/precision/recall/F1/MCC/CI,
per-condition breakdown) from a cached per-scene results file, using the
EXACT SAME scoring functions as scripts/frame_benchmark.py (imported, not
reimplemented) — so this is not a metric redefinition, just re-deriving the
official numbers from already-computed per-scene predictions instead of
re-running inference.

Run: python3 -m scripts.report_from_per_scene <per_scene_results.json> <label>
"""
import json
import sys
from collections import defaultdict

from scripts.frame_benchmark import confusion, metrics_from_confusion, cluster_bootstrap_ci, per_scene_accuracy


def build_report(per_scene_path):
    with open(per_scene_path) as f:
        all_results = json.load(f)

    report = {}
    for system_name, scene_results in all_results.items():
        all_pred, all_truth = [], []
        scene_accuracies = []
        by_condition = defaultdict(lambda: {"pred": [], "truth": []})

        for s in scene_results:
            pred, truth = s["pred"], s["truth"]
            all_pred.extend(pred)
            all_truth.extend(truth)
            scene_accuracies.append(per_scene_accuracy(pred, truth))
            by_condition[s["condition"]]["pred"].extend(pred)
            by_condition[s["condition"]]["truth"].extend(truth)

        overall = metrics_from_confusion(*confusion(all_pred, all_truth))
        ci_lo, ci_hi = cluster_bootstrap_ci(scene_accuracies)
        overall["accuracy_ci95"] = [ci_lo, ci_hi]
        overall["ci_width"] = round(ci_hi - ci_lo, 2)

        by_cond_metrics = {}
        for cond, d in by_condition.items():
            by_cond_metrics[cond] = metrics_from_confusion(*confusion(d["pred"], d["truth"]))

        report[system_name] = {"overall": overall, "by_condition": by_cond_metrics}

    return report


def print_report(report, label):
    print(f"\n=== {label} ===")
    print(f"{'Model':<22}{'Acc%':<8}{'95% CI':<18}{'CIwidth':<9}{'Prec%':<8}{'Rec%':<8}{'F1%':<8}{'MCC':<8}")
    for name, r in report.items():
        o = r["overall"]
        ci = f"[{o['accuracy_ci95'][0]},{o['accuracy_ci95'][1]}]"
        print(f"{name:<22}{o['accuracy']:<8}{ci:<18}{o['ci_width']:<9}{o['precision']:<8}"
              f"{o['recall']:<8}{o['f1']:<8}{o['mcc']:<8}")

    conditions = list(next(iter(report.values()))["by_condition"].keys())
    print(f"\nPer-condition accuracy %:")
    print(f"{'Model':<22}" + "".join(f"{c:<12}" for c in conditions))
    for name, r in report.items():
        row = "".join(f"{r['by_condition'][c]['accuracy']:<12}" for c in conditions)
        print(f"{name:<22}{row}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "reports/per_scene_test_v2.json"
    label = sys.argv[2] if len(sys.argv) > 2 else path
    report = build_report(path)
    print_report(report, label)
    out_path = path.replace(".json", "_full_metrics.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to {out_path}")
