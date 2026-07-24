"""
SINGLE evaluation of the val-fit logistic ensemble on test_v2. Weights are
loaded from reports/ensemble_frame_probs_val_combination_results.json
(frozen from val, Part 1) — NOT refit here. This is the first and only
contact test_v2 has had with this specific ensemble decision.

Also reports NOVA-VAD-v2 alone (raw prob, no hysteresis) on the SAME
test_v2 frames, for a direct apples-to-apples before/after comparison
(matching the val-side comparison methodology exactly).

Run: python3 -m scripts.ensemble_evaluate_test_v2
"""
import json
from collections import defaultdict

import numpy as np

from scripts.ensemble_fit_and_eval import confusion_metrics
from scripts.frame_benchmark import cluster_bootstrap_ci, per_scene_accuracy


def main():
    with open("reports/ensemble_frame_probs_val_combination_results.json") as f:
        fit = json.load(f)
    w = fit["logistic_weights"]
    print(f"Using FROZEN logistic weights from val (not refit): {w}")

    with open("reports/ensemble_frame_probs_test_v2.json") as f:
        scenes = json.load(f)
    print(f"Loaded {len(scenes)} test_v2 scenes — FIRST evaluation of this ensemble on test_v2")

    overall_nova, overall_ens = defaultdict(list), defaultdict(list)
    by_cond_nova, by_cond_ens = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    scene_acc_nova, scene_acc_ens = [], []

    for s in scenes:
        nova = np.array(s["nova_v2_prob"])
        silero = np.array(s["silero_prob"])
        pyannote = np.array(s["pyannote_mask"])
        truth = np.array(s["truth"])
        cond = s["condition"]

        nova_pred = (nova >= 0.5).astype(int)

        logit = w["intercept"] + w["nova"] * nova + w["silero"] * silero + w["pyannote"] * pyannote
        ens_prob = 1 / (1 + np.exp(-logit))
        ens_pred = (ens_prob >= 0.5).astype(int)

        overall_nova["pred"].extend(nova_pred.tolist())
        overall_nova["truth"].extend(truth.tolist())
        overall_ens["pred"].extend(ens_pred.tolist())
        overall_ens["truth"].extend(truth.tolist())

        by_cond_nova[cond]["pred"].extend(nova_pred.tolist())
        by_cond_nova[cond]["truth"].extend(truth.tolist())
        by_cond_ens[cond]["pred"].extend(ens_pred.tolist())
        by_cond_ens[cond]["truth"].extend(truth.tolist())

        scene_acc_nova.append(per_scene_accuracy(nova_pred.tolist(), truth.tolist()))
        scene_acc_ens.append(per_scene_accuracy(ens_pred.tolist(), truth.tolist()))

    nova_metrics = confusion_metrics(overall_nova["pred"], overall_nova["truth"])
    ens_metrics = confusion_metrics(overall_ens["pred"], overall_ens["truth"])
    nova_ci = cluster_bootstrap_ci(scene_acc_nova)
    ens_ci = cluster_bootstrap_ci(scene_acc_ens)

    print(f"\n{'System':<45}{'Acc%':<8}{'95%CI':<16}{'Prec%':<8}{'Rec%':<8}{'F1%':<8}{'MCC':<8}")
    print(f"{'NOVA-VAD-v2 alone (raw, no hysteresis)':<45}{nova_metrics['accuracy']:<8}"
          f"{str(list(nova_ci)):<16}{nova_metrics['precision']:<8}{nova_metrics['recall']:<8}"
          f"{nova_metrics['f1']:<8}{nova_metrics['mcc']:<8}")
    print(f"{'Logistic ensemble (NOVA+Silero+Pyannote)':<45}{ens_metrics['accuracy']:<8}"
          f"{str(list(ens_ci)):<16}{ens_metrics['precision']:<8}{ens_metrics['recall']:<8}"
          f"{ens_metrics['f1']:<8}{ens_metrics['mcc']:<8}")
    print(f"\nMCC delta (ensemble - NOVA alone): {ens_metrics['mcc'] - nova_metrics['mcc']:+.4f}")

    print(f"\nPer-condition accuracy:")
    conditions = list(by_cond_nova.keys())
    print(f"{'System':<45}" + "".join(f"{c:<12}" for c in conditions))
    nova_row = [confusion_metrics(by_cond_nova[c]["pred"], by_cond_nova[c]["truth"])["accuracy"] for c in conditions]
    ens_row = [confusion_metrics(by_cond_ens[c]["pred"], by_cond_ens[c]["truth"])["accuracy"] for c in conditions]
    print(f"{'NOVA-VAD-v2 alone':<45}" + "".join(f"{v:<12}" for v in nova_row))
    print(f"{'Logistic ensemble':<45}" + "".join(f"{v:<12}" for v in ens_row))

    out = {
        "nova_alone": {"overall": nova_metrics, "ci95": list(nova_ci),
                       "by_condition": {c: confusion_metrics(by_cond_nova[c]["pred"], by_cond_nova[c]["truth"]) for c in conditions}},
        "logistic_ensemble": {"overall": ens_metrics, "ci95": list(ens_ci),
                               "by_condition": {c: confusion_metrics(by_cond_ens[c]["pred"], by_cond_ens[c]["truth"]) for c in conditions}},
        "weights_used": w,
    }
    with open("reports/ensemble_test_v2_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to reports/ensemble_test_v2_result.json")


if __name__ == "__main__":
    main()
