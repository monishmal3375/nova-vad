"""
Fits and evaluates candidate NOVA-VAD-v2 + Silero + Pyannote combination
rules on the VAL split (never test/test_v2), starting with the simplest
things first per the round-3 instruction:

  1. Simple average of NOVA-VAD-v2 and Silero probabilities, threshold 0.5.
  2. Majority vote of the three systems' binary decisions (>=2 of 3 say speech).
  3. Logistic regression on [nova_prob, silero_prob, pyannote_mask] -> label,
     fit on val, evaluated on val (in-sample fit — appropriate here since the
     "model" being tested is a 3-weight logistic regression, not something
     with enough capacity to memorize 40 scenes' worth of frames in a way
     that would invalidate the comparison; still disclosed explicitly).

Baseline for comparison: NOVA-VAD-v2 alone, on the SAME val frames (not
test_v2's numbers — apples to apples for this decision).

Run: python3 -m scripts.ensemble_fit_and_eval reports/ensemble_frame_probs_val.json
"""
import json
import math
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression


def confusion_metrics(pred, truth):
    pred = np.array(pred)
    truth = np.array(truth)
    tp = int(np.sum((pred == 1) & (truth == 1)))
    tn = int(np.sum((pred == 0) & (truth == 0)))
    fp = int(np.sum((pred == 1) & (truth == 0)))
    fn = int(np.sum((pred == 0) & (truth == 1)))
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0
    return {
        "accuracy": round(accuracy * 100, 2), "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2), "f1": round(f1 * 100, 2), "mcc": round(mcc, 4),
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "reports/ensemble_frame_probs_val.json"
    with open(path) as f:
        scenes = json.load(f)

    nova_all, silero_all, pyannote_all, truth_all = [], [], [], []
    for s in scenes:
        nova_all.extend(s["nova_v2_prob"])
        silero_all.extend(s["silero_prob"])
        pyannote_all.extend(s["pyannote_mask"])
        truth_all.extend(s["truth"])

    nova = np.array(nova_all)
    silero = np.array(silero_all)
    pyannote = np.array(pyannote_all)
    truth = np.array(truth_all)
    print(f"Total val frames: {len(truth)}")

    results = {}

    # Baseline: NOVA-VAD-v2 alone (using its ALREADY-TUNED hysteresis threshold's
    # underlying raw probability at 0.5 as the simplest apples-to-apples baseline —
    # NOT re-running the full hysteresis post-processing here, just the raw
    # per-frame probability vs 0.5, since the combinations below are also raw
    # per-frame, no hysteresis applied to any of them, for a fair comparison)
    nova_pred = (nova >= 0.5).astype(int)
    results["NOVA-VAD-v2 alone (raw prob >= 0.5, no hysteresis)"] = confusion_metrics(nova_pred, truth)

    # Candidate 1: simple average of NOVA + Silero probabilities
    avg_prob = (nova + silero) / 2
    avg_pred = (avg_prob >= 0.5).astype(int)
    results["Average(NOVA-v2, Silero) >= 0.5"] = confusion_metrics(avg_pred, truth)

    # Candidate 2: majority vote of 3 binary decisions
    nova_bin = (nova >= 0.5).astype(int)
    silero_bin = (silero >= 0.5).astype(int)
    vote_sum = nova_bin + silero_bin + pyannote
    vote_pred = (vote_sum >= 2).astype(int)
    results["Majority vote (NOVA-v2, Silero, Pyannote), >=2 of 3"] = confusion_metrics(vote_pred, truth)

    # Candidate 3: OR combination (favors recall)
    or_pred = ((nova_bin + silero_bin + pyannote) >= 1).astype(int)
    results["OR (any of 3 says speech)"] = confusion_metrics(or_pred, truth)

    # Candidate 4: logistic regression on [nova_prob, silero_prob, pyannote_mask]
    X = np.column_stack([nova, silero, pyannote])
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, truth)
    logit_pred = clf.predict(X)
    results["Logistic regression [nova_prob, silero_prob, pyannote_mask]"] = confusion_metrics(logit_pred, truth)
    print(f"\nLogistic regression weights: nova={clf.coef_[0][0]:.3f}, "
          f"silero={clf.coef_[0][1]:.3f}, pyannote={clf.coef_[0][2]:.3f}, "
          f"intercept={clf.intercept_[0]:.3f}")

    print(f"\n{'Rule':<62}{'Acc%':<8}{'Prec%':<8}{'Rec%':<8}{'F1%':<8}{'MCC':<8}")
    for name, m in results.items():
        print(f"{name:<62}{m['accuracy']:<8}{m['precision']:<8}{m['recall']:<8}{m['f1']:<8}{m['mcc']:<8}")

    baseline_mcc = results["NOVA-VAD-v2 alone (raw prob >= 0.5, no hysteresis)"]["mcc"]
    print(f"\nBaseline (NOVA-VAD-v2 alone) MCC: {baseline_mcc}")
    for name, m in results.items():
        if name.startswith("NOVA-VAD-v2 alone"):
            continue
        delta = m["mcc"] - baseline_mcc
        print(f"  {name}: MCC delta vs baseline = {delta:+.4f}")

    out_path = path.replace(".json", "_combination_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "results": results,
            "logistic_weights": {
                "nova": float(clf.coef_[0][0]), "silero": float(clf.coef_[0][1]),
                "pyannote": float(clf.coef_[0][2]), "intercept": float(clf.intercept_[0]),
            },
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
