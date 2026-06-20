import os
from src.vad import detect_speech

def evaluate(speech_dir: str, noise_dir: str) -> dict:
    """
    Runs VAD on all files in speech_dir and noise_dir.
    Compares predictions against true labels.
    Returns accuracy metrics.
    """
    results = []

    # speech files → true label is 1
    print("Evaluating speech files...")
    for filename in sorted(os.listdir(speech_dir)):
        if filename.endswith(".wav"):
            path = os.path.join(speech_dir, filename)
            result = detect_speech(path)
            result["true_label"] = 1
            result["correct"] = result["prediction"] == 1
            results.append(result)
            status = "✓" if result["correct"] else "✗"
            print(f"  {status} {filename} → {result['label']} ({result['speech_ratio']})")

    # noise files → true label is 0
    print("\nEvaluating noise files...")
    for filename in sorted(os.listdir(noise_dir)):
        if filename.endswith(".wav"):
            path = os.path.join(noise_dir, filename)
            result = detect_speech(path)
            result["true_label"] = 0
            result["correct"] = result["prediction"] == 0
            results.append(result)
            status = "✓" if result["correct"] else "✗"
            print(f"  {status} {filename} → {result['label']} ({result['speech_ratio']})")

    # calculate metrics
    total     = len(results)
    correct   = sum(1 for r in results if r["correct"])
    accuracy  = correct / total * 100

    # true positives, false positives etc
    tp = sum(1 for r in results if r["true_label"] == 1 and r["prediction"] == 1)
    tn = sum(1 for r in results if r["true_label"] == 0 and r["prediction"] == 0)
    fp = sum(1 for r in results if r["true_label"] == 0 and r["prediction"] == 1)
    fn = sum(1 for r in results if r["true_label"] == 1 and r["prediction"] == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "total":     total,
        "correct":   correct,
        "accuracy":  round(accuracy, 2),
        "precision": round(precision * 100, 2),
        "recall":    round(recall * 100, 2),
        "f1_score":  round(f1 * 100, 2),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }

    return metrics, results