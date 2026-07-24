"""
Fair threshold tuning for Silero, Pyannote, and SpeechBrain (Plan Section
7.4) -- same principle as the WebRTC mode fix: each baseline evaluated at
its own fairly-tuned operating point, not just its library default.
Tuned on `val` ONLY, never `test_v2`. Selection objective: frame F1, same
as every other threshold-tuning decision in this project (NOVA-VAD's
hysteresis, WebRTC's mode selection) -- not switched to a different metric
for these baselines.

Silero: `threshold` param on get_speech_timestamps(), directly tunable.
SpeechBrain: activation_th/deactivation_th, tunable -- and the library
  exposes get_speech_prob_file() (raw probabilities, no thresholding) +
  apply_threshold()/get_boundaries() (cheap post-hoc thresholding), so the
  expensive neural inference runs ONCE per scene, then many threshold
  combos are swept for free.
Pyannote: onset/offset are NOT tunable for pyannote/segmentation-3.0 (a
  powerset model) -- confirmed by direct experimentation, see
  scripts/frame_vad_adapters.py's build_pyannote_pipeline() docstring.
  Only min_duration_on/min_duration_off are genuinely tunable; each combo
  requires a fresh pipeline instantiation + full inference (no cheap
  post-hoc path exists for this pipeline), so the grid here is smaller.

Run: python3 -m scripts.tune_baselines_val
"""
import glob
import json
import math
import os
import warnings

warnings.filterwarnings("ignore")

from scripts.frame_vad_adapters import (
    predict_mask_silero, predict_mask_pyannote, build_pyannote_pipeline,
    _mask_from_intervals_ms,
)

VAL_DIR = "data/scenes/val"
OUT_PATH = "reports/baseline_tuning_val.json"


def load_val_scenes():
    scenes = []
    for json_path in sorted(glob.glob(os.path.join(VAL_DIR, "*.json"))):
        with open(json_path) as f:
            meta = json.load(f)
        wav_path = json_path.replace(".json", ".wav")
        scenes.append((wav_path, meta))
    return scenes


def frame_f1(pred, truth):
    n = min(len(pred), len(truth))
    pred, truth = pred[:n], truth[:n]
    tp = sum(1 for p, t in zip(pred, truth) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(pred, truth) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(pred, truth) if p == 0 and t == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def score_all(scenes, predict_fn):
    all_pred, all_truth = [], []
    for wav_path, meta in scenes:
        pred = predict_fn(wav_path, meta)
        all_pred.extend(pred)
        all_truth.extend(meta["frame_labels_10ms"])
    return frame_f1(all_pred, all_truth)


def tune_silero(scenes):
    from silero_vad import load_silero_vad
    model = load_silero_vad()
    print("\n=== Silero threshold sweep ===")
    results = {}
    for th in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        f1 = score_all(scenes, lambda wav, meta: predict_mask_silero(
            wav, model, meta["duration_ms"], threshold=th))
        results[th] = f1
        print(f"  threshold={th}: val F1={f1*100:.2f}%")
    best_th = max(results, key=results.get)
    print(f"  BEST: threshold={best_th}, F1={results[best_th]*100:.2f}% "
          f"(default 0.5 gave F1={results[0.5]*100:.2f}%)")
    return {"best_threshold": best_th, "sweep": results, "default_f1": results[0.5]}


def tune_speechbrain(scenes):
    from speechbrain.inference.VAD import VAD
    model = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="models/speechbrain_vad")
    print("\n=== SpeechBrain threshold sweep (probabilities cached, thresholds swept cheaply) ===")

    tmp_path = "data/tmp_sb_tune.wav"
    import librosa
    probs_by_scene = []
    for i, (wav_path, meta) in enumerate(scenes):
        audio, sr = librosa.load(wav_path, sr=16000, mono=True)
        import soundfile as sf
        sf.write(tmp_path, audio, sr)
        prob = model.get_speech_prob_file(tmp_path)
        probs_by_scene.append((prob, meta))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(scenes)} scenes' probabilities computed")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    results = {}
    for act_th in [0.3, 0.5, 0.7]:
        for deact_th in [0.15, 0.25, 0.35]:
            if deact_th >= act_th:
                continue
            all_pred, all_truth = [], []
            for prob, meta in probs_by_scene:
                prob_th = model.apply_threshold(prob, activation_th=act_th, deactivation_th=deact_th)
                boundaries = model.get_boundaries(prob_th, output_value="seconds")
                intervals_ms = [(float(b[0]) * 1000, float(b[1]) * 1000) for b in boundaries]
                mask = _mask_from_intervals_ms(meta["duration_ms"], intervals_ms)
                all_pred.extend(mask)
                all_truth.extend(meta["frame_labels_10ms"])
            f1 = frame_f1(all_pred, all_truth)
            results[f"{act_th},{deact_th}"] = f1
            print(f"  activation_th={act_th}, deactivation_th={deact_th}: val F1={f1*100:.2f}%")

    best_key = max(results, key=results.get)
    best_act, best_deact = [float(x) for x in best_key.split(",")]
    default_key = "0.5,0.25"
    print(f"  BEST: activation_th={best_act}, deactivation_th={best_deact}, F1={results[best_key]*100:.2f}% "
          f"(default 0.5/0.25 gave F1={results.get(default_key, float('nan'))*100:.2f}%)")
    return {"best_activation_th": best_act, "best_deactivation_th": best_deact,
            "sweep": results, "default_f1": results.get(default_key)}


def tune_pyannote(scenes):
    print("\n=== Pyannote min_duration_on/off sweep (onset/offset NOT tunable -- see docstring) ===")
    results = {}
    combos = [(0.0, 0.0), (0.1, 0.1), (0.0, 0.25), (0.25, 0.0)]
    for min_on, min_off in combos:
        pipeline = build_pyannote_pipeline(min_duration_on=min_on, min_duration_off=min_off)
        f1 = score_all(scenes, lambda wav, meta: predict_mask_pyannote(wav, pipeline, meta["duration_ms"]))
        results[f"{min_on},{min_off}"] = f1
        print(f"  min_duration_on={min_on}, min_duration_off={min_off}: val F1={f1*100:.2f}%")
    best_key = max(results, key=results.get)
    best_on, best_off = [float(x) for x in best_key.split(",")]
    print(f"  BEST: min_duration_on={best_on}, min_duration_off={best_off}, F1={results[best_key]*100:.2f}% "
          f"(default 0.0/0.0 gave F1={results['0.0,0.0']*100:.2f}%)")
    return {"best_min_duration_on": best_on, "best_min_duration_off": best_off,
            "sweep": results, "default_f1": results["0.0,0.0"],
            "note": "onset/offset are hardcoded to 0.5 for this powerset model and are NOT "
                    "exposed as tunable pipeline parameters -- confirmed via direct experimentation "
                    "(instantiate({'onset': ...}) raises ValueError). Only min_duration_on/off tuned."}


def main():
    scenes = load_val_scenes()
    print(f"Loaded {len(scenes)} val scenes")

    results = {
        "silero": tune_silero(scenes),
        "speechbrain": tune_speechbrain(scenes),
        "pyannote": tune_pyannote(scenes),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
