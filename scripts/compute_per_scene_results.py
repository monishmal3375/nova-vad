"""
Computes and caches per-scene (predicted mask, accuracy) for all 7 systems
on a given scene directory. Shared by the LONO swing analysis (Part 1) and
the error-profile comparison (Part 3) so each system only has to run once
per test-set version, not once per analysis.

Run: python3 -m scripts.compute_per_scene_results <scene_dir> <output_json>
Example: python3 -m scripts.compute_per_scene_results data/scenes/test reports/per_scene_test_original.json
"""
import glob
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np

from scripts.frame_vad_adapters import (
    predict_mask_nova, predict_mask_webrtc, predict_mask_silero,
    predict_mask_pyannote, predict_mask_speechbrain,
)
from scripts.frame_vad_v1 import predict_mask_frame_v1, load_tuned_params as load_v1_params
from scripts.frame_vad_v2 import predict_mask_frame_v2, load_tuned_params as load_v2_params


def load_scenes(scene_dir):
    scenes = []
    for json_path in sorted(glob.glob(os.path.join(scene_dir, "*.json"))):
        with open(json_path) as f:
            meta = json.load(f)
        wav_path = json_path.replace(".json", ".wav")
        scenes.append((wav_path, meta))
    return scenes


def scene_accuracy(pred, truth):
    n = min(len(pred), len(truth))
    pred, truth = pred[:n], truth[:n]
    return float(np.mean(np.array(pred) == np.array(truth))) * 100, pred, truth


def run_all_systems(scene_dir, output_path):
    scenes = load_scenes(scene_dir)
    print(f"Loaded {len(scenes)} scenes from {scene_dir}")

    systems = {}

    rf0 = joblib.load("models/nova_vad_rf.pkl")
    gbt0 = joblib.load("models/nova_vad_gbt.pkl")
    scaler0 = joblib.load("models/nova_vad_scaler.pkl")
    systems["NOVA-VAD"] = lambda wav, meta: predict_mask_nova(wav, rf0, gbt0, scaler0, meta["duration_ms"])

    systems["WebRTC VAD"] = lambda wav, meta: predict_mask_webrtc(wav, meta["duration_ms"])

    rf1 = joblib.load("models/registry/nova-vad-frame-v1/frame_vad_v1_rf.pkl")
    gbt1 = joblib.load("models/registry/nova-vad-frame-v1/frame_vad_v1_gbt.pkl")
    scaler1 = joblib.load("models/registry/nova-vad-frame-v1/frame_vad_v1_scaler.pkl")
    params1 = load_v1_params()
    systems["NOVA-VAD-frame-v1"] = lambda wav, meta: predict_mask_frame_v1(
        wav, rf1, gbt1, scaler1, meta["duration_ms"], params1)

    rf2 = joblib.load("models/registry/nova-vad-frame-v2/frame_vad_v2_rf.pkl")
    gbt2 = joblib.load("models/registry/nova-vad-frame-v2/frame_vad_v2_gbt.pkl")
    scaler2 = joblib.load("models/registry/nova-vad-frame-v2/frame_vad_v2_scaler.pkl")
    params2 = load_v2_params()
    systems["NOVA-VAD-frame-v2"] = lambda wav, meta: predict_mask_frame_v2(
        wav, rf2, gbt2, scaler2, meta["duration_ms"], params2)

    from silero_vad import load_silero_vad
    silero_model = load_silero_vad()
    systems["Silero VAD"] = lambda wav, meta: predict_mask_silero(wav, silero_model, meta["duration_ms"])

    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection
    token = os.environ.get("HF_TOKEN")
    pmodel = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=token)
    pyannote_pipeline = VoiceActivityDetection(segmentation=pmodel)
    pyannote_pipeline.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})
    systems["Pyannote VAD"] = lambda wav, meta: predict_mask_pyannote(wav, pyannote_pipeline, meta["duration_ms"])

    from speechbrain.inference.VAD import VAD
    sb_model = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="models/speechbrain_vad")
    systems["SpeechBrain VAD"] = lambda wav, meta: predict_mask_speechbrain(wav, sb_model, meta["duration_ms"])

    results = {}
    for name, predict_fn in systems.items():
        print(f"\n  Running {name} on {len(scenes)} scenes...")
        scene_results = []
        for wav_path, meta in scenes:
            pred = predict_fn(wav_path, meta)
            acc, pred_trim, truth_trim = scene_accuracy(pred, meta["frame_labels_10ms"])
            scene_results.append({
                "scene_id": meta["scene_id"],
                "condition": meta["condition"],
                "source_noise_file": meta["source_noise_file"],
                "accuracy": round(acc, 2),
                "pred": pred_trim,
                "truth": truth_trim,
            })
        results[name] = scene_results
        mean_acc = np.mean([s["accuracy"] for s in scene_results])
        print(f"    {name}: mean per-scene accuracy = {mean_acc:.2f}%")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f)
    print(f"\nSaved per-scene results to {output_path}")


if __name__ == "__main__":
    scene_dir = sys.argv[1] if len(sys.argv) > 1 else "data/scenes/test"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "reports/per_scene_results.json"
    run_all_systems(scene_dir, output_path)
