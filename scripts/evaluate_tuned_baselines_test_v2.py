"""
SINGLE evaluation of the val-tuned baseline configs on test_v2. Configs are
loaded from reports/baseline_tuning_val.json (frozen from val, Item 3) --
not refit here. First and only test_v2 contact for this decision.

Run: python3 -m scripts.evaluate_tuned_baselines_test_v2
"""
import json
import warnings

warnings.filterwarnings("ignore")

from scripts.frame_benchmark import load_test_scenes, run_system
from scripts.frame_vad_adapters import predict_mask_silero, predict_mask_pyannote, build_pyannote_pipeline

TEST_V2_DIR = "data/scenes/test_v2"


def load_test_v2_scenes():
    import glob
    import os
    scenes = []
    for json_path in sorted(glob.glob(os.path.join(TEST_V2_DIR, "*.json"))):
        with open(json_path) as f:
            meta = json.load(f)
        wav_path = json_path.replace(".json", ".wav")
        scenes.append((wav_path, meta))
    return scenes


def main():
    with open("reports/baseline_tuning_val.json") as f:
        tuning = json.load(f)

    scenes = load_test_v2_scenes()
    print(f"Loaded {len(scenes)} test_v2 scenes -- FIRST contact for tuned-baseline evaluation")

    results = {}

    # Silero, tuned threshold
    from silero_vad import load_silero_vad
    silero_model = load_silero_vad()
    silero_th = tuning["silero"]["best_threshold"]
    print(f"\nSilero fair threshold: {silero_th} (val F1={tuning['silero']['sweep'][str(silero_th)]*100:.2f}%, "
          f"default 0.5 val F1={tuning['silero']['default_f1']*100:.2f}%)")
    results["Silero VAD (fair, tuned)"] = run_system(
        "Silero VAD (fair, tuned)",
        lambda wav, meta: predict_mask_silero(wav, silero_model, meta["duration_ms"], threshold=silero_th),
        scenes,
    )

    # SpeechBrain -- default WAS the best in the val grid, so "tuned" == default here, reported as-is
    from speechbrain.inference.VAD import VAD
    sb_model = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="models/speechbrain_vad")
    act_th = tuning["speechbrain"]["best_activation_th"]
    deact_th = tuning["speechbrain"]["best_deactivation_th"]
    print(f"\nSpeechBrain fair thresholds: activation={act_th}, deactivation={deact_th} "
          f"(same as library default -- no improvement found in the val grid)")
    from scripts.frame_vad_adapters import predict_mask_speechbrain
    results["SpeechBrain VAD (fair, tuned)"] = run_system(
        "SpeechBrain VAD (fair, tuned)",
        lambda wav, meta: predict_mask_speechbrain(
            wav, sb_model, meta["duration_ms"], activation_th=act_th, deactivation_th=deact_th),
        scenes,
    )

    # Pyannote, tuned min_duration_on/off
    min_on = tuning["pyannote"]["best_min_duration_on"]
    min_off = tuning["pyannote"]["best_min_duration_off"]
    print(f"\nPyannote fair min_duration_on/off: {min_on}/{min_off} "
          f"(onset/offset not tunable for this model -- see reports/decision_v7.md)")
    pyannote_pipeline = build_pyannote_pipeline(min_duration_on=min_on, min_duration_off=min_off)
    results["Pyannote VAD (fair, tuned)"] = run_system(
        "Pyannote VAD (fair, tuned)",
        lambda wav, meta: predict_mask_pyannote(wav, pyannote_pipeline, meta["duration_ms"]),
        scenes,
    )

    with open("reports/tuned_baselines_test_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to reports/tuned_baselines_test_v2.json")

    print(f"\n{'System':<35}{'Acc%':<8}{'Prec%':<8}{'Rec%':<8}{'F1%':<8}{'MCC':<8}")
    for name, r in results.items():
        o = r["overall"]
        print(f"{name:<35}{o['accuracy']:<8}{o['precision']:<8}{o['recall']:<8}{o['f1']:<8}{o['mcc']:<8}")


if __name__ == "__main__":
    main()
