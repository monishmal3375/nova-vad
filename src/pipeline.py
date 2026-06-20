import os
from src.denoiser import denoise_folder
from src.vad import detect_speech
from src.classifier import train_and_evaluate

def run_pipeline():
    print("=" * 50)
    print("   NOVA-VAD PIPELINE")
    print("=" * 50)

    RAW_SPEECH   = "data/speech"
    RAW_NOISE    = "data/noise"
    CLEAN_SPEECH = "data/clean_speech"
    CLEAN_NOISE  = "data/clean_noise"

    # step 1 — denoise
    print("\n[ STEP 1 ] Denoising audio files...\n")
    denoise_folder(RAW_SPEECH, CLEAN_SPEECH)
    denoise_folder(RAW_NOISE,  CLEAN_NOISE)

    # step 2 — WebRTC baseline
    print("\n[ STEP 2 ] WebRTC VAD Baseline...\n")
    results = []
    speech_files = sorted([f for f in os.listdir(CLEAN_SPEECH) if f.endswith(".wav")])
    noise_files  = sorted([f for f in os.listdir(CLEAN_NOISE)  if f.endswith(".wav")])

    for f in speech_files:
        r = detect_speech(os.path.join(CLEAN_SPEECH, f))
        r["true_label"] = 1
        r["correct"]    = r["prediction"] == 1
        results.append(r)
        if len(results) % 50 == 0:
            print(f"  WebRTC: {len(results)} files evaluated...")

    for f in noise_files:
        r = detect_speech(os.path.join(CLEAN_NOISE, f))
        r["true_label"] = 0
        r["correct"]    = r["prediction"] == 0
        results.append(r)
        if len(results) % 50 == 0:
            print(f"  WebRTC: {len(results)} files evaluated...")

    total           = len(results)
    correct         = sum(1 for r in results if r["correct"])
    webrtc_accuracy = round(correct / total * 100, 2)
    print(f"\n  WebRTC accuracy: {webrtc_accuracy}%")

    # step 3 — NOVA-VAD
    print("\n[ STEP 3 ] NOVA-VAD (150+ features + Ensemble)...\n")
    metrics = train_and_evaluate(CLEAN_SPEECH, CLEAN_NOISE)

    # final results
    print("\n" + "=" * 50)
    print("   FINAL RESULTS")
    print("=" * 50)
    print(f"  Total files:              {metrics['total']}")
    print(f"  WebRTC VAD (baseline):    {webrtc_accuracy}%")
    print(f"  NOVA-VAD (ours):          {metrics['accuracy']}%")
    print(f"  Improvement:              +{round(metrics['accuracy'] - webrtc_accuracy, 2)}%")
    print(f"\n  Precision:  {metrics['precision']}%")
    print(f"  Recall:     {metrics['recall']}%")
    print(f"  F1 Score:   {metrics['f1_score']}%")
    print(f"\n  TP: {metrics['tp']}  TN: {metrics['tn']}  FP: {metrics['fp']}  FN: {metrics['fn']}")
    print("=" * 50)

if __name__ == "__main__":
    run_pipeline()