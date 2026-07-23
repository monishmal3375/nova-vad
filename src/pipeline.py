import os
from src.denoiser import denoise_folder
from src.vad import detect_speech
from src.classifier import train_and_evaluate

def run_pipeline(denoise: bool = False):
    """
    denoise: if True, run the noisereduce preprocessing step first and train
    on the denoised copies (data/clean_speech, data/clean_noise). Defaults to
    False — raw audio is the primary path, since the denoiser uses the first
    0.5s of every clip as its noise profile, which can remove real speech
    energy from clips that start talking immediately (see src/denoiser.py).
    Denoising is available as an explicit, opt-in experiment, not a default.
    """
    print("=" * 50)
    print("   NOVA-VAD PIPELINE")
    print("=" * 50)

    RAW_SPEECH   = "data/speech"
    RAW_NOISE    = "data/noise"
    CLEAN_SPEECH = "data/clean_speech"
    CLEAN_NOISE  = "data/clean_noise"

    if denoise:
        print("\n[ STEP 1 ] Denoising audio files (opt-in)...\n")
        denoise_folder(RAW_SPEECH, CLEAN_SPEECH)
        denoise_folder(RAW_NOISE,  CLEAN_NOISE)
        SPEECH_DIR, NOISE_DIR = CLEAN_SPEECH, CLEAN_NOISE
    else:
        print("\n[ STEP 1 ] Using raw audio (denoising skipped — pass denoise=True to opt in)\n")
        SPEECH_DIR, NOISE_DIR = RAW_SPEECH, RAW_NOISE

    # step 2 — WebRTC baseline
    print("\n[ STEP 2 ] WebRTC VAD Baseline...\n")
    results = []
    speech_files = sorted([f for f in os.listdir(SPEECH_DIR) if f.endswith(".wav")])
    noise_files  = sorted([f for f in os.listdir(NOISE_DIR)  if f.endswith(".wav")])

    for f in speech_files:
        r = detect_speech(os.path.join(SPEECH_DIR, f))
        r["true_label"] = 1
        r["correct"]    = r["prediction"] == 1
        results.append(r)
        if len(results) % 50 == 0:
            print(f"  WebRTC: {len(results)} files evaluated...")

    for f in noise_files:
        r = detect_speech(os.path.join(NOISE_DIR, f))
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
    print("\n[ STEP 3 ] NOVA-VAD (106 features + Ensemble)...\n")
    metrics = train_and_evaluate(SPEECH_DIR, NOISE_DIR)

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--denoise", action="store_true",
                         help="Opt in to noisereduce preprocessing (off by default; see run_pipeline docstring)")
    args = parser.parse_args()
    run_pipeline(denoise=args.denoise)