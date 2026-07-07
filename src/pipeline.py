import os
import subprocess
import sys


def _count_wavs(path):
    if not os.path.isdir(path):
        return 0
    return len([f for f in os.listdir(path) if f.endswith(".wav")])


def _run(cmd, description):
    print("\n" + "=" * 60)
    print(f"  {description}")
    print("=" * 60)
    subprocess.run(cmd, check=True)


def run_pipeline():
    """
    End-to-end: download the same dataset scale, run the same leakage-safe
    train/held-out split, and train the same ensemble used for this repo's
    published benchmark (see README.md and ROADMAP.md's "Dataset integrity"
    section). Safe to re-run -- each step skips or tops-up rather than
    re-downloading/re-training from scratch once already satisfied.

    Note on exact reproducibility: each fresh run of the *_expand scripts
    streams a fresh random subsample from the source archives, so the exact
    files (and therefore the exact accuracy to the second decimal place)
    will vary slightly run to run -- this is expected. What stays constant
    is the methodology: same two licensed sources, same dataset scale, same
    duration-standardization, same source-recording/speaker-grouped
    held-out split, same default (untuned) hyperparameters.
    """
    print("=" * 60)
    print("   NOVA-VAD PIPELINE")
    print("=" * 60)

    if _count_wavs("data/speech") == 0:
        _run([sys.executable, "download_data.py"],
             "STEP 1a: Downloading base speech data (Google Speech Commands)")
    if _count_wavs("data/noise") == 0:
        _run([sys.executable, "download_noise.py"],
             "STEP 1b: Downloading base noise data (UrbanSound8K)")

    _run([sys.executable, "download_speech_expand.py"],
         "STEP 2a: Expanding speech dataset to the published dataset scale")
    _run([sys.executable, "download_noise_expand.py"],
         "STEP 2b: Expanding noise dataset to the published dataset scale")

    _run([sys.executable, "backfill_fsid.py"],
         "STEP 3a: Recovering UrbanSound8K source-recording IDs (for leakage-safe splitting)")
    _run([sys.executable, "backfill_speaker_id.py"],
         "STEP 3b: Recovering Speech Commands speaker IDs (for leakage-safe splitting)")

    cache_path = "data/_feature_cache.joblib"
    if os.path.exists(cache_path):
        # Stale features silently produce wrong results after a data change --
        # always clear this before (re)training.
        os.remove(cache_path)

    _run([sys.executable, "-m", "src.experiment", "final"],
         "STEP 4: Training + evaluating on the leakage-safe held-out split")

    print("\n" + "=" * 60)
    print("   DONE")
    print("=" * 60)
    print("  Exact numbers: results/final_model_report.json")
    print("  Explain a single file:  python3 -m src.explainer data/speech/speech_0001.wav")
    print("  Compare against every baseline on identical audio:")
    print("      python3 -m src.fair_comparison")


if __name__ == "__main__":
    run_pipeline()
