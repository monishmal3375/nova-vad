import urllib.request
import tarfile
import os
import shutil
import random

# ── paths ──────────────────────────────────────────────────────────────────
SPEECH_DIR = "data/speech"
NOISE_DIR  = "data/noise"
TMP_DIR    = "data/tmp"

os.makedirs(SPEECH_DIR, exist_ok=True)
os.makedirs(NOISE_DIR,  exist_ok=True)
os.makedirs(TMP_DIR,    exist_ok=True)

# ── 1. SPEECH FILES — Google Speech Commands ───────────────────────────────
print("=" * 50)
print("Downloading Speech Commands dataset...")
print("=" * 50)

SPEECH_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
SPEECH_TAR = os.path.join(TMP_DIR, "speech_commands.tar.gz")

urllib.request.urlretrieve(SPEECH_URL, SPEECH_TAR)
print("Extracting...")

SPEECH_TMP = os.path.join(TMP_DIR, "speech_commands")
os.makedirs(SPEECH_TMP, exist_ok=True)

with tarfile.open(SPEECH_TAR, "r:gz") as tar:
    tar.extractall(SPEECH_TMP)

# grab 250 random .wav files across different word folders
all_speech_wavs = []
for root, dirs, files in os.walk(SPEECH_TMP):
    # skip background noise folder inside speech commands
    if "_background_noise_" in root:
        continue
    for f in files:
        if f.endswith(".wav"):
            all_speech_wavs.append(os.path.join(root, f))

selected_speech = random.sample(all_speech_wavs, 250)
for i, src in enumerate(selected_speech):
    dst = os.path.join(SPEECH_DIR, f"speech_{i+1:03d}.wav")
    shutil.copy(src, dst)
    if (i + 1) % 50 == 0:
        print(f"  ✓ {i+1}/250 speech files copied")

print(f"\n✅ 250 speech files saved to {SPEECH_DIR}/")

# Noise comes exclusively from UrbanSound8K via download_noise.py -- this
# script used to also pull 250 files from MUSAN (noise+music subfolders) into
# data/noise/, which was dead/inconsistent with the rest of the pipeline:
# every benchmark number in the README, the fsID-based leakage checks, and
# src/experiment.py's held-out split are all built around UrbanSound8K's 10
# categories specifically. Mixing in MUSAN clips here would have both
# silently changed what "noisy real-world audio" means in the benchmark and
# risked filename collisions with download_noise.py's own noise_XXX.wav
# numbering. Removed rather than left in as an unused code path.

# ── 2. CLEANUP ─────────────────────────────────────────────────────────────
print("\nCleaning up temp files...")
shutil.rmtree(TMP_DIR)

print("\n" + "=" * 50)
print("SPEECH DATASET READY")
print("=" * 50)
print(f"  data/speech/ → {len(os.listdir(SPEECH_DIR))} files")
print("  Run download_noise.py next to fetch UrbanSound8K noise files.")