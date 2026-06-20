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

# ── 2. NOISE FILES — MUSAN ─────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Downloading MUSAN dataset...")
print("=" * 50)

MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
MUSAN_TAR = os.path.join(TMP_DIR, "musan.tar.gz")

urllib.request.urlretrieve(MUSAN_URL, MUSAN_TAR)
print("Extracting...")

MUSAN_TMP = os.path.join(TMP_DIR, "musan")
os.makedirs(MUSAN_TMP, exist_ok=True)

with tarfile.open(MUSAN_TAR, "r:gz") as tar:
    tar.extractall(MUSAN_TMP)

# grab from BOTH noise AND music subfolders — music is what broke WebRTC
all_noise_wavs = []

noise_root = os.path.join(MUSAN_TMP, "musan", "noise")
for root, dirs, files in os.walk(noise_root):
    for f in files:
        if f.endswith(".wav"):
            all_noise_wavs.append(os.path.join(root, f))

music_root = os.path.join(MUSAN_TMP, "musan", "music")
for root, dirs, files in os.walk(music_root):
    for f in files:
        if f.endswith(".wav"):
            all_noise_wavs.append(os.path.join(root, f))

selected_noise = random.sample(all_noise_wavs, 250)
for i, src in enumerate(selected_noise):
    dst = os.path.join(NOISE_DIR, f"noise_{i+1:03d}.wav")
    shutil.copy(src, dst)
    if (i + 1) % 50 == 0:
        print(f"  ✓ {i+1}/250 noise files copied")

print(f"\n✅ 250 noise files saved to {NOISE_DIR}/")

# ── 3. CLEANUP ─────────────────────────────────────────────────────────────
print("\nCleaning up temp files...")
shutil.rmtree(TMP_DIR)

print("\n" + "=" * 50)
print("DATASET READY")
print("=" * 50)
print(f"  data/speech/ → {len(os.listdir(SPEECH_DIR))} files")
print(f"  data/noise/  → {len(os.listdir(NOISE_DIR))} files")
print(f"  Total:         {len(os.listdir(SPEECH_DIR)) + len(os.listdir(NOISE_DIR))} files")