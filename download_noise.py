import urllib.request
import zipfile
import os
import shutil
import random

NOISE_DIR = "data/noise"
TMP_DIR   = "data/tmp"

os.makedirs(NOISE_DIR, exist_ok=True)
os.makedirs(TMP_DIR,   exist_ok=True)

# UrbanSound8K — real world noise, perfect for VAD testing
print("Downloading UrbanSound8K dataset...")
URL     = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz"
TAR     = os.path.join(TMP_DIR, "urbansound.tar.gz")

urllib.request.urlretrieve(URL, TAR)
print("Download complete. Extracting...")

TMP_EXTRACT = os.path.join(TMP_DIR, "urbansound")
os.makedirs(TMP_EXTRACT, exist_ok=True)

import tarfile
with tarfile.open(TAR, "r:gz") as tar:
    tar.extractall(TMP_EXTRACT, filter="data")

# collect all wav files
all_wavs = []
for root, dirs, files in os.walk(TMP_EXTRACT):
    for f in files:
        if f.endswith(".wav"):
            all_wavs.append(os.path.join(root, f))

print(f"Found {len(all_wavs)} noise files total")

take     = min(250, len(all_wavs))
selected = random.sample(all_wavs, take)

for i, src in enumerate(selected):
    dst = os.path.join(NOISE_DIR, f"noise_{i+1:03d}.wav")
    shutil.copy(src, dst)
    if (i + 1) % 50 == 0:
        print(f"  ✓ {i+1}/{take} files copied")

print(f"\n✅ {take} noise files saved to data/noise/")

print("Cleaning up...")
shutil.rmtree(TMP_DIR)

print(f"\nFinal count:")
print(f"  data/speech/ → {len(os.listdir('data/speech'))} files")
print(f"  data/noise/  → {len(os.listdir('data/noise'))} files")