import urllib.request
import os
import shutil
import random
import tarfile
import re

NOISE_DIR = "data/noise"
TMP_DIR   = "data/tmp"

os.makedirs(NOISE_DIR, exist_ok=True)
os.makedirs(TMP_DIR,   exist_ok=True)

# ── UrbanSound8K class map ─────────────────────────────────────────────────
# UrbanSound8K filenames are named "<fsID>-<classID>-<occurrenceID>-<sliceID>.wav"
# so the class is recoverable straight from the filename, no metadata CSV needed.
# See https://urbansounddataset.weebly.com/urbansound8k.html
URBANSOUND_CLASSES = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music",
}

# Per-category cap so no single noise class (e.g. engine_idling, which has the
# most clips in UrbanSound8K) dominates the held-out test set. Broadens
# coverage vs. the previous "250 random files, whatever categories fall out"
# approach, which in practice skewed toward traffic/siren/jackhammer/AC/
# construction-heavy classes.
PER_CATEGORY_TARGET = 30
TOTAL_TARGET         = 250

FILENAME_RE = re.compile(r"^\d+-(\d+)-\d+-\d+\.wav$")


def classify_by_filename(filename: str) -> int:
    """
    Returns the UrbanSound8K classID encoded in a slice filename, or None
    if the filename doesn't match the expected pattern.
    """
    m = FILENAME_RE.match(filename)
    if not m:
        return None
    return int(m.group(1))


def stratified_sample(wavs_by_class: dict, per_category: int, total_target: int) -> list:
    """
    Samples up to `per_category` files from each class, then tops up from
    the remaining pool (in round-robin fashion across classes) until
    `total_target` files are collected or the pool is exhausted.
    """
    random.seed(42)
    selected = []
    leftover_by_class = {}

    for class_id, files in wavs_by_class.items():
        files = files[:]
        random.shuffle(files)
        take = files[:per_category]
        selected.extend(take)
        leftover_by_class[class_id] = files[per_category:]

    # top up round-robin from leftovers if we haven't hit the total target
    class_ids = list(leftover_by_class.keys())
    idx = 0
    while len(selected) < total_target and any(leftover_by_class.values()):
        class_id = class_ids[idx % len(class_ids)]
        pool = leftover_by_class[class_id]
        if pool:
            selected.append(pool.pop())
        idx += 1

    random.shuffle(selected)
    return selected[:total_target]


def main():
    print("Downloading + extracting UrbanSound8K dataset (streamed, no local .tar.gz kept)...")
    URL = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz"

    TMP_EXTRACT = os.path.join(TMP_DIR, "urbansound")
    os.makedirs(TMP_EXTRACT, exist_ok=True)

    # Stream the download straight into tarfile's extractor instead of
    # saving the ~6GB .tar.gz to disk first. This roughly halves peak
    # disk usage during setup (extracted-only vs. tar + extracted).
    with urllib.request.urlopen(URL) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as tar:
            tar.extractall(TMP_EXTRACT, filter="data")

    print("Download + extraction complete.")

    # collect all wav files, bucketed by UrbanSound8K category
    wavs_by_class = {class_id: [] for class_id in URBANSOUND_CLASSES}
    unclassified  = []

    for root, dirs, files in os.walk(TMP_EXTRACT):
        for f in files:
            if not f.endswith(".wav"):
                continue
            path     = os.path.join(root, f)
            class_id = classify_by_filename(f)
            if class_id is not None and class_id in wavs_by_class:
                wavs_by_class[class_id].append(path)
            else:
                unclassified.append(path)

    print("\nFiles found per category:")
    for class_id, name in URBANSOUND_CLASSES.items():
        print(f"  [{class_id}] {name:<18} {len(wavs_by_class[class_id])} files")
    if unclassified:
        print(f"  (unclassified, filename didn't match pattern: {len(unclassified)})")

    selected = stratified_sample(wavs_by_class, PER_CATEGORY_TARGET, TOTAL_TARGET)

    # keep a manifest of which category each copied file came from, so the
    # benchmark can report per-category breakdowns later
    manifest_path = os.path.join(NOISE_DIR, "_category_manifest.csv")
    with open(manifest_path, "w") as manifest:
        manifest.write("noise_filename,category\n")
        for i, src in enumerate(selected):
            dst = os.path.join(NOISE_DIR, f"noise_{i+1:03d}.wav")
            shutil.copy(src, dst)
            class_id = classify_by_filename(os.path.basename(src))
            category = URBANSOUND_CLASSES.get(class_id, "unknown")
            manifest.write(f"noise_{i+1:03d}.wav,{category}\n")
            if (i + 1) % 50 == 0:
                print(f"  ✓ {i+1}/{len(selected)} files copied")

    print(f"\n✅ {len(selected)} noise files saved to {NOISE_DIR}/")
    print(f"   Category manifest written to {manifest_path}")

    print("Cleaning up...")
    shutil.rmtree(TMP_DIR)

    print(f"\nFinal count:")
    if os.path.exists("data/speech"):
        print(f"  data/speech/ → {len(os.listdir('data/speech'))} files")
    print(f"  data/noise/  → {len([f for f in os.listdir(NOISE_DIR) if f.endswith('.wav')])} files")


if __name__ == "__main__":
    main()
