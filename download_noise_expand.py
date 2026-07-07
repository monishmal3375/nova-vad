import urllib.request
import os
import shutil
import random
import tarfile
import re
import hashlib

NOISE_DIR = "data/noise"
TMP_DIR   = "data/tmp_expand"

os.makedirs(NOISE_DIR, exist_ok=True)
os.makedirs(TMP_DIR,   exist_ok=True)

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

# Per-category FINAL totals (not "add N more") -- car_horn (429 total) and
# gun_shot (374 total) are the hard ceilings in UrbanSound8K, so their targets
# sit close to that ceiling; the other 8 categories have ~1,000 clips each in
# the source archive. These are the per-category totals in the dataset that
# produced this repo's published 99.80% benchmark (~4,091 noise files). This
# script computes the delta against whatever's already on disk per category
# (read from the manifest), so it works correctly against an empty
# data/noise/ (fresh clone) or a partially-populated one alike.
PER_CATEGORY_TOTAL_TARGET = {
    "air_conditioner": 420,
    "car_horn": 400,        # ~429 total in UrbanSound8K -- near the ceiling
    "children_playing": 420,
    "dog_bark": 420,
    "drilling": 420,
    "engine_idling": 420,
    "gun_shot": 350,        # ~374 total in UrbanSound8K -- near the ceiling
    "jackhammer": 420,
    "siren": 420,
    "street_music": 420,
}

FILENAME_RE = re.compile(r"^\d+-(\d+)-\d+-\d+\.wav$")


def classify_by_filename(filename: str):
    m = FILENAME_RE.match(filename)
    if not m:
        return None
    return int(m.group(1))


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # Build set of existing noise file hashes so we never double-count an
    # already-present clip as "new", even if UrbanSound8K's own random
    # ordering happens to resample the same source file.
    existing_hashes = set()
    for f in os.listdir(NOISE_DIR):
        if f.endswith(".wav"):
            existing_hashes.add(md5_of(os.path.join(NOISE_DIR, f)))
    print(f"Existing noise files on disk: {len(existing_hashes)} (hashed for overlap check)")

    # Read existing per-category counts from the manifest (if any) so the
    # delta-to-target below works whether data/noise/ is empty (fresh clone)
    # or already partially populated.
    existing_per_category = {name: 0 for name in URBANSOUND_CLASSES.values()}
    manifest_path = os.path.join(NOISE_DIR, "_category_manifest.csv")
    if os.path.exists(manifest_path):
        import csv
        with open(manifest_path) as fh:
            for row in csv.DictReader(fh):
                cat = row.get("category")
                if cat in existing_per_category:
                    existing_per_category[cat] += 1
    print("Existing per-category counts:", existing_per_category)

    print("Downloading + streaming UrbanSound8K dataset (no local .tar.gz kept)...")
    URL = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz"

    wavs_by_class = {class_id: [] for class_id in URBANSOUND_CLASSES}

    # Stream-extract member by member so we never materialize the whole
    # ~5.6GB archive on disk at once -- extract each wav, hash it, either
    # keep it (if new) or discard it immediately (if a dup of what we have).
    kept_by_class = {class_id: [] for class_id in URBANSOUND_CLASSES}
    target_by_class = {
        cid: max(0, PER_CATEGORY_TOTAL_TARGET[name] - existing_per_category[name])
        for cid, name in URBANSOUND_CLASSES.items()
    }
    print("Per-category download targets (delta to reach final total):", target_by_class)

    # Two-pass isn't feasible without holding the whole archive, so we do a
    # single streaming pass: extract every wav to a scratch path, hash+classify,
    # and either move it into a per-class staging list (up to some overhead
    # buffer beyond target, to allow later random subsampling for unbiased
    # selection) or delete it right away once we have "enough" oversample
    # per class to randomly draw the final target from.
    OVERSAMPLE_FACTOR = 1.6  # collect a bit more than target per class, then subsample randomly
    stop_collecting = {cid: False for cid in URBANSOUND_CLASSES}

    count_seen = 0
    with urllib.request.urlopen(URL) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".wav"):
                    continue
                fname = os.path.basename(member.name)
                class_id = classify_by_filename(fname)
                if class_id is None or class_id not in URBANSOUND_CLASSES:
                    continue
                if stop_collecting[class_id]:
                    continue

                cap = int(target_by_class[class_id] * OVERSAMPLE_FACTOR) + 20
                if len(kept_by_class[class_id]) >= cap:
                    stop_collecting[class_id] = True
                    if all(stop_collecting.values()):
                        print("Collected enough oversample for all categories, stopping stream early.")
                        break
                    continue

                fh = tar.extractfile(member)
                if fh is None:
                    continue
                data = fh.read()
                h = hashlib.md5(data).hexdigest()
                count_seen += 1
                if count_seen % 500 == 0:
                    print(f"  ...scanned {count_seen} class-matched clips from stream")

                if h in existing_hashes:
                    continue  # already have this exact clip -- true overlap check

                tmp_path = os.path.join(TMP_DIR, f"{class_id}_{len(kept_by_class[class_id])}.wav")
                with open(tmp_path, "wb") as out:
                    out.write(data)
                kept_by_class[class_id].append((tmp_path, h))
                existing_hashes.add(h)  # prevent re-adding same clip twice within this run

    print("\nOversampled candidates collected per category:")
    for cid, name in URBANSOUND_CLASSES.items():
        print(f"  {name:<18} {len(kept_by_class[cid])} candidates (target {target_by_class[cid]})")

    # random subsample down to the exact target per category
    random.seed(42)
    manifest_path = os.path.join(NOISE_DIR, "_category_manifest.csv")
    existing_manifest_rows = []
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            existing_manifest_rows = fh.readlines()

    # figure out next available numeric index for noise_XXXX.wav naming
    existing_nums = []
    for f in os.listdir(NOISE_DIR):
        m = re.match(r"noise_(\d+)\.wav$", f)
        if m:
            existing_nums.append(int(m.group(1)))
    next_idx = max(existing_nums, default=0) + 1

    new_rows = []
    total_added = 0
    for cid, name in URBANSOUND_CLASSES.items():
        candidates = kept_by_class[cid]
        random.shuffle(candidates)
        take = candidates[: target_by_class[cid]]
        for tmp_path, h in take:
            dst = os.path.join(NOISE_DIR, f"noise_{next_idx:04d}.wav")
            shutil.move(tmp_path, dst)
            new_rows.append(f"noise_{next_idx:04d}.wav,{name}\n")
            next_idx += 1
            total_added += 1
        # clean up any un-taken oversample candidates for this class
        for tmp_path, h in candidates[target_by_class[cid]:]:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    with open(manifest_path, "w") as fh:
        if existing_manifest_rows:
            fh.writelines(existing_manifest_rows)
        else:
            fh.write("noise_filename,category\n")
        fh.writelines(new_rows)

    print(f"\nAdded {total_added} genuinely new (non-overlapping) noise files.")
    print(f"Manifest updated at {manifest_path}")

    shutil.rmtree(TMP_DIR, ignore_errors=True)

    final_count = len([f for f in os.listdir(NOISE_DIR) if f.endswith(".wav")])
    print(f"\ndata/noise/ now has {final_count} total files")


if __name__ == "__main__":
    main()
