import urllib.request
import tarfile
import os
import shutil
import random
import re
import hashlib

SPEECH_DIR = "data/speech"
TMP_DIR    = "data/tmp_speech_expand"

os.makedirs(SPEECH_DIR, exist_ok=True)
os.makedirs(TMP_DIR,    exist_ok=True)

SPEECH_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

# Google Speech Commands v0.02 has ~105,000 utterances total across ~35 word
# folders (this is the same corpus src/experiment.py and download_data.py
# already use, CC BY 4.0). Currently 900 files are in use. Target a
# substantial, non-overlapping expansion -- more than double the current set --
# balanced against reasonable download/processing time (streaming ~2.3GB tar).
NEW_TARGET = 1000


def md5_bytes(data):
    return hashlib.md5(data).hexdigest()


def main():
    existing_hashes = set()
    for f in os.listdir(SPEECH_DIR):
        if f.endswith(".wav"):
            h = hashlib.md5()
            with open(os.path.join(SPEECH_DIR, f), "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            existing_hashes.add(h.hexdigest())
    print(f"Existing speech files on disk: {len(existing_hashes)} (hashed for overlap check)")

    print("Streaming Speech Commands v0.02 (no local .tar.gz kept)...")

    candidates = []  # list of (tmp_path, hash)
    seen_count = 0

    # Oversample a bit beyond NEW_TARGET so the final selection is a random
    # subsample (unbiased), not just "whatever came first in the tar".
    OVERSAMPLE_CAP = int(NEW_TARGET * 1.3) + 50

    with urllib.request.urlopen(SPEECH_URL) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".wav"):
                    continue
                if "_background_noise_" in member.name:
                    continue
                if len(candidates) >= OVERSAMPLE_CAP:
                    # We have plenty of oversample candidates already -- stop
                    # reading the rest of the stream early to save bandwidth.
                    break

                fh = tar.extractfile(member)
                if fh is None:
                    continue
                data = fh.read()
                seen_count += 1
                if seen_count % 2000 == 0:
                    print(f"  ...scanned {seen_count} speech clips from stream, "
                          f"{len(candidates)} new candidates so far")

                h = md5_bytes(data)
                if h in existing_hashes:
                    continue  # true overlap check against what's already on disk

                tmp_path = os.path.join(TMP_DIR, f"{len(candidates)}.wav")
                with open(tmp_path, "wb") as out:
                    out.write(data)
                candidates.append((tmp_path, h))
                existing_hashes.add(h)

    print(f"\nCollected {len(candidates)} new-candidate speech files (target {NEW_TARGET}).")

    random.seed(42)
    random.shuffle(candidates)
    take = candidates[:NEW_TARGET]
    leftover = candidates[NEW_TARGET:]

    existing_nums = []
    for f in os.listdir(SPEECH_DIR):
        m = re.match(r"speech_(\d+)\.wav$", f)
        if m:
            existing_nums.append(int(m.group(1)))
    next_idx = max(existing_nums, default=0) + 1

    added = 0
    for tmp_path, h in take:
        dst = os.path.join(SPEECH_DIR, f"speech_{next_idx:04d}.wav")
        shutil.move(tmp_path, dst)
        next_idx += 1
        added += 1

    for tmp_path, h in leftover:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print(f"\nAdded {added} genuinely new (non-overlapping) speech files.")
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    final_count = len([f for f in os.listdir(SPEECH_DIR) if f.endswith(".wav")])
    print(f"data/speech/ now has {final_count} total files")


if __name__ == "__main__":
    main()
