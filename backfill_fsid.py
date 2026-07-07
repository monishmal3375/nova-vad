"""
Backfills the UrbanSound8K source-recording ID (fsID) into
data/noise/_category_manifest.csv for every noise_XXXX.wav file already on
disk, WITHOUT re-downloading or duplicating any audio.

Why this exists: UrbanSound8K clips are 4-second slices cut from longer
original field recordings, named "<fsID>-<classID>-<occurrenceID>-<sliceID>.wav".
Multiple slices from the same source recording (fsID) can be highly
acoustically correlated -- this is exactly why UrbanSound8K's own creators
provide official predefined folds grouped by fsID, so users don't naively
split by individual clip. download_noise.py / download_noise_expand.py
renamed files to generic noise_XXX.wav and only tracked "category" in the
manifest, discarding fsID entirely.

Since the audio is streamed directly from the source archive (never kept as
a local raw cache -- see download_noise.py / download_noise_expand.py), the
only way to recover fsID for files already on disk is to re-stream the
archive, parse each member's original filename as we encounter it, and match
it against our existing data/noise/*.wav files by content hash (MD5) -- the
same content-hash approach download_noise_expand.py already uses for its
dedup check.

Usage: python3 backfill_fsid.py
Output: rewrites data/noise/_category_manifest.csv with an added fsID column.
"""
import os
import re
import csv
import hashlib
import tarfile
import urllib.request

NOISE_DIR = "data/noise"
MANIFEST_PATH = os.path.join(NOISE_DIR, "_category_manifest.csv")

FILENAME_RE = re.compile(r"^(\d+)-(\d+)-(\d+)-(\d+)\.wav$")

URL = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz"


def md5_of_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # 1. Load existing manifest (noise_filename -> category)
    if not os.path.exists(MANIFEST_PATH):
        raise SystemExit(f"Manifest not found at {MANIFEST_PATH}")

    with open(MANIFEST_PATH) as fh:
        rows = list(csv.DictReader(fh))
    print(f"Loaded {len(rows)} rows from existing manifest.")

    # 2. Hash every noise_XXXX.wav file already on disk
    print("Hashing existing noise files on disk...")
    hash_to_filenames = {}  # md5 -> list of noise_XXXX.wav filenames (dup-safe)
    for row in rows:
        fname = row["noise_filename"]
        path = os.path.join(NOISE_DIR, fname)
        if not os.path.exists(path):
            print(f"  WARNING: {fname} listed in manifest but missing on disk, skipping")
            continue
        h = md5_of_file(path)
        hash_to_filenames.setdefault(h, []).append(fname)
    print(f"Hashed {sum(len(v) for v in hash_to_filenames.values())} files "
          f"({len(hash_to_filenames)} unique hashes).")

    # 3. Re-stream the UrbanSound8K archive, parse fsID from each member's
    #    original filename, hash its content, and match against our files.
    fname_to_fsid = {}
    matched_hashes = set()
    n_members_seen = 0

    print("Re-streaming UrbanSound8K archive to recover fsID by content hash "
          "(no local copy kept, nothing re-downloaded into data/)...")
    with urllib.request.urlopen(URL) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".wav"):
                    continue
                base = os.path.basename(member.name)
                m = FILENAME_RE.match(base)
                if not m:
                    continue
                fs_id = m.group(1)

                n_members_seen += 1
                if n_members_seen % 500 == 0:
                    print(f"  ...scanned {n_members_seen} source archive members, "
                          f"{len(fname_to_fsid)}/{len(hash_to_filenames)} unique hashes matched so far")

                fh = tar.extractfile(member)
                if fh is None:
                    continue
                data = fh.read()
                h = hashlib.md5(data).hexdigest()

                if h in hash_to_filenames and h not in matched_hashes:
                    for noise_fname in hash_to_filenames[h]:
                        fname_to_fsid[noise_fname] = fs_id
                    matched_hashes.add(h)

                # Stop early once every unique hash on disk has been matched --
                # no need to stream the rest of the ~5.6GB archive.
                if len(matched_hashes) >= len(hash_to_filenames):
                    print("All existing files matched -- stopping stream early.")
                    break

    n_matched = sum(1 for row in rows if row["noise_filename"] in fname_to_fsid)
    print(f"\nMatched fsID for {n_matched}/{len(rows)} manifest rows "
          f"({len(matched_hashes)}/{len(hash_to_filenames)} unique hashes).")

    unmatched = [row["noise_filename"] for row in rows if row["noise_filename"] not in fname_to_fsid]
    if unmatched:
        print(f"WARNING: {len(unmatched)} files could not be matched to a source "
              f"archive member (kept as fsID=unknown): {unmatched[:10]}"
              f"{'...' if len(unmatched) > 10 else ''}")

    # 4. Rewrite the manifest with the new fsID column
    with open(MANIFEST_PATH, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["noise_filename", "category", "fsID"])
        for row in rows:
            fs_id = fname_to_fsid.get(row["noise_filename"], "unknown")
            writer.writerow([row["noise_filename"], row["category"], fs_id])

    print(f"\nManifest updated with fsID column at {MANIFEST_PATH}")

    # quick summary: how many distinct source recordings, and how skewed
    n_unique_fsids = len(set(fname_to_fsid.values()))
    print(f"\n{n_matched} clips backfilled map to {n_unique_fsids} distinct source recordings (fsID).")


if __name__ == "__main__":
    main()
