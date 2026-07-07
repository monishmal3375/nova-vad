"""
Backfills the Google Speech Commands speaker ID into a new
data/speech/_speaker_manifest.csv for every speech_XXXX.wav file already on
disk, WITHOUT re-downloading or duplicating any audio.

Why this exists: Google Speech Commands v0.02's original archive filenames
are "<word>/<speaker_hash>_nohash_<n>.wav" (e.g. "eight/1b88bf70_nohash_0.wav"),
where <speaker_hash> is a stable per-speaker identifier. Multiple utterances
from the same speaker could currently be split across train and test --
same class of leakage risk as UrbanSound8K's fsID. download_data.py /
download_speech_expand.py renamed files to generic speech_XXX.wav and never
tracked speaker ID anywhere.

Since the audio is streamed directly from the source archive (never kept as
a local raw cache), the only way to recover speaker ID for files already on
disk is to re-stream the archive, parse each member's original filename as
we encounter it, and match it against our existing data/speech/*.wav files
by content hash (MD5) -- same approach as backfill_fsid.py /
download_speech_expand.py's existing dedup logic.

Usage: python3 backfill_speaker_id.py
Output: writes data/speech/_speaker_manifest.csv (speech_filename,speaker_id)
"""
import os
import re
import csv
import hashlib
import tarfile
import urllib.request

SPEECH_DIR = "data/speech"
MANIFEST_PATH = os.path.join(SPEECH_DIR, "_speaker_manifest.csv")

# "<speaker_hash>_nohash_<n>.wav" is the dominant Speech Commands naming
# convention. A minority of files use "<speaker_hash>_<word>_<n>.wav" style
# names in some releases, so capture the leading hash generically: everything
# before the first underscore.
SPEAKER_RE = re.compile(r"^([0-9a-f]+)_.*\.wav$")

SPEECH_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"


def md5_of_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not os.path.isdir(SPEECH_DIR):
        raise SystemExit(f"{SPEECH_DIR} not found")

    speech_files = sorted(f for f in os.listdir(SPEECH_DIR) if f.endswith(".wav"))
    print(f"Found {len(speech_files)} speech_XXXX.wav files on disk.")

    print("Hashing existing speech files on disk...")
    hash_to_filenames = {}
    for fname in speech_files:
        h = md5_of_file(os.path.join(SPEECH_DIR, fname))
        hash_to_filenames.setdefault(h, []).append(fname)
    print(f"Hashed {len(speech_files)} files ({len(hash_to_filenames)} unique hashes).")

    fname_to_speaker = {}
    matched_hashes = set()
    n_members_seen = 0

    print("Re-streaming Speech Commands v0.02 archive to recover speaker ID "
          "by content hash (no local copy kept, nothing re-downloaded into data/)...")
    with urllib.request.urlopen(SPEECH_URL) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".wav"):
                    continue
                if "_background_noise_" in member.name:
                    continue
                base = os.path.basename(member.name)
                m = SPEAKER_RE.match(base)
                if not m:
                    continue
                speaker_id = m.group(1)

                n_members_seen += 1
                if n_members_seen % 2000 == 0:
                    print(f"  ...scanned {n_members_seen} source archive members, "
                          f"{len(matched_hashes)}/{len(hash_to_filenames)} unique hashes matched so far")

                fh = tar.extractfile(member)
                if fh is None:
                    continue
                data = fh.read()
                h = hashlib.md5(data).hexdigest()

                if h in hash_to_filenames and h not in matched_hashes:
                    for speech_fname in hash_to_filenames[h]:
                        fname_to_speaker[speech_fname] = speaker_id
                    matched_hashes.add(h)

                if len(matched_hashes) >= len(hash_to_filenames):
                    print("All existing files matched -- stopping stream early.")
                    break

    n_matched = sum(1 for f in speech_files if f in fname_to_speaker)
    print(f"\nMatched speaker ID for {n_matched}/{len(speech_files)} files "
          f"({len(matched_hashes)}/{len(hash_to_filenames)} unique hashes).")

    unmatched = [f for f in speech_files if f not in fname_to_speaker]
    if unmatched:
        print(f"WARNING: {len(unmatched)} files could not be matched to a source "
              f"archive member (kept as speaker_id=unknown): {unmatched[:10]}"
              f"{'...' if len(unmatched) > 10 else ''}")

    with open(MANIFEST_PATH, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["speech_filename", "speaker_id"])
        for fname in speech_files:
            writer.writerow([fname, fname_to_speaker.get(fname, "unknown")])

    print(f"\nSpeaker manifest written to {MANIFEST_PATH}")

    n_unique_speakers = len(set(fname_to_speaker.values()))
    print(f"\n{n_matched} clips backfilled map to {n_unique_speakers} distinct speakers.")


if __name__ == "__main__":
    main()
