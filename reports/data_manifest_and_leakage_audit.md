# Data manifest and leakage audit — all splits

**Purpose:** durable record of every scene split's composition and the
leakage checks between them, consolidating what was previously verified
piecemeal across `decision_v3.md`/`decision_v4.md` chat-visible output
into one canonical, re-runnable document, per the Phase A release-gate
checklist (plan Section 7.9, item 1).

**Re-generated fresh on 2026-07-23** (round 3) rather than copied from
earlier rounds' chat output, so this reflects the actual current state of
`data/scenes/` on disk.

## Split composition

| Split | Scenes | Unique noise files | Unique speech files | Role |
|---|---|---|---|---|
| `train` | 100 | 78 | 151 | v1's original training data (seed=42) |
| `train2` | 300 | 149 | 199 | v2's additional training volume (seed=44) |
| `train3` | 150 | 104 | 177 | targeted 0dB/-5dB data for v3 — **not deployed, see below** (seed=46) |
| `dev` | 20 | 19 | 46 | v1's hysteresis threshold tuning only (seed=42, continued sequence) |
| `val` | 40 | 38 | 96 | v2/v3's threshold tuning + round-3 ensemble fitting (seed=43) |
| `test` | 40 | 27 | 43 | original locked test set — historical record only, not used for any decision after round 2 (seed=42, continued sequence) |
| `test_v2` | 100 | 48 | 49 | **current locked test set** (seed=45, without-replacement noise sampling) |

All splits are grouped by source recording: every scene's `.json` manifest
records `source_noise_file` and `source_speech_files`, and every train-side
split is drawn from `split_dataset()`'s `train_speech`/`train_noise` pool
(200+200 files), every test-side split from the same function's
`test_speech`/`test_noise` pool (50+50 files) — a single seeded 80/20
division (`src/benchmark.py:split_dataset`, seed=42) reused everywhere, so
"train-side" and "test-side" can never accidentally diverge across splits.

## Pairwise leakage checks (every train-side split vs. every test-side split)

Verification method: set-intersection of `source_noise_file` values and
`source_speech_files` values between each pair. Re-run live for this
document:

```
train      vs test      : noise_overlap=0, speech_overlap=0  [OK]
train      vs test_v2   : noise_overlap=0, speech_overlap=0  [OK]
train2     vs test      : noise_overlap=0, speech_overlap=0  [OK]
train2     vs test_v2   : noise_overlap=0, speech_overlap=0  [OK]
train3     vs test      : noise_overlap=0, speech_overlap=0  [OK]
train3     vs test_v2   : noise_overlap=0, speech_overlap=0  [OK]
dev        vs test      : noise_overlap=0, speech_overlap=0  [OK]
dev        vs test_v2   : noise_overlap=0, speech_overlap=0  [OK]
val        vs test      : noise_overlap=0, speech_overlap=0  [OK]
val        vs test_v2   : noise_overlap=0, speech_overlap=0  [OK]
```

**Result: zero overlap on every pair, on both noise and speech source
files.** No leakage between any train-side split and any test-side split.

## Reproduction

```bash
python3 -c "
import json, glob
def files(split):
    noise, speech, n = set(), set(), 0
    for p in glob.glob(f'data/scenes/{split}/*.json'):
        with open(p) as f: m = json.load(f)
        noise.add(m['source_noise_file']); speech.update(m['source_speech_files']); n += 1
    return {'noise': noise, 'speech': speech, 'n_scenes': n}
splits = ['train','train2','train3','dev','val','test','test_v2']
data = {s: files(s) for s in splits}
for tr in ['train','train2','train3','dev','val']:
    for te in ['test','test_v2']:
        no = data[tr]['noise'] & data[te]['noise']
        so = data[tr]['speech'] & data[te]['speech']
        print(f'{tr} vs {te}: noise_overlap={len(no)}, speech_overlap={len(so)}')
"
```

## Generation seeds (for full reproducibility of the splits themselves)

| Split | Script | Seed |
|---|---|---|
| dev, test, train | `scripts/generate_scenes.py` | 42 (single `random.Random(42)`, consumed sequentially: dev → test → train) |
| val | `scripts/generate_val_split.py` | 43 |
| train2 | `scripts/generate_train2_split.py` | 44 |
| test_v2 | `scripts/generate_test_v2.py` | 45 |
| train3 | `scripts/generate_train3_targeted.py` | 46 |

## Note on train3 / NOVA-VAD-frame-v3

`train3` exists on disk and is included in this audit for completeness,
but the model trained on it (NOVA-VAD-frame-v3) regressed on every test
condition and was **not deployed** — see
`models/registry/nova-vad-frame-v3/model_card.md` and `reports/decision_v4.md`.
It is kept as a documented negative result, not deleted.
