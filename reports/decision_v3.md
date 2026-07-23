# Decision v3: closing the noise-robustness gap — did it work, and at what cost?

**Date:** 2026-07-23
**Goal:** close the MCC/accuracy gap with Silero/Pyannote via noise-robust
features + more training data + better post-processing, without touching
clean-audio accuracy or the way the benchmark measures results.
**Constraint discipline:** see "Integrity confirmation" section at the end —
read that before trusting any number above it.

## What changed, and why

| Change | Why |
|---|---|
| Audited train scene composition | Before touching anything, checked whether class imbalance explained the noise gap. |
| New 'val' split (40 scenes, seed=43, train-source files) | A second validation set, separate from the already-published 'dev' split used for v1's tuning, so this iteration's tuning has its own clean provenance. |
| 4 new features: periodicity_strength, estimated_f0_hz, spectral_flatness (mean+std) | Structural/harmonic features chosen from acoustic theory (periodicity survives additive noise better than energy does), not from inspecting test failures. |
| 300 additional train scenes ('train2', 75/condition, seed=44) | More noisy training volume, matching eval conditions (clean/10dB/0dB/-5dB), train-only noise sources. |
| Retrained RF+GBT (v1's architecture, v2's 62 features, train+train2 = 400 scenes / 48,000 windows) | — |
| Re-tuned hysteresis (T_on, T_off) + added median-filter option, grid-searched on 'val' only | Target precision specifically (v1's 42.77% precision was well behind its 68.86% recall). |

## Step 1: Training data composition audit — no imbalance found

25 scenes per condition (clean/10dB/0dB/-5dB) in the original train split, all
in the 22–26% speech-ratio range. **This was not the cause of the noise gap.**
Reported honestly rather than manufacturing an imbalance fix for a problem
that didn't exist — see conversation history for the exact per-condition
numbers.

## Step 2: New features — first-principles validation, not test-tuned

Before touching any project audio, `periodicity_strength` was validated
against synthetic signals:

| Signal | periodicity_strength | estimated F0 |
|---|---|---|
| Pure 150Hz tone | 0.979 | 149.5 Hz (true: 150) |
| White noise | 0.037 | 0 Hz (correctly none) |
| 150Hz tone + heavy noise | 0.442 | 150.9 Hz (survives, degraded but present) |

This is the whole premise for why the feature should help exactly where
energy-based features fail: periodicity degrades gracefully under additive
noise while raw magnitude features get swamped. Automated tests for this:
`tests/test_frame_features_v2.py`.

**Honest caveat:** in the trained Random Forest, these 4 new features ranked
29th, 37th, 38th, and 52nd of 62 by importance — not dominant features. The
downstream result improved anyway (see below), most likely because of the
combination of more training volume + retuned thresholds, not because the
new features became primary decision drivers. Flagging this rather than
overstating the new features' individual contribution.

## Step 3: Leakage checks (before training)

```
train2 noise  ∩ test noise : set()
train2 speech ∩ test speech: set()
val noise     ∩ test noise : set()
val speech    ∩ test speech: set()
train noise   ∩ test noise : set()   (already true from v1, reconfirmed)
train speech  ∩ test speech: set()
```
All empty. Test/dev scene `.wav` files verified byte-identical (SHA-256)
before and after every regeneration step in this iteration.

## Step 4: Validation-set tuning (data/scenes/val, 40 scenes — NOT test)

Grid search over T_on ∈ {0.35..0.75}, T_off ∈ {0.15..0.45}, median filter
size ∈ {1,3,5}, selecting by frame F1 (same objective as v1's tuning — not
switched post-hoc). 60 combinations tested, full log in conversation
history.

**Winner:** T_on=0.55, T_off=0.45, median_filter_size=1 (i.e. the median
filter did not help on val at any tested size — honestly reporting a
negative result instead of omitting it).

**Validation-set result:** F1=62.14%, precision=71.42%, recall=55.00%.

## Step 5: Final test-set result (SINGLE evaluation, first test contact for v2)

| Metric | v1 (before) | v2 (after) | Change |
|---|---|---|---|
| Accuracy | 69.88% | **78.99%** | +9.11pp |
| 95% CI | [61.88, 77.21] | **[73.06, 83.75]** | narrower (15.3pp → 10.7pp wide) |
| Precision | 42.77% | **57.10%** | +14.33pp |
| Recall | 68.86% | 56.24% | **-12.62pp** |
| F1 | 52.77% | 56.67% | +3.90pp |
| MCC | 0.3437 | **0.428** | +0.084 |
| False positives | 10,805 | 4,955 | **-54%** |
| False negatives | 3,651 | 5,131 | **+41%** |

Now nearly matching SpeechBrain (MCC 0.44), still behind Pyannote (0.53)
and Silero (0.57).

### Per-condition accuracy

| Condition | v1 | v2 | Change |
|---|---|---|---|
| clean | 90.61% | **89.53%** | **-1.08pp** |
| 10dB | 69.93% | 73.67% | +3.74pp |
| 0dB | 55.37% | 78.78% | +23.41pp |
| -5dB | 63.62% | 73.98% | +10.36pp |

## Explicit flags — read these before trusting the headline numbers

1. **Clean-audio accuracy moved, by -1.08pp.** The goal was explicitly "without
   touching accuracy on clean audio." It technically did move. With only 10
   clean test scenes and a per-scene std-dev of 2.7% observed for v1 on this
   condition, a ~1pp shift is plausibly within scene-to-scene noise rather
   than a real regression — but I have not proven that statistically (no
   significance test was run on this specific delta), so this is flagged as
   an open item, not waved away.
2. **Precision gain came with a real recall cost**, not for free: the
   T_on/T_off thresholds moved from (0.45, 0.35) to (0.55, 0.45) — a
   stricter trigger. This is an expected, honest trade-off from threshold
   placement, not a "both metrics improved" free lunch. Net effect on F1 was
   positive (+3.9pp) and MCC was clearly positive (+0.084), so the overall
   trade was favorable, but recall did drop by 12.6pp and that has to be
   weighed against the use case (an application that can't tolerate missed
   speech should not adopt this threshold as-is).
3. **The 0dB condition's +23.4pp jump is the largest single number in this
   report and deserves scrutiny, not celebration.** Combined with the
   noise-file diagnostic below, this is most plausibly explained by v2
   simply handling `noise_176.wav` (which appears in multiple 0dB and -5dB
   scenes and was independently identified as an unusually hard, tonal
   noise file) better than v1 did — not by a general, stable improvement at
   exactly 0dB SNR. The per-condition breakdown has only 10 scenes and 8
   unique noise files per bucket; treat per-condition deltas as directional,
   not precise.
4. **This was one run, one seed, one train/val/test split** — not repeated
   across multiple seeds. Per the no-cherry-picking constraint, this is
   disclosed as a single run, not implied to be a stable mean.

## Diagnosis: the non-monotonic SNR curve (both v1 and v2)

v1 showed 0dB (55.4%) scoring *worse* than -5dB (63.6%) — non-monotonic.
v2 shows a different non-monotonic shape: 0dB (78.8%) now scores *better*
than both its neighbors, 10dB (73.7%) and -5dB (74.0%).

Investigated two ways, both using only structural properties or already-
frozen model outputs — neither influenced any tuning decision above:

**A. Test-scene composition (structural, no scores involved):** 0dB and
-5dB test scenes differ only slightly in clip count/duration/gaps, but only
share 1 of 8 noise files. Measuring the *raw* noise files directly (before
any mixing, before any model involvement): the 0dB condition's 8 noise
files average periodicity_strength=0.379 / spectral_flatness=0.051, vs.
-5dB's 0.347 / 0.059. The 0dB bucket got noise that happens to be more
tonal/music-like — exactly the acoustic property that makes a noise source
harder to distinguish from speech, independent of its loudness (SNR).

**B. Per-scene accuracy (v1, already-frozen model, not re-tuned from this):**
`noise_176.wav` appears in 3 of the model's 8 worst-scoring test scenes
(16.5%, 20.2%, 28.5% accuracy) spanning both the 0dB and -5dB conditions.
`noise_180.wav` similarly drags down two 10dB scenes. Per-condition
accuracy std-dev is 21–27% for the three noisy conditions vs. 2.7% for
clean — performance is dominated by *which specific noise file* landed in
a scene, not by a smooth function of nominal SNR.

**Conclusion:** the non-monotonicity in both v1 and v2 is most plausibly a
small-sample noise-file-identity confound (only 8 unique noise files per
condition, 10 scenes), not a genuine "medium SNR is uniquely hard for VAD"
effect. This is a benchmark statistical-power limitation, not something
"fixed" by this iteration's changes — and no feature or threshold in this
report was designed by targeting `noise_176.wav` or any other specific
failing scene, consistent with the no-overfitting constraint. The
actionable conclusion is that the benchmark itself needs more unique noise
sources per condition before per-condition deltas smaller than ~10pp should
be trusted.

## Integrity confirmation

- **No test-set contact during tuning.** `data/scenes/test/` was read
  exactly once, in `scripts/evaluate_frame_vad_v2.py`, after every model
  and threshold decision was already frozen from `data/scenes/val/`
  results. Verified by SHA-256 that `data/scenes/test/` and
  `data/scenes/dev/` files are byte-identical to before this iteration
  started (dev untouched entirely; test only read, never written).
- **No leakage.** Noise/speech source file sets for train, train2, and val
  have zero intersection with test's source file sets (shown above).
- **Threshold tuned on a pre-registered validation split**, frozen before
  test contact (Step 4 → Step 5 ordering, not reversed).
- **Single run reported**, not a best-of-N. No alternate seeds or
  checkpoints were tried and discarded.
- **Same metric code** (`scripts/frame_benchmark.py`'s `compute_metrics`/
  `run_system`/cluster-bootstrap CI, imported directly, not reimplemented)
  used for both v1 and v2 — no redefinition.
- **No test-scene-specific patching.** The 4 new features were designed and
  validated against synthetic tones/noise (Step 2) before ever being run on
  project audio; the noise-file diagnostic (this section, part B) happened
  *after* the model and threshold were already frozen and is reported as an
  explanation, not fed back into a further tuning round.
- **Reproduction command** (exact order, this machine, this seed set):
  ```
  python3 -m scripts.generate_val_split      # val, seed=43
  python3 -m scripts.generate_train2_split   # train2, seed=44
  python3 -m scripts.train_frame_vad_v2      # trains on train+train2
  python3 -m scripts.tune_frame_vad_v2       # tunes on val only
  python3 -m scripts.evaluate_frame_vad_v2   # single test-set run
  ```

## Bottom line

Real, verified progress on the stated goal: MCC 0.34→0.43, accuracy
69.9%→79.0%, precision 42.8%→57.1% (54% fewer false positives), CI
tightened. Still behind Pyannote (0.53) and Silero (0.57). Cost: recall
dropped 12.6pp, and clean-audio accuracy moved by -1.1pp against an explicit
"don't touch it" constraint — small, plausibly noise, but not proven so and
disclosed rather than hidden. The largest single per-condition change
(+23.4pp at 0dB) is most likely a noise-file-specific effect rather than a
stable SNR-band improvement, and the benchmark's per-condition granularity
(8 unique noise files/condition) isn't statistically powered to fully
separate those explanations — that's the most concrete next fix if
per-condition precision matters more than the overall number.
