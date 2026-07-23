# Decision v4 — Round 2: fixing benchmark validity, resolving open flags, and a negative result

**Date:** 2026-07-23
**Two goals, equal priority, in mandated order:** (1) fix the benchmark's
ability to measure truthfully, (2) push the model further — Part 1 done and
locked before any Part 2/3 model number was produced.

---

## Part 1 — Benchmark validity fix

### Step 1: How much can one noise file swing a condition's accuracy? (original 40-scene test set, all 7 systems)

Leave-one-noise-file-out (LONO): for each condition, removed each unique
noise file's scenes in turn and measured the accuracy shift. Full table:
`reports/per_scene_test_original_lono.json`. Headline numbers (max swing,
percentage points):

| System | clean | 10dB | 0dB | -5dB |
|---|---|---|---|---|
| NOVA-VAD (v0) | 2.89 | 6.23 | 4.56 | 4.30 |
| WebRTC VAD | 1.05 | 5.53 | **8.81** | 3.50 |
| NOVA-VAD-frame-v1 | 0.66 | **10.96** | 7.76 | 5.24 |
| NOVA-VAD-frame-v2 | 1.07 | **11.90** | 5.76 | 2.70 |
| Silero VAD | 0.60 | 0.72 | 1.32 | 0.98 |
| Pyannote VAD | 0.78 | 4.67 | 5.34 | 1.03 |
| SpeechBrain VAD | 1.77 | 1.56 | 4.14 | 3.30 |

**This affected every system, not just NOVA-VAD** — max swings up to
**11.9pp** on only 8 unique noise files per condition. Clean condition was
already stable everywhere (max 2.9pp). The weaker/more brittle systems
(both NOVA-VAD versions, WebRTC) show the largest swings; Silero's
inherent robustness makes it far less sensitive to which specific file
appears (never above 1.32pp) — i.e. the measurement problem was partly
masked by Silero/Pyannote being good enough that no single hard file could
move them much, but it was real and large for the systems in the middle of
the pack, which is exactly where the interesting comparisons (v1 vs v2 vs
WebRTC vs SpeechBrain) are happening.

### Step 2: Expanded test set ('test_v2')

Sized the fix from Step 1's numbers: swing scales roughly as 1/n_files, so
diluting the worst case (~12pp at 8 files) to the ~3-5pp target needs
roughly 8 × (12/4) ≈ 24-25 files. Built **25 unique noise files per
condition** (up from 8) — sampled *without replacement* within each
condition (the original generator sampled with replacement, which is what
caused only 8 unique files out of 10 draws by collision). 100 scenes total
(up from 40), drawn exclusively from the test-only 50-file noise pool
(confirmed by set-intersection check against every train-side split: zero
overlap). Original `data/scenes/test/` (40 scenes) is untouched, kept as
historical record — it's what the LONO analysis above was run against.

### Step 3: All 7 systems re-run on test_v2 — this is a MEASUREMENT change, not a model change

No model was retrained for this table. Same frozen v0/v1/v2 artifacts,
same WebRTC/Silero/Pyannote/SpeechBrain adapters, same scoring code
(`scripts/frame_benchmark.py`'s functions, reused not reimplemented) — only
the audio being scored changed.

| Model | test (orig, 40 scenes) Acc / MCC | test_v2 (100 scenes) Acc / MCC | Δ Acc |
|---|---|---|---|
| NOVA-VAD | 33.26% / -0.280 | 38.03% / -0.219 | +4.77pp |
| WebRTC VAD | 53.76% / 0.197 | 54.75% / 0.224 | +0.99pp |
| NOVA-VAD-frame-v1 | 69.88% / 0.344 | 68.94% / 0.345 | -0.94pp |
| NOVA-VAD-frame-v2 | 78.99% / 0.428 | 78.88% / 0.438 | -0.11pp |
| Silero VAD | 85.24% / 0.565 | 84.06% / 0.522 | -1.18pp |
| Pyannote VAD | 83.81% / 0.529 | 84.54% / 0.541 | +0.73pp |
| SpeechBrain VAD | 73.53% / 0.442 | 71.21% / 0.390 | -2.32pp |

**Overall accuracy moved by less than 5pp for every system** — the
headline numbers were directionally trustworthy even before this fix.
**The real damage was in the per-condition breakdown**, which is where the
measurement was actually unreliable:

| Model | Condition | test (orig) | test_v2 | Δ |
|---|---|---|---|---|
| NOVA-VAD-frame-v1 | 0dB | 55.37% | 63.03% | +7.66pp |
| NOVA-VAD-frame-v1 | -5dB | 63.62% | 57.73% | -5.89pp |
| NOVA-VAD-frame-v2 | 0dB | 78.78% | 72.63% | -6.15pp |
| WebRTC VAD | 10dB | 38.62% | 50.19% | +11.57pp |

Full tables: `reports/per_scene_test_original_full_metrics.json` and
`reports/per_scene_test_v2_full_metrics.json`.

**Per-condition curves are now much closer to monotonic** (decreasing
accuracy as SNR drops), which is what real model behavior should look like
and the original 8-file test set did not show:

| Model | clean | 10dB | 0dB | -5dB | Monotonic? |
|---|---|---|---|---|---|
| NOVA-VAD-frame-v1 (test_v2) | 85.06 | 69.95 | 63.03 | 57.73 | **Yes** (was not, on original) |
| NOVA-VAD-frame-v2 (test_v2) | 87.86 | 81.90 | 72.63 | 73.14 | Nearly (0dB/-5dB now a near-tie, not a dramatic reversal) |
| WebRTC VAD (test_v2) | 89.93 | 50.19 | 39.54 | 39.33 | **Yes** (was not, on original) |

### Step 4: Confidence intervals — did they tighten? Yes, for every system.

| Model | CI width (original, 40 scenes) | CI width (test_v2, 100 scenes) | Tightened by |
|---|---|---|---|
| NOVA-VAD | 11.25pp | 7.18pp | 36% |
| WebRTC VAD | 17.19pp | 10.99pp | 36% |
| NOVA-VAD-frame-v1 | 15.33pp | 9.63pp | 37% |
| NOVA-VAD-frame-v2 | 10.69pp | 6.29pp | **41%** |
| Silero VAD | 3.37pp | 2.56pp | 24% |
| Pyannote VAD | 7.41pp | 3.68pp | **50%** |
| SpeechBrain VAD | 7.43pp | 5.37pp | 28% |

Every single system's CI tightened. Stated plainly since the instructions
required it either way: **the fix worked as intended.**

**test_v2 is now the locked benchmark for everything below this line.**
`data/scenes/test/` (original) remains as historical record only.

---

## Part 2 — Resolving the two open flags

### Flag 1: Clean-audio regression (v1 90.6% vs v2 89.5%, -1.1pp on the original 10-scene clean set)

Per the round-2 instruction: check if it persists on more clean scenes
before doing anything more expensive.

| | v1 | v2 | Δ |
|---|---|---|---|
| Original clean (10 scenes, 9 noise files) | 90.61% | 89.53% | -1.08pp |
| test_v2 clean (25 scenes, 25 noise files) | 85.06% | **87.86%** | **+2.80pp** |

**It doesn't persist — it reverses.** On a 2.5x larger, far more
noise-diverse clean sample, v2 is *ahead* of v1, not behind. This is strong
evidence the round-1 finding was small-sample noise from only 10 scenes,
not a real cost of v2's changes. Per the round-2 instruction's own
conditional ("if it persists... diagnose... ablate"), since it does not
persist, the expensive 5-model feature-ablation retrain was **not run** —
flagging this decision explicitly rather than silently skipping it. Happy
to still run the ablation if further confirmation is wanted, but the
evidence available is already fairly clean.

### Flag 2: Precision/recall trade-off — full threshold curve

Swept 84 (T_on, T_off) combinations on `data/scenes/val` (never test or
test_v2). Full curve: `reports/threshold_sweep_v2.json`. Round-1's chosen
point (T_on=0.55, T_off=0.45) reproduced exactly (F1=62.14%), confirming
the earlier tuning run. The sweep's actual F1-argmax is marginally higher:

| Point | Acc% | Prec% | Rec% | F1% | MCC |
|---|---|---|---|---|---|
| (0.45, 0.40) — sweep F1-argmax | 80.79 | 63.11 | 62.49 | **62.80** | 0.499 |
| (0.55, 0.45) — round-1 deployed | 82.62 | 71.42 | 55.00 | 62.14 | 0.518 |

+0.66pp F1 is not "clearly better" on a 40-scene validation set — **not
switching the deployed threshold on this basis**, per the explicit
instruction not to move something just to have moved it.

**Stated assumption for the recommendation:** NOVA-VAD is the first-stage
gate in front of NOVA Verify's (not-yet-built) authenticity/anti-spoofing
model — its job is to hand off "is there speech here" to a more expensive
downstream analysis. In that role, a missed speech segment (false
negative) means that audio is never analyzed at all — a silent safety gap.
A false positive (flagging noise as speech) means the downstream model
runs on garbage input, which its own quality/OOD gating (per the
NOVA-Verify engineering plan) should catch and abstain on — a wasted
compute cycle, not a safety gap. **Under this assumption, recall matters
more than precision for this specific stage.** This is my inference from
the product context, not a specification given directly for this
question — stated explicitly as required.

Recall-weighted (F2) candidate from the same sweep:

| Point | Acc% | Prec% | Rec% | F1% | F2% | MCC |
|---|---|---|---|---|---|---|
| (0.55, 0.45) — round-1 deployed | 82.62 | 71.42 | 55.00 | 62.14 | 57.65 | 0.518 |
| (0.40, 0.35) — recall-priority candidate | 78.41 | 56.92 | 68.98 | 62.37 | 66.18 | 0.479 |

**Recommendation:** if the recall-priority assumption above matches how
NOVA Verify actually intends to use the VAD stage, (0.40, 0.35) is a
reasonable alternative — real recall gain (+14pp) for a moderate accuracy/
MCC cost (-4.2pp / -0.04). **Not deploying this as the new default in this
report** — it's a recommendation contingent on confirming the product
assumption above, not something the evidence alone makes an obvious
universal choice. The currently-registered v2 threshold (0.55/0.45) stays
as the shipped default until that's confirmed.

---

## Part 3 — Continuing to close the gap

### Error-profile comparison (test_v2, already-computed per-scene results, no new test contact)

| Comparison | Shared difficulty | Only NOVA-VAD-v2 weak | Only baseline weak | Shared strength | Per-scene correlation |
|---|---|---|---|---|---|
| v2 vs Silero | 37% | 13% | 13% | 37% | 0.387 (partially shared) |
| v2 vs Pyannote | 32% | 18% | 18% | 32% | **0.253 (mostly independent)** |

36% of scenes (Pyannote comparison) split — NOVA-VAD-v2 struggles where
Pyannote doesn't, or vice versa, roughly equally often. Low correlation
(0.25-0.39, not close to 1.0) suggests genuinely different failure modes
rather than "NOVA-VAD is just a strictly worse version of the same thing"
— consistent with an ensembling or feature-borrowing opportunity being
real, though building that ensemble wasn't attempted in this round (scope).

### SNR-conditional feature weighting (gating) — deprioritized, with reasoning

Considered but **not implemented**: a mixture-of-experts-style gate that
switches feature reliance based on estimated SNR. Reasoning for
deprioritizing: RF/GBT trees already perform conditional feature selection
natively (a tree can split on an energy feature first, then decide whether
to trust periodicity vs. spectral features in the resulting leaf) — an
explicit gating layer would likely be largely redundant with what the
existing ensemble already does implicitly, for a large implementation and
overfitting-risk cost. Flagging this as a deliberate scope decision, not a
silent omission.

### Targeted training data at the weak bands (0dB, -5dB) — attempted, NEGATIVE RESULT

150 additional scenes at 0dB/-5dB only (train-only noise, zero overlap
with test_v2 — confirmed), retrained as NOVA-VAD-frame-v3, tuned on val,
evaluated once on test_v2.

**Regressed on every condition, including the two it targeted:**

| Condition | v2 | v3 | Δ |
|---|---|---|---|
| clean | 87.86% | 83.37% | -4.49pp |
| 10dB | 81.90% | 73.22% | -8.68pp |
| 0dB | 72.63% | 66.04% | -6.59pp |
| -5dB | 73.14% | 63.70% | -9.44pp |
| Overall MCC | 0.4379 | 0.3713 | -0.0666 |

Likely mechanism (hypothesis, not proven by a controlled ablation): the
added scenes shifted training composition from a balanced 25/25/25/25% to
~18/18/32/32%, plausibly biasing the model's global calibration toward
over-predicting speech everywhere (recall rose, precision collapsed
universally) rather than selectively improving the targeted bands.

**NOVA-VAD-frame-v3 is not deployed. NOVA-VAD-frame-v2 remains the current
best model.** Full model card with the negative result kept for the
record: `models/registry/nova-vad-frame-v3/model_card.md`. Recommended
(not executed) next approach: sample re-weighting within a balanced scene
mix, rather than skewed oversampling by scene count.

---

## Integrity confirmation

- **Part 1 fully completed and locked before any Part 2/3 model evaluation
  touched test_v2** — v2's threshold sweep and v3's training both happened
  entirely on train/val; test_v2 was read exactly twice total: once in
  Part 1 Step 3 (scoring the 4 already-frozen systems from round 1, no
  new training involved) and once in Part 3 for v3's single evaluation.
- **No leakage.** Verified via set-intersection (all empty) between
  test_v2's noise/speech source files and every train-side split
  (train, train2, train3, val) — shown in each generation step above.
- **No threshold fitting on test.** Both v2's original threshold and v3's
  new threshold were selected entirely on `data/scenes/val`.
- **No cherry-picked runs.** v3's regression is reported as the single run
  it was — not discarded, not re-run with a different seed to look for a
  better outcome.
- **No silent metric redefinition.** All numbers in this report use
  `scripts/frame_benchmark.py`'s original `confusion`/
  `metrics_from_confusion`/`cluster_bootstrap_ci` functions, imported
  directly wherever a metric was computed (see
  `scripts/report_from_per_scene.py`, `scripts/evaluate_frame_vad_v3.py`).
- **CI methodology unchanged** — same cluster-bootstrap-over-scenes
  approach, same N_BOOTSTRAP=2000, same seed, for both the original and
  expanded test sets.
- **Reproduction commands** (exact order):
  ```
  # Part 1
  python3 -m scripts.compute_per_scene_results data/scenes/test reports/per_scene_test_original.json
  python3 -m scripts.lono_analysis reports/per_scene_test_original.json
  python3 -m scripts.generate_test_v2                      # seed=45, test-only noise
  python3 -m scripts.compute_per_scene_results data/scenes/test_v2 reports/per_scene_test_v2.json
  python3 -m scripts.report_from_per_scene reports/per_scene_test_v2.json

  # Part 2
  python3 -m scripts.threshold_sweep_v2               # val only

  # Part 3
  python3 -m scripts.error_profile_comparison reports/per_scene_test_v2.json
  python3 -m scripts.generate_train3_targeted          # seed=46, train-only noise
  python3 -m scripts.train_frame_vad_v3
  python3 -m scripts.tune_frame_vad_v3                 # val only
  python3 -m scripts.evaluate_frame_vad_v3             # single test_v2 run
  ```

## Bottom line

The benchmark was genuinely unreliable at the per-condition level (up to
11.9pp swing from a single noise file, affecting all 7 systems), now fixed
with 3x the noise diversity and CIs tightened 24-50% across every system.
Both open flags from round 1 resolved: the clean-audio "regression" was
sampling noise (reverses on more data), and the precision/recall
trade-off is now backed by a full curve with an explicit, stated product
assumption rather than a single blind operating point. The one new thing
attempted to close the noise gap further — targeted training data — failed
outright and is reported as a failure. **Current best model remains
NOVA-VAD-frame-v2** (test_v2: 78.88% accuracy, MCC 0.438, precision 56.2%,
recall 59.5%), still behind Pyannote (MCC 0.541) and Silero (MCC 0.522),
roughly tied with SpeechBrain (MCC 0.390) — now on a benchmark trustworthy
enough that these comparisons hold up under scrutiny at the per-condition
level, not just in aggregate.
