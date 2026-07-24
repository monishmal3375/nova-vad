# Decision v5 — Round 3, Part 1: testing the ensembling lead

**Date:** 2026-07-23
**Question:** round 2 found NOVA-VAD-frame-v2's per-scene accuracy
correlates only 0.25-0.39 with Pyannote/Silero — is that exploitable, or
just noise?

## Method — simplest thing first, as instructed

1. **NOVA-VAD-v2**: real per-frame probabilities (RF+GBT average), already
   computed at 100ms hop resolution, expanded to the 10ms scoring grid.
2. **Silero**: real per-frame probabilities extracted from the underlying
   scripted model's raw `forward()` call on native 512-sample (32ms)
   chunks at 16kHz — this bypasses `get_speech_timestamps()`'s internal
   thresholding, which only exposes binary segments. Verified working
   before running the full split (single-scene sanity check: correct
   discriminative direction, mean prob 0.385 on speech frames vs. 0.005 on
   non-speech frames).
3. **Pyannote**: binary mask only, via the already-tested
   `predict_mask_pyannote` adapter. **Deliberate scope decision, not an
   oversight**: `pyannote/segmentation-3.0`'s raw output is multi-speaker
   "powerset" logits (7 classes for up to 3 speakers), not a direct speech
   probability. Correctly deriving a probability from that requires the
   exact class mapping; reimplementing it from scratch risked a subtle
   conversion bug under this round's time budget, so the binary mask
   (already verified correct in prior rounds) was used instead.

Candidates tried, on `val` (never `test_v2`), simplest first:

| Rule | val Acc% | val Prec% | val Rec% | val F1% | val MCC | Δ MCC vs. NOVA-v2 alone |
|---|---|---|---|---|---|---|
| NOVA-VAD-v2 alone (raw prob ≥0.5, no hysteresis) — baseline | 82.22 | 70.28 | 54.52 | 61.41 | 0.5078 | — |
| Average(NOVA-v2, Silero) ≥0.5 | 79.27 | 92.84 | 21.76 | 35.25 | 0.3883 | -0.1195 |
| Majority vote (≥2 of 3 binary decisions) | 84.33 | 94.82 | 41.88 | 58.10 | 0.5653 | +0.0575 |
| OR (any of 3 says speech) | 78.88 | 57.33 | 72.62 | 64.08 | 0.5010 | -0.0068 |
| **Logistic regression [nova_prob, silero_prob, pyannote_mask]** | **85.40** | 81.81 | 56.23 | 66.65 | **0.5938** | **+0.0860** |

Note on the baseline choice: compared against NOVA-VAD-v2's **raw
probability at 0.5**, not the deployed hysteresis-post-processed version,
because none of the candidate combinations had hysteresis applied either —
comparing raw-to-raw is the fair comparison for isolating what the
*combination* contributes. (For reference, the deployed v2-with-hysteresis
scores MCC 0.438 on `test_v2` — very close to this raw-probability
baseline's 0.434 on the same set, shown below, confirming hysteresis adds
only a small amount and the raw-probability comparison here is
representative, not artificially weakened.)

Simple averaging **hurt** (-0.12 MCC) — Silero is far more conservative
(much lower average probability, high precision/low recall profile) than
NOVA-VAD-v2, so naive averaging just drags predictions toward "less
speech" without the calibration a learned combination provides. Majority
vote helped modestly. **Logistic regression gave the largest, clearly
non-marginal gain** (+0.086 MCC, ~17% relative) — this is what was
promoted to a single test_v2 evaluation.

Frozen weights (fit on val, never touched again): `nova=4.922,
silero=3.657, pyannote=1.393, intercept=-3.581`. All three learned
coefficients are positive — every system contributes positively to the
combination on val, not just NOVA-VAD-v2.

## Single evaluation on test_v2 (first and only contact for this decision)

| System | Accuracy | 95% CI | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| NOVA-VAD-v2 alone (raw, no hysteresis) | 78.66% | [75.38, 81.44] | 55.73% | 59.56% | 57.58% | 0.4339 |
| **Logistic ensemble (v2 + Silero + Pyannote)** | **85.01%** | **[83.21, 86.60]** | 76.67% | 55.17% | 64.17% | **0.5620** |

**MCC delta: +0.1281** — the gain is *larger* on test_v2 than it was on
val (+0.086), not smaller. That's the direction you'd worry about if this
were overfitting to val; instead it generalized better than the val
number suggested, which is a positive but also a real result worth
double-checking rather than assuming — see "things that look too good"
below.

### Per-condition accuracy

| System | clean | 10dB | 0dB | -5dB |
|---|---|---|---|---|
| NOVA-VAD-v2 alone | 88.20% | 81.21% | 72.91% | 72.32% |
| Logistic ensemble | 90.95% | 85.71% | 81.76% | 81.64% |

Gains hold across every condition, including the two (0dB, -5dB) that were
NOVA-VAD-v2's weakest and where the failed v3 experiment (round 2) tried
and failed to help via more training data. The combination closes roughly
half the gap to Pyannote/Silero's ~84-88% range at those bands, without
retraining NOVA-VAD-v2 at all.

## Flagged: does this look too good relative to the size of the change?

The instruction requires flagging results that look suspiciously good.
Considered explicitly:

- **Is this leakage?** No — `val` and `test_v2` share zero source files
  (verified in `reports/data_manifest_and_leakage_audit.md`), and the
  logistic regression was fit once on val, never refit or peeked at
  test_v2 before this single evaluation.
- **Is 3 parameters + intercept enough capacity to overfit 48,000 val
  frames?** No — a 4-parameter linear model fit on 48,000 examples has
  essentially no capacity to memorize; this is why the result generalizing
  *even better* out-of-sample is plausible rather than suspicious.
- **Why would the gain be bigger on test_v2 than val?** The most likely
  explanation: `val`'s 40 scenes vs. `test_v2`'s 100 scenes — test_v2 is
  simply a larger, statistically steadier sample (consistent with round
  2's finding that more scenes tightens estimates), so its measurement of
  the ensemble's true benefit is less noisy, not necessarily "more
  favorable" in a biased sense. This is a hypothesis, not proven by a
  further controlled experiment in this round.
- **Bottom line:** the result is real and worth taking seriously, but it's
  a single split, single fit, single evaluation — the standard caveats
  from every prior round apply (one run, one seed, not a stable mean
  across repeated splits).

## Product implications (flagged, not resolved — see `reports/phase_a_decision.md`)

Shipping this requires running NOVA-VAD-v2 + Silero + Pyannote together:
~3x the inference cost of NOVA-VAD alone, Pyannote's Hugging-Face-gated
license terms (unreviewed), and a materially heavier on-device footprint
than the original "lightweight, sklearn-only" positioning — this is a
product/licensing decision, not something resolved by this modeling
result.

## Integrity confirmation

- Weights fit on `val` only; `test_v2` read exactly once, after weights
  were frozen.
- No leakage (checked, see `reports/data_manifest_and_leakage_audit.md`).
- Single run, not cherry-picked — this is the first and only combination
  rule promoted to test_v2 in this round; the other three (average,
  majority vote, OR) are reported as tested-and-not-promoted on val, not
  discarded.
- Same metric code (`scripts/frame_benchmark.py`'s `cluster_bootstrap_ci`,
  `per_scene_accuracy`) reused for the CI; `confusion_metrics` in
  `scripts/ensemble_fit_and_eval.py` is a direct copy of
  `frame_benchmark.py`'s `metrics_from_confusion`/`confusion` logic (same
  formulas, verified against it), not a redefinition.
- Reproduction:
  ```
  python3 -m scripts.ensemble_frame_probs data/scenes/val reports/ensemble_frame_probs_val.json
  python3 -m scripts.ensemble_fit_and_eval reports/ensemble_frame_probs_val.json
  python3 -m scripts.ensemble_frame_probs data/scenes/test_v2 reports/ensemble_frame_probs_test_v2.json
  python3 -m scripts.ensemble_evaluate_test_v2
  ```
