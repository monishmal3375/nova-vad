# Model Card — NOVA-VAD-frame-v3 — NOT DEPLOYED, NEGATIVE RESULT

## Do not use this model. It regressed vs v2 on every condition.

This experiment (round 2, Part 3.2) added 150 targeted training scenes at
the two SNR bands `test_v2` showed were genuinely weak (0dB, -5dB), on top
of v2's 400-scene train+train2 pool. Hypothesis was that more targeted
training data would improve the weak bands without hurting the rest.

**The hypothesis was wrong.** Single evaluation on the locked `test_v2` set
(never touched before this run):

| Condition | v2 (deployed) | v3 (this, not deployed) | Change |
|---|---|---|---|
| clean | 87.86% | 83.37% | **-4.49pp** |
| 10dB | 81.90% | 73.22% | **-8.68pp** |
| 0dB | 72.63% | 66.04% | **-6.59pp** |
| -5dB | 73.14% | 63.70% | **-9.44pp** |
| **Overall MCC** | 0.4379 | 0.3713 | **-0.0666** |
| **Overall precision** | 56.22% | 44.62% | **-11.60pp** |

Regressed everywhere, including the two conditions it targeted. This is a
genuine negative result, kept in the registry (not deleted) as a record of
what was tried and why it didn't work — see `reports/decision_v4.md` for
the full round-2 writeup.

## Likely mechanism (hypothesis, not proven via a controlled ablation)

Adding 150 scenes only at 0dB/-5dB shifted the training composition from a
balanced 25/25/25/25% (clean/10dB/0dB/-5dB) to roughly 18/18/32/32%. This
plausibly biased the ensemble's global probability calibration toward
predicting speech more often across the board (recall rose 59.5%→69.9%
overall, but precision collapsed 56.2%→44.6% — including on clean audio,
where the model wasn't even given new targeted data). A skewed training
mix appears to have degraded overall calibration rather than selectively
improving the targeted bands.

## Recommended next step, not executed in this round

If pursuing targeted SNR-band improvement again: use sample re-weighting
(keep the original balanced scene mix, but up-weight 0dB/-5dB examples in
the loss/training objective) rather than pure oversampling by scene count,
which would add signal at the weak bands without distorting the overall
training distribution the way this attempt did.
