# Master comparison table — current, authoritative, updated as of round 5

**This supersedes `reports/frame_level_benchmark_v1.md`/`.json` as the
reference table.** That file was discovered in round 4 to still hold the
*original* 40-scene `test` set's numbers for every system except v3 —
it was never regenerated against `test_v2` despite the filename. It is
left as-is (not deleted, not silently corrected) as a historical artifact;
this document is the current source of truth going forward, and will
continue to be updated as later rounds add fairly-tuned baselines, more
hard-negative categories, etc.

All numbers below: `test_v2` (100 scenes, 25 unique noise files/condition,
locked since round 2), same scoring code throughout
(`scripts/frame_benchmark.py`'s `confusion`/`metrics_from_confusion`/
`cluster_bootstrap_ci`, reused not reimplemented).

## Overall (pooled across all conditions)

| System | Accuracy | 95% CI | Precision | Recall | MCC |
|---|---|---|---|---|---|
| NOVA-VAD (v0, whole-file features on frames) | 38.03% | [34.56, 41.74] | 15.86% | 35.96% | -0.2186 |
| **WebRTC VAD — mode 2 (fair)** | 52.15% | — | 31.93% | 85.50% | **0.2424** |
| WebRTC VAD — mode 3 (used in every round 1-4 table, NOT the fair pick) | 54.75% | [49.35, 60.34] | 32.30% | 78.52% | 0.2236 |
| SpeechBrain VAD (default = fair; val grid found no better point) | 71.21% | [68.44, 73.81] | 44.51% | 74.47% | 0.3898 |
| NOVA-VAD-frame-v1 | 68.94% | [63.98, 73.61] | 41.89% | 71.55% | 0.3452 |
| NOVA-VAD-frame-v2 | 78.88% | [75.52, 81.81] | 56.22% | 59.54% | 0.4379 |
| Silero VAD — threshold=0.5 (default) | 84.06% | [82.74, 85.30] | 85.07% | 41.82% | **0.5218** |
| Silero VAD — threshold=0.3 (**fair, val-tuned — scores worse here, see round 5 Item 3**) | 83.43% | [81.50, 85.17] | 73.46% | 49.91% | 0.5096 |
| Pyannote VAD — default (onset/offset not tunable for this model, min_dur=0.0/0.0) | 84.54% | [82.58, 86.26] | 78.99% | 49.66% | 0.5414 |
| Pyannote VAD — **fair (min_dur=0.0/0.25)** | 84.56% | [82.58, 86.28] | 78.94% | 49.80% | 0.5420 |
| **Logistic ensemble (v2 + Silero + Pyannote)** | **85.01%** | [83.21, 86.60] | 76.67% | 55.17% | **0.5620** |

**Round 5 Item 3 update:** all three baselines' public interfaces checked
for a tunable threshold (Pyannote's onset/offset turned out to NOT be
tunable for this powerset model — confirmed by direct experimentation,
`instantiate()` raises `ValueError`). Fair, val-tuned numbers now shown
alongside defaults. Notably, Silero's fair-tuned threshold scores *worse*
on test_v2 than its default (0.5096 vs 0.5218) — reported honestly, not
suppressed; full reasoning in `reports/decision_v7.md` Item 3. No ranking
changes as a result — every conclusion in this table holds under either
number. The ensemble is unaffected by any of this (it uses Silero's raw
probabilities directly, not `get_speech_timestamps()`'s threshold param).

## Item 0 (round 5): retroactive WebRTC mode fix

Round 4 discovered WebRTC mode 3 (used in every table through round 4)
was not its best mode by aggregate MCC — mode 2 is (0.2424 vs 0.2236).
That finding was documented but never actually swapped into a live
comparison table until now.

**Does this change any conclusion? No — checked explicitly, not assumed.**

- Keep/reduce/replace decision (`reports/phase_a_decision.md`): unaffected.
  WebRTC was never a contender for "which model to keep" in any round —
  it was always the weakest system in aggregate MCC terms. Mode 2's 0.2424
  is still far below NOVA-VAD-v2 (0.4379), SpeechBrain (0.3898), Pyannote
  (0.5414), Silero (0.5218), and the ensemble (0.5620). Ordering unchanged.
- Ensembling result (`reports/decision_v5.md`): unaffected. The ensemble
  doesn't use WebRTC as an input (only NOVA-VAD-v2, Silero, Pyannote) —
  WebRTC's mode has zero mechanical effect on that result.
- **NOVA-VAD-v2-beats-WebRTC framing: gets slightly weaker, not stronger,
  under the fair comparison** — the MCC gap shrinks from 0.4379-0.2236=
  0.2143 (mode 3) to 0.4379-0.2424=0.1955 (mode 2). Still a clear, real
  gap in NOVA-VAD-v2's favor overall, just marginally smaller.

**A genuine, disclosed nuance the fair-mode exercise surfaced**: per-
condition, WebRTC's *best* mode is not the same for every condition.
Mode 3 (not the aggregate-best mode 2) has the highest MCC of any WebRTC
mode specifically on **clean** audio (0.7112) — which is *higher* than
NOVA-VAD-frame-v2's own clean MCC (0.6559). Picking one fair operating
point per the plan's instruction (not cherry-picking per-condition) means
mode 2 is used above, but it would be inaccurate to claim "NOVA-VAD-v2
beats WebRTC in every condition" — it doesn't, on clean audio, against
WebRTC's own best-for-that-condition setting. Full per-mode, per-condition
table: `reports/webrtc_all_modes_test_v2.json`.

| WebRTC mode | Overall Acc | Overall MCC | Clean MCC | 10dB MCC | 0dB MCC | -5dB MCC |
|---|---|---|---|---|---|---|
| 0 | 46.93% | 0.2319 | 0.5601 | 0.1576 | 0.1279 | 0.0932 |
| 1 | 48.66% | 0.2353 | 0.5868 | 0.1515 | 0.1245 | 0.1026 |
| **2 (fair pick, best aggregate)** | 52.15% | **0.2424** | 0.6387 | 0.1767 | 0.1280 | 0.1032 |
| 3 (used through round 4) | 54.75% | 0.2236 | **0.7112** | **0.1806** | 0.1183 | 0.0991 |
