# Phase A decision: keep, reduce, fine-tune, or replace NOVA-VAD?

**Date:** 2026-07-23
**Status:** Phase A release-gate checklist assessed below. This document is
the explicit "keep/reduce/fine-tune/replace" deliverable the checklist
requires (plan Section 7.9, last bullet) — it did not exist as a
standalone document before this round; the reasoning was scattered across
`decision_v1.md`–`v4.md`. This consolidates it and adds the round-3
ensembling finding.
**This document does not authorize starting Phase B.** Per the explicit
instruction for this round, that sign-off is the project owner's decision
after reviewing this report and the evidence package
(`reports/evidence_package_index.md`) — not something to infer from a
clean checklist.

## The full trajectory, in numbers (all on frame-level 10ms-grid scoring)

| Model | Test set | Accuracy | MCC | Precision | Recall |
|---|---|---|---|---|---|
| NOVA-VAD v0 (whole-file features, misapplied to frames) | test (orig) | 33.3% | **-0.28** | 14.6% | 35.6% |
| NOVA-VAD-frame-v1 (true frame labels, causal features) | test (orig) | 69.9% | 0.34 | 42.8% | 68.9% |
| NOVA-VAD-frame-v2 (+ noise-robust features, 4x data) | test_v2 | 78.9% | 0.44 | 56.2% | 59.5% |
| NOVA-VAD-frame-v3 (+ targeted low-SNR data) | test_v2 | 71.6% | 0.37 | 44.6% | 69.9% |
| **NOVA-VAD-frame-v3: REGRESSED, not deployed** | | | | | |
| Logistic ensemble (v2 + Silero + Pyannote) | test_v2 | **85.0%** | **0.56** | 76.7% | 55.2% |
| — for reference — | | | | | |
| WebRTC VAD (**fair: aggressiveness=2**, was mode 3 through round 4 — see round 5 update below) | test_v2 | 52.2% | **0.24** | 31.9% | 85.5% |
| SpeechBrain VAD (default) | test_v2 | 71.2% | 0.39 | 44.5% | 74.5% |
| Pyannote VAD (default) | test_v2 | 84.5% | 0.54 | 79.0% | 49.7% |
| Silero VAD (default) | test_v2 | 84.1% | 0.52 | 85.1% | 41.8% |

## Decision: fine-tune-in-place is exhausted; the productive path is combination, not further standalone scaling

**NOVA-VAD as a standalone model should be considered feature/architecture-
complete for this phase.** Three consecutive standalone iterations
(v0→v1→v2→v3) show diminishing and then negative returns:
- v0→v1: fixing a fundamental task mismatch (whole-file features on a
  frame task) — large, real gain (MCC -0.28→0.34).
- v1→v2: noise-robust features + 4x data — real but smaller gain
  (0.34→0.44).
- v2→v3: targeted low-SNR data — **regression** (0.44→0.37), on every
  condition including the ones targeted. The standalone architecture did
  not have more headroom to extract from more of the same kind of data;
  it got worse.

This pattern (large gain fixing a real bug → smaller gain from genuine
improvement → negative return from more of the same) is a reasonably
strong signal that the 62-feature causal RF+GBT architecture is near its
ceiling for this task, not that it's one more dataset tweak away from
closing the gap to Pyannote/Silero (whose MCCs of 0.52-0.54 remain 0.08-0.10
above v2's 0.44).

**The combination result changes the actionable conclusion.** The
per-scene correlation finding from round 2 (0.25-0.39 with
Pyannote/Silero — different failure modes, not a strict subset) was
tested directly this round: a 3-weight logistic combination of
NOVA-VAD-v2 + Silero + Pyannote frame probabilities, fit on `val` only
and evaluated once on `test_v2`, reached **MCC 0.56, beating every
individual system tested including Pyannote and Silero standalone.** This
is a real, non-cherry-picked, out-of-sample result (frozen weights from
val, single test_v2 evaluation) — full methodology and numbers in
`reports/decision_v5.md` (round 3).

**Recommendation: NOVA-VAD-frame-v2 is kept, not replaced, not further
scaled standalone.** Its role changes from "the detector" to "one signal
in a combination" where it demonstrably adds value (the combination beats
Silero+Pyannote-style baselines alone, and NOVA-VAD-v2 alone beats WebRTC
outright). Whether to actually ship the 3-system combination is a product
decision, not a modeling one — see the explicit trade-off below.

## The trade-off the combination result creates (flagged, not resolved — product decision)

Running the combination means running NOVA-VAD-v2 **and** Silero **and**
Pyannote together, not NOVA-VAD alone. Concretely:

- **Compute cost:** roughly 3x the inference cost of NOVA-VAD alone per
  file (NOVA-VAD ~6-7s, Pyannote ~6s, Silero ~0.9s on the hardware used
  for this benchmark — see `reports/frame_level_benchmark_v1.md` timing
  columns from earlier rounds).
- **Dependency/licensing:** Pyannote's `segmentation-3.0` model is gated
  on Hugging Face (requires `HF_TOKEN` in this project's setup) and its
  license terms for commercial redistribution have not been checked in
  any round of this work — this needs review before any product decision
  that depends on shipping Pyannote.
- **On-device feasibility (plan Phase E):** this directly contradicts the
  original README's positioning of NOVA-VAD as "scikit-learn only,
  lightweight" vs. "Silero: PyTorch required 200MB+" — the combination
  requires shipping *two* PyTorch-based neural models plus NOVA-VAD's own
  sklearn ensemble, which is a materially heavier on-device footprint than
  any standalone option tested.

This report does not recommend for or against shipping the combination —
that decision needs the compute/licensing/on-device answers above, which
are product and legal questions, not modeling questions.

## Phase A release-gate checklist (plan Section 7.9) — item by item

| Item | Status | Evidence |
|---|---|---|
| Locked train/development/test manifests with grouped sources | ✅ Done | `reports/data_manifest_and_leakage_audit.md` (this round — consolidates and re-verifies checks previously only shown in chat output across rounds 2-3) |
| Frame-level ground truth and a common scoring grid | ✅ Done | `frame_labels_10ms` in every scene manifest; all 7 systems scored on this same 10ms grid via `scripts/frame_vad_adapters.py` + `scripts/frame_benchmark.py` |
| Raw, degraded, hard-negative, and actual-transmission results | 🟡 **Partial — narrower gap than before, still not closed** (updated round 4, see below) | Layer 1 (clean) ✅, Layer 2 (noise/SNR) ✅. Layer 3: simulated codec (G.711 A-law/mu-law, Opus 24kbps) ✅ **now tested** (`reports/decision_v6.md`); actual RTC/transmission (packet loss, jitter, real network conditions) still ❌ not tested — a materially different, larger undertaking than simulated codec round-trip. Layer 4: 1 of ~8 hard-negative categories (DTMF) ✅ now tested; 7 remain (laughter, coughing, crying, singing, TV, hold music, breathing, overlapping speech) ❌ explicitly deferred — need real audio sources this project doesn't have yet. |
| Pinned and fairly tuned baselines | 🟡 **Partial — WebRTC sub-gap closed, broader gap remains** (updated round 4) | Versions pinned ✅ (unchanged). WebRTC now tested at all 4 aggressiveness modes ✅ (`reports/decision_v6.md`) — and the previously-reported mode 3 turned out to be worst-by-MCC of the four (0.2236 vs. best mode 2's 0.2424), meaning every prior round's WebRTC number in the comparison tables was real but not WebRTC's best achievable result. **Silero/Pyannote/SpeechBrain are still only run at out-of-box default settings, not tuned on `dev`/`val`** — this remains open, and finding a real, non-trivial gap in WebRTC's "obvious" default this round is a reason to take that remaining gap more seriously, not less. |
| Confidence intervals and per-condition failure analysis | ✅ Done | Cluster-bootstrap CIs (`scripts/frame_benchmark.py:cluster_bootstrap_ci`) and per-condition breakdowns throughout `decision_v3.md`/`v4.md`; round 2's LONO analysis additionally validated the per-condition numbers are now trustworthy (`reports/per_scene_test_original_lono.json`) |
| A documented decision to keep, reduce, fine-tune, or replace the current ensemble | ✅ Done (this document) | Decision above: keep NOVA-VAD-frame-v2, combination is the productive path, standalone scaling is not |

**Honest summary: 4 of 6 items fully done, 2 partial with specific, named gaps** (transmission/hard-negative test coverage; baseline tuning fairness/WebRTC mode coverage). Neither partial item invalidates the numbers reported so far — they define the boundary of what those numbers can currently claim, which is: performance under clean and additive-noise conditions on the tested noise/music sources, not performance under codec degradation, real transmission, or non-music hard negatives, and not a claim that the baselines were pushed to their best achievable operating points.

## What this document does NOT decide

Per the explicit instruction for this round: this document does not start
or authorize Phase B (anti-spoofing) work, regardless of how the checklist
above reads. That is Gate 1 sign-off, made by the project owner after
reviewing this report and the evidence package.

---

## Round 4 update (2026-07-23): checklist re-assessed, gate is NOT fully closeable yet

Full methodology and numbers: `reports/decision_v6.md`. Honest verdict,
stated plainly per this round's explicit instruction not to treat
"attempted" as "closed":

**Real, verified progress on both previously-partial items:**
- Codec degradation (G.711 A-law/mu-law, Opus) tested for the first time
  — closes the "zero codec testing" gap specifically. Finding: codec
  degradation barely hurts any system tested, NOVA-VAD-frame-v2 included
  (MCC -0.025 vs. clean, far smaller than the -0.22-ish drop seen under
  additive noise).
- One hard-negative category (DTMF) tested — NOVA-VAD-frame-v2 handles it
  well (6.67% false-positive rate, better than v1's 17.0%, competitive
  with WebRTC/Pyannote, though behind Silero's 0% and SpeechBrain's 0.81%).
- WebRTC now tested at all 4 aggressiveness modes — the "only 1 mode
  tested" sub-gap is fully closed. Real finding: the previously-reported
  mode (3) was not WebRTC's best (mode 2 is, by MCC).

**What remains open, not closed by this round:**
1. Actual RTC/transmission testing (packet loss, jitter, real network
   conditions) — not started, and explicitly not conflated with the
   simulated-codec work that was done.
2. 7 of 8 hard-negative categories (laughter, coughing, crying, singing,
   TV, hold music, breathing, overlapping speech) — not started, deferred
   because they need real audio sources this project doesn't currently
   have and rushing that sourcing risks licensing/leakage problems.
3. Silero/Pyannote/SpeechBrain threshold tuning on `dev`/`val` — not
   started. Only WebRTC's operating-point selection was addressed. The
   WebRTC finding (a real, non-trivial gap hiding in an "obvious" default)
   is a reason to weight this remaining item as more important, not less.

**Verdict: Phase A's release-gate checklist is not fully met.** Two items
moved from "wide open" to "meaningfully narrowed, real gaps clearly
bounded" — that is genuine progress, not just an attempt, but it is not
the same as closure. The items marked ✅ in the table above (manifests,
scoring grid, CIs/per-condition analysis, the keep/reduce/replace
decision itself) remain solid. The two 🟡 items are real, named,
partially-closed gaps — not yet ✅. This document continues to not
authorize Phase B; the remaining gap list above is offered as the
concrete input for the project owner's Gate 1 decision, not a
recommendation to proceed.

---

## Round 5 update (2026-07-23): pushed to close all three remaining gaps — one closes, two advance substantially but remain open

Full methodology and numbers: `reports/decision_v7.md`,
`reports/master_comparison_table.md`. Per the explicit instruction for
this round, reporting exactly what closed and what didn't — a partial
close is a fine outcome to report, a false "done" is not.

### Item 0 (WebRTC mode fix) — ✅ fully closed
Retroactively fixed in the master comparison table. Checked explicitly,
not assumed: no keep/reduce/replace or ensembling conclusion changes.
One real nuance surfaced and disclosed: WebRTC's best mode differs by
condition (mode 3 actually beats NOVA-VAD-frame-v2 on clean audio
specifically, even though mode 2 is the fair aggregate pick).

### Item 1 (real RTC transmission, Plan Section 7.3 Layer 3) — 🟡 substantial, real progress; NOT fully closed
A genuine `RTCPeerConnection` media path was built and verified working
end-to-end this round — real ICE negotiation against a real STUN server
(confirmed via a real public IP in the gathered candidates), real Opus
encoding via the browser's native WebRTC audio pipeline, real RTP
transport (confirmed via `getStats()` packet counts), captured and scored
against ground truth for **5 scenes**. Two real implementation bugs were
caught and fixed before trusting any result (`BroadcastChannel` couldn't
clone `RTCSessionDescription`/`RTCIceCandidate` objects; `MediaRecorder`
needed a playback-element sink to reliably capture a remote track's
data), plus a false alignment match caught and corrected via the same
verify-before-trust discipline used for round 4's codec bugs.

**What's genuinely closed:** the claim "actual transmission has never been
tested, only simulated codec transforms" is no longer true — it has been
tested, with full sender/receiver/route/platform/capture documentation
sufficient for independent verification.

**What's still open, disclosed explicitly, not glossed over:**
- Scale: 5 scenes is a pilot, not the ~25-scene statistical power the
  codec test achieved. CIs here are wide (up to 34.5pp).
- Network conditions: this was a real but easy path (localhost-resolved
  host candidates, sub-millisecond RTT). Packet loss, jitter, and
  congestion — the conditions where transmission testing matters most —
  remain untested.
- Device scope: both peers were tabs in the same browser process, not
  genuinely different devices. A concrete next-step design is in
  `reports/decision_v7.md` (two VoIP clients on separate machines with
  virtual audio routing) but was not attempted this round.

### Item 2 (hard negatives, Plan Section 7.2 Layer 4) — 🟡 substantial, real progress; NOT fully closed
3 more categories added (overlapping speech, breathing, hold music) with
full source/license documentation for each — bringing the total to 4 of
~8 listed categories tested (DTMF from round 4 + these 3).
NOVA-VAD-frame-v2 performed well: 0% false-positive rate on both pure
hard-negative categories (matching Silero/Pyannote/SpeechBrain), and
72.62% recall on overlapping speech (behind SpeechBrain's 89.15% but
ahead of v1/Silero/Pyannote).

**What's still open:** laughter, coughing, crying, singing (need
authentic recordings this environment has no safe/verified way to source
this round) and television (deferred specifically for licensing risk, as
anticipated). 4-5 categories genuinely untested, not silently ignored —
named explicitly above and in `reports/decision_v7.md`.

### Item 3 (fair baseline tuning, Plan Section 7.4) — ✅ closed, within what each library's interface actually allows
All three remaining baselines' public interfaces checked directly for a
tunable threshold. Silero and SpeechBrain both have one and were tuned on
`val`. **Pyannote's onset/offset turned out to not be tunable at all for
`segmentation-3.0`** (a powerset model) — confirmed by direct
experimentation (`instantiate()` raises `ValueError: parameter 'onset'
does not exist`), not assumed from documentation; only its post-processing
durations were tunable, and those were tuned. Single test_v2 evaluation:
**Silero's fair-tuned threshold scored worse than its default** (MCC
0.5096 vs 0.5218) — reported plainly as a real, non-flattering result,
not suppressed or re-run for a better number. No ranking conclusions
changed. This item is as closed as it can be given each library's actual
capabilities — there is no further tunable surface being left unexplored.

### Updated overall verdict

**Phase A's release gate still does not fully close.** One of the three
items pushed this round (fair baseline tuning) is now genuinely done.
The other two (real transmission, hard negatives) both moved from "not
attempted" to "real, verified, honestly-scoped partial progress with a
named, bounded remainder" — which is exactly what was asked for when full
closure wasn't guaranteed in one round. The concrete remainder, for
whoever picks this up next:
1. Scale the real-RTC pipeline (already built and working) to more scenes,
   and add adverse network conditions and/or genuine cross-device
   transmission.
2. Source authentic laughter/coughing/crying/singing audio (needs a
   verified clean-license path this round didn't have) and decide whether
   television is worth pursuing given its IP risk.

This document still does not authorize Phase B.
