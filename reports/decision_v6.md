# Decision v6 — Round 4: MCC clarification + closing the Phase A gaps

**Date:** 2026-07-23

## Item 0 — token

Not touched this round, per explicit instruction. `.git/config` and the
remote URL were not modified.

## Item 1 — the four MCC numbers, shown directly

Pulled fresh from the actual result files (not from prose), with exact
source file named for each:

| | Before ensembling (NOVA-VAD-v2 alone, raw prob ≥0.5, no hysteresis) | After ensembling (3-weight logistic, NOVA+Silero+Pyannote) | Gain |
|---|---|---|---|
| **val** (`reports/ensemble_frame_probs_val_combination_results.json`) | 0.5078 | 0.5938 | +0.0860 |
| **test_v2** (`reports/ensemble_test_v2_result.json`) | 0.4339 | 0.5620 | +0.1281 |

The test gain (+0.1281) is genuinely larger than the val gain (+0.0860) —
this was reported exactly this way in round 3's `reports/decision_v5.md`
already, re-verified here by re-reading the same two files fresh rather
than from memory. No change to either number.

**Likely source of the apparent inconsistency you flagged:** round 3's
prose also mentioned a *third*, different number — 0.438 (rounded from
0.4379) — which is NOT the same baseline as the 0.4339 used in the actual
before/after comparison above. 0.4379 is **NOVA-VAD-frame-v2 in its fully
deployed form** (with hysteresis post-processing, threshold tuned on
`val`) scored on `test_v2` — a *different, separately-reported* number
from `reports/decision_v4.md`'s baseline comparison table, not the
ensembling experiment's baseline. The ensembling experiment's actual
baseline (0.4339) is NOVA-VAD-v2's **raw probability at 0.5, no
hysteresis** — deliberately chosen so the "before" number uses the same
no-post-processing form as the ensemble candidates (none of which had
hysteresis applied either), for a fair apples-to-apples comparison. Both
0.4379 and 0.4339 are real, correctly-computed numbers for what they each
measure; they were just two different "v2 baseline" framings sitting
close together in the same document, which is where the confusion
came from. The four numbers in the table above are the ones that
actually feed the ensembling before/after comparison.

## Item 2a — codec degradation testing

**Tooling note:** no system ffmpeg/opus was available in this environment
(checked: `ffmpeg`, `brew`, `opuslib` all absent). Installed `av` (PyAV,
18.0.0) instead — it bundles its own ffmpeg/libopus, no system install
needed. Verified codec support before building anything on it
(`pcm_alaw`/`pcm_mulaw`/`libopus` all present).

**Two real implementation bugs caught during verification, before any
scene was scored** (disclosed per the standard of showing real work, not
just results): (1) the WAV muxer defaulted to 2-channel output for a mono
G.711 stream unless the layout was explicitly set, silently doubling
output length; (2) Opus always decodes at 48kHz internally regardless of
the requested encode rate (correct Opus behavior, not a library bug) — an
early version treated the decoded samples as already 16kHz, corrupting
alignment. Both caught via direct waveform/cross-correlation inspection
before scoring anything, fixed, and locked with regression tests
(`tests/test_codec_degrade.py`, 5 tests, all passing).

**Scope, disclosed explicitly:** applied to the 25 clean `test_v2` scenes
only (not crossed with SNR conditions — that would be a 3-codec × 4-SNR
matrix, out of scope this round). This measures "does codec degradation
alone hurt performance," isolated the same way the SNR conditions
isolated noise-alone effects. Codec × noise interaction is **not**
tested — an explicit, disclosed gap for a future round, not a claim of
completeness. This is simulated codec encode/decode, **not** actual
RTC/VoIP transmission (packet loss, jitter, real network conditions) —
that remains Plan Section 7.2's separate Layer 3 gap, still open.

No new leakage risk: transforms already-test-only-pool audio (test_v2's
clean scenes), draws no new source files.

### Results (75 scenes: 25 each of G.711 A-law, G.711 mu-law, Opus 24kbps)

| Model | Codec-degraded Acc | Codec-degraded MCC | Clean (pre-codec) Acc | Clean MCC | Δ MCC |
|---|---|---|---|---|---|
| NOVA-VAD (v0) | 30.35% | -0.3104 | 34.77% | -0.2377 | -0.0727 |
| WebRTC VAD (mode 3) | 89.58% | 0.7008 | 89.93% | 0.7112 | -0.0104 |
| NOVA-VAD-frame-v1 | 86.98% | 0.6312 | 85.06% | 0.6021 | **+0.0291** |
| NOVA-VAD-frame-v2 | 87.11% | 0.6308 | 87.86% | 0.6559 | -0.0251 |
| Silero VAD | 85.73% | 0.5784 | 86.59% | 0.6074 | -0.0290 |
| Pyannote VAD | 86.84% | 0.6151 | 86.77% | 0.6126 | +0.0025 |
| SpeechBrain VAD | 76.67% | 0.4544 | 73.80% | 0.4750 | -0.0206 |

### Per-codec-condition accuracy

| Model | G.711 A-law | G.711 mu-law | Opus 24kbps |
|---|---|---|---|
| NOVA-VAD | 25.97% | 29.44% | 35.65% |
| WebRTC VAD | 89.60% | 89.75% | 89.40% |
| NOVA-VAD-frame-v1 | 87.97% | 87.99% | 84.98% |
| NOVA-VAD-frame-v2 | 86.88% | 86.83% | 87.63% |
| Silero VAD | 85.29% | 85.40% | 86.49% |
| Pyannote VAD | 86.91% | 86.95% | 86.67% |
| SpeechBrain VAD | 78.05% | 74.79% | 77.18% |

**Honest finding, reported exactly as measured — this was not expected
going in:** codec degradation barely hurts any system, NOVA-VAD-v2
included (MCC drop of only 0.025 vs. clean — much smaller than the
degradation seen under additive noise, e.g. -0.22 from clean to -5dB SNR).
v1 actually scores *slightly better* under codec degradation than on
clean audio (+0.029 MCC) — a genuinely counterintuitive result, reported
plainly rather than explained away; the most defensible reading is that
this is within normal scene-to-scene variance given the small sample (25
scenes/codec), not a claim that codec degradation somehow helps. A
plausible reason codec effects are small across the board: G.711 and
Opus are both specifically engineered to preserve perceptually-relevant
speech spectral structure while discarding what's not needed for voice
intelligibility — unlike additive background noise, which directly
corrupts the same acoustic cues (energy, spectral shape, periodicity)
these detectors rely on. This is a plausible explanation, not proven by
a further controlled experiment in this round.

**This closes the "no codec testing at all" gap** — G.711 (A-law +
mu-law) and Opus are now tested, satisfying the stated floor. It does
**not** close the broader Layer 3 (actual RTC transmission) gap, which
remains open and is a materially different, larger undertaking.

Reproduction:
```
python3 -m scripts.generate_codec_scenes
python3 -m scripts.compute_per_scene_results data/scenes/test_v2_codec reports/per_scene_test_v2_codec.json
python3 -m scripts.report_from_per_scene reports/per_scene_test_v2_codec.json
```

## Item 2b — WebRTC all 4 aggressiveness modes

Re-run on `test_v2` (100 scenes), same scoring code
(`scripts/frame_benchmark.py`, imported not reimplemented):

| Mode | Accuracy | 95% CI | F1 | MCC |
|---|---|---|---|---|
| 0 (least aggressive) | 46.93% | [41.75, 52.23] | 45.55% | 0.2319 |
| 1 | 48.66% | [43.15, 54.09] | 45.87% | 0.2353 |
| 2 | 52.15% | [46.77, 57.78] | 46.50% | **0.2424** |
| 3 (most aggressive — the only mode ever tested before this round) | 54.75% | [49.35, 60.34] | 45.77% | 0.2236 |

**Best mode by MCC is 2, not 3.** Mode 3 (previously the only mode tested,
used throughout `decision_v3.md`/`v4.md`/`v5.md`'s comparison tables) is
actually the *worst* of the four by MCC, despite having the highest raw
accuracy — accuracy alone was misleading here because mode 3's higher
recall came with proportionally more false positives, which MCC penalizes
and accuracy doesn't weight as heavily. The gap between best (0.2424) and
previously-reported (0.2236) is real but modest (+0.019) — it does not
change WebRTC's standing relative to NOVA-VAD/the ensemble/Pyannote/Silero
(WebRTC remains clearly behind all of them at any mode), but the
previously-reported WebRTC number in every prior round's comparison table
was not WebRTC's best achievable result, and should be read as such
going forward.

Reproduction: `python3 -m scripts.webrtc_all_modes`

## Item 2c — hard negatives (secondary priority, attempted after 2a/2b)

**Scope, disclosed explicitly:** plan Section 7.2 Layer 4 lists 8+
categories (laughter, coughing, crying, singing, TV, DTMF, hold music,
breathing, overlapping speech). Only **DTMF** was added this round — it's
the one category synthesizable programmatically with zero new
data-sourcing or licensing risk within this round's time budget. **The
other ~7 categories (laughter, coughing, crying, singing, TV, hold music,
breathing, overlapping speech) need real audio this project does not
currently have, and are explicitly DEFERRED to a future round — not
silently skipped.** Sourcing them cleanly (right licensing, no leakage
risk, actually representative samples) is real additional work, not
something to rush in the time remaining.

**Why DTMF is a meaningful test, not a token gesture:** verified directly
(not assumed) that DTMF tones are strongly periodic — a representative
digit ('5', 770+1336Hz) measures `periodicity_strength=0.968`, essentially
indistinguishable from a real voiced-speech tone's 0.979 (from round 1's
synthetic-signal validation). This is exactly the acoustic property
NOVA-VAD-frame-v2's periodicity feature relies on to detect speech under
noise — DTMF is a plausible, targeted adversarial case for that specific
design choice, not an arbitrary hard negative.

10 scenes, synthesized DTMF digit sequences (3-6 tones/scene, 150-350ms
each, realistic press durations) over silence, ground truth = no speech
anywhere. Metric that matters: **false-positive rate** (MCC/F1 are
degenerate when the ground truth never contains the positive class).

### Results

| System | False-positive rate (10ms frames) |
|---|---|
| NOVA-VAD (v0) | **75.21%** — catastrophic, consistent with v0's general unreliability |
| NOVA-VAD-frame-v1 | 17.00% |
| Pyannote VAD | 9.22% |
| WebRTC VAD (mode 3) | 7.07% |
| **NOVA-VAD-frame-v2** | **6.67%** |
| SpeechBrain VAD | 0.81% |
| Silero VAD | **0.00%** — perfect |

**Honest finding:** NOVA-VAD-frame-v2 handles this adversarial case
reasonably well — better than v1 (which lacks the periodicity feature
entirely), better than WebRTC, and better than Pyannote. This is
reassuring given the specific concern that motivated testing DTMF at all
(periodicity-based features might mistake tonal DTMF for periodic voiced
speech) — the concern was reasonable to test, but the empirical result
doesn't show it materializing as v2's biggest weakness. That said, v1
scoring *worse* than v2 despite lacking the periodicity feature suggests
periodicity isn't the dominant driver of DTMF false positives for either
model — some other shared feature (likely energy/spectral-magnitude ones
common to both v1 and v2) is more responsible, though this round did not
run a feature-ablation to confirm that directly. Silero and SpeechBrain
remain the most robust systems on this specific test.

Reproduction:
```
python3 -m scripts.generate_dtmf_scenes
python3 -m scripts.score_hardneg_dtmf
```

## Item 3 — Phase A checklist re-close

See `reports/phase_a_decision.md` (updated this round) for the full
re-assessment. Summary: the two named gaps from round 3 are now
**partially closed**, not fully closed — see that document for the exact
boundary of what "partially" means here, and for a new gap this round's
work surfaced (WebRTC's previously-reported operating point was not
actually its best one).
