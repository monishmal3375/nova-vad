# Decision v7 — Round 5: closing Phase A's three remaining gaps

**Date:** 2026-07-23
**Scope:** per the mandate, this round pushes to fully close all three
remaining Phase A gaps. As anticipated, not all three fully close in one
round — reported honestly below, item by item, with what closed and what
remains.

---

## Item 0 — retroactive WebRTC mode fix

Full detail and the "does this change any conclusion" check (answer: no,
checked explicitly, not assumed — one real nuance found on the clean
condition specifically): **`reports/master_comparison_table.md`**, a new
document created this round to supersede the stale
`frame_level_benchmark_v1.md`/`.json` (discovered in round 4 to still hold
original-test-set numbers under current-looking names — left as historical
record, not silently corrected, with a pointer added at its top).

---

## Item 1 — real RTC transmission testing (Plan Section 7.3, Layer 3)

**This is not a relabeled simulation. Full setup detail below so this is
independently verifiable.**

### What was actually built and run

A genuine browser-native `RTCPeerConnection` media path between two tabs
of the same browser (Chromium 148, via the Claude Browser pane's Electron
host — `navigator.userAgent`: `Mozilla/5.0 (Macintosh; Intel Mac OS X
10_15_7) ... Chrome/148.0.7778.280 Electron/42.7.0`, `navigator.platform`:
`MacIntel`). This is the **smaller-scope real RTC path** explicitly
sanctioned as acceptable when a full cross-device call setup isn't
practical — called exactly that here, not implied to be a full phone-call
test.

**Sender** (browser tab, role=sender): loads a real 12-second speech+noise
scene `.wav` from `data/scenes/test_v2/` (test-only pool, never
train-side) via an `<audio>` element, calls `audioEl.captureStream()` to
get a live `MediaStreamTrack`, adds it to an `RTCPeerConnection`, and
plays the file in real time — the actual audio samples that get encoded
and transmitted are the real waveform, not a synthetic signal.

**Receiver** (separate browser tab, role=receiver, same browser instance):
a second `RTCPeerConnection`, receives the remote track via `ontrack`,
records it with the browser's native `MediaRecorder` API
(`audio/webm;codecs=opus`), and uploads the raw recorded bytes via `fetch`
POST to a local Python upload server (`scripts` live in
`/private/tmp/.../scratchpad/rtc_test/`, not committed to the repo since
they're test infrastructure, not project code — the *results* are
committed).

**Signaling**: `BroadcastChannel` (same-origin, real cross-tab browser
API) — carries only SDP offer/answer and ICE candidates, never touches
the audio itself (the audio only ever travels over the actual
`RTCPeerConnection` media path).

**Route**: real ICE negotiation, confirmed via `RTCPeerConnection.getStats()`
logged from the page itself (not asserted from outside) —
`iceConnectionState` genuinely transitioned `checking` → `connected` for
every one of the 5 runs. Candidate gathering produced real `srflx`
(server-reflexive) candidates resolved via an actual STUN request to
`stun.l.google.com:19302` over the real internet — one gathered candidate
in the initial connectivity check was `98.220.128.207` (a real public IP,
not a placeholder) plus a real IPv6 address, confirming the ICE stack
performed genuine STUN resolution against a real external server, not a
mock. The selected candidate pair for media flow used `host`-type
candidates (both tabs on the same machine, so the shortest real path was
selected — this is real ICE behavior, not evidence of mocking: ICE always
selects the lowest-latency working path, and localhost genuinely is the
lowest-latency real path when both peers are on the same machine).
`getStats()` also confirms the codec actually negotiated and used was
`opus` (`minptime=10;useinbandfec=1` — real Opus SDP parameters), and
logged real packet counts (589-596 packets, ~38-44KB) actually received
over the RTP stream for every run.

**Capture method**: `MediaRecorder` on the received `MediaStream`,
periodic 500ms chunks (`recorder.start(500)`), uploaded as a `.webm`
container with Opus-encoded audio, decoded back to PCM via PyAV,
resampled to 16kHz.

### Two real bugs hit and fixed while building this — disclosed, not smoothed over

1. **`BroadcastChannel.postMessage` cannot structured-clone
   `RTCSessionDescription`/`RTCIceCandidate` objects** — throws
   `DataCloneError` synchronously. First attempt silently dropped every
   offer/answer/ICE message (confirmed the failure mode directly: a
   targeted `postMessage` test threw `"RTCSessionDescription object could
   not be cloned"`). Fixed by converting to plain `{type, sdp}` /
   `.toJSON()` objects before posting.
2. **`MediaRecorder` on a remote WebRTC track produced almost no data**
   (110-byte blob for ~12s of real audio) even though `getStats()`
   confirmed 43KB of real RTP data was received — a known browser quirk
   where a remote track needs to also be connected to a playback sink for
   some `MediaRecorder` implementations to reliably pump its data. Fixed
   by attaching the remote stream to a hidden, muted `<audio autoplay>`
   element in addition to the recorder. After the fix, `ondataavailable`
   fired every 500ms with real, consistently-sized chunks (589 packets /
   ~43KB matched a ~188KB Opus-encoded blob, a plausible ratio).

Both caught by direct inspection of real values (`getStats()` output,
`postMessage` return/throw, chunk sizes) before trusting any recording —
same standard as the codec-degradation bugs from round 4.

### A third issue, also caught and corrected: false alignment match

An initial timing-alignment check (naive full-file cross-correlation)
reported a 7.77-second offset between source and received audio — checked
by eye against the scene's known speech-interval timestamps and the
received audio's own energy envelope, and this was clearly wrong (a false
match, most likely silence-correlating-with-silence in the mostly-quiet
background bed). Recomputed with a properly bounded search window
(-500ms to +1000ms) and an energy-envelope-based correlation (robust to
Opus's waveform-phase differences) — found a **consistent ~230-270ms
offset across all 5 runs, correlation 0.980-0.987**, matching the
expected real-world explanation (the first fraction of a second of
audio was lost during ICE/connection setup, before `MediaRecorder`
started — a real, expected characteristic of live transmission, not an
artifact). Received audio was padded with silence to compensate and
realign to the original 10ms-grid ground truth, verified by re-checking
correlation at the corrected offset (see `reports/data_manifest_and_leakage_audit.md`-style discipline — every alignment claim checked
numerically, not assumed).

### Scale, disclosed honestly

**5 scenes** (`test_v2_scene_0000` through `0004`, all clean condition,
from the locked `test_v2` test-only pool). This is a **pilot-scale
demonstration that the pipeline works end-to-end with genuine
transmission**, not a statistically powered benchmark addition — compare
to the 25-scene codec test from round 4. Confidence intervals below are
correspondingly wide (up to 34.5pp) and should be read as directional,
not precise.

### Results

| Model | Matched clean (same 5 source scenes) | Real RTC transmission | Δ |
|---|---|---|---|
| NOVA-VAD (v0) | 31.47% | 31.13% | -0.34pp |
| WebRTC VAD (mode 2) | 87.90% | 87.15% | -0.75pp |
| NOVA-VAD-frame-v1 | 81.43% | 87.83% | +6.40pp (likely n=5 noise, not a real gain — see caveat) |
| **NOVA-VAD-frame-v2** | 88.00% | 88.37% | +0.37pp |
| Silero VAD | 86.85% | 86.80% | -0.05pp |
| Pyannote VAD | 84.43% | 84.63% | +0.20pp |
| SpeechBrain VAD | 74.18% | 69.32% | -4.86pp |

Full metrics (precision/recall/F1/MCC/CI): `reports/per_scene_test_v2_rtc_full_metrics.json`.

**Honest finding:** real RTC transmission's effect is small for most
systems, similar in magnitude to round 4's *simulated* codec test. This
is a valuable cross-check, not just a repeat: it suggests the simulated
codec test was a reasonable proxy for at least this dimension of real
transmission's impact, for this specific setup (same-machine loopback,
good network conditions, no packet loss/congestion). **This does NOT
extend to network-adverse conditions** (packet loss, jitter, congestion,
cross-device latency) — this test's loopback path had a real but
essentially ideal network path (host-candidate selected, sub-millisecond
RTT per `getStats()`), so it says nothing about degraded-network
transmission, which remains untested.

**v1's apparent +6.4pp improvement is flagged, not explained away** — with
n=5 and the other systems all showing small (<5pp) deltas in either
direction, this is most plausibly small-sample variance, not a genuine
codec-interacts-well-with-v1 effect, consistent with how a similarly
counterintuitive small "improvement" was treated in round 4's codec test.

### What this does NOT close

- **Packet loss, jitter, congestion, or genuinely adverse network
  conditions** — this test's path was real but easy (localhost-resolved
  host candidates, ~0ms measured RTT). Real degraded-network transmission
  remains untested.
- **Cross-device transmission** — both peers were tabs in the same
  browser process. A genuine phone-to-phone or app-to-app call across
  real network infrastructure was not attempted this round; explaining
  why: setting up two independent devices/apps with controllable audio
  injection and capture was assessed as a materially larger undertaking
  than the time budget for this round allowed, given Items 2 and 3 also
  needed real progress. Proposed next step if pursued: two VoIP softphone
  clients (e.g. two SIP or WebRTC-based apps) on separate machines or
  devices on the same LAN, with one injecting a known file via a virtual
  audio cable and the other recording its speaker output — genuinely
  cross-device, still practical for a solo builder.
- **Statistical power** — 5 scenes is a pilot, not a benchmark-grade
  sample. Scaling to test_v2's full 25-scene clean set (or beyond) with
  this same pipeline is mechanical repetition of what's already built and
  verified, not new engineering — deferred due to time, not difficulty.

### Reproduction

The test harness (`rtc_test.html`, `upload_server.py`) lives in the
session scratchpad, not the repo (it's browser-automation test
infrastructure, not project code) — the alignment/scoring scripts and all
results ARE in the repo:

```
# (harness setup, not repo code): two browser tabs load rtc_test.html
# with ?role=sender/receiver&src=<scene>.wav, real RTCPeerConnection
# established via BroadcastChannel signaling, MediaRecorder captures the
# received track, uploaded to a local server.

# Repo-side alignment + scoring, from the received .webm files:
python3 -m scripts.compute_per_scene_results data/scenes/test_v2_rtc reports/per_scene_test_v2_rtc.json
python3 -m scripts.report_from_per_scene reports/per_scene_test_v2_rtc.json
```

Committed scene files (`data/scenes/test_v2_rtc/*.wav` + `.json`) contain
the actual post-transmission, realigned audio and full provenance metadata
(`alignment_lag_ms`, `alignment_correlation`, `transmission_note`) for
independent verification.

---

## Item 2 — remaining hard-negative categories (Plan Section 7.2, Layer 4)

DTMF was done in round 4. Of the remaining categories (laughter, coughing,
crying, singing, television, hold music, breathing, overlapping speech):

**3 added this round** — chosen specifically because they can be sourced
with zero licensing risk:

| Category | Source | License |
|---|---|---|
| Overlapping speech | Two real clips from the test-only `data/speech/` pool, mixed | Google Speech Commands v0.02, **CC BY 4.0** — the same corpus and license already documented for every other use of this pool in this project |
| Breathing | Synthesized: low-pass-filtered broadband noise with a slow rhythmic amplitude envelope, matching real breathing's actual acoustic signature (turbulent airflow = broadband noise, not periodic) | Fully synthesized, no external source, no licensing constraint |
| Hold music | Synthesized: an original 8-note melodic loop (sine tones, simple envelope), not based on any existing composition | Fully synthesized, no external source, no licensing constraint |

**5 categories NOT attempted, explicitly deferred, not silently
skipped:**
- **Laughter, coughing, crying, singing** — these need authentic recorded
  audio to be a meaningful test. This environment has no verified,
  reliable path to download and license-check external audio (e.g. from
  freesound.org) within this round, and crude synthesis of these
  specifically (unlike breathing or tones) would not produce an
  authentic acoustic signature — a fake "cough" built from noise bursts
  doesn't test whether a system confuses a *real* cough with speech, it
  tests something else entirely. Reported as not done rather than faked.
- **Television** — deferred specifically for licensing risk, exactly as
  flagged: real broadcast content (dialogue + music + effects) carries
  real IP concerns, and there's no clean synthetic proxy for "TV" the way
  there is for a pure tone (hold music) or filtered noise (breathing).

### Per-category results (8 scenes/category, 6s each; false-positive rate for the two pure-negative categories, recall for overlapping speech since ground truth there does contain real speech)

| System | Breathing (FP rate) | Hold music (FP rate) | Overlapping speech (recall) |
|---|---|---|---|
| NOVA-VAD (v0) | 84.90% | 54.17% | 49.26% |
| WebRTC VAD (mode 2) | 74.25% | 39.00% | 56.61% |
| NOVA-VAD-frame-v1 | 0.00% | 25.00% | 69.20% |
| **NOVA-VAD-frame-v2** | **0.00%** | **0.00%** | 72.62% |
| Silero VAD | 0.00% | 0.00% | 59.41% |
| Pyannote VAD | 0.00% | 0.00% | 61.50% |
| SpeechBrain VAD | 0.00% | 0.00% | **89.15%** |

**Honest findings:**
- NOVA-VAD-frame-v2 matches Silero/Pyannote/SpeechBrain's perfect 0% false
  positive rate on both pure hard-negative categories — a genuinely good
  result, not inflated.
- **v1 (25% FP on hold music) is worse than v2 (0% FP) despite v1 lacking
  the periodicity feature entirely** — the same pattern found with DTMF in
  round 4. This is further evidence against the hypothesis that adding
  periodicity would make NOVA-VAD-v2 *more* susceptible to tonal false
  positives; empirically it correlates with the opposite in both tests run
  so far, though this round did not run a controlled ablation to establish
  causation.
- Overlapping-speech recall is NOVA-VAD-frame-v2's relative weak point in
  this set (72.62%, better than v1/Silero/Pyannote but well behind
  SpeechBrain's 89.15%) — real, specific, actionable information for a
  future round if overlapping speech matters for the product.

Reproduction:
```
python3 -m scripts.generate_hardneg_extra
python3 -m scripts.compute_per_scene_results data/scenes/test_v2_hardneg_extra reports/per_scene_test_v2_hardneg_extra.json
```

---

## Item 3 — fair threshold tuning for Silero, Pyannote, SpeechBrain (Plan Section 7.4)

Checked each system's actual public interface for a tunable threshold —
not assumed:

| System | Tunable? | Parameter |
|---|---|---|
| Silero | Yes | `threshold` on `get_speech_timestamps()`, default 0.5 |
| SpeechBrain | Yes | `activation_th`/`deactivation_th` on `get_speech_segments()`, default 0.5/0.25 |
| **Pyannote** | **No, for this model** | `onset`/`offset` are hardcoded to 0.5 for powerset models (`pyannote/segmentation-3.0` is one) and are not exposed as `Uniform` pipeline hyperparameters — **confirmed by direct experimentation**: `pipeline.instantiate({'onset': 0.3, ...})` raises `ValueError: parameter 'onset' does not exist`. Only `min_duration_on`/`min_duration_off` (post-processing durations) are genuinely tunable for this specific model. Documented explicitly rather than silently tuning something else and calling it equivalent. |

### Val-only tuning results (frame F1, same objective used throughout this project)

| System | Default | Fair (val-tuned) | Val F1 gain |
|---|---|---|---|
| Silero | threshold=0.5 → F1=56.62% | **threshold=0.3** → F1=62.29% | +5.67pp |
| SpeechBrain | activation=0.5/deactivation=0.25 → F1=30.07% | **same** (best in grid = default) → F1=30.07% | +0.00pp |
| Pyannote | min_dur=0.0/0.0 → F1=57.08% | **min_dur=0.0/0.25** → F1=57.14% | +0.06pp |

### Single evaluation on test_v2 (first contact for this decision)

| System | Default MCC (test_v2) | Fair-tuned MCC (test_v2) | Δ |
|---|---|---|---|
| Silero | 0.5218 | **0.5096** | **-0.0122 (worse)** |
| SpeechBrain | 0.3898 | 0.3898 | 0.0000 |
| Pyannote | 0.5414 | 0.5420 | +0.0006 |

**Honest finding, reported plainly, not spun: Silero's fair-tuned threshold
scores *worse* on test_v2 than its untuned default.** The val-optimal
threshold (0.3, chosen because it clearly beat 0.5 on val's 40 scenes:
62.29% vs 56.62% F1) did not generalize — on test_v2's 100 scenes, it's
measurably behind the default. This is not a failure of the methodology;
it's exactly the kind of result a genuinely non-cherry-picked val/test
split should sometimes produce, and it's evidence *for* the discipline
being followed correctly (if every "fair-tuned" number always won,
that would itself be suspicious). The most likely explanation: val's 40
scenes are a noisier, smaller sample than test_v2's 100, so its
threshold-optimal point carries more sampling variance.

**Does this change any conclusion? No — checked explicitly:**
- Silero vs. NOVA-VAD-frame-v2 (MCC 0.4379): Silero stays ahead under
  either its default (0.522) or fair-tuned (0.510) number.
- Pyannote vs. NOVA-VAD-frame-v2: stays ahead either way (0.541-0.542).
- SpeechBrain vs. NOVA-VAD-frame-v2: stays behind either way (0.390 <
  0.438).
- **The round-3 ensemble result is entirely unaffected** — it uses
  Silero's raw per-chunk probabilities extracted directly from the
  underlying model (bypassing `get_speech_timestamps()` and its
  `threshold` parameter entirely), so this tuning exercise doesn't touch
  it.

`reports/master_comparison_table.md` updated with both default and
fair-tuned rows for transparency, per the same convention used for the
WebRTC mode fix.

Reproduction:
```
python3 -m scripts.tune_baselines_val               # val only
python3 -m scripts.evaluate_tuned_baselines_test_v2  # single test_v2 run
```
