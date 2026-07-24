# Evidence package — NOVA-VAD-frame-v2 (current best standalone model)

Per plan Section 14.3's release-evidence-package list. This is an index
cross-referencing artifacts already produced and verified across rounds
1-3 — no new claims are made here, only assembly. Where an item genuinely
doesn't exist yet, that's stated plainly rather than filled in.

**Subject:** NOVA-VAD-frame-v2, the current best *standalone* model (per
`reports/phase_a_decision.md`). The round-3 logistic-ensemble result
(MCC 0.56) is a documented finding, not folded in here as "the model" —
shipping it is a flagged, unresolved product/licensing decision, not
something this evidence package should presume. Its own evidence lives in
`reports/decision_v5.md`.

| Item (plan Section 14.3) | Status | Location |
|---|---|---|
| Model artifact(s) | ✅ | `models/registry/nova-vad-frame-v2/frame_vad_v2_rf.pkl`, `frame_vad_v2_gbt.pkl`, `frame_vad_v2_scaler.pkl` |
| Preprocessing schema | 🟡 **Gap** | Feature *code* is fully specified and tested (`scripts/frame_features_v2.py`, `tests/test_frame_features_v2.py`), but unlike v0 (which has `models/registry/nova-vad-v0.1/feature_schema.json`), v2 has no standalone itemized JSON schema of its 62 feature names/indices. Not fabricated here since Part 3 is assembly-only — flagged as a real, cheap-to-fix gap for a future round. |
| Calibration and thresholds | ✅ | `models/registry/nova-vad-frame-v2/frame_vad_v2_hysteresis.json` (T_on=0.55, T_off=0.45, tuned on `val` — see `reports/decision_v3.md` Step 4) |
| Source code commit | ✅ | `68da84d` ("Noise-robustness pass: NOVA-VAD-frame-v2 (MCC +0.34 -> +0.43)") for the model training code; current HEAD `c328de9` for the full evaluation/analysis pipeline used to produce every number in `decision_v3.md`-`v5.md` |
| Training configuration | ✅ | `scripts/train_frame_vad_v2.py` (RF: 200 trees/depth 10, GBT: 100 estimators/lr 0.1, class_weight=balanced, trained on `data/scenes/train`+`train2`, 400 scenes / 48,000 windows) |
| Training/development/calibration/test manifest hashes | ✅ (this round) | `reports/scene_manifest_checksums.txt` — SHA-256 of `data/scenes/{manifest,val_manifest,train2_manifest,train3_manifest,test_v2_manifest}.json`. Per-split composition and pairwise leakage checks: `reports/data_manifest_and_leakage_audit.md` |
| Predictions on locked evaluation sets | ✅ | `reports/per_scene_test_v2.json` (raw per-frame predictions for NOVA-VAD-frame-v2 and all other systems on `test_v2`, from round 2's `compute_per_scene_results.py`) |
| Full metric tables and confidence intervals | ✅ | `reports/per_scene_test_v2_full_metrics.json`, tables in `reports/decision_v4.md` (accuracy/precision/recall/F1/MCC, 95% CI via cluster bootstrap, per-condition breakdown) |
| Latency/memory/device report | ❌ **Not done — out of scope for Phase A** | This is Phase E (Core ML / on-device deployment) work per the plan's own phasing; nothing to report here yet, stated plainly rather than improvised |
| Model card and dataset card | 🟡 **Partial** | Model card ✅: `models/registry/nova-vad-frame-v2/model_card.md`. Dataset card ❌: no standalone dataset card exists; `reports/data_manifest_and_leakage_audit.md` covers composition/leakage but not licensing/consent/provenance in the format a dataset card implies |
| Known limitations and prohibited claims | ✅ | `models/registry/nova-vad-frame-v2/model_card.md`'s "Known issues, disclosed not hidden" section; `reports/phase_a_decision.md`'s checklist gaps (no codec/transmission testing, no hard-negative testing beyond music, baselines not fairly tuned, WebRTC only tested at one aggressiveness mode) |
| Licenses and third-party notices | 🟡 **Gap** | Repo's own `LICENSE` (MIT) exists. **Third-party data/model licenses have not been reviewed in any round**: Google Speech Commands and MUSAN's exact license terms for this use, and Pyannote `segmentation-3.0`'s Hugging-Face-gated license terms (relevant if the round-3 ensemble is ever shipped) — flagged in `reports/phase_a_decision.md`, not resolved |

## Honest summary

**7 of 11 items fully done, 1 done this round (manifest hashes), 3 with
named gaps** (preprocessing schema JSON, dataset card, third-party
licenses) and 1 correctly out of scope for this phase (latency/memory).
None of the gaps invalidate the model numbers already reported — they mark
what a reviewer would still need before a production release decision,
consistent with `reports/phase_a_decision.md`'s framing that this
checklist assessment, not a green checklist by itself, is what should
inform the Gate 1 sign-off.
