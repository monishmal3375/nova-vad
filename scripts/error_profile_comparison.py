"""
Compares NOVA-VAD-frame-v2's per-scene failures against Pyannote and
Silero's, using the per-scene results already computed for the expanded
test set (Part 1.3) — no new test-set contact, this is analysis of an
already-final evaluation.

For each pair (NOVA-VAD-frame-v2, {Pyannote, Silero}), scenes are bucketed
into: both do well, both do poorly, only NOVA-VAD does poorly, only the
baseline does poorly — using each system's own median accuracy as its
"doing poorly" threshold (below its own median = the worse half of its
performance), so the comparison isn't biased by the baselines' generally
higher accuracy.

Run: python3 -m scripts.error_profile_comparison <per_scene_results.json>
"""
import json
import sys

import numpy as np


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "reports/per_scene_test_v2.json"
    with open(path) as f:
        all_results = json.load(f)

    nova = {s["scene_id"]: s["accuracy"] for s in all_results["NOVA-VAD-frame-v2"]}

    for baseline_name in ["Silero VAD", "Pyannote VAD"]:
        baseline = {s["scene_id"]: s["accuracy"] for s in all_results[baseline_name]}
        scene_ids = sorted(set(nova) & set(baseline))

        nova_median = np.median([nova[s] for s in scene_ids])
        baseline_median = np.median([baseline[s] for s in scene_ids])

        both_poor, only_nova_poor, only_baseline_poor, both_good = [], [], [], []
        for sid in scene_ids:
            nova_poor = nova[sid] < nova_median
            baseline_poor = baseline[sid] < baseline_median
            if nova_poor and baseline_poor:
                both_poor.append(sid)
            elif nova_poor and not baseline_poor:
                only_nova_poor.append(sid)
            elif not nova_poor and baseline_poor:
                only_baseline_poor.append(sid)
            else:
                both_good.append(sid)

        n = len(scene_ids)
        print(f"\n=== NOVA-VAD-frame-v2 vs {baseline_name} ({n} shared scenes) ===")
        print(f"  NOVA-VAD-frame-v2 median accuracy: {nova_median:.2f}%")
        print(f"  {baseline_name} median accuracy: {baseline_median:.2f}%")
        print(f"  Both below their own median (shared difficulty):  {len(both_poor):3d} ({len(both_poor)/n*100:.0f}%)")
        print(f"  Only NOVA-VAD-frame-v2 below median (NOVA-specific weakness): {len(only_nova_poor):3d} ({len(only_nova_poor)/n*100:.0f}%)")
        print(f"  Only {baseline_name} below median (baseline-specific weakness): {len(only_baseline_poor):3d} ({len(only_baseline_poor)/n*100:.0f}%)")
        print(f"  Both above their own median (shared strength):    {len(both_good):3d} ({len(both_good)/n*100:.0f}%)")

        # correlation between the two systems' per-scene accuracy — high correlation
        # means they struggle on the same scenes (shared hard cases, less room for
        # ensembling); low/negative correlation means complementary failure modes.
        nova_vals = [nova[s] for s in scene_ids]
        baseline_vals = [baseline[s] for s in scene_ids]
        corr = float(np.corrcoef(nova_vals, baseline_vals)[0, 1])
        print(f"  Per-scene accuracy correlation: {corr:.3f} "
              f"({'shared failure modes' if corr > 0.5 else 'complementary/independent failure modes' if corr < 0.3 else 'partially shared'})")


if __name__ == "__main__":
    main()
