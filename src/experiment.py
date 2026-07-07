"""
Rigorous train/tune/test methodology for NOVA-VAD model improvement work.

Why this file exists (separate from src/benchmark.py):
  benchmark.py's job is a single honest head-to-head snapshot against other
  VAD systems on one held-out split — that's the number that goes in the
  README. This file's job is different: it's the *methodology* for improving
  NOVA-VAD itself without overfitting to that snapshot.

Three-way split, strictly separated:
  1. HELD-OUT TEST SET  — carved off first, stratified by class and (for
     noise) by UrbanSound8K category. Never touched by feature selection,
     hyperparameter search, or model selection. Only ever scored once we've
     locked in a final pipeline.
  2. TRAIN/VAL POOL      — everything else. All hyperparameter tuning and
     feature-pruning decisions are made using cross-validation *within this
     pool only*.
  3. K-FOLD CV over the train/val pool — reports mean +/- std accuracy
     across folds, so a claimed improvement is a distribution, not a single
     number that could be +/-2 files of noise.

Usage:
  python3 -m src.experiment cv           # CV report on current feature set
  python3 -m src.experiment search       # hyperparameter random search (train/val pool only)
  python3 -m src.experiment final        # train final model on train/val pool, score ONCE on held-out test
  python3 -m src.experiment importances  # feature importance + correlation pruning analysis
"""
import os
import sys
import json
import time
import random
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from src.classifier import extract_features, extract_features_from_array
from src.explainer import get_feature_names

SPEECH_DIR = "data/speech"
NOISE_DIR = "data/noise"
CACHE_PATH = "data/_feature_cache.joblib"
RESULTS_DIR = "results"
HELD_OUT_FRACTION = 0.25
RANDOM_STATE = 42
TARGET_SR = 16000

# NOTE on a duration confound discovered during this work: data/speech comes
# from Google Speech Commands (~1s single-word clips) and data/noise comes
# from UrbanSound8K (~3.5-4s environmental clips). A duration-only classifier
# (RandomForest on clip length alone) gets ~99% accuracy separating these two
# classes — meaning "how long is this clip" is a near-perfect shortcut that
# has nothing to do with whether the audio contains speech. A feature-based
# model can (and, empirically, does — see results/importances output where
# a single whole-clip std-based feature dominates importance) partially
# exploit this shortcut instead of learning real speech-vs-noise acoustics.
# This confound predates this change (it's present in the original
# 250/250 dataset that produced the 94.0% headline number too), but this
# expansion is the first time it was measured directly.
#
# Fix: standardize every clip to a fixed WINDOW_SECONDS window before
# feature extraction — pad short clips with (low-amplitude) repetition,
# center-crop long clips. This matches how the model is actually used at
# inference time: src/stream.py always feeds fixed 1-second chunks. Any
# accuracy this removes was never real generalization — it was the model
# fitting to which corpus a clip came from.
WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = int(TARGET_SR * WINDOW_SECONDS)


def _standardize_duration(audio: np.ndarray) -> np.ndarray:
    """
    Fixes every clip to exactly WINDOW_SAMPLES samples:
      - longer clips: take a random-seeded center crop (deterministic here)
      - shorter clips: tile/repeat to fill the window (keeps the signal's
        own spectral character rather than zero-padding, which would just
        introduce a different artificial "silence ratio" shortcut)
    """
    n = len(audio)
    if n == WINDOW_SAMPLES:
        return audio
    if n > WINDOW_SAMPLES:
        start = (n - WINDOW_SAMPLES) // 2
        return audio[start:start + WINDOW_SAMPLES]
    # shorter than window — tile to fill, then trim to exact length
    reps = int(np.ceil(WINDOW_SAMPLES / max(n, 1)))
    return np.tile(audio, reps)[:WINDOW_SAMPLES]


def _extract_features_windowed(file_path: str) -> np.ndarray:
    audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    audio = _standardize_duration(audio)
    return extract_features_from_array(audio, sr)


# ── Feature caching (feature extraction is the slow part; cache once) ─────
def load_or_build_features():
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached features from {CACHE_PATH}")
        return joblib.load(CACHE_PATH)

    print(f"Extracting features for all files at a standardized {WINDOW_SECONDS}s "
          f"window (this is cached after first run)...")
    speech_files = sorted(f for f in os.listdir(SPEECH_DIR) if f.endswith(".wav"))
    noise_files = sorted(f for f in os.listdir(NOISE_DIR) if f.endswith(".wav"))

    import csv

    # noise category + fsID (source-recording ID) manifest, written by
    # download_noise.py / download_noise_expand.py and backfilled by
    # backfill_fsid.py. fsID is the UrbanSound8K source-recording grouping
    # key -- multiple 4-second slices can come from the same original field
    # recording, so held_out_split() must group by fsID, not just category,
    # or slices from the same recording can leak across train/test.
    manifest_path = os.path.join(NOISE_DIR, "_category_manifest.csv")
    noise_category = {}
    noise_fsid = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            for row in csv.DictReader(fh):
                noise_category[row["noise_filename"]] = row["category"]
                noise_fsid[row["noise_filename"]] = row.get("fsID", "unknown")

    # speech speaker-ID manifest, written by backfill_speaker_id.py from
    # Google Speech Commands' original "<speaker_hash>_nohash_<n>.wav" file
    # naming. Same class of leakage risk as fsID -- multiple utterances from
    # the same speaker split across train/test.
    speaker_manifest_path = os.path.join(SPEECH_DIR, "_speaker_manifest.csv")
    speech_speaker = {}
    if os.path.exists(speaker_manifest_path):
        with open(speaker_manifest_path) as fh:
            for row in csv.DictReader(fh):
                speech_speaker[row["speech_filename"]] = row.get("speaker_id", "unknown")

    X, y, filenames, groups, split_keys = [], [], [], [], []

    for i, f in enumerate(speech_files):
        X.append(_extract_features_windowed(os.path.join(SPEECH_DIR, f)))
        y.append(1)
        filenames.append(f)
        groups.append("speech")
        # group-by key for held_out_split: speaker_id if known, else fall
        # back to per-file uniqueness (no grouping benefit, but never
        # silently mixes real speaker-grouping with fabricated grouping)
        speaker = speech_speaker.get(f, "unknown")
        split_keys.append(f"speaker:{speaker}" if speaker != "unknown" else f"speech_nogroup:{f}")
        if (i + 1) % 100 == 0:
            print(f"  speech {i+1}/{len(speech_files)}")

    for i, f in enumerate(noise_files):
        X.append(_extract_features_windowed(os.path.join(NOISE_DIR, f)))
        y.append(0)
        filenames.append(f)
        groups.append(noise_category.get(f, "unknown"))
        fs_id = noise_fsid.get(f, "unknown")
        split_keys.append(f"fsid:{fs_id}" if fs_id != "unknown" else f"noise_nogroup:{f}")
        if (i + 1) % 100 == 0:
            print(f"  noise {i+1}/{len(noise_files)}")

    data = {
        "X": np.array(X),
        "y": np.array(y),
        "filenames": filenames,
        "groups": groups,  # "speech" for speech files, UrbanSound8K category for noise (stratification tier)
        "split_keys": split_keys,  # fine-grained no-leak grouping key: fsID / speaker_id (falls back to per-file)
    }
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    joblib.dump(data, CACHE_PATH)
    print(f"Cached feature matrix to {CACHE_PATH} — delete this file to force re-extraction.")
    return data


# ── Stratified held-out split (by class/category, grouped by source-recording
#    or speaker so no group ever spans both train and test) ────────────────
def held_out_split(y, groups, held_out_fraction=HELD_OUT_FRACTION, seed=RANDOM_STATE, split_keys=None):
    """
    Carves off a held-out test set stratified by group (speech vs. each
    UrbanSound8K noise category), so the test set has proportional
    representation of every category, not just overall class balance.

    If `split_keys` is provided (fine-grained grouping key -- UrbanSound8K
    fsID for noise, speaker_id for speech, see load_or_build_features()),
    the split additionally guarantees that no single source recording /
    speaker has clips in BOTH train and test. This matters because
    UrbanSound8K's own creators warn that multiple 4-second slices cut from
    the same original field recording (same fsID) can be highly acoustically
    correlated -- their official folds are grouped by fsID for exactly this
    reason. Same class of risk applies to multiple utterances from the same
    Speech Commands speaker.

    Whole split_key groups are assigned entirely to train or entirely to
    test (never split), while still targeting the requested
    held_out_fraction within each stratification tier (category/"speech").
    Returns (train_idx, test_idx).
    """
    rng = random.Random(seed)
    by_group = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)

    if split_keys is None:
        # legacy behavior: no fine-grained grouping info available, fall
        # back to grouping by category/"speech" tier only (pre-fix behavior)
        split_keys = list(groups)

    # Build GLOBAL clusters by split_key first (across ALL groups/categories,
    # not per-group). This matters because a single UrbanSound8K source
    # recording (fsID) can legitimately contain slices labeled under two
    # DIFFERENT categories (e.g. the same field recording yields both
    # car_horn and engine_idling clips) -- if clusters were decided per-group
    # independently, the same fsID could end up assigned to train in one
    # category's pass and test in another's, which is still a leak even
    # though each individual group-loop looks "clean" in isolation.
    global_clusters = {}
    for i, sk in enumerate(split_keys):
        global_clusters.setdefault(sk, []).append(i)

    # Decide each cluster's assignment (train/test) exactly once, globally,
    # while still targeting each group's (category/"speech") held-out
    # fraction as closely as possible. Process groups in a stable order and,
    # within each group, only decide clusters that haven't been decided yet
    # (a cluster whose split_key already appeared in an earlier group keeps
    # whatever side it was already assigned).
    cluster_assignment = {}  # split_key -> "test" or "train"

    for g in sorted(by_group.keys()):
        idxs = by_group[g]
        n_target = max(1, round(len(idxs) * held_out_fraction))

        n_already_test = sum(
            1 for i in idxs if cluster_assignment.get(split_keys[i]) == "test"
        )

        undecided_keys = sorted({
            split_keys[i] for i in idxs if split_keys[i] not in cluster_assignment
        })
        rng.shuffle(undecided_keys)

        n_test_so_far = n_already_test
        for ck in undecided_keys:
            if n_test_so_far >= n_target:
                cluster_assignment[ck] = "train"
                continue
            cluster_assignment[ck] = "test"
            n_test_so_far += len(global_clusters[ck])

    test_idx = sorted(
        i for i, sk in enumerate(split_keys) if cluster_assignment.get(sk) == "test"
    )
    test_set = set(test_idx)
    train_idx = [i for i in range(len(y)) if i not in test_set]

    # invariant check: no split_key should ever appear on both sides
    train_keys = {split_keys[i] for i in train_idx}
    test_keys = {split_keys[i] for i in test_idx}
    overlap = train_keys & test_keys
    assert not overlap, f"held_out_split leaked {len(overlap)} split_key group(s) across train/test: {list(overlap)[:5]}"

    return np.array(train_idx), np.array(test_idx)


def compute_all_metrics(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "f1": round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
    }


# ── Current default model definition (kept in sync with classifier.py) ────
def make_models(rf_params=None, gbt_params=None):
    default_rf = dict(n_estimators=200, max_depth=10, random_state=RANDOM_STATE)
    default_gbt = dict(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE)
    if rf_params:
        default_rf.update(rf_params)
    if gbt_params:
        default_gbt.update(gbt_params)
    rf = RandomForestClassifier(**default_rf)
    gbt = GradientBoostingClassifier(**default_gbt)
    return rf, gbt


# ── Ensemble combination strategies ─────────────────────────────────────────
# "average"    — current production approach: mean of RF and GBT probabilities,
#                fixed 0.5 cutoff.
# "stacking"   — logistic regression meta-learner over [rf_prob, gbt_prob],
#                fit on out-of-fold predictions from the training split only
#                (so the meta-learner never sees predictions from models that
#                were trained on the same rows — avoids leakage).
# "threshold"  — same averaging as "average", but the 0.5 cutoff is replaced
#                by a threshold chosen on a held-back slice of the training
#                split to maximize accuracy (cost-sensitive: optimizes for
#                whatever FP/FN balance the training data actually shows,
#                rather than assuming FP and FN are equally costly by default).
def _fit_predict_ensemble(X_train, y_train, X_val, rf_params, gbt_params, mode):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    rf, gbt = make_models(rf_params, gbt_params)

    if mode == "stacking":
        cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        rf_oof = cross_val_predict(rf, X_train, y_train, cv=cv_inner, method="predict_proba", n_jobs=-1)[:, 1]
        gbt_oof = cross_val_predict(gbt, X_train, y_train, cv=cv_inner, method="predict_proba", n_jobs=-1)[:, 1]
        meta_X_train = np.column_stack([rf_oof, gbt_oof])

        rf.fit(X_train, y_train)
        gbt.fit(X_train, y_train)
        meta = LogisticRegression()
        meta.fit(meta_X_train, y_train)

        meta_X_val = np.column_stack([rf.predict_proba(X_val)[:, 1], gbt.predict_proba(X_val)[:, 1]])
        probs = meta.predict_proba(meta_X_val)[:, 1]
        preds = (probs > 0.5).astype(int)
        return preds, probs

    rf.fit(X_train, y_train)
    gbt.fit(X_train, y_train)
    rf_probs_val = rf.predict_proba(X_val)[:, 1]
    gbt_probs_val = gbt.predict_proba(X_val)[:, 1]
    avg_probs = (rf_probs_val + gbt_probs_val) / 2

    if mode == "threshold":
        # pick threshold on a held-back slice of the TRAINING split (never
        # the validation fold) to avoid tuning against the data we score on
        from sklearn.model_selection import train_test_split
        X_tr2, X_thresh, y_tr2, y_thresh = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
        )
        rf2, gbt2 = make_models(rf_params, gbt_params)
        rf2.fit(X_tr2, y_tr2)
        gbt2.fit(X_tr2, y_tr2)
        thresh_probs = (rf2.predict_proba(X_thresh)[:, 1] + gbt2.predict_proba(X_thresh)[:, 1]) / 2

        best_thresh, best_acc = 0.5, -1
        for t in np.arange(0.30, 0.71, 0.02):
            acc = accuracy_score(y_thresh, (thresh_probs > t).astype(int))
            if acc > best_acc:
                best_acc, best_thresh = acc, t
        preds = (avg_probs > best_thresh).astype(int)
        return preds, avg_probs

    # default: naive averaging, fixed 0.5 cutoff (matches current production behavior)
    preds = (avg_probs > 0.5).astype(int)
    return preds, avg_probs


# ── K-fold CV report on the train/val pool ─────────────────────────────────
def run_cv(X, y, groups=None, rf_params=None, gbt_params=None, n_splits=5, label="current", ensemble_mode="average"):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        y_train, y_val = y[train_idx], y[val_idx]

        preds, _ = _fit_predict_ensemble(X_train, y_train, X_val, rf_params, gbt_params, ensemble_mode)

        m = compute_all_metrics(y_val, preds)
        fold_metrics.append(m)
        print(f"  [{label}] fold {fold+1}/{n_splits}: acc={m['accuracy']}% f1={m['f1']}%")

    accs = [m["accuracy"] for m in fold_metrics]
    f1s = [m["f1"] for m in fold_metrics]
    precs = [m["precision"] for m in fold_metrics]
    recs = [m["recall"] for m in fold_metrics]

    summary = {
        "label": label,
        "n_splits": n_splits,
        "acc_mean": round(float(np.mean(accs)), 2),
        "acc_std": round(float(np.std(accs)), 2),
        "f1_mean": round(float(np.mean(f1s)), 2),
        "f1_std": round(float(np.std(f1s)), 2),
        "precision_mean": round(float(np.mean(precs)), 2),
        "recall_mean": round(float(np.mean(recs)), 2),
        "fold_accuracies": accs,
    }
    print(f"\n  [{label}] CV accuracy: {summary['acc_mean']}% +/- {summary['acc_std']}% "
          f"(F1: {summary['f1_mean']}% +/- {summary['f1_std']}%)")
    return summary


# ── Hyperparameter random search on train/val pool only ────────────────────
def run_search(X_trainval, y_trainval, n_iter=40):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_trainval)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rf_param_dist = {
        "n_estimators": [100, 200, 300, 400, 500, 600],
        "max_depth": [6, 8, 10, 12, 16, 20, None],
        "min_samples_leaf": [1, 2, 3, 4, 6, 8],
        "min_samples_split": [2, 4, 6, 8, 10],
        "max_features": ["sqrt", "log2", 0.3, 0.5, None],
        "class_weight": [None, "balanced"],
    }
    gbt_param_dist = {
        "n_estimators": [50, 100, 150, 200, 300],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
        "max_depth": [2, 3, 4, 5],
        "min_samples_leaf": [1, 2, 4, 8],
        "subsample": [0.6, 0.8, 0.9, 1.0],
        "max_features": ["sqrt", "log2", None],
    }

    print("\nSearching Random Forest hyperparameters (5-fold CV, train/val pool only)...")
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        rf_param_dist, n_iter=n_iter, cv=cv, scoring="accuracy",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
    )
    rf_search.fit(X_scaled, y_trainval)
    print(f"  Best RF params: {rf_search.best_params_}")
    print(f"  Best RF CV accuracy: {rf_search.best_score_*100:.2f}%")

    print("\nSearching Gradient Boosting hyperparameters (5-fold CV, train/val pool only)...")
    gbt_search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        gbt_param_dist, n_iter=n_iter, cv=cv, scoring="accuracy",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
    )
    gbt_search.fit(X_scaled, y_trainval)
    print(f"  Best GBT params: {gbt_search.best_params_}")
    print(f"  Best GBT CV accuracy: {gbt_search.best_score_*100:.2f}%")

    return rf_search.best_params_, gbt_search.best_params_


# ── Joint accuracy/latency hyperparameter search ────────────────────────────
# `run_search` above (mode "search") optimizes accuracy alone via
# RandomizedSearchCV -- its output (results/best_hyperparams.json) is NOT
# what's used in the trusted baseline, which deliberately uses plain
# defaults (n_estimators=200/max_depth=10 RF, n_estimators=100 GBT). Since
# inference latency scales with model complexity (more/deeper trees means
# more splits to walk per prediction), there is a real, separate question
# from "what's most accurate": what's the best ACCURACY-PER-MILLISECOND
# tradeoff. This searches a smaller, latency-aware candidate grid (biased
# toward cheaper models than run_search's grid, which goes up to
# n_estimators=600/max_depth=20), scores each candidate on the SAME 5-fold
# CV methodology as run_cv/run_search (train/val pool only, held-out test
# set never touched), and additionally measures each candidate's REAL
# inference latency (predict_proba wall-clock time on the actual fitted
# models, not a proxy like tree count) so the tradeoff is grounded in a
# measured number, not a guess.
#
# Objective: acc_mean - LATENCY_PENALTY_PER_MS * inference_latency_ms.
# LATENCY_PENALTY_PER_MS=0.05 means "1ms of extra inference latency must buy
# at least 0.05 accuracy points to be worth it" -- a candidate that trades
# 0.02% accuracy for a 5ms latency cut is preferred; one that trades 2%
# accuracy for the same 5ms is not. This is a judgment call, not a derived
# constant -- reported alongside raw accuracy/latency for every candidate so
# the tradeoff is inspectable, not hidden inside a single score.
LATENCY_PENALTY_PER_MS = 0.05


def _measure_inference_latency_ms(rf, gbt, scaler, X_sample, n_repeats=3):
    """
    Real wall-clock inference latency (scaler transform + RF/GBT
    predict_proba + averaging) per sample, matching what final_model_report
    actually times for the "inference" portion of end-to-end latency (see
    src/experiment.py mode "final"). Measured on already-extracted features
    (X_sample rows), so this isolates model-complexity cost from
    feature-extraction cost, which doesn't change with hyperparameters.
    Repeated a few times and averaged to reduce timer noise from a single
    run (same spirit as re-running latency checks 2-3x under system load,
    per this repo's established practice -- see commit 765c859).
    """
    X_scaled = scaler.transform(X_sample)
    per_sample_times = []
    for _ in range(n_repeats):
        t0 = time.time()
        rf_probs = rf.predict_proba(X_scaled)[:, 1]
        gbt_probs = gbt.predict_proba(X_scaled)[:, 1]
        _ = (rf_probs + gbt_probs) / 2
        elapsed = time.time() - t0
        per_sample_times.append(elapsed / len(X_sample) * 1000)
    return float(np.median(per_sample_times))


def run_latency_aware_search(X_trainval, y_trainval, n_candidates=24, latency_penalty=LATENCY_PENALTY_PER_MS):
    """
    Joint accuracy/latency hyperparameter search. Returns (best_rf_params,
    best_gbt_params, all_candidate_results) where all_candidate_results is
    every candidate's CV accuracy, measured inference latency, and combined
    score, sorted best-first -- so a "didn't beat the baseline" outcome is
    visible in the saved artifact, not just the winner.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_trainval)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Latency-aware candidate grid: biased toward SMALLER/SHALLOWER models
    # than run_search's accuracy-only grid (which searches up to
    # n_estimators=600, max_depth=20/None). Includes the current production
    # defaults (RF 200/10, GBT 100/0.1) as an explicit candidate so the
    # search can honestly conclude "the default is already the best
    # tradeoff" instead of being structurally unable to pick it.
    rng = np.random.RandomState(RANDOM_STATE)
    rf_grid = {
        "n_estimators": [50, 75, 100, 150, 200],
        "max_depth": [4, 6, 8, 10, 12],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    gbt_grid = {
        "n_estimators": [30, 50, 75, 100],
        "learning_rate": [0.05, 0.1, 0.15, 0.2],
        "max_depth": [2, 3, 4],
        "subsample": [0.8, 1.0],
    }

    def sample_params(grid):
        # .item() / native cast so candidates are JSON-serializable (np.int64
        # / np.float64 from rng.choice() otherwise break json.dump later)
        out = {}
        for k, v in grid.items():
            if isinstance(v[0], str):
                out[k] = v[rng.randint(len(v))]
            else:
                chosen = rng.choice(v)
                out[k] = chosen.item() if hasattr(chosen, "item") else chosen
        return out

    candidates = [
        {"rf": dict(n_estimators=200, max_depth=10),
         "gbt": dict(n_estimators=100, learning_rate=0.1)},  # current production default
    ]
    seen = {("default",)}
    while len(candidates) < n_candidates:
        rf_p = sample_params(rf_grid)
        gbt_p = sample_params(gbt_grid)
        key = (tuple(sorted(rf_p.items())), tuple(sorted(gbt_p.items())))
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"rf": rf_p, "gbt": gbt_p})

    print(f"\nJoint accuracy/latency search: {len(candidates)} candidates, "
          f"5-fold CV each (train/val pool only), latency_penalty={latency_penalty}/ms")

    # Reusable held-back latency-measurement sample: 200 rows from the
    # train/val pool (never the held-out test set), same features every
    # candidate is timed on so latency comparisons are apples-to-apples.
    lat_sample_idx = rng.choice(len(X_trainval), size=min(200, len(X_trainval)), replace=False)
    X_lat_sample = X_trainval[lat_sample_idx]

    results = []
    for i, cand in enumerate(candidates):
        rf_params, gbt_params = cand["rf"], cand["gbt"]
        fold_accs = []
        for train_idx, val_idx in cv.split(X_scaled, y_trainval):
            rf, gbt = make_models(rf_params, gbt_params)
            rf.fit(X_scaled[train_idx], y_trainval[train_idx])
            gbt.fit(X_scaled[train_idx], y_trainval[train_idx])
            rf_probs = rf.predict_proba(X_scaled[val_idx])[:, 1]
            gbt_probs = gbt.predict_proba(X_scaled[val_idx])[:, 1]
            preds = ((rf_probs + gbt_probs) / 2 > 0.5).astype(int)
            fold_accs.append(accuracy_score(y_trainval[val_idx], preds))
        acc_mean = float(np.mean(fold_accs)) * 100
        acc_std = float(np.std(fold_accs)) * 100

        # refit on the full train/val pool once for a realistic latency
        # measurement (fold-sized models would understate real model size)
        rf_full, gbt_full = make_models(rf_params, gbt_params)
        rf_full.fit(X_scaled, y_trainval)
        gbt_full.fit(X_scaled, y_trainval)
        inf_latency_ms = _measure_inference_latency_ms(rf_full, gbt_full, scaler, X_lat_sample)

        score = acc_mean - latency_penalty * inf_latency_ms
        label = "default" if cand["rf"] == dict(n_estimators=200, max_depth=10) and \
            cand["gbt"] == dict(n_estimators=100, learning_rate=0.1) else f"candidate_{i}"
        results.append({
            "label": label, "rf_params": rf_params, "gbt_params": gbt_params,
            "cv_acc_mean": round(acc_mean, 3), "cv_acc_std": round(acc_std, 3),
            "inference_latency_ms": round(inf_latency_ms, 4), "score": round(score, 4),
        })
        print(f"  [{i+1}/{len(candidates)}] {label:<14} acc={acc_mean:.2f}%+/-{acc_std:.2f}  "
              f"inf_latency={inf_latency_ms:.3f}ms  score={score:.3f}  "
              f"rf={rf_params}  gbt={gbt_params}")

    results.sort(key=lambda r: r["score"], reverse=True)
    best = results[0]
    default_result = next(r for r in results if r["label"] == "default")
    print(f"\nBest by joint score: {best['label']} "
          f"(acc={best['cv_acc_mean']}%, inf_latency={best['inference_latency_ms']}ms, score={best['score']})")
    print(f"Default for comparison: acc={default_result['cv_acc_mean']}%, "
          f"inf_latency={default_result['inference_latency_ms']}ms, score={default_result['score']}")
    if best["label"] == "default":
        print("Joint search did NOT find anything beating the current production default -- "
              "keeping defaults is the honest outcome here.")

    return best["rf_params"], best["gbt_params"], results


# ── Feature importance + correlation pruning analysis ──────────────────────
def run_importances(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf, gbt = make_models()
    rf.fit(X_scaled, y)
    gbt.fit(X_scaled, y)

    names = get_feature_names()
    if len(names) < X.shape[1]:
        names += [f"feature_{i}" for i in range(len(names), X.shape[1])]
    names = names[: X.shape[1]]

    rf_imp = rf.feature_importances_
    gbt_imp = gbt.feature_importances_
    combined = (rf_imp + gbt_imp) / 2

    ranked = np.argsort(combined)[::-1]
    print("\nTop 20 features by combined RF+GBT importance:")
    for idx in ranked[:20]:
        print(f"  {names[idx]:<30} {combined[idx]*100:.2f}%")

    print("\nBottom 15 features by combined RF+GBT importance (pruning candidates):")
    for idx in ranked[-15:]:
        print(f"  {names[idx]:<30} {combined[idx]*100:.3f}%")

    # correlation-based redundancy check among top features
    corr = np.corrcoef(X_scaled.T)
    print("\nHighly correlated feature pairs (|r| > 0.95, both in top 40 importance):")
    top40 = set(ranked[:40])
    seen = set()
    for i in top40:
        for j in top40:
            if i >= j or (i, j) in seen:
                continue
            seen.add((i, j))
            if abs(corr[i, j]) > 0.95:
                print(f"  {names[i]:<28} <-> {names[j]:<28} r={corr[i,j]:.3f}")

    return {"names": names, "rf_importances": rf_imp.tolist(), "gbt_importances": gbt_imp.tolist()}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "cv"
    data = load_or_build_features()
    X, y, groups = data["X"], data["y"], data["groups"]
    split_keys = data.get("split_keys")
    print(f"\nTotal dataset: {len(y)} files ({int(np.sum(y==1))} speech, {int(np.sum(y==0))} noise)")

    train_idx, test_idx = held_out_split(y, groups, split_keys=split_keys)
    print(f"Held-out test set: {len(test_idx)} files (stratified by class/category, "
          f"grouped by fsID/speaker so no source recording or speaker spans train/test, "
          f"never used for tuning)")
    print(f"Train/val pool:    {len(train_idx)} files")

    X_trainval, y_trainval = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if mode == "cv":
        summary = run_cv(X_trainval, y_trainval, label="baseline-hyperparams")
        with open(os.path.join(RESULTS_DIR, "cv_report.json"), "w") as f:
            json.dump(summary, f, indent=2)

    elif mode == "search":
        rf_best, gbt_best = run_search(X_trainval, y_trainval)
        with open(os.path.join(RESULTS_DIR, "best_hyperparams.json"), "w") as f:
            json.dump({"rf": rf_best, "gbt": gbt_best}, f, indent=2)
        print(f"\nSaved best hyperparameters to results/best_hyperparams.json")

        print("\nRe-running CV with tuned hyperparameters for comparison:")
        run_cv(X_trainval, y_trainval, rf_params=rf_best, gbt_params=gbt_best, label="tuned-hyperparams")

    elif mode == "importances":
        run_importances(X_trainval, y_trainval)

    elif mode == "search_latency":
        n_candidates = int(sys.argv[2]) if len(sys.argv) > 2 else 24
        penalty = float(sys.argv[3]) if len(sys.argv) > 3 else LATENCY_PENALTY_PER_MS
        rf_best, gbt_best, all_results = run_latency_aware_search(
            X_trainval, y_trainval, n_candidates=n_candidates, latency_penalty=penalty
        )
        out_path = os.path.join(RESULTS_DIR, "latency_aware_search.json")
        with open(out_path, "w") as f:
            json.dump({
                "latency_penalty_per_ms": penalty,
                "n_candidates": n_candidates,
                "best_rf_params": rf_best,
                "best_gbt_params": gbt_best,
                "all_candidates_sorted_by_score": all_results,
            }, f, indent=2)
        print(f"\nSaved all candidate results to {out_path}")
        print("NOTE: this does NOT overwrite results/best_hyperparams.json -- "
              "inspect results before deciding whether to adopt these params "
              "for the trusted baseline (mode 'final' only uses "
              "best_hyperparams.json if it exists).")

    elif mode == "ensemble":
        # compare naive averaging vs stacking vs cost-sensitive threshold
        # tuning, all via CV on the train/val pool only, so the comparison
        # itself doesn't touch the held-out test set.
        params_path = os.path.join(RESULTS_DIR, "best_hyperparams.json")
        rf_params = gbt_params = None
        if os.path.exists(params_path):
            with open(params_path) as f:
                best = json.load(f)
            rf_params, gbt_params = best["rf"], best["gbt"]
            print(f"Using tuned hyperparameters from {params_path}")

        results = {}
        for strategy in ["average", "stacking", "threshold"]:
            print(f"\n--- Ensemble strategy: {strategy} ---")
            results[strategy] = run_cv(
                X_trainval, y_trainval, rf_params=rf_params, gbt_params=gbt_params,
                label=strategy, ensemble_mode=strategy,
            )

        print("\nEnsemble strategy comparison (5-fold CV, train/val pool):")
        for strategy, summary in results.items():
            print(f"  {strategy:<10} acc={summary['acc_mean']}% +/- {summary['acc_std']}%  "
                  f"f1={summary['f1_mean']}% +/- {summary['f1_std']}%")

        with open(os.path.join(RESULTS_DIR, "ensemble_comparison.json"), "w") as f:
            json.dump(results, f, indent=2)
        print("\nSaved to results/ensemble_comparison.json")

    elif mode == "final":
        params_path = os.path.join(RESULTS_DIR, "best_hyperparams.json")
        rf_params = gbt_params = None
        if os.path.exists(params_path):
            with open(params_path) as f:
                best = json.load(f)
            rf_params, gbt_params = best["rf"], best["gbt"]
            print(f"Using tuned hyperparameters from {params_path}")
        else:
            print("No tuned hyperparameters found, using current defaults.")

        ensemble_mode = sys.argv[2] if len(sys.argv) > 2 else "average"
        print(f"Using ensemble strategy: {ensemble_mode}")

        # report CV distribution on train/val pool for statistical honesty
        cv_summary = run_cv(
            X_trainval, y_trainval, rf_params=rf_params, gbt_params=gbt_params,
            label="final-cv", ensemble_mode=ensemble_mode,
        )

        # train final model on ALL of train/val pool, score once on held-out test
        scaler = StandardScaler()
        X_trainval_scaled = scaler.fit_transform(X_trainval)
        X_test_scaled = scaler.transform(X_test)

        preds, probs = _fit_predict_ensemble(
            X_trainval_scaled, y_trainval, X_test_scaled, rf_params, gbt_params, ensemble_mode
        )

        test_metrics = compute_all_metrics(y_test, preds)
        print(f"\nFINAL held-out test set ({len(y_test)} files, never used for tuning):")
        print(f"  Accuracy:  {test_metrics['accuracy']}%")
        print(f"  Precision: {test_metrics['precision']}%")
        print(f"  Recall:    {test_metrics['recall']}%")
        print(f"  F1:        {test_metrics['f1']}%")

        # refit the production models on the full train/val pool for saving +
        # latency measurement (mirrors what pipeline.py / benchmark.py do)
        rf, gbt = make_models(rf_params, gbt_params)
        rf.fit(X_trainval_scaled, y_trainval)
        gbt.fit(X_trainval_scaled, y_trainval)

        test_filenames = [data["filenames"][i] for i in test_idx]
        latencies = []
        speech_dirset = set(os.listdir(SPEECH_DIR))
        for fname in test_filenames[:50]:
            path = os.path.join(SPEECH_DIR, fname) if fname in speech_dirset else os.path.join(NOISE_DIR, fname)
            t0 = time.time()
            feats = extract_features(path)
            fs = scaler.transform([feats])
            _ = (rf.predict_proba(fs)[0][1] + gbt.predict_proba(fs)[0][1]) / 2
            latencies.append((time.time() - t0) * 1000)
        mean_latency = round(float(np.mean(latencies)), 2)
        print(f"  Mean latency (feature extraction + inference): {mean_latency}ms over {len(latencies)} files")

        os.makedirs("models", exist_ok=True)
        joblib.dump(rf, "models/nova_vad_rf.pkl")
        joblib.dump(gbt, "models/nova_vad_gbt.pkl")
        joblib.dump(scaler, "models/nova_vad_scaler.pkl")
        model_size = sum(
            os.path.getsize(p) for p in
            ["models/nova_vad_rf.pkl", "models/nova_vad_gbt.pkl", "models/nova_vad_scaler.pkl"]
        )
        print(f"  Model size on disk: {model_size/1024:.1f}KB")

        result = {
            "ensemble_mode": ensemble_mode,
            "cv_summary": cv_summary,
            "held_out_test_metrics": test_metrics,
            "n_test_files": len(y_test),
            "n_trainval_files": len(y_trainval),
            "mean_latency_ms": mean_latency,
            "model_size_bytes": model_size,
            "rf_params": rf_params,
            "gbt_params": gbt_params,
        }
        with open(os.path.join(RESULTS_DIR, "final_model_report.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to results/final_model_report.json and models/")

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
