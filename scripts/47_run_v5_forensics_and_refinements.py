#!/usr/bin/env python3
"""
scripts/47_run_v5_forensics_and_refinements.py
----------------------------------------------
Neuroinformatics v5 Complete Forensic & Benchmarking Pipeline:
  1. Study 2 (Figshare 4244171) unfiltered EDF header & 0.5-80 Hz batch PSD audit.
  2. Primary SHA-256 deduplication of Study 2 (removing mumtaz_hc_030 & mumtaz_mdd_034),
     yielding N = 564 unique primary participants, N = 675 primary sessions, and
     N = 733 total sessions w/ Study 2b EO (13,927 unique channel spectra), while
     retaining nominal 58/60 manifests in Study 2 duplicate sensitivity tables.
  3. Re-runs Module A (Biophysics, Bootstrap Delta_d, ANCOVA, Meta-Analysis K=6 and
     K=5 Omit-Study2-BatchConfounded, IAPF, UNM EC/EO, Table S1 OMST, Heuristic Slowing).
  4. Full 2D specparam fit-window grid ([1.0..3.0] x [25.0..40.0] Hz + 45.0 Hz on
     Study 2 unfiltered EDFs) and empirical fooof v1.1.1 vs specparam v2.0.0rc7 check.
  5. Empirical-variance-calibrated Monte Carlo leakage simulation (SD(chi)=0.16,
     SD(theta)=0.12, SD(alpha)=0.20, N=29 vs 27).
  6. True 10x5 Repeated Participant-Grouped CV across 13 models (adding GCN_Weighted_PLI
     via adj*pli and fixing Linear_SVM/RBF_SVM decision_function sigmoid scoring),
     paired Delta_AUC 95% CIs + TOST equivalence (+/-0.02, +/-0.03), and Table S10/S12A
     10x5 mean/SD integration.
  7. Aligned 1,000-permutation tests, Split-Target-HC transfer, 10x5 leakage ablation,
     GLI SPD audit (675 primary sessions), true 19x19 sample spatial covariance
     Riemannian tangent-space baseline (Riemannian_Spatial_Cov_190D), gain stress tests,
     and 1,000-permutation p-values for clinical severity regressions.
  8. Exports release_artifacts/derived_session_features_733.csv, cv_fold_indices_split0.csv,
     oof_predictions_split0.csv, and requirements_lock.txt.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import hashlib
import json
import multiprocessing as mp
import random
import shutil
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mne
import networkx as nx
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.stats import pearsonr, wilcoxon
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, SVR
from specparam import SpectralGroupModel
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore", category=UserWarning)
mne.set_log_level("ERROR")

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = BENCHMARK_ROOT / "src"
sys.path.append(str(SRC_DIR))
from harmonizer_and_graph import CHANNELS_19, LEGACY_1020_MAP, NormativeGCN, compute_fold_normative_z
import multi_cohort_loader as mcl

DATA_DIR = BENCHMARK_ROOT / "data" / "derivatives"
BACKUP_DIR = DATA_DIR / "v4_backups"
REPORTS_DIR = BENCHMARK_ROOT / "reports"
RELEASE_DIR = BENCHMARK_ROOT / "release_artifacts"
SCRATCH_DIR = Path("/Users/tan/.gemini/antigravity/scratch")
FOOOF_PY = SCRATCH_DIR / "fooof_venv" / "bin" / "python"

MUMTAZ_DIR = Path("/Users/tan/Downloads/4244171")
SLEEP_FIF_DIR = Path("/Users/tan/Documents/qwen-agent/aCAgqOVe0x/default/sleep_deprivation_eeg/data/derivatives/preprocessed_v3_conservative")
EEG_ROOT = Path("/Users/tan/Documents/eeg_datastes")
CAVANAGH_DIR = EEG_ROOT / "depression_new"
DEMENTIA_DIR = EEG_ROOT / "ds004504"
UNM_DIR = EEG_ROOT / "ds003490"
IOWA_DIR = EEG_ROOT / "ds004584"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
RELEASE_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

FOCAL_5 = [CHANNELS_19.index(c) for c in ["Fp1", "Fp2", "F3", "F4", "Fz"]]
FOCAL_3 = [CHANNELS_19.index(c) for c in ["F3", "Fz", "F4"]]
POSTERIOR_5 = [CHANNELS_19.index(c) for c in ["P3", "P4", "Pz", "O1", "O2"]]
REPEATS_10 = [42, 123, 456, 789, 2026, 101, 202, 303, 404, 505]
DUP_DROP_IDS = {"mumtaz_hc_030", "mumtaz_mdd_034"}

SPECPARAM_KWARGS = {
    "aperiodic_mode": "fixed",
    "peak_width_limits": [1.0, 12.0],
    "max_n_peaks": 6,
    "min_peak_height": 0.05,
    "peak_threshold": 2.0,
    "verbose": False,
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def welch_df(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1) / nx, np.var(y, ddof=1) / ny
    denom = (vx ** 2) / max(1, nx - 1) + (vy ** 2) / max(1, ny - 1)
    return float(((vx + vy) ** 2) / denom) if denom > 0 else float(nx + ny - 2)


def bootstrap_auc_ci(y_true, y_prob, n_boot: int = 1000, seed: int = 42):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan
    base_auc = float(roc_auc_score(y_true, y_prob))
    rng = np.random.RandomState(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not boot:
        return base_auc, base_auc, base_auc
    return base_auc, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def cluster_bootstrap_auc_ci(y_true, y_prob, groups, n_boot: int = 1000, seed: int = 42):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    g2idx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.RandomState(seed)
    boot = []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([g2idx[g] for g in sampled])
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not boot:
        base = float(roc_auc_score(y_true, y_prob))
        return base, base
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def get_grouped_splits(groups, y, n_splits: int = 5, seed: int = 42):
    groups = np.asarray(groups)
    y = np.asarray(y, dtype=int)
    rng = np.random.RandomState(seed)
    uniq_g, seen = [], set()
    for g in groups:
        if g not in seen:
            seen.add(g)
            uniq_g.append(g)
    uniq_g = np.asarray(uniq_g)

    sig_buckets = {}
    for g in uniq_g:
        ys = set(y[groups == g].tolist())
        sig = ys.pop() if len(ys) == 1 else 2
        sig_buckets.setdefault(sig, []).append(g)

    fold_groups = [[] for _ in range(n_splits)]
    for sig in sorted(sig_buckets.keys()):
        arr = np.asarray(sig_buckets[sig])
        rng.shuffle(arr)
        for idx, g in enumerate(arr):
            fold_groups[idx % n_splits].append(g)

    splits = []
    all_idx = np.arange(len(y))
    for k in range(n_splits):
        val_set = set(fold_groups[k])
        va_mask = np.array([g in val_set for g in groups], dtype=bool)
        tr_idx = all_idx[~va_mask]
        va_idx = all_idx[va_mask]
        if len(tr_idx) > 0 and len(va_idx) > 0:
            assert len(set(groups[tr_idx]) & set(groups[va_idx])) == 0
        splits.append((tr_idx, va_idx))
    return splits


def make_edge_matched_random_adj(num_nodes: int, num_edges: int, seed: int = 42) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    m = max(1, min(int(num_edges), len(edges)))
    for idx in rng.choice(len(edges), size=m, replace=False):
        u, v = edges[idx]
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return torch.tensor(adj, dtype=torch.float32)


def make_dense_adj(num_nodes: int = 19) -> torch.Tensor:
    adj = np.ones((num_nodes, num_nodes), dtype=np.float32)
    np.fill_diagonal(adj, 0.0)
    return torch.tensor(adj, dtype=torch.float32)


def train_eval_gcn(train_z, train_adj, train_y, val_z, val_adj, in_dim=4, hidden_dim=32, epochs=50, lr=0.005, weight_decay=1e-4, seed=42):
    set_seed(seed)
    model = NormativeGCN(in_node_features=in_dim, hidden_dim=hidden_dim, num_classes=2, dropout=0.25)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(model(train_z, train_adj), train_y)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(val_z, val_adj)
        probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
        preds = torch.argmax(logits, dim=-1).numpy()
    return probs, preds


def compute_es(cases, ctrls, fn, paired=False):
    vc = np.array([fn(g) for g in cases], dtype=float)
    vr = np.array([fn(g) for g in ctrls], dtype=float)
    nc, nr = len(vc), len(vr)
    if paired:
        diff = vc - vr
        d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))
        se = float(np.sqrt(1.0 / nc + (d ** 2) / (2.0 * nc)))
        t_st, p_val = stats.ttest_rel(vc, vr)
        df_v = float(nc - 1)
    else:
        sp_sd = np.sqrt(((nc - 1) * np.var(vc, ddof=1) + (nr - 1) * np.var(vr, ddof=1)) / max(1, nc + nr - 2)) + 1e-12
        d = float((np.mean(vc) - np.mean(vr)) / sp_sd)
        se = float(np.sqrt((nc + nr) / (nc * nr) + (d ** 2) / (2.0 * (nc + nr))))
        t_st, p_val = stats.ttest_ind(vc, vr, equal_var=False)
        df_v = welch_df(vc, vr)
    return {
        "n_case": nc,
        "n_ctrl": nr,
        "cohens_d": d,
        "se_d": se,
        "ci_lower": d - 1.96 * se,
        "ci_upper": d + 1.96 * se,
        "t_statistic": float(t_st),
        "df_satterthwaite": df_v,
        "p_value": float(p_val),
        "mean_case": float(np.mean(vc)),
        "mean_ctrl": float(np.mean(vr)),
    }


def bootstrap_delta_d(cases, ctrls, fn_a, fn_b, paired=False, n_boot=1000, seed=42):
    a_c = np.array([fn_a(g) for g in cases], dtype=float)
    a_r = np.array([fn_a(g) for g in ctrls], dtype=float)
    b_c = np.array([fn_b(g) for g in cases], dtype=float)
    b_r = np.array([fn_b(g) for g in ctrls], dtype=float)

    def d_pair(xc, xr, is_p):
        if is_p:
            df = xc - xr
            return float(df.mean() / (df.std(ddof=1) + 1e-12))
        nc, nr = len(xc), len(xr)
        s = np.sqrt(((nc - 1) * np.var(xc, ddof=1) + (nr - 1) * np.var(xr, ddof=1)) / (nc + nr - 2)) + 1e-12
        return float((np.mean(xc) - np.mean(xr)) / s)

    obs_a, obs_b = d_pair(a_c, a_r, paired), d_pair(b_c, b_r, paired)
    obs_delta = obs_a - obs_b
    rng = np.random.RandomState(seed)
    deltas = []
    nc, nr = len(a_c), len(a_r)
    for _ in range(n_boot):
        if paired:
            idx = rng.randint(0, nc, nc)
            da = d_pair(a_c[idx], a_r[idx], True)
            db = d_pair(b_c[idx], b_r[idx], True)
        else:
            ic = rng.randint(0, nc, nc)
            ir = rng.randint(0, nr, nr)
            da = d_pair(a_c[ic], a_r[ir], False)
            db = d_pair(b_c[ic], b_r[ir], False)
        deltas.append(da - db)
    arr = np.asarray(deltas)
    lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    p_two = float(2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0)))
    return obs_a, obs_b, obs_delta, lo, hi, min(1.0, max(1.0 / (n_boot + 1), p_two))


# ==============================================================================
# PART 1: STUDY 2 UNFILTERED EDF HEADER & 0.5-80 Hz BATCH PSD AUDIT
# ==============================================================================
def run_study2_unfiltered_audit():
    print("\n=== [Step 1] Study 2 Raw Unfiltered EDF Header & 0.5-80 Hz Batch Audit ===")
    manifest_ec = mcl.get_mumtaz_mdd_manifest(MUMTAZ_DIR, state="EC")
    assert len(manifest_ec) == 58

    def map_19ch_unfiltered(raw_in: mne.io.BaseRaw):
        rename_dict = {}
        for ch in raw_in.ch_names:
            c = ch.upper().replace("EEG ", "").replace("-LE", "").replace("-REF", "").replace("-AVG", "").strip()
            for k in LEGACY_1020_MAP:
                if c == k:
                    c = LEGACY_1020_MAP[k]
            for canon in CHANNELS_19:
                if c == canon.upper():
                    rename_dict[ch] = canon
                    break
        r = raw_in.copy()
        r.pick(list(rename_dict.keys()))
        r.rename_channels(rename_dict)
        r.reorder_channels(CHANNELS_19)
        r.set_eeg_reference("average", projection=False, verbose=False)
        return r

    audit_rows = []
    raw_unfilt_psds = []
    batch_curves = {"HC_2011": [], "HC_2013_01": [], "HC_2013_04_07": [], "MDD_2013_04_07": []}
    freqs_ref = None

    for item in manifest_ec:
        sid = item["subject_id"]
        fpath = item["path"]
        sha = hashlib.sha256(fpath.read_bytes()).hexdigest()[:16]
        raw = mne.io.read_raw_edf(str(fpath), preload=True, verbose=False)
        mdate = str(raw.info["meas_date"])[:10]
        sfreq = float(raw.info["sfreq"])
        hp = float(raw.info.get("highpass", np.nan))
        lp = float(raw.info.get("lowpass", np.nan))
        pref = ""
        try:
            extras = raw._raw_extras[0]
            pfs = extras.get("prefiltering", [])
            if pfs:
                pref = str(pfs[0]).strip()
        except Exception:
            pref = ""

        r19 = map_19ch_unfiltered(raw)
        # Compute clean 4s epoch indices via 1-40 Hz copy (identical rejection rule without altering unfiltered PSD)
        r_filt = r19.copy().filter(1.0, 40.0, fir_design="firwin", verbose=False)
        ep_len = int(4.0 * sfreq)
        dat_unfilt = r19.get_data()
        dat_filt = r_filt.get_data()
        n_ep = dat_unfilt.shape[1] // ep_len
        ep_u = np.stack([dat_unfilt[:, i * ep_len:(i + 1) * ep_len] for i in range(n_ep)])
        ep_f = np.stack([dat_filt[:, i * ep_len:(i + 1) * ep_len] for i in range(n_ep)])
        ptp = np.ptp(ep_f, axis=-1) * 1e6
        keep = np.max(ptp, axis=-1) <= 150.0
        if np.sum(keep) < 5:
            order = np.argsort(np.max(ptp, axis=-1))[:min(10, len(ep_u))]
            ep_clean = ep_u[order]
        else:
            ep_clean = ep_u[keep]

        freqs, pxx = signal.welch(ep_clean, fs=sfreq, nperseg=int(2.0 * sfreq), detrend="constant", axis=-1)
        mean_psd_19 = np.mean(pxx, axis=0).astype(np.float64)  # [19, n_freqs]
        mean_logp = np.log10(np.maximum(mean_psd_19, 1e-25)).mean(axis=0)  # global 19ch log10 PSD
        freqs_ref = freqs

        if item["group"] == "MDD":
            batch = "MDD_2013_04_07"
        elif mdate.startswith("2011"):
            batch = "HC_2011"
        elif mdate.startswith("2013-01"):
            batch = "HC_2013_01"
        else:
            batch = "HC_2013_04_07"

        def at_f(fval):
            return float(mean_logp[int(np.argmin(np.abs(freqs - fval)))])

        is_dup = sid in DUP_DROP_IDS
        p45, p50, p60 = at_f(45.0), at_f(50.0), at_f(60.0)
        audit_rows.append({
            "subject_id": sid,
            "filename": fpath.name,
            "group": item["group"],
            "y": int(item["label"]),
            "meas_date": mdate,
            "batch": batch,
            "is_duplicate_sha256": is_dup,
            "sha256_prefix": sha,
            "sfreq": sfreq,
            "highpass_hdr": hp,
            "lowpass_hdr": lp,
            "prefiltering_hdr": pref,
            "n_clean_epochs": int(len(ep_clean)),
            "logp_1hz": at_f(1.0),
            "logp_2hz": at_f(2.0),
            "logp_6hz": at_f(6.0),
            "logp_10hz": at_f(10.0),
            "logp_20hz": at_f(20.0),
            "logp_30hz": at_f(30.0),
            "logp_40hz": at_f(40.0),
            "logp_45hz": p45,
            "logp_50hz": p50,
            "logp_60hz": p60,
            "logp_80hz": at_f(80.0),
            "drop_2_to_1hz": at_f(1.0) - at_f(2.0),
            "drop_30_to_40hz": at_f(40.0) - at_f(30.0),
            "line_50hz_excess": p50 - 0.5 * (p45 + p60),
        })
        if not is_dup:
            f65 = (freqs >= 0.5) & (freqs <= 65.0)
            batch_curves[batch].append(mean_logp[f65])
            raw_unfilt_psds.append({
                "subject_id": sid,
                "y": int(item["label"]),
                "batch": batch,
                "freqs": freqs.astype(np.float64),
                "mean_psd": mean_psd_19,
            })

    df_audit = pd.DataFrame(audit_rows)
    df_audit.to_csv(REPORTS_DIR / "study2_unfiltered_psd_and_header_audit.csv", index=False)

    f65_vals = freqs_ref[(freqs_ref >= 0.5) & (freqs_ref <= 65.0)]
    batch_json = {"freqs": f65_vals.tolist(), "batches": {}}
    for bname, lst in batch_curves.items():
        arr = np.stack(lst)
        batch_json["batches"][bname] = {
            "n": len(lst),
            "mean": arr.mean(axis=0).tolist(),
            "sem": (arr.std(axis=0, ddof=1) / np.sqrt(len(lst))).tolist(),
        }
    with open(REPORTS_DIR / "study2_unfiltered_batch_spectra.json", "w", encoding="utf-8") as f:
        json.dump(batch_json, f)

    print(df_audit[~df_audit["is_duplicate_sha256"]].groupby("batch")[["logp_1hz", "logp_2hz", "logp_30hz", "logp_40hz", "logp_50hz", "logp_80hz", "drop_2_to_1hz", "drop_30_to_40hz", "line_50hz_excess"]].mean().round(3).to_string())
    return df_audit, raw_unfilt_psds


# ==============================================================================
# PART 2: DEDUPLICATE DERIVATIVES + LOAD CANONICAL BUNDLES + UPDATE AUX CSVs
# ==============================================================================
def backup_and_deduplicate():
    print("\n=== [Step 2] Deduplicating Study 2 (mumtaz_hc_030, mumtaz_mdd_034) Across Bundles ===")
    targets = [
        "all_7cohorts_welch_psds.pt",
        "specparam_vs_masked_ols_7cohorts.pt",
        "specparam_1to40_sensitivity_7cohorts.pt",
        "specparam_canonical_7cohorts.pt",
    ]
    for fn in targets:
        src = DATA_DIR / fn
        dst = BACKUP_DIR / fn
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"  Backed up v4 {fn} -> {dst.name}")

    # Load from backups (always start cleanly from nominal 58/60 snapshot so idempotent across re-runs!)
    psds = torch.load(BACKUP_DIR / "all_7cohorts_welch_psds.pt", weights_only=False)
    sp = torch.load(BACKUP_DIR / "specparam_vs_masked_ols_7cohorts.pt", weights_only=False)
    sp_140 = torch.load(BACKUP_DIR / "specparam_1to40_sensitivity_7cohorts.pt", weights_only=False)
    unm_ec_map = torch.load(DATA_DIR / "unm_ec_sensitivity_specparam.pt", weights_only=False)

    # Save unfiltered-dedup Study 2 sensitivity table BEFORE removing duplicates
    df_mum_all_diag = pd.read_csv(REPORTS_DIR / "study2_unfiltered_psd_and_header_audit.csv")
    # Recompute mumtaz_batch_and_pli_diagnostic.csv cleanly from nominal 58 -> annotate is_duplicate_sha256
    mum_pli_rows = []
    date_map = df_mum_all_diag.set_index("subject_id")["meas_date"].to_dict()
    for g in sp["mumtaz_ec"]:
        pm = g["pli"].numpy().astype(np.float64)
        C = 0.5 * (pm + pm.T)
        np.fill_diagonal(C, 1.0)
        ev = np.linalg.eigvalsh(C)
        mum_pli_rows.append({
            "subject_id": g["subject_id"],
            "group": "H" if int(g["y"]) == 0 else "MDD",
            "meas_date": date_map.get(g["subject_id"], ""),
            "is_duplicate_sha256": g["subject_id"] in DUP_DROP_IDS,
            "n_clean_epochs": int(g["n_clean_epochs"]),
            "min_eig_I_plus_PLI": float(ev.min()),
            "is_indefinite": bool(ev.min() <= 0.0),
            "median_pli": float(np.median(pm[np.triu_indices(19, k=1)])),
        })
    df_mum_diag = pd.DataFrame(mum_pli_rows)
    df_mum_diag.to_csv(REPORTS_DIR / "mumtaz_batch_and_pli_diagnostic.csv", index=False)

    # Run updated Study 2 duplicate & batch sensitivity with exact 95% CIs
    def cohens_d_ci_welch(a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        na, nb = len(a), len(b)
        sp_sd = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / max(1, na + nb - 2)) + 1e-12
        d = float((a.mean() - b.mean()) / sp_sd)
        se = float(np.sqrt((na + nb) / (na * nb) + (d ** 2) / (2.0 * (na + nb))))
        _, p = stats.ttest_ind(a, b, equal_var=False)
        return d, d - 1.96 * se, d + 1.96 * se, float(p)

    def eval_10x5_tab_safe(glist):
        vg = [g for g in glist if int(g["y"]) in (0, 1)]
        X = torch.stack([g["x_specparam_resid"] for g in vg])
        y = np.array([int(g["y"]) for g in vg])
        n0, n1 = int(np.sum(y == 0)), int(np.sum(y == 1))
        n_spl = min(5, n0, n1)
        aucs_lr, aucs_rf = [], []
        for s in REPEATS_10:
            skf = StratifiedKFold(n_splits=n_spl, shuffle=True, random_state=s)
            p_lr, p_rf = np.zeros(len(vg)), np.zeros(len(vg))
            for fold, (tr, va) in enumerate(skf.split(vg, y)):
                tz, vz, _, _ = compute_fold_normative_z(X[tr], torch.tensor(y[tr], dtype=torch.long), X[va])
                Xtr, Xva, ytr = tz.view(len(tr), -1).numpy(), vz.view(len(va), -1).numpy(), y[tr]
                min_tr = min(int(np.sum(ytr == 0)), int(np.sum(ytr == 1)))
                if min_tr >= 2:
                    icv = StratifiedKFold(n_splits=min(3, min_tr), shuffle=True, random_state=s + fold)
                    lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=s), {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}, cv=icv, scoring="roc_auc").fit(Xtr, ytr)
                else:
                    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=s).fit(Xtr, ytr)
                rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=s).fit(Xtr, ytr)
                p_lr[va] = lr.predict_proba(Xva)[:, 1]
                p_rf[va] = rf.predict_proba(Xva)[:, 1]
            aucs_lr.append(float(roc_auc_score(y, p_lr)))
            aucs_rf.append(float(roc_auc_score(y, p_rf)))
        return float(np.mean(aucs_lr)), float(np.std(aucs_lr, ddof=1)), float(np.mean(aucs_rf)), float(np.std(aucs_rf, ddof=1))

    diag_idx = df_mum_diag.set_index("subject_id")
    ec_nom = sp["mumtaz_ec"]
    eo_nom = sp["mumtaz_eo"]
    ec_dedup = [g for g in ec_nom if g["subject_id"] not in DUP_DROP_IDS]
    eo_dedup = [g for g in eo_nom if g["subject_id"] not in DUP_DROP_IDS]
    ec_no_jan13 = [g for g in ec_dedup if g["y"] == 1 or not str(diag_idx.loc[g["subject_id"], "meas_date"]).startswith("2013-01")]
    ec_vs_2011 = [g for g in ec_dedup if g["y"] == 1 or str(diag_idx.loc[g["subject_id"], "meas_date"]).startswith("2011")]
    ec_vs_jan13 = [g for g in ec_dedup if g["y"] == 1 or str(diag_idx.loc[g["subject_id"], "meas_date"]).startswith("2013-01")]
    ec_vs_late13 = [g for g in ec_dedup if g["y"] == 1 or str(diag_idx.loc[g["subject_id"], "meas_date"]).startswith(("2013-04", "2013-05", "2013-06", "2013-07"))]
    ec_strict_psd = [g for g in ec_dedup if not bool(diag_idx.loc[g["subject_id"], "is_indefinite"])]

    s2_rows = []
    for lbl, glist in [
        ("Study 2a EC: Public Manifest (30 MDD vs 28 HC, N=58)", ec_nom),
        ("Study 2a EC: Deduplicated Unique (29 MDD vs 27 HC, N=56)", ec_dedup),
        ("Study 2a EC: Deduplicated vs Jan-2013 HC Only (29 MDD vs 14 HC, N=43)", ec_vs_jan13),
        ("Study 2a EC: Deduplicated Excl. Jan-2013 HC Batch (29 MDD vs 13 HC, N=42)", ec_no_jan13),
        ("Study 2a EC: Deduplicated vs 2011 HC Batch Only (29 MDD vs 9 HC, N=38)", ec_vs_2011),
        ("Study 2a EC: Deduplicated vs Apr-Jul 2013 Date-Matched HC (29 MDD vs 4 HC, N=33)", ec_vs_late13),
        ("Study 2a EC: Deduplicated + Strictly PSD PLI Both Groups (22 MDD vs 12 HC, N=34)", ec_strict_psd),
        ("Study 2b EO: Public Manifest (32 MDD vs 28 HC, N=60)", eo_nom),
        ("Study 2b EO: Deduplicated Unique (31 MDD vs 27 HC, N=58)", eo_dedup),
    ]:
        cs = [g for g in glist if g["y"] == 1]
        rs = [g for g in glist if g["y"] == 0]
        thd, thlo, thhi, thp = cohens_d_ci_welch([g["x_specparam_resid"].numpy()[FOCAL_5, 0].mean() for g in cs], [g["x_specparam_resid"].numpy()[FOCAL_5, 0].mean() for g in rs])
        chd, chlo, chhi, chp = cohens_d_ci_welch([g["x_specparam_resid"].numpy()[:, 3].mean() for g in cs], [g["x_specparam_resid"].numpy()[:, 3].mean() for g in rs])
        lrm, lrs, rfm, rfs = eval_10x5_tab_safe(glist)
        s2_rows.append({
            "subset": lbl, "n_total": len(glist), "n_case": len(cs), "n_ctrl": len(rs),
            "theta_d": thd, "theta_ci_lo": thlo, "theta_ci_hi": thhi, "theta_p": thp,
            "chi_d": chd, "chi_ci_lo": chlo, "chi_ci_hi": chhi, "chi_p": chp,
            "lr_10x5_mean": lrm, "lr_10x5_sd": lrs, "rf_10x5_mean": rfm, "rf_10x5_sd": rfs,
        })
    # Also HC-only Jan-2013 vs 2011 batch check
    hc_2011 = [g for g in ec_dedup if g["y"] == 0 and str(diag_idx.loc[g["subject_id"], "meas_date"]).startswith("2011")]
    hc_jan13 = [g for g in ec_dedup if g["y"] == 0 and str(diag_idx.loc[g["subject_id"], "meas_date"]).startswith("2013-01")]
    th_hc_d, th_hc_lo, th_hc_hi, th_hc_p = cohens_d_ci_welch([g["x_specparam_resid"].numpy()[FOCAL_5, 0].mean() for g in hc_jan13], [g["x_specparam_resid"].numpy()[FOCAL_5, 0].mean() for g in hc_2011])
    ch_hc_d, ch_hc_lo, ch_hc_hi, ch_hc_p = cohens_d_ci_welch([g["x_specparam_resid"].numpy()[:, 3].mean() for g in hc_jan13], [g["x_specparam_resid"].numpy()[:, 3].mean() for g in hc_2011])
    s2_rows.append({
        "subset": "Study 2 HC-Only Batch Effect: Jan-2013 HC (n=14) vs 2011 HC (n=9)",
        "n_total": len(hc_jan13) + len(hc_2011), "n_case": len(hc_jan13), "n_ctrl": len(hc_2011),
        "theta_d": th_hc_d, "theta_ci_lo": th_hc_lo, "theta_ci_hi": th_hc_hi, "theta_p": th_hc_p,
        "chi_d": ch_hc_d, "chi_ci_lo": ch_hc_lo, "chi_ci_hi": ch_hc_hi, "chi_p": ch_hc_p,
        "lr_10x5_mean": np.nan, "lr_10x5_sd": np.nan, "rf_10x5_mean": np.nan, "rf_10x5_sd": np.nan,
    })
    pd.DataFrame(s2_rows).to_csv(REPORTS_DIR / "study2_duplicate_and_batch_sensitivity.csv", index=False)
    # Save SHA-256 duplicates table cleanly
    by_hash = {}
    for p in sorted(MUMTAZ_DIR.glob("*.edf")):
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        by_hash.setdefault(h, []).append(p.name)
    dup_rows = [{"sha256_prefix": h, "n_files": len(ns), "filenames": "; ".join(ns)} for h, ns in sorted(by_hash.items(), key=lambda kv: kv[1][0]) if len(ns) > 1]
    pd.DataFrame(dup_rows).to_csv(REPORTS_DIR / "mumtaz_duplicate_edf_hashes.csv", index=False)

    # Apply deduplication across primary bundles
    for d_obj in (psds, sp, sp_140):
        d_obj["mumtaz_ec"] = [r for r in d_obj["mumtaz_ec"] if r["subject_id"] not in DUP_DROP_IDS]
        d_obj["mumtaz_eo"] = [r for r in d_obj["mumtaz_eo"] if r["subject_id"] not in DUP_DROP_IDS]
        assert len(d_obj["mumtaz_ec"]) == 56 and len(d_obj["mumtaz_eo"]) == 58

    cav_tsv = pd.read_csv(CAVANAGH_DIR / "participants.tsv", sep="\t")
    cav_age = {str(r["participant_id"]): (float(r["age"]) if pd.notnull(r["age"]) else np.nan) for _, r in cav_tsv.iterrows()}
    ucsd_tsv = pd.read_csv(EEG_ROOT / "participants.tsv", sep="\t")
    ucsd_age = {str(r["participant_id"]): (float(r["age"]) if pd.notnull(r["age"]) else np.nan) for _, r in ucsd_tsv.iterrows()}

    for ckey, recs in sp.items():
        psd_recs = psds[ckey]
        s140_recs = sp_140[ckey]
        freqs = np.asarray(psd_recs[0]["freqs"], dtype=float)
        fg = SpectralGroupModel(**SPECPARAM_KWARGS)
        post_psds = np.stack([np.maximum(np.asarray(pr["mean_psd"], dtype=float)[POSTERIOR_5], 1e-25) for pr in psd_recs])
        fg.fit(freqs, post_psds.reshape(len(recs) * len(POSTERIOR_5), -1), [2.0, 30.0])

        for idx, r in enumerate(recs):
            assert np.all(r["sp_fit_ok"]), f"Failed fit in {ckey} idx={idx}"
            r["x"] = r["x_specparam_resid"].clone()
            r["x_specparam_1to40"] = s140_recs[idx]["x_specparam_resid"].clone()
            r["x_masked_ols_1to40"] = s140_recs[idx]["x_masked_ols"].clone()
            r["x_masked_ols_2to30"] = r["x_masked_ols_win"].clone()
            r["adj_weighted"] = (r["adj"] * r["pli"]).clone()
            r["omst_edge_count"] = int(np.sum(np.triu(r["adj"].numpy() > 0.5, k=1)))

            if ckey == "cavanagh_ds003478":
                r["age"] = cav_age.get(r["subject_id"].replace("unm_", ""), np.nan)
            elif ckey == "ucsd_ds002778":
                r["age"] = ucsd_age.get(r["bids_id"], np.nan)

            ch_peaks = []
            for ch_off in range(len(POSTERIOR_5)):
                m_res = fg.get_model(idx * len(POSTERIOR_5) + ch_off, regenerate=False)
                pks = m_res.results.params.periodic.params
                if pks.size > 0:
                    apks = pks[(pks[:, 0] >= 7.0) & (pks[:, 0] <= 13.5)]
                    if len(apks) > 0:
                        ch_peaks.append(float(apks[np.argmax(apks[:, 1]), 0]))
            r["iapf_posterior"] = float(np.median(ch_peaks)) if ch_peaks else np.nan

    torch.save(psds, DATA_DIR / "all_7cohorts_welch_psds.pt")
    torch.save(sp, DATA_DIR / "specparam_vs_masked_ols_7cohorts.pt")
    torch.save(sp_140, DATA_DIR / "specparam_1to40_sensitivity_7cohorts.pt")
    torch.save(sp, DATA_DIR / "specparam_canonical_7cohorts.pt")

    # Update deduplicated specparam_fit_diagnostics_by_cohort.csv & cohort_epoch_duration_audit.csv
    diag_fit_rows = []
    for win_lbl, d_obj in [("1.0-40.0", sp_140), ("2.0-30.0", sp)]:
        for ckey, recs in d_obj.items():
            r2_arr = np.concatenate([np.asarray(r["sp_r2"], dtype=float) for r in recs])
            mae_arr = np.concatenate([np.asarray(r["sp_mae"], dtype=float) for r in recs])
            npk_arr = np.concatenate([np.asarray(r["sp_n_peaks"], dtype=float) for r in recs])
            chi_o = np.concatenate([np.asarray(r["x_masked_ols"][:, 3].numpy(), dtype=float) for r in recs])
            chi_s = np.concatenate([np.asarray(r["x_specparam_resid"][:, 3].numpy(), dtype=float) for r in recs])
            diag_fit_rows.append({
                "fit_window_hz": win_lbl,
                "cohort_key": ckey,
                "n_sessions": len(recs),
                "n_spectra": len(recs) * 19,
                "n_failed": int(np.sum([int(np.sum(~r["sp_fit_ok"])) for r in recs])),
                "pct_r2_lt_080": float(100.0 * np.mean(r2_arr < 0.80)),
                "r2_p05": float(np.percentile(r2_arr, 5)),
                "r2_median": float(np.median(r2_arr)),
                "r2_mean": float(np.mean(r2_arr)),
                "mae_median": float(np.median(mae_arr)),
                "mae_mean": float(np.mean(mae_arr)),
                "mean_n_peaks": float(np.mean(npk_arr)),
                "mean_chi_ols": float(np.mean(chi_o)),
                "mean_chi_specparam": float(np.mean(chi_s)),
            })
    pd.DataFrame(diag_fit_rows).to_csv(REPORTS_DIR / "specparam_fit_diagnostics_by_cohort.csv", index=False)

    ep_rows = []
    for c_lbl, c_key in [
        ("Study 1: Sleep Deprivation (ds004902)", "sleep_ds004902"),
        ("Study 2a: Outpatient MDD EC (4244171)", "mumtaz_ec"),
        ("Study 2b: Outpatient MDD EO (4244171)", "mumtaz_eo"),
        ("Study 3: Univ. Depression (ds003478)", "cavanagh_ds003478"),
        ("Study 4: UCSD PD ON/OFF (ds002778)", "ucsd_ds002778"),
        ("Study 5: AHEPA Dementia (ds004504)", "dementia_ds004504"),
        ("Study 6: UNM PD Pure EO (ds003490)", "unm_ds003490"),
        ("Study 7: Iowa PD (ds004584)", "iowa_ds004584"),
    ]:
        recs = sp[c_key]
        ep0 = [float(r["n_clean_epochs"]) for r in recs if int(r["y"]) == 0]
        ep1 = [float(r["n_clean_epochs"]) for r in recs if int(r["y"]) == 1]
        all_ep = [float(r["n_clean_epochs"]) for r in recs]
        _, p_ep = stats.ttest_rel(ep1, ep0) if c_key == "sleep_ds004902" else stats.ttest_ind(ep1, ep0, equal_var=False)
        ep_rows.append({
            "cohort": c_lbl, "n_sessions": len(recs),
            "all_mean": float(np.mean(all_ep)), "all_sd": float(np.std(all_ep, ddof=1)),
            "all_min": int(np.min(all_ep)), "all_max": int(np.max(all_ep)),
            "case_mean": float(np.mean(ep1)), "case_sd": float(np.std(ep1, ddof=1)),
            "ctrl_mean": float(np.mean(ep0)), "ctrl_sd": float(np.std(ep0, ddof=1)),
            "case_vs_ctrl_p": float(p_ep),
        })
    pd.DataFrame(ep_rows).to_csv(REPORTS_DIR / "cohort_epoch_duration_audit.csv", index=False)

    print("✓ Deduplicated derivatives: EC=56 (29 vs 27), EO=58 (31 vs 27); Primary=675 sessions (564 participants; 733 total = 13,927 spectra).")
    return sp, sp_140, psds, unm_ec_map


# ==============================================================================
# PART 3: MODULE A ON DEDUPLICATED DATA + K=5 OMIT-STUDY2 META-ANALYSIS + AUX
# ==============================================================================
def make_ucsd_subj_mean(ucsd):
    by_bid = {}
    for g in ucsd:
        if "PD" in g.get("group", ""):
            by_bid.setdefault(g["bids_id"], []).append(g)
    pd_mean = []
    for bid in sorted(by_bid.keys()):
        pair = by_bid[bid]
        rec = {
            "bids_id": bid, "subject_id": bid, "group": "PD_SubjMean",
            "age": float(pair[0].get("age", np.nan)),
            "n_clean_epochs": float(np.mean([p["n_clean_epochs"] for p in pair])),
            "gce": float(np.mean([float(p["gce"]) for p in pair])),
            "iapf_posterior": float(np.nanmean([p.get("iapf_posterior", np.nan) for p in pair])),
        }
        for k in ["x", "x_specparam_resid", "x_specparam_peaks", "x_specparam_1to40", "x_masked_ols_1to40", "x_masked_ols_2to30"]:
            rec[k] = torch.stack([p[k] for p in pair]).mean(dim=0)
        pd_mean.append(rec)
    return pd_mean, [g for g in ucsd if g.get("group") == "HC"]


def run_module_a_and_aux(sp, psds, unm_ec_map):
    print("\n=== [Step 3] Module A Biophysics, K=6 & K=5 Meta-Analysis, Table S1, Extra Biophysics ===")
    sleep, mum_ec, mum_eo = sp["sleep_ds004902"], sp["mumtaz_ec"], sp["mumtaz_eo"]
    cav, ucsd, dem, unm, iowa = sp["cavanagh_ds003478"], sp["ucsd_ds002778"], sp["dementia_ds004504"], sp["unm_ds003490"], sp["iowa_ds004584"]

    all_primary = sleep + mum_ec + cav + ucsd + dem + unm + iowa
    assert len(all_primary) == 675

    sd_s = sorted([g for g in sleep if g["condition"] == "SD"], key=lambda r: r["subject_id"])
    ns_s = sorted([g for g in sleep if g["condition"] == "NS"], key=lambda r: r["subject_id"])
    ucsd_mean, ucsd_hc = make_ucsd_subj_mean(ucsd)
    ucsd_off = [g for g in ucsd if g.get("group") == "PD_OFF"]
    ucsd_on = [g for g in ucsd if g.get("group") == "PD_ON"]
    unm_off = [g for g in unm if g.get("med_state") == "OFF"]
    unm_on = [g for g in unm if g.get("med_state") == "ON"]
    unm_hc = [g for g in unm if g.get("med_state") == "CTL"]

    # Update extra_biophysics_summary.json dynamically
    sl_gth = compute_es(sd_s, ns_s, lambda g: float(g["x"].numpy()[:, 0].mean()), paired=True)
    uc_all_th = compute_es(ucsd_off + ucsd_on, ucsd_hc, lambda g: float(g["x"].numpy()[FOCAL_5, 0].mean()), paired=False)
    eo_cs, eo_rs = [g for g in mum_eo if g["y"] == 1], [g for g in mum_eo if g["y"] == 0]
    eo_th = compute_es(eo_cs, eo_rs, lambda g: float(g["x"].numpy()[FOCAL_5, 0].mean()))
    eo_dl = compute_es(eo_cs, eo_rs, lambda g: float(g["x"].numpy()[:, 1].mean()))
    eo_ch = compute_es(eo_cs, eo_rs, lambda g: float(g["x"].numpy()[:, 3].mean()))
    extra_bio = {
        "Sleep_global_theta": {"cohens_d": sl_gth["cohens_d"], "p_value": sl_gth["p_value"]},
        "UCSD_All_theta": {"cohens_d": uc_all_th["cohens_d"], "p_value": uc_all_th["p_value"]},
        "Mumtaz_EO_theta": {"cohens_d": eo_th["cohens_d"], "p_value": eo_th["p_value"]},
        "Mumtaz_EO_delta": {"cohens_d": eo_dl["cohens_d"], "p_value": eo_dl["p_value"]},
        "Mumtaz_EO_chi": {"cohens_d": eo_ch["cohens_d"], "p_value": eo_ch["p_value"]},
    }
    with open(REPORTS_DIR / "extra_biophysics_summary.json", "w", encoding="utf-8") as f:
        json.dump(extra_bio, f, indent=2)

    contrasts_10 = [
        ("Study 1: Acute Sleep Deprivation", sd_s, ns_s, True),
        ("Study 2: Outpatient MDD", [g for g in mum_ec if g["y"] == 1], [g for g in mum_ec if g["y"] == 0], False),
        ("Study 3: Univ. Depression Screening", [g for g in cav if g["y"] == 1], [g for g in cav if g["y"] == 0], False),
        ("Study 4: UCSD Parkinson All", ucsd_mean, ucsd_hc, False),
        ("Study 5a: Alzheimer's Disease (AD)", [g for g in dem if g["diagnosis"] == "Alzheimer"], [g for g in dem if g["diagnosis"] == "Control"], False),
        ("Study 5b: Frontotemporal Dementia (FTD)", [g for g in dem if g["diagnosis"] == "FTD"], [g for g in dem if g["diagnosis"] == "Control"], False),
        ("Study 5c: Clinical Dementia (All)", [g for g in dem if g["y"] == 1], [g for g in dem if g["y"] == 0], False),
        ("Study 6a: UNM Parkinson OFF", unm_off, unm_hc, False),
        ("Study 6b: UNM Parkinson ON", unm_on, unm_hc, False),
        ("Study 7: Iowa Parkinson Cohort", [g for g in iowa if g["y"] == 1], [g for g in iowa if g["y"] == 0], False),
    ]

    marker_defs = [
        ("frontal_theta", "Frontal Periodic Theta (4–8 Hz)", lambda g, k="x": float(g[k].numpy()[FOCAL_5, 0].mean())),
        ("global_delta", "Global Low-Freq Periodic Residual (2–4 Hz)", lambda g, k="x": float(g[k].numpy()[:, 1].mean())),
        ("posterior_alpha", "Posterior Periodic Alpha (8–13 Hz)", lambda g, k="x": float(g[k].numpy()[POSTERIOR_5, 2].mean())),
        ("global_chi", "Global Aperiodic Exponent (χ)", lambda g, k="x": float(g[k].numpy()[:, 3].mean())),
        ("omst_gce", "OMST Global Cost Efficiency (GCE)", lambda g, k="x": float(g["gce"])),
    ]
    exp_rows = []
    for clbl, cs, rs, isp in contrasts_10:
        for mk, mn, fn in marker_defs:
            exp_rows.append({"cohort": clbl, "marker_key": mk, "marker": mn, **compute_es(cs, rs, fn, paired=isp)})
    df_exp = pd.DataFrame(exp_rows)
    _, q50, _, _ = multipletests(df_exp["p_value"].values, method="fdr_bh")
    df_exp["fdr_q_50"] = q50
    df_exp["fdr_q_10"] = np.nan
    for mk, idx in df_exp.groupby("marker_key").groups.items():
        _, q10, _, _ = multipletests(df_exp.loc[idx, "p_value"].values, method="fdr_bh")
        df_exp.loc[idx, "fdr_q_10"] = q10
    df_exp.to_csv(REPORTS_DIR / "multicohort_biophysical_effect_sizes_expanded.csv", index=False)

    sub_rows = [{"cohort": clbl, **compute_es(cs, rs, lambda g: float(g["x"].numpy()[FOCAL_3, 0].mean()), paired=isp)} for clbl, cs, rs, isp in contrasts_10]
    pd.DataFrame(sub_rows).to_csv(REPORTS_DIR / "biophysics_f3_fz_f4_sensitivity.csv", index=False)

    methods = [
        ("masked_ols_1to40", "x_masked_ols_1to40"),
        ("masked_ols_2to30", "x_masked_ols_2to30"),
        ("specparam_resid_2to30", "x_specparam_resid"),
        ("specparam_peaks_2to30", "x_specparam_peaks"),
        ("specparam_resid_1to40", "x_specparam_1to40"),
    ]
    ab_rows, boot_rows = [], []
    for clbl, cs, rs, isp in contrasts_10:
        for mk, idx_col, chans in [("frontal_theta_5ch", 0, FOCAL_5), ("global_chi", 3, None)]:
            for m_name, xk in methods:
                fn = (lambda g, col=idx_col, ch=chans, key=xk: float(g[key].numpy()[:, col].mean() if ch is None else g[key].numpy()[ch, col].mean()))
                ab_rows.append({"method": m_name, "cohort": clbl, "marker": mk, **compute_es(cs, rs, fn, paired=isp)})
            fn_sp = lambda g, col=idx_col, ch=chans: float(g["x_specparam_resid"].numpy()[:, col].mean() if ch is None else g["x_specparam_resid"].numpy()[ch, col].mean())
            fn_o1 = lambda g, col=idx_col, ch=chans: float(g["x_masked_ols_1to40"].numpy()[:, col].mean() if ch is None else g["x_masked_ols_1to40"].numpy()[ch, col].mean())
            fn_o2 = lambda g, col=idx_col, ch=chans: float(g["x_masked_ols_2to30"].numpy()[:, col].mean() if ch is None else g["x_masked_ols_2to30"].numpy()[ch, col].mean())
            fn_s140 = lambda g, col=idx_col, ch=chans: float(g["x_specparam_1to40"].numpy()[:, col].mean() if ch is None else g["x_specparam_1to40"].numpy()[ch, col].mean())
            for cmp_lbl, fn_alt in [("OLS_1_40_minus_SP_2_30", fn_o1), ("OLS_2_30_minus_SP_2_30", fn_o2), ("SP_1_40_minus_SP_2_30", fn_s140)]:
                da, db, dd, lo, hi, pv = bootstrap_delta_d(cs, rs, fn_alt, fn_sp, paired=isp)
                boot_rows.append({"cohort": clbl, "marker": mk, "comparison": cmp_lbl, "d_alt": da, "d_sp_2_30": db, "delta_d": dd, "ci_lower": lo, "ci_upper": hi, "boot_p": pv})
    df_ab = pd.DataFrame(ab_rows)
    df_ab["fdr_q_10"] = np.nan
    for (_, _), idx in df_ab.groupby(["method", "marker"]).groups.items():
        _, q10, _, _ = multipletests(df_ab.loc[idx, "p_value"].values, method="fdr_bh")
        df_ab.loc[idx, "fdr_q_10"] = q10
    df_ab.to_csv(REPORTS_DIR / "ablation_specparam_vs_ols_biophysics.csv", index=False)
    df_ab_w = df_ab.pivot(index=["marker", "cohort"], columns="method", values=["cohens_d", "p_value", "fdr_q_10"])
    df_ab_w.columns = [f"{a}_{b}" for a, b in df_ab_w.columns]
    df_ab_w.reset_index().to_csv(REPORTS_DIR / "ablation_specparam_vs_ols_biophysics_wide.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(REPORTS_DIR / "ablation_specparam_vs_ols_bootstrap_deltas.csv", index=False)

    def dl_pool(items):
        ds = np.array([x[0] for x in items], dtype=float)
        vs = np.array([x[1] for x in items], dtype=float)
        k = len(ds)
        w = 1.0 / vs
        if k == 1:
            d_re = float(ds[0])
            se_re = float(np.sqrt(vs[0]))
            p_dl = float(2.0 * (1.0 - stats.norm.cdf(abs(d_re / se_re))))
            return {
                "k_studies": 1, "pooled_d": d_re, "se_re": se_re,
                "ci_lower": d_re - 1.96 * se_re, "ci_upper": d_re + 1.96 * se_re,
                "hksj_ci_lower": d_re - 1.96 * se_re, "hksj_ci_upper": d_re + 1.96 * se_re,
                "p_value": p_dl, "p_hksj": p_dl, "Q_stat": 0.0, "Q_df": 0, "tau2": 0.0, "I2_percent": 0.0,
            }
        fe = float(np.sum(w * ds) / np.sum(w))
        Q = float(np.sum(w * (ds - fe) ** 2))
        df_q = k - 1
        c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
        tau2 = float(max(0.0, (Q - df_q) / c)) if c > 0 else 0.0
        w_re = 1.0 / (vs + tau2)
        d_re = float(np.sum(w_re * ds) / np.sum(w_re))
        se_re = float(np.sqrt(1.0 / np.sum(w_re)))
        p_dl = float(2.0 * (1.0 - stats.norm.cdf(abs(d_re / se_re))))
        q_hksj = float(max(1.0, Q / df_q))
        se_hk = float(np.sqrt(q_hksj) * se_re)
        t_c = float(stats.t.ppf(0.975, df=df_q))
        p_hk = float(2.0 * (1.0 - stats.t.cdf(abs(d_re / se_hk), df=df_q)))
        i2 = float(max(0.0, (Q - df_q) / Q * 100.0)) if Q > 0 else 0.0
        return {
            "k_studies": k, "pooled_d": d_re, "se_re": se_re,
            "ci_lower": d_re - 1.96 * se_re, "ci_upper": d_re + 1.96 * se_re,
            "hksj_ci_lower": d_re - t_c * se_hk, "hksj_ci_upper": d_re + t_c * se_hk,
            "p_value": p_dl, "p_hksj": p_hk, "Q_stat": Q, "Q_df": df_q, "tau2": tau2, "I2_percent": i2,
        }

    meta_rows = []
    for m_name, xk in [("specparam_2to30_resid", "x_specparam_resid"), ("specparam_1to40_resid", "x_specparam_1to40"), ("masked_ols_1to40", "x_masked_ols_1to40")]:
        th_fn = lambda g, k=xk: float(g[k].numpy()[FOCAL_5, 0].mean())
        es_map = {
            "Study2_Mumtaz_MDD": compute_es([g for g in mum_ec if g["y"] == 1], [g for g in mum_ec if g["y"] == 0], th_fn),
            "Study3_Cavanagh_Dep": compute_es([g for g in cav if g["y"] == 1], [g for g in cav if g["y"] == 0], th_fn),
            "Study4_UCSD_SubjMean": compute_es(ucsd_mean, ucsd_hc, th_fn),
            "Study4_UCSD_OFF": compute_es(ucsd_off, ucsd_hc, th_fn),
            "Study4_UCSD_ON": compute_es(ucsd_on, ucsd_hc, th_fn),
            "Study5_Dementia_All": compute_es([g for g in dem if g["y"] == 1], [g for g in dem if g["y"] == 0], th_fn),
            "Study5a_Dementia_AD": compute_es([g for g in dem if g["diagnosis"] == "Alzheimer"], [g for g in dem if g["diagnosis"] == "Control"], th_fn),
            "Study5b_Dementia_FTD": compute_es([g for g in dem if g["diagnosis"] == "FTD"], [g for g in dem if g["diagnosis"] == "Control"], th_fn),
            "Study6a_UNM_PD_OFF": compute_es(unm_off, unm_hc, th_fn),
            "Study6b_UNM_PD_ON": compute_es(unm_on, unm_hc, th_fn),
            "Study7_Iowa_PD": compute_es([g for g in iowa if g["y"] == 1], [g for g in iowa if g["y"] == 0], th_fn),
        }
        v_fn = lambda k: (es_map[k]["cohens_d"], es_map[k]["se_d"] ** 2)
        s_sd = np.array([th_fn(g) for g in sd_s])
        s_ns = np.array([th_fn(g) for g in ns_s])
        d_sl = float((s_sd.mean() - s_ns.mean()) / np.sqrt(0.5 * (np.var(s_sd, ddof=1) + np.var(s_ns, ddof=1))))
        v_sl = float((71 + 71) / (71 * 71) + (d_sl ** 2) / (2 * 142))

        dep_k2 = [v_fn("Study2_Mumtaz_MDD"), v_fn("Study3_Cavanagh_Dep")]
        dep_k1 = [v_fn("Study3_Cavanagh_Dep")]
        configs = [
            ("Primary_K6_SubjMean_PD_OFF", dep_k2, [v_fn("Study4_UCSD_SubjMean"), v_fn("Study5_Dementia_All"), v_fn("Study6a_UNM_PD_OFF"), v_fn("Study7_Iowa_PD")]),
            ("Sens_K5_Omit_Study2_BatchConfounded", dep_k1, [v_fn("Study4_UCSD_SubjMean"), v_fn("Study5_Dementia_All"), v_fn("Study6a_UNM_PD_OFF"), v_fn("Study7_Iowa_PD")]),
            ("Sens1_K6_UCSD_OFF", dep_k2, [v_fn("Study4_UCSD_OFF"), v_fn("Study5_Dementia_All"), v_fn("Study6a_UNM_PD_OFF"), v_fn("Study7_Iowa_PD")]),
            ("Sens2_K6_PD_ON_Med", dep_k2, [v_fn("Study4_UCSD_ON"), v_fn("Study5_Dementia_All"), v_fn("Study6b_UNM_PD_ON"), v_fn("Study7_Iowa_PD")]),
            ("Sens3_K6_AD_Only", dep_k2, [v_fn("Study4_UCSD_SubjMean"), v_fn("Study5a_Dementia_AD"), v_fn("Study6a_UNM_PD_OFF"), v_fn("Study7_Iowa_PD")]),
            ("Sens4_K6_FTD_Only", dep_k2, [v_fn("Study4_UCSD_SubjMean"), v_fn("Study5b_Dementia_FTD"), v_fn("Study6a_UNM_PD_OFF"), v_fn("Study7_Iowa_PD")]),
            ("Sens5_K7_Plus_Sleep", dep_k2, [(d_sl, v_sl), v_fn("Study4_UCSD_SubjMean"), v_fn("Study5_Dementia_All"), v_fn("Study6a_UNM_PD_OFF"), v_fn("Study7_Iowa_PD")]),
            ("LOO_Omit_UCSD", dep_k2, [v_fn("Study5_Dementia_All"), v_fn("Study6a_UNM_PD_OFF"), v_fn("Study7_Iowa_PD")]),
            ("LOO_Omit_Dementia", dep_k2, [v_fn("Study4_UCSD_SubjMean"), v_fn("Study6a_UNM_PD_OFF"), v_fn("Study7_Iowa_PD")]),
            ("LOO_Omit_UNM", dep_k2, [v_fn("Study4_UCSD_SubjMean"), v_fn("Study5_Dementia_All"), v_fn("Study7_Iowa_PD")]),
            ("LOO_Omit_Iowa", dep_k2, [v_fn("Study4_UCSD_SubjMean"), v_fn("Study5_Dementia_All"), v_fn("Study6a_UNM_PD_OFF")]),
        ]
        for cfg, d1, d2 in configs:
            p1, p2, pall = dl_pool(d1), dl_pool(d2), dl_pool(d1 + d2)
            qb = float(((p1["pooled_d"] - p2["pooled_d"]) ** 2) / (p1["se_re"] ** 2 + p2["se_re"] ** 2))
            pb = float(1.0 - stats.chi2.cdf(qb, df=1))
            sub1_lbl = "Depression_Cohorts_k1" if len(d1) == 1 else "Depression_Cohorts_k2"
            for sub_lbl, pd_obj in [(sub1_lbl, p1), ("PD_Dementia_Cohorts", p2), ("Overall_Pooled", pall)]:
                meta_rows.append({"method": m_name, "analysis_set": cfg, "subgroup": sub_lbl, **pd_obj, "Q_between": qb, "p_between": pb})
    pd.DataFrame(meta_rows).to_csv(REPORTS_DIR / "meta_analysis_frontal_theta_polarity.csv", index=False)

    # ANCOVA (Duration & Age)
    anc_rows = []
    age_cohorts = [
        ("Study 3: Univ. Depression (N=118)", [g for g in cav if pd.notnull(g.get("age"))]),
        ("Study 4a: UCSD PD-OFF (N=31)", [g for g in ucsd_off + ucsd_hc if pd.notnull(g.get("age"))]),
        ("Study 4b: UCSD PD-ON (N=31)", [g for g in ucsd_on + ucsd_hc if pd.notnull(g.get("age"))]),
        ("Study 5c: Dementia All (N=88)", dem),
        ("Study 6a: UNM PD-OFF (N=50)", unm_off + unm_hc),
        ("Study 6b: UNM PD-ON (N=50)", unm_on + unm_hc),
        ("Study 7: Iowa PD (N=149)", iowa),
    ]
    for clbl, glist in age_cohorts:
        yv = np.array([g["x"].numpy()[FOCAL_5, 0].mean() for g in glist])
        grp = np.array([float(g["y"]) for g in glist])
        age = np.array([float(g["age"]) for g in glist])
        ep = np.array([float(g["n_clean_epochs"]) for g in glist])
        m_age = sm.OLS(yv, sm.add_constant(np.column_stack([grp, age]))).fit()
        m_aep = sm.OLS(yv, sm.add_constant(np.column_stack([grp, age, ep]))).fit()
        anc_rows.append({
            "cohort": clbl, "n": len(glist), "marker": "Frontal_Theta_2to30",
            "age_only_grp_t": float(m_age.tvalues[1]), "age_only_grp_p": float(m_age.pvalues[1]), "age_p": float(m_age.pvalues[2]),
            "age_plus_epoch_grp_t": float(m_aep.tvalues[1]), "age_plus_epoch_grp_p": float(m_aep.pvalues[1]), "epoch_p": float(m_aep.pvalues[3]),
        })
    yv2 = np.array([g["x"].numpy()[FOCAL_5, 0].mean() for g in mum_ec])
    grp2 = np.array([float(g["y"]) for g in mum_ec])
    ep2 = np.array([float(g["n_clean_epochs"]) for g in mum_ec])
    m2 = sm.OLS(yv2, sm.add_constant(np.column_stack([grp2, ep2]))).fit()
    anc_rows.append({
        "cohort": "Study 2a: Outpatient MDD EC (N=56)", "n": len(mum_ec), "marker": "Frontal_Theta_2to30",
        "age_only_grp_t": np.nan, "age_only_grp_p": np.nan, "age_p": np.nan,
        "age_plus_epoch_grp_t": float(m2.tvalues[1]), "age_plus_epoch_grp_p": float(m2.pvalues[1]), "epoch_p": float(m2.pvalues[2]),
    })
    pd.DataFrame(anc_rows).to_csv(REPORTS_DIR / "multicohort_age_and_duration_ancova.csv", index=False)

    anc_full = []
    for clbl, glist in age_cohorts:
        grp = np.array([float(g["y"]) for g in glist])
        age = np.array([float(g["age"]) for g in glist])
        X = sm.add_constant(np.column_stack([grp, age]))
        for m_lbl, fn in [
            ("Frontal Periodic Theta (4–8 Hz)", lambda g: float(g["x"].numpy()[FOCAL_5, 0].mean())),
            ("Global Low-Freq Residual (2–4 Hz)", lambda g: float(g["x"].numpy()[:, 1].mean())),
            ("Posterior Periodic Alpha (8–13 Hz)", lambda g: float(g["x"].numpy()[POSTERIOR_5, 2].mean())),
            ("Global Aperiodic Exponent (χ)", lambda g: float(g["x"].numpy()[:, 3].mean())),
        ]:
            mod = sm.OLS(np.array([fn(g) for g in glist]), X).fit()
            anc_full.append({
                "cohort": clbl, "marker": m_lbl,
                "group_coef": float(mod.params[1]), "group_se": float(mod.bse[1]),
                "group_t": float(mod.tvalues[1]), "group_p": float(mod.pvalues[1]),
                "age_coef": float(mod.params[2]), "age_p": float(mod.pvalues[2]), "r2": float(mod.rsquared),
            })
    pd.DataFrame(anc_full).to_csv(REPORTS_DIR / "multicohort_age_ancova_complete.csv", index=False)

    med_rows = []
    for c_lbl, on_gs, off_gs in [
        ("Study 6: UNM PD Pure EO (>=15h Washout) (N=25)", sorted(unm_on, key=lambda r: r["subject_id"]), sorted(unm_off, key=lambda r: r["subject_id"])),
        ("Study 4: UCSD PD (Acute Washout) (N=15)", sorted(ucsd_on, key=lambda r: r["bids_id"]), sorted(ucsd_off, key=lambda r: r["bids_id"])),
    ]:
        for mlbl, fn in [
            ("Frontal Periodic Theta (4–8 Hz)", lambda g: float(g["x"].numpy()[FOCAL_5, 0].mean())),
            ("Global Periodic Theta (4–8 Hz)", lambda g: float(g["x"].numpy()[:, 0].mean())),
            ("Global Low-Freq Residual (2–4 Hz)", lambda g: float(g["x"].numpy()[:, 1].mean())),
            ("Posterior Periodic Alpha (8–13 Hz)", lambda g: float(g["x"].numpy()[POSTERIOR_5, 2].mean())),
            ("Global Periodic Alpha (8–13 Hz)", lambda g: float(g["x"].numpy()[:, 2].mean())),
            ("Global Aperiodic Exponent (χ)", lambda g: float(g["x"].numpy()[:, 3].mean())),
            ("OMST Global Cost Efficiency (GCE)", lambda g: float(g["gce"])),
        ]:
            vo, vf = np.array([fn(g) for g in on_gs]), np.array([fn(g) for g in off_gs])
            st = compute_es(on_gs, off_gs, fn, paired=True)
            _, wp = wilcoxon(vo, vf)
            med_rows.append({
                "cohort": c_lbl, "marker": mlbl, "mean_off": float(vf.mean()), "mean_on": float(vo.mean()),
                "mean_diff": float((vo - vf).mean()), "paired_dz": st["cohens_d"],
                "ci_lower": st["ci_lower"], "ci_upper": st["ci_upper"], "paired_p": st["p_value"], "wilcoxon_p": float(wp),
            })
    pd.DataFrame(med_rows).to_csv(REPORTS_DIR / "unm_ucsd_paired_medication_biophysics.csv", index=False)

    unm_ec_230 = unm_ec_map["2.0-30.0"]
    unm_sens_rows = []
    for cond, cs, rs in [
        ("UNM Pure EO (N=25 PD-OFF vs 25 CTL)", unm_off, unm_hc),
        ("UNM Pure EO (N=25 PD-ON vs 25 CTL)", unm_on, unm_hc),
        ("UNM Pure EC (N=25 PD-OFF vs 24 CTL)", [g for g in unm_ec_230 if g["med_state"] == "OFF"], [g for g in unm_ec_230 if g["med_state"] == "CTL"]),
        ("UNM Pure EC (N=25 PD-ON vs 24 CTL)", [g for g in unm_ec_230 if g["med_state"] == "ON"], [g for g in unm_ec_230 if g["med_state"] == "CTL"]),
    ]:
        st_th = compute_es(cs, rs, lambda g: float(g["x_specparam_resid"].numpy()[FOCAL_5, 0].mean()))
        st_al = compute_es(cs, rs, lambda g: float(g["x_specparam_resid"].numpy()[POSTERIOR_5, 2].mean()))
        st_ch = compute_es(cs, rs, lambda g: float(g["x_specparam_resid"].numpy()[:, 3].mean()))
        unm_sens_rows.append({"condition": cond, "theta_d": st_th["cohens_d"], "theta_p": st_th["p_value"], "alpha_d": st_al["cohens_d"], "alpha_p": st_al["p_value"], "chi_d": st_ch["cohens_d"], "chi_p": st_ch["p_value"]})
    pd.DataFrame(unm_sens_rows).to_csv(REPORTS_DIR / "unm_ec_vs_eo_sensitivity.csv", index=False)

    iapf_rows = []
    for clbl, cs, rs, isp in contrasts_10:
        if isp:
            pairs = [(a.get("iapf_posterior", np.nan), b.get("iapf_posterior", np.nan)) for a, b in zip(cs, rs)]
            pairs = [(x, y) for x, y in pairs if not (np.isnan(x) or np.isnan(y))]
            vc, vr = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
            d_val = float((vc - vr).mean() / ((vc - vr).std(ddof=1) + 1e-12))
            _, p_v = stats.ttest_rel(vc, vr)
        else:
            vc = np.array([g.get("iapf_posterior", np.nan) for g in cs if not np.isnan(g.get("iapf_posterior", np.nan))])
            vr = np.array([g.get("iapf_posterior", np.nan) for g in rs if not np.isnan(g.get("iapf_posterior", np.nan))])
            sp_sd = np.sqrt(((len(vc) - 1) * np.var(vc, ddof=1) + (len(vr) - 1) * np.var(vr, ddof=1)) / (len(vc) + len(vr) - 2)) + 1e-12
            d_val = float((vc.mean() - vr.mean()) / sp_sd)
            _, p_v = stats.ttest_ind(vc, vr, equal_var=False)
        iapf_rows.append({
            "cohort": clbl, "n_valid_case": len(vc), "n_valid_ctrl": len(vr),
            "mean_iapf_case_hz": float(vc.mean()), "sd_iapf_case_hz": float(vc.std(ddof=1)),
            "mean_iapf_ctrl_hz": float(vr.mean()), "sd_iapf_ctrl_hz": float(vr.std(ddof=1)),
            "diff_hz": float(vc.mean() - vr.mean()), "cohens_d": d_val, "p_value": float(p_v),
        })
    pd.DataFrame(iapf_rows).to_csv(REPORTS_DIR / "iapf_biophysics_summary.csv", index=False)

    # Grand-average spectra JSON (Figure S4) & master_results.json (deduplicated 564 / 675 / 733)
    freqs_full = np.asarray(psds["sleep_ds004902"][0]["freqs"], dtype=float)
    fmask, fmask_in = (freqs_full >= 1.0) & (freqs_full <= 40.0), (freqs_full >= 2.0) & (freqs_full <= 30.0)
    f_plot, f_in = freqs_full[fmask], freqs_full[fmask_in]

    def extract_grp_spec(sp_list, psd_list, sel_fn):
        raw_c, res_c = [], []
        fg = SpectralGroupModel(**SPECPARAM_KWARGS)
        sel_idx = [i for i, r in enumerate(sp_list) if sel_fn(r)]
        sel_psd = np.stack([np.maximum(np.asarray(psd_list[i]["mean_psd"], dtype=float)[:, fmask_in], 1e-25).mean(axis=0) for i in sel_idx])
        fg.fit(f_in, sel_psd, [2.0, 30.0])
        for k_i, idx in enumerate(sel_idx):
            lp_full = np.log10(np.maximum(np.asarray(psd_list[idx]["mean_psd"], dtype=float)[:, fmask], 1e-25).mean(axis=0))
            m = fg.get_model(k_i, regenerate=True)
            raw_c.append(lp_full)
            res_c.append(m.data.power_spectrum - m.results.model._ap_fit)
        ar, ares = np.stack(raw_c), np.stack(res_c)
        return {
            "n": len(sel_idx),
            "raw_mean": ar.mean(axis=0).tolist(), "raw_sem": (ar.std(axis=0, ddof=1) / np.sqrt(len(sel_idx))).tolist(),
            "res_mean": ares.mean(axis=0).tolist(), "res_sem": (ares.std(axis=0, ddof=1) / np.sqrt(len(sel_idx))).tolist(),
        }

    spectra_specs = [
        ("Study 1: Sleep Deprivation", "sleep_ds004902", "SD (N=71)", lambda r: r["condition"] == "SD", "NS (N=71)", lambda r: r["condition"] == "NS"),
        ("Study 2: Outpatient MDD (EC)", "mumtaz_ec", "MDD (N=29)", lambda r: r["group"] == "MDD", "HC (N=27)", lambda r: r["group"] == "HC"),
        ("Study 3: Univ. Depression", "cavanagh_ds003478", "Dep (N=44)", lambda r: r["group"] == "Depressed", "HC (N=75)", lambda r: r["group"] == "Control"),
        ("Study 4: UCSD Parkinson", "ucsd_ds002778", "PD All (N=30)", lambda r: "PD" in r["group"], "HC (N=16)", lambda r: r["group"] == "HC"),
        ("Study 5: Clinical Dementia", "dementia_ds004504", "AD+FTD (N=59)", lambda r: r["diagnosis"] in ("Alzheimer", "FTD"), "HC (N=29)", lambda r: r["diagnosis"] == "Control"),
        ("Study 6a: UNM PD-OFF (EO)", "unm_ds003490", "PD-OFF (N=25)", lambda r: r["med_state"] == "OFF", "HC (N=25)", lambda r: r["med_state"] == "CTL"),
        ("Study 7: Iowa Parkinson", "iowa_ds004584", "PD (N=100)", lambda r: r["group"] == "PD", "HC (N=49)", lambda r: r["group"] == "Control"),
    ]
    scache = {"freqs": f_plot.tolist(), "freqs_res": f_in.tolist(), "cohorts": []}
    for title, ck, cl, cf, hl, hf in spectra_specs:
        scache["cohorts"].append({"title": title, "cohort_key": ck, "case_label": cl, "ctrl_label": hl, "case": extract_grp_spec(sp[ck], psds[ck], cf), "ctrl": extract_grp_spec(sp[ck], psds[ck], hf)})
    with open(REPORTS_DIR / "real_grand_average_spectra.json", "w", encoding="utf-8") as f:
        json.dump(scache, f)

    omst_trees = [g["omst_trees"] for g in all_primary]
    omst_edges = [g["omst_edge_count"] for g in all_primary]
    gces = [g["gce"] for g in all_primary]
    master_payload = {
        "spectral_backend": "specparam_v2.0.0rc7_interior_2to30Hz",
        "accounting": {
            "total_unique_participants": 564,
            "total_analyzed_sessions": 675,
            "total_with_mumtaz_eo": 733,
            "total_spectra": 733 * 19,
            "nominal_pre_dedup_participants": 566,
            "nominal_pre_dedup_primary_sessions": 677,
            "nominal_pre_dedup_total_sessions": 737,
            "enrolled_participants": 574,
            "excluded_participants": 10,
            "cohorts": {
                "Study 1 (Sleep ds004902)": {"enrolled": 71, "excluded": 0, "analyzed_pts": 71, "analyzed_sessions": 142, "design": "Within-subject paired (71 SD, 71 NS)"},
                "Study 2 (Mumtaz MDD 4244171)": {"enrolled": 64, "excluded": 8, "qc_excluded": 6, "dedup_excluded": 2, "analyzed_pts": 56, "analyzed_sessions": 56, "design": "Deduplicated EC (29 MDD, 27 HC); 58 EO (31 MDD, 27 HC)"},
                "Study 3 (Cavanagh ds003478)": {"enrolled": 121, "excluded": 2, "analyzed_pts": 119, "analyzed_sessions": 119, "design": "Between-subject (44 Depressed, 75 HC)"},
                "Study 4 (UCSD PD ds002778)": {"enrolled": 31, "excluded": 0, "analyzed_pts": 31, "analyzed_sessions": 46, "design": "15 PD (paired ON/OFF) + 16 HC"},
                "Study 5 (AHEPA Dementia ds004504)": {"enrolled": 88, "excluded": 0, "analyzed_pts": 88, "analyzed_sessions": 88, "design": "Between-subject (36 AD, 23 FTD, 29 HC)"},
                "Study 6 (UNM PD ds003490)": {"enrolled": 50, "excluded": 0, "analyzed_pts": 50, "analyzed_sessions": 75, "design": "25 PD (paired ON/OFF EO) + 25 HC (EO)"},
                "Study 7 (Iowa PD ds004584)": {"enrolled": 149, "excluded": 0, "analyzed_pts": 149, "analyzed_sessions": 149, "design": "Between-subject EO (100 PD ON, 49 HC)"},
            },
        },
        "omst_topology": {
            "total_graphs": 675,
            "total_node_instances": 675 * 19,
            "mean_trees": float(np.mean(omst_trees)), "sd_trees": float(np.std(omst_trees, ddof=1)),
            "pct_tree_1": float(100.0 * np.mean(np.array(omst_trees) == 1)),
            "pct_tree_2": float(100.0 * np.mean(np.array(omst_trees) >= 2)),
            "mean_edges": float(np.mean(omst_edges)), "sd_edges": float(np.std(omst_edges, ddof=1)),
            "mean_gce": float(np.mean(gces)), "sd_gce": float(np.std(gces, ddof=1)),
            "isolated_node_rate_pct": 0.0,
        },
    }
    with open(REPORTS_DIR / "master_results.json", "w", encoding="utf-8") as f:
        json.dump(master_payload, f, indent=2)

    # Gain invariance audit on deduplicated [2, 30] Hz
    sample_recs = psds["ucsd_ds002778"][:5] + psds["mumtaz_ec"][:5] + psds["iowa_ds004584"][:5]
    freqs_g = np.asarray(sample_recs[0]["freqs"], dtype=float)
    psds_g = np.vstack([np.maximum(np.asarray(r["mean_psd"], dtype=float), 1e-25) for r in sample_recs])
    fg_b = SpectralGroupModel(**SPECPARAM_KWARGS)
    fg_b.fit(freqs_g, psds_g, [2.0, 30.0], n_jobs=1)
    ap_b = fg_b.get_params("aperiodic")
    th_m_g = (fg_b.data.freqs >= 4.0) & (fg_b.data.freqs < 8.0)
    th_b = np.array([float(np.mean((fg_b.get_model(i, regenerate=True).data.power_spectrum - fg_b.get_model(i, regenerate=True).results.model._ap_fit)[th_m_g])) for i in range(len(psds_g))])
    gain_chk_rows = []
    for k_val in [0.01, 0.50, 1.00, 2.00, 5.00, 100.00]:
        fg_k = SpectralGroupModel(**SPECPARAM_KWARGS)
        fg_k.fit(freqs_g, (k_val ** 2) * psds_g, [2.0, 30.0], n_jobs=1)
        ap_k = fg_k.get_params("aperiodic")
        th_k = np.array([float(np.mean((fg_k.get_model(i, regenerate=True).data.power_spectrum - fg_k.get_model(i, regenerate=True).results.model._ap_fit)[th_m_g])) for i in range(len(psds_g))])
        gain_chk_rows.append({
            "gain_k": k_val, "n_spectra": len(psds_g), "expected_offset_shift": float(2.0 * np.log10(k_val)),
            "measured_offset_shift_mean": float(np.mean(ap_k[:, 0] - ap_b[:, 0])),
            "max_abs_diff_chi": float(np.max(np.abs(ap_k[:, 1] - ap_b[:, 1]))),
            "max_abs_diff_theta_resid": float(np.max(np.abs(th_k - th_b))),
        })
    pd.DataFrame(gain_chk_rows).to_csv(REPORTS_DIR / "specparam_gain_invariance_check.csv", index=False)

    # Update Table S1 OMST empirical channel metrics (N=675) & Heuristic slowing CSV
    CH_COORDS = {
        "Fp1": ("Prefrontal (Left)", -0.28, +0.78), "Fp2": ("Prefrontal (Right)", +0.28, +0.78),
        "F7": ("Inferior Frontal (Left)", -0.72, +0.44), "F3": ("Dorsal Frontal (Left)", -0.35, +0.40),
        "Fz": ("Frontal Midline", 0.00, +0.40), "F4": ("Dorsal Frontal (Right)", +0.35, +0.40),
        "F8": ("Inferior Frontal (Right)", +0.72, +0.44), "T7": ("Mid-Temporal (Left)", -0.84, 0.00),
        "C3": ("Central (Left)", -0.42, 0.00), "Cz": ("Central Vertex", 0.00, 0.00),
        "C4": ("Central (Right)", +0.42, 0.00), "T8": ("Mid-Temporal (Right)", +0.84, 0.00),
        "P7": ("Post-Temporal (Left)", -0.72, -0.44), "P3": ("Parietal (Left)", -0.35, -0.40),
        "Pz": ("Parietal Midline", 0.00, -0.40), "P4": ("Parietal (Right)", +0.35, -0.40),
        "P8": ("Post-Temporal (Right)", +0.72, -0.44), "O1": ("Occipital (Left)", -0.28, -0.78),
        "O2": ("Occipital (Right)", +0.28, -0.78),
    }
    T1_EDGES = [("Fp1", "F3"), ("Fp2", "F4"), ("F7", "F3"), ("F8", "F4"), ("F7", "T7"), ("F8", "T8"), ("F3", "Fz"),
                ("F4", "Fz"), ("Fz", "Cz"), ("C3", "Cz"), ("C4", "Cz"), ("Cz", "Pz"), ("Pz", "P3"), ("Pz", "P4"),
                ("T7", "P7"), ("T8", "P8"), ("P3", "O1"), ("P4", "O2")]
    T2_EDGES = [("Fp1", "Fp2"), ("Fp1", "F7"), ("Fp2", "F8"), ("F3", "C3"), ("F4", "C4"), ("T7", "C3"), ("T8", "C4"),
                ("C3", "P3"), ("C4", "P4"), ("P7", "P3"), ("P8", "P4"), ("P7", "O1"), ("P8", "O2"), ("O1", "O2"),
                ("Fz", "Pz"), ("F3", "F4"), ("C3", "C4"), ("Cz", "O1")]
    degs, betw, clus = np.zeros((len(all_primary), 19)), np.zeros((len(all_primary), 19)), np.zeros((len(all_primary), 19))
    for i, g in enumerate(all_primary):
        A = (g["adj"].numpy() > 0.5).astype(int)
        np.fill_diagonal(A, 0)
        G = nx.from_numpy_array(A)
        dmap, bmap, cmap = dict(G.degree()), nx.betweenness_centrality(G), nx.clustering(G)
        for j in range(19):
            degs[i, j], betw[i, j], clus[i, j] = dmap[j], bmap[j], cmap[j]
    t1_deg, t2_deg = {c: 0 for c in CHANNELS_19}, {c: 0 for c in CHANNELS_19}
    for u, v in T1_EDGES: t1_deg[u] += 1; t1_deg[v] += 1
    for u, v in T2_EDGES: t2_deg[u] += 1; t2_deg[v] += 1
    s1_rows = []
    for j, ch in enumerate(CHANNELS_19):
        reg, px, py = CH_COORDS[ch]
        s1_rows.append({
            "channel": ch, "region": reg, "proj_x": px, "proj_y": py,
            "tree1_deg": t1_deg[ch], "tree2_deg": t2_deg[ch], "combined_deg": t1_deg[ch] + t2_deg[ch],
            "emp_deg_mean": float(degs[:, j].mean()), "emp_deg_sd": float(degs[:, j].std(ddof=1)),
            "betw_mean": float(betw[:, j].mean()), "betw_sd": float(betw[:, j].std(ddof=1)),
            "clus_mean": float(clus[:, j].mean()), "clus_sd": float(clus[:, j].std(ddof=1)),
        })
    pd.DataFrame(s1_rows).to_csv(REPORTS_DIR / "table_s1_omst_channel_metrics.csv", index=False)

    specs_heur = [
        ("Study 1: Sleep Deprivation (ds004902)", sleep), ("Study 2: Mumtaz MDD (Eyes Closed)", mum_ec),
        ("Study 2b: Mumtaz MDD (Eyes Open)", mum_eo), ("Study 3: Cavanagh University Depression", cav),
        ("Study 4: UCSD Parkinson All Sessions", ucsd),
        ("Study 5a: Dementia AD vs Control", [g for g in dem if g["diagnosis"] in ("Alzheimer", "Control")]),
        ("Study 5b: Dementia FTD vs Control", [g for g in dem if g["diagnosis"] in ("FTD", "Control")]),
        ("Study 5c: Dementia All (AD+FTD) vs Control", dem),
        ("Study 6a: UNM PD-OFF vs Control", unm_off + unm_hc), ("Study 6b: UNM PD-ON vs Control", unm_on + unm_hc),
        ("Study 6c: UNM PD All Sessions", unm), ("Study 7: Iowa Parkinson (ds004584)", iowa),
    ]
    hrows = []
    for cname, glist in specs_heur:
        vg = [g for g in glist if int(g["y"]) in (0, 1)]
        y_all = np.array([int(g["y"]) for g in vg])
        grps = np.array([g.get("bids_id", g.get("sub_name", g["subject_id"])) for g in vg])
        X_all = torch.stack([g["x"] for g in vg])
        aucs_raw, aucs_ori, baccs, aucs_th = [], [], [], []
        for s in REPEATS_10:
            splits = get_grouped_splits(grps, y_all, n_splits=5, seed=s)
            oof_h, oof_t, oof_y = np.zeros(len(vg)), np.zeros(len(vg)), np.zeros(len(vg))
            for fold, (tr, va) in enumerate(splits):
                ty = torch.tensor(y_all[tr], dtype=torch.long)
                tz, vz, _, _ = compute_fold_normative_z(X_all[tr], ty, X_all[va])
                h_tr = (tz[:, FOCAL_5, 0].mean(dim=-1) - tz[:, FOCAL_5, 2].mean(dim=-1)).numpy()
                h_va = (vz[:, FOCAL_5, 0].mean(dim=-1) - vz[:, FOCAL_5, 2].mean(dim=-1)).numpy()
                oof_h[va] = h_va
                oof_y[va] = LogisticRegression(random_state=s).fit(h_tr.reshape(-1, 1), y_all[tr]).predict(h_va.reshape(-1, 1))
                t_tr, t_va = tz[:, FOCAL_5, 0].mean(dim=-1, keepdim=True).numpy(), vz[:, FOCAL_5, 0].mean(dim=-1, keepdim=True).numpy()
                oof_t[va] = LogisticRegression(random_state=s).fit(t_tr, y_all[tr]).predict_proba(t_va)[:, 1]
            ar, at = float(roc_auc_score(y_all, oof_h)), float(roc_auc_score(y_all, oof_t))
            aucs_raw.append(ar); aucs_ori.append(max(ar, 1.0 - ar))
            baccs.append(float(balanced_accuracy_score(y_all, oof_y))); aucs_th.append(max(at, 1.0 - at))
        hrows.append({
            "cohort": cname, "n_samples": len(vg), "heuristic_rule": "Frontal_Theta_Alpha_Difference",
            "auc_raw": float(np.mean(aucs_raw)), "auc_oriented": float(np.mean(aucs_ori)),
            "balanced_accuracy": float(np.mean(baccs)), "single_theta_auc_oriented": float(np.mean(aucs_th)),
        })
    pd.DataFrame(hrows).to_csv(REPORTS_DIR / "benchmark_heuristic_slowing.csv", index=False)
    print("✓ Step 3 complete: Module A, K=6 & K=5 Meta-Analysis, ANCOVA, Table S1, Heuristic CSVs updated.")
    return master_payload


# ==============================================================================
# PART 4: 2D FIT-WINDOW GRID & FOOOF v1.1.1 vs SPECPARAM v2.0.0rc7 EQUIVALENCE
# ==============================================================================
def run_2d_grid_and_fooof_check(psds, raw_unfilt_psds):
    print("\n=== [Step 4] 2D Fit-Window Grid ([1..3] x [25..45] Hz) & fooof v1.1.1 Equivalence ===")
    grid_rows = []
    # 1) Passband-filtered 1-40 Hz cached Welch PSDs across Study 2 (56), Study 3 (119), Study 5c (88), Study 7 (149)
    cohorts_cfg = [
        ("Study 2: Outpatient MDD (N=56, Passband 1-40Hz)", "Passband_Filtered_1to40Hz", psds["mumtaz_ec"], [25.0, 30.0, 35.0, 40.0]),
        ("Study 3: Univ. Depression (N=119, Passband 1-40Hz)", "Passband_Filtered_1to40Hz", psds["cavanagh_ds003478"], [25.0, 30.0, 35.0, 40.0]),
        ("Study 5c: Dementia All (N=88, Passband 1-40Hz)", "Passband_Filtered_1to40Hz", psds["dementia_ds004504"], [25.0, 30.0, 35.0, 40.0]),
        ("Study 7: Iowa PD (N=149, Passband 1-40Hz)", "Passband_Filtered_1to40Hz", psds["iowa_ds004584"], [25.0, 30.0, 35.0, 40.0]),
        ("Study 2: Outpatient MDD (N=56, Raw Unfiltered EDF)", "Unfiltered_Raw_EDF", raw_unfilt_psds, [25.0, 30.0, 35.0, 40.0, 45.0]),
    ]
    f_mins = [1.0, 1.5, 2.0, 2.5, 3.0]

    for cname, psd_src, recs, f_maxs in cohorts_cfg:
        freqs = np.asarray(recs[0]["freqs"], dtype=float)
        ys = np.array([int(r["y"]) for r in recs])
        flat_psds = np.vstack([np.maximum(np.asarray(r["mean_psd"], dtype=float), 1e-25) for r in recs])
        n_s = len(recs)
        for fl in f_mins:
            for fh in f_maxs:
                sgm = SpectralGroupModel(**SPECPARAM_KWARGS)
                sgm.fit(freqs, flat_psds, [fl, fh], n_jobs=1)
                f_fit = sgm.data.freqs
                th_m = (f_fit >= 4.0) & (f_fit < 8.0)
                aps = sgm.get_params("aperiodic")
                r2s = sgm.get_metrics("gof", "rsquared")
                maes = sgm.get_metrics("error", "mae")
                null_set = set(sgm.results.null_inds or [])

                th_vals, ch_vals = np.full(n_s, np.nan), np.full(n_s, np.nan)
                r2_valid, mae_valid = [], []
                for si in range(n_s):
                    f_th, g_ch = [], []
                    for ci in range(19):
                        gi = si * 19 + ci
                        if gi in null_set or np.isnan(aps[gi, 1]):
                            continue
                        m_res = sgm.get_model(gi, regenerate=True)
                        resid = m_res.data.power_spectrum - m_res.results.model._ap_fit
                        if ci in FOCAL_5:
                            f_th.append(float(np.mean(resid[th_m])))
                        g_ch.append(float(aps[gi, 1]))
                        r2_valid.append(float(r2s[gi]))
                        mae_valid.append(float(maes[gi]))
                    if f_th:
                        th_vals[si] = float(np.mean(f_th))
                    if g_ch:
                        ch_vals[si] = float(np.mean(g_ch))

                def es_arr(vals):
                    vc = vals[(ys == 1) & ~np.isnan(vals)]
                    vr = vals[(ys == 0) & ~np.isnan(vals)]
                    sp_sd = np.sqrt(((len(vc) - 1) * np.var(vc, ddof=1) + (len(vr) - 1) * np.var(vr, ddof=1)) / (len(vc) + len(vr) - 2)) + 1e-12
                    d = float((vc.mean() - vr.mean()) / sp_sd)
                    _, p = stats.ttest_ind(vc, vr, equal_var=False)
                    return d, float(p)

                th_d, th_p = es_arr(th_vals)
                ch_d, ch_p = es_arr(ch_vals)
                grid_rows.append({
                    "cohort": cname, "psd_source": psd_src, "f_min_hz": fl, "f_max_hz": fh,
                    "n_sessions": n_s, "n_spectra": n_s * 19,
                    "n_failed": len(null_set), "pct_failed": float(100.0 * len(null_set) / (n_s * 19)),
                    "r2_median": float(np.median(r2_valid)), "mae_median": float(np.median(mae_valid)),
                    "theta_d": th_d, "theta_p": th_p, "chi_d": ch_d, "chi_p": ch_p,
                })
    df_grid = pd.DataFrame(grid_rows)
    df_grid.to_csv(REPORTS_DIR / "specparam_2d_window_grid_sensitivity.csv", index=False)
    print(f"✓ 2D Grid Complete ({len(df_grid)} configurations, max n_failed={df_grid['n_failed'].max()}):")
    print(df_grid[df_grid["cohort"].str.contains("Study 2")][["psd_source", "f_min_hz", "f_max_hz", "r2_median", "theta_d", "chi_d"]].round(3).to_string(index=False))

    # 2) Empirical fooof 1.1.1 vs specparam 2.0.0rc7 comparison on 5 sessions x 7 cohorts = 665 real spectra
    sample_specs = [
        ("sleep_ds004902", psds["sleep_ds004902"][:5]),
        ("mumtaz_ec", psds["mumtaz_ec"][:5]),
        ("cavanagh_ds003478", psds["cavanagh_ds003478"][:5]),
        ("ucsd_ds002778", psds["ucsd_ds002778"][:5]),
        ("dementia_ds004504", psds["dementia_ds004504"][:5]),
        ("unm_ds003490", psds["unm_ds003490"][:5]),
        ("iowa_ds004584", psds["iowa_ds004584"][:5]),
    ]
    freqs_eq = np.asarray(psds["sleep_ds004902"][0]["freqs"], dtype=float)
    stacked, tags = [], []
    for ck, s_recs in sample_specs:
        for r in s_recs:
            stacked.append(np.maximum(np.asarray(r["mean_psd"], dtype=float), 1e-25))
            tags.extend([ck] * 19)
    arr_psds = np.vstack(stacked)
    npz_in = SCRATCH_DIR / "fooof_vs_sp_input.npz"
    npz_out = SCRATCH_DIR / "fooof_vs_sp_output.npz"
    np.savez(npz_in, freqs=freqs_eq, psds=arr_psds)

    fooof_runner = """
import numpy as np, warnings
warnings.filterwarnings("ignore")
from fooof import FOOOFGroup
dat = np.load(r"%s")
freqs, psds = dat["freqs"], dat["psds"]
out = {}
for tag, fl, fh in [("2to30", 2.0, 30.0), ("1to40", 1.0, 40.0)]:
    fg = FOOOFGroup(aperiodic_mode="fixed", peak_width_limits=[1.0, 12.0], max_n_peaks=6, min_peak_height=0.05, peak_threshold=2.0, verbose=False)
    fg.fit(freqs, psds, [fl, fh], n_jobs=1)
    ff = fg.freqs
    th_m = (ff >= 4.0) & (ff < 8.0)
    aps = fg.get_params("aperiodic_params")
    r2s = fg.get_results()
    chi = np.array([aps[i, 1] for i in range(len(psds))])
    r2 = np.array([r2s[i].r_squared for i in range(len(psds))])
    th = np.zeros(len(psds))
    for i in range(len(psds)):
        fm = fg.get_fooof(i, regenerate=True)
        th[i] = float(np.mean((fm.power_spectrum - fm._ap_fit)[th_m]))
    out[f"{tag}_chi"] = chi
    out[f"{tag}_theta"] = th
    out[f"{tag}_r2"] = r2
np.savez(r"%s", **out)
""" % (npz_in, npz_out)
    subprocess.run([str(FOOOF_PY), "-c", fooof_runner], check=True)
    f_res = np.load(npz_out)

    eq_rows = []
    tags_arr = np.asarray(tags)
    for tag, fl, fh in [("2to30", 2.0, 30.0), ("1to40", 1.0, 40.0)]:
        sgm = SpectralGroupModel(**SPECPARAM_KWARGS)
        sgm.fit(freqs_eq, arr_psds, [fl, fh], n_jobs=1)
        th_m = (sgm.data.freqs >= 4.0) & (sgm.data.freqs < 8.0)
        sp_chi = sgm.get_params("aperiodic")[:, 1]
        sp_r2 = sgm.get_metrics("gof", "rsquared")
        sp_th = np.array([float(np.mean((sgm.get_model(i, regenerate=True).data.power_spectrum - sgm.get_model(i, regenerate=True).results.model._ap_fit)[th_m])) for i in range(len(arr_psds))])
        fo_chi, fo_th, fo_r2 = f_res[f"{tag}_chi"], f_res[f"{tag}_theta"], f_res[f"{tag}_r2"]

        for lbl, mask in [("All_7Cohorts_Pooled", np.ones(len(arr_psds), dtype=bool))] + [(ck, tags_arr == ck) for ck, _ in sample_specs]:
            eq_rows.append({
                "fit_window_hz": f"{fl:.1f}-{fh:.1f}", "cohort": lbl, "n_spectra": int(np.sum(mask)),
                "pearson_r_chi": float(pearsonr(fo_chi[mask], sp_chi[mask])[0]),
                "mean_abs_diff_chi": float(np.mean(np.abs(fo_chi[mask] - sp_chi[mask]))),
                "max_abs_diff_chi": float(np.max(np.abs(fo_chi[mask] - sp_chi[mask]))),
                "pearson_r_theta_resid": float(pearsonr(fo_th[mask], sp_th[mask])[0]),
                "mean_abs_diff_theta_resid": float(np.mean(np.abs(fo_th[mask] - sp_th[mask]))),
                "max_abs_diff_theta_resid": float(np.max(np.abs(fo_th[mask] - sp_th[mask]))),
                "max_abs_diff_r2": float(np.max(np.abs(fo_r2[mask] - sp_r2[mask]))),
            })
    df_eq = pd.DataFrame(eq_rows)
    df_eq.to_csv(REPORTS_DIR / "fooof_v1_vs_specparam_v2_equivalence.csv", index=False)
    print(df_eq[df_eq["cohort"] == "All_7Cohorts_Pooled"].to_string(index=False))


# ==============================================================================
# PART 5: CALIBRATED MONTE CARLO SIMULATION (MODULE B)
# ==============================================================================
def run_module_b_calibrated_simulation():
    print("\n=== [Step 5] Empirical-Variance-Calibrated Monte Carlo Simulation ===")
    freqs = np.arange(1.0, 40.5, 0.5)
    f_log = np.log10(freqs)
    th_m_40 = (freqs >= 4.0) & (freqs < 8.0)
    mask_in = (freqs >= 2.0) & (freqs <= 30.0)
    f_in = freqs[mask_in]
    th_m_in = (f_in >= 4.0) & (f_in < 8.0)

    def gen_psd(rng, b, chi, th_amp, th_cf, th_bw, al_amp, al_cf, al_bw, rolloff_low=0.0, rolloff_high=0.0, noise_sd=0.035):
        ap = b - chi * f_log
        pk_th = th_amp * np.exp(-0.5 * ((freqs - th_cf) / th_bw) ** 2)
        pk_al = al_amp * np.exp(-0.5 * ((freqs - al_cf) / al_bw) ** 2)
        att_low = -rolloff_low * np.maximum(0.0, 2.0 - freqs) ** 2
        att_high = -rolloff_high * np.maximum(0.0, (freqs - 30.0) / 10.0) ** 2
        return 10.0 ** (ap + pk_th + pk_al + att_low + att_high + rng.normal(0.0, noise_sd, size=len(freqs)))

    rng = np.random.RandomState(2026)
    n_trials = 150
    n_case, n_ctrl = 29, 27  # Exact deduplicated Study 2 sample size

    scenarios = [
        ("S1_TrueNullTheta_TrueExponentDrop", -0.15, 0.00, +0.05, 0.0, 0.0),
        ("S2_TrueNullTheta_BroadAlphaAndSlowCurve", -0.12, 0.00, -0.22, 0.0, 0.0),
        ("S3_FilterEdgeAttenuation_Study2Replica", -0.16, -0.14, -0.12, 0.80, 1.10),
    ]
    sim_rows = []
    for s_name, d_chi, d_th, d_al, rl_c, rh_c in scenarios:
        psds_batch, y_batch, true_chi_batch, true_th_batch = [], [], [], []
        for _ in range(n_trials):
            for grp, n_sub in [(0, n_ctrl), (1, n_case)]:
                for _ in range(n_sub):
                    chi_i = rng.normal(1.10 + (d_chi if grp == 1 else 0.0), 0.16)
                    th_i = max(0.01, rng.normal(0.22 + (d_th if grp == 1 else 0.0), 0.12))
                    al_i = max(0.03, rng.normal(0.60 + (d_al if grp == 1 else 0.0), 0.20))
                    b_i = rng.normal(-11.0, 0.30)
                    rl = rl_c if grp == 1 else 0.05
                    rh = rh_c if grp == 1 else 0.10
                    psds_batch.append(gen_psd(rng, b_i, chi_i, th_i, 6.0, 1.2, al_i, 10.0, 1.6, rolloff_low=rl, rolloff_high=rh))
                    y_batch.append(grp)
                    true_chi_batch.append(chi_i)
                    true_th_batch.append(float(np.mean(th_i * np.exp(-0.5 * ((f_in[th_m_in] - 6.0) / 1.2) ** 2))))

        psds_arr = np.asarray(psds_batch)
        fg40 = SpectralGroupModel(**SPECPARAM_KWARGS)
        fg40.fit(freqs, psds_arr, [1.0, 40.0], n_jobs=1)
        fg30 = SpectralGroupModel(**SPECPARAM_KWARGS)
        fg30.fit(freqs, psds_arr, [2.0, 30.0], n_jobs=1)

        aps40, r2_40_all = fg40.get_params("aperiodic"), fg40.get_metrics("gof", "rsquared")
        aps30, r2_30_all = fg30.get_params("aperiodic"), fg30.get_metrics("gof", "rsquared")
        ols_mask = (freqs < 7.0) | (freqs > 14.0)
        lf_ols = f_log[ols_mask]

        d_th_ols40, d_ch_ols40 = [], []
        d_th_sp40, d_ch_sp40, r2_sp40 = [], [], []
        d_th_sp30, d_ch_sp30, r2_sp30 = [], [], []
        chi_bias_ols40, chi_bias_sp40, chi_bias_sp30 = [], [], []
        true_d_th_trials, true_d_chi_trials = [], []

        N_per = n_case + n_ctrl
        for t in range(n_trials):
            i0, i1 = t * N_per, (t + 1) * N_per
            yt = np.asarray(y_batch[i0:i1])

            def cd_arr(vals):
                v = np.asarray(vals)
                vc, vr = v[yt == 1], v[yt == 0]
                return float((vc.mean() - vr.mean()) / (np.sqrt(((len(vc) - 1) * vc.var(ddof=1) + (len(vr) - 1) * vr.var(ddof=1)) / (len(vc) + len(vr) - 2)) + 1e-12))

            th_o40, ch_o40, th_s40, ch_s40, th_s30, ch_s30 = [], [], [], [], [], []
            for k in range(i0, i1):
                lp40 = np.log10(psds_arr[k])
                sl, ic = np.polyfit(lf_ols, lp40[ols_mask], 1)
                th_o40.append(float(np.mean((lp40 - (ic + sl * f_log))[th_m_40])))
                ch_o40.append(float(-sl))

                m40 = fg40.get_model(k, regenerate=True)
                th_s40.append(float(np.mean((m40.data.power_spectrum - m40.results.model._ap_fit)[th_m_40])))
                ch_s40.append(float(aps40[k, 1]))

                m30 = fg30.get_model(k, regenerate=True)
                th_s30.append(float(np.mean((m30.data.power_spectrum - m30.results.model._ap_fit)[th_m_in])))
                ch_s30.append(float(aps30[k, 1]))

            true_d_th_trials.append(cd_arr(true_th_batch[i0:i1]))
            true_d_chi_trials.append(cd_arr(true_chi_batch[i0:i1]))
            d_th_ols40.append(cd_arr(th_o40)); d_ch_ols40.append(cd_arr(ch_o40))
            d_th_sp40.append(cd_arr(th_s40)); d_ch_sp40.append(cd_arr(ch_s40))
            d_th_sp30.append(cd_arr(th_s30)); d_ch_sp30.append(cd_arr(ch_s30))
            r2_sp40.append(float(np.median(r2_40_all[i0:i1])))
            r2_sp30.append(float(np.median(r2_30_all[i0:i1])))
            c_true = np.asarray(true_chi_batch[i0:i1])
            chi_bias_ols40.append(float(np.mean(np.asarray(ch_o40) - c_true)))
            chi_bias_sp40.append(float(np.mean(np.asarray(ch_s40) - c_true)))
            chi_bias_sp30.append(float(np.mean(np.asarray(ch_s30) - c_true)))

        true_d_th = float(np.mean(true_d_th_trials))
        true_d_chi = float(np.mean(true_d_chi_trials))
        for est_lbl, dth_arr, dch_arr, cb_arr, r2_arr in [
            ("Masked_OLS_1to40", d_th_ols40, d_ch_ols40, chi_bias_ols40, [np.nan]),
            ("specparam_1to40", d_th_sp40, d_ch_sp40, chi_bias_sp40, r2_sp40),
            ("specparam_2to30", d_th_sp30, d_ch_sp30, chi_bias_sp30, r2_sp30),
        ]:
            sim_rows.append({
                "scenario": s_name, "estimator": est_lbl, "n_trials": n_trials,
                "true_theta_d": true_d_th, "est_theta_d_mean": float(np.mean(dth_arr)), "est_theta_d_sd": float(np.std(dth_arr, ddof=1)),
                "theta_d_bias": float(np.mean(dth_arr) - true_d_th),
                "true_chi_d": true_d_chi, "est_chi_d_mean": float(np.mean(dch_arr)), "est_chi_d_sd": float(np.std(dch_arr, ddof=1)),
                "chi_d_bias": float(np.mean(dch_arr) - true_d_chi),
                "chi_raw_bias": float(np.nanmean(cb_arr)),
                "median_r2": float(np.nanmean(r2_arr)),
            })
    df_sim = pd.DataFrame(sim_rows)
    df_sim.to_csv(REPORTS_DIR / "synthetic_spectral_leakage_simulation.csv", index=False)
    print(df_sim.to_string(index=False))


# ==============================================================================
# PART 6: MODULE C (10x5 CV, 13 MODELS w/ GCN_Weighted_PLI, FIXED SVMs, TOST)
# ==============================================================================
def run_module_c_v5(sp):
    print("\n=== [Step 6] Module C: 10x5 Repeated Grouped CV (13 Models, Fixed SVMs, GCN_Weighted_PLI, CIs & TOST) ===")
    sleep, mum_ec, mum_eo = sp["sleep_ds004902"], sp["mumtaz_ec"], sp["mumtaz_eo"]
    cav, ucsd, dem, unm, iowa = sp["cavanagh_ds003478"], sp["ucsd_ds002778"], sp["dementia_ds004504"], sp["unm_ds003490"], sp["iowa_ds004584"]

    specs_12 = [
        ("Study 1: Sleep Deprivation (ds004902)", sleep, True),
        ("Study 2: Mumtaz MDD (Eyes Closed)", mum_ec, False),
        ("Study 2b: Mumtaz MDD (Eyes Open)", mum_eo, False),
        ("Study 3: Cavanagh University Depression", cav, False),
        ("Study 4: UCSD Parkinson All Sessions", ucsd, True),
        ("Study 5a: Dementia AD vs Control", [g for g in dem if g["diagnosis"] in ("Alzheimer", "Control")], False),
        ("Study 5b: Dementia FTD vs Control", [g for g in dem if g["diagnosis"] in ("FTD", "Control")], False),
        ("Study 5c: Dementia All (AD+FTD) vs Control", dem, False),
        ("Study 6a: UNM PD-OFF vs Control", [g for g in unm if g["med_state"] in ("OFF", "CTL")], False),
        ("Study 6b: UNM PD-ON vs Control", [g for g in unm if g["med_state"] in ("ON", "CTL")], False),
        ("Study 6c: UNM PD All Sessions", unm, True),
        ("Study 7: Iowa Parkinson (ds004584)", iowa, False),
    ]
    models_13 = [
        "GCN_OMST", "GCN_Weighted_PLI", "GCN_Identity", "GCN_Random_EdgeMatched", "GCN_Full",
        "Tuned_LogisticRegression", "Tuned_Ridge", "Linear_SVM", "RBF_SVM",
        "RandomForest", "GBDT", "Single_Theta", "Single_Chi",
    ]
    adj_id = torch.zeros((19, 19), dtype=torch.float32)
    adj_full = make_dense_adj(19)

    repeat_rows, lofo_rows, split0_cls_rows = [], [], []
    fold_index_rows, oof_split0_rows, eo_split0_rows = [], [], []

    for cname, glist, is_grp in specs_12:
        t_c = time.time()
        vg = [g for g in glist if int(g["y"]) in (0, 1)]
        n = len(vg)
        y_all = np.array([int(g["y"]) for g in vg])
        groups = np.array([g.get("bids_id", g.get("sub_name", g.get("subject_id", str(i)))) for i, g in enumerate(vg)])
        X_all = torch.stack([g["x"] for g in vg])
        adj_omst = torch.stack([g["adj"] for g in vg])
        adj_wpli = torch.stack([g["adj_weighted"] for g in vg])
        edge_counts = [max(1, int(g.get("omst_edge_count", 27))) for g in vg]

        for r_idx, seed in enumerate(REPEATS_10):
            set_seed(seed)
            splits = get_grouped_splits(groups, y_all, n_splits=5, seed=seed)
            if r_idx == 0:
                fold_map = np.zeros(n, dtype=int)
                for f_k, (_, va_k) in enumerate(splits):
                    fold_map[va_k] = f_k
                for i_s in range(n):
                    fold_index_rows.append({"cohort": cname, "repeat_seed": seed, "row_idx": i_s, "subject_id": vg[i_s]["subject_id"], "group_id": groups[i_s], "y": int(y_all[i_s]), "fold": int(fold_map[i_s])})

            adj_rnd_all = torch.stack([make_edge_matched_random_adj(19, edge_counts[i], seed=seed * 1000 + i) for i in range(n)])
            oof_p = {m: np.zeros(n) for m in models_13}
            oof_y = {m: np.zeros(n) for m in models_13}
            oof_lofo = {k: np.zeros(n) for k in ["Single_Delta", "Single_Alpha", "Single_GCE", "GCN_Ablate_Theta", "GCN_Ablate_Delta", "GCN_Ablate_Alpha", "GCN_Ablate_Chi"]} if r_idx == 0 else None

            for fold, (tr, va) in enumerate(splits):
                tx, vx = X_all[tr], X_all[va]
                ty = torch.tensor(y_all[tr], dtype=torch.long)
                tz, vz, _, _ = compute_fold_normative_z(tx, ty, vx)
                X_tr, X_va = tz.view(len(tr), -1).numpy(), vz.view(len(va), -1).numpy()
                y_tr = y_all[tr]

                oof_p["GCN_OMST"][va], oof_y["GCN_OMST"][va] = train_eval_gcn(tz, adj_omst[tr], ty, vz, adj_omst[va], seed=seed + fold)
                oof_p["GCN_Weighted_PLI"][va], oof_y["GCN_Weighted_PLI"][va] = train_eval_gcn(tz, adj_wpli[tr], ty, vz, adj_wpli[va], seed=seed + fold)
                oof_p["GCN_Identity"][va], oof_y["GCN_Identity"][va] = train_eval_gcn(
                    tz, adj_id.unsqueeze(0).expand(len(tr), -1, -1), ty,
                    vz, adj_id.unsqueeze(0).expand(len(va), -1, -1), seed=seed + fold,
                )
                oof_p["GCN_Random_EdgeMatched"][va], oof_y["GCN_Random_EdgeMatched"][va] = train_eval_gcn(
                    tz, adj_rnd_all[tr], ty, vz, adj_rnd_all[va], seed=seed + fold,
                )
                oof_p["GCN_Full"][va], oof_y["GCN_Full"][va] = train_eval_gcn(
                    tz, adj_full.unsqueeze(0).expand(len(tr), -1, -1), ty,
                    vz, adj_full.unsqueeze(0).expand(len(va), -1, -1), seed=seed + fold,
                )

                icv = get_grouped_splits(groups[tr], y_tr, n_splits=3, seed=seed + fold)
                gs_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=seed), {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}, cv=icv, scoring="roc_auc").fit(X_tr, y_tr)
                oof_p["Tuned_LogisticRegression"][va], oof_y["Tuned_LogisticRegression"][va] = gs_lr.predict_proba(X_va)[:, 1], gs_lr.predict(X_va)

                gs_rc = GridSearchCV(RidgeClassifier(random_state=seed), {"alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}, cv=icv, scoring="roc_auc").fit(X_tr, y_tr)
                oof_p["Tuned_Ridge"][va], oof_y["Tuned_Ridge"][va] = 1.0 / (1.0 + np.exp(-gs_rc.decision_function(X_va))), gs_rc.predict(X_va)

                # Fixed SVM scoring: use decision_function + logistic link instead of Platt scaling (fixes small-sample < 0.50 flip)
                gs_lsvm = GridSearchCV(SVC(kernel="linear", probability=False, max_iter=10000, random_state=seed), {"C": [0.01, 0.1, 1.0, 10.0]}, cv=icv, scoring="roc_auc").fit(X_tr, y_tr)
                dec_l = gs_lsvm.decision_function(X_va)
                oof_p["Linear_SVM"][va], oof_y["Linear_SVM"][va] = 1.0 / (1.0 + np.exp(-dec_l)), gs_lsvm.predict(X_va)

                gs_rbf = GridSearchCV(SVC(kernel="rbf", probability=False, max_iter=10000, random_state=seed), {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto", 0.01, 0.1]}, cv=icv, scoring="roc_auc").fit(X_tr, y_tr)
                dec_r = gs_rbf.decision_function(X_va)
                oof_p["RBF_SVM"][va], oof_y["RBF_SVM"][va] = 1.0 / (1.0 + np.exp(-dec_r)), gs_rbf.predict(X_va)

                rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed).fit(X_tr, y_tr)
                oof_p["RandomForest"][va], oof_y["RandomForest"][va] = rf.predict_proba(X_va)[:, 1], rf.predict(X_va)

                gb = HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=seed).fit(X_tr, y_tr)
                oof_p["GBDT"][va], oof_y["GBDT"][va] = gb.predict_proba(X_va)[:, 1], gb.predict(X_va)

                th_tr, th_va = tz[:, FOCAL_5, 0].mean(dim=-1, keepdim=True).numpy(), vz[:, FOCAL_5, 0].mean(dim=-1, keepdim=True).numpy()
                lth = LogisticRegression(random_state=seed).fit(th_tr, y_tr)
                oof_p["Single_Theta"][va], oof_y["Single_Theta"][va] = lth.predict_proba(th_va)[:, 1], lth.predict(th_va)

                ch_tr, ch_va = tz[:, :, 3].mean(dim=-1, keepdim=True).numpy(), vz[:, :, 3].mean(dim=-1, keepdim=True).numpy()
                lch = LogisticRegression(random_state=seed).fit(ch_tr, y_tr)
                oof_p["Single_Chi"][va], oof_y["Single_Chi"][va] = lch.predict_proba(ch_va)[:, 1], lch.predict(ch_va)

                if r_idx == 0:
                    dl_tr, dl_va = tz[:, :, 1].mean(dim=-1, keepdim=True).numpy(), vz[:, :, 1].mean(dim=-1, keepdim=True).numpy()
                    oof_lofo["Single_Delta"][va] = LogisticRegression(random_state=42).fit(dl_tr, y_tr).predict_proba(dl_va)[:, 1]
                    al_tr, al_va = tz[:, POSTERIOR_5, 2].mean(dim=-1, keepdim=True).numpy(), vz[:, POSTERIOR_5, 2].mean(dim=-1, keepdim=True).numpy()
                    oof_lofo["Single_Alpha"][va] = LogisticRegression(random_state=42).fit(al_tr, y_tr).predict_proba(al_va)[:, 1]
                    g_tr = np.array([vg[i]["gce"] for i in tr]).reshape(-1, 1)
                    g_va = np.array([vg[i]["gce"] for i in va]).reshape(-1, 1)
                    mg, sg = g_tr[y_tr == 0].mean(), g_tr[y_tr == 0].std() + 1e-6
                    oof_lofo["Single_GCE"][va] = LogisticRegression(random_state=42).fit((g_tr - mg) / sg, y_tr).predict_proba((g_va - mg) / sg)[:, 1]
                    for kab, idxs in [("GCN_Ablate_Theta", [1, 2, 3]), ("GCN_Ablate_Delta", [0, 2, 3]), ("GCN_Ablate_Alpha", [0, 1, 3]), ("GCN_Ablate_Chi", [0, 1, 2])]:
                        oof_lofo[kab][va], _ = train_eval_gcn(tz[:, :, idxs], adj_omst[tr], ty, vz[:, :, idxs], adj_omst[va], in_dim=3, seed=seed + fold)

            om_auc = float(roc_auc_score(y_all, oof_p["GCN_OMST"]))
            for m in models_13:
                auc_m = float(roc_auc_score(y_all, oof_p[m]))
                bacc_m = float(balanced_accuracy_score(y_all, oof_y[m]))
                acc_m = float(accuracy_score(y_all, oof_y[m]))
                repeat_rows.append({
                    "cohort": cname, "repeat": r_idx, "seed": seed, "model": m, "n_sessions": n, "n_participants": len(set(groups)),
                    "auc": auc_m, "balanced_accuracy": bacc_m, "accuracy": acc_m, "delta_vs_omst": auc_m - om_auc,
                })
                if r_idx == 0:
                    if is_grp:
                        auc0 = auc_m
                        lo0, hi0 = cluster_bootstrap_auc_ci(y_all, oof_p[m], groups)
                    else:
                        auc0, lo0, hi0 = bootstrap_auc_ci(y_all, oof_p[m])
                    rec0 = {"cohort": cname, "model": m, "auc": auc0, "ci_lower": lo0, "ci_upper": hi0, "accuracy": acc_m, "balanced_accuracy": bacc_m}
                    if cname == "Study 2b: Mumtaz MDD (Eyes Open)":
                        r_eo = dict(rec0)
                        r_eo["cohort"] = "Study 2b: Mumtaz Outpatient MDD (Eyes-Open)"
                        eo_split0_rows.append(r_eo)
                    split0_cls_rows.append(rec0)
            if r_idx == 0:
                if cname != "Study 2b: Mumtaz MDD (Eyes Open)":
                    rl = {"cohort": cname, "n_samples": n, "GCN_Full4_auc": om_auc, "Single_Theta_auc": float(roc_auc_score(y_all, oof_p["Single_Theta"])), "Single_Chi_auc": float(roc_auc_score(y_all, oof_p["Single_Chi"])), "Tuned_LR76_auc": float(roc_auc_score(y_all, oof_p["Tuned_LogisticRegression"]))}
                    for k, pr in oof_lofo.items():
                        rl[f"{k}_auc"] = float(roc_auc_score(y_all, pr))
                    lofo_rows.append(rl)
                for i_s in range(n):
                    oof_split0_rows.append({"cohort": cname, "subject_id": vg[i_s]["subject_id"], "y": int(y_all[i_s]), "prob_GCN_OMST": float(oof_p["GCN_OMST"][i_s]), "prob_GCN_Weighted_PLI": float(oof_p["GCN_Weighted_PLI"][i_s]), "prob_Tuned_LR": float(oof_p["Tuned_LogisticRegression"][i_s]), "prob_RF": float(oof_p["RandomForest"][i_s])})

        sub_g = [r["auc"] for r in repeat_rows if r["cohort"] == cname and r["model"] == "GCN_OMST"]
        sub_w = [r["auc"] for r in repeat_rows if r["cohort"] == cname and r["model"] == "GCN_Weighted_PLI"]
        sub_l = [r["auc"] for r in repeat_rows if r["cohort"] == cname and r["model"] == "Linear_SVM"]
        sub_r = [r["auc"] for r in repeat_rows if r["cohort"] == cname and r["model"] == "RBF_SVM"]
        print(f"  {cname:<42} | OMST={np.mean(sub_g):.3f} | W-PLI={np.mean(sub_w):.3f} | LinSVM={np.mean(sub_l):.3f} | RBF={np.mean(sub_r):.3f} ({time.time()-t_c:.1f}s)")

    df_rep = pd.DataFrame(repeat_rows)
    df_rep.to_csv(REPORTS_DIR / "benchmark_repeated_10x5_cv.csv", index=False)

    sum_rows = []
    for (cname, m), grp in df_rep.groupby(["cohort", "model"], sort=False):
        sum_rows.append({
            "cohort": cname, "model": m,
            "n_sessions": int(grp["n_sessions"].iloc[0]), "n_participants": int(grp["n_participants"].iloc[0]),
            "seed42_auc": float(grp[grp["seed"] == 42]["auc"].iloc[0]),
            "mean_auc": float(grp["auc"].mean()), "std_auc": float(grp["auc"].std(ddof=1)),
            "min_auc": float(grp["auc"].min()), "max_auc": float(grp["auc"].max()),
            "mean_bacc": float(grp["balanced_accuracy"].mean()), "mean_acc": float(grp["accuracy"].mean()),
            "mean_delta_vs_omst": float(grp["delta_vs_omst"].mean()), "std_delta_vs_omst": float(grp["delta_vs_omst"].std(ddof=1)),
        })
    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(REPORTS_DIR / "benchmark_repeated_10x5_summary.csv", index=False)
    df_sum.to_csv(REPORTS_DIR / "benchmark_multiseed_summary.csv", index=False)

    # Embed 10x5 mean_auc & sd_auc cleanly into Table S10 and Table S12A CSVs
    sum_idx = df_sum.set_index(["cohort", "model"])
    df_s10 = pd.DataFrame(split0_cls_rows)
    df_s10["mean_auc_10x5"] = [float(sum_idx.loc[(r["cohort"], r["model"]), "mean_auc"]) for _, r in df_s10.iterrows()]
    df_s10["sd_auc_10x5"] = [float(sum_idx.loc[(r["cohort"], r["model"]), "std_auc"]) for _, r in df_s10.iterrows()]
    df_s10.to_csv(REPORTS_DIR / "multicohort_7cohort_extended_model_comparison.csv", index=False)

    df_eo = pd.DataFrame(eo_split0_rows)
    df_eo["mean_auc_10x5"] = [float(sum_idx.loc[("Study 2b: Mumtaz MDD (Eyes Open)", r["model"]), "mean_auc"]) for _, r in df_eo.iterrows()]
    df_eo["sd_auc_10x5"] = [float(sum_idx.loc[("Study 2b: Mumtaz MDD (Eyes Open)", r["model"]), "std_auc"]) for _, r in df_eo.iterrows()]
    df_eo.to_csv(REPORTS_DIR / "mumtaz_eo_model_comparison.csv", index=False)

    pd.DataFrame(lofo_rows).to_csv(REPORTS_DIR / "multicohort_7cohort_feature_ablation.csv", index=False)
    pd.DataFrame(fold_index_rows).to_csv(RELEASE_DIR / "cv_fold_indices_split0.csv", index=False)
    pd.DataFrame(oof_split0_rows).to_csv(RELEASE_DIR / "oof_predictions_split0.csv", index=False)

    # Wilcoxon + Paired Delta 95% CIs + TOST Equivalence (+/-0.02, +/-0.03)
    piv_rep = df_sum[df_sum["cohort"] != "Study 2b: Mumtaz MDD (Eyes Open)"].pivot(index="cohort", columns="model", values="mean_auc")
    primary_9 = [c for c in piv_rep.index if c not in ("Study 6b: UNM PD-ON vs Control", "Study 6c: UNM PD All Sessions")]
    nonnested_8 = [c for c in primary_9 if c != "Study 5c: Dementia All (AD+FTD) vs Control"]
    all_11 = list(piv_rep.index)
    others = [m for m in models_13 if m != "GCN_OMST"]

    w_rows = []
    for sname, idxs in [("Primary_9_Contrasts_10x5Mean", primary_9), ("NonNested_8_Contrasts_10x5Mean", nonnested_8), ("All_11_Contrasts_10x5Mean", all_11)]:
        plist, tmp = [], []
        k_n = len(idxs)
        t_crit = float(stats.t.ppf(0.975, df=k_n - 1))
        for m in others:
            x, y = piv_rep.loc[idxs, "GCN_OMST"].values, piv_rep.loc[idxs, m].values
            diff = x - y
            md = float(diff.mean())
            se = float(diff.std(ddof=1) / np.sqrt(k_n)) + 1e-12
            ci_lo, ci_hi = md - t_crit * se, md + t_crit * se
            tost02 = max(float(1.0 - stats.t.cdf((md - (-0.02)) / se, df=k_n - 1)), float(stats.t.cdf((md - 0.02) / se, df=k_n - 1)))
            tost03 = max(float(1.0 - stats.t.cdf((md - (-0.03)) / se, df=k_n - 1)), float(stats.t.cdf((md - 0.03) / se, df=k_n - 1)))
            w_st, p_v = wilcoxon(x, y) if np.any(np.abs(diff) > 1e-8) else (0.0, 1.0)
            plist.append(p_v)
            tmp.append({
                "scope": sname, "n_benchmarks": k_n, "comparator_model": m,
                "gcn_mean_auc": float(x.mean()), "comp_mean_auc": float(y.mean()),
                "mean_diff_auc": md, "median_diff_auc": float(np.median(diff)),
                "ci95_diff_lo": ci_lo, "ci95_diff_hi": ci_hi,
                "tost_p_002": tost02, "tost_p_003": tost03,
                "wins_ties_losses": f"{int(np.sum(diff > 1e-6))} / {int(np.sum(np.abs(diff) <= 1e-6))} / {int(np.sum(diff < -1e-6))}",
                "wilcoxon_W": float(w_st), "raw_p": float(p_v),
            })
        _, qvals, _, _ = multipletests(plist, method="fdr_bh")
        for r, qv in zip(tmp, qvals):
            r["fdr_q"] = float(qv)
            w_rows.append(r)
    pd.DataFrame(w_rows).to_csv(REPORTS_DIR / "multicohort_wilcoxon_model_comparisons.csv", index=False)

    # GCN Hyperparameter sensitivity (deduplicated Study 2)
    grid = [
        (0.005, 50, 32, "Baseline (lr=0.005, ep=50, h=32)"),
        (0.001, 50, 32, "Lower LR (lr=0.001, ep=50, h=32)"),
        (0.010, 50, 32, "Higher LR (lr=0.010, ep=50, h=32)"),
        (0.005, 30, 32, "Fewer Epochs (lr=0.005, ep=30, h=32)"),
        (0.005, 100, 32, "More Epochs (lr=0.005, ep=100, h=32)"),
        (0.005, 50, 16, "Smaller Capacity (lr=0.005, ep=50, h=16)"),
        (0.005, 50, 64, "Larger Capacity (lr=0.005, ep=50, h=64)"),
    ]
    g_cohorts = [
        ("Study 2: Mumtaz Outpatient MDD (EC)", mum_ec),
        ("Study 3: Cavanagh Univ. Depression (EO)", cav),
        ("Study 4: UCSD Parkinson All", ucsd),
        ("Study 6a: UNM PD-OFF vs CTL", [g for g in unm if g["med_state"] in ("OFF", "CTL")]),
        ("Study 7: Iowa Parkinson", iowa),
    ]
    g_rows = []
    for cname, glist in g_cohorts:
        vg = [g for g in glist if int(g["y"]) in (0, 1)]
        y_all = np.array([int(g["y"]) for g in vg])
        grps = np.array([g.get("bids_id", g.get("sub_name", g["subject_id"])) for g in vg])
        splits = get_grouped_splits(grps, y_all, n_splits=5, seed=42)
        for lr, ep, hd, clbl in grid:
            oof = np.zeros(len(vg))
            for fold, (tr, va) in enumerate(splits):
                tx, vx = torch.stack([vg[i]["x"] for i in tr]), torch.stack([vg[i]["x"] for i in va])
                ty = torch.tensor(y_all[tr], dtype=torch.long)
                tadj, vadj = torch.stack([vg[i]["adj"] for i in tr]), torch.stack([vg[i]["adj"] for i in va])
                tz, vz, _, _ = compute_fold_normative_z(tx, ty, vx)
                oof[va], _ = train_eval_gcn(tz, tadj, ty, vz, vadj, hidden_dim=hd, epochs=ep, lr=lr, seed=42 + fold)
            g_rows.append({"cohort": cname, "n_samples": len(vg), "config_label": clbl, "learning_rate": lr, "epochs": ep, "hidden_dim": hd, "auc": float(roc_auc_score(y_all, oof))})
    pd.DataFrame(g_rows).to_csv(REPORTS_DIR / "benchmark_gcn_hyperparameter_sensitivity.csv", index=False)
    print("✓ Module C complete.")


# ==============================================================================
# PART 7: MODULE D ALIGNED PERMUTATIONS + SPATIAL COVARIANCE EXTRACTION + MODULE E
# ==============================================================================
def _perm_worker_aligned(args):
    tid, X_src, adj_src, y_src, grps_src, X_tgt, adj_tgt, y_tgt, is_transfer, seed_list = args
    torch.set_num_threads(1)
    null_gcn, null_lr = [], []
    for pseed in seed_list:
        rng = np.random.RandomState(pseed)
        if is_transfer:
            y_perm_tr = rng.permutation(y_src)
            ty_t = torch.tensor(y_perm_tr, dtype=torch.long)
            if int((ty_t == 0).sum()) < 2 or int((ty_t == 1).sum()) < 2:
                null_gcn.append(0.5); null_lr.append(0.5); continue
            hc_idx, pt_idx = np.where(y_tgt == 0)[0], np.where(y_tgt == 1)[0]
            perm_hc = np.random.RandomState(42).permutation(hc_idx)
            cal_hc, eval_hc = perm_hc[:len(perm_hc) // 2], perm_hc[len(perm_hc) // 2:]
            eval_idx = np.concatenate([pt_idx, eval_hc])
            tz, _, _, _ = compute_fold_normative_z(X_src, ty_t, X_src)
            mu_hc, sd_hc = X_tgt[cal_hc].mean(dim=0, keepdim=True), X_tgt[cal_hc].std(dim=0, keepdim=True) + 1e-6
            vz_eval = (X_tgt[eval_idx] - mu_hc) / sd_hc
            pr_gcn, _ = train_eval_gcn(tz, adj_src, ty_t, vz_eval, adj_tgt[eval_idx], epochs=40, seed=42)
            null_gcn.append(float(roc_auc_score(y_tgt[eval_idx], pr_gcn)))
            icv = get_grouped_splits(grps_src, y_perm_tr, n_splits=3, seed=42)
            gs_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}, cv=icv, scoring="roc_auc").fit(tz.view(len(ty_t), -1).numpy(), y_perm_tr)
            null_lr.append(float(roc_auc_score(y_tgt[eval_idx], gs_lr.predict_proba(vz_eval.view(len(eval_idx), -1).numpy())[:, 1])))
        else:
            uniq_g = np.unique(grps_src)
            g_map = {ug: int(y_src[grps_src == ug][0]) for ug in uniq_g}
            pmap = dict(zip(uniq_g, rng.permutation(list(g_map.values()))))
            y_perm = np.array([pmap[g] for g in grps_src], dtype=int)
            splits = get_grouped_splits(grps_src, y_perm, n_splits=5, seed=42)
            oof_g, oof_l = np.zeros(len(y_perm)), np.zeros(len(y_perm))
            valid_perm = True
            for fold, (tr, va) in enumerate(splits):
                ty_t = torch.tensor(y_perm[tr], dtype=torch.long)
                if int((ty_t == 0).sum()) < 2 or int((ty_t == 1).sum()) < 2:
                    valid_perm = False; break
                tz, vz, _, _ = compute_fold_normative_z(X_src[tr], ty_t, X_src[va])
                oof_g[va], _ = train_eval_gcn(tz, adj_src[tr], ty_t, vz, adj_src[va], epochs=50, seed=42 + fold)
                icv = get_grouped_splits(grps_src[tr], y_perm[tr], n_splits=3, seed=42 + fold)
                gs_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}, cv=icv, scoring="roc_auc").fit(tz.view(len(tr), -1).numpy(), y_perm[tr])
                oof_l[va] = gs_lr.predict_proba(vz.view(len(va), -1).numpy())[:, 1]
            if valid_perm:
                null_gcn.append(float(roc_auc_score(y_perm, oof_g)))
                null_lr.append(float(roc_auc_score(y_perm, oof_l)))
            else:
                null_gcn.append(0.5); null_lr.append(0.5)
    return tid, null_gcn, null_lr


def run_module_d_permutations(sp):
    print("\n=== [Step 7] Module D: Aligned 1,000-Permutation Classification Tests ===")
    dem, iowa, unm, cav, sleep = sp["dementia_ds004504"], sp["iowa_ds004584"], sp["unm_ds003490"], sp["cavanagh_ds003478"], sp["sleep_ds004902"]
    unm_off = [g for g in unm if g["med_state"] in ("OFF", "CTL")]

    def pack_cv(tid, label, glist):
        vg = [g for g in glist if int(g["y"]) in (0, 1)]
        X, A = torch.stack([g["x"] for g in vg]), torch.stack([g["adj"] for g in vg])
        y = np.array([int(g["y"]) for g in vg])
        grps = np.array([g.get("bids_id", g.get("sub_name", g["subject_id"])) for g in vg])
        splits = get_grouped_splits(grps, y, n_splits=5, seed=42)
        og, ol = np.zeros(len(vg)), np.zeros(len(vg))
        for fold, (tr, va) in enumerate(splits):
            tz, vz, _, _ = compute_fold_normative_z(X[tr], torch.tensor(y[tr]), X[va])
            og[va], _ = train_eval_gcn(tz, A[tr], torch.tensor(y[tr]), vz, A[va], epochs=50, seed=42 + fold)
            icv = get_grouped_splits(grps[tr], y[tr], n_splits=3, seed=42 + fold)
            gs_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}, cv=icv, scoring="roc_auc").fit(tz.view(len(tr), -1).numpy(), y[tr])
            ol[va] = gs_lr.predict_proba(vz.view(len(va), -1).numpy())[:, 1]
        return {"id": tid, "label": label, "is_transfer": False, "obs_gcn": float(roc_auc_score(y, og)), "obs_lr": float(roc_auc_score(y, ol)), "X_src": X, "A_src": A, "y_src": y, "grps_src": grps, "X_tgt": None, "A_tgt": None, "y_tgt": None}

    def pack_tr(tid, label, src_gs, tgt_gs):
        sv, tv = [g for g in src_gs if int(g["y"]) in (0, 1)], [g for g in tgt_gs if int(g["y"]) in (0, 1)]
        Xs, As = torch.stack([g["x"] for g in sv]), torch.stack([g["adj"] for g in sv])
        ys = np.array([int(g["y"]) for g in sv])
        grps_s = np.array([g.get("bids_id", g.get("sub_name", g["subject_id"])) for g in sv])
        Xt, At = torch.stack([g["x"] for g in tv]), torch.stack([g["adj"] for g in tv])
        yt = np.array([int(g["y"]) for g in tv])
        hc_idx, pt_idx = np.where(yt == 0)[0], np.where(yt == 1)[0]
        perm_hc = np.random.RandomState(42).permutation(hc_idx)
        cal_hc, eval_hc = perm_hc[:len(perm_hc) // 2], perm_hc[len(perm_hc) // 2:]
        eval_idx = np.concatenate([pt_idx, eval_hc])
        tz, _, _, _ = compute_fold_normative_z(Xs, torch.tensor(ys), Xs)
        mu_hc, sd_hc = Xt[cal_hc].mean(dim=0, keepdim=True), Xt[cal_hc].std(dim=0, keepdim=True) + 1e-6
        vz_eval = (Xt[eval_idx] - mu_hc) / sd_hc
        pr_g, _ = train_eval_gcn(tz, As, torch.tensor(ys), vz_eval, At[eval_idx], epochs=40, seed=42)
        icv = get_grouped_splits(grps_s, ys, n_splits=3, seed=42)
        gs_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}, cv=icv, scoring="roc_auc").fit(tz.view(len(sv), -1).numpy(), ys)
        pr_l = gs_lr.predict_proba(vz_eval.view(len(eval_idx), -1).numpy())[:, 1]
        return {"id": tid, "label": label, "is_transfer": True, "obs_gcn": float(roc_auc_score(yt[eval_idx], pr_g)), "obs_lr": float(roc_auc_score(yt[eval_idx], pr_l)), "X_src": Xs, "A_src": As, "y_src": ys, "grps_src": grps_s, "X_tgt": Xt, "A_tgt": At, "y_tgt": yt}

    tasks = [
        pack_cv("A", "Study 5c: Clinical Dementia All (Split-0 CV)", dem),
        pack_cv("B", "Study 7: Iowa Parkinson's (Split-0 CV)", iowa),
        pack_cv("C", "Study 6a: UNM PD-OFF EO (Split-0 CV)", unm_off),
        pack_cv("D", "Study 3: Univ. Depression Screening (Split-0 CV)", cav),
        pack_tr("E", "Transfer (Split-HC): Iowa PD → UNM PD-OFF", iowa, unm_off),
        pack_tr("F", "Transfer (Split-HC): Sleep Debt → Dementia", sleep, dem),
    ]
    n_perms, chunk = 1000, 50
    jobs = [(t["id"], t["X_src"], t["A_src"], t["y_src"], t["grps_src"], t["X_tgt"], t["A_tgt"], t["y_tgt"], t["is_transfer"], [10000 + s for s in range(st, min(st + chunk, n_perms))]) for t in tasks for st in range(0, n_perms, chunk)]
    agg = {t["id"]: {"gcn": [], "lr": []} for t in tasks}
    n_workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as ex:
        for tid, ng, nl in ex.map(_perm_worker_aligned, jobs):
            agg[tid]["gcn"].extend(ng); agg[tid]["lr"].extend(nl)

    perm_out = {}
    for t in tasks:
        tid = t["id"]
        ng, nl = np.asarray(agg[tid]["gcn"]), np.asarray(agg[tid]["lr"])
        pg = float((np.sum(np.abs(ng - 0.5) >= abs(t["obs_gcn"] - 0.5)) + 1.0) / (len(ng) + 1.0))
        pl = float((np.sum(np.abs(nl - 0.5) >= abs(t["obs_lr"] - 0.5)) + 1.0) / (len(nl) + 1.0))
        perm_out[tid] = {
            "label": t["label"], "n_perms": len(ng),
            "obs_gcn_auc": t["obs_gcn"], "obs_lr_auc": t["obs_lr"],
            "null_gcn_mean": float(ng.mean()), "null_gcn_sd": float(ng.std(ddof=1)), "p_perm_gcn": pg,
            "null_lr_mean": float(nl.mean()), "null_lr_sd": float(nl.std(ddof=1)), "p_perm_lr": pl,
            "null_gcn_aucs": ng.tolist(), "null_lr_aucs": nl.tolist(),
        }
        print(f"  [{tid}] {t['label']:<44} | GCN={t['obs_gcn']:.3f} (p={pg:.4f}) | TunedLR={t['obs_lr']:.3f} (p={pl:.4f})")
    with open(REPORTS_DIR / "real_permutation_distributions.json", "w", encoding="utf-8") as f:
        json.dump(perm_out, f)


def extract_or_load_spatial_covariances(sp):
    cache_p = DATA_DIR / "spatial_covariances_7cohorts.pt"
    if cache_p.exists():
        covs = torch.load(cache_p, weights_only=False)
        # Ensure counts align with deduplicated sp
        ok = all(len(covs.get(ck, {})) == len(sp[ck]) for ck in ["sleep_ds004902", "mumtaz_ec", "cavanagh_ds003478", "ucsd_ds002778", "dementia_ds004504", "unm_ds003490", "iowa_ds004584"])
        if ok:
            print("✓ Loaded cached 19x19 sample spatial covariances (675/675 sessions).")
            return covs

    print("\n=== Extracting 19x19 Sample Spatial Covariance Matrices Across All 7 Cohorts ===")
    t0 = time.time()
    adapter_150 = mcl.Universal19ChannelAdapter(target_sfreq=250.0, l_freq=1.0, h_freq=40.0, epoch_duration=4.0, ptp_threshold_uv=150.0)
    adapter_200 = mcl.Universal19ChannelAdapter(target_sfreq=250.0, l_freq=1.0, h_freq=40.0, epoch_duration=4.0, ptp_threshold_uv=200.0)

    def epochs_to_cov(ep_data: np.ndarray) -> np.ndarray:
        # Reshape [N_ep, 19, T] -> [19, N_ep*T], demean per channel, scale-normalized sample covariance + 1e-5*tr/19 I
        X = np.transpose(ep_data, (1, 0, 2)).reshape(19, -1).astype(np.float64) * 1e6  # in uV for numeric stability
        X = X - X.mean(axis=1, keepdims=True)
        C = (X @ X.T) / max(1, X.shape[1])
        C = 0.5 * (C + C.T)
        tr = float(np.trace(C))
        C = C + (1e-5 * (tr / 19.0)) * np.eye(19)
        return C.astype(np.float64)

    def standardize_iowa_pz(raw):
        canon_map = {ch: c for c in raw.ch_names for ch in CHANNELS_19 if ch.lower() == c.lower()}
        present = [c for c in CHANNELS_19 if c in canon_map]
        r = raw.copy()
        r.pick([canon_map[c] for c in present])
        r.rename_channels({canon_map[c]: c for c in present})
        if "Pz" not in r.ch_names:
            info_pz = mne.create_info(["Pz"], sfreq=r.info["sfreq"], ch_types=["eeg"])
            pz_raw = mne.io.RawArray(np.zeros((1, r.n_times)), info_pz, verbose=False)
            r.add_channels([pz_raw], force_update_info=True)
        r.reorder_channels(CHANNELS_19)
        r.set_eeg_reference("average", projection=False, verbose=False)
        r.filter(1.0, 40.0, fir_design="firwin", verbose=False)
        if abs(r.info["sfreq"] - 250.0) > 1e-3:
            r.resample(250.0, verbose=False)
        d = r.get_data()
        ep_len = int(4.0 * 250.0)
        n_ep = d.shape[1] // ep_len
        eps = np.stack([d[:, i * ep_len:(i + 1) * ep_len] for i in range(n_ep)])
        ptp = np.ptp(eps, axis=-1) * 1e6
        keep = np.max(ptp, axis=-1) <= 200.0
        return eps[keep] if np.sum(keep) >= 5 else eps[np.argsort(np.max(ptp, axis=-1))[:min(10, len(eps))]]

    def unm_eo_contig_blocks(df_ev):
        eo = df_ev[df_ev["trial_type"].astype(str).str.contains("Eyes Open", na=False)]
        onsets = np.sort(eo["onset"].to_numpy(dtype=float))
        if len(onsets) == 0:
            return [(0.0, 60.0)]
        blocks, st, prev = [], onsets[0], onsets[0]
        for o in onsets[1:]:
            if o - prev > 3.0:
                blocks.append((st, prev + 1.0))
                st = o
            prev = o
        blocks.append((st, prev + 1.0))
        return blocks

    covs = {}
    # 1. Sleep
    m_sleep = {x["subject_id"]: x["path"] for x in mcl.get_sleep_deprivation_manifest(SLEEP_FIF_DIR)}
    covs["sleep_ds004902"] = {}
    for r in sp["sleep_ds004902"]:
        ep = mne.read_epochs(str(m_sleep[r["subject_id"]]), preload=True, verbose=False).pick(CHANNELS_19).get_data(copy=True)
        covs["sleep_ds004902"][r["subject_id"]] = epochs_to_cov(ep)

    # 2. Mumtaz EC (deduplicated)
    m_mum = {x["subject_id"]: x["path"] for x in mcl.get_mumtaz_mdd_manifest(MUMTAZ_DIR, state="EC")}
    covs["mumtaz_ec"] = {}
    for r in sp["mumtaz_ec"]:
        ep, _ = adapter_150.standardize_raw_to_epochs(mne.io.read_raw_edf(str(m_mum[r["subject_id"]]), preload=True, verbose=False))
        covs["mumtaz_ec"][r["subject_id"]] = epochs_to_cov(ep)

    # 3. Cavanagh
    m_cav = {x["subject_id"]: x["path"] for x in mcl.get_cavanagh_depression_manifest(CAVANAGH_DIR)}
    covs["cavanagh_ds003478"] = {}
    for r in sp["cavanagh_ds003478"]:
        ep, _ = adapter_150.standardize_raw_to_epochs(mne.io.read_raw_eeglab(str(m_cav[r["subject_id"]]), preload=True, verbose=False))
        covs["cavanagh_ds003478"][r["subject_id"]] = epochs_to_cov(ep)

    # 4. UCSD
    m_ucsd = {x["subject_id"]: x["path"] for x in mcl.get_ucsd_parkinson_manifest(EEG_ROOT)}
    covs["ucsd_ds002778"] = {}
    for r in sp["ucsd_ds002778"]:
        ep, _ = adapter_150.standardize_raw_to_epochs(mne.io.read_raw_bdf(str(m_ucsd[r["subject_id"]]), preload=True, verbose=False))
        covs["ucsd_ds002778"][r["subject_id"]] = epochs_to_cov(ep)

    # 5. Dementia
    covs["dementia_ds004504"] = {}
    for r in sp["dementia_ds004504"]:
        sid = r["subject_id"]
        sf = list((DEMENTIA_DIR / sid / "eeg").glob("*task-eyesclosed*_eeg.set")) or list((DEMENTIA_DIR / sid / "eeg").glob("*.set"))
        ep, _ = adapter_150.standardize_raw_to_epochs(mne.io.read_raw_eeglab(str(sf[0]), preload=True, verbose=False))
        covs["dementia_ds004504"][sid] = epochs_to_cov(ep)

    # 6. UNM EO (keyed by canonical sub_ses UID)
    covs["unm_ds003490"] = {}
    for r in sp["unm_ds003490"]:
        sid, ses = r["subject_id"], r["session"]
        uid = f"{sid}_{ses}" if r.get("group") == "PD" else sid
        sf = sorted((UNM_DIR / sid / ses / "eeg").glob("*task-Rest*.set"))[0]
        ef = sorted((UNM_DIR / sid / ses / "eeg").glob("*events.tsv"))[0]
        t_a, t_b = unm_eo_contig_blocks(pd.read_csv(ef, sep="\t"))[0]
        raw_eo = mne.io.read_raw_eeglab(str(sf), preload=True, verbose=False)
        raw_eo = raw_eo.crop(tmin=t_a, tmax=min(t_b, float(raw_eo.times[-1])))
        ep, _ = adapter_150.standardize_raw_to_epochs(raw_eo)
        covs["unm_ds003490"][uid] = epochs_to_cov(ep)

    # 7. Iowa
    covs["iowa_ds004584"] = {}
    for r in sp["iowa_ds004584"]:
        sid = r["subject_id"]
        sf = list((IOWA_DIR / sid / "eeg").glob("*_eeg.set"))[0]
        ep = standardize_iowa_pz(mne.io.read_raw_eeglab(str(sf), preload=True, verbose=False))
        covs["iowa_ds004584"][sid] = epochs_to_cov(ep)

    torch.save(covs, cache_p)
    print(f"✓ Extracted & cached all 19x19 sample spatial covariances in {time.time() - t0:.1f}s.")
    return covs


def _reg_perm_worker(args):
    jkey, mname, X, y, seed_list = args
    r_null = []
    for ps in seed_list:
        yp = np.random.RandomState(ps).permutation(y)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(y))
        for tr, va in kf.split(X, yp):
            mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-6
            if mname == "GBDT_Regressor":
                est = HistGradientBoostingRegressor(max_iter=100, max_depth=4, random_state=42)
            elif mname == "Random_Forest_Regressor":
                est = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=1)
            elif mname == "SVR_RBF":
                est = SVR(C=1.0, epsilon=0.1)
            else:
                est = Ridge(alpha=10.0)
            est.fit((X[tr] - mu) / sd, yp[tr])
            oof[va] = est.predict((X[va] - mu) / sd)
        rv_p, _ = pearsonr(yp, oof)
        r_null.append(0.0 if np.isnan(rv_p) else float(rv_p))
    return jkey, mname, r_null


def run_module_e_rest(sp, psds, spatial_covs):
    print("\n=== [Step 8] Module E: Split-HC Transfer, 10x5 Leakage, PLI SPD, Spatial Cov Riemannian, Gain, Perm Regressions ===")
    sleep, mum_ec, mum_eo = sp["sleep_ds004902"], sp["mumtaz_ec"], sp["mumtaz_eo"]
    cav, ucsd, dem, unm, iowa = sp["cavanagh_ds003478"], sp["ucsd_ds002778"], sp["dementia_ds004504"], sp["unm_ds003490"], sp["iowa_ds004584"]
    unm_off = [g for g in unm if g["med_state"] in ("OFF", "CTL")]

    def eval_transfer_pair(src_gs, tgt_gs, sn, tn, ttype):
        sv = [g for g in src_gs if int(g["y"]) in (0, 1)]
        tv = [g for g in tgt_gs if int(g["y"]) in (0, 1)]
        Xs, As = torch.stack([g["x"] for g in sv]), torch.stack([g["adj"] for g in sv])
        ys_t = torch.tensor([int(g["y"]) for g in sv], dtype=torch.long)
        grps_s = np.array([g.get("bids_id", g.get("sub_name", g["subject_id"])) for g in sv])
        tz, _, _, _ = compute_fold_normative_z(Xs, ys_t, Xs)
        Xtr, ytr = tz.view(len(sv), -1).numpy(), ys_t.numpy()
        icv = get_grouped_splits(grps_s, ytr, n_splits=3, seed=42)
        lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}, cv=icv, scoring="roc_auc").fit(Xtr, ytr)
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(Xtr, ytr)
        gb = HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=42).fit(Xtr, ytr)

        set_seed(42)
        gcn_omst = NormativeGCN(4, 32, 2, 0.25)
        opt1 = optim.Adam(gcn_omst.parameters(), lr=0.005, weight_decay=1e-4)
        gcn_omst.train()
        for _ in range(40):
            opt1.zero_grad(); nn.CrossEntropyLoss()(gcn_omst(tz, As), ys_t).backward(); opt1.step()
        gcn_omst.eval()

        set_seed(42)
        gcn_id = NormativeGCN(4, 32, 2, 0.25)
        adj_id_s = torch.zeros((len(sv), 19, 19), dtype=torch.float32)
        opt2 = optim.Adam(gcn_id.parameters(), lr=0.005, weight_decay=1e-4)
        gcn_id.train()
        for _ in range(40):
            opt2.zero_grad(); nn.CrossEntropyLoss()(gcn_id(tz, adj_id_s), ys_t).backward(); opt2.step()
        gcn_id.eval()

        Xt, At = torch.stack([g["x"] for g in tv]), torch.stack([g["adj"] for g in tv])
        yt = np.array([int(g["y"]) for g in tv])
        hc_idx, pt_idx = np.where(yt == 0)[0], np.where(yt == 1)[0]
        mu_f, sd_f = Xt[hc_idx].mean(dim=0, keepdim=True), Xt[hc_idx].std(dim=0, keepdim=True) + 1e-6
        vz_f = (Xt - mu_f) / sd_f
        with torch.no_grad():
            pf_gcn = torch.softmax(gcn_omst(vz_f, At), dim=-1)[:, 1].numpy()
            pf_id = torch.softmax(gcn_id(vz_f, torch.zeros((len(tv), 19, 19))), dim=-1)[:, 1].numpy()
        Xte_f = vz_f.view(len(tv), -1).numpy()
        oracle = {
            "GCN_OMST": float(roc_auc_score(yt, pf_gcn)), "GCN_Identity": float(roc_auc_score(yt, pf_id)),
            "Tuned_LR": float(roc_auc_score(yt, lr.predict_proba(Xte_f)[:, 1])),
            "RandomForest": float(roc_auc_score(yt, rf.predict_proba(Xte_f)[:, 1])),
            "GBDT": float(roc_auc_score(yt, gb.predict_proba(Xte_f)[:, 1])),
        }

        rng = np.random.RandomState(42)
        split_aucs = {m: [] for m in oracle}
        split_baccs = {m: [] for m in oracle}
        for _ in range(20):
            ph = rng.permutation(hc_idx)
            cal_h, ev_h = ph[:len(ph) // 2], ph[len(ph) // 2:]
            ev_idx = np.concatenate([pt_idx, ev_h])
            mu_s, sd_s = Xt[cal_h].mean(dim=0, keepdim=True), Xt[cal_h].std(dim=0, keepdim=True) + 1e-6
            vz_s = (Xt[ev_idx] - mu_s) / sd_s
            Xte_s, yte_s = vz_s.view(len(ev_idx), -1).numpy(), yt[ev_idx]
            with torch.no_grad():
                ps_g = torch.softmax(gcn_omst(vz_s, At[ev_idx]), dim=-1)[:, 1].numpy()
                ps_i = torch.softmax(gcn_id(vz_s, torch.zeros((len(ev_idx), 19, 19))), dim=-1)[:, 1].numpy()
            pm = {
                "GCN_OMST": ps_g, "GCN_Identity": ps_i,
                "Tuned_LR": lr.predict_proba(Xte_s)[:, 1], "RandomForest": rf.predict_proba(Xte_s)[:, 1], "GBDT": gb.predict_proba(Xte_s)[:, 1],
            }
            for m, pr in pm.items():
                split_aucs[m].append(float(roc_auc_score(yte_s, pr)))
                split_baccs[m].append(float(balanced_accuracy_score(yte_s, (pr >= 0.5).astype(int))))
        rows = []
        for m in oracle:
            arr = np.asarray(split_aucs[m])
            rows.append({
                "transfer_type": ttype, "source_cohort": sn, "target_cohort": tn,
                "n_source": len(sv), "n_target": len(tv), "model": m,
                "auc": float(arr.mean()), "split_hc_sd": float(arr.std(ddof=1)),
                "split_hc_p05": float(np.percentile(arr, 5)), "split_hc_p95": float(np.percentile(arr, 95)),
                "balanced_accuracy": float(np.mean(split_baccs[m])), "oracle_full_hc_auc": oracle[m],
            })
        return rows

    pd_cohorts = {"UCSD_PD": ucsd, "UNM_PD_OFF": unm_off, "Iowa_PD": iowa}
    all_tr = [r for s in pd_cohorts for t in pd_cohorts if s != t for r in eval_transfer_pair(pd_cohorts[s], pd_cohorts[t], s, t, "Within_Disorder_PD_Cross_Site")]
    pd.DataFrame(all_tr).to_csv(REPORTS_DIR / "cross_site_pd_transfer_benchmarks.csv", index=False)

    all_td = list(all_tr)
    for sg, tg, sn, tn in [(mum_ec, cav, "Mumtaz_MDD", "Cavanagh_MDD"), (cav, mum_ec, "Cavanagh_MDD", "Mumtaz_MDD")]:
        all_td.extend(eval_transfer_pair(sg, tg, sn, tn, "Within_Disorder_MDD_Cross_Site"))
    trans = {"Sleep_Debt": sleep, "Mumtaz_MDD": mum_ec, "Iowa_PD": iowa, "Dementia_All": dem}
    for s in trans:
        for t in trans:
            if s != t and not any(r["source_cohort"] == s and r["target_cohort"] == t for r in all_td):
                all_td.extend(eval_transfer_pair(trans[s], trans[t], s, t, "Cross_Disorder_Transdiagnostic"))
    pd.DataFrame(all_td).to_csv(REPORTS_DIR / "comprehensive_transdiagnostic_transfer_matrix.csv", index=False)

    ec_m, eo_m = {g["subject_id"]: g for g in mum_ec}, {g["subject_id"]: g for g in mum_eo}
    sh_ids = sorted(set(ec_m) & set(eo_m))
    conc_rows = []
    for lbl, fn in [
        ("Frontal Periodic Theta (4–8 Hz Residual)", lambda g: float(g["x"].numpy()[FOCAL_5, 0].mean())),
        ("Frontal Aperiodic Exponent (χ)", lambda g: float(g["x"].numpy()[FOCAL_5, 3].mean())),
        ("Global Aperiodic Exponent (χ)", lambda g: float(g["x"].numpy()[:, 3].mean())),
        ("Global Cost Efficiency (GCE, OMST)", lambda g: float(g["gce"])),
    ]:
        r, p = pearsonr([fn(ec_m[i]) for i in sh_ids], [fn(eo_m[i]) for i in sh_ids])
        conc_rows.append({"marker": lbl, "n_shared": len(sh_ids), "pearson_r": float(r), "p_value": float(p)})
    pd.DataFrame(conc_rows).to_csv(REPORTS_DIR / "mumtaz_ec_eo_concordance.csv", index=False)

    eo_tr = []
    for sn, tn, sg, tg in [
        ("Mumtaz_EC (Eyes-Closed)", "Cavanagh_EO (Eyes-Open)", mum_ec, cav),
        ("Mumtaz_EO (Eyes-Open)", "Cavanagh_EO (Eyes-Open)", mum_eo, cav),
        ("Cavanagh_EO (Eyes-Open)", "Mumtaz_EC (Eyes-Closed)", cav, mum_ec),
        ("Cavanagh_EO (Eyes-Open)", "Mumtaz_EO (Eyes-Open)", cav, mum_eo),
        ("Mumtaz_EO (Eyes-Open)", "Mumtaz_EC (Eyes-Closed)", mum_eo, mum_ec),
        ("Mumtaz_EC (Eyes-Closed)", "Mumtaz_EO (Eyes-Open)", mum_ec, mum_eo),
    ]:
        rr = {r["model"]: r for r in eval_transfer_pair(sg, tg, sn, tn, "State_Matched")}
        eo_tr.append({
            "source_cohort": sn, "target_cohort": tn,
            "gcn_auc": rr["GCN_OMST"]["auc"], "gcn_sd": rr["GCN_OMST"]["split_hc_sd"], "gcn_bal_acc": rr["GCN_OMST"]["balanced_accuracy"],
            "lr_auc": rr["Tuned_LR"]["auc"], "lr_sd": rr["Tuned_LR"]["split_hc_sd"], "lr_bal_acc": rr["Tuned_LR"]["balanced_accuracy"],
        })
    pd.DataFrame(eo_tr).to_csv(REPORTS_DIR / "mumtaz_eo_state_matched_transfer.csv", index=False)

    # 2. 10x5 Leakage ablation
    leak_rows = []
    for cname, dtype, glist, gfn in [
        ("Study 1: Sleep Deprivation", "Paired_Crossover_Discordant", sleep, lambda g: g.get("sub_name", g["subject_id"])),
        ("Study 4: UCSD Parkinson", "Repeated_Sessions_Concordant", ucsd, lambda g: g.get("bids_id", g["subject_id"])),
        ("Study 6c: UNM Parkinson All", "Repeated_Sessions_Concordant", unm, lambda g: g["subject_id"]),
    ]:
        vg = [g for g in glist if int(g["y"]) in (0, 1)]
        y_all = np.array([int(g["y"]) for g in vg])
        grps = np.array([gfn(g) for g in vg])
        X_all, A_all = torch.stack([g["x"] for g in vg]), torch.stack([g["adj"] for g in vg])
        gl_z = (X_all - X_all.mean(0, keepdims=True)) / (X_all.std(0, keepdims=True) + 1e-6)
        per_rep = []
        for r_idx, seed in enumerate(REPEATS_10):
            set_seed(seed)
            for reg in ["Strict_Leak_Free", "Leaky_Pre_Standardization", "Leaky_Subject_Mixing"]:
                splits = get_grouped_splits(grps, y_all, n_splits=5, seed=seed) if reg != "Leaky_Subject_Mixing" else list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(vg, y_all))
                ov = 100.0 * sum(sum(1 for gid in grps[va] if gid in set(grps[tr])) for tr, va in splits) / len(vg)
                oof = {m: np.zeros(len(vg)) for m in ["Tuned_LogisticRegression", "RandomForest", "GCN_OMST"]}
                for fold, (tr, va) in enumerate(splits):
                    tx, vx = X_all[tr], X_all[va]
                    ty = torch.tensor(y_all[tr], dtype=torch.long)
                    tz, vz = (gl_z[tr], gl_z[va]) if reg == "Leaky_Pre_Standardization" else compute_fold_normative_z(tx, ty, vx)[:2]
                    Xtr, Xva = tz.view(len(tr), -1).numpy(), vz.view(len(va), -1).numpy()
                    icv = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + fold).split(Xtr, y_all[tr])) if reg == "Leaky_Subject_Mixing" else get_grouped_splits(grps[tr], y_all[tr], n_splits=3, seed=seed + fold)
                    oof["Tuned_LogisticRegression"][va] = GridSearchCV(LogisticRegression(max_iter=1000, random_state=seed), {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}, cv=icv, scoring="roc_auc").fit(Xtr, y_all[tr]).predict_proba(Xva)[:, 1]
                    oof["RandomForest"][va] = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed).fit(Xtr, y_all[tr]).predict_proba(Xva)[:, 1]
                    oof["GCN_OMST"][va], _ = train_eval_gcn(tz, A_all[tr], ty, vz, A_all[va], epochs=50, seed=seed + fold)
                for m, pr in oof.items():
                    per_rep.append({"cohort": cname, "design_type": dtype, "regime": reg, "model": m, "repeat": r_idx, "overlap": ov, "auc": float(roc_auc_score(y_all, pr))})
        df_pr = pd.DataFrame(per_rep)
        s_map = df_pr[df_pr["regime"] == "Strict_Leak_Free"].set_index(["model", "repeat"])["auc"].to_dict()
        df_pr["delta"] = df_pr.apply(lambda r: r["auc"] - s_map[(r["model"], r["repeat"])], axis=1)
        for (reg, m), grp in df_pr.groupby(["regime", "model"], sort=False):
            leak_rows.append({
                "cohort": cname, "design_type": dtype, "regime": reg, "model": m,
                "subject_overlap_pct": float(grp["overlap"].mean()),
                "mean_auc": float(grp["auc"].mean()), "sd_auc": float(grp["auc"].std(ddof=1)),
                "mean_delta_vs_strict": float(grp["delta"].mean()), "sd_delta_vs_strict": float(grp["delta"].std(ddof=1)),
                "delta_ci_lo": float(grp["delta"].mean() - 1.96 * grp["delta"].std(ddof=1) / np.sqrt(len(REPEATS_10))),
                "delta_ci_hi": float(grp["delta"].mean() + 1.96 * grp["delta"].std(ddof=1) / np.sqrt(len(REPEATS_10))),
            })
    pd.DataFrame(leak_rows).to_csv(REPORTS_DIR / "benchmark_leakage_ablation.csv", index=False)

    # 3. PLI SPD Audit (675 primary sessions) + Riemannian (incl. Spatial Covariance!)
    eig_rows, all_eigs = [], []
    for clbl, glist in [
        ("Study 1: Sleep Deprivation (ds004902)", sleep), ("Study 2: Outpatient MDD (EC)", mum_ec),
        ("Study 3: Cavanagh Depression (ds003478)", cav), ("Study 4: UCSD Parkinson (ds002778)", ucsd),
        ("Study 5: Clinical Dementia (ds004504)", dem), ("Study 6: UNM Parkinson EO (ds003490)", unm),
        ("Study 7: Iowa Parkinson (ds004584)", iowa),
    ]:
        mins = []
        for g in glist:
            pm = g["pli"].numpy().astype(np.float64)
            C = 0.5 * (pm + pm.T); np.fill_diagonal(C, 1.0)
            ev = np.linalg.eigvalsh(C)
            mins.append(float(ev.min())); all_eigs.append(float(ev.min()))
        arr = np.asarray(mins)
        eig_rows.append({
            "cohort": clbl, "n_sessions": len(arr), "min_eigenvalue": float(arr.min()), "median_min_eigenvalue": float(np.median(arr)),
            "n_indefinite_raw": int(np.sum(arr <= 0.0)), "pct_indefinite_raw": float(100.0 * np.mean(arr <= 0.0)),
            "n_indefinite_after_1e4_shift": int(np.sum(arr + 1e-4 <= 0.0)),
        })
    pd.DataFrame(eig_rows).to_csv(REPORTS_DIR / "pli_spd_eigenvalue_audit.csv", index=False)
    aa = np.asarray(all_eigs)
    pd.DataFrame([{"cohort": "Total Across All 7 Cohorts", "n_sessions": len(aa), "min_eigenvalue": float(aa.min()), "median_min_eigenvalue": float(np.median(aa)), "n_indefinite_raw": int(np.sum(aa <= 0.0)), "pct_indefinite_raw": float(100.0 * np.mean(aa <= 0.0)), "n_indefinite_after_1e4_shift": int(np.sum(aa + 1e-4 <= 0.0))}]).to_csv(REPORTS_DIR / "pli_spd_eigenvalue_summary.csv", index=False)

    def upper_vec(M):
        i, j = np.triu_indices_from(M, k=0)
        return M[i, j] * np.where(i == j, 1.0, np.sqrt(2.0))
    i_off, j_off = np.triu_indices(19, k=1)

    riem_rows = []
    for cname, ckey, glist in [
        ("Study 1: Sleep Deprivation (ds004902)", "sleep_ds004902", sleep),
        ("Study 2: Mumtaz MDD (Eyes Closed)", "mumtaz_ec", mum_ec),
        ("Study 3: Cavanagh University Depression", "cavanagh_ds003478", cav),
        ("Study 4: UCSD Parkinson All Sessions", "ucsd_ds002778", ucsd),
        ("Study 5a: Dementia AD vs Control", "dementia_ds004504", [g for g in dem if g["diagnosis"] in ("Alzheimer", "Control")]),
        ("Study 5b: Dementia FTD vs Control", "dementia_ds004504", [g for g in dem if g["diagnosis"] in ("FTD", "Control")]),
        ("Study 5c: Dementia All (AD+FTD) vs Control", "dementia_ds004504", dem),
        ("Study 6a: UNM PD-OFF vs Control", "unm_ds003490", unm_off),
        ("Study 6b: UNM PD-ON vs Control", "unm_ds003490", [g for g in unm if g["med_state"] in ("ON", "CTL")]),
        ("Study 7: Iowa Parkinson (ds004584)", "iowa_ds004584", iowa),
    ]:
        vg = [g for g in glist if int(g["y"]) in (0, 1)]
        y_all = np.array([int(g["y"]) for g in vg])
        grps = np.array([g.get("bids_id", g.get("sub_name", g["subject_id"])) for g in vg])
        cov_dict = spatial_covs[ckey]
        Xe, Xa, Xh, Xs = [], [], [], []
        for g in vg:
            pm = g["pli"].numpy().astype(np.float64); C = 0.5 * (pm + pm.T); np.fill_diagonal(C, 1.0)
            Xe.append(C[i_off, j_off])
            vals, vecs = np.linalg.eigh(C)
            sh = max(0.0, 0.01 - float(vals.min()))
            Xa.append(upper_vec((vecs * np.log(vals + sh)) @ vecs.T))
            Xh.append(upper_vec((vecs * np.log(np.maximum(vals, 1e-4))) @ vecs.T))

            key_cov = (f"{g['subject_id']}_{g['session']}" if g.get("group") == "PD" else g["subject_id"]) if ckey == "unm_ds003490" else g["subject_id"]
            S_mat = cov_dict[key_cov]
            svs, svecs = np.linalg.eigh(S_mat)
            Xs.append(upper_vec((svecs * np.log(np.maximum(svs, 1e-12))) @ svecs.T))

        for rname, Xrep in [
            ("Euclidean_UpperTri_171D", np.asarray(Xe)),
            ("Riemannian_Gershgorin_SPD_190D", np.asarray(Xa)),
            ("Riemannian_Higham_SPD_190D", np.asarray(Xh)),
            ("Riemannian_Spatial_Cov_190D", np.asarray(Xs)),
        ]:
            aucs_rep = {k: [] for k in ("LR", "Ridge", "RF")}
            for s in REPEATS_10:
                splits = get_grouped_splits(grps, y_all, n_splits=5, seed=s)
                oof_p = {k: np.zeros(len(vg)) for k in ("LR", "Ridge", "RF")}
                for fold, (tr, va) in enumerate(splits):
                    mu, sd = Xrep[tr].mean(0, keepdims=True), Xrep[tr].std(0, keepdims=True) + 1e-6
                    Xtr, Xva, ytr = (Xrep[tr] - mu) / sd, (Xrep[va] - mu) / sd, y_all[tr]
                    icv = get_grouped_splits(grps[tr], ytr, n_splits=3, seed=s + fold)
                    oof_p["LR"][va] = GridSearchCV(LogisticRegression(max_iter=1000, random_state=s), {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}, cv=icv, scoring="roc_auc").fit(Xtr, ytr).predict_proba(Xva)[:, 1]
                    oof_p["Ridge"][va] = 1.0 / (1.0 + np.exp(-GridSearchCV(RidgeClassifier(random_state=s), {"alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}, cv=icv, scoring="roc_auc").fit(Xtr, ytr).decision_function(Xva)))
                    oof_p["RF"][va] = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=s).fit(Xtr, ytr).predict_proba(Xva)[:, 1]
                for ck in ("LR", "Ridge", "RF"):
                    aucs_rep[ck].append(float(roc_auc_score(y_all, oof_p[ck])))
            for ck in ("LR", "Ridge", "RF"):
                arr = np.asarray(aucs_rep[ck])
                riem_rows.append({"cohort": cname, "representation": rname, "classifier": ck, "n_samples": len(vg), "auc": float(arr.mean()), "sd_auc": float(arr.std(ddof=1)), "split0_auc": float(arr[0])})
    pd.DataFrame(riem_rows).to_csv(REPORTS_DIR / "benchmark_riemannian_tangent_space.csv", index=False)

    # 4. Gain calibration & stress test (including mean_feat_drift, lr_mean_prob_drift, rf_mean_prob_drift from 45d!)
    def build_gain_feats(recs, ftype, gain_k):
        feats = []
        for r, p_rec in recs:
            psd = p_rec["mean_psd"] * (gain_k ** 2)
            freqs = p_rec["freqs"]
            f30 = (freqs >= 2.0) & (freqs <= 30.0)
            fv = freqs[f30]
            if ftype == "specparam_periodic_residual":
                x = r["x"].numpy().copy()
            elif ftype == "masked_ols_residual":
                x = r["x_masked_ols_1to40"].numpy().copy()
            elif ftype == "raw_log_band_power":
                th_m, dl_m, al_m = (fv >= 4.0) & (fv < 8.0), (fv >= 2.0) & (fv < 4.0), (fv >= 8.0) & (fv <= 13.0)
                lp = np.log10(np.maximum(psd[:, f30], 1e-25))
                x = np.zeros((19, 4), dtype=np.float32)
                x[:, 0], x[:, 1], x[:, 2], x[:, 3] = lp[:, th_m].mean(1), lp[:, dl_m].mean(1), lp[:, al_m].mean(1), r["x"].numpy()[:, 3]
            else:
                th_m, dl_m, al_m = (fv >= 4.0) & (fv < 8.0), (fv >= 2.0) & (fv < 4.0), (fv >= 8.0) & (fv <= 13.0)
                tot = np.maximum(psd[:, f30].sum(axis=1, keepdims=True), 1e-25)
                x = np.zeros((19, 4), dtype=np.float32)
                x[:, 0] = np.log10(np.maximum(psd[:, f30][:, th_m].mean(1, keepdims=True) / tot, 1e-12)).ravel()
                x[:, 1] = np.log10(np.maximum(psd[:, f30][:, dl_m].mean(1, keepdims=True) / tot, 1e-12)).ravel()
                x[:, 2] = np.log10(np.maximum(psd[:, f30][:, al_m].mean(1, keepdims=True) / tot, 1e-12)).ravel()
                x[:, 3] = r["x"].numpy()[:, 3]
            feats.append(x.flatten())
        return np.asarray(feats, dtype=np.float64)

    pd_sp_psds = {
        "UCSD": list(zip(ucsd, psds["ucsd_ds002778"])),
        "UNM_OFF": [(r, p) for r, p in zip(unm, psds["unm_ds003490"]) if r["med_state"] in ("OFF", "CTL")],
        "Iowa": list(zip(iowa, psds["iowa_ds004584"])),
    }
    gain_rows = []
    for sn in pd_sp_psds:
        for tn in pd_sp_psds:
            if sn == tn: continue
            s_pairs, t_pairs = pd_sp_psds[sn], pd_sp_psds[tn]
            ys, yt = np.array([int(r["y"]) for r, _ in s_pairs]), np.array([int(r["y"]) for r, _ in t_pairs])
            for ftype in ["specparam_periodic_residual", "masked_ols_residual", "relative_band_power", "raw_log_band_power"]:
                Xs_base, Xt_base = build_gain_feats(s_pairs, ftype, 1.0), build_gain_feats(t_pairs, ftype, 1.0)
                lr_u = LogisticRegression(C=0.1, max_iter=1000, random_state=42).fit(Xs_base, ys)
                rf_u = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(Xs_base, ys)
                p_lr_ref, p_rf_ref = lr_u.predict_proba(Xt_base)[:, 1], rf_u.predict_proba(Xt_base)[:, 1]
                for k_lbl, k_fn in [("0.5", lambda n: np.full(n, 0.5)), ("1.0", lambda n: np.ones(n)), ("2.0", lambda n: np.full(n, 2.0)), ("5.0", lambda n: np.full(n, 5.0)), ("Heterogeneous_0.5x_5.0x", lambda n: np.array([0.5 if i % 2 == 0 else 5.0 for i in range(n)]))]:
                    ks = k_fn(len(t_pairs))
                    Xt_k = np.vstack([build_gain_feats([t_pairs[i]], ftype, ks[i]) for i in range(len(t_pairs))])
                    pr_lr, pr_rf = lr_u.predict_proba(Xt_k)[:, 1], rf_u.predict_proba(Xt_k)[:, 1]
                    gain_rows.append({
                        "experiment": "Amplifier_Gain_Ablation", "source": sn, "target": tn, "calibration": "Uncalibrated_Raw",
                        "feature_type": ftype, "gain_k": k_lbl,
                        "lr_auc": float(roc_auc_score(yt, pr_lr)), "lr_bacc": float(balanced_accuracy_score(yt, (pr_lr >= 0.5).astype(int))), "lr_brier": float(brier_score_loss(yt, pr_lr)),
                        "rf_auc": float(roc_auc_score(yt, pr_rf)),
                        "mean_feat_drift": float(np.mean(np.abs(Xt_k - Xt_base))), "max_feat_drift": float(np.max(np.abs(Xt_k - Xt_base))),
                        "lr_mean_prob_drift": float(np.mean(np.abs(pr_lr - p_lr_ref))), "lr_max_prob_drift": float(np.max(np.abs(pr_lr - p_lr_ref))),
                        "rf_mean_prob_drift": float(np.mean(np.abs(pr_rf - p_rf_ref))), "rf_max_prob_drift": float(np.max(np.abs(pr_rf - p_rf_ref))),
                    })
                for cal_lbl in ["Uncalibrated_Raw", "Global_Cohort_ZScore", "Target_HC_Normative"]:
                    if cal_lbl == "Uncalibrated_Raw":
                        Xs_c, Xt_c = Xs_base, Xt_base
                    elif cal_lbl == "Global_Cohort_ZScore":
                        Xs_c = (Xs_base - Xs_base.mean(0, keepdims=True)) / (Xs_base.std(0, keepdims=True) + 1e-6)
                        Xt_c = (Xt_base - Xt_base.mean(0, keepdims=True)) / (Xt_base.std(0, keepdims=True) + 1e-6)
                    else:
                        s_hc, t_hc = Xs_base[ys == 0], Xt_base[yt == 0]
                        Xs_c = (Xs_base - s_hc.mean(0, keepdims=True)) / (s_hc.std(0, keepdims=True) + 1e-6)
                        Xt_c = (Xt_base - t_hc.mean(0, keepdims=True)) / (t_hc.std(0, keepdims=True) + 1e-6)
                    lr_c = LogisticRegression(C=0.1, max_iter=1000, random_state=42).fit(Xs_c, ys)
                    rf_c = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(Xs_c, ys)
                    pr_lc, pr_rc = lr_c.predict_proba(Xt_c)[:, 1], rf_c.predict_proba(Xt_c)[:, 1]
                    gain_rows.append({
                        "experiment": "Calibration_Strategy", "source": sn, "target": tn, "calibration": cal_lbl,
                        "feature_type": ftype, "gain_k": "1.0",
                        "lr_auc": float(roc_auc_score(yt, pr_lc)), "lr_bacc": float(balanced_accuracy_score(yt, (pr_lc >= 0.5).astype(int))), "lr_brier": float(brier_score_loss(yt, pr_lc)),
                        "rf_auc": float(roc_auc_score(yt, pr_rc)),
                        "mean_feat_drift": 0.0, "max_feat_drift": 0.0,
                        "lr_mean_prob_drift": 0.0, "lr_max_prob_drift": 0.0, "rf_mean_prob_drift": 0.0, "rf_max_prob_drift": 0.0,
                    })
    pd.DataFrame(gain_rows).to_csv(REPORTS_DIR / "benchmark_gain_and_calibration_ablation.csv", index=False)
    run_module_e_tail(sp)


def _ridge_perm_batch(X: np.ndarray, y: np.ndarray, alpha: float = 10.0, n_perms: int = 1000):
    """Exact vectorized 5-fold KFold Ridge permutation null (0.02s for 1,000 permutations)."""
    n = len(y)
    Y_perm = np.column_stack([np.random.RandomState(20000 + i).permutation(y) for i in range(n_perms)])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    OOF = np.zeros_like(Y_perm)
    for tr, va in kf.split(X, y):
        mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-6
        Xtr, Xva = (X[tr] - mu) / sd, (X[va] - mu) / sd
        y_mean = Y_perm[tr].mean(axis=0, keepdims=True)
        Yc = Y_perm[tr] - y_mean
        W = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]), Xtr.T @ Yc)
        OOF[va] = Xva @ W + y_mean
    yp_c = Y_perm - Y_perm.mean(axis=0, keepdims=True)
    oo_c = OOF - OOF.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(yp_c ** 2, axis=0) * np.sum(oo_c ** 2, axis=0))
    rs = np.where(denom > 1e-12, np.sum(yp_c * oo_c, axis=0) / np.maximum(denom, 1e-12), 0.0)
    return rs.astype(float).tolist()


def run_module_e_tail(sp):
    dem, iowa = sp["dementia_ds004504"], sp["iowa_ds004584"]
    dem_pts = [g for g in dem if g.get("diagnosis") in ("Alzheimer", "FTD")]
    iowa_pts = [g for g in iowa if g.get("group") == "PD"]
    reg_tasks = [
        ("Dementia_MMSE_All_N88", "Study 5: Dementia All (ds004504)", "MMSE (4.0–30.0)", dem, "mmse"),
        ("Dementia_MMSE_Patients_Only_N59", "Study 5: Dementia Patients Only", "MMSE (4.0–29.0)", dem_pts, "mmse"),
        ("Iowa_PD_MoCA_All_N149", "Study 7: Iowa PD All (ds004584)", "MoCA (9.0–30.0)", iowa, "moca"),
        ("Iowa_PD_MoCA_Patients_Only_N100", "Study 7: Iowa PD Patients Only", "MoCA (9.0–30.0)", iowa_pts, "moca"),
        ("Iowa_PD_UPDRS_Patients_N100", "Study 7: Iowa PD Patients Only", "UPDRS-III (1.0–28.0)", iowa_pts, "updrs"),
    ]
    reg_rows = []
    perm_jobs = []
    perm_agg = {}
    for jkey, clbl, sclbl, glist, skey in reg_tasks:
        valid = [g for g in glist if pd.notnull(g.get(skey))]
        y = np.array([float(g[skey]) for g in valid])
        X = np.stack([g["x"].numpy().flatten() for g in valid])
        for mname, fn_est in [
            ("GBDT_Regressor", lambda s: HistGradientBoostingRegressor(max_iter=100, max_depth=4, random_state=s)),
            ("Random_Forest_Regressor", lambda s: RandomForestRegressor(n_estimators=100, max_depth=5, random_state=s, n_jobs=1)),
            ("SVR_RBF", lambda s: SVR(C=1.0, epsilon=0.1)),
            ("Ridge_Regression", lambda s: Ridge(alpha=10.0)),
        ]:
            rs, ps, r2s, maes = [], [], [], []
            r0, p0, r20, mae0 = 0, 1, 0, 0
            for r_i, s in enumerate(REPEATS_10):
                kf = KFold(n_splits=5, shuffle=True, random_state=s)
                oof = np.zeros(len(valid))
                for tr, va in kf.split(X, y):
                    mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-6
                    est = fn_est(s).fit((X[tr] - mu) / sd, y[tr])
                    oof[va] = est.predict((X[va] - mu) / sd)
                rv, pv = pearsonr(y, oof); r2v = r2_score(y, oof); mv = mean_absolute_error(y, oof)
                rs.append(float(rv)); ps.append(float(pv)); r2s.append(float(r2v)); maes.append(float(mv))
                if r_i == 0:
                    r0, p0, r20, mae0 = rv, pv, r2v, mv
            reg_rows.append({
                "cohort": clbl, "cohort_key": jkey, "scale": sclbl, "scale_key": skey, "n_samples": len(valid), "model": mname,
                "pearson_r_mean": float(np.mean(rs)), "pearson_r_sd": float(np.std(rs, ddof=1)), "median_p": float(np.median(ps)),
                "r2_mean": float(np.mean(r2s)), "mae_mean": float(np.mean(maes)),
                "split0_r": float(r0), "split0_p": float(p0), "split0_r2": float(r20), "split0_mae": float(mae0),
            })
            if mname == "Ridge_Regression":
                perm_agg[(jkey, mname)] = _ridge_perm_batch(X, y, alpha=10.0, n_perms=1000)
            else:
                for st in range(0, 1000, 100):
                    perm_jobs.append((jkey, mname, X, y, list(range(20000 + st, 20000 + min(st + 100, 1000)))))

    print(f"  Running spawn ProcessPool 1,000-permutation tests across {len(perm_jobs)} chunks...")
    t_p = time.time()
    done_c = 0
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4), mp_context=mp.get_context("spawn")) as ex:
        for jk, mn, r_null in ex.map(_reg_perm_worker, perm_jobs):
            perm_agg.setdefault((jk, mn), []).extend(r_null)
            done_c += 1
            if done_c % 30 == 0 or done_c == len(perm_jobs):
                print(f"    Regression perms: {done_c}/{len(perm_jobs)} chunks ({time.time() - t_p:.1f}s)")
    for r in reg_rows:
        arr = np.asarray(perm_agg[(r["cohort_key"], r["model"])], dtype=float)
        r["perm_p"] = float((np.sum(arr >= r["split0_r"]) + 1.0) / (len(arr) + 1.0))
        r["perm_p_two_sided"] = float((np.sum(np.abs(arr) >= abs(r["split0_r"])) + 1.0) / (len(arr) + 1.0))
        r["perm_null_mean_r"] = float(arr.mean())
        r["perm_null_sd_r"] = float(arr.std(ddof=1))
    pd.DataFrame(reg_rows).to_csv(REPORTS_DIR / "multicohort_clinical_regression_benchmarks.csv", index=False)

    old_737 = RELEASE_DIR / "derived_session_features_737.csv"
    if old_737.exists():
        old_737.unlink()
    feat_rows = []
    for ckey, recs in sp.items():
        for g in recs:
            arr = g["x"].numpy()
            feat_rows.append({
                "cohort_key": ckey, "subject_id": g["subject_id"],
                "group": g.get("group", g.get("diagnosis", g.get("condition", ""))),
                "med_state": g.get("med_state", ""), "y": int(g["y"]), "age": g.get("age", np.nan),
                "n_clean_epochs": int(g.get("n_clean_epochs", -1)),
                "omst_trees": int(g["omst_trees"]), "omst_edges": int(g["omst_edge_count"]),
                "gce": float(g["gce"]),
                "frontal_theta_2to30": float(arr[FOCAL_5, 0].mean()),
                "global_lowfreq_2to4": float(arr[:, 1].mean()),
                "posterior_alpha": float(arr[POSTERIOR_5, 2].mean()),
                "global_chi_2to30": float(arr[:, 3].mean()),
                "iapf_hz": g.get("iapf_posterior", np.nan),
                "mean_r2_2to30": float(np.mean(g["sp_r2"])),
                "mean_mae_2to30": float(np.mean(g["sp_mae"])),
            })
    pd.DataFrame(feat_rows).to_csv(RELEASE_DIR / "derived_session_features_733.csv", index=False)
    try:
        req_txt = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (RELEASE_DIR / "requirements_lock.txt").write_text(req_txt, encoding="utf-8")
        (REPORTS_DIR / "requirements_lock.txt").write_text(req_txt, encoding="utf-8")
    except Exception as exc:
        print(f"Warning pip freeze: {exc}")
    print("✓ Step 8 complete: Module E + Spatial Cov Riemannian + 1,000-Perm Regression + derived_session_features_733.csv.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail-only", action="store_true", help="Run only regression permutations & release artifact export using cached specparam_canonical_7cohorts.pt")
    args = parser.parse_args()
    t0 = time.time()
    if args.tail_only:
        sp = torch.load(DATA_DIR / "specparam_canonical_7cohorts.pt", weights_only=False)
        run_module_e_tail(sp)
    else:
        _, raw_unfilt_psds = run_study2_unfiltered_audit()
        sp, sp_140, psds, unm_ec_map = backup_and_deduplicate()
        run_module_a_and_aux(sp, psds, unm_ec_map)
        run_2d_grid_and_fooof_check(psds, raw_unfilt_psds)
        run_module_b_calibrated_simulation()
        run_module_c_v5(sp)
        run_module_d_permutations(sp)
        spatial_covs = extract_or_load_spatial_covariances(sp)
        run_module_e_rest(sp, psds, spatial_covs)
    print(f"\n🎉 COMPLETE v5 FORENSIC & BENCHMARKING PIPELINE FINISHED IN {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
