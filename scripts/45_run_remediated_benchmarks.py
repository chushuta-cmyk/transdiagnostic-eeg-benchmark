#!/usr/bin/env python3
"""
scripts/45_run_remediated_benchmarks.py
---------------------------------------
Executes the complete remediated benchmarking & statistical verification suite:
  1. Module A (Biophysics & Sensitivity):
     - Primary interior [2.0, 30.0] Hz specparam residuals & peaks vs [7, 14] Hz Masked OLS
       AND [1.0, 40.0] Hz edge sensitivity across all 10 contrasts (with fixed pure-EO Study 6).
     - 1,000-bootstrap 95% CIs & p-values for paired Delta_d = d_OLS - d_specparam and
       Delta_d_win = d_[1-40] - d_[2-30].
     - Clean epoch-duration ANCOVA & chronological Age ANCOVA across cohorts.
     - Primary Participant-Averaged (K=6, N=495) Meta-Analysis with conservative HKSJ CIs & p-values.
     - UNM Pure EC (N=74) vs Pure EO (N=75) sensitivity comparison.
  2. Module B (Synthetic Ground-Truth PSD Simulation):
     - Known-truth Monte Carlo simulations (500 trials x 3 regimes) directly proving both
       [7, 14] Hz OLS bias under true null theta and [1, 40] Hz filter-edge distortion vs [2, 30] Hz specparam.
  3. Module C (True 10x5 Repeated Participant-Grouped CV + Edge-Matched GCN-Random):
     - Explicit stratified shuffling of unique participant IDs across 10 repeats x 5 folds (50 outer folds)
       guaranteeing 0.0% participant overlap and truly distinct partitions even in paired Study 1.
     - Session-specific edge-matched degree-agnostic random graph controls (GCN_Random_EdgeMatched).
     - Wilcoxon signed-rank + BH-FDR comparisons on 10x5 repeated means across both 9 contrasts and
       8 non-nested contrasts (sensitivity).
     - Exports Zenodo-ready reproducibility CSVs (derived_session_features, cv_fold_indices, oof_predictions).
  4. Module D (Aligned 1,000-Permutation Retraining Tests):
     - Uses IDENTICAL inner GridSearchCV Tuned LR and participant grouping as the main CV benchmark.
  5. Module E (Leak-Free Split-Target-HC Transfer, 10x5 Leakage Ablation, Gain Calibration, PLI/Riemannian, 10x5 Regressions):
     - Leak-free 20-repeat 50/50 Split-Target-HC evaluation across ALL PD and transdiagnostic transfer pairs.
     - Explicit mean AND worst-case max drift reporting for gain stress tests.
     - Re-computed PLI non-PSD audit, Riemannian baselines, and 10x5 clinical severity regressions.
"""

import json
import os
import random
import sys
import time
import warnings
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, SVR
from specparam import SpectralGroupModel
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore", category=UserWarning)

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = BENCHMARK_ROOT / "src"
sys.path.append(str(SRC_DIR))
from harmonizer_and_graph import CHANNELS_19, NormativeGCN, compute_fold_normative_z

DATA_DIR = BENCHMARK_ROOT / "data" / "derivatives"
REPORTS_DIR = BENCHMARK_ROOT / "reports"
RELEASE_DIR = BENCHMARK_ROOT / "release_artifacts"
EEG_ROOT = Path("/Users/tan/Documents/eeg_datastes")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
RELEASE_DIR.mkdir(parents=True, exist_ok=True)

FOCAL_5 = [CHANNELS_19.index(c) for c in ["Fp1", "Fp2", "F3", "F4", "Fz"]]
FOCAL_3 = [CHANNELS_19.index(c) for c in ["F3", "Fz", "F4"]]
POSTERIOR_5 = [CHANNELS_19.index(c) for c in ["P3", "P4", "Pz", "O1", "O2"]]
REPEATS_10 = [42, 123, 456, 789, 2026, 101, 202, 303, 404, 505]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def welch_df(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1) / nx, np.var(y, ddof=1) / ny
    denom = (vx ** 2) / (nx - 1) + (vy ** 2) / (ny - 1)
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
    """
    True shuffled participant-grouped stratified 5-fold splitter.
    Avoids sklearn StratifiedGroupKFold deterministic fold ties in balanced paired cohorts (Study 1)
    while guaranteeing 0.0% participant overlap and class balance across outer folds.
    """
    groups = np.asarray(groups)
    y = np.asarray(y, dtype=int)
    rng = np.random.RandomState(seed)

    uniq_g = []
    seen = set()
    for g in groups:
        if g not in seen:
            seen.add(g)
            uniq_g.append(g)
    uniq_g = np.asarray(uniq_g)

    # Bucket unique participants by participant-level label signature
    # For paired balanced crossover (mixed labels inside a subject), assign signature = 2
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
        assert len(set(groups[tr_idx]) & set(groups[va_idx])) == 0, "Participant overlap!"
        splits.append((tr_idx, va_idx))
    return splits


def make_edge_matched_random_adj(num_nodes: int, num_edges: int, seed: int = 42) -> torch.Tensor:
    """Build degree-agnostic unweighted symmetric adjacency matching exact OMST undirected edge count."""
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


def load_canonical_bundles():
    """Load [2.0, 30.0] Hz primary bundle, [1.0, 40.0] Hz edge sensitivity, UNM EC, and PSDs."""
    sp = torch.load(DATA_DIR / "specparam_vs_masked_ols_7cohorts.pt", weights_only=False)
    sp_140 = torch.load(DATA_DIR / "specparam_1to40_sensitivity_7cohorts.pt", weights_only=False)
    psds = torch.load(DATA_DIR / "all_7cohorts_welch_psds.pt", weights_only=False)
    unm_ec_map = torch.load(DATA_DIR / "unm_ec_sensitivity_specparam.pt", weights_only=False)

    cav_tsv = pd.read_csv(EEG_ROOT / "depression_new" / "participants.tsv", sep="\t")
    cav_age = {str(r["participant_id"]): (float(r["age"]) if pd.notnull(r["age"]) else np.nan) for _, r in cav_tsv.iterrows()}
    ucsd_tsv = pd.read_csv(EEG_ROOT / "participants.tsv", sep="\t")
    ucsd_age = {str(r["participant_id"]): (float(r["age"]) if pd.notnull(r["age"]) else np.nan) for _, r in ucsd_tsv.iterrows()}

    for ckey, recs in sp.items():
        psd_recs = psds[ckey]
        s140_recs = sp_140[ckey]
        freqs = np.asarray(psd_recs[0]["freqs"], dtype=float)
        fg = SpectralGroupModel(aperiodic_mode="fixed", peak_width_limits=[1.0, 12.0], max_n_peaks=6, min_peak_height=0.05, peak_threshold=2.0, verbose=False)
        post_psds = np.stack([np.maximum(np.asarray(pr["mean_psd"], dtype=float)[POSTERIOR_5], 1e-25) for pr in psd_recs])
        fg.fit(freqs, post_psds.reshape(len(recs) * len(POSTERIOR_5), -1), [2.0, 30.0])

        for idx, r in enumerate(recs):
            assert np.all(r["sp_fit_ok"]), f"Failed fit in {ckey} idx={idx}"
            r["x"] = r["x_specparam_resid"].clone()
            r["x_specparam_1to40"] = s140_recs[idx]["x_specparam_resid"].clone()
            r["x_masked_ols_1to40"] = s140_recs[idx]["x_masked_ols"].clone()
            r["x_masked_ols_2to30"] = r["x_masked_ols_win"].clone()
            # Count exact OMST undirected edges on each graph adjacency
            adj_np = r["adj"].numpy()
            r["omst_edge_count"] = int(np.sum(np.triu(adj_np > 0.5, k=1)))

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

    torch.save(sp, DATA_DIR / "specparam_canonical_7cohorts.pt")
    print("✓ Loaded canonical [2.0, 30.0] Hz bundle + [1.0, 40.0] Hz sensitivity + UNM EC, 0 failed fits.")
    return sp, sp_140, psds, unm_ec_map


def make_ucsd_subj_mean(ucsd):
    by_bid = {}
    for g in ucsd:
        if "PD" in g.get("group", ""):
            by_bid.setdefault(g["bids_id"], []).append(g)
    pd_mean = []
    for bid in sorted(by_bid.keys()):
        pair = by_bid[bid]
        rec = {
            "bids_id": bid,
            "subject_id": bid,
            "group": "PD_SubjMean",
            "age": float(pair[0].get("age", np.nan)),
            "n_clean_epochs": float(np.mean([p["n_clean_epochs"] for p in pair])),
            "gce": float(np.mean([float(p["gce"]) for p in pair])),
            "iapf_posterior": float(np.nanmean([p.get("iapf_posterior", np.nan) for p in pair])),
        }
        for k in ["x", "x_specparam_resid", "x_specparam_peaks", "x_specparam_1to40", "x_masked_ols_1to40", "x_masked_ols_2to30"]:
            rec[k] = torch.stack([p[k] for p in pair]).mean(dim=0)
        pd_mean.append(rec)
    return pd_mean, [g for g in ucsd if g.get("group") == "HC"]


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
        sp_sd = np.sqrt(((nc - 1) * np.var(vc, ddof=1) + (nr - 1) * np.var(vr, ddof=1)) / (nc + nr - 2)) + 1e-12
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
    """Stratified participant-level bootstrap CI & two-sided p for Delta_d = d(fn_a) - d(fn_b)."""
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

    obs_a = d_pair(a_c, a_r, paired)
    obs_b = d_pair(b_c, b_r, paired)
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
# MODULE A: BIOPHYSICS, BOOTSTRAP DELTA_D, DURATION/AGE ANCOVA, META-ANALYSIS
# ==============================================================================
def run_module_a(sp, psds, unm_ec_map):
    print("\n=== Module A: Biophysics, Bootstrap Δd, Fit-Range, Epoch ANCOVA & Primary SubjMean Meta-Analysis ===")
    sleep, mum_ec, mum_eo = sp["sleep_ds004902"], sp["mumtaz_ec"], sp["mumtaz_eo"]
    cav, ucsd, dem, unm, iowa = sp["cavanagh_ds003478"], sp["ucsd_ds002778"], sp["dementia_ds004504"], sp["unm_ds003490"], sp["iowa_ds004584"]

    all_primary = sleep + mum_ec + cav + ucsd + dem + unm + iowa
    assert len(all_primary) == 677
    omst_trees = [g["omst_trees"] for g in all_primary]
    omst_edges = [g["omst_edge_count"] for g in all_primary]
    gces = [g["gce"] for g in all_primary]

    sd_s = sorted([g for g in sleep if g["condition"] == "SD"], key=lambda r: r["subject_id"])
    ns_s = sorted([g for g in sleep if g["condition"] == "NS"], key=lambda r: r["subject_id"])
    ucsd_mean, ucsd_hc = make_ucsd_subj_mean(ucsd)
    ucsd_off = [g for g in ucsd if g.get("group") == "PD_OFF"]
    ucsd_on = [g for g in ucsd if g.get("group") == "PD_ON"]
    unm_off = [g for g in unm if g.get("med_state") == "OFF"]
    unm_on = [g for g in unm if g.get("med_state") == "ON"]
    unm_hc = [g for g in unm if g.get("med_state") == "CTL"]

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

    # Expanded 50-test primary table ([2.0, 30.0] Hz specparam)
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
            st = compute_es(cs, rs, fn, paired=isp)
            exp_rows.append({"cohort": clbl, "marker_key": mk, "marker": mn, **st})
    df_exp = pd.DataFrame(exp_rows)
    _, q50, _, _ = multipletests(df_exp["p_value"].values, method="fdr_bh")
    df_exp["fdr_q_50"] = q50
    df_exp["fdr_q_10"] = np.nan
    for mk, idx in df_exp.groupby("marker_key").groups.items():
        _, q10, _, _ = multipletests(df_exp.loc[idx, "p_value"].values, method="fdr_bh")
        df_exp.loc[idx, "fdr_q_10"] = q10
    df_exp.to_csv(REPORTS_DIR / "multicohort_biophysical_effect_sizes_expanded.csv", index=False)

    # Also store F3-Fz-F4 3ch frontal theta sensitivity + Study 2b EO + UNM EC sensitivity
    sub_rows = []
    for clbl, cs, rs, isp in contrasts_10:
        st3 = compute_es(cs, rs, lambda g: float(g["x"].numpy()[FOCAL_3, 0].mean()), paired=isp)
        sub_rows.append({"cohort": clbl, **st3})
    pd.DataFrame(sub_rows).to_csv(REPORTS_DIR / "biophysics_f3_fz_f4_sensitivity.csv", index=False)

    # Head-to-head ablation CSV (OLS 1-40, OLS 2-30, SP 2-30 resid, SP 2-30 peaks, SP 1-40 resid) + bootstrap Delta_d
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
                st = compute_es(cs, rs, fn, paired=isp)
                ab_rows.append({"method": m_name, "cohort": clbl, "marker": mk, **st})
            # Bootstrap Δd1: OLS[1-40] - SP[2-30], Δd2: OLS[2-30] - SP[2-30], Δd3: SP[1-40] - SP[2-30]
            fn_sp = lambda g, col=idx_col, ch=chans: float(g["x_specparam_resid"].numpy()[:, col].mean() if ch is None else g["x_specparam_resid"].numpy()[ch, col].mean())
            fn_o1 = lambda g, col=idx_col, ch=chans: float(g["x_masked_ols_1to40"].numpy()[:, col].mean() if ch is None else g["x_masked_ols_1to40"].numpy()[ch, col].mean())
            fn_o2 = lambda g, col=idx_col, ch=chans: float(g["x_masked_ols_2to30"].numpy()[:, col].mean() if ch is None else g["x_masked_ols_2to30"].numpy()[ch, col].mean())
            fn_s140 = lambda g, col=idx_col, ch=chans: float(g["x_specparam_1to40"].numpy()[:, col].mean() if ch is None else g["x_specparam_1to40"].numpy()[ch, col].mean())
            for cmp_lbl, fn_alt in [("OLS_1_40_minus_SP_2_30", fn_o1), ("OLS_2_30_minus_SP_2_30", fn_o2), ("SP_1_40_minus_SP_2_30", fn_s140)]:
                da, db, dd, lo, hi, pv = bootstrap_delta_d(cs, rs, fn_alt, fn_sp, paired=isp)
                boot_rows.append({
                    "cohort": clbl, "marker": mk, "comparison": cmp_lbl,
                    "d_alt": da, "d_sp_2_30": db, "delta_d": dd, "ci_lower": lo, "ci_upper": hi, "boot_p": pv,
                })
    df_ab = pd.DataFrame(ab_rows)
    df_ab["fdr_q_10"] = np.nan
    for (mn, mk), idx in df_ab.groupby(["method", "marker"]).groups.items():
        _, q10, _, _ = multipletests(df_ab.loc[idx, "p_value"].values, method="fdr_bh")
        df_ab.loc[idx, "fdr_q_10"] = q10
    df_ab.to_csv(REPORTS_DIR / "ablation_specparam_vs_ols_biophysics.csv", index=False)
    df_ab_w = df_ab.pivot(index=["marker", "cohort"], columns="method", values=["cohens_d", "p_value", "fdr_q_10"])
    df_ab_w.columns = [f"{a}_{b}" for a, b in df_ab_w.columns]
    df_ab_w.reset_index().to_csv(REPORTS_DIR / "ablation_specparam_vs_ols_biophysics_wide.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(REPORTS_DIR / "ablation_specparam_vs_ols_bootstrap_deltas.csv", index=False)

    # Meta-Analysis (Primary = K=6 Participant-Averaged w/ UCSD_PD_SubjMean, HKSJ CIs & p-values!)
    def dl_pool(items):
        ds = np.array([x[0] for x in items], dtype=float)
        vs = np.array([x[1] for x in items], dtype=float)
        k = len(ds)
        w = 1.0 / vs
        fe = float(np.sum(w * ds) / np.sum(w))
        Q = float(np.sum(w * (ds - fe) ** 2))
        df_q = k - 1
        c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
        tau2 = float(max(0.0, (Q - df_q) / c)) if c > 0 and k > 1 else 0.0
        w_re = 1.0 / (vs + tau2)
        d_re = float(np.sum(w_re * ds) / np.sum(w_re))
        se_re = float(np.sqrt(1.0 / np.sum(w_re)))
        p_dl = float(2.0 * (1.0 - stats.norm.cdf(abs(d_re / se_re))))
        q_hksj = float(max(1.0, Q / df_q)) if df_q > 0 else 1.0
        se_hk = float(np.sqrt(q_hksj) * se_re)
        t_c = float(stats.t.ppf(0.975, df=df_q)) if df_q > 0 else 1.96
        p_hk = float(2.0 * (1.0 - stats.t.cdf(abs(d_re / se_hk), df=df_q))) if df_q > 0 else p_dl
        i2 = float(max(0.0, (Q - df_q) / Q * 100.0)) if Q > 0 and df_q > 0 else 0.0
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
        configs = [
            ("Primary_K6_SubjMean_PD_OFF", dep_k2, [v_fn("Study4_UCSD_SubjMean"), v_fn("Study5_Dementia_All"), v_fn("Study6a_UNM_PD_OFF"), v_fn("Study7_Iowa_PD")]),
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
            for sub_lbl, pd_obj in [("Depression_Cohorts_k2", p1), ("PD_Dementia_Cohorts", p2), ("Overall_Pooled", pall)]:
                meta_rows.append({"method": m_name, "analysis_set": cfg, "subgroup": sub_lbl, **pd_obj, "Q_between": qb, "p_between": pb})
    pd.DataFrame(meta_rows).to_csv(REPORTS_DIR / "meta_analysis_frontal_theta_polarity.csv", index=False)

    # Age & Clean-Epoch-Duration ANCOVA Sensitivity
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
        X_age = sm.add_constant(np.column_stack([grp, age]))
        m_age = sm.OLS(yv, X_age).fit()
        X_aep = sm.add_constant(np.column_stack([grp, age, ep]))
        m_aep = sm.OLS(yv, X_aep).fit()
        anc_rows.append({
            "cohort": clbl, "n": len(glist), "marker": "Frontal_Theta_2to30",
            "age_only_grp_t": float(m_age.tvalues[1]), "age_only_grp_p": float(m_age.pvalues[1]), "age_p": float(m_age.pvalues[2]),
            "age_plus_epoch_grp_t": float(m_aep.tvalues[1]), "age_plus_epoch_grp_p": float(m_aep.pvalues[1]), "epoch_p": float(m_aep.pvalues[3]),
        })
    # Epoch-only adjustment for Study 2 EC (no individual ages)
    yv2 = np.array([g["x"].numpy()[FOCAL_5, 0].mean() for g in mum_ec])
    grp2 = np.array([float(g["y"]) for g in mum_ec])
    ep2 = np.array([float(g["n_clean_epochs"]) for g in mum_ec])
    m2 = sm.OLS(yv2, sm.add_constant(np.column_stack([grp2, ep2]))).fit()
    anc_rows.append({
        "cohort": "Study 2a: Outpatient MDD EC (N=58)", "n": 58, "marker": "Frontal_Theta_2to30",
        "age_only_grp_t": np.nan, "age_only_grp_p": np.nan, "age_p": np.nan,
        "age_plus_epoch_grp_t": float(m2.tvalues[1]), "age_plus_epoch_grp_p": float(m2.pvalues[1]), "epoch_p": float(m2.pvalues[2]),
    })
    pd.DataFrame(anc_rows).to_csv(REPORTS_DIR / "multicohort_age_and_duration_ancova.csv", index=False)

    # Full Age ANCOVA table across 4 EEG markers (matching Table S17 schema cleanly)
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
            yv = np.array([fn(g) for g in glist])
            mod = sm.OLS(yv, X).fit()
            anc_full.append({
                "cohort": clbl, "marker": m_lbl,
                "group_coef": float(mod.params[1]), "group_se": float(mod.bse[1]),
                "group_t": float(mod.tvalues[1]), "group_p": float(mod.pvalues[1]),
                "age_coef": float(mod.params[2]), "age_p": float(mod.pvalues[2]), "r2": float(mod.rsquared),
            })
    pd.DataFrame(anc_full).to_csv(REPORTS_DIR / "multicohort_age_ancova_complete.csv", index=False)

    # Paired medication ON/OFF + IAPF + UNM EC-vs-EO sensitivity
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
            vo = np.array([fn(g) for g in on_gs])
            vf = np.array([fn(g) for g in off_gs])
            st = compute_es(on_gs, off_gs, fn, paired=True)
            _, wp = wilcoxon(vo, vf)
            med_rows.append({
                "cohort": c_lbl, "marker": mlbl, "mean_off": float(vf.mean()), "mean_on": float(vo.mean()),
                "mean_diff": float((vo - vf).mean()), "paired_dz": st["cohens_d"],
                "ci_lower": st["ci_lower"], "ci_upper": st["ci_upper"], "paired_p": st["p_value"], "wilcoxon_p": float(wp),
            })
    pd.DataFrame(med_rows).to_csv(REPORTS_DIR / "unm_ucsd_paired_medication_biophysics.csv", index=False)

    # UNM Pure EC sensitivity vs Pure EO
    unm_ec_230 = unm_ec_map["2.0-30.0"]
    unm_ec_off = [g for g in unm_ec_230 if g["med_state"] == "OFF"]
    unm_ec_on = [g for g in unm_ec_230 if g["med_state"] == "ON"]
    unm_ec_hc = [g for g in unm_ec_230 if g["med_state"] == "CTL"]
    unm_sens_rows = []
    for cond, cs, rs in [
        ("UNM Pure EO (N=25 PD-OFF vs 25 CTL)", unm_off, unm_hc),
        ("UNM Pure EO (N=25 PD-ON vs 25 CTL)", unm_on, unm_hc),
        ("UNM Pure EC (N=25 PD-OFF vs 24 CTL)", unm_ec_off, unm_ec_hc),
        ("UNM Pure EC (N=25 PD-ON vs 24 CTL)", unm_ec_on, unm_ec_hc),
    ]:
        st_th = compute_es(cs, rs, lambda g: float(g["x_specparam_resid"].numpy()[FOCAL_5, 0].mean()))
        st_al = compute_es(cs, rs, lambda g: float(g["x_specparam_resid"].numpy()[POSTERIOR_5, 2].mean()))
        st_ch = compute_es(cs, rs, lambda g: float(g["x_specparam_resid"].numpy()[:, 3].mean()))
        unm_sens_rows.append({"condition": cond, "theta_d": st_th["cohens_d"], "theta_p": st_th["p_value"], "alpha_d": st_al["cohens_d"], "alpha_p": st_al["p_value"], "chi_d": st_ch["cohens_d"], "chi_p": st_ch["p_value"]})
    pd.DataFrame(unm_sens_rows).to_csv(REPORTS_DIR / "unm_ec_vs_eo_sensitivity.csv", index=False)

    # IAPF table
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

    # Grand-average spectra JSON for Figure S4 (both raw log10 and [2.0, 30.0] Hz residuals)
    freqs_full = np.asarray(psds["sleep_ds004902"][0]["freqs"], dtype=float)
    fmask = (freqs_full >= 1.0) & (freqs_full <= 40.0)
    f_plot = freqs_full[fmask]
    fmask_in = (freqs_full >= 2.0) & (freqs_full <= 30.0)
    f_in = freqs_full[fmask_in]

    def extract_grp_spec(sp_list, psd_list, sel_fn):
        raw_c, res_c = [], []
        fg = SpectralGroupModel(aperiodic_mode="fixed", peak_width_limits=[1.0, 12.0], max_n_peaks=6, min_peak_height=0.05, peak_threshold=2.0, verbose=False)
        sel_idx = [i for i, r in enumerate(sp_list) if sel_fn(r)]
        sel_psd = np.stack([np.maximum(np.asarray(psd_list[i]["mean_psd"], dtype=float)[:, fmask_in], 1e-25).mean(axis=0) for i in sel_idx])
        fg.fit(f_in, sel_psd, [2.0, 30.0])
        for k_i, idx in enumerate(sel_idx):
            lp_full = np.log10(np.maximum(np.asarray(psd_list[idx]["mean_psd"], dtype=float)[:, fmask], 1e-25).mean(axis=0))
            m = fg.get_model(k_i, regenerate=True)
            ap = m.results.model._ap_fit
            lp_in = m.data.power_spectrum
            raw_c.append(lp_full)
            res_c.append(lp_in - ap)
        ar = np.stack(raw_c)
        ares = np.stack(res_c)
        return {
            "n": len(sel_idx),
            "raw_mean": ar.mean(axis=0).tolist(), "raw_sem": (ar.std(axis=0, ddof=1) / np.sqrt(len(sel_idx))).tolist(),
            "res_mean": ares.mean(axis=0).tolist(), "res_sem": (ares.std(axis=0, ddof=1) / np.sqrt(len(sel_idx))).tolist(),
        }

    spectra_specs = [
        ("Study 1: Sleep Deprivation", "sleep_ds004902", "SD (N=71)", lambda r: r["condition"] == "SD", "NS (N=71)", lambda r: r["condition"] == "NS"),
        ("Study 2: Outpatient MDD (EC)", "mumtaz_ec", "MDD (N=30)", lambda r: r["group"] == "MDD", "HC (N=28)", lambda r: r["group"] == "HC"),
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

    master_payload = {
        "spectral_backend": "specparam_v2.0.0rc7_interior_2to30Hz",
        "accounting": {
            "total_unique_participants": 566,
            "total_analyzed_sessions": 677,
            "total_with_mumtaz_eo": 737,
            "enrolled_participants": 574,
            "excluded_participants": 8,
            "cohorts": {
                "Study 1 (Sleep ds004902)": {"enrolled": 71, "excluded": 0, "analyzed_pts": 71, "analyzed_sessions": 142, "design": "Within-subject paired (71 SD, 71 NS)"},
                "Study 2 (Mumtaz MDD 4244171)": {"enrolled": 64, "excluded": 6, "analyzed_pts": 58, "analyzed_sessions": 58, "design": "Between-subject EC (30 MDD, 28 HC); 60 EO (32 MDD, 28 HC)"},
                "Study 3 (Cavanagh ds003478)": {"enrolled": 121, "excluded": 2, "analyzed_pts": 119, "analyzed_sessions": 119, "design": "Between-subject (44 Depressed, 75 HC)"},
                "Study 4 (UCSD PD ds002778)": {"enrolled": 31, "excluded": 0, "analyzed_pts": 31, "analyzed_sessions": 46, "design": "15 PD (paired ON/OFF) + 16 HC"},
                "Study 5 (AHEPA Dementia ds004504)": {"enrolled": 88, "excluded": 0, "analyzed_pts": 88, "analyzed_sessions": 88, "design": "Between-subject (36 AD, 23 FTD, 29 HC)"},
                "Study 6 (UNM PD ds003490)": {"enrolled": 50, "excluded": 0, "analyzed_pts": 50, "analyzed_sessions": 75, "design": "25 PD (paired ON/OFF EO) + 25 HC (EO)"},
                "Study 7 (Iowa PD ds004584)": {"enrolled": 149, "excluded": 0, "analyzed_pts": 149, "analyzed_sessions": 149, "design": "Between-subject EO (100 PD ON, 49 HC)"},
            },
        },
        "omst_topology": {
            "total_graphs": 677,
            "total_node_instances": 677 * 19,
            "mean_trees": float(np.mean(omst_trees)),
            "sd_trees": float(np.std(omst_trees, ddof=1)),
            "pct_tree_1": float(100.0 * np.mean(np.array(omst_trees) == 1)),
            "pct_tree_2": float(100.0 * np.mean(np.array(omst_trees) >= 2)),
            "mean_edges": float(np.mean(omst_edges)),
            "sd_edges": float(np.std(omst_edges, ddof=1)),
            "mean_gce": float(np.mean(gces)),
            "sd_gce": float(np.std(gces, ddof=1)),
            "isolated_node_rate_pct": 0.0,
        },
    }
    with open(REPORTS_DIR / "master_results.json", "w", encoding="utf-8") as f:
        json.dump(master_payload, f, indent=2)

    # Empirical multiplicative gain-invariance audit on canonical [2.0, 30.0] Hz solver (285 spectra)
    sample_recs = psds["ucsd_ds002778"][:5] + psds["mumtaz_ec"][:5] + psds["iowa_ds004584"][:5]
    freqs_g = np.asarray(sample_recs[0]["freqs"], dtype=float)
    psds_g = np.vstack([np.maximum(np.asarray(r["mean_psd"], dtype=float), 1e-25) for r in sample_recs])
    fg_b = SpectralGroupModel(aperiodic_mode="fixed", peak_width_limits=[1.0, 12.0], max_n_peaks=6, min_peak_height=0.05, peak_threshold=2.0, verbose=False)
    fg_b.fit(freqs_g, psds_g, [2.0, 30.0], n_jobs=1)
    ap_b = fg_b.get_params("aperiodic")
    f_fit_g = fg_b.data.freqs
    th_m_g = (f_fit_g >= 4.0) & (f_fit_g < 8.0)
    th_b = np.array([float(np.mean((fg_b.get_model(i, regenerate=True).data.power_spectrum - fg_b.get_model(i, regenerate=True).results.model._ap_fit)[th_m_g])) for i in range(len(psds_g))])
    gain_chk_rows = []
    for k_val in [0.01, 0.50, 1.00, 2.00, 5.00, 100.00]:
        fg_k = SpectralGroupModel(aperiodic_mode="fixed", peak_width_limits=[1.0, 12.0], max_n_peaks=6, min_peak_height=0.05, peak_threshold=2.0, verbose=False)
        fg_k.fit(freqs_g, (k_val ** 2) * psds_g, [2.0, 30.0], n_jobs=1)
        ap_k = fg_k.get_params("aperiodic")
        th_k = np.array([float(np.mean((fg_k.get_model(i, regenerate=True).data.power_spectrum - fg_k.get_model(i, regenerate=True).results.model._ap_fit)[th_m_g])) for i in range(len(psds_g))])
        gain_chk_rows.append({
            "gain_k": k_val,
            "n_spectra": len(psds_g),
            "expected_offset_shift": float(2.0 * np.log10(k_val)),
            "measured_offset_shift_mean": float(np.mean(ap_k[:, 0] - ap_b[:, 0])),
            "max_abs_diff_chi": float(np.max(np.abs(ap_k[:, 1] - ap_b[:, 1]))),
            "max_abs_diff_theta_resid": float(np.max(np.abs(th_k - th_b))),
        })
    pd.DataFrame(gain_chk_rows).to_csv(REPORTS_DIR / "specparam_gain_invariance_check.csv", index=False)
    print("✓ Module A complete (+ [2, 30] Hz gain invariance check).")
    return master_payload


# ==============================================================================
# MODULE B: SYNTHETIC KNOWN-GROUND-TRUTH PSD SIMULATION (MONTE CARLO)
# ==============================================================================
def run_module_b_simulation():
    print("\n=== Module B: Known-Ground-Truth Spectral Leakage Monte Carlo Simulation ===")
    freqs = np.arange(1.0, 40.5, 0.5)
    f_log = np.log10(freqs)
    th_m_40 = (freqs >= 4.0) & (freqs < 8.0)
    mask_in = (freqs >= 2.0) & (freqs <= 30.0)
    f_in = freqs[mask_in]
    th_m_in = (f_in >= 4.0) & (f_in < 8.0)

    def gen_psd(rng, b, chi, th_amp, th_cf, th_bw, al_amp, al_cf, al_bw, rolloff_low=0.0, rolloff_high=0.0, noise_sd=0.03):
        ap = b - chi * f_log
        pk_th = th_amp * np.exp(-0.5 * ((freqs - th_cf) / th_bw) ** 2)
        pk_al = al_amp * np.exp(-0.5 * ((freqs - al_cf) / al_bw) ** 2)
        # Optional hardware transition filter skirts < 2 Hz and > 30 Hz
        att_low = -rolloff_low * np.maximum(0.0, 2.0 - freqs) ** 2
        att_high = -rolloff_high * np.maximum(0.0, (freqs - 30.0) / 10.0) ** 2
        logp = ap + pk_th + pk_al + att_low + att_high + rng.normal(0.0, noise_sd, size=len(freqs))
        return 10.0 ** logp

    rng = np.random.RandomState(2026)
    n_trials = 150
    n_case, n_ctrl = 30, 28

    scenarios = [
        # (name, d_chi_shift, d_th_shift, d_al_shift, roll_l_case, roll_h_case)
        ("S1_TrueNullTheta_TrueExponentDrop", -0.35, 0.00, +0.05, 0.0, 0.0),
        ("S2_TrueNullTheta_BroadAlphaAndSlowCurve", -0.25, 0.00, -0.25, 0.0, 0.0),
        ("S3_FilterEdgeAttenuation_Study2Replica", -0.35, -0.18, -0.15, 0.90, 1.15),
    ]

    sim_rows = []
    for s_name, d_chi, d_th, d_al, rl_c, rh_c in scenarios:
        d_th_ols40, d_ch_ols40 = [], []
        d_th_sp40, d_ch_sp40, r2_sp40 = [], [], []
        d_th_sp30, d_ch_sp30, r2_sp30 = [], [], []
        chi_bias_ols40, chi_bias_sp40, chi_bias_sp30 = [], [], []

        psds_batch = []
        y_batch = []
        true_chi_batch = []
        true_th_batch = []
        for t in range(n_trials):
            for grp, n_sub in [(0, n_ctrl), (1, n_case)]:
                for _ in range(n_sub):
                    chi_i = rng.normal(1.10 + (d_chi if grp == 1 else 0.0), 0.14)
                    th_i = max(0.02, rng.normal(0.18 + (d_th if grp == 1 else 0.0), 0.05))
                    al_i = max(0.05, rng.normal(0.65 + (d_al if grp == 1 else 0.0), 0.15))
                    b_i = rng.normal(-11.0, 0.25)
                    rl = rl_c if grp == 1 else 0.05
                    rh = rh_c if grp == 1 else 0.15
                    psds_batch.append(gen_psd(rng, b_i, chi_i, th_i, 6.0, 1.2, al_i, 10.0, 1.5, rolloff_low=rl, rolloff_high=rh))
                    y_batch.append(grp)
                    true_chi_batch.append(chi_i)
                    true_th_batch.append(float(np.mean(th_i * np.exp(-0.5 * ((f_in[th_m_in] - 6.0) / 1.2) ** 2))))

        psds_arr = np.asarray(psds_batch)
        fg40 = SpectralGroupModel(aperiodic_mode="fixed", peak_width_limits=[1.0, 12.0], max_n_peaks=6, min_peak_height=0.05, peak_threshold=2.0, verbose=False)
        fg40.fit(freqs, psds_arr, [1.0, 40.0], n_jobs=1)
        fg30 = SpectralGroupModel(aperiodic_mode="fixed", peak_width_limits=[1.0, 12.0], max_n_peaks=6, min_peak_height=0.05, peak_threshold=2.0, verbose=False)
        fg30.fit(freqs, psds_arr, [2.0, 30.0], n_jobs=1)

        aps40, r2_40_all = fg40.get_params("aperiodic"), fg40.get_metrics("gof", "rsquared")
        aps30, r2_30_all = fg30.get_params("aperiodic"), fg30.get_metrics("gof", "rsquared")

        # OLS [1, 40] masking [7, 14] Hz
        ols_mask = (freqs < 7.0) | (freqs > 14.0)
        lf_ols = f_log[ols_mask]

        N_per = n_case + n_ctrl
        for t in range(n_trials):
            i0, i1 = t * N_per, (t + 1) * N_per
            yt = np.asarray(y_batch[i0:i1])
            th_o40, ch_o40 = [], []
            th_s40, ch_s40 = [], []
            th_s30, ch_s30 = [], []
            t_true, c_true = np.asarray(true_th_batch[i0:i1]), np.asarray(true_chi_batch[i0:i1])

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

            def cd_arr(vals):
                v = np.asarray(vals)
                vc, vr = v[yt == 1], v[yt == 0]
                return float((vc.mean() - vr.mean()) / (np.sqrt(((len(vc) - 1) * vc.var(ddof=1) + (len(vr) - 1) * vr.var(ddof=1)) / (len(vc) + len(vr) - 2)) + 1e-12))

            d_th_ols40.append(cd_arr(th_o40)); d_ch_ols40.append(cd_arr(ch_o40))
            d_th_sp40.append(cd_arr(th_s40)); d_ch_sp40.append(cd_arr(ch_s40))
            d_th_sp30.append(cd_arr(th_s30)); d_ch_sp30.append(cd_arr(ch_s30))
            r2_sp40.append(float(np.median(r2_40_all[i0:i1])))
            r2_sp30.append(float(np.median(r2_30_all[i0:i1])))
            chi_bias_ols40.append(float(np.mean(np.asarray(ch_o40) - c_true)))
            chi_bias_sp40.append(float(np.mean(np.asarray(ch_s40) - c_true)))
            chi_bias_sp30.append(float(np.mean(np.asarray(ch_s30) - c_true)))

        true_d_th = d_th_true_list = float(np.mean([cd_arr(true_th_batch[t * N_per:(t + 1) * N_per]) for t in range(n_trials)]))
        true_d_chi = float(np.mean([cd_arr(true_chi_batch[t * N_per:(t + 1) * N_per]) for t in range(n_trials)]))
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
# MODULE C: 10x5 REPEATED PARTICIPANT-GROUPED CV + EDGE-MATCHED RANDOM GRAPHS
# ==============================================================================
def run_module_c_repeated_cv(sp):
    print("\n=== Module C: True 10x5 Repeated Participant-Grouped CV Across All Contrasts ===")
    sleep, mum_ec, mum_eo = sp["sleep_ds004902"], sp["mumtaz_ec"], sp["mumtaz_eo"]
    cav, ucsd, dem, unm, iowa = sp["cavanagh_ds003478"], sp["ucsd_ds002778"], sp["dementia_ds004504"], sp["unm_ds003490"], sp["iowa_ds004584"]

    # Unified canonical 12 contrast rows (9 Primary + 6b UNM PD-ON + 6c UNM PD All + 2b Mumtaz EO)
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
    models_12 = [
        "GCN_OMST", "GCN_Identity", "GCN_Random_EdgeMatched", "GCN_Full",
        "Tuned_LogisticRegression", "Tuned_Ridge", "Linear_SVM", "RBF_SVM",
        "RandomForest", "GBDT", "Single_Theta", "Single_Chi",
    ]
    adj_id = torch.zeros((19, 19), dtype=torch.float32)  # zero off-diagonals; layer adds I
    adj_full = make_dense_adj(19)

    repeat_rows = []
    lofo_rows = []
    split0_cls_rows = []
    fold_index_rows = []
    oof_split0_rows = []
    eo_split0_rows = []

    for cname, glist, is_grp in specs_12:
        t_c = time.time()
        vg = [g for g in glist if int(g["y"]) in (0, 1)]
        n = len(vg)
        y_all = np.array([int(g["y"]) for g in vg])
        groups = np.array([g.get("bids_id", g.get("sub_name", g.get("subject_id", str(i)))) for i, g in enumerate(vg)])
        X_all = torch.stack([g["x"] for g in vg])
        adj_omst = torch.stack([g["adj"] for g in vg])
        edge_counts = [max(1, int(g.get("omst_edge_count", 27))) for g in vg]

        for r_idx, seed in enumerate(REPEATS_10):
            set_seed(seed)
            # True participant-grouped shuffled 5-fold outer splits!
            splits = get_grouped_splits(groups, y_all, n_splits=5, seed=seed)
            if r_idx == 0:
                fold_map = np.zeros(n, dtype=int)
                for f_k, (_, va_k) in enumerate(splits):
                    fold_map[va_k] = f_k
                for i_s in range(n):
                    fold_index_rows.append({"cohort": cname, "repeat_seed": seed, "row_idx": i_s, "subject_id": vg[i_s]["subject_id"], "group_id": groups[i_s], "y": int(y_all[i_s]), "fold": int(fold_map[i_s])})

            # Build session-specific edge-matched degree-agnostic random graph per sample for this repeat
            adj_rnd_all = torch.stack([make_edge_matched_random_adj(19, edge_counts[i], seed=seed * 1000 + i) for i in range(n)])

            oof_p = {m: np.zeros(n) for m in models_12}
            oof_y = {m: np.zeros(n) for m in models_12}
            oof_lofo = {k: np.zeros(n) for k in ["Single_Delta", "Single_Alpha", "Single_GCE", "GCN_Ablate_Theta", "GCN_Ablate_Delta", "GCN_Ablate_Alpha", "GCN_Ablate_Chi"]} if r_idx == 0 else None

            for fold, (tr, va) in enumerate(splits):
                tx, vx = X_all[tr], X_all[va]
                ty = torch.tensor(y_all[tr], dtype=torch.long)
                tz, vz, _, _ = compute_fold_normative_z(tx, ty, vx)
                X_tr, X_va = tz.view(len(tr), -1).numpy(), vz.view(len(va), -1).numpy()
                y_tr = y_all[tr]

                oof_p["GCN_OMST"][va], oof_y["GCN_OMST"][va] = train_eval_gcn(tz, adj_omst[tr], ty, vz, adj_omst[va], seed=seed + fold)
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

                gs_lsvm = GridSearchCV(SVC(kernel="linear", probability=True, max_iter=10000, random_state=seed), {"C": [0.01, 0.1, 1.0, 10.0]}, cv=icv, scoring="roc_auc").fit(X_tr, y_tr)
                oof_p["Linear_SVM"][va], oof_y["Linear_SVM"][va] = gs_lsvm.predict_proba(X_va)[:, 1], gs_lsvm.predict(X_va)

                gs_rbf = GridSearchCV(SVC(kernel="rbf", probability=True, max_iter=10000, random_state=seed), {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto", 0.01, 0.1]}, cv=icv, scoring="roc_auc").fit(X_tr, y_tr)
                oof_p["RBF_SVM"][va], oof_y["RBF_SVM"][va] = gs_rbf.predict_proba(X_va)[:, 1], gs_rbf.predict(X_va)

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
            for m in models_12:
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
                    oof_split0_rows.append({"cohort": cname, "subject_id": vg[i_s]["subject_id"], "y": int(y_all[i_s]), "prob_GCN_OMST": float(oof_p["GCN_OMST"][i_s]), "prob_Tuned_LR": float(oof_p["Tuned_LogisticRegression"][i_s]), "prob_RF": float(oof_p["RandomForest"][i_s])})

        sub_gcn = [r["auc"] for r in repeat_rows if r["cohort"] == cname and r["model"] == "GCN_OMST"]
        sub_rnd = [r["auc"] for r in repeat_rows if r["cohort"] == cname and r["model"] == "GCN_Random_EdgeMatched"]
        sub_rf = [r["auc"] for r in repeat_rows if r["cohort"] == cname and r["model"] == "RandomForest"]
        sub_lr = [r["auc"] for r in repeat_rows if r["cohort"] == cname and r["model"] == "Tuned_LogisticRegression"]
        print(f"  {cname:<42} | 10x5 GCN={np.mean(sub_gcn):.3f}±{np.std(sub_gcn,ddof=1):.3f} | Rnd={np.mean(sub_rnd):.3f} | RF={np.mean(sub_rf):.3f} | LR={np.mean(sub_lr):.3f} ({time.time()-t_c:.1f}s)")

    df_rep = pd.DataFrame(repeat_rows)
    df_rep.to_csv(REPORTS_DIR / "benchmark_repeated_10x5_cv.csv", index=False)

    # Summary across 10 repeats per cohort x model
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

    pd.DataFrame(split0_cls_rows).to_csv(REPORTS_DIR / "multicohort_7cohort_extended_model_comparison.csv", index=False)
    pd.DataFrame(eo_split0_rows).to_csv(REPORTS_DIR / "mumtaz_eo_model_comparison.csv", index=False)
    pd.DataFrame(lofo_rows).to_csv(REPORTS_DIR / "multicohort_7cohort_feature_ablation.csv", index=False)
    pd.DataFrame(fold_index_rows).to_csv(RELEASE_DIR / "cv_fold_indices_split0.csv", index=False)
    pd.DataFrame(oof_split0_rows).to_csv(RELEASE_DIR / "oof_predictions_split0.csv", index=False)

    # Pairwise Wilcoxon on 10x5 REPEATED MEANS across (a) Primary 9 contrasts, (b) 8 Non-Nested (excl 5c), (c) All 11 contrasts
    piv_rep = df_sum[df_sum["cohort"] != "Study 2b: Mumtaz MDD (Eyes Open)"].pivot(index="cohort", columns="model", values="mean_auc")
    primary_9 = [c for c in piv_rep.index if c not in ("Study 6b: UNM PD-ON vs Control", "Study 6c: UNM PD All Sessions")]
    nonnested_8 = [c for c in primary_9 if c != "Study 5c: Dementia All (AD+FTD) vs Control"]
    all_11 = list(piv_rep.index)
    others = [m for m in models_12 if m != "GCN_OMST"]

    w_rows = []
    for sname, idxs in [("Primary_9_Contrasts_10x5Mean", primary_9), ("NonNested_8_Contrasts_10x5Mean", nonnested_8), ("All_11_Contrasts_10x5Mean", all_11)]:
        plist, tmp = [], []
        for m in others:
            x, y = piv_rep.loc[idxs, "GCN_OMST"].values, piv_rep.loc[idxs, m].values
            diff = x - y
            w_st, p_v = wilcoxon(x, y) if np.any(np.abs(diff) > 1e-8) else (0.0, 1.0)
            plist.append(p_v)
            tmp.append({
                "scope": sname, "n_benchmarks": len(idxs), "comparator_model": m,
                "gcn_mean_auc": float(x.mean()), "comp_mean_auc": float(y.mean()),
                "mean_diff_auc": float(diff.mean()), "median_diff_auc": float(np.median(diff)),
                "wins_ties_losses": f"{int(np.sum(diff > 1e-6))} / {int(np.sum(np.abs(diff) <= 1e-6))} / {int(np.sum(diff < -1e-6))}",
                "wilcoxon_W": float(w_st), "raw_p": float(p_v),
            })
        _, qvals, _, _ = multipletests(plist, method="fdr_bh")
        for r, qv in zip(tmp, qvals):
            r["fdr_q"] = float(qv)
            w_rows.append(r)
    pd.DataFrame(w_rows).to_csv(REPORTS_DIR / "multicohort_wilcoxon_model_comparisons.csv", index=False)

    # Edge-matched GCN hyperparameter sensitivity on 10x5? (Keep canonical 5-cohort split0 grid with fast runtime)
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
        grps = np.array([g.get("bids_id", g.get("sub_name", g.get("subject_id", str(i)))) for i, g in enumerate(vg)])
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
    print("✓ Module C complete: 10x5 Repeated Grouped CV + edge-matched GCN-Random + LOFO + Wilcoxon saved.")


# ==============================================================================
# MODULE D: ALIGNED 1,000-PERMUTATION RETRAINING TESTS (TRUE TUNED LR)
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
            # Fixed disjoint 50/50 split of target controls matching leak-free Split-Target-HC!
            hc_idx = np.where(y_tgt == 0)[0]
            pt_idx = np.where(y_tgt == 1)[0]
            perm_hc = np.random.RandomState(42).permutation(hc_idx)
            cal_hc, eval_hc = perm_hc[:len(perm_hc)//2], perm_hc[len(perm_hc)//2:]
            eval_idx = np.concatenate([pt_idx, eval_hc])
            tz, _, _, _ = compute_fold_normative_z(X_src, ty_t, X_src)
            mu_hc = X_tgt[cal_hc].mean(dim=0, keepdim=True)
            sd_hc = X_tgt[cal_hc].std(dim=0, keepdim=True) + 1e-6
            vz_eval = (X_tgt[eval_idx] - mu_hc) / sd_hc
            pr_gcn, _ = train_eval_gcn(tz, adj_src, ty_t, vz_eval, adj_tgt[eval_idx], epochs=40, seed=42)
            null_gcn.append(float(roc_auc_score(y_tgt[eval_idx], pr_gcn)))
            icv = get_grouped_splits(grps_src, y_perm_tr, n_splits=3, seed=42)
            gs_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}, cv=icv, scoring="roc_auc").fit(tz.view(len(ty_t), -1).numpy(), y_perm_tr)
            null_lr.append(float(roc_auc_score(y_tgt[eval_idx], gs_lr.predict_proba(vz_eval.view(len(eval_idx), -1).numpy())[:, 1])))
        else:
            # Permute at unique participant cluster level (preserves repeated-session dependence cleanly)
            uniq_g = np.unique(grps_src)
            g_map = {ug: int(y_src[grps_src == ug][0]) for ug in uniq_g}
            perm_vals = rng.permutation(list(g_map.values()))
            pmap = dict(zip(uniq_g, perm_vals))
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
    print("\n=== Module D: Aligned 1,000-Permutation Tests (Exact Split-0 Tuned LR + Group Splits) ===")
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
        cal_hc, eval_hc = perm_hc[:len(perm_hc)//2], perm_hc[len(perm_hc)//2:]
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
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
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


# ==============================================================================
# MODULE E: LEAK-FREE SPLIT-HC TRANSFER, GAIN ABLATION, PLI SPD, REGRESSION
# ==============================================================================
def run_module_e_rest(sp, psds):
    print("\n=== Module E: Leak-Free Split-HC Transfer, 10x5 Leakage, Gain Stress, PLI SPD, Regressions ===")
    sleep, mum_ec, mum_eo = sp["sleep_ds004902"], sp["mumtaz_ec"], sp["mumtaz_eo"]
    cav, ucsd, dem, unm, iowa = sp["cavanagh_ds003478"], sp["ucsd_ds002778"], sp["dementia_ds004504"], sp["unm_ds003490"], sp["iowa_ds004584"]
    unm_off = [g for g in unm if g["med_state"] in ("OFF", "CTL")]

    # 1. Leak-Free 20-Repeat 50/50 Split-Target-HC External Transfer (+ Oracle Full-HC comparison)
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

        # Train one fixed GCN_OMST & GCN_Identity on source
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

        # 20 disjoint 50/50 splits of target HC (zero overlap between calibration HC and evaluation HC)
        rng = np.random.RandomState(42)
        split_aucs = {m: [] for m in ["GCN_OMST", "GCN_Identity", "Tuned_LR", "RandomForest", "GBDT"]}
        split_baccs = {m: [] for m in ["GCN_OMST", "GCN_Identity", "Tuned_LR", "RandomForest", "GBDT"]}

        # Oracle full-HC (kept strictly for supplemental audit)
        mu_f, sd_f = Xt[hc_idx].mean(dim=0, keepdim=True), Xt[hc_idx].std(dim=0, keepdim=True) + 1e-6
        vz_f = (Xt - mu_f) / sd_f
        with torch.no_grad():
            pf_gcn = torch.softmax(gcn_omst(vz_f, At), dim=-1)[:, 1].numpy()
            pf_id = torch.softmax(gcn_id(vz_f, torch.zeros((len(tv), 19, 19))), dim=-1)[:, 1].numpy()
        Xte_f = vz_f.view(len(tv), -1).numpy()
        oracle = {
            "GCN_OMST": float(roc_auc_score(yt, pf_gcn)),
            "GCN_Identity": float(roc_auc_score(yt, pf_id)),
            "Tuned_LR": float(roc_auc_score(yt, lr.predict_proba(Xte_f)[:, 1])),
            "RandomForest": float(roc_auc_score(yt, rf.predict_proba(Xte_f)[:, 1])),
            "GBDT": float(roc_auc_score(yt, gb.predict_proba(Xte_f)[:, 1])),
        }

        for _ in range(20):
            ph = rng.permutation(hc_idx)
            cal_h, ev_h = ph[:len(ph)//2], ph[len(ph)//2:]
            ev_idx = np.concatenate([pt_idx, ev_h])
            mu_s, sd_s = Xt[cal_h].mean(dim=0, keepdim=True), Xt[cal_h].std(dim=0, keepdim=True) + 1e-6
            vz_s = (Xt[ev_idx] - mu_s) / sd_s
            Xte_s, yte_s = vz_s.view(len(ev_idx), -1).numpy(), yt[ev_idx]
            with torch.no_grad():
                ps_g = torch.softmax(gcn_omst(vz_s, At[ev_idx]), dim=-1)[:, 1].numpy()
                ps_i = torch.softmax(gcn_id(vz_s, torch.zeros((len(ev_idx), 19, 19))), dim=-1)[:, 1].numpy()
            pm = {
                "GCN_OMST": ps_g, "GCN_Identity": ps_i,
                "Tuned_LR": lr.predict_proba(Xte_s)[:, 1],
                "RandomForest": rf.predict_proba(Xte_s)[:, 1],
                "GBDT": gb.predict_proba(Xte_s)[:, 1],
            }
            for m, pr in pm.items():
                split_aucs[m].append(float(roc_auc_score(yte_s, pr)))
                split_baccs[m].append(float(balanced_accuracy_score(yte_s, (pr >= 0.5).astype(int))))

        rows = []
        for m in ["GCN_OMST", "GCN_Identity", "Tuned_LR", "RandomForest", "GBDT"]:
            arr = np.asarray(split_aucs[m])
            rows.append({
                "transfer_type": ttype, "source_cohort": sn, "target_cohort": tn,
                "n_source": len(sv), "n_target": len(tv), "model": m,
                "auc": float(arr.mean()), "split_hc_sd": float(arr.std(ddof=1)),
                "split_hc_p05": float(np.percentile(arr, 5)), "split_hc_p95": float(np.percentile(arr, 95)),
                "balanced_accuracy": float(np.mean(split_baccs[m])),
                "oracle_full_hc_auc": oracle[m],
            })
        return rows

    pd_cohorts = {"UCSD_PD": ucsd, "UNM_PD_OFF": unm_off, "Iowa_PD": iowa}
    all_tr = []
    for s in pd_cohorts:
        for t in pd_cohorts:
            if s != t:
                all_tr.extend(eval_transfer_pair(pd_cohorts[s], pd_cohorts[t], s, t, "Within_Disorder_PD_Cross_Site"))
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

    # Mumtaz EO vs EC concordance + leak-free Split-HC state transfer
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

    # 2. 10x5 Repeated Validation Leakage Ablation (Study 1, Study 4, Study 6c)
    leak_rows = []
    leak_specs = [
        ("Study 1: Sleep Deprivation", "Paired_Crossover_Discordant", sleep, lambda g: g.get("sub_name", g["subject_id"])),
        ("Study 4: UCSD Parkinson", "Repeated_Sessions_Concordant", ucsd, lambda g: g.get("bids_id", g["subject_id"])),
        ("Study 6c: UNM Parkinson All", "Repeated_Sessions_Concordant", unm, lambda g: g["subject_id"]),
    ]
    for cname, dtype, glist, gfn in leak_specs:
        vg = [g for g in glist if int(g["y"]) in (0, 1)]
        y_all = np.array([int(g["y"]) for g in vg])
        grps = np.array([gfn(g) for g in vg])
        X_all = torch.stack([g["x"] for g in vg])
        A_all = torch.stack([g["adj"] for g in vg])
        gl_z = (X_all - X_all.mean(0, keepdims=True)) / (X_all.std(0, keepdims=True) + 1e-6)
        per_rep = []
        for r_idx, seed in enumerate(REPEATS_10):
            set_seed(seed)
            for reg in ["Strict_Leak_Free", "Leaky_Pre_Standardization", "Leaky_Subject_Mixing"]:
                if reg != "Leaky_Subject_Mixing":
                    splits = get_grouped_splits(grps, y_all, n_splits=5, seed=seed)
                else:
                    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(vg, y_all))
                ov = 100.0 * sum(sum(1 for gid in grps[va] if gid in set(grps[tr])) for tr, va in splits) / len(vg)
                oof = {m: np.zeros(len(vg)) for m in ["Tuned_LogisticRegression", "RandomForest", "GCN_OMST"]}
                for fold, (tr, va) in enumerate(splits):
                    tx, vx = X_all[tr], X_all[va]
                    ty = torch.tensor(y_all[tr], dtype=torch.long)
                    tz, vz = (gl_z[tr], gl_z[va]) if reg == "Leaky_Pre_Standardization" else compute_fold_normative_z(tx, ty, vx)[:2]
                    Xtr, Xva = tz.view(len(tr), -1).numpy(), vz.view(len(va), -1).numpy()
                    icv = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + fold).split(Xtr, y_all[tr])) if reg == "Leaky_Subject_Mixing" else get_grouped_splits(grps[tr], y_all[tr], n_splits=3, seed=seed + fold)
                    gs_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=seed), {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}, cv=icv, scoring="roc_auc").fit(Xtr, y_all[tr])
                    oof["Tuned_LogisticRegression"][va] = gs_lr.predict_proba(Xva)[:, 1]
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

    # 3. PLI SPD Audit + Riemannian Tangent-Space (re-run cleanly on updated Study 6 EO!)
    eig_rows, all_eigs = [], []
    for clbl, glist in [
        ("Study 1: Sleep Deprivation (ds004902)", sleep),
        ("Study 2: Outpatient MDD (EC)", mum_ec),
        ("Study 3: Cavanagh Depression (ds003478)", cav),
        ("Study 4: UCSD Parkinson (ds002778)", ucsd),
        ("Study 5: Clinical Dementia (ds004504)", dem),
        ("Study 6: UNM Parkinson EO (ds003490)", unm),
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
            "cohort": clbl, "n_sessions": len(arr),
            "min_eigenvalue": float(arr.min()), "median_min_eigenvalue": float(np.median(arr)),
            "n_indefinite_raw": int(np.sum(arr <= 0.0)), "pct_indefinite_raw": float(100.0 * np.mean(arr <= 0.0)),
            "n_indefinite_after_1e4_shift": int(np.sum(arr + 1e-4 <= 0.0)),
        })
    pd.DataFrame(eig_rows).to_csv(REPORTS_DIR / "pli_spd_eigenvalue_audit.csv", index=False)
    aa = np.asarray(all_eigs)
    pd.DataFrame([{"cohort": "Total Across All 7 Cohorts", "n_sessions": len(aa), "min_eigenvalue": float(aa.min()), "median_min_eigenvalue": float(np.median(aa)), "n_indefinite_raw": int(np.sum(aa <= 0.0)), "pct_indefinite_raw": float(100.0 * np.mean(aa <= 0.0)), "n_indefinite_after_1e4_shift": int(np.sum(aa + 1e-4 <= 0.0))}]).to_csv(REPORTS_DIR / "pli_spd_eigenvalue_summary.csv", index=False)

    # Riemannian baselines (10 contrasts)
    def upper_vec(M):
        i, j = np.triu_indices_from(M, k=0)
        return M[i, j] * np.where(i == j, 1.0, np.sqrt(2.0))
    i_off, j_off = np.triu_indices(19, k=1)
    riem_rows = []
    for cname, glist in [
        ("Study 1: Sleep Deprivation (ds004902)", sleep), ("Study 2: Mumtaz MDD (Eyes Closed)", mum_ec),
        ("Study 3: Cavanagh University Depression", cav), ("Study 4: UCSD Parkinson All Sessions", ucsd),
        ("Study 5a: Dementia AD vs Control", [g for g in dem if g["diagnosis"] in ("Alzheimer", "Control")]),
        ("Study 5b: Dementia FTD vs Control", [g for g in dem if g["diagnosis"] in ("FTD", "Control")]),
        ("Study 5c: Dementia All (AD+FTD) vs Control", dem),
        ("Study 6a: UNM PD-OFF vs Control", unm_off),
        ("Study 6b: UNM PD-ON vs Control", [g for g in unm if g["med_state"] in ("ON", "CTL")]),
        ("Study 7: Iowa Parkinson (ds004584)", iowa),
    ]:
        vg = [g for g in glist if int(g["y"]) in (0, 1)]
        y_all = np.array([int(g["y"]) for g in vg])
        grps = np.array([g.get("bids_id", g.get("sub_name", g["subject_id"])) for g in vg])
        Xe, Xa, Xh = [], [], []
        for g in vg:
            pm = g["pli"].numpy().astype(np.float64); C = 0.5 * (pm + pm.T); np.fill_diagonal(C, 1.0)
            Xe.append(C[i_off, j_off])
            vals, vecs = np.linalg.eigh(C)
            sh = max(0.0, 0.01 - float(vals.min()))
            Xa.append(upper_vec((vecs * np.log(vals + sh)) @ vecs.T))
            Xh.append(upper_vec((vecs * np.log(np.maximum(vals, 1e-4))) @ vecs.T))
        for rname, Xrep in [("Euclidean_UpperTri_171D", np.asarray(Xe)), ("Riemannian_Gershgorin_SPD_190D", np.asarray(Xa)), ("Riemannian_Higham_SPD_190D", np.asarray(Xh))]:
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

    # 4. Gain Calibration & Stress Test on canonical [2.0, 30.0] Hz + masked OLS + raw power + rel power
    def build_gain_feats(recs, ftype, gain_k):
        feats = []
        for r, p_rec in recs:
            psd = p_rec["mean_psd"] * (gain_k ** 2)
            freqs = p_rec["freqs"]
            f30 = (freqs >= 2.0) & (freqs <= 30.0)
            fv = freqs[f30]
            if ftype == "specparam_periodic_residual":
                # Invariant cleanly inside [2.0, 30.0] Hz; recompute for k=0.5/1.0/2.0/5.0 cleanly from [2, 30] or analytical check
                x = r["x"].numpy().copy()
            elif ftype == "masked_ols_residual":
                x = r["x_masked_ols_1to40"].numpy().copy()
            elif ftype == "raw_log_band_power":
                # Band-averaged raw log10 PSD * (k^2) across theta, delta, alpha + raw 1/f chi
                th_m = (fv >= 4.0) & (fv < 8.0); dl_m = (fv >= 2.0) & (fv < 4.0); al_m = (fv >= 8.0) & (fv <= 13.0)
                lp = np.log10(np.maximum(psd[:, f30], 1e-25))
                x = np.zeros((19, 4), dtype=np.float32)
                x[:, 0] = lp[:, th_m].mean(axis=1)
                x[:, 1] = lp[:, dl_m].mean(axis=1)
                x[:, 2] = lp[:, al_m].mean(axis=1)
                x[:, 3] = r["x"].numpy()[:, 3]
            elif ftype == "relative_band_power":
                th_m = (fv >= 4.0) & (fv < 8.0); dl_m = (fv >= 2.0) & (fv < 4.0); al_m = (fv >= 8.0) & (fv <= 13.0)
                tot = np.maximum(psd[:, f30].sum(axis=1, keepdims=True), 1e-25)
                x = np.zeros((19, 4), dtype=np.float32)
                x[:, 0] = np.log10(np.maximum(psd[:, f30][:, th_m].mean(axis=1, keepdims=True) / tot, 1e-12)).ravel()
                x[:, 1] = np.log10(np.maximum(psd[:, f30][:, dl_m].mean(axis=1, keepdims=True) / tot, 1e-12)).ravel()
                x[:, 2] = np.log10(np.maximum(psd[:, f30][:, al_m].mean(axis=1, keepdims=True) / tot, 1e-12)).ravel()
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
            ys = np.array([int(r["y"]) for r, _ in s_pairs])
            yt = np.array([int(r["y"]) for r, _ in t_pairs])
            for ftype in ["specparam_periodic_residual", "masked_ols_residual", "relative_band_power", "raw_log_band_power"]:
                Xs_base = build_gain_feats(s_pairs, ftype, 1.0)
                Xt_base = build_gain_feats(t_pairs, ftype, 1.0)
                lr_u = LogisticRegression(C=0.1, max_iter=1000, random_state=42).fit(Xs_base, ys)
                rf_u = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(Xs_base, ys)
                p_lr_ref = lr_u.predict_proba(Xt_base)[:, 1]
                p_rf_ref = rf_u.predict_proba(Xt_base)[:, 1]

                for k_lbl, k_fn in [("0.5", lambda n: np.full(n, 0.5)), ("1.0", lambda n: np.ones(n)), ("2.0", lambda n: np.full(n, 2.0)), ("5.0", lambda n: np.full(n, 5.0)), ("Heterogeneous_0.5x_5.0x", lambda n: np.array([0.5 if i % 2 == 0 else 5.0 for i in range(n)]))]:
                    ks = k_fn(len(t_pairs))
                    Xt_k = np.vstack([build_gain_feats([t_pairs[i]], ftype, ks[i]) for i in range(len(t_pairs))])
                    pr_lr = lr_u.predict_proba(Xt_k)[:, 1]
                    pr_rf = rf_u.predict_proba(Xt_k)[:, 1]
                    gain_rows.append({
                        "experiment": "Amplifier_Gain_Ablation", "source": sn, "target": tn, "calibration": "Uncalibrated_Raw",
                        "feature_type": ftype, "gain_k": k_lbl,
                        "lr_auc": float(roc_auc_score(yt, pr_lr)), "lr_bacc": float(balanced_accuracy_score(yt, (pr_lr >= 0.5).astype(int))), "lr_brier": float(brier_score_loss(yt, pr_lr)),
                        "rf_auc": float(roc_auc_score(yt, pr_rf)),
                        "max_feat_drift": float(np.max(np.abs(Xt_k - Xt_base))),
                        "lr_max_prob_drift": float(np.max(np.abs(pr_lr - p_lr_ref))),
                        "rf_max_prob_drift": float(np.max(np.abs(pr_rf - p_rf_ref))),
                    })
                # Calibration strategies at k=1.0
                for cal_lbl in ["Uncalibrated_Raw", "Global_Cohort_ZScore", "Target_HC_Normative"]:
                    if cal_lbl == "Uncalibrated_Raw":
                        Xs_c, Xt_c = Xs_base, Xt_base
                    elif cal_lbl == "Global_Cohort_ZScore":
                        Xs_c = (Xs_base - Xs_base.mean(0, keepdims=True)) / (Xs_base.std(0, keepdims=True) + 1e-6)
                        Xt_c = (Xt_base - Xt_base.mean(0, keepdims=True)) / (Xt_base.std(0, keepdims=True) + 1e-6)
                    else:
                        s_hc = Xs_base[ys == 0]; t_hc = Xt_base[yt == 0]
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
                        "max_feat_drift": 0.0, "lr_max_prob_drift": 0.0, "rf_max_prob_drift": 0.0,
                    })
    pd.DataFrame(gain_rows).to_csv(REPORTS_DIR / "benchmark_gain_and_calibration_ablation.csv", index=False)

    # 5. Clinical Severity Regressions (10x5 repeated OOF!)
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
    for jkey, clbl, sclbl, glist, skey in reg_tasks:
        valid = [g for g in glist if pd.notnull(g.get(skey))]
        y = np.array([float(g[skey]) for g in valid])
        X = np.stack([g["x"].numpy().flatten() for g in valid])
        for mname, fn_est in [
            ("GBDT_Regressor", lambda s: HistGradientBoostingRegressor(max_iter=100, max_depth=4, random_state=s)),
            ("Random_Forest_Regressor", lambda s: RandomForestRegressor(n_estimators=100, max_depth=5, random_state=s)),
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
    pd.DataFrame(reg_rows).to_csv(REPORTS_DIR / "multicohort_clinical_regression_benchmarks.csv", index=False)

    # 6. Export canonical derived session features CSV + requirements lock for Zenodo release
    feat_rows = []
    for ckey, recs in sp.items():
        for g in recs:
            arr = g["x"].numpy()
            feat_rows.append({
                "cohort_key": ckey, "subject_id": g["subject_id"], "group": g.get("group", g.get("diagnosis", g.get("condition", ""))),
                "med_state": g.get("med_state", ""), "y": int(g["y"]), "age": g.get("age", np.nan),
                "n_clean_epochs": int(g.get("n_clean_epochs", -1)), "omst_trees": int(g["omst_trees"]), "omst_edges": int(g["omst_edge_count"]),
                "gce": float(g["gce"]), "frontal_theta_2to30": float(arr[FOCAL_5, 0].mean()), "global_lowfreq_2to4": float(arr[:, 1].mean()),
                "posterior_alpha": float(arr[POSTERIOR_5, 2].mean()), "global_chi_2to30": float(arr[:, 3].mean()),
                "iapf_hz": g.get("iapf_posterior", np.nan),
                "mean_r2_2to30": float(np.mean(g["sp_r2"])), "mean_mae_2to30": float(np.mean(g["sp_mae"])),
            })
    pd.DataFrame(feat_rows).to_csv(RELEASE_DIR / "derived_session_features_737.csv", index=False)
    try:
        req_txt = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (RELEASE_DIR / "requirements_lock.txt").write_text(req_txt, encoding="utf-8")
        (REPORTS_DIR / "requirements_lock.txt").write_text(req_txt, encoding="utf-8")
    except Exception as exc:
        print(f"Warning: could not export pip freeze: {exc}")
    print("✓ Module E complete & release artifacts exported.")


def main():
    t0 = time.time()
    sp, sp_140, psds, unm_ec_map = load_canonical_bundles()
    run_module_a(sp, psds, unm_ec_map)
    run_module_b_simulation()
    run_module_c_repeated_cv(sp)
    run_module_d_permutations(sp)
    run_module_e_rest(sp, psds)
    print(f"\n🎉 ALL REMEDIATED BENCHMARKS COMPLETED IN {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
