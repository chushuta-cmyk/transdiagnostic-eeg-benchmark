#!/usr/bin/env python3
"""
Unified Genuine specparam (v2.0.0rc7) Multi-Cohort Benchmark Pipeline.

Eliminates the legacy [7, 14] Hz-masked OLS split-data contamination by
re-running every Paper 2 downstream analysis from genuine specparam periodic
residuals and aperiodic exponents (x_specparam_resid in
data/derivatives/specparam_vs_masked_ols_7cohorts.pt):
  1. Archives legacy pre-Nov-14 masked-OLS reports to reports/archive_cibm_masked_ols/
  2. Attaches BIDS participant ages & computes posterior IAPF from Welch PSDs
  3. Recomputes canonical biophysics (50-test & 10-test BH-FDR), 5-cohort Age ANCOVA,
     paired PD medication tests, and dynamic clinical scale regressions -> master_results.json
  4. Runs 12-model x 11-contrast (+ Mumtaz EO) within-cohort CV, LOFO ablations, Wilcoxon tests
  5. Runs 5-seed repeated CV (350 runs) & NormativeGCN hyperparameter sensitivity (35 runs)
  6. Runs cross-site PD (6 pairs), transdiagnostic transfer (14 pairs), EO/EC state-matched
     transfer, and Split-Target-HC holdout calibration checks
  7. Runs leakage ablation, heuristic slowing baseline, PLI SPD eigenvalue audit,
     Riemannian tangent-space baselines, and DL/HKSJ polarity meta-analysis
  8. Computes real grand-average spectra (Fig S4) and real 1,000-permutation null
     distributions (Fig S6) via multi-core worker execution
"""

import json
import os
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy.linalg as linalg
import scipy.stats as stats
import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.optim as optim
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
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.svm import SVC, SVR
from specparam import SpectralGroupModel
from statsmodels.stats.multitest import multipletests

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = BENCHMARK_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
from harmonizer_and_graph import CHANNELS_19, NormativeGCN, compute_fold_normative_z

DATA_DIR = BENCHMARK_ROOT / "data" / "derivatives"
REPORTS_DIR = BENCHMARK_ROOT / "reports"
ARCHIVE_DIR = REPORTS_DIR / "archive_cibm_masked_ols"
EEG_ROOT = Path("/Users/tan/Documents/eeg_datastes")

FOCAL_5 = [CHANNELS_19.index(c) for c in ["Fp1", "Fp2", "F3", "F4", "Fz"]]
FRONTAL_3 = [CHANNELS_19.index(c) for c in ["F3", "Fz", "F4"]]
POSTERIOR_5 = [CHANNELS_19.index(c) for c in ["P3", "P4", "Pz", "O1", "O2"]]
CENTRAL_3 = [CHANNELS_19.index(c) for c in ["C3", "Cz", "C4"]]

SEEDS_5 = [42, 123, 456, 789, 2026]


def set_seed(seed: int = 42):
    """Set deterministic RNG states across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def archive_legacy_reports():
    """Copy pre-Nov-14 masked-OLS reports once to archive_cibm_masked_ols/."""
    if ARCHIVE_DIR.exists():
        return
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    legacy_names = [
        "master_results.json",
        "multicohort_biophysical_effect_sizes_expanded.csv",
        "multicohort_7cohort_extended_model_comparison.csv",
        "multicohort_7cohort_feature_ablation.csv",
        "multicohort_wilcoxon_model_comparisons.csv",
        "multicohort_age_ancova_complete.csv",
        "unm_ucsd_paired_medication_biophysics.csv",
        "cross_site_pd_transfer_benchmarks.csv",
        "comprehensive_transdiagnostic_transfer_matrix.csv",
        "multicohort_clinical_regression_benchmarks.csv",
        "mumtaz_eo_model_comparison.csv",
        "mumtaz_eo_state_matched_transfer.csv",
        "benchmark_leakage_ablation.csv",
        "benchmark_heuristic_slowing.csv",
        "benchmark_multiseed_repeated_cv.csv",
        "benchmark_multiseed_summary.csv",
        "pli_spd_eigenvalue_audit.csv",
        "benchmark_riemannian_tangent_space.csv",
        "meta_analysis_frontal_theta_polarity.csv",
        "benchmark_gcn_hyperparameter_sensitivity.csv",
    ]
    copied = 0
    for name in legacy_names:
        src = REPORTS_DIR / name
        if src.exists():
            shutil.copy2(src, ARCHIVE_DIR / name)
            copied += 1
    print(f"Archived {copied} legacy masked-OLS files to {ARCHIVE_DIR}")


def bootstrap_auc_ci(y_true, y_prob, n_boot: int = 1000, seed: int = 42):
    """Compute ROC-AUC with 1,000-sample percentile bootstrap 95% CI."""
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
    """Compute cluster bootstrap 95% CI resampling unique participant IDs."""
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


def welch_df(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Welch-Satterthwaite degrees of freedom."""
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1) / nx, np.var(y, ddof=1) / ny
    denom = (vx ** 2) / (nx - 1) + (vy ** 2) / (ny - 1)
    return float(((vx + vy) ** 2) / denom) if denom > 0 else float(nx + ny - 2)


def make_random_adj(num_nodes: int = 19, num_edges: int = 18, seed: int = 42) -> torch.Tensor:
    """Construct degree-normalized fixed random 18-edge graph adjacency."""
    rng = np.random.RandomState(seed)
    adj = np.eye(num_nodes, dtype=np.float32)
    edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    for idx in rng.choice(len(edges), size=num_edges, replace=False):
        u, v = edges[idx]
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    deg = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(deg, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = np.diag(d_inv_sqrt)
    return torch.tensor(d_mat @ adj @ d_mat, dtype=torch.float32)


def train_eval_gcn(
    train_z,
    train_adj,
    train_y,
    val_z,
    val_adj,
    in_dim: int = 4,
    hidden_dim: int = 32,
    epochs: int = 50,
    lr: float = 0.005,
    weight_decay: float = 1e-4,
    seed: int = 42,
):
    """Train and evaluate NormativeGCN on normalized fold tensors."""
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


def load_canonical_specparam_cohorts():
    """Load genuine specparam cohort bundle, verify zero NaNs, attach BIDS ages & IAPF."""
    sp = torch.load(DATA_DIR / "specparam_vs_masked_ols_7cohorts.pt", weights_only=False)
    psds = torch.load(DATA_DIR / "all_7cohorts_welch_psds.pt", weights_only=False)

    # Inject participant-level ages for Study 3 (ds003478) & Study 4 (ds002778)
    cav_tsv = pd.read_csv(EEG_ROOT / "depression_new" / "participants.tsv", sep="\t")
    cav_age = {
        str(r["participant_id"]): (float(r["age"]) if pd.notnull(r["age"]) else np.nan)
        for _, r in cav_tsv.iterrows()
    }
    ucsd_tsv = pd.read_csv(EEG_ROOT / "participants.tsv", sep="\t")
    ucsd_age = {
        str(r["participant_id"]): (float(r["age"]) if pd.notnull(r["age"]) else np.nan)
        for _, r in ucsd_tsv.iterrows()
    }

    # Verify zero failed specparam fits and compute genuine posterior IAPF (7.0-13.5 Hz)
    iapf_all = {}
    for ckey, recs in sp.items():
        psd_recs = psds[ckey]
        assert len(recs) == len(psd_recs), f"Length mismatch in {ckey}"
        freqs = np.asarray(psd_recs[0]["freqs"], dtype=float)
        fg = SpectralGroupModel(
            aperiodic_mode="fixed",
            peak_width_limits=[1.0, 12.0],
            max_n_peaks=6,
            min_peak_height=0.05,
            peak_threshold=2.0,
            verbose=False,
        )
        post_psds = np.stack([np.maximum(np.asarray(pr["mean_psd"], dtype=float)[POSTERIOR_5], 1e-20) for pr in psd_recs])
        # Fit all posterior spectra in batch [n_sess * 5, n_freqs]
        n_sess = len(recs)
        fg.fit(freqs, post_psds.reshape(n_sess * len(POSTERIOR_5), -1), [1.0, 40.0])

        sess_iapfs = []
        for idx, r in enumerate(recs):
            assert np.all(r["sp_fit_ok"]), f"Failed fit in {ckey} idx={idx}"
            x_sp = r["x_specparam_resid"]
            assert not torch.isnan(x_sp).any(), f"NaN in x_specparam_resid {ckey} idx={idx}"
            # Promote genuine specparam residuals to canonical 'x'
            r["x"] = x_sp.clone()
            if ckey == "cavanagh_ds003478":
                sub_id = r["subject_id"].replace("unm_", "")
                r["age"] = cav_age.get(sub_id, np.nan)
            elif ckey == "ucsd_ds002778":
                r["age"] = ucsd_age.get(r["bids_id"], np.nan)

            # Extract highest-power alpha-band peak (7.0-13.5 Hz) across 5 posterior channels
            ch_peaks = []
            for ch_off in range(len(POSTERIOR_5)):
                m_res = fg.get_model(idx * len(POSTERIOR_5) + ch_off, regenerate=False)
                pks = m_res.results.params.periodic.params
                if pks.size > 0:
                    alpha_pks = pks[(pks[:, 0] >= 7.0) & (pks[:, 0] <= 13.5)]
                    if len(alpha_pks) > 0:
                        best_row = alpha_pks[np.argmax(alpha_pks[:, 1])]
                        ch_peaks.append(float(best_row[0]))
            iapf_val = float(np.median(ch_peaks)) if ch_peaks else np.nan
            r["iapf_posterior"] = iapf_val
            sess_iapfs.append(iapf_val)
        iapf_all[ckey] = sess_iapfs

    torch.save(sp, DATA_DIR / "specparam_canonical_7cohorts.pt")
    print("✓ Loaded all 8 cohort splits (737 sessions, 0 failed specparam fits), attached ages & posterior IAPF.")
    return sp, psds


# ==============================================================================
# MODULE A: BIOPHYSICS, MASTER RESULTS JSON, AGE ANCOVA, PAIRED MED, IAPF
# ==============================================================================
def compute_group_contrast(cases, ctrls, ext_fn, paired: bool = False):
    """Compute standardized effect size, analytical CI, t-test, and Satterthwaite df."""
    vc = np.array([ext_fn(g) for g in cases], dtype=float)
    vr = np.array([ext_fn(g) for g in ctrls], dtype=float)
    nc, nr = len(vc), len(vr)
    if paired:
        diff = vc - vr
        d = float(np.mean(diff) / np.std(diff, ddof=1))
        se = float(np.sqrt(1.0 / nc + (d ** 2) / (2.0 * nc)))
        t_stat, p_val = stats.ttest_rel(vc, vr)
        _, w_p = wilcoxon(vc, vr)
        return {
            "cohens_d": d,
            "ci_lower": d - 1.96 * se,
            "ci_upper": d + 1.96 * se,
            "paired_t": float(t_stat),
            "paired_p": float(p_val),
            "wilcoxon_p": float(w_p),
            "t_statistic": float(t_stat),
            "df_satterthwaite": float(nc - 1),
            "p_value": float(p_val),
            "mean_case": float(np.mean(vc)),
            "mean_ctrl": float(np.mean(vr)),
            "mean_diff": float(np.mean(diff)),
            "n_case": nc,
            "n_ctrl": nr,
            "n_pairs": nc,
        }
    sp_sd = np.sqrt(((nc - 1) * np.var(vc, ddof=1) + (nr - 1) * np.var(vr, ddof=1)) / (nc + nr - 2))
    d = float((np.mean(vc) - np.mean(vr)) / sp_sd)
    se = float(np.sqrt((nc + nr) / (nc * nr) + (d ** 2) / (2.0 * (nc + nr))))
    t_w, p_w = stats.ttest_ind(vc, vr, equal_var=False)
    t_s, p_s = stats.ttest_ind(vc, vr, equal_var=True)
    return {
        "cohens_d": d,
        "ci_lower": d - 1.96 * se,
        "ci_upper": d + 1.96 * se,
        "welch_t": float(t_w),
        "welch_p": float(p_w),
        "student_t": float(t_s),
        "student_p": float(p_s),
        "t_statistic": float(t_w),
        "df_satterthwaite": welch_df(vc, vr),
        "p_value": float(p_w),
        "mean_case": float(np.mean(vc)),
        "mean_ctrl": float(np.mean(vr)),
        "n_case": nc,
        "n_ctrl": nr,
    }


def make_ucsd_subj_mean(ucsd):
    """Average repeated PD sessions (ON + OFF) per participant for independent N=15 vs N=16 test."""
    by_bid = {}
    for g in ucsd:
        if "PD" in g.get("group", ""):
            by_bid.setdefault(g["bids_id"], []).append(g)
    pd_mean = []
    for bid in sorted(by_bid.keys()):
        pair = by_bid[bid]
        pd_mean.append({
            "bids_id": bid,
            "group": "PD_SubjMean",
            "x": torch.stack([p["x"] for p in pair]).mean(dim=0),
            "gce": float(np.mean([float(p["gce"]) for p in pair])),
            "age": float(pair[0].get("age", np.nan)),
            "iapf_posterior": float(np.nanmean([p.get("iapf_posterior", np.nan) for p in pair])),
        })
    return pd_mean, [g for g in ucsd if g.get("group") == "HC"]


def compute_all_regressions(dem, iowa):
    """Dynamically compute all 20 5-fold CV clinical severity regression benchmarks."""
    dem_pts = [g for g in dem if g.get("diagnosis") in ("Alzheimer", "FTD")]
    iowa_pts = [g for g in iowa if g.get("group") == "PD"]
    tasks = [
        ("Dementia_MMSE_All_N88", "**Study 5: Dementia All (ds004504)**", "MMSE (4.0–30.0)", dem, "mmse"),
        ("Dementia_MMSE_Patients_Only_N59", "**Study 5: Dementia Patients Only**", "MMSE (4.0–29.0)", dem_pts, "mmse"),
        ("Iowa_PD_MoCA_All_N149", "**Study 7: Iowa PD All (ds004584)**", "MoCA (9.0–30.0)", iowa, "moca"),
        ("Iowa_PD_MoCA_Patients_Only_N100", "**Study 7: Iowa PD Patients Only**", "MoCA (9.0–30.0)", iowa_pts, "moca"),
        ("Iowa_PD_UPDRS_Patients_N100", "**Study 7: Iowa PD Patients Only**", "UPDRS-III (1.0–28.0)", iowa_pts, "updrs"),
    ]
    reg_json = {}
    reg_rows = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for jkey, c_lbl, sc_lbl, glist, s_key in tasks:
        valid = [g for g in glist if pd.notnull(g.get(s_key))]
        y = np.array([float(g[s_key]) for g in valid])
        X = np.stack([g["x"].numpy().flatten() for g in valid])
        models = [
            ("GBDT_Regressor", "GBDT", HistGradientBoostingRegressor(max_iter=100, max_depth=4, random_state=42)),
            ("Random_Forest_Regressor", "RF", RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)),
            ("SVR_RBF", "SVR", SVR(C=1.0, epsilon=0.1)),
            ("Ridge_Regression", "Ridge", Ridge(alpha=10.0)),
        ]
        oof = {m_csv: np.zeros(len(valid)) for m_csv, _, _ in models}
        for tr, va in kf.split(X, y):
            mu = X[tr].mean(axis=0, keepdims=True)
            sd = X[tr].std(axis=0, keepdims=True) + 1e-6
            Xtr, Xva = (X[tr] - mu) / sd, (X[va] - mu) / sd
            for m_csv, _, est in models:
                est.fit(Xtr, y[tr])
                oof[m_csv][va] = est.predict(Xva)
        reg_json[jkey] = {"n": len(valid), "models": {}}
        for m_csv, m_short, _ in models:
            r, p = pearsonr(y, oof[m_csv])
            r2 = r2_score(y, oof[m_csv])
            mae = mean_absolute_error(y, oof[m_csv])
            reg_json[jkey]["models"][m_short] = {"r": float(r), "p": float(p), "r2": float(r2), "mae": float(mae)}
            reg_rows.append({
                "cohort": c_lbl,
                "cohort_key": jkey,
                "scale": sc_lbl,
                "scale_key": s_key,
                "n_samples": len(valid),
                "model": m_csv,
                "pearson_r": float(r),
                "p_value": float(p),
                "r2_score": float(r2),
                "mae": float(mae),
            })
    pd.DataFrame(reg_rows).to_csv(REPORTS_DIR / "multicohort_clinical_regression_benchmarks.csv", index=False)
    return reg_json


def run_module_a(sp):
    """Export master_results.json, 50-test FDR CSV, Age ANCOVA, Paired Med, and IAPF."""
    print("\n=== Module A: Biophysics, Master JSON, Age ANCOVA, Paired Med, IAPF ===")
    sleep = sp["sleep_ds004902"]
    mum_ec = sp["mumtaz_ec"]
    mum_eo = sp["mumtaz_eo"]
    cav = sp["cavanagh_ds003478"]
    ucsd = sp["ucsd_ds002778"]
    dem = sp["dementia_ds004504"]
    unm = sp["unm_ds003490"]
    iowa = sp["iowa_ds004584"]

    all_primary = sleep + mum_ec + cav + ucsd + dem + unm + iowa
    assert len(all_primary) == 677
    omst_trees = [g["omst_trees"] for g in all_primary]
    gces = [g["gce"] for g in all_primary]

    accounting = {
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
            "Study 6 (UNM PD ds003490)": {"enrolled": 50, "excluded": 0, "analyzed_pts": 50, "analyzed_sessions": 75, "design": "25 PD (paired ON/OFF) + 25 HC"},
            "Study 7 (Iowa PD ds004584)": {"enrolled": 149, "excluded": 0, "analyzed_pts": 149, "analyzed_sessions": 149, "design": "Between-subject (100 PD, 49 HC)"},
        },
    }
    omst_summary = {
        "total_graphs": 677,
        "total_node_instances": 677 * 19,
        "mean_trees": float(np.mean(omst_trees)),
        "sd_trees": float(np.std(omst_trees)),
        "pct_tree_1": float(np.mean([1 if t == 1 else 0 for t in omst_trees]) * 100),
        "pct_tree_2": float(np.mean([1 if t >= 2 else 0 for t in omst_trees]) * 100),
        "mean_gce": float(np.mean(gces)),
        "sd_gce": float(np.std(gces)),
        "isolated_node_rate_pct": 0.0,
    }

    extractors = {
        "frontal_theta_5ch": lambda g: float(np.mean(g["x"].numpy()[FOCAL_5, 0])),
        "frontal_theta_3ch_no_fp": lambda g: float(np.mean(g["x"].numpy()[FRONTAL_3, 0])),
        "global_theta": lambda g: float(np.mean(g["x"].numpy()[:, 0])),
        "global_delta": lambda g: float(np.mean(g["x"].numpy()[:, 1])),
        "posterior_alpha": lambda g: float(np.mean(g["x"].numpy()[POSTERIOR_5, 2])),
        "global_alpha": lambda g: float(np.mean(g["x"].numpy()[:, 2])),
        "global_chi": lambda g: float(np.mean(g["x"].numpy()[:, 3])),
        "omst_gce": lambda g: float(g.get("gce", 0.0)),
    }

    ucsd_pd_mean, ucsd_hc = make_ucsd_subj_mean(ucsd)
    contrasts = {
        "Sleep_ds004902_SD_vs_NS": (True, sorted([g for g in sleep if g["condition"] == "SD"], key=lambda g: g["subject_id"]), sorted([g for g in sleep if g["condition"] == "NS"], key=lambda g: g["subject_id"])),
        "Mumtaz_MDD_EC_vs_HC": (False, [g for g in mum_ec if g["group"] == "MDD"], [g for g in mum_ec if g["group"] == "HC"]),
        "Mumtaz_MDD_EO_vs_HC": (False, [g for g in mum_eo if g["group"] == "MDD"], [g for g in mum_eo if g["group"] == "HC"]),
        "Cavanagh_DEP_vs_HC": (False, [g for g in cav if g["group"] == "Depressed"], [g for g in cav if g["group"] == "Control"]),
        "UCSD_PD_All_vs_HC": (False, [g for g in ucsd if "PD" in g.get("group", "")], ucsd_hc),
        "UCSD_PD_SubjMean_vs_HC": (False, ucsd_pd_mean, ucsd_hc),
        "UCSD_PD_OFF_vs_HC": (False, [g for g in ucsd if g.get("group") == "PD_OFF"], ucsd_hc),
        "UCSD_PD_ON_vs_HC": (False, [g for g in ucsd if g.get("group") == "PD_ON"], ucsd_hc),
        "UCSD_PD_ON_vs_OFF_Paired": (True, sorted([g for g in ucsd if g.get("group") == "PD_ON"], key=lambda g: g["bids_id"]), sorted([g for g in ucsd if g.get("group") == "PD_OFF"], key=lambda g: g["bids_id"])),
        "Dementia_AD_vs_HC": (False, [g for g in dem if g["diagnosis"] == "Alzheimer"], [g for g in dem if g["diagnosis"] == "Control"]),
        "Dementia_FTD_vs_HC": (False, [g for g in dem if g["diagnosis"] == "FTD"], [g for g in dem if g["diagnosis"] == "Control"]),
        "Dementia_All_vs_HC": (False, [g for g in dem if g["diagnosis"] in ("Alzheimer", "FTD")], [g for g in dem if g["diagnosis"] == "Control"]),
        "UNM_PD_OFF_vs_HC": (False, [g for g in unm if g["group"] == "PD" and g["med_state"] == "OFF"], [g for g in unm if g["group"] == "CTL"]),
        "UNM_PD_ON_vs_HC": (False, [g for g in unm if g["group"] == "PD" and g["med_state"] == "ON"], [g for g in unm if g["group"] == "CTL"]),
        "UNM_PD_ON_vs_OFF_Paired": (True, sorted([g for g in unm if g["group"] == "PD" and g["med_state"] == "ON"], key=lambda g: g["subject_id"]), sorted([g for g in unm if g["group"] == "PD" and g["med_state"] == "OFF"], key=lambda g: g["subject_id"])),
        "Iowa_PD_vs_HC": (False, [g for g in iowa if g["group"] == "PD"], [g for g in iowa if g["group"] == "Control"]),
    }

    biophys = {}
    for cname, (is_paired, cases, ctrls) in contrasts.items():
        biophys[cname] = {fname: compute_group_contrast(cases, ctrls, fn, paired=is_paired) for fname, fn in extractors.items()}

    # Build 10-contrast x 5-marker expanded CSV (exact q_10 + q_50)
    comparisons_10 = [
        ("Study 1: Acute Sleep Deprivation", "ds004902", "Sleep_ds004902_SD_vs_NS", "SD", "NS", True),
        ("Study 2: Outpatient MDD", "4244171", "Mumtaz_MDD_EC_vs_HC", "MDD", "Control", False),
        ("Study 3: Univ. Depression Screening", "ds003478", "Cavanagh_DEP_vs_HC", "Depressive-Symptom", "Control", False),
        ("Study 4: UCSD Parkinson All", "ds002778", "UCSD_PD_SubjMean_vs_HC", "PD (Participant Mean)", "Control", False),
        ("Study 5a: Alzheimer's Disease (AD)", "ds004504", "Dementia_AD_vs_HC", "AD", "Control", False),
        ("Study 5b: Frontotemporal Dementia (FTD)", "ds004504", "Dementia_FTD_vs_HC", "FTD", "Control", False),
        ("Study 5c: Clinical Dementia (All)", "ds004504", "Dementia_All_vs_HC", "Dementia", "Control", False),
        ("Study 6a: UNM Parkinson OFF", "ds003490", "UNM_PD_OFF_vs_HC", "PD_OFF", "Control", False),
        ("Study 6b: UNM Parkinson ON", "ds003490", "UNM_PD_ON_vs_HC", "PD_ON", "Control", False),
        ("Study 7: Iowa Parkinson Cohort", "ds004584", "Iowa_PD_vs_HC", "PD", "Control", False),
    ]
    marker_defs = [
        ("frontal_theta", "frontal_theta_5ch", "Frontal Periodic Theta (4–8 Hz)"),
        ("global_delta", "global_delta", "Global Periodic Delta (1–4 Hz)"),
        ("posterior_alpha", "posterior_alpha", "Posterior Periodic Alpha (8–13 Hz)"),
        ("global_chi", "global_chi", "Global Aperiodic Exponent (χ)"),
        ("omst_gce", "omst_gce", "OMST Global Cost Efficiency (GCE)"),
    ]
    rec_50 = []
    for clabel, dsid, bkey, cname, rname, is_paired in comparisons_10:
        for mkey, bfeat, mlabel in marker_defs:
            r = biophys[bkey][bfeat]
            rec_50.append({
                "cohort": clabel,
                "dataset_id": dsid,
                "case_group": cname,
                "ref_group": rname,
                "n_case": int(r["n_case"]),
                "n_control": int(r["n_ctrl"]),
                "is_paired": is_paired,
                "marker_key": mkey,
                "marker_label": mlabel,
                "mean_case": r["mean_case"],
                "mean_control": r["mean_ctrl"],
                "cohens_d": r["cohens_d"],
                "ci_lower": r["ci_lower"],
                "ci_upper": r["ci_upper"],
                "t_statistic": r["t_statistic"],
                "df_satterthwaite": r["df_satterthwaite"],
                "p_value": r["p_value"],
            })
    df_50 = pd.DataFrame(rec_50)
    _, df_50["fdr_q_50"], _, _ = multipletests(df_50["p_value"].values, method="fdr_bh")
    # Also compute q_10 per marker family cleanly
    q10_map = {}
    for mkey, _, _ in marker_defs:
        mask = df_50["marker_key"] == mkey
        _, q10, _, _ = multipletests(df_50.loc[mask, "p_value"].values, method="fdr_bh")
        for idx_r, qv in zip(df_50.index[mask], q10):
            q10_map[idx_r] = float(qv)
    df_50["fdr_q_10"] = [q10_map[i] for i in df_50.index]
    df_50.to_csv(REPORTS_DIR / "multicohort_biophysical_effect_sizes_expanded.csv", index=False)

    # Complete 5-cohort independent-sample Age ANCOVA
    ancova_contrasts = [
        ("Study 3: Univ. Depression Screening (ds003478)", cav),
        ("Study 4a: UCSD Parkinson OFF vs. HC (ds002778)", [g for g in ucsd if g["group"] in ("PD_OFF", "HC")]),
        ("Study 4b: UCSD Parkinson ON vs. HC (ds002778)", [g for g in ucsd if g["group"] in ("PD_ON", "HC")]),
        ("Study 5: AHEPA Dementia All vs. HC (ds004504)", dem),
        ("Study 6a: UNM Parkinson OFF vs. HC (ds003490)", [g for g in unm if g["med_state"] in ("OFF", "CTL")]),
        ("Study 6b: UNM Parkinson ON vs. HC (ds003490)", [g for g in unm if g["med_state"] in ("ON", "CTL")]),
        ("Study 7: Iowa Parkinson vs. HC (ds004584)", iowa),
    ]
    ancova_rows = []
    for clabel, glist in ancova_contrasts:
        valid = [g for g in glist if g.get("age") is not None and not np.isnan(float(g["age"]))]
        y_grp = np.array([1.0 if int(g["y"]) == 1 else 0.0 for g in valid])
        age = np.array([float(g["age"]) for g in valid])
        X = sm.add_constant(np.column_stack([y_grp, age]))
        for _, bfeat, mlabel in marker_defs[:4]:
            vals = np.array([extractors[bfeat](g) for g in valid])
            res = sm.OLS(vals, X).fit()
            ancova_rows.append({
                "cohort": clabel,
                "n_valid_age": len(valid),
                "marker": mlabel,
                "group_coef": float(res.params[1]),
                "group_se": float(res.bse[1]),
                "group_t": float(res.tvalues[1]),
                "group_p": float(res.pvalues[1]),
                "age_coef": float(res.params[2]),
                "age_se": float(res.bse[2]),
                "age_t": float(res.tvalues[2]),
                "age_p": float(res.pvalues[2]),
                "r_squared": float(res.rsquared),
            })
    pd.DataFrame(ancova_rows).to_csv(REPORTS_DIR / "multicohort_age_ancova_complete.csv", index=False)

    # Paired dopaminergic medication effects (UNM N=25 & UCSD N=15)
    med_rows = []
    med_sets = [
        ("Study 6: UNM Parkinson (≥15h Overnight Washout)",
         sorted([g for g in unm if g["group"] == "PD" and g["med_state"] == "ON"], key=lambda g: g["subject_id"]),
         sorted([g for g in unm if g["group"] == "PD" and g["med_state"] == "OFF"], key=lambda g: g["subject_id"])),
        ("Study 4: UCSD Parkinson (Acute Medication Comparison)",
         sorted([g for g in ucsd if g.get("group") == "PD_ON"], key=lambda g: g["bids_id"]),
         sorted([g for g in ucsd if g.get("group") == "PD_OFF"], key=lambda g: g["bids_id"])),
    ]
    med_markers = [
        ("Frontal Periodic Theta (4–8 Hz)", "frontal_theta_5ch"),
        ("Global Periodic Theta (4–8 Hz)", "global_theta"),
        ("Global Periodic Delta (1–4 Hz)", "global_delta"),
        ("Posterior Periodic Alpha (8–13 Hz)", "posterior_alpha"),
        ("Global Periodic Alpha (8–13 Hz)", "global_alpha"),
        ("Global Aperiodic Exponent (χ)", "global_chi"),
        ("OMST Global Cost Efficiency (GCE)", "omst_gce"),
    ]
    for clabel, on_g, off_g in med_sets:
        n = len(on_g)
        tcrit = stats.t.ppf(0.975, df=n - 1)
        for mlabel, ek in med_markers:
            von = np.array([extractors[ek](g) for g in on_g])
            voff = np.array([extractors[ek](g) for g in off_g])
            diff = von - voff
            md = float(np.mean(diff))
            sd = float(np.std(diff, ddof=1))
            se = sd / np.sqrt(n)
            dz = md / sd
            se_dz = float(np.sqrt(1.0 / n + (dz ** 2) / (2.0 * n)))
            t_stat, p_val = stats.ttest_rel(von, voff)
            _, w_p = wilcoxon(diff)
            med_rows.append({
                "cohort": clabel,
                "n_pairs": n,
                "marker": mlabel,
                "mean_off": float(np.mean(voff)),
                "mean_on": float(np.mean(von)),
                "mean_diff_on_minus_off": md,
                "diff_ci_lower": md - tcrit * se,
                "diff_ci_upper": md + tcrit * se,
                "cohens_dz": dz,
                "dz_ci_lower": dz - 1.96 * se_dz,
                "dz_ci_upper": dz + 1.96 * se_dz,
                "paired_t": float(t_stat),
                "paired_p": float(p_val),
                "wilcoxon_p": float(w_p),
            })
    pd.DataFrame(med_rows).to_csv(REPORTS_DIR / "unm_ucsd_paired_medication_biophysics.csv", index=False)

    # Posterior IAPF Summary (7.0-13.5 Hz peak center frequency)
    iapf_rows = []
    for clabel, dsid, bkey, cname, rname, is_paired in comparisons_10:
        _, cases, ctrls = contrasts[bkey]
        vc = np.array([g["iapf_posterior"] for g in cases if pd.notnull(g.get("iapf_posterior"))])
        vr = np.array([g["iapf_posterior"] for g in ctrls if pd.notnull(g.get("iapf_posterior"))])
        if len(vc) >= 5 and len(vr) >= 5:
            sp_sd = np.sqrt(((len(vc) - 1) * np.var(vc, ddof=1) + (len(vr) - 1) * np.var(vr, ddof=1)) / (len(vc) + len(vr) - 2))
            d_val = (np.mean(vc) - np.mean(vr)) / sp_sd
            t_v, p_v = stats.ttest_rel(vc, vr) if (is_paired and len(vc) == len(vr)) else stats.ttest_ind(vc, vr, equal_var=False)
            iapf_rows.append({
                "cohort": clabel,
                "n_case_valid": len(vc),
                "n_ctrl_valid": len(vr),
                "mean_iapf_case_hz": float(np.mean(vc)),
                "sd_iapf_case_hz": float(np.std(vc, ddof=1)),
                "mean_iapf_ctrl_hz": float(np.mean(vr)),
                "sd_iapf_ctrl_hz": float(np.std(vr, ddof=1)),
                "diff_hz": float(np.mean(vc) - np.mean(vr)),
                "cohens_d": float(d_val),
                "p_value": float(p_v),
            })
    pd.DataFrame(iapf_rows).to_csv(REPORTS_DIR / "iapf_biophysics_summary.csv", index=False)

    # Dynamic regressions + Master JSON
    reg_json = compute_all_regressions(dem, iowa)
    master_payload = {
        "spectral_backend": "specparam_v2.0.0rc7_resid",
        "accounting": accounting,
        "omst_topology": omst_summary,
        "biophysics": biophys,
        "regressions": reg_json,
    }
    with open(REPORTS_DIR / "master_results.json", "w", encoding="utf-8") as f:
        json.dump(master_payload, f, indent=2)
    print("✓ Module A complete: master_results.json, 50-test FDR, Age ANCOVA, Paired Med, IAPF, 20 Regressions.")
    return master_payload


# ==============================================================================
# MODULE B: WITHIN-COHORT 12-MODEL BENCHMARK, LOFO ABLATIONS, WILCOXON, EO
# ==============================================================================
def get_within_cohort_specs(sp):
    """Canonical list of 11 primary/medication contrasts plus Study 2b EO."""
    sleep = sp["sleep_ds004902"]
    mum_ec = sp["mumtaz_ec"]
    mum_eo = sp["mumtaz_eo"]
    cav = sp["cavanagh_ds003478"]
    ucsd = sp["ucsd_ds002778"]
    dem = sp["dementia_ds004504"]
    unm = sp["unm_ds003490"]
    iowa = sp["iowa_ds004584"]
    return [
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


def eval_12models_and_lofo_single_cohort(cname, glist, use_grp):
    """Run all 12 models + LOFO + Single-feature baselines for one cohort."""
    set_seed(42)
    vg = [g for g in glist if int(g["y"]) in (0, 1)]
    y_all = np.array([int(g["y"]) for g in vg])
    n = len(vg)
    groups = [g.get("bids_id", g.get("sub_name", g.get("subject_id", str(i)))) for i, g in enumerate(vg)]
    if use_grp:
        splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(vg, y_all, groups=groups))
    else:
        splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(vg, y_all))

    models_12 = [
        "GCN_OMST", "GCN_Identity", "GCN_Random", "GCN_Full",
        "Tuned_LogisticRegression", "Tuned_Ridge", "Linear_SVM", "RBF_SVM",
        "RandomForest", "GBDT", "Single_Theta", "Single_Chi",
    ]
    oof_p12 = {m: np.zeros(n) for m in models_12}
    oof_y12 = {m: np.zeros(n) for m in models_12}
    oof_lofo = {k: np.zeros(n) for k in [
        "Single_Theta", "Single_Delta", "Single_Alpha", "Single_Chi", "Single_GCE",
        "GCN_Full4", "GCN_Ablate_Theta", "GCN_Ablate_Delta", "GCN_Ablate_Alpha", "GCN_Ablate_Chi",
    ]}

    adj_id = torch.eye(19, dtype=torch.float32)
    adj_rnd = make_random_adj(19, 18, seed=42)
    adj_full = torch.ones((19, 19), dtype=torch.float32) / 19.0

    for fold, (tr, va) in enumerate(splits):
        tx = torch.stack([vg[i]["x"] for i in tr])
        tadj = torch.stack([vg[i]["adj"] for i in tr])
        ty = torch.tensor([int(vg[i]["y"]) for i in tr], dtype=torch.long)
        vx = torch.stack([vg[i]["x"] for i in va])
        vadj = torch.stack([vg[i]["adj"] for i in va])

        tz, vz, _, _ = compute_fold_normative_z(tx, ty, vx)
        X_tr, X_va = tz.view(len(tr), -1).numpy(), vz.view(len(va), -1).numpy()
        y_tr = ty.numpy()

        # 4 Graph topologies
        oof_p12["GCN_OMST"][va], oof_y12["GCN_OMST"][va] = train_eval_gcn(tz, tadj, ty, vz, vadj, seed=42 + fold)
        oof_lofo["GCN_Full4"][va] = oof_p12["GCN_OMST"][va]
        oof_p12["GCN_Identity"][va], oof_y12["GCN_Identity"][va] = train_eval_gcn(
            tz, adj_id.unsqueeze(0).expand(len(tr), -1, -1), ty,
            vz, adj_id.unsqueeze(0).expand(len(va), -1, -1), seed=42 + fold,
        )
        oof_p12["GCN_Random"][va], oof_y12["GCN_Random"][va] = train_eval_gcn(
            tz, adj_rnd.unsqueeze(0).expand(len(tr), -1, -1), ty,
            vz, adj_rnd.unsqueeze(0).expand(len(va), -1, -1), seed=42 + fold,
        )
        oof_p12["GCN_Full"][va], oof_y12["GCN_Full"][va] = train_eval_gcn(
            tz, adj_full.unsqueeze(0).expand(len(tr), -1, -1), ty,
            vz, adj_full.unsqueeze(0).expand(len(va), -1, -1), seed=42 + fold,
        )

        # Leak-free inner CV (StratifiedGroupKFold when outer is grouped!)
        if use_grp:
            tr_grps = [groups[i] for i in tr]
            inner_cv = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(X_tr, y_tr, groups=tr_grps))
        else:
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        gs_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}, cv=inner_cv, scoring="roc_auc").fit(X_tr, y_tr)
        oof_p12["Tuned_LogisticRegression"][va] = gs_lr.predict_proba(X_va)[:, 1]
        oof_y12["Tuned_LogisticRegression"][va] = gs_lr.predict(X_va)

        gs_rc = GridSearchCV(RidgeClassifier(random_state=42), {"alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}, cv=inner_cv, scoring="roc_auc").fit(X_tr, y_tr)
        oof_p12["Tuned_Ridge"][va] = 1.0 / (1.0 + np.exp(-gs_rc.decision_function(X_va)))
        oof_y12["Tuned_Ridge"][va] = gs_rc.predict(X_va)

        gs_lsvm = GridSearchCV(SVC(kernel="linear", probability=True, random_state=42), {"C": [0.01, 0.1, 1.0, 10.0]}, cv=inner_cv, scoring="roc_auc").fit(X_tr, y_tr)
        oof_p12["Linear_SVM"][va] = gs_lsvm.predict_proba(X_va)[:, 1]
        oof_y12["Linear_SVM"][va] = gs_lsvm.predict(X_va)

        gs_rbf = GridSearchCV(SVC(kernel="rbf", probability=True, random_state=42), {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto", 0.01, 0.1]}, cv=inner_cv, scoring="roc_auc").fit(X_tr, y_tr)
        oof_p12["RBF_SVM"][va] = gs_rbf.predict_proba(X_va)[:, 1]
        oof_y12["RBF_SVM"][va] = gs_rbf.predict(X_va)

        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, y_tr)
        oof_p12["RandomForest"][va], oof_y12["RandomForest"][va] = rf.predict_proba(X_va)[:, 1], rf.predict(X_va)

        gbdt = HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=42).fit(X_tr, y_tr)
        oof_p12["GBDT"][va], oof_y12["GBDT"][va] = gbdt.predict_proba(X_va)[:, 1], gbdt.predict(X_va)

        # Univariate baselines
        th_tr, th_va = tz[:, FOCAL_5, 0].mean(dim=-1, keepdim=True).numpy(), vz[:, FOCAL_5, 0].mean(dim=-1, keepdim=True).numpy()
        lr_th = LogisticRegression(random_state=42).fit(th_tr, y_tr)
        oof_p12["Single_Theta"][va], oof_y12["Single_Theta"][va] = lr_th.predict_proba(th_va)[:, 1], lr_th.predict(th_va)
        oof_lofo["Single_Theta"][va] = oof_p12["Single_Theta"][va]

        ch_tr, ch_va = tz[:, :, 3].mean(dim=-1, keepdim=True).numpy(), vz[:, :, 3].mean(dim=-1, keepdim=True).numpy()
        lr_ch = LogisticRegression(random_state=42).fit(ch_tr, y_tr)
        oof_p12["Single_Chi"][va], oof_y12["Single_Chi"][va] = lr_ch.predict_proba(ch_va)[:, 1], lr_ch.predict(ch_va)
        oof_lofo["Single_Chi"][va] = oof_p12["Single_Chi"][va]

        dl_tr, dl_va = tz[:, :, 1].mean(dim=-1, keepdim=True).numpy(), vz[:, :, 1].mean(dim=-1, keepdim=True).numpy()
        oof_lofo["Single_Delta"][va] = LogisticRegression(random_state=42).fit(dl_tr, y_tr).predict_proba(dl_va)[:, 1]

        al_tr, al_va = tz[:, POSTERIOR_5, 2].mean(dim=-1, keepdim=True).numpy(), vz[:, POSTERIOR_5, 2].mean(dim=-1, keepdim=True).numpy()
        oof_lofo["Single_Alpha"][va] = LogisticRegression(random_state=42).fit(al_tr, y_tr).predict_proba(al_va)[:, 1]

        gce_tr = np.array([vg[i]["gce"] for i in tr]).reshape(-1, 1)
        gce_va = np.array([vg[i]["gce"] for i in va]).reshape(-1, 1)
        mu_g, sd_g = gce_tr[y_tr == 0].mean(), gce_tr[y_tr == 0].std() + 1e-6
        oof_lofo["Single_GCE"][va] = LogisticRegression(random_state=42).fit((gce_tr - mu_g) / sd_g, y_tr).predict_proba((gce_va - mu_g) / sd_g)[:, 1]

        # LOFO 3-feature GCNs
        for key_ab, idxs in [
            ("GCN_Ablate_Theta", [1, 2, 3]),
            ("GCN_Ablate_Delta", [0, 2, 3]),
            ("GCN_Ablate_Alpha", [0, 1, 3]),
            ("GCN_Ablate_Chi", [0, 1, 2]),
        ]:
            oof_lofo[key_ab][va], _ = train_eval_gcn(tz[:, :, idxs], tadj, ty, vz[:, :, idxs], vadj, in_dim=3, seed=42 + fold)

    rows_12 = []
    for m in models_12:
        auc, lo, hi = bootstrap_auc_ci(y_all, oof_p12[m])
        rows_12.append({
            "cohort": cname,
            "model": m,
            "auc": auc,
            "ci_lower": lo,
            "ci_upper": hi,
            "accuracy": float(accuracy_score(y_all, oof_y12[m])),
            "balanced_accuracy": float(balanced_accuracy_score(y_all, oof_y12[m])),
        })
    row_lofo = {"cohort": cname, "n_samples": n}
    for k, probs in oof_lofo.items():
        a, l, u = bootstrap_auc_ci(y_all, probs)
        row_lofo[f"{k}_auc"] = a
        row_lofo[f"{k}_ci_low"] = l
        row_lofo[f"{k}_ci_high"] = u
    lr_r = [r for r in rows_12 if r["model"] == "Tuned_LogisticRegression"][0]
    row_lofo["Tuned_LR76_auc"] = lr_r["auc"]
    row_lofo["Tuned_LR76_ci_low"] = lr_r["ci_lower"]
    row_lofo["Tuned_LR76_ci_high"] = lr_r["ci_upper"]
    return rows_12, row_lofo


def run_module_b(sp):
    """Run all 12 models + LOFO + Wilcoxon + Study 2b Mumtaz EO."""
    print("\n=== Module B: Within-Cohort 12-Model CV, LOFO, Wilcoxon, Mumtaz EO ===")
    specs = get_within_cohort_specs(sp)
    all_12, all_lofo, eo_rows = [], [], []
    for cname, glist, use_grp in specs:
        r12, rlofo = eval_12models_and_lofo_single_cohort(cname, glist, use_grp)
        if cname == "Study 2b: Mumtaz MDD (Eyes Open)":
            for r in r12:
                r_eo = dict(r)
                r_eo["cohort"] = "Study 2b: Mumtaz Outpatient MDD (Eyes-Open)"
                eo_rows.append(r_eo)
            all_12.extend(r12)
        else:
            all_12.extend(r12)
            all_lofo.append(rlofo)
        gcn_a = [r["auc"] for r in r12 if r["model"] == "GCN_OMST"][0]
        rf_a = [r["auc"] for r in r12 if r["model"] == "RandomForest"][0]
        lr_a = [r["auc"] for r in r12 if r["model"] == "Tuned_LogisticRegression"][0]
        print(f"  {cname:<42} | GCN={gcn_a:.3f} | LR={lr_a:.3f} | RF={rf_a:.3f}")

    df_ext = pd.DataFrame(all_12)
    df_ext.to_csv(REPORTS_DIR / "multicohort_7cohort_extended_model_comparison.csv", index=False)
    pd.DataFrame(eo_rows).to_csv(REPORTS_DIR / "mumtaz_eo_model_comparison.csv", index=False)
    pd.DataFrame(all_lofo).to_csv(REPORTS_DIR / "multicohort_7cohort_feature_ablation.csv", index=False)

    # Wilcoxon on primary 9 and 11 contrasts (excluding Study 2b EO so Table 2 / Table S11 match)
    df_11 = df_ext[df_ext["cohort"] != "Study 2b: Mumtaz MDD (Eyes Open)"].copy()
    piv = df_11.pivot(index="cohort", columns="model", values="auc")
    primary_9 = [c for c in piv.index if c not in ("Study 6b: UNM PD-ON vs Control", "Study 6c: UNM PD All Sessions")]
    others = [
        "GCN_Identity", "GCN_Random", "GCN_Full",
        "Tuned_LogisticRegression", "Tuned_Ridge", "Linear_SVM", "RBF_SVM",
        "RandomForest", "GBDT", "Single_Theta", "Single_Chi",
    ]
    w_rows = []
    for sname, idxs in [("Primary_9_Contrasts", primary_9), ("All_11_Contrasts", list(piv.index))]:
        p_list, tmp = [], []
        for m in others:
            x, y = piv.loc[idxs, "GCN_OMST"].values, piv.loc[idxs, m].values
            diff = x - y
            w_stat, p_val = wilcoxon(x, y)
            p_list.append(p_val)
            tmp.append({
                "scope": sname,
                "n_benchmarks": len(idxs),
                "comparator_model": m,
                "gcn_mean_auc": float(np.mean(x)),
                "comp_mean_auc": float(np.mean(y)),
                "mean_diff_auc": float(np.mean(diff)),
                "median_diff_auc": float(np.median(diff)),
                "wins_ties_losses": f"{int(np.sum(diff > 1e-6))} / {int(np.sum(np.abs(diff) <= 1e-6))} / {int(np.sum(diff < -1e-6))}",
                "wilcoxon_W": float(w_stat),
                "raw_p": float(p_val),
            })
        _, qvals, _, _ = multipletests(p_list, method="fdr_bh")
        for r, qv in zip(tmp, qvals):
            r["fdr_q"] = float(qv)
            w_rows.append(r)
    pd.DataFrame(w_rows).to_csv(REPORTS_DIR / "multicohort_wilcoxon_model_comparisons.csv", index=False)
    print("✓ Module B complete: 12-model CV, Study 2b EO, LOFO, Wilcoxon saved.")
    return df_11


# ==============================================================================
# MODULE C: 5-SEED REPEATED CV & GCN HYPERPARAMETER SENSITIVITY
# ==============================================================================
def run_module_c(sp, df_11):
    """Run 350 5-seed evaluations and 35 GCN hyperparameter grid runs."""
    print("\n=== Module C: 5-Seed Repeated CV (350 runs) & GCN Hyperparam Grid (35 runs) ===")
    sleep, mum_ec, cav = sp["sleep_ds004902"], sp["mumtaz_ec"], sp["cavanagh_ds003478"]
    ucsd, dem, unm, iowa = sp["ucsd_ds002778"], sp["dementia_ds004504"], sp["unm_ds003490"], sp["iowa_ds004584"]

    contrasts_10 = [
        ("Study 1: Sleep Deprivation (ds004902)", sleep, True),
        ("Study 2: Mumtaz MDD (Eyes Closed)", mum_ec, False),
        ("Study 3: Cavanagh University Depression", cav, False),
        ("Study 4: UCSD Parkinson All Sessions", ucsd, True),
        ("Study 5a: Dementia AD vs Control", [g for g in dem if g["diagnosis"] in ("Alzheimer", "Control")], False),
        ("Study 5b: Dementia FTD vs Control", [g for g in dem if g["diagnosis"] in ("FTD", "Control")], False),
        ("Study 5c: Dementia All (AD+FTD) vs Control", dem, False),
        ("Study 6a: UNM PD-OFF vs Control", [g for g in unm if g["med_state"] in ("OFF", "CTL")], False),
        ("Study 6b: UNM PD-ON vs Control", [g for g in unm if g["med_state"] in ("ON", "CTL")], False),
        ("Study 7: Iowa Parkinson (ds004584)", iowa, False),
    ]
    models_7 = [
        "GCN_OMST", "GCN_Identity", "GCN_Random",
        "Tuned_LogisticRegression", "Tuned_Ridge", "RandomForest", "Single_Theta",
    ]
    adj_id = torch.eye(19, dtype=torch.float32)
    records = []

    for cname, glist, use_grp in contrasts_10:
        vg = [g for g in glist if int(g["y"]) in (0, 1)]
        y_all = np.array([int(g["y"]) for g in vg])
        n = len(vg)
        groups = [g.get("bids_id", g.get("sub_name", g.get("subject_id", str(i)))) for i, g in enumerate(vg)]
        n_part = len(set(groups))
        for seed in SEEDS_5:
            set_seed(seed)
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed) if use_grp else StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            splits = list(splitter.split(vg, y_all, groups=groups)) if use_grp else list(splitter.split(vg, y_all))
            oof_p = {m: np.zeros(n) for m in models_7}
            oof_y = {m: np.zeros(n) for m in models_7}
            adj_rnd = make_random_adj(19, 18, seed=seed)

            for fold, (tr, va) in enumerate(splits):
                tx = torch.stack([vg[i]["x"] for i in tr])
                tadj = torch.stack([vg[i]["adj"] for i in tr])
                ty = torch.tensor([int(vg[i]["y"]) for i in tr], dtype=torch.long)
                vx = torch.stack([vg[i]["x"] for i in va])
                vadj = torch.stack([vg[i]["adj"] for i in va])
                tz, vz, _, _ = compute_fold_normative_z(tx, ty, vx)
                X_tr, X_va = tz.view(len(tr), -1).numpy(), vz.view(len(va), -1).numpy()
                y_tr = ty.numpy()

                oof_p["GCN_OMST"][va], oof_y["GCN_OMST"][va] = train_eval_gcn(tz, tadj, ty, vz, vadj, seed=seed + fold)
                oof_p["GCN_Identity"][va], oof_y["GCN_Identity"][va] = train_eval_gcn(
                    tz, adj_id.unsqueeze(0).expand(len(tr), -1, -1), ty,
                    vz, adj_id.unsqueeze(0).expand(len(va), -1, -1), seed=seed + fold,
                )
                oof_p["GCN_Random"][va], oof_y["GCN_Random"][va] = train_eval_gcn(
                    tz, adj_rnd.unsqueeze(0).expand(len(tr), -1, -1), ty,
                    vz, adj_rnd.unsqueeze(0).expand(len(va), -1, -1), seed=seed + fold,
                )

                if use_grp:
                    inner_cv = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed).split(X_tr, y_tr, groups=[groups[i] for i in tr]))
                else:
                    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

                gs_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=seed), {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}, cv=inner_cv, scoring="roc_auc").fit(X_tr, y_tr)
                oof_p["Tuned_LogisticRegression"][va], oof_y["Tuned_LogisticRegression"][va] = gs_lr.predict_proba(X_va)[:, 1], gs_lr.predict(X_va)

                gs_rc = GridSearchCV(RidgeClassifier(random_state=seed), {"alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}, cv=inner_cv, scoring="roc_auc").fit(X_tr, y_tr)
                oof_p["Tuned_Ridge"][va] = 1.0 / (1.0 + np.exp(-gs_rc.decision_function(X_va)))
                oof_y["Tuned_Ridge"][va] = gs_rc.predict(X_va)

                rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed).fit(X_tr, y_tr)
                oof_p["RandomForest"][va], oof_y["RandomForest"][va] = rf.predict_proba(X_va)[:, 1], rf.predict(X_va)

                th_tr, th_va = tz[:, FOCAL_5, 0].mean(dim=-1, keepdim=True).numpy(), vz[:, FOCAL_5, 0].mean(dim=-1, keepdim=True).numpy()
                lr_th = LogisticRegression(random_state=seed).fit(th_tr, y_tr)
                oof_p["Single_Theta"][va], oof_y["Single_Theta"][va] = lr_th.predict_proba(th_va)[:, 1], lr_th.predict(th_va)

            om_auc = float(roc_auc_score(y_all, oof_p["GCN_OMST"]))
            for m in models_7:
                auc_m = float(roc_auc_score(y_all, oof_p[m]))
                records.append({
                    "cohort": cname,
                    "seed": seed,
                    "model": m,
                    "n_sessions": n,
                    "n_participants": n_part,
                    "auc": auc_m,
                    "balanced_accuracy": float(balanced_accuracy_score(y_all, oof_y[m])),
                    "delta_vs_omst": auc_m - om_auc,
                })

    df_ms = pd.DataFrame(records)
    df_ms.to_csv(REPORTS_DIR / "benchmark_multiseed_repeated_cv.csv", index=False)

    # Assert seed=42 reproduces df_11 identically
    s42 = df_ms[df_ms["seed"] == 42].set_index(["cohort", "model"])["auc"]
    ref = df_11.set_index(["cohort", "model"])["auc"]
    for idx_k, v in s42.items():
        assert abs(v - float(ref.loc[idx_k])) < 1e-6, f"Seed=42 mismatch at {idx_k}: {v} vs {ref.loc[idx_k]}"

    sum_rows = []
    for cname, glist, _ in contrasts_10:
        for m in models_7:
            sub = df_ms[(df_ms["cohort"] == cname) & (df_ms["model"] == m)]
            sum_rows.append({
                "cohort": cname,
                "model": m,
                "n_sessions": int(sub["n_sessions"].iloc[0]),
                "n_participants": int(sub["n_participants"].iloc[0]),
                "seed42_auc": float(sub[sub["seed"] == 42]["auc"].iloc[0]),
                "mean_auc": float(sub["auc"].mean()),
                "std_auc": float(sub["auc"].std(ddof=1)),
                "min_auc": float(sub["auc"].min()),
                "max_auc": float(sub["auc"].max()),
                "mean_bacc": float(sub["balanced_accuracy"].mean()),
                "mean_delta_vs_omst": float(sub["delta_vs_omst"].mean()),
                "std_delta_vs_omst": float(sub["delta_vs_omst"].std(ddof=1)),
            })
    pd.DataFrame(sum_rows).to_csv(REPORTS_DIR / "benchmark_multiseed_summary.csv", index=False)

    # GCN Hyperparameter sensitivity grid (5 representative cohorts x 7 configs)
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
        ("Study 2: Mumtaz Outpatient MDD (EC)", mum_ec, False),
        ("Study 3: Cavanagh Univ. Depression (EO)", cav, False),
        ("Study 4: UCSD Parkinson All", ucsd, True),
        ("Study 6a: UNM PD-OFF vs CTL", [g for g in unm if g["med_state"] in ("OFF", "CTL")], False),
        ("Study 7: Iowa Parkinson", iowa, False),
    ]
    g_rows = []
    for cname, glist, is_grp in g_cohorts:
        vg = [g for g in glist if int(g["y"]) in (0, 1)]
        y_all = np.array([int(g["y"]) for g in vg])
        grps = [g.get("bids_id", g.get("sub_name", g.get("subject_id", str(i)))) for i, g in enumerate(vg)]
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) if is_grp else StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(splitter.split(vg, y_all, groups=grps if is_grp else None))
        for lr, epochs, hdim, clbl in grid:
            oof = np.zeros(len(vg))
            for fold, (tr, va) in enumerate(splits):
                tx = torch.stack([vg[i]["x"] for i in tr])
                vx = torch.stack([vg[i]["x"] for i in va])
                ty = torch.tensor([int(vg[i]["y"]) for i in tr], dtype=torch.long)
                tadj = torch.stack([vg[i]["adj"] for i in tr])
                vadj = torch.stack([vg[i]["adj"] for i in va])
                tz, vz, _, _ = compute_fold_normative_z(tx, ty, vx)
                oof[va], _ = train_eval_gcn(tz, tadj, ty, vz, vadj, hidden_dim=hdim, epochs=epochs, lr=lr, seed=42 + fold)
            g_rows.append({
                "cohort": cname,
                "n_samples": len(vg),
                "config_label": clbl,
                "learning_rate": lr,
                "epochs": epochs,
                "hidden_dim": hdim,
                "auc": float(roc_auc_score(y_all, oof)),
            })
    pd.DataFrame(g_rows).to_csv(REPORTS_DIR / "benchmark_gcn_hyperparameter_sensitivity.csv", index=False)
    print("✓ Module C complete: 350 5-seed runs & 35 GCN grid runs saved.")


# ==============================================================================
# MODULE D: CROSS-SITE PD, TRANSDIAGNOSTIC TRANSFER, MUMTAZ EO, SPLIT-TARGET-HC
# ==============================================================================
def eval_pair_transfer(src_gs, tgt_gs, s_name, t_name, t_type):
    """Evaluate zero-retraining target-HC calibrated transfer for 5 models."""
    set_seed(42)
    sv = [g for g in src_gs if int(g["y"]) in (0, 1)]
    tv = [g for g in tgt_gs if int(g["y"]) in (0, 1)]
    tx = torch.stack([g["x"] for g in sv])
    tadj = torch.stack([g["adj"] for g in sv])
    ty = torch.tensor([int(g["y"]) for g in sv], dtype=torch.long)
    vx = torch.stack([g["x"] for g in tv])
    vadj = torch.stack([g["adj"] for g in tv])
    vy_t = torch.tensor([int(g["y"]) for g in tv], dtype=torch.long)
    vy = vy_t.numpy()

    tz, _, _, _ = compute_fold_normative_z(tx, ty, tx)
    vz, _, _, _ = compute_fold_normative_z(vx, vy_t, vx)

    adj_id_tr = torch.eye(19, dtype=torch.float32).unsqueeze(0).expand(len(sv), -1, -1)
    adj_id_va = torch.eye(19, dtype=torch.float32).unsqueeze(0).expand(len(tv), -1, -1)

    p_gcn, y_gcn = train_eval_gcn(tz, tadj, ty, vz, vadj, epochs=40, seed=42)
    p_id, y_id = train_eval_gcn(tz, adj_id_tr, ty, vz, adj_id_va, epochs=40, seed=42)

    X_tr, X_te = tz.view(len(sv), -1).numpy(), vz.view(len(tv), -1).numpy()
    y_tr = ty.numpy()

    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42).fit(X_tr, y_tr)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, y_tr)
    gb = HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=42).fit(X_tr, y_tr)

    preds_map = {
        "GCN_OMST": (p_gcn, y_gcn),
        "GCN_Identity": (p_id, y_id),
        "Tuned_LR": (lr.predict_proba(X_te)[:, 1], lr.predict(X_te)),
        "RandomForest": (rf.predict_proba(X_te)[:, 1], rf.predict(X_te)),
        "GBDT": (gb.predict_proba(X_te)[:, 1], gb.predict(X_te)),
    }
    out = []
    for m, (pr, pd_cls) in preds_map.items():
        auc, lo, hi = bootstrap_auc_ci(vy, pr)
        out.append({
            "transfer_type": t_type,
            "source_cohort": s_name,
            "target_cohort": t_name,
            "n_source": len(sv),
            "n_target": len(tv),
            "model": m,
            "auc": auc,
            "ci_lower": lo,
            "ci_upper": hi,
            "accuracy": float(accuracy_score(vy, pd_cls)),
            "balanced_accuracy": float(balanced_accuracy_score(vy, pd_cls)),
        })
    return out


def run_module_d(sp):
    """Run PD transfer, 4x4 transdiagnostic matrix, Mumtaz EO/EC transfer, and Split-Target-HC."""
    print("\n=== Module D: PD Transfer, Transdiagnostic Matrix, Mumtaz EO, Split-Target-HC ===")
    sleep, mum_ec, mum_eo = sp["sleep_ds004902"], sp["mumtaz_ec"], sp["mumtaz_eo"]
    cav, ucsd, dem = sp["cavanagh_ds003478"], sp["ucsd_ds002778"], sp["dementia_ds004504"]
    unm, iowa = sp["unm_ds003490"], sp["iowa_ds004584"]
    unm_off = [g for g in unm if g["med_state"] in ("OFF", "CTL")]

    pd_cohorts = {"UCSD_PD": ucsd, "UNM_PD_OFF": unm_off, "Iowa_PD": iowa}
    pd_res = []
    for s in pd_cohorts:
        for t in pd_cohorts:
            if s != t:
                pd_res.extend(eval_pair_transfer(pd_cohorts[s], pd_cohorts[t], s, t, "Within_Disorder_PD_Cross_Site"))
    pd.DataFrame(pd_res).to_csv(REPORTS_DIR / "cross_site_pd_transfer_benchmarks.csv", index=False)

    mdd_res = []
    for s_gs, t_gs, sn, tn in [(mum_ec, cav, "Mumtaz_MDD", "Cavanagh_MDD"), (cav, mum_ec, "Cavanagh_MDD", "Mumtaz_MDD")]:
        mdd_res.extend(eval_pair_transfer(s_gs, t_gs, sn, tn, "Within_Disorder_MDD_Cross_Site"))

    trans = {"Sleep_Debt": sleep, "Mumtaz_MDD": mum_ec, "Iowa_PD": iowa, "Dementia_All": dem}
    all_tr = list(pd_res) + list(mdd_res)
    for s in trans:
        for t in trans:
            if s != t and not any(r["source_cohort"] == s and r["target_cohort"] == t for r in all_tr):
                all_tr.extend(eval_pair_transfer(trans[s], trans[t], s, t, "Cross_Disorder_Transdiagnostic"))
    pd.DataFrame(all_tr).to_csv(REPORTS_DIR / "comprehensive_transdiagnostic_transfer_matrix.csv", index=False)

    # Mumtaz EO vs EC concordance + state-matched transfers (Table S13)
    ec_by_id = {g["subject_id"]: g for g in mum_ec}
    eo_by_id = {g["subject_id"]: g for g in mum_eo}
    shared_ids = sorted(set(ec_by_id) & set(eo_by_id))
    conc_rows = []
    for m_lbl, fn in [
        ("Frontal Periodic Theta (4–8 Hz Residual)", lambda g: float(np.mean(g["x"].numpy()[FOCAL_5, 0]))),
        ("Frontal Aperiodic Exponent (χ)", lambda g: float(np.mean(g["x"].numpy()[FOCAL_5, 3]))),
        ("Global Aperiodic Exponent (χ)", lambda g: float(np.mean(g["x"].numpy()[:, 3]))),
        ("Global Cost Efficiency (GCE, OMST)", lambda g: float(g["gce"])),
    ]:
        vec = np.array([fn(ec_by_id[sid]) for sid in shared_ids])
        veo = np.array([fn(eo_by_id[sid]) for sid in shared_ids])
        r, p = pearsonr(vec, veo)
        conc_rows.append({"marker": m_lbl, "n_shared": len(shared_ids), "pearson_r": float(r), "p_value": float(p)})
    pd.DataFrame(conc_rows).to_csv(REPORTS_DIR / "mumtaz_ec_eo_concordance.csv", index=False)

    eo_transfers = [
        ("Mumtaz_EC (Eyes-Closed)", "Cavanagh_EO (Eyes-Open)", mum_ec, cav),
        ("Mumtaz_EO (Eyes-Open)", "Cavanagh_EO (Eyes-Open)", mum_eo, cav),
        ("Cavanagh_EO (Eyes-Open)", "Mumtaz_EC (Eyes-Closed)", cav, mum_ec),
        ("Cavanagh_EO (Eyes-Open)", "Mumtaz_EO (Eyes-Open)", cav, mum_eo),
        ("Mumtaz_EO (Eyes-Open)", "Mumtaz_EC (Eyes-Closed)", mum_eo, mum_ec),
        ("Mumtaz_EC (Eyes-Closed)", "Mumtaz_EO (Eyes-Open)", mum_ec, mum_eo),
    ]
    eo_tr_rows = []
    for sn, tn, s_gs, t_gs in eo_transfers:
        res_pair = eval_pair_transfer(s_gs, t_gs, sn, tn, "State_Matched_Affective")
        rmap = {r["model"]: r for r in res_pair}
        eo_tr_rows.append({
            "source_cohort": sn,
            "target_cohort": tn,
            "gcn_auc": rmap["GCN_OMST"]["auc"],
            "gcn_ci_lower": rmap["GCN_OMST"]["ci_lower"],
            "gcn_ci_upper": rmap["GCN_OMST"]["ci_upper"],
            "gcn_bal_acc": rmap["GCN_OMST"]["balanced_accuracy"],
            "lr_auc": rmap["Tuned_LR"]["auc"],
            "lr_ci_lower": rmap["Tuned_LR"]["ci_lower"],
            "lr_ci_upper": rmap["Tuned_LR"]["ci_upper"],
            "lr_bal_acc": rmap["Tuned_LR"]["balanced_accuracy"],
        })
    pd.DataFrame(eo_tr_rows).to_csv(REPORTS_DIR / "mumtaz_eo_state_matched_transfer.csv", index=False)

    # Split-Target-HC Holdout Check (20 random 50/50 HC splits across all 6 PD pairs, x_specparam_resid)
    hold_rows = []
    for sn in pd_cohorts:
        for tn in pd_cohorts:
            if sn == tn:
                continue
            sv, tv = pd_cohorts[sn], pd_cohorts[tn]
            tx = torch.stack([g["x"] for g in sv])
            ty = torch.tensor([int(g["y"]) for g in sv], dtype=torch.long)
            tz, _, _, _ = compute_fold_normative_z(tx, ty, tx)
            lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42).fit(tz.view(len(sv), -1).numpy(), ty.numpy())
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(tz.view(len(sv), -1).numpy(), ty.numpy())

            vx = torch.stack([g["x"] for g in tv])
            vy = np.array([int(g["y"]) for g in tv])
            hc_idx = np.where(vy == 0)[0]
            pt_idx = np.where(vy == 1)[0]
            rng = np.random.RandomState(42)
            lr_aucs, rf_aucs = [], []
            for _ in range(20):
                perm_hc = rng.permutation(hc_idx)
                half = len(perm_hc) // 2
                cal_hc = perm_hc[:half]
                eval_hc = perm_hc[half:]
                eval_idx = np.concatenate([pt_idx, eval_hc])
                mu = vx[cal_hc].mean(dim=0, keepdim=True)
                sd = vx[cal_hc].std(dim=0, keepdim=True) + 1e-6
                X_eval = ((vx[eval_idx] - mu) / sd).view(len(eval_idx), -1).numpy()
                y_eval = vy[eval_idx]
                lr_aucs.append(roc_auc_score(y_eval, lr.predict_proba(X_eval)[:, 1]))
                rf_aucs.append(roc_auc_score(y_eval, rf.predict_proba(X_eval)[:, 1]))
            hold_rows.append({
                "source_cohort": sn,
                "target_cohort": tn,
                "lr_split_hc_mean_auc": float(np.mean(lr_aucs)),
                "lr_split_hc_sd_auc": float(np.std(lr_aucs, ddof=1)),
                "rf_split_hc_mean_auc": float(np.mean(rf_aucs)),
                "rf_split_hc_sd_auc": float(np.std(rf_aucs, ddof=1)),
            })
    pd.DataFrame(hold_rows).to_csv(REPORTS_DIR / "split_target_hc_holdout_check.csv", index=False)
    print("✓ Module D complete: PD transfer, 4x4 transdiagnostic matrix, Mumtaz EO, Split-Target-HC saved.")


# ==============================================================================
# MODULE E: LEAKAGE, HEURISTIC SLOWING, PLI SPD AUDIT, RIEMANNIAN, META-ANALYSIS
# ==============================================================================
def run_module_e(sp, master_payload):
    """Run leakage ablation, heuristic slowing, PLI SPD audit, Riemannian, and DL/HKSJ meta-analysis."""
    print("\n=== Module E: Leakage, Heuristic Slowing, PLI SPD, Riemannian, Meta-Analysis ===")
    sleep, mum_ec, mum_eo = sp["sleep_ds004902"], sp["mumtaz_ec"], sp["mumtaz_eo"]
    cav, ucsd, dem = sp["cavanagh_ds003478"], sp["ucsd_ds002778"], sp["dementia_ds004504"]
    unm, iowa = sp["unm_ds003490"], sp["iowa_ds004584"]

    # 1. Leakage Ablation (Table S19)
    leak_cohorts = [
        ("Study 1: Sleep Deprivation", "Paired_Crossover_Discordant", sleep, lambda g: g.get("sub_name", g["subject_id"])),
        ("Study 4: UCSD Parkinson", "Repeated_Sessions_Concordant", ucsd, lambda g: g.get("bids_id", g["subject_id"])),
        ("Study 6c: UNM Parkinson All", "Repeated_Sessions_Concordant", unm, lambda g: g["subject_id"]),
    ]
    regimes = ["Strict_Leak_Free", "Leaky_Pre_Standardization", "Leaky_Subject_Mixing"]
    models_lk = ["Tuned_LogisticRegression", "RandomForest", "GCN_OMST"]
    leak_rows = []
    for cname, dtype, glist, gfn in leak_cohorts:
        vg = [g for g in glist if int(g["y"]) in (0, 1)]
        y_all = np.array([int(g["y"]) for g in vg])
        groups = np.array([gfn(g) for g in vg])
        all_x = torch.stack([g["x"] for g in vg])
        gl_mu, gl_sd = all_x.mean(dim=0, keepdim=True), all_x.std(dim=0, keepdim=True) + 1e-6
        gl_z = (all_x - gl_mu) / gl_sd
        for reg in regimes:
            set_seed(42)
            oof_p = {m: np.zeros(len(vg)) for m in models_lk}
            oof_y = {m: np.zeros(len(vg)) for m in models_lk}
            if reg in ("Strict_Leak_Free", "Leaky_Pre_Standardization"):
                splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(vg, y_all, groups=groups))
            else:
                splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(vg, y_all))
            n_leaked = sum(sum(1 for gid in groups[va] if gid in set(groups[tr])) for tr, va in splits)
            overlap_pct = 100.0 * n_leaked / len(vg)
            for fold, (tr, va) in enumerate(splits):
                tx = torch.stack([vg[i]["x"] for i in tr])
                tadj = torch.stack([vg[i]["adj"] for i in tr])
                ty = torch.tensor([int(vg[i]["y"]) for i in tr], dtype=torch.long)
                vx = torch.stack([vg[i]["x"] for i in va])
                vadj = torch.stack([vg[i]["adj"] for i in va])
                if reg == "Leaky_Pre_Standardization":
                    tz, vz = gl_z[tr], gl_z[va]
                else:
                    tz, vz, _, _ = compute_fold_normative_z(tx, ty, vx)
                X_tr, X_va, y_tr = tz.view(len(tr), -1).numpy(), vz.view(len(va), -1).numpy(), ty.numpy()
                if reg != "Leaky_Subject_Mixing":
                    icv = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(X_tr, y_tr, groups=[groups[i] for i in tr]))
                else:
                    icv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                gs_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}, cv=icv, scoring="roc_auc").fit(X_tr, y_tr)
                oof_p["Tuned_LogisticRegression"][va], oof_y["Tuned_LogisticRegression"][va] = gs_lr.predict_proba(X_va)[:, 1], gs_lr.predict(X_va)
                rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, y_tr)
                oof_p["RandomForest"][va], oof_y["RandomForest"][va] = rf.predict_proba(X_va)[:, 1], rf.predict(X_va)
                oof_p["GCN_OMST"][va], oof_y["GCN_OMST"][va] = train_eval_gcn(tz, tadj, ty, vz, vadj, epochs=50, seed=42 + fold)
            for m in models_lk:
                auc, lo, hi = bootstrap_auc_ci(y_all, oof_p[m])
                clo, chi = cluster_bootstrap_auc_ci(y_all, oof_p[m], groups)
                leak_rows.append({
                    "cohort": cname,
                    "design_type": dtype,
                    "regime": reg,
                    "model": m,
                    "n_sessions": len(vg),
                    "n_participants": len(np.unique(groups)),
                    "subject_overlap_pct": overlap_pct,
                    "auc": auc,
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "cluster_ci_lower": clo,
                    "cluster_ci_upper": chi,
                    "accuracy": float(accuracy_score(y_all, oof_y[m])),
                    "balanced_accuracy": float(balanced_accuracy_score(y_all, oof_y[m])),
                })
    df_leak = pd.DataFrame(leak_rows)
    smap = df_leak[df_leak["regime"] == "Strict_Leak_Free"].set_index(["cohort", "model"])["auc"].to_dict()
    df_leak["delta_auc_vs_strict"] = df_leak.apply(lambda r: r["auc"] - smap[(r["cohort"], r["model"])], axis=1)
    df_leak.to_csv(REPORTS_DIR / "benchmark_leakage_ablation.csv", index=False)

    # 2. Heuristic Zero-Training Slowing Baseline
    heur_rows = []
    for cname, glist, _ in get_within_cohort_specs(sp):
        vg = [g for g in glist if int(g["y"]) in (0, 1)]
        y_t = np.array([int(g["y"]) for g in vg])
        th = np.array([float(g["x"][FOCAL_5, 0].mean()) for g in vg])
        al = np.array([float(g["x"][FOCAL_5, 2].mean()) for g in vg])
        slow = th - al
        auc_th = roc_auc_score(y_t, th)
        auc_sl = roc_auc_score(y_t, slow)
        best_b = 0.5
        for thr in np.percentile(slow, np.linspace(10, 90, 81)):
            pr = (slow >= thr).astype(int)
            best_b = max(best_b, balanced_accuracy_score(y_t, pr), balanced_accuracy_score(y_t, 1 - pr))
        heur_rows.append({
            "cohort": cname,
            "n_samples": len(vg),
            "heuristic_rule": "Frontal_Theta_Alpha_Difference",
            "auc_raw": float(auc_sl),
            "auc_oriented": float(max(auc_sl, 1.0 - auc_sl)),
            "balanced_accuracy": float(best_b),
            "single_theta_auc_oriented": float(max(auc_th, 1.0 - auc_th)),
        })
    pd.DataFrame(heur_rows).to_csv(REPORTS_DIR / "benchmark_heuristic_slowing.csv", index=False)

    # 3. PLI SPD Eigenvalue Audit & Riemannian Tangent Space (Table S22)
    audit_cohorts = [
        ("Study 1: Sleep Deprivation (ds004902)", sleep),
        ("Study 2: Outpatient MDD (EC)", mum_ec),
        ("Study 3: Cavanagh Depression (ds003478)", cav),
        ("Study 4: UCSD Parkinson (ds002778)", ucsd),
        ("Study 5: Clinical Dementia (ds004504)", dem),
        ("Study 6: UNM Parkinson (ds003490)", unm),
        ("Study 7: Iowa Parkinson (ds004584)", iowa),
    ]
    eig_rows, all_min_eigs = [], []
    for clbl, glist in audit_cohorts:
        mins = []
        for g in glist:
            p_m = g["pli"].numpy().astype(np.float64)
            C = 0.5 * (p_m + p_m.T)
            np.fill_diagonal(C, 1.0)
            ev = np.linalg.eigvalsh(C)
            mins.append(float(ev.min()))
            all_min_eigs.append(float(ev.min()))
        arr = np.asarray(mins)
        n_ind = int(np.sum(arr <= 0.0))
        n_shift = int(np.sum(arr + 1e-4 <= 0.0))
        eig_rows.append({
            "cohort": clbl,
            "n_sessions": len(arr),
            "min_eigenvalue": float(arr.min()),
            "median_min_eigenvalue": float(np.median(arr)),
            "max_min_eigenvalue": float(arr.max()),
            "n_indefinite_raw": n_ind,
            "pct_indefinite_raw": 100.0 * n_ind / len(arr),
            "n_indefinite_after_1e4_shift": n_shift,
        })
    pd.DataFrame(eig_rows).to_csv(REPORTS_DIR / "pli_spd_eigenvalue_audit.csv", index=False)
    arr_all = np.asarray(all_min_eigs)
    pd.DataFrame([{
        "cohort": "Total Across All 7 Cohorts",
        "n_sessions": len(arr_all),
        "min_eigenvalue": float(arr_all.min()),
        "median_min_eigenvalue": float(np.median(arr_all)),
        "max_min_eigenvalue": float(arr_all.max()),
        "n_indefinite_raw": int(np.sum(arr_all <= 0.0)),
        "pct_indefinite_raw": 100.0 * int(np.sum(arr_all <= 0.0)) / len(arr_all),
        "n_indefinite_after_1e4_shift": int(np.sum(arr_all + 1e-4 <= 0.0)),
    }]).to_csv(REPORTS_DIR / "pli_spd_eigenvalue_summary.csv", index=False)

    def upper_vec(M):
        i, j = np.triu_indices_from(M, k=0)
        w = np.where(i == j, 1.0, np.sqrt(2.0))
        return M[i, j] * w

    riem_rows = []
    riem_contrasts = [
        ("Study 1: Sleep Deprivation (ds004902)", sleep, True),
        ("Study 2: Mumtaz MDD (Eyes Closed)", mum_ec, False),
        ("Study 3: Cavanagh University Depression", cav, False),
        ("Study 4: UCSD Parkinson All Sessions", ucsd, True),
        ("Study 5a: Dementia AD vs Control", [g for g in dem if g["diagnosis"] in ("Alzheimer", "Control")], False),
        ("Study 5b: Dementia FTD vs Control", [g for g in dem if g["diagnosis"] in ("FTD", "Control")], False),
        ("Study 5c: Dementia All (AD+FTD) vs Control", dem, False),
        ("Study 6a: UNM PD-OFF vs Control", [g for g in unm if g["med_state"] in ("OFF", "CTL")], False),
        ("Study 6b: UNM PD-ON vs Control", [g for g in unm if g["med_state"] in ("ON", "CTL")], False),
        ("Study 7: Iowa Parkinson (ds004584)", iowa, False),
    ]
    i_off, j_off = np.triu_indices(19, k=1)
    for cname, glist, is_grp in riem_contrasts:
        vg = [g for g in glist if int(g["y"]) in (0, 1)]
        y_all = np.array([int(g["y"]) for g in vg])
        grps = np.array([g.get("bids_id", g.get("sub_name", g["subject_id"])) for g in vg])
        X_euc, X_ada, X_hig = [], [], []
        for g in vg:
            p_m = g["pli"].numpy().astype(np.float64)
            C = 0.5 * (p_m + p_m.T)
            np.fill_diagonal(C, 1.0)
            X_euc.append(C[i_off, j_off])
            vals, vecs = np.linalg.eigh(C)
            shift = max(0.0, 0.01 - float(vals.min()))
            log_ada = (vecs * np.log(vals + shift)) @ vecs.T
            X_ada.append(upper_vec(log_ada))
            log_hig = (vecs * np.log(np.maximum(vals, 1e-4))) @ vecs.T
            X_hig.append(upper_vec(log_hig))
        reps = [
            ("Euclidean_UpperTri_171D", np.asarray(X_euc, dtype=np.float64)),
            ("Riemannian_Gershgorin_SPD_190D", np.asarray(X_ada, dtype=np.float64)),
            ("Riemannian_Higham_SPD_190D", np.asarray(X_hig, dtype=np.float64)),
        ]
        splits = list((StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) if is_grp else StratifiedKFold(n_splits=5, shuffle=True, random_state=42)).split(vg, y_all, groups=grps if is_grp else None))
        for rname, X_rep in reps:
            oof_p = {k: np.zeros(len(vg)) for k in ("LR", "Ridge", "RF")}
            oof_y = {k: np.zeros(len(vg)) for k in ("LR", "Ridge", "RF")}
            for tr, va in splits:
                mu = X_rep[tr].mean(axis=0, keepdims=True)
                sd = X_rep[tr].std(axis=0, keepdims=True) + 1e-6
                X_tr, X_va, y_tr = (X_rep[tr] - mu) / sd, (X_rep[va] - mu) / sd, y_all[tr]
                icv = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(X_tr, y_tr, groups=grps[tr])) if is_grp else StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                gs_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}, cv=icv, scoring="roc_auc").fit(X_tr, y_tr)
                oof_p["LR"][va], oof_y["LR"][va] = gs_lr.predict_proba(X_va)[:, 1], gs_lr.predict(X_va)
                gs_rc = GridSearchCV(RidgeClassifier(random_state=42), {"alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}, cv=icv, scoring="roc_auc").fit(X_tr, y_tr)
                oof_p["Ridge"][va], oof_y["Ridge"][va] = 1.0 / (1.0 + np.exp(-gs_rc.decision_function(X_va))), gs_rc.predict(X_va)
                rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, y_tr)
                oof_p["RF"][va], oof_y["RF"][va] = rf.predict_proba(X_va)[:, 1], rf.predict(X_va)
            for ck in ("LR", "Ridge", "RF"):
                a, lo, hi = bootstrap_auc_ci(y_all, oof_p[ck])
                riem_rows.append({
                    "cohort": cname,
                    "representation": rname,
                    "classifier": ck,
                    "n_samples": len(vg),
                    "auc": a,
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "balanced_accuracy": float(balanced_accuracy_score(y_all, oof_y[ck])),
                })
    pd.DataFrame(riem_rows).to_csv(REPORTS_DIR / "benchmark_riemannian_tangent_space.csv", index=False)

    # 4. Random-effects Meta-Analysis from genuine specparam master_payload (Table S23)
    bp = master_payload["biophysics"]

    def get_es(ckey, paired=False):
        r = bp[ckey]["frontal_theta_5ch"]
        d = r["cohens_d"]
        se = (r["ci_upper"] - r["ci_lower"]) / (2.0 * 1.96)
        return float(d), float(se * se)

    def dl_pool(items):
        ds = np.array([x[0] for x in items])
        vs = np.array([x[1] for x in items])
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
        z = d_re / se_re
        p_dl = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
        q_hksj = float(max(1.0, Q / df_q)) if df_q > 0 else 1.0
        se_hk = float(np.sqrt(q_hksj) * se_re)
        t_c = float(stats.t.ppf(0.975, df=df_q)) if df_q > 0 else 1.96
        p_hk = float(2.0 * (1.0 - stats.t.cdf(abs(d_re / se_hk), df=df_q))) if df_q > 0 else p_dl
        i2 = float(max(0.0, (Q - df_q) / Q * 100.0)) if Q > 0 and df_q > 0 else 0.0
        return {
            "k_studies": k,
            "pooled_d": d_re,
            "se_re": se_re,
            "ci_lower": d_re - 1.96 * se_re,
            "ci_upper": d_re + 1.96 * se_re,
            "hksj_ci_lower": d_re - t_c * se_hk,
            "hksj_ci_upper": d_re + t_c * se_hk,
            "z_score": z,
            "p_value": p_dl,
            "p_hksj": p_hk,
            "Q_stat": Q,
            "Q_df": df_q,
            "tau2": tau2,
            "tau": float(np.sqrt(tau2)),
            "I2_percent": i2,
        }

    def run_meta_scope(sname, dep_items, slow_items, slow_lbl="Neurodegenerative_Slowing_Domain"):
        p_dep = dl_pool(dep_items)
        p_slw = dl_pool(slow_items)
        p_all = dl_pool(dep_items + slow_items)
        q_b = float(((p_dep["pooled_d"] - p_slw["pooled_d"]) ** 2) / (p_dep["se_re"] ** 2 + p_slw["se_re"] ** 2))
        p_b = float(1.0 - stats.chi2.cdf(q_b, df=1))
        rows = []
        for gname, pdict, qb, pb in [
            ("Depressive_HypoOscillatory_Domain", p_dep, q_b, p_b),
            (slow_lbl, p_slw, q_b, p_b),
            ("Overall_Pooled", p_all, q_b, p_b),
        ]:
            row = {"analysis_set": sname, "domain_group": gname}
            row.update(pdict)
            row["Q_between"] = qb
            row["p_between"] = pb
            rows.append(row)
        return rows

    dep_k2 = [get_es("Mumtaz_MDD_EC_vs_HC"), get_es("Cavanagh_DEP_vs_HC")]
    e_ucsd_off = get_es("UCSD_PD_OFF_vs_HC")
    e_ucsd_on = get_es("UCSD_PD_ON_vs_HC")
    e_ucsd_avg = get_es("UCSD_PD_SubjMean_vs_HC")
    e_dem_all = get_es("Dementia_All_vs_HC")
    e_dem_ad = get_es("Dementia_AD_vs_HC")
    e_dem_ftd = get_es("Dementia_FTD_vs_HC")
    e_unm_off = get_es("UNM_PD_OFF_vs_HC")
    e_unm_on = get_es("UNM_PD_ON_vs_HC")
    e_iowa = get_es("Iowa_PD_vs_HC")
    # Sleep d_IG (between-group pooled-SD scaling matching baseline SD for Morris-DeShon scale)
    s_th_sd = np.array([float(np.mean(g["x"].numpy()[FOCAL_5, 0])) for g in sleep if g["condition"] == "SD"])
    s_th_ns = np.array([float(np.mean(g["x"].numpy()[FOCAL_5, 0])) for g in sleep if g["condition"] == "NS"])
    sp_sl = np.sqrt((np.var(s_th_sd, ddof=1) + np.var(s_th_ns, ddof=1)) / 2.0)
    d_ig_sl = float((np.mean(s_th_sd) - np.mean(s_th_ns)) / sp_sl)
    v_ig_sl = float((71 + 71) / (71 * 71) + (d_ig_sl ** 2) / (2 * 142))

    meta_rows = []
    meta_rows.extend(run_meta_scope("Primary_Independent_K6_PD_OFF", dep_k2, [e_ucsd_off, e_dem_all, e_unm_off, e_iowa]))
    meta_rows.extend(run_meta_scope("Sens1_UCSD_Participant_Avg", dep_k2, [e_ucsd_avg, e_dem_all, e_unm_off, e_iowa]))
    meta_rows.extend(run_meta_scope("Sens2_PD_ON_Medication", dep_k2, [e_ucsd_on, e_dem_all, e_unm_on, e_iowa]))
    meta_rows.extend(run_meta_scope("Sens3_Dementia_AD_Only", dep_k2, [e_ucsd_off, e_dem_ad, e_unm_off, e_iowa]))
    meta_rows.extend(run_meta_scope("Sens4_Dementia_FTD_Only", dep_k2, [e_ucsd_off, e_dem_ftd, e_unm_off, e_iowa]))
    meta_rows.extend(run_meta_scope("Sens5_Including_Sleep_Deprivation_K7", dep_k2, [(d_ig_sl, v_ig_sl), e_ucsd_off, e_dem_all, e_unm_off, e_iowa], slow_lbl="Pathological_and_SleepPressure_Slowing_Domain"))
    meta_rows.extend(run_meta_scope("LeaveOneOut_Omit_UCSD", dep_k2, [e_dem_all, e_unm_off, e_iowa]))
    meta_rows.extend(run_meta_scope("LeaveOneOut_Omit_Dementia", dep_k2, [e_ucsd_off, e_unm_off, e_iowa]))
    meta_rows.extend(run_meta_scope("LeaveOneOut_Omit_UNM", dep_k2, [e_ucsd_off, e_dem_all, e_iowa]))
    meta_rows.extend(run_meta_scope("LeaveOneOut_Omit_Iowa", dep_k2, [e_ucsd_off, e_dem_all, e_unm_off]))
    pd.DataFrame(meta_rows).to_csv(REPORTS_DIR / "meta_analysis_frontal_theta_polarity.csv", index=False)
    print("✓ Module E complete: leakage, heuristic, PLI SPD, Riemannian, and DL/HKSJ meta-analysis saved.")


# ==============================================================================
# MODULE F: REAL GRAND-AVERAGE SPECTRA (FIG S4) & REAL 1,000 PERMUTATIONS (FIG S6)
# ==============================================================================
def _perm_worker_task(args):
    """Run one chunk of full-pipeline label-permutation refits cleanly."""
    task_id, X_src, adj_src, y_src, grps_src, X_tgt, adj_tgt, y_tgt, is_transfer, is_grp, seed_list = args
    torch.set_num_threads(1)
    null_gcn = []
    null_lr = []
    for pseed in seed_list:
        rng = np.random.RandomState(pseed)
        if is_transfer:
            # Permute source diagnostic labels cleanly while keeping held-out target & HC norm fixed
            y_perm_tr = rng.permutation(y_src)
            ty_t = torch.tensor(y_perm_tr, dtype=torch.long)
            vy_t = torch.tensor(y_tgt, dtype=torch.long)
            if int((ty_t == 0).sum()) < 2 or int((vy_t == 0).sum()) < 2:
                null_gcn.append(0.5)
                null_lr.append(0.5)
                continue
            tz, _, _, _ = compute_fold_normative_z(X_src, ty_t, X_src)
            vz, _, _, _ = compute_fold_normative_z(X_tgt, vy_t, X_tgt)
            pr_gcn, _ = train_eval_gcn(tz, adj_src, ty_t, vz, adj_tgt, epochs=40, seed=pseed)
            null_gcn.append(float(roc_auc_score(y_tgt, pr_gcn)))
            lr = LogisticRegression(C=0.1, max_iter=1000, random_state=pseed).fit(tz.view(len(ty_t), -1).numpy(), y_perm_tr)
            null_lr.append(float(roc_auc_score(y_tgt, lr.predict_proba(vz.view(len(vy_t), -1).numpy())[:, 1])))
        else:
            # Within-cohort permutation
            if is_grp:
                # Permute labels at unique participant cluster level (concordant cohorts) or session level
                uniq_g = np.unique(grps_src)
                # Flip labels per cluster to preserve within-subject clustering cleanly
                y_perm = np.array(y_src, copy=True)
                g_map = {}
                for ug in uniq_g:
                    g_map[ug] = int(y_src[grps_src == ug][0])
                perm_vals = rng.permutation(list(g_map.values()))
                pmap = dict(zip(uniq_g, perm_vals))
                y_perm = np.array([pmap[g] for g in grps_src], dtype=int)
                splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                splits = list(splitter.split(np.zeros(len(y_perm)), y_perm, groups=grps_src))
            else:
                y_perm = rng.permutation(y_src)
                splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                splits = list(splitter.split(np.zeros(len(y_perm)), y_perm))

            oof_g = np.zeros(len(y_perm))
            oof_l = np.zeros(len(y_perm))
            valid_perm = True
            for fold, (tr, va) in enumerate(splits):
                ty_t = torch.tensor(y_perm[tr], dtype=torch.long)
                if int((ty_t == 0).sum()) < 2 or int((ty_t == 1).sum()) < 1:
                    valid_perm = False
                    break
                tz, vz, _, _ = compute_fold_normative_z(X_src[tr], ty_t, X_src[va])
                oof_g[va], _ = train_eval_gcn(tz, adj_src[tr], ty_t, vz, adj_src[va], epochs=50, seed=42 + fold)
                lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42).fit(tz.view(len(tr), -1).numpy(), y_perm[tr])
                oof_l[va] = lr.predict_proba(vz.view(len(va), -1).numpy())[:, 1]
            if valid_perm:
                null_gcn.append(float(roc_auc_score(y_perm, oof_g)))
                null_lr.append(float(roc_auc_score(y_perm, oof_l)))
            else:
                null_gcn.append(0.5)
                null_lr.append(0.5)
    return task_id, null_gcn, null_lr


def run_module_f(sp, psds):
    """Build real grand-average spectra cache (Fig S4) and real 1,000-permutation distributions (Fig S6)."""
    print("\n=== Module F: Real Grand-Average Spectra & 1,000 Full-Pipeline Permutations ===")
    # 1. Real grand-average raw log10 PSD and specparam-flattened periodic residual spectra (1.0-40.0 Hz)
    # Using Mean across canonical 19 channels (and frontal 5 / posterior 5) for clear grand averages
    freqs_full = np.asarray(psds["sleep_ds004902"][0]["freqs"], dtype=float)
    fmask = (freqs_full >= 1.0) & (freqs_full <= 40.0)
    f_plot = freqs_full[fmask]

    def extract_group_spectra(sp_list, psd_list, selector_fn):
        raw_curves, resid_curves = [], []
        fg = SpectralGroupModel(aperiodic_mode="fixed", peak_width_limits=[1.0, 12.0], max_n_peaks=6, min_peak_height=0.05, peak_threshold=2.0, verbose=False)
        sel_idx = [i for i, r in enumerate(sp_list) if selector_fn(r)]
        # Average across frontal 5 channels for theta-sensitive studies, or all 19 channels?
        # Using 19-channel global average shows both frontal theta and posterior alpha cleanly
        sel_psd = np.stack([np.maximum(np.asarray(psd_list[i]["mean_psd"], dtype=float)[:, fmask], 1e-20).mean(axis=0) for i in sel_idx])
        fg.fit(f_plot, sel_psd, [1.0, 40.0])
        for k_i in range(len(sel_idx)):
            lp = np.log10(sel_psd[k_i])
            ap = fg.get_model(k_i, regenerate=True).results.model.get_component("aperiodic", "log")
            raw_curves.append(lp)
            resid_curves.append(lp - ap)
        arr_raw = np.stack(raw_curves)
        arr_res = np.stack(resid_curves)
        return {
            "n": len(sel_idx),
            "raw_mean": arr_raw.mean(axis=0).tolist(),
            "raw_sem": (arr_raw.std(axis=0, ddof=1) / np.sqrt(len(sel_idx))).tolist(),
            "res_mean": arr_res.mean(axis=0).tolist(),
            "res_sem": (arr_res.std(axis=0, ddof=1) / np.sqrt(len(sel_idx))).tolist(),
        }

    spectra_specs = [
        ("Study 1: Sleep Deprivation", "sleep_ds004902", "SD (N=71)", lambda r: r["condition"] == "SD", "NS (N=71)", lambda r: r["condition"] == "NS"),
        ("Study 2: Outpatient MDD (EC)", "mumtaz_ec", "MDD (N=30)", lambda r: r["group"] == "MDD", "HC (N=28)", lambda r: r["group"] == "HC"),
        ("Study 3: Univ. Depression", "cavanagh_ds003478", "Dep (N=44)", lambda r: r["group"] == "Depressed", "HC (N=75)", lambda r: r["group"] == "Control"),
        ("Study 4: UCSD Parkinson", "ucsd_ds002778", "PD All (N=30)", lambda r: "PD" in r["group"], "HC (N=16)", lambda r: r["group"] == "HC"),
        ("Study 5: Clinical Dementia", "dementia_ds004504", "AD+FTD (N=59)", lambda r: r["diagnosis"] in ("Alzheimer", "FTD"), "HC (N=29)", lambda r: r["diagnosis"] == "Control"),
        ("Study 6a: UNM PD-OFF", "unm_ds003490", "PD-OFF (N=25)", lambda r: r["med_state"] == "OFF", "HC (N=25)", lambda r: r["med_state"] == "CTL"),
        ("Study 7: Iowa Parkinson", "iowa_ds004584", "PD (N=100)", lambda r: r["group"] == "PD", "HC (N=49)", lambda r: r["group"] == "Control"),
    ]
    spectra_cache = {"freqs": f_plot.tolist(), "cohorts": []}
    for title, ckey, c_lbl, c_fn, h_lbl, h_fn in spectra_specs:
        spectra_cache["cohorts"].append({
            "title": title,
            "cohort_key": ckey,
            "case_label": c_lbl,
            "ctrl_label": h_lbl,
            "case": extract_group_spectra(sp[ckey], psds[ckey], c_fn),
            "ctrl": extract_group_spectra(sp[ckey], psds[ckey], h_fn),
        })
    with open(REPORTS_DIR / "real_grand_average_spectra.json", "w", encoding="utf-8") as f:
        json.dump(spectra_cache, f)
    print("✓ Computed real 7-cohort grand-average PSD & specparam residual spectra.")

    # 2. Real 1,000 full-pipeline label-permutation refits across 6 representative tasks
    dem = sp["dementia_ds004504"]
    iowa = sp["iowa_ds004584"]
    unm_off = [g for g in sp["unm_ds003490"] if g["med_state"] in ("OFF", "CTL")]
    cav = sp["cavanagh_ds003478"]
    mum_ec = sp["mumtaz_ec"]
    sleep = sp["sleep_ds004902"]

    def pack_cv(tid, label, glist):
        vg = [g for g in glist if int(g["y"]) in (0, 1)]
        X = torch.stack([g["x"] for g in vg])
        A = torch.stack([g["adj"] for g in vg])
        y = np.array([int(g["y"]) for g in vg])
        grps = np.array([g.get("bids_id", g.get("sub_name", g["subject_id"])) for g in vg])
        # Compute observed GCN_OMST & LR(C=0.1) AUC under exact protocol
        splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(vg, y))
        og, ol = np.zeros(len(vg)), np.zeros(len(vg))
        for f_i, (tr, va) in enumerate(splits):
            tz, vz, _, _ = compute_fold_normative_z(X[tr], torch.tensor(y[tr]), X[va])
            og[va], _ = train_eval_gcn(tz, A[tr], torch.tensor(y[tr]), vz, A[va], epochs=50, seed=42 + f_i)
            lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42).fit(tz.view(len(tr), -1).numpy(), y[tr])
            ol[va] = lr.predict_proba(vz.view(len(va), -1).numpy())[:, 1]
        return {
            "id": tid,
            "label": label,
            "is_transfer": False,
            "is_grp": False,
            "obs_gcn": float(roc_auc_score(y, og)),
            "obs_lr": float(roc_auc_score(y, ol)),
            "X_src": X,
            "A_src": A,
            "y_src": y,
            "grps_src": grps,
            "X_tgt": None,
            "A_tgt": None,
            "y_tgt": None,
        }

    def pack_tr(tid, label, src_gs, tgt_gs):
        sv = [g for g in src_gs if int(g["y"]) in (0, 1)]
        tv = [g for g in tgt_gs if int(g["y"]) in (0, 1)]
        Xs, As = torch.stack([g["x"] for g in sv]), torch.stack([g["adj"] for g in sv])
        ys = np.array([int(g["y"]) for g in sv])
        Xt, At = torch.stack([g["x"] for g in tv]), torch.stack([g["adj"] for g in tv])
        yt = np.array([int(g["y"]) for g in tv])
        tz, _, _, _ = compute_fold_normative_z(Xs, torch.tensor(ys), Xs)
        vz, _, _, _ = compute_fold_normative_z(Xt, torch.tensor(yt), Xt)
        pr_g, _ = train_eval_gcn(tz, As, torch.tensor(ys), vz, At, epochs=40, seed=42)
        lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42).fit(tz.view(len(sv), -1).numpy(), ys)
        return {
            "id": tid,
            "label": label,
            "is_transfer": True,
            "is_grp": False,
            "obs_gcn": float(roc_auc_score(yt, pr_g)),
            "obs_lr": float(roc_auc_score(yt, lr.predict_proba(vz.view(len(tv), -1).numpy())[:, 1])),
            "X_src": Xs,
            "A_src": As,
            "y_src": ys,
            "grps_src": None,
            "X_tgt": Xt,
            "A_tgt": At,
            "y_tgt": yt,
        }

    tasks = [
        pack_cv("A", "Study 5c: Clinical Dementia All (5-Fold CV)", dem),
        pack_cv("B", "Study 7: Iowa Parkinson's (5-Fold CV)", iowa),
        pack_cv("C", "Study 6a: UNM PD-OFF (5-Fold CV)", unm_off),
        pack_cv("D", "Study 3: Univ. Depression Screening (5-Fold CV)", cav),
        pack_tr("E", "Transfer: Iowa PD → UNM PD-OFF", iowa, unm_off),
        pack_tr("F", "Transfer: Sleep Debt → Dementia All", sleep, dem),
    ]

    n_perms = 1000
    chunk_size = 50
    jobs = []
    for t in tasks:
        for start in range(0, n_perms, chunk_size):
            seeds = [10000 + s for s in range(start, min(start + chunk_size, n_perms))]
            jobs.append((
                t["id"], t["X_src"], t["A_src"], t["y_src"], t["grps_src"],
                t["X_tgt"], t["A_tgt"], t["y_tgt"], t["is_transfer"], t["is_grp"], seeds
            ))

    t_p0 = time.time()
    n_workers = min(8, os.cpu_count() or 4)
    print(f"Running 1,000 full-pipeline retraining permutations across 6 tasks ({len(jobs)} jobs) on {n_workers} workers...")
    agg = {t["id"]: {"gcn": [], "lr": []} for t in tasks}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_perm_worker_task, j) for j in jobs]
        for f in as_completed(futs):
            tid, ng, nl = f.result()
            agg[tid]["gcn"].extend(ng)
            agg[tid]["lr"].extend(nl)

    perm_out = {}
    for t in tasks:
        tid = t["id"]
        ng = np.asarray(agg[tid]["gcn"])
        nl = np.asarray(agg[tid]["lr"])
        # Two-sided and right-tailed exact permutation p-values around 0.5
        p_gcn = float((np.sum(np.abs(ng - 0.5) >= abs(t["obs_gcn"] - 0.5)) + 1.0) / (len(ng) + 1.0))
        p_lr = float((np.sum(np.abs(nl - 0.5) >= abs(t["obs_lr"] - 0.5)) + 1.0) / (len(nl) + 1.0))
        p_gcn_gt = float((np.sum(ng >= t["obs_gcn"]) + 1.0) / (len(ng) + 1.0))
        p_lr_gt = float((np.sum(nl >= t["obs_lr"]) + 1.0) / (len(nl) + 1.0))
        perm_out[tid] = {
            "label": t["label"],
            "n_perms": len(ng),
            "obs_gcn_auc": t["obs_gcn"],
            "obs_lr_auc": t["obs_lr"],
            "null_gcn_mean": float(ng.mean()),
            "null_gcn_sd": float(ng.std(ddof=1)),
            "p_perm_gcn": p_gcn,
            "p_perm_gcn_greater": p_gcn_gt,
            "null_lr_mean": float(nl.mean()),
            "null_lr_sd": float(nl.std(ddof=1)),
            "p_perm_lr": p_lr,
            "p_perm_lr_greater": p_lr_gt,
            "null_gcn_aucs": ng.tolist(),
            "null_lr_aucs": nl.tolist(),
        }
        print(f"  [{tid}] {t['label']:<42} | GCN={t['obs_gcn']:.3f} (p={p_gcn:.4f}) | LR={t['obs_lr']:.3f} (p={p_lr:.4f})")

    with open(REPORTS_DIR / "real_permutation_distributions.json", "w", encoding="utf-8") as f:
        json.dump(perm_out, f)
    print(f"✓ Module F complete in {time.time() - t_p0:.1f}s (1,000 real permutations x 6 tasks).")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run genuine specparam Paper 2 benchmark pipeline.")
    parser.add_argument("--only-module-f", action="store_true", help="Load canonical derivatives and run only Module F.")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("RUNNING UNIFIED GENUINE SPECPARAM (v2.0.0rc7) PAPER 2 BENCHMARK PIPELINE")
    print("=" * 78)
    if args.only_module_f:
        sp = torch.load(DATA_DIR / "specparam_canonical_7cohorts.pt", weights_only=False)
        psds = torch.load(DATA_DIR / "all_7cohorts_welch_psds.pt", weights_only=False)
        run_module_f(sp, psds)
    else:
        archive_legacy_reports()
        sp, psds = load_canonical_specparam_cohorts()
        master = run_module_a(sp)
        df_11 = run_module_b(sp)
        run_module_c(sp, df_11)
        run_module_d(sp)
        run_module_e(sp, master)
        run_module_f(sp, psds)
    print(f"\nALL MODULES COMPLETED SUCCESSFULLY in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
