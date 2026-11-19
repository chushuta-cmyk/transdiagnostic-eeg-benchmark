#!/usr/bin/env python3
"""
scripts/52_run_v7_reviewer_remediations.py
------------------------------------------
v7 Targeted Statistical, Manifold, Deep-Baseline, and Learning-Curve Remediations:
  1. Per-Contrast Nadeau-Bengio Paired Inference & Contrast-Level DerSimonian-Laird + HKSJ
     Random-Effects TOST across k=8 non-nested contrasts (and k=7 shared-HC / Study 2 splits).
  2. True Fold-Wise Affine-Invariant Riemannian Metric (AIRM) Fréchet-Mean Whitening
     Tangent-Space Audit on 19x19 Spatial Covariances (vs. Identity Log-Euclidean & Global Leaky AIRM).
  3. Standard Compact Deep Spectral-Spatial CNN Baseline (ShallowSpecCNN, modeled after EEGNet /
     ShallowConvNet) and Inner-3-Fold Tuned GCN-OMST-Tuned across all k=8 contrasts (10x5 grouped CV).
  4. Sample-Size Learning Curves (25%, 50%, 75%, 100% training participants across 5x5 grouped CV)
     on the 3 largest cohorts (Study 7 N=149, Study 1 N=142 sess, Study 3 N=119).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import importlib.util
import multiprocessing as mp
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from statsmodels.stats.multitest import multipletests

torch.set_num_threads(1)
warnings.filterwarnings("ignore", category=UserWarning)

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = BENCHMARK_ROOT / "src"
sys.path.append(str(SRC_DIR))
from harmonizer_and_graph import NormativeGCN, compute_fold_normative_z

DATA_DIR = BENCHMARK_ROOT / "data" / "derivatives"
REPORTS_DIR = BENCHMARK_ROOT / "reports"

REPEATS_10 = [42, 123, 456, 789, 2026, 101, 202, 303, 404, 505]
REPEATS_5 = [42, 123, 456, 789, 2026]

NONNESTED_8 = [
    ("Study 1: Sleep Deprivation (ds004902)", "sleep_ds004902"),
    ("Study 2: Mumtaz MDD (Eyes Closed)", "mumtaz_ec"),
    ("Study 3: Cavanagh University Depression", "cavanagh_ds003478"),
    ("Study 4: UCSD Parkinson All Sessions", "ucsd_ds002778"),
    ("Study 5a: Dementia AD vs Control", "dementia_ds004504"),
    ("Study 5b: Dementia FTD vs Control", "dementia_ds004504"),
    ("Study 6a: UNM PD-OFF vs Control", "unm_ds003490"),
    ("Study 7: Iowa Parkinson (ds004584)", "iowa_ds004584"),
]


def _load_v6_helpers():
    spec = importlib.util.spec_from_file_location("v6_sens", BENCHMARK_ROOT / "scripts" / "49_run_v6_sensitivities.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V6 = _load_v6_helpers()


def set_seed(seed: int = 42):
    V6.set_seed(seed)


def get_grouped_splits(groups, y, n_splits: int = 5, seed: int = 42):
    return V6.get_grouped_splits(groups, y, n_splits=n_splits, seed=seed)


def make_edge_matched_random_adj(num_nodes: int, num_edges: int, seed: int = 42) -> torch.Tensor:
    return V6.make_edge_matched_random_adj(num_nodes, num_edges, seed=seed)


def _filter_cohort_records(cname: str, ckey: str, records: list):
    if cname == "Study 5a: Dementia AD vs Control":
        return [g for g in records if g["diagnosis"] in ("Alzheimer", "Control")]
    if cname == "Study 5b: Dementia FTD vs Control":
        return [g for g in records if g["diagnosis"] in ("FTD", "Control")]
    if cname == "Study 6a: UNM PD-OFF vs Control":
        return [g for g in records if g["med_state"] in ("OFF", "CTL")]
    return [g for g in records if int(g["y"]) in (0, 1)]


# ==============================================================================
# Step 1: Per-Contrast Nadeau-Bengio & Random-Effects (DL + HKSJ) TOST
# ==============================================================================
def _dl_hksj_pool(yi: np.ndarray, vi: np.ndarray):
    yi = np.asarray(yi, dtype=float)
    vi = np.maximum(np.asarray(vi, dtype=float), 1e-12)
    k = len(yi)
    w = 1.0 / vi
    mu_fe = float(np.sum(w * yi) / np.sum(w))
    q_stat = float(np.sum(w * (yi - mu_fe) ** 2))
    df_q = k - 1
    c_val = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = float(max(0.0, (q_stat - df_q) / c_val)) if (c_val > 0 and df_q > 0) else 0.0
    w_re = 1.0 / (vi + tau2)
    mu_re = float(np.sum(w_re * yi) / np.sum(w_re))
    se_re_dl = float(np.sqrt(1.0 / np.sum(w_re)))
    q_hk = float(np.sum(w_re * (yi - mu_re) ** 2) / df_q) if df_q > 0 else 1.0
    se_hk = float(np.sqrt(max(1.0, q_hk)) * se_re_dl)
    i2 = float(max(0.0, (q_stat - df_q) / q_stat * 100.0)) if q_stat > df_q else 0.0
    t90 = float(stats.t.ppf(0.95, df_q))
    t95 = float(stats.t.ppf(0.975, df_q))
    p02 = max(float(1.0 - stats.t.cdf((mu_re + 0.02) / se_hk, df_q)), float(stats.t.cdf((mu_re - 0.02) / se_hk, df_q)))
    p04 = max(float(1.0 - stats.t.cdf((mu_re + 0.04) / se_hk, df_q)), float(stats.t.cdf((mu_re - 0.04) / se_hk, df_q)))
    return {
        "mu_re": mu_re, "se_re_dl": se_re_dl, "se_hk": se_hk,
        "ci90_lo": mu_re - t90 * se_hk, "ci90_hi": mu_re + t90 * se_hk,
        "ci95_lo": mu_re - t95 * se_hk, "ci95_hi": mu_re + t95 * se_hk,
        "tau2": tau2, "i2_percent": i2, "tost_p_002": p02, "tost_p_004": p04,
    }


def run_per_contrast_and_mixed_effects_tost():
    print("\n=== [v7 Step 1] Per-Contrast Nadeau-Bengio & Random-Effects (DL + HKSJ) TOST ===")
    df_cv = pd.read_csv(REPORTS_DIR / "benchmark_repeated_10x5_cv.csv")
    nn8_names = [c for c, _ in NONNESTED_8]
    sub = df_cv[df_cv["cohort"].isin(nn8_names)].copy()
    gnn_comps = ["GCN_Weighted_PLI", "GCN_Identity", "GCN_Random_EdgeMatched", "GCN_Full"]
    all_comps = [m for m in V6.MODELS_13 if m != "GCN_OMST"]

    rows = []
    per_c_cache = {}
    t90_r = float(stats.t.ppf(0.95, 9))
    t95_r = float(stats.t.ppf(0.975, 9))

    for m in all_comps:
        piv = sub[sub["model"].isin(["GCN_OMST", m])].pivot(index=["cohort", "repeat"], columns="model", values="auc")
        per_c_cache[m] = {}
        for c in nn8_names:
            x = piv.loc[c, "GCN_OMST"].values
            y = piv.loc[c, m].values
            d_r = x - y
            md = float(np.mean(d_r))
            sd_r = float(np.std(d_r, ddof=1))
            se_naive = float(sd_r / np.sqrt(10.0))
            # Nadeau-Bengio (2003) SE correction for 10 repeats of 5-fold CV (n_test / n_train = 0.25)
            se_nb = float(sd_r * np.sqrt(1.0 / 10.0 + 0.25)) + 1e-12
            p02 = max(float(1.0 - stats.t.cdf((md + 0.02) / se_nb, 9)), float(stats.t.cdf((md - 0.02) / se_nb, 9)))
            p04 = max(float(1.0 - stats.t.cdf((md + 0.04) / se_nb, 9)), float(stats.t.cdf((md - 0.04) / se_nb, 9)))
            rec = {
                "record_type": "Per_Contrast", "scope": "Single_Contrast", "cohort": c, "k_contrasts": 1,
                "comparator_model": m, "is_gnn_family": m in gnn_comps,
                "gcn_mean_auc": float(np.mean(x)), "comp_mean_auc": float(np.mean(y)),
                "mean_diff_auc": md, "repeat_sd": sd_r, "se_naive": se_naive, "se_nb": se_nb,
                "se_re_dl": np.nan, "se_hk": se_nb,
                "ci90_lo": md - t90_r * se_nb, "ci90_hi": md + t90_r * se_nb,
                "ci95_lo": md - t95_r * se_nb, "ci95_hi": md + t95_r * se_nb,
                "tau2": np.nan, "i2_percent": np.nan,
                "tost_p_002": p02, "tost_p_004": p04,
                "tost02_holm_gnn4": np.nan, "tost04_holm_gnn4": np.nan,
                "n_contrasts_abs_le_002": int(abs(md) <= 0.02),
                "n_contrasts_abs_le_004": int(abs(md) <= 0.04),
                "n_contrasts_tost04_pass": int(p04 < 0.05),
            }
            rows.append(rec)
            per_c_cache[m][c] = rec

    scopes = [
        ("k=8 Non-Nested", nn8_names),
        ("k=7 Drop 5b (AD Only)", [c for c in nn8_names if not c.startswith("Study 5b:")]),
        ("k=7 Drop 5a (FTD Only)", [c for c in nn8_names if not c.startswith("Study 5a:")]),
        ("k=7 Excl. Study 2", [c for c in nn8_names if not c.startswith("Study 2:")]),
    ]
    for scope_lbl, idxs in scopes:
        tmp = []
        for m in all_comps:
            c_recs = [per_c_cache[m][c] for c in idxs]
            yi = np.array([r["mean_diff_auc"] for r in c_recs])
            vi = np.array([r["se_nb"] ** 2 for r in c_recs])
            pool = _dl_hksj_pool(yi, vi)
            tmp.append({
                "record_type": "Random_Effects_Pool", "scope": scope_lbl, "cohort": "Pooled_DL_HKSJ_NB", "k_contrasts": len(idxs),
                "comparator_model": m, "is_gnn_family": m in gnn_comps,
                "gcn_mean_auc": float(np.mean([r["gcn_mean_auc"] for r in c_recs])),
                "comp_mean_auc": float(np.mean([r["comp_mean_auc"] for r in c_recs])),
                "mean_diff_auc": pool["mu_re"], "repeat_sd": float(np.std(yi, ddof=1)),
                "se_naive": np.nan, "se_nb": float(np.mean([r["se_nb"] for r in c_recs])),
                "se_re_dl": pool["se_re_dl"], "se_hk": pool["se_hk"],
                "ci90_lo": pool["ci90_lo"], "ci90_hi": pool["ci90_hi"],
                "ci95_lo": pool["ci95_lo"], "ci95_hi": pool["ci95_hi"],
                "tau2": pool["tau2"], "i2_percent": pool["i2_percent"],
                "tost_p_002": pool["tost_p_002"], "tost_p_004": pool["tost_p_004"],
                "tost02_holm_gnn4": np.nan, "tost04_holm_gnn4": np.nan,
                "n_contrasts_abs_le_002": int(sum(r["n_contrasts_abs_le_002"] for r in c_recs)),
                "n_contrasts_abs_le_004": int(sum(r["n_contrasts_abs_le_004"] for r in c_recs)),
                "n_contrasts_tost04_pass": int(sum(r["n_contrasts_tost04_pass"] for r in c_recs)),
            })
        gnn_idx = [i for i, r in enumerate(tmp) if r["is_gnn_family"]]
        _, h02, _, _ = multipletests([tmp[i]["tost_p_002"] for i in gnn_idx], method="holm")
        _, h04, _, _ = multipletests([tmp[i]["tost_p_004"] for i in gnn_idx], method="holm")
        for j, i in enumerate(gnn_idx):
            tmp[i]["tost02_holm_gnn4"] = float(h02[j])
            tmp[i]["tost04_holm_gnn4"] = float(h04[j])
        rows.extend(tmp)

    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_DIR / "v7_per_contrast_and_mixed_effects_tost.csv", index=False)
    print(out[(out["scope"] == "k=8 Non-Nested") & (out["is_gnn_family"])][
        ["comparator_model", "mean_diff_auc", "ci90_lo", "ci90_hi", "i2_percent", "tost02_holm_gnn4", "tost04_holm_gnn4", "n_contrasts_abs_le_002"]
    ].round(4).to_string(index=False))
    print("  ✓ Saved reports/v7_per_contrast_and_mixed_effects_tost.csv")


# ==============================================================================
# Step 2: True Fold-Wise AIRM Fréchet-Mean Whitening Audit
# ==============================================================================
def _spd_pow(mat: np.ndarray, p: float) -> np.ndarray:
    w, v = np.linalg.eigh(0.5 * (mat + mat.T))
    return (v * (np.maximum(w, 1e-12) ** p)) @ v.T


def _spd_log(mat: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(0.5 * (mat + mat.T))
    return (v * np.log(np.maximum(w, 1e-12))) @ v.T


def _spd_exp(sym: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(0.5 * (sym + sym.T))
    return (v * np.exp(np.clip(w, -30.0, 30.0))) @ v.T


def compute_airm_frechet_mean(mats: np.ndarray, max_iter: int = 12, tol: float = 1e-7) -> np.ndarray:
    """Karcher / Affine-Invariant Riemannian Metric (AIRM) geometric Fréchet mean on 19x19 SPD."""
    m = _spd_exp(np.mean([_spd_log(c) for c in mats], axis=0))
    for _ in range(max_iter):
        m_inv2 = _spd_pow(m, -0.5)
        m_pos2 = _spd_pow(m, 0.5)
        step = np.mean([_spd_log(m_inv2 @ c @ m_inv2) for c in mats], axis=0)
        m = m_pos2 @ _spd_exp(step) @ m_pos2
        if np.linalg.norm(step, "fro") < tol:
            break
    return 0.5 * (m + m.T)


def _upper_vec(mat: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(19)
    w = np.ones_like(mat) * np.sqrt(2.0)
    np.fill_diagonal(w, 1.0)
    return (mat * w)[idx].astype(np.float32)


def _airm_repeat_worker(task):
    cname, r_idx, seed, y_all, grps, c_all, x_id_all, x_leaky_raw = task
    torch.set_num_threads(1)
    set_seed(seed)
    splits = get_grouped_splits(grps, y_all, n_splits=5, seed=seed)
    regs = ("Identity_LogEuclidean_Z", "Foldwise_AIRM_Frechet_Z", "Foldwise_AIRM_Frechet_Raw", "Leaky_Global_AIRM_Raw")
    clfs = ("LR", "Ridge", "RF")
    oof = {(reg, clf): np.zeros(len(y_all)) for reg in regs for clf in clfs}

    for fold, (tr, va) in enumerate(splits):
        ytr = y_all[tr]
        icv = get_grouped_splits(grps[tr], ytr, n_splits=3, seed=seed + fold)

        # 1. Identity Log-Euclidean + z-score
        mu_i, sd_i = x_id_all[tr].mean(0, keepdims=True), x_id_all[tr].std(0, keepdims=True) + 1e-6
        xtr_i, xva_i = (x_id_all[tr] - mu_i) / sd_i, (x_id_all[va] - mu_i) / sd_i

        # 2. Strict Fold-wise AIRM Fréchet mean congruence whitening
        m_tr = compute_airm_frechet_mean(c_all[tr], max_iter=12, tol=1e-7)
        m_tr_inv2 = _spd_pow(m_tr, -0.5)
        x_airm_raw = np.stack([_upper_vec(_spd_log(m_tr_inv2 @ c @ m_tr_inv2)) for c in c_all])
        xtr_ar, xva_ar = x_airm_raw[tr], x_airm_raw[va]
        mu_a, sd_a = xtr_ar.mean(0, keepdims=True), xtr_ar.std(0, keepdims=True) + 1e-6
        xtr_az, xva_az = (xtr_ar - mu_a) / sd_a, (xva_ar - mu_a) / sd_a

        # 3. Leaky global AIRM (fit mean across tr+va, raw tangent coords)
        xtr_lk, xva_lk = x_leaky_raw[tr], x_leaky_raw[va]

        feat_map = {
            "Identity_LogEuclidean_Z": (xtr_i, xva_i),
            "Foldwise_AIRM_Frechet_Z": (xtr_az, xva_az),
            "Foldwise_AIRM_Frechet_Raw": (xtr_ar, xva_ar),
            "Leaky_Global_AIRM_Raw": (xtr_lk, xva_lk),
        }
        for reg_name, (xtr, xva) in feat_map.items():
            gs_lr = GridSearchCV(
                LogisticRegression(max_iter=1000, random_state=seed),
                {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}, cv=icv, scoring="roc_auc", n_jobs=1,
            ).fit(xtr, ytr)
            oof[(reg_name, "LR")][va] = gs_lr.predict_proba(xva)[:, 1]
            gs_rc = GridSearchCV(
                RidgeClassifier(random_state=seed),
                {"alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}, cv=icv, scoring="roc_auc", n_jobs=1,
            ).fit(xtr, ytr)
            oof[(reg_name, "Ridge")][va] = 1.0 / (1.0 + np.exp(-gs_rc.decision_function(xva)))
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed, n_jobs=1).fit(xtr, ytr)
            oof[(reg_name, "RF")][va] = rf.predict_proba(xva)[:, 1]

    res = {}
    for k, prob in oof.items():
        res[k] = float(roc_auc_score(y_all, prob))
    return cname, r_idx, res


def run_airm_frechet_audit(sp):
    print("\n=== [v7 Step 2] True Fold-Wise AIRM Fréchet-Mean Whitening Tangent-Space Audit ===")
    t0 = time.time()
    covs = torch.load(DATA_DIR / "spatial_covariances_7cohorts.pt", weights_only=False)
    tasks = []
    cohort_names = []
    for cname, ckey in NONNESTED_8:
        vg = _filter_cohort_records(cname, ckey, sp[ckey])
        y_all = np.array([int(g["y"]) for g in vg])
        grps = np.array([g.get("bids_id", g.get("sub_name", g["subject_id"])) for g in vg])
        c_list = []
        for g in vg:
            sid = g["subject_id"]
            k = (f"{sid}_{g['session']}" if g.get("group") == "PD" else sid) if ckey == "unm_ds003490" else sid
            c_mat = np.asarray(covs[ckey][k], dtype=np.float64)
            c_list.append(0.5 * (c_mat + c_mat.T))
        c_all = np.stack(c_list)
        x_id_all = np.stack([_upper_vec(_spd_log(c)) for c in c_all])
        m_glob = compute_airm_frechet_mean(c_all, max_iter=12, tol=1e-7)
        m_glob_inv2 = _spd_pow(m_glob, -0.5)
        x_leaky_raw = np.stack([_upper_vec(_spd_log(m_glob_inv2 @ c @ m_glob_inv2)) for c in c_all])
        cohort_names.append(cname)
        for r_idx, seed in enumerate(REPEATS_10):
            tasks.append((cname, r_idx, seed, y_all, grps, c_all, x_id_all, x_leaky_raw))

    ctx = mp.get_context("spawn")
    per_cohort = {c: [] for c in cohort_names}
    with ProcessPoolExecutor(max_workers=8, mp_context=ctx) as ex:
        for cname, r_idx, res in ex.map(_airm_repeat_worker, tasks):
            per_cohort[cname].append(res)

    bench_rows, compat_rows = [], []
    regs = ("Identity_LogEuclidean_Z", "Foldwise_AIRM_Frechet_Z", "Foldwise_AIRM_Frechet_Raw", "Leaky_Global_AIRM_Raw")
    for cname in cohort_names:
        rlist = per_cohort[cname]
        for clf in ("LR", "Ridge", "RF"):
            id_aucs = [r[("Identity_LogEuclidean_Z", clf)] for r in rlist]
            fz_aucs = [r[("Foldwise_AIRM_Frechet_Z", clf)] for r in rlist]
            fr_aucs = [r[("Foldwise_AIRM_Frechet_Raw", clf)] for r in rlist]
            lk_aucs = [r[("Leaky_Global_AIRM_Raw", clf)] for r in rlist]
            id_m, fz_m, fr_m, lk_m = float(np.mean(id_aucs)), float(np.mean(fz_aucs)), float(np.mean(fr_aucs)), float(np.mean(lk_aucs))
            for reg_lbl, vals in [
                ("Identity_LogEuclidean_Z", id_aucs),
                ("Foldwise_AIRM_Frechet_Z", fz_aucs),
                ("Foldwise_AIRM_Frechet_Raw", fr_aucs),
                ("Leaky_Global_AIRM_Raw", lk_aucs),
            ]:
                bench_rows.append({
                    "cohort": cname, "representation": reg_lbl, "classifier": clf,
                    "auc_mean": float(np.mean(vals)), "auc_sd": float(np.std(vals, ddof=1)),
                    "delta_vs_id_z": float(np.mean(vals) - id_m),
                })
            compat_rows.append({
                "cohort": cname, "classifier": clf,
                "auc_identity_ref": id_m, "sd_identity": float(np.std(id_aucs, ddof=1)),
                "auc_fold_frechet_ref": fz_m, "sd_fold_frechet": float(np.std(fz_aucs, ddof=1)),
                "delta_frechet_minus_id": fz_m - id_m,
                "auc_fold_airm_raw": fr_m, "auc_leaky_global_airm_raw": lk_m,
                "delta_leaky_airm_leakage": lk_m - fr_m,
            })

    df_b = pd.DataFrame(bench_rows)
    df_c = pd.DataFrame(compat_rows)
    df_b.to_csv(REPORTS_DIR / "v7_airm_foldwise_riemannian_benchmark.csv", index=False)
    df_c.to_csv(REPORTS_DIR / "spatial_cov_foldwise_frechet_check.csv", index=False)
    print(df_c.groupby("classifier")[["auc_identity_ref", "auc_fold_frechet_ref", "delta_frechet_minus_id", "auc_fold_airm_raw", "delta_leaky_airm_leakage"]].mean().round(4).to_string())
    print(f"  ✓ Saved AIRM benchmarks ({time.time() - t0:.1f}s)")


# ==============================================================================
# Step 3: Compact Spectral-Spatial CNN (ShallowSpecCNN) & Inner-Tuned GCN-OMST
# ==============================================================================
class ShallowSpecCNN(nn.Module):
    """Compact spectral-spatial CNN modeled after EEGNet / ShallowConvNet on [19 ch x 57 freq bins]."""

    def __init__(self, n_ch: int = 19, n_classes: int = 2, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=(n_ch, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(16, 16, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_eval_shallow_spec_cnn(tz_psd: torch.Tensor, ty: torch.Tensor, vz_psd: torch.Tensor, epochs: int = 45, lr: float = 0.003, wd: float = 1e-3, seed: int = 42):
    set_seed(seed)
    model = ShallowSpecCNN(n_ch=tz_psd.shape[2], n_classes=2, dropout=0.25)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(model(tz_psd), ty)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(vz_psd)
        return torch.softmax(logits, dim=-1)[:, 1].numpy()


def train_eval_gcn_cfg(tz, tadj, ty, vz, vadj, hidden_dim=32, lr=0.005, wd=1e-4, dropout=0.25, epochs=50, seed=42):
    set_seed(seed)
    model = NormativeGCN(in_node_features=tz.shape[-1], hidden_dim=hidden_dim, num_classes=2, dropout=dropout)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(model(tz, tadj), ty)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(vz, vadj)
        return torch.softmax(logits, dim=-1)[:, 1].numpy()


def train_eval_gcn_omst_tuned(tx, tadj, ty, tgrps, vx, vadj, seed=42):
    cands = [
        (16, 0.005, 1e-3, 0.25, 40),
        (32, 0.005, 1e-4, 0.25, 50),
        (32, 0.003, 1e-3, 0.10, 50),
    ]
    icv = get_grouped_splits(tgrps, ty.numpy(), n_splits=3, seed=seed)
    best_auc, best_cfg = -1.0, cands[1]
    for cfg in cands:
        hd, lr, wd, dr, ep = cfg
        oof_in = np.zeros(len(ty))
        for itr, iva in icv:
            tz_in, vz_in, _, _ = compute_fold_normative_z(tx[itr], ty[itr], tx[iva])
            oof_in[iva] = train_eval_gcn_cfg(
                tz_in, tadj[itr], ty[itr], vz_in, tadj[iva],
                hidden_dim=hd, lr=lr, wd=wd, dropout=dr, epochs=ep, seed=seed,
            )
        auc_in = float(roc_auc_score(ty.numpy(), oof_in))
        if auc_in > best_auc:
            best_auc = auc_in
            best_cfg = cfg
    tz_out, vz_out, _, _ = compute_fold_normative_z(tx, ty, vx)
    hd, lr, wd, dr, ep = best_cfg
    return train_eval_gcn_cfg(
        tz_out, tadj, ty, vz_out, vadj,
        hidden_dim=hd, lr=lr, wd=wd, dropout=dr, epochs=ep, seed=seed,
    )


def _deep_gcn_repeat_worker(task):
    cname, r_idx, seed, y_all, grps, x_gcn, adj_omst, log_psd_all = task
    torch.set_num_threads(1)
    set_seed(seed)
    splits = get_grouped_splits(grps, y_all, n_splits=5, seed=seed)
    oof_cnn = np.zeros(len(y_all))
    oof_tuned_gcn = np.zeros(len(y_all))

    for fold, (tr, va) in enumerate(splits):
        ty = torch.tensor(y_all[tr], dtype=torch.long)
        # 1. ShallowSpecCNN on fold-wise HC-z-scored [2, 30] Hz log-PSD [B, 1, 19, 57]
        tz_p, vz_p, _, _ = compute_fold_normative_z(log_psd_all[tr], ty, log_psd_all[va])
        oof_cnn[va] = train_eval_shallow_spec_cnn(tz_p.unsqueeze(1), ty, vz_p.unsqueeze(1), epochs=45, lr=0.003, wd=1e-3, seed=seed + fold)

        # 2. Inner 3-fold tuned GCN-OMST
        oof_tuned_gcn[va] = train_eval_gcn_omst_tuned(
            x_gcn[tr], adj_omst[tr], ty, grps[tr], x_gcn[va], adj_omst[va], seed=seed + fold,
        )
    return cname, r_idx, float(roc_auc_score(y_all, oof_cnn)), float(roc_auc_score(y_all, oof_tuned_gcn))


def run_deep_and_tuned_gcn(sp, ps):
    print("\n=== [v7 Step 3] ShallowSpecCNN & Inner-Tuned GCN-OMST-Tuned across k=8 Contrasts (10x5 CV) ===")
    t0 = time.time()
    tasks = []
    meta_c = {}
    for cname, ckey in NONNESTED_8:
        vg_sp = _filter_cohort_records(cname, ckey, sp[ckey])
        ps_map = {g["subject_id"] + "_" + str(g.get("session", "")): g for g in ps[ckey]}
        ps_list = []
        for g in vg_sp:
            pk = g["subject_id"] + "_" + str(g.get("session", ""))
            rec_p = ps_map[pk]
            freqs = rec_p["freqs"]
            m230 = (freqs >= 2.0) & (freqs <= 30.0)
            lp = np.log10(np.maximum(rec_p["mean_psd"][:, m230], 1e-25)).astype(np.float32)
            ps_list.append(torch.tensor(lp))
        y_all = np.array([int(g["y"]) for g in vg_sp])
        grps = np.array([g.get("bids_id", g.get("sub_name", g["subject_id"])) for g in vg_sp])
        x_gcn = torch.stack([g["x"] for g in vg_sp])
        adj_omst = torch.stack([g["adj"] for g in vg_sp])
        log_psd_all = torch.stack(ps_list)
        meta_c[cname] = (len(vg_sp), len(np.unique(grps)))
        for r_idx, seed in enumerate(REPEATS_10):
            tasks.append((cname, r_idx, seed, y_all, grps, x_gcn, adj_omst, log_psd_all))

    ctx = mp.get_context("spawn")
    res_map = {c: {"ShallowSpecCNN": [], "GCN_OMST_Tuned": []} for c, _ in NONNESTED_8}
    with ProcessPoolExecutor(max_workers=8, mp_context=ctx) as ex:
        for cname, r_idx, auc_cnn, auc_tgcn in ex.map(_deep_gcn_repeat_worker, tasks):
            res_map[cname]["ShallowSpecCNN"].append(auc_cnn)
            res_map[cname]["GCN_OMST_Tuned"].append(auc_tgcn)

    df_sum = pd.read_csv(REPORTS_DIR / "benchmark_repeated_10x5_summary.csv")
    df_sc = pd.read_csv(REPORTS_DIR / "spatial_cov_foldwise_frechet_check.csv")
    piv_s = df_sum.pivot(index="cohort", columns="model", values="mean_auc")
    piv_sd = df_sum.pivot(index="cohort", columns="model", values="std_auc")
    airm_rf = df_sc[df_sc["classifier"] == "RF"].set_index("cohort")

    out_rows = []
    for cname, _ in NONNESTED_8:
        ns, n_p = meta_c[cname]
        omst_m = float(piv_s.loc[cname, "GCN_OMST"])
        svm_m = float(piv_s.loc[cname, "Linear_SVM"])
        for m in ["ShallowSpecCNN", "GCN_OMST_Tuned"]:
            vals = res_map[cname][m]
            m_auc, sd_auc = float(np.mean(vals)), float(np.std(vals, ddof=1))
            out_rows.append({
                "cohort": cname, "model": m, "n_sessions": ns, "n_participants": n_p,
                "mean_auc": m_auc, "sd_auc": sd_auc,
                "delta_vs_gcn_omst": m_auc - omst_m, "delta_vs_linear_svm": m_auc - svm_m,
            })
        for m in ["GCN_OMST", "GCN_Random_EdgeMatched", "Linear_SVM", "RandomForest"]:
            m_auc, sd_auc = float(piv_s.loc[cname, m]), float(piv_sd.loc[cname, m])
            out_rows.append({
                "cohort": cname, "model": m, "n_sessions": ns, "n_participants": n_p,
                "mean_auc": m_auc, "sd_auc": sd_auc,
                "delta_vs_gcn_omst": m_auc - omst_m, "delta_vs_linear_svm": m_auc - svm_m,
            })
        m_auc = float(airm_rf.loc[cname, "auc_fold_frechet_ref"])
        sd_auc = float(airm_rf.loc[cname, "sd_fold_frechet"])
        out_rows.append({
            "cohort": cname, "model": "Foldwise_AIRM_RF", "n_sessions": ns, "n_participants": n_p,
            "mean_auc": m_auc, "sd_auc": sd_auc,
            "delta_vs_gcn_omst": m_auc - omst_m, "delta_vs_linear_svm": m_auc - svm_m,
        })

    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(REPORTS_DIR / "v7_deep_baseline_and_tuned_gcn_10x5.csv", index=False)
    print(df_out.groupby("model")[["mean_auc", "delta_vs_gcn_omst", "delta_vs_linear_svm"]].mean().round(4).to_string())
    print(f"  ✓ Saved reports/v7_deep_baseline_and_tuned_gcn_10x5.csv ({time.time() - t0:.1f}s)")


# ==============================================================================
# Step 4: Sample-Size Learning Curves (5x5 Grouped CV, Frac 0.25..1.00)
# ==============================================================================
def _lc_repeat_worker(task):
    cname, r_idx, seed, y_all, grps, x_gcn, adj_omst, edge_counts, log_psd_all = task
    torch.set_num_threads(1)
    set_seed(seed)
    splits = get_grouped_splits(grps, y_all, n_splits=5, seed=seed)
    adj_rnd_all = torch.stack([make_edge_matched_random_adj(19, edge_counts[i], seed=seed * 1000 + i) for i in range(len(y_all))])
    fracs = [0.25, 0.50, 0.75, 1.00]
    models = ["GCN_OMST", "GCN_Random_EdgeMatched", "ShallowSpecCNN", "Linear_SVM", "RandomForest"]
    oof = {(f, m): np.zeros(len(y_all)) for f in fracs for m in models}
    tr_sizes = {f: [] for f in fracs}

    for fold, (tr, va) in enumerate(splits):
        tr_grps_uniq = []
        seen_g = set()
        for idx_tr in tr:
            g_id = grps[idx_tr]
            if g_id not in seen_g:
                seen_g.add(g_id)
                tr_grps_uniq.append((g_id, int(y_all[idx_tr])))
        cls0 = [g for g, yv in tr_grps_uniq if yv == 0]
        cls1 = [g for g, yv in tr_grps_uniq if yv == 1]

        for f in fracs:
            if f >= 0.999:
                sub_tr = tr
            else:
                rng = np.random.RandomState(seed * 100 + fold * 10 + int(round(f * 100)))
                n0 = max(3, int(round(len(cls0) * f)))
                n1 = max(3, int(round(len(cls1) * f)))
                keep_g = set(rng.choice(cls0, size=min(n0, len(cls0)), replace=False).tolist() +
                             rng.choice(cls1, size=min(n1, len(cls1)), replace=False).tolist())
                sub_tr = np.array([idx_tr for idx_tr in tr if grps[idx_tr] in keep_g], dtype=int)
            tr_sizes[f].append(len(sub_tr))

            ty = torch.tensor(y_all[sub_tr], dtype=torch.long)
            tz, vz, _, _ = compute_fold_normative_z(x_gcn[sub_tr], ty, x_gcn[va])
            oof[(f, "GCN_OMST")][va] = train_eval_gcn_cfg(tz, adj_omst[sub_tr], ty, vz, adj_omst[va], seed=seed + fold)
            oof[(f, "GCN_Random_EdgeMatched")][va] = train_eval_gcn_cfg(tz, adj_rnd_all[sub_tr], ty, vz, adj_rnd_all[va], seed=seed + fold)

            tz_p, vz_p, _, _ = compute_fold_normative_z(log_psd_all[sub_tr], ty, log_psd_all[va])
            oof[(f, "ShallowSpecCNN")][va] = train_eval_shallow_spec_cnn(tz_p.unsqueeze(1), ty, vz_p.unsqueeze(1), seed=seed + fold)

            xtr_flat, xva_flat = tz.view(len(sub_tr), -1).numpy(), vz.view(len(va), -1).numpy()
            y_sub = y_all[sub_tr]
            icv = get_grouped_splits(grps[sub_tr], y_sub, n_splits=min(3, len(np.unique(grps[sub_tr])) // 2 or 2), seed=seed + fold)
            # Ensure every inner validation split has both classes
            valid_icv = [(itr, iva) for itr, iva in icv if len(np.unique(y_sub[itr])) == 2 and len(np.unique(y_sub[iva])) == 2]
            if len(valid_icv) < 2:
                valid_icv = 2
            gs_svm = GridSearchCV(SVC(kernel="linear", probability=True, random_state=seed), {"C": [0.005, 0.01, 0.05, 0.1, 0.5, 1.0]}, cv=valid_icv, scoring="roc_auc", n_jobs=1).fit(xtr_flat, y_sub)
            oof[(f, "Linear_SVM")][va] = gs_svm.predict_proba(xva_flat)[:, 1]
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed, n_jobs=1).fit(xtr_flat, y_sub)
            oof[(f, "RandomForest")][va] = rf.predict_proba(xva_flat)[:, 1]

    aucs = {(f, m): float(roc_auc_score(y_all, oof[(f, m)])) for f in fracs for m in models}
    mean_tr = {f: float(np.mean(tr_sizes[f])) for f in fracs}
    return cname, r_idx, aucs, mean_tr


def run_learning_curves(sp, ps):
    print("\n=== [v7 Step 4] Sample-Size Learning Curves (3 Largest Cohorts, 5x5 Grouped CV) ===")
    t0 = time.time()
    lc_cohorts = [
        ("Study 7: Iowa Parkinson (ds004584)", "iowa_ds004584"),
        ("Study 1: Sleep Deprivation (ds004902)", "sleep_ds004902"),
        ("Study 3: Cavanagh University Depression", "cavanagh_ds003478"),
    ]
    tasks = []
    n_tot_map = {}
    for cname, ckey in lc_cohorts:
        vg_sp = _filter_cohort_records(cname, ckey, sp[ckey])
        ps_map = {g["subject_id"] + "_" + str(g.get("session", "")): g for g in ps[ckey]}
        ps_list = []
        for g in vg_sp:
            rec_p = ps_map[g["subject_id"] + "_" + str(g.get("session", ""))]
            freqs = rec_p["freqs"]
            m230 = (freqs >= 2.0) & (freqs <= 30.0)
            ps_list.append(torch.tensor(np.log10(np.maximum(rec_p["mean_psd"][:, m230], 1e-25)).astype(np.float32)))
        y_all = np.array([int(g["y"]) for g in vg_sp])
        grps = np.array([g.get("bids_id", g.get("sub_name", g["subject_id"])) for g in vg_sp])
        x_gcn = torch.stack([g["x"] for g in vg_sp])
        adj_omst = torch.stack([g["adj"] for g in vg_sp])
        edge_counts = [max(1, int(g.get("omst_edge_count", 27))) for g in vg_sp]
        log_psd_all = torch.stack(ps_list)
        n_tot_map[cname] = len(vg_sp)
        for r_idx, seed in enumerate(REPEATS_5):
            tasks.append((cname, r_idx, seed, y_all, grps, x_gcn, adj_omst, edge_counts, log_psd_all))

    ctx = mp.get_context("spawn")
    res_c = {c: [] for c, _ in lc_cohorts}
    tr_c = {c: [] for c, _ in lc_cohorts}
    with ProcessPoolExecutor(max_workers=8, mp_context=ctx) as ex:
        for cname, r_idx, aucs, mtr in ex.map(_lc_repeat_worker, tasks):
            res_c[cname].append(aucs)
            tr_c[cname].append(mtr)

    fracs = [0.25, 0.50, 0.75, 1.00]
    models = ["GCN_OMST", "GCN_Random_EdgeMatched", "ShallowSpecCNN", "Linear_SVM", "RandomForest"]
    rows = []
    for cname, _ in lc_cohorts:
        for f in fracs:
            mtr_f = float(np.mean([d[f] for d in tr_c[cname]]))
            omst_vals = [d[(f, "GCN_OMST")] for d in res_c[cname]]
            rnd_vals = [d[(f, "GCN_Random_EdgeMatched")] for d in res_c[cname]]
            svm_vals = [d[(f, "Linear_SVM")] for d in res_c[cname]]
            delta_rnd = float(np.mean(omst_vals) - np.mean(rnd_vals))
            delta_svm = float(np.mean(omst_vals) - np.mean(svm_vals))
            for m in models:
                vals = [d[(f, m)] for d in res_c[cname]]
                rows.append({
                    "cohort": cname, "n_total_samples": n_tot_map[cname],
                    "train_fraction": f, "mean_n_train": mtr_f, "model": m,
                    "mean_auc": float(np.mean(vals)), "sd_auc": float(np.std(vals, ddof=1)),
                    "delta_omst_minus_rnd": delta_rnd, "delta_omst_minus_svm": delta_svm,
                })

    df_lc = pd.DataFrame(rows)
    df_lc.to_csv(REPORTS_DIR / "v7_sample_size_learning_curves.csv", index=False)
    print(df_lc.pivot_table(index=["cohort", "train_fraction", "mean_n_train"], columns="model", values="mean_auc")[models].round(3).to_string())
    print(f"  ✓ Saved reports/v7_sample_size_learning_curves.csv ({time.time() - t0:.1f}s)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-step", type=int, default=1)
    args = parser.parse_args()

    sp = torch.load(DATA_DIR / "specparam_canonical_7cohorts.pt", weights_only=False)
    ps = torch.load(DATA_DIR / "all_7cohorts_welch_psds.pt", weights_only=False)

    if args.from_step <= 1:
        run_per_contrast_and_mixed_effects_tost()
    if args.from_step <= 2:
        run_airm_frechet_audit(sp)
    if args.from_step <= 3:
        run_deep_and_tuned_gcn(sp, ps)
    if args.from_step <= 4:
        run_learning_curves(sp, ps)
    print("\n🎉 ALL v7 TARGETED REMEDIATION ANALYSES COMPLETE!")


if __name__ == "__main__":
    main()
