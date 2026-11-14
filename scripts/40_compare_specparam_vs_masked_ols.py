#!/usr/bin/env python3
"""
Step 40: Head-to-Head Ablation of Genuine specparam (v2.0.0rc7) vs. [7, 14] Hz-Masked OLS
Across Biophysical Effect Sizes, Polarity Meta-Analysis, 7-Cohort Classification,
and Tri-Center Parkinson Transfer + Gain Perturbation.

Reads:
  - data/derivatives/specparam_vs_masked_ols_7cohorts.pt
  - data/derivatives/all_7cohorts_welch_psds.pt

Outputs:
  - reports/ablation_specparam_vs_ols_biophysics.csv
  - reports/ablation_specparam_vs_ols_biophysics_wide.csv
  - reports/ablation_specparam_vs_ols_meta_analysis.csv
  - reports/ablation_specparam_vs_ols_classification.csv
  - reports/ablation_specparam_vs_ols_tricenter_transfer.csv
"""

import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from specparam import SpectralGroupModel
from statsmodels.stats.multitest import multipletests
import torch
import torch.nn as nn
import torch.optim as optim

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = BENCHMARK_ROOT / "src"
sys.path.append(str(SRC_DIR))
from harmonizer_and_graph import CHANNELS_19, NormativeGCN, compute_fold_normative_z

DATA_DIR = BENCHMARK_ROOT / "data" / "derivatives"
REPORTS_DIR = BENCHMARK_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FOCAL_5 = [CHANNELS_19.index(c) for c in ["Fp1", "Fp2", "F3", "F4", "Fz"]]
FOCAL_3 = [CHANNELS_19.index(c) for c in ["F3", "Fz", "F4"]]
POSTERIOR_5 = [CHANNELS_19.index(c) for c in ["P3", "P4", "Pz", "O1", "O2"]]

METHODS = [
    ("masked_ols", "x_masked_ols"),
    ("specparam_resid", "x_specparam_resid"),
    ("specparam_peaks", "x_specparam_peaks"),
]

SPECPARAM_KWARGS = {
    "aperiodic_mode": "fixed",
    "periodic_mode": "gaussian",
    "peak_width_limits": [1.0, 12.0],
    "max_n_peaks": 6,
    "min_peak_height": 0.05,
    "peak_threshold": 2.0,
    "metrics": ["error_mae", "gof_rsquared"],
    "verbose": False,
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def welch_df(x: np.ndarray, y: np.ndarray) -> float:
    n1, n2 = len(x), len(y)
    v1, v2 = np.var(x, ddof=1) / n1, np.var(y, ddof=1) / n2
    return float(((v1 + v2) ** 2) / ((v1 ** 2) / (n1 - 1) + (v2 ** 2) / (n2 - 1) + 1e-15))


def calc_stats(cases: List[dict], ctrls: List[dict], x_key: str, col_idx: int, ch_idx, is_paired: bool) -> dict:
    def extract_val(g):
        arr = g[x_key].numpy()
        vals = arr[:, col_idx] if ch_idx is None else arr[ch_idx, col_idx]
        return float(np.nanmean(vals))

    vc = np.array([extract_val(g) for g in cases], dtype=np.float64)
    vr = np.array([extract_val(g) for g in ctrls], dtype=np.float64)
    nc, nr = len(vc), len(vr)
    if is_paired:
        diff = vc - vr
        d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))
        se_d = float(np.sqrt(1.0 / nc + (d ** 2) / (2.0 * nc)))
        t_stat, p_val = stats.ttest_rel(vc, vr)
        df_val = float(nc - 1)
    else:
        sp = np.sqrt(((nc - 1) * np.var(vc, ddof=1) + (nr - 1) * np.var(vr, ddof=1)) / (nc + nr - 2)) + 1e-12
        d = float((np.mean(vc) - np.mean(vr)) / sp)
        se_d = float(np.sqrt((nc + nr) / (nc * nr) + (d ** 2) / (2.0 * (nc + nr))))
        t_stat, p_val = stats.ttest_ind(vc, vr, equal_var=False)
        df_val = welch_df(vc, vr)
    return {
        "n_case": nc,
        "n_control": nr,
        "is_paired": is_paired,
        "mean_case": float(np.mean(vc)),
        "mean_control": float(np.mean(vr)),
        "cohens_d": d,
        "se_d": se_d,
        "ci_lower": d - 1.96 * se_d,
        "ci_upper": d + 1.96 * se_d,
        "t_statistic": float(t_stat),
        "df": df_val,
        "p_value": float(p_val),
    }


def make_ucsd_subj_mean(ucsd_recs: List[dict]) -> List[dict]:
    by_bid: Dict[str, List[dict]] = {}
    for g in ucsd_recs:
        if "PD" in g.get("group", ""):
            by_bid.setdefault(g.get("bids_id", g["subject_id"]), []).append(g)
    out = []
    for bid in sorted(by_bid.keys()):
        pair = by_bid[bid]
        d = {"bids_id": bid, "subject_id": bid, "group": "PD_SubjMean"}
        for _, xk in METHODS:
            d[xk] = torch.nanmean(torch.stack([g[xk] for g in pair]), dim=0)
        out.append(d)
    return out


def run_biophysics_and_meta(data: Dict[str, List[dict]]):
    print("\n" + "=" * 78)
    print("📊 [Part A & B] Biophysical Contrasts & Meta-Analysis: Masked OLS vs Genuine specparam")
    print("=" * 78)

    sleep = data["sleep_ds004902"]
    mumtaz = data["mumtaz_ec"]
    cav = data["cavanagh_ds003478"]
    ucsd = data["ucsd_ds002778"]
    dem = data["dementia_ds004504"]
    unm = data["unm_ds003490"]
    iowa = data["iowa_ds004584"]

    sd_sleep = sorted([g for g in sleep if g.get("condition") == "SD"], key=lambda g: g["subject_id"])
    ns_sleep = sorted([g for g in sleep if g.get("condition") == "NS"], key=lambda g: g["subject_id"])
    ucsd_pd_mean = make_ucsd_subj_mean(ucsd)
    ucsd_hc = [g for g in ucsd if g.get("group") == "HC"]

    comparisons_10 = [
        ("Study 1: Acute Sleep Deprivation", sd_sleep, ns_sleep, True),
        ("Study 2: Outpatient MDD", [g for g in mumtaz if g.get("group") == "MDD"], [g for g in mumtaz if g.get("group") == "HC"], False),
        ("Study 3: Univ. Depression Screening", [g for g in cav if g.get("group") == "Depressed"], [g for g in cav if g.get("group") == "Control"], False),
        ("Study 4: UCSD Parkinson All", ucsd_pd_mean, ucsd_hc, False),
        ("Study 5a: Alzheimer's Disease (AD)", [g for g in dem if g.get("diagnosis") == "Alzheimer"], [g for g in dem if g.get("diagnosis") == "Control"], False),
        ("Study 5b: Frontotemporal Dementia (FTD)", [g for g in dem if g.get("diagnosis") == "FTD"], [g for g in dem if g.get("diagnosis") == "Control"], False),
        ("Study 5c: Clinical Dementia (All)", [g for g in dem if g.get("diagnosis") in ("Alzheimer", "FTD")], [g for g in dem if g.get("diagnosis") == "Control"], False),
        ("Study 6a: UNM Parkinson OFF", [g for g in unm if g.get("group") == "PD" and g.get("med_state") == "OFF"], [g for g in unm if g.get("group") == "CTL"], False),
        ("Study 6b: UNM Parkinson ON", [g for g in unm if g.get("group") == "PD" and g.get("med_state") == "ON"], [g for g in unm if g.get("group") == "CTL"], False),
        ("Study 7: Iowa Parkinson Cohort", [g for g in iowa if g.get("group") == "PD"], [g for g in iowa if g.get("group") == "Control"], False),
    ]

    markers = [
        ("frontal_theta_5ch", 0, FOCAL_5),
        ("frontal_theta_3ch", 0, FOCAL_3),
        ("global_delta", 1, None),
        ("posterior_alpha", 2, POSTERIOR_5),
        ("global_chi", 3, None),
    ]

    rows = []
    for m_name, x_key in METHODS:
        for c_label, cases, ctrls, is_paired in comparisons_10:
            for mk_name, col_idx, ch_idx in markers:
                st = calc_stats(cases, ctrls, x_key, col_idx, ch_idx, is_paired)
                rows.append({"method": m_name, "cohort": c_label, "marker": mk_name, **st})

    df_bp = pd.DataFrame(rows)
    # Compute within-(method, marker) 10-contrast BH-FDR (matching Table 1 q_10)
    df_bp["fdr_q_10"] = np.nan
    for (m_name, mk_name), idx in df_bp.groupby(["method", "marker"]).groups.items():
        _, q10, _, _ = multipletests(df_bp.loc[idx, "p_value"].values, method="fdr_bh")
        df_bp.loc[idx, "fdr_q_10"] = q10

    # Also compute 40-test BH-FDR across the 4 primary EEG markers x 10 contrasts per method
    df_bp["fdr_q_40_eeg"] = np.nan
    primary4 = ["frontal_theta_5ch", "global_delta", "posterior_alpha", "global_chi"]
    for m_name in [m[0] for m in METHODS]:
        mask = (df_bp["method"] == m_name) & (df_bp["marker"].isin(primary4))
        _, q40, _, _ = multipletests(df_bp.loc[mask, "p_value"].values, method="fdr_bh")
        df_bp.loc[mask, "fdr_q_40_eeg"] = q40

    bp_csv = REPORTS_DIR / "ablation_specparam_vs_ols_biophysics.csv"
    df_bp.to_csv(bp_csv, index=False)

    # Wide side-by-side summary table for key markers
    piv = df_bp.pivot(index=["marker", "cohort"], columns="method", values=["cohens_d", "p_value", "fdr_q_10"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    df_wide = piv.reset_index()
    df_wide.to_csv(REPORTS_DIR / "ablation_specparam_vs_ols_biophysics_wide.csv", index=False)

    print("\n--- Frontal Periodic Theta (5-ch) Head-to-Head Across 10 Contrasts ---")
    ft = df_wide[df_wide["marker"] == "frontal_theta_5ch"].set_index("cohort")
    for c_label, _, _, _ in comparisons_10:
        r = ft.loc[c_label]
        print(
            f"  {c_label:<38} | OLS: d={r['cohens_d_masked_ols']:+.3f} (q={r['fdr_q_10_masked_ols']:.3e}) | "
            f"SP-Resid: d={r['cohens_d_specparam_resid']:+.3f} (q={r['fdr_q_10_specparam_resid']:.3e}) | "
            f"SP-Peaks: d={r['cohens_d_specparam_peaks']:+.3f} (q={r['fdr_q_10_specparam_peaks']:.3e})"
        )

    print("\n--- Global Aperiodic Exponent chi Head-to-Head Across 10 Contrasts ---")
    gc = df_wide[df_wide["marker"] == "global_chi"].set_index("cohort")
    for c_label, _, _, _ in comparisons_10:
        r = gc.loc[c_label]
        print(
            f"  {c_label:<38} | OLS: d={r['cohens_d_masked_ols']:+.3f} (q={r['fdr_q_10_masked_ols']:.3e}) | "
            f"SP-Real: d={r['cohens_d_specparam_resid']:+.3f} (q={r['fdr_q_10_specparam_resid']:.3e})"
        )

    # Part B: Random-effects meta-analysis on frontal_theta_5ch (exact scripts/34 DerSimonian-Laird)
    def dl_meta(d_vals, var_vals):
        d_a, v_a = np.array(d_vals, dtype=float), np.array(var_vals, dtype=float)
        k = len(d_a)
        w = 1.0 / v_a
        mu_fe = np.sum(w * d_a) / np.sum(w)
        q = float(np.sum(w * (d_a - mu_fe) ** 2))
        df = k - 1
        c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w)) if df > 0 else 1.0
        tau2 = max(0.0, (q - df) / c) if df > 0 else 0.0
        w_re = 1.0 / (v_a + tau2)
        d_re = float(np.sum(w_re * d_a) / np.sum(w_re))
        se_re = float(np.sqrt(1.0 / np.sum(w_re)))
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(d_re / se_re))))
        i2 = max(0.0, 100.0 * (q - df) / q) if q > 0 and df > 0 else 0.0
        return {
            "k": k, "pooled_d": d_re, "se_d": se_re,
            "ci_lower": d_re - 1.96 * se_re, "ci_upper": d_re + 1.96 * se_re,
            "p_value": p_val, "Q_stat": q, "I2_percent": i2, "tau2": tau2,
        }

    def study_theta(m_name, x_key, cases, ctrls, is_paired=False):
        st = calc_stats(cases, ctrls, x_key, 0, FOCAL_5, is_paired)
        return st["cohens_d"], st["se_d"] ** 2

    def sleep_theta_ig(x_key):
        vc = np.array([float(np.nanmean(g[x_key].numpy()[FOCAL_5, 0])) for g in sd_sleep])
        vr = np.array([float(np.nanmean(g[x_key].numpy()[FOCAL_5, 0])) for g in ns_sleep])
        r = float(np.corrcoef(vc, vr)[0, 1])
        sp = np.sqrt(0.5 * (np.var(vc, ddof=1) + np.var(vr, ddof=1)))
        d_ig = float((np.mean(vc) - np.mean(vr)) / sp)
        n = len(vc)
        var_ig = float((1.0 / n + (d_ig ** 2) / (2.0 * n)) * 2.0 * (1.0 - r))
        return d_ig, var_ig

    meta_rows = []
    print("\n--- Independent-Cohort Polarity Meta-Analysis (Frontal Periodic Theta 5-ch) ---")
    for m_name, x_key in METHODS:
        s_map = {
            "Mumtaz_MDD_EC": study_theta(m_name, x_key, [g for g in mumtaz if g.get("group") == "MDD"], [g for g in mumtaz if g.get("group") == "HC"]),
            "Cavanagh_DEP": study_theta(m_name, x_key, [g for g in cav if g.get("group") == "Depressed"], [g for g in cav if g.get("group") == "Control"]),
            "UCSD_PD_OFF": study_theta(m_name, x_key, [g for g in ucsd if g.get("group") == "PD_OFF"], [g for g in ucsd if g.get("group") == "HC"]),
            "UCSD_PD_ON": study_theta(m_name, x_key, [g for g in ucsd if g.get("group") == "PD_ON"], [g for g in ucsd if g.get("group") == "HC"]),
            "UCSD_PD_SubjMean": study_theta(m_name, x_key, ucsd_pd_mean, ucsd_hc),
            "Dementia_All": study_theta(m_name, x_key, [g for g in dem if g.get("diagnosis") in ("Alzheimer", "FTD")], [g for g in dem if g.get("diagnosis") == "Control"]),
            "Dementia_AD": study_theta(m_name, x_key, [g for g in dem if g.get("diagnosis") == "Alzheimer"], [g for g in dem if g.get("diagnosis") == "Control"]),
            "Dementia_FTD": study_theta(m_name, x_key, [g for g in dem if g.get("diagnosis") == "FTD"], [g for g in dem if g.get("diagnosis") == "Control"]),
            "UNM_PD_OFF": study_theta(m_name, x_key, [g for g in unm if g.get("group") == "PD" and g.get("med_state") == "OFF"], [g for g in unm if g.get("group") == "CTL"]),
            "UNM_PD_ON": study_theta(m_name, x_key, [g for g in unm if g.get("group") == "PD" and g.get("med_state") == "ON"], [g for g in unm if g.get("group") == "CTL"]),
            "Iowa_PD": study_theta(m_name, x_key, [g for g in iowa if g.get("group") == "PD"], [g for g in iowa if g.get("group") == "Control"]),
            "Sleep_SD_NS_dIG": sleep_theta_ig(x_key),
        }

        configs = [
            ("Primary_K6_PD_OFF", ["Mumtaz_MDD_EC", "Cavanagh_DEP"], ["UCSD_PD_OFF", "Dementia_All", "UNM_PD_OFF", "Iowa_PD"]),
            ("Sens1_K6_UCSD_SubjMean", ["Mumtaz_MDD_EC", "Cavanagh_DEP"], ["UCSD_PD_SubjMean", "Dementia_All", "UNM_PD_OFF", "Iowa_PD"]),
            ("Sens2_K6_PD_ON", ["Mumtaz_MDD_EC", "Cavanagh_DEP"], ["UCSD_PD_ON", "Dementia_All", "UNM_PD_ON", "Iowa_PD"]),
            ("Sens3_K6_AD_Only", ["Mumtaz_MDD_EC", "Cavanagh_DEP"], ["UCSD_PD_OFF", "Dementia_AD", "UNM_PD_OFF", "Iowa_PD"]),
            ("Sens4_K6_FTD_Only", ["Mumtaz_MDD_EC", "Cavanagh_DEP"], ["UCSD_PD_OFF", "Dementia_FTD", "UNM_PD_OFF", "Iowa_PD"]),
            ("Sens5_K7_Plus_Sleep", ["Mumtaz_MDD_EC", "Cavanagh_DEP"], ["Sleep_SD_NS_dIG", "UCSD_PD_OFF", "Dementia_All", "UNM_PD_OFF", "Iowa_PD"]),
        ]

        for cfg_name, g1_keys, g2_keys in configs:
            d1, v1 = zip(*[s_map[k] for k in g1_keys])
            d2, v2 = zip(*[s_map[k] for k in g2_keys])
            r1 = dl_meta(d1, v1)
            r2 = dl_meta(d2, v2)
            r_all = dl_meta(list(d1) + list(d2), list(v1) + list(v2))
            q_bet = max(0.0, r_all["Q_stat"] - (r1["Q_stat"] + r2["Q_stat"]))
            p_bet = float(1.0 - stats.chi2.cdf(q_bet, 1))
            for dom, res in [("Depressive", r1), ("Slowing", r2), ("Pooled", r_all)]:
                meta_rows.append({
                    "method": m_name, "config": cfg_name, "domain": dom,
                    **res, "Q_between": q_bet, "p_between": p_bet,
                })
            if cfg_name in ("Primary_K6_PD_OFF", "Sens1_K6_UCSD_SubjMean", "Sens5_K7_Plus_Sleep"):
                print(
                    f"  {m_name:<16} | {cfg_name:<22} | Dep d={r1['pooled_d']:+.3f} (p={r1['p_value']:.3f}) | "
                    f"Slow d={r2['pooled_d']:+.3f} (p={r2['p_value']:.3e}) | Q_bet={q_bet:.2f} (p={p_bet:.3e})"
                )

    meta_csv = REPORTS_DIR / "ablation_specparam_vs_ols_meta_analysis.csv"
    pd.DataFrame(meta_rows).to_csv(meta_csv, index=False)
    print(f"✅ Saved {bp_csv.name} & {meta_csv.name}")


def train_eval_gcn(train_z, train_adj, train_y, val_z, val_adj, epochs=50, lr=0.005, weight_decay=1e-4, seed=42):
    set_seed(seed)
    model = NormativeGCN(in_node_features=4, hidden_dim=32, num_classes=2, dropout=0.25)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(train_z, train_adj), train_y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(val_z, val_adj)
        return torch.softmax(logits, dim=-1)[:, 1].numpy(), torch.argmax(logits, dim=-1).numpy()


def bootstrap_auc_ci(y_true, y_prob, n_boot: int = 1000):
    base = float(roc_auc_score(y_true, y_prob))
    rng = np.random.RandomState(42)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return base, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def eval_12cohort_cv(data: Dict[str, List[dict]]):
    print("\n" + "=" * 78)
    print("🤖 [Part C] Within-Cohort 5-Fold CV Across 12 Contrasts x 7 Models x 3 Feature Methods")
    print("=" * 78)

    sleep = data["sleep_ds004902"]
    mum_ec = data["mumtaz_ec"]
    mum_eo = data["mumtaz_eo"]
    cav = data["cavanagh_ds003478"]
    ucsd = data["ucsd_ds002778"]
    dem = data["dementia_ds004504"]
    unm = data["unm_ds003490"]
    iowa = data["iowa_ds004584"]

    contrasts = [
        ("Study 1: Sleep Deprivation (ds004902)", sleep, True),
        ("Study 2: Mumtaz MDD (Eyes Closed)", mum_ec, False),
        ("Study 2b: Mumtaz MDD (Eyes Open)", mum_eo, False),
        ("Study 3: Cavanagh University Depression", cav, False),
        ("Study 4: UCSD Parkinson All Sessions", ucsd, True),
        ("Study 5a: Dementia AD vs Control", [g for g in dem if g.get("diagnosis") in ("Alzheimer", "Control")], False),
        ("Study 5b: Dementia FTD vs Control", [g for g in dem if g.get("diagnosis") in ("FTD", "Control")], False),
        ("Study 5c: Dementia All (AD+FTD) vs Control", dem, False),
        ("Study 6a: UNM PD-OFF vs Control", [g for g in unm if g.get("med_state") in ("OFF", "CTL")], False),
        ("Study 6b: UNM PD-ON vs Control", [g for g in unm if g.get("med_state") in ("ON", "CTL")], False),
        ("Study 6c: UNM PD All Sessions", unm, True),
        ("Study 7: Iowa Parkinson (ds004584)", iowa, False),
    ]

    models = ["GCN_OMST", "GCN_Identity", "Tuned_LogisticRegression", "Tuned_Ridge", "RandomForest", "Single_Theta", "Single_Chi"]
    adj_identity = torch.eye(19, dtype=torch.float32)
    rows = []

    for c_name, g_list, use_group in contrasts:
        valid = [g for g in g_list if int(g["y"]) in (0, 1)]
        y_all = np.array([int(g["y"]) for g in valid])
        n = len(valid)
        groups = [g.get("bids_id", g.get("sub_name", g.get("subject_id", str(i)))) for i, g in enumerate(valid)]
        if use_group:
            splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(valid, y_all, groups=groups))
        else:
            splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(valid, y_all))

        for m_name, x_key in METHODS:
            set_seed(42)
            oof_p = {m: np.zeros(n) for m in models}
            oof_y = {m: np.zeros(n) for m in models}

            # Handle any rare channel NaNs cleanly by imputing train-fold channel mean if ever present
            x_tensor = torch.stack([g[x_key].clone() for g in valid])
            adj_omst = torch.stack([g["adj"] for g in valid])

            for fold, (tr_i, va_i) in enumerate(splits):
                tx = x_tensor[tr_i].clone()
                vx = x_tensor[va_i].clone()
                if torch.isnan(tx).any() or torch.isnan(vx).any():
                    col_means = torch.nanmean(tx, dim=0, keepdim=True)
                    col_means = torch.nan_to_num(col_means, nan=0.0)
                    tx = torch.where(torch.isnan(tx), col_means.expand_as(tx), tx)
                    vx = torch.where(torch.isnan(vx), col_means.expand_as(vx), vx)

                ty = torch.tensor(y_all[tr_i], dtype=torch.long)
                tz, vz, _, _ = compute_fold_normative_z(tx, ty, vx)
                X_tr = tz.view(len(tr_i), -1).numpy()
                X_va = vz.view(len(va_i), -1).numpy()
                y_tr = y_all[tr_i]

                p, pr = train_eval_gcn(tz, adj_omst[tr_i], ty, vz, adj_omst[va_i], seed=42 + fold)
                oof_p["GCN_OMST"][va_i], oof_y["GCN_OMST"][va_i] = p, pr

                tr_id = adj_identity.unsqueeze(0).expand(len(tr_i), -1, -1)
                va_id = adj_identity.unsqueeze(0).expand(len(va_i), -1, -1)
                p, pr = train_eval_gcn(tz, tr_id, ty, vz, va_id, seed=42 + fold)
                oof_p["GCN_Identity"][va_i], oof_y["GCN_Identity"][va_i] = p, pr

                inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                glr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}, cv=inner, scoring="roc_auc").fit(X_tr, y_tr)
                oof_p["Tuned_LogisticRegression"][va_i] = glr.predict_proba(X_va)[:, 1]
                oof_y["Tuned_LogisticRegression"][va_i] = glr.predict(X_va)

                grc = GridSearchCV(RidgeClassifier(random_state=42), {"alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}, cv=inner, scoring="roc_auc").fit(X_tr, y_tr)
                oof_p["Tuned_Ridge"][va_i] = 1.0 / (1.0 + np.exp(-grc.decision_function(X_va)))
                oof_y["Tuned_Ridge"][va_i] = grc.predict(X_va)

                rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, y_tr)
                oof_p["RandomForest"][va_i], oof_y["RandomForest"][va_i] = rf.predict_proba(X_va)[:, 1], rf.predict(X_va)

                th_tr = tz[:, FOCAL_5, 0].mean(dim=-1).unsqueeze(-1).numpy()
                th_va = vz[:, FOCAL_5, 0].mean(dim=-1).unsqueeze(-1).numpy()
                lrt = LogisticRegression(random_state=42).fit(th_tr, y_tr)
                oof_p["Single_Theta"][va_i], oof_y["Single_Theta"][va_i] = lrt.predict_proba(th_va)[:, 1], lrt.predict(th_va)

                ch_tr = tz[:, :, 3].mean(dim=-1).unsqueeze(-1).numpy()
                ch_va = vz[:, :, 3].mean(dim=-1).unsqueeze(-1).numpy()
                lrc = LogisticRegression(random_state=42).fit(ch_tr, y_tr)
                oof_p["Single_Chi"][va_i], oof_y["Single_Chi"][va_i] = lrc.predict_proba(ch_va)[:, 1], lrc.predict(ch_va)

            for mod in models:
                auc, lo, hi = bootstrap_auc_ci(y_all, oof_p[mod])
                bacc = balanced_accuracy_score(y_all, oof_y[mod])
                rows.append({
                    "cohort": c_name, "n_sessions": n, "method": m_name, "model": mod,
                    "auc": auc, "ci_lower": lo, "ci_upper": hi, "balanced_accuracy": bacc,
                })
        sub = [r for r in rows if r["cohort"] == c_name and r["model"] in ("GCN_OMST", "RandomForest", "Tuned_LogisticRegression", "Single_Theta")]
        summary_str = " | ".join(f"{r['method'][:6]}-{r['model'][:4]}:{r['auc']:.3f}" for r in sub)
        print(f"  {c_name[:34]:<34} | {summary_str}")

    df_cv = pd.DataFrame(rows)
    cv_csv = REPORTS_DIR / "ablation_specparam_vs_ols_classification.csv"
    df_cv.to_csv(cv_csv, index=False)

    print("\n--- Grand Mean AUC Across All 12 Contrasts (Masked OLS vs Genuine specparam) ---")
    gm = df_cv.groupby(["model", "method"])["auc"].mean().unstack()
    print(gm.round(4).to_string())
    print(f"✅ Saved {cv_csv.name}")


def compute_five_reps_from_psd(freqs: np.ndarray, mean_psd: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute masked OLS, relative power, and raw log power for a [19, n_freqs] PSD (specparam computed via batch SGM)."""
    vm = (freqs >= 1.0) & (freqs <= 40.0)
    f = freqs[vm]
    lf = np.log10(f)
    m_ap = (f < 7.0) | (f > 14.0)
    th = (f >= 4.0) & (f < 8.0)
    dl = (f >= 1.0) & (f < 4.0)
    al = (f >= 8.0) & (f <= 13.0)

    x_ols = np.zeros((19, 4), dtype=np.float32)
    x_rel = np.zeros((19, 4), dtype=np.float32)
    x_raw = np.zeros((19, 4), dtype=np.float32)
    for c in range(19):
        psd_c = np.maximum(mean_psd[c, vm], 1e-20)
        lp = np.log10(psd_c)
        s, b = np.polyfit(lf[m_ap], lp[m_ap], 1)
        fl = lp - (b + s * lf)
        x_ols[c] = [np.mean(fl[th]), np.mean(fl[dl]), np.mean(fl[al]), -s]

        tot = float(np.sum(psd_c)) + 1e-25
        lt, la = float(np.mean(psd_c[th])), float(np.mean(psd_c[al]))
        x_rel[c] = [np.sum(psd_c[th]) / tot, np.sum(psd_c[dl]) / tot, np.sum(psd_c[al]) / tot, np.log10((lt + 1e-25) / (la + 1e-25))]
        x_raw[c] = [np.mean(lp[th]), np.mean(lp[dl]), np.mean(lp[al]), np.mean(lp)]
    return {"masked_ols": x_ols, "relative_band_power": x_rel, "raw_log_band_power": x_raw}


def compute_batch_specparam(rec_list: List[dict], k_vals: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = rec_list[0]["freqs"]
    psds = []
    for r, kv in zip(rec_list, k_vals):
        psds.append(np.maximum((kv ** 2) * r["mean_psd"], 1e-25))
    sgm = SpectralGroupModel(**SPECPARAM_KWARGS)
    sgm.fit(freqs, np.vstack(psds), [1.0, 40.0], n_jobs=1)
    ff = sgm.data.freqs
    th, dl, al = (ff >= 4.0) & (ff < 8.0), (ff >= 1.0) & (ff < 4.0), (ff >= 8.0) & (ff <= 13.0)
    ap = sgm.get_params("aperiodic")
    null_s = set(sgm.results.null_inds or [])

    res_list, pk_list = [], []
    for si in range(len(rec_list)):
        xr = np.zeros((19, 4), dtype=np.float32)
        xp = np.zeros((19, 4), dtype=np.float32)
        for ch in range(19):
            fi = si * 19 + ch
            if fi in null_s:
                xr[ch] = np.nan
                xp[ch] = np.nan
            else:
                mi = sgm.get_model(fi)
                resid = mi.data.power_spectrum - mi.results.model._ap_fit
                pkf = mi.results.model._peak_fit
                chi = float(ap[fi, 1])
                xr[ch] = [np.mean(resid[th]), np.mean(resid[dl]), np.mean(resid[al]), chi]
                xp[ch] = [np.mean(pkf[th]), np.mean(pkf[dl]), np.mean(pkf[al]), chi]
        res_list.append(torch.tensor(xr, dtype=torch.float32))
        pk_list.append(torch.tensor(xp, dtype=torch.float32))
    return torch.nan_to_num(torch.stack(res_list), nan=0.0), torch.nan_to_num(torch.stack(pk_list), nan=0.0)


def eval_tricenter_transfer(all_psds: Dict[str, List[dict]]):
    print("\n" + "=" * 78)
    print("🌐 [Part D] Tri-Center PD Cross-Site Transfer & Gain Perturbation (5 Representations)")
    print("=" * 78)

    site_recs = {
        "UCSD": [r for r in all_psds["ucsd_ds002778"] if int(r["y"]) in (0, 1)],
        "UNM_OFF": [r for r in all_psds["unm_ds003490"] if r.get("med_state") in ("OFF", "CTL") and int(r["y"]) in (0, 1)],
        "Iowa": [r for r in all_psds["iowa_ds004584"] if int(r["y"]) in (0, 1)],
    }

    rep_names = ["specparam_resid", "specparam_peaks", "masked_ols", "relative_band_power", "raw_log_band_power"]

    def build_site_tensors(k_mode=1.0):
        out = {}
        for s_name, recs in site_recs.items():
            ks = [(0.5 if i % 2 == 0 else 5.0) if k_mode == "Heterogeneous_0.5x_5.0x" else float(k_mode) for i in range(len(recs))]
            sp_res, sp_pks = compute_batch_specparam(recs, ks)
            ols_l, rel_l, raw_l = [], [], []
            for r, kv in zip(recs, ks):
                reps = compute_five_reps_from_psd(r["freqs"], (kv ** 2) * r["mean_psd"])
                ols_l.append(torch.tensor(reps["masked_ols"], dtype=torch.float32))
                rel_l.append(torch.tensor(reps["relative_band_power"], dtype=torch.float32))
                raw_l.append(torch.tensor(reps["raw_log_band_power"], dtype=torch.float32))
            out[s_name] = {
                "specparam_resid": sp_res,
                "specparam_peaks": sp_pks,
                "masked_ols": torch.stack(ols_l),
                "relative_band_power": torch.stack(rel_l),
                "raw_log_band_power": torch.stack(raw_l),
                "adj": torch.stack([r["adj"] for r in recs]),
                "y": torch.tensor([int(r["y"]) for r in recs], dtype=torch.long),
                "y_np": np.array([int(r["y"]) for r in recs]),
            }
        return out

    base_tensors = build_site_tensors(1.0)
    pairs = [("UCSD", "UNM_OFF"), ("UCSD", "Iowa"), ("UNM_OFF", "UCSD"), ("UNM_OFF", "Iowa"), ("Iowa", "UCSD"), ("Iowa", "UNM_OFF")]
    records = []

    # Calibration comparison at k = 1.0
    for src_n, tgt_n in pairs:
        src, tgt = base_tensors[src_n], base_tensors[tgt_n]
        sy, ty = src["y"], tgt["y_np"]
        for rep in rep_names:
            for calib in ["Uncalibrated_Raw", "Global_Cohort_ZScore", "Target_HC_Normative"]:
                set_seed(42)
                sx, tx = src[rep], tgt[rep]
                if calib == "Uncalibrated_Raw":
                    tz, vz = sx, tx
                elif calib == "Global_Cohort_ZScore":
                    tz = (sx - sx.mean(dim=0, keepdim=True)) / (sx.std(dim=0, keepdim=True) + 1e-6)
                    vz = (tx - tx.mean(dim=0, keepdim=True)) / (tx.std(dim=0, keepdim=True) + 1e-6)
                else:
                    _, _, smu, ssd = compute_fold_normative_z(sx, sy, sx)
                    _, _, tmu, tsd = compute_fold_normative_z(tx, tgt["y"], tx)
                    tz, vz = (sx - smu) / ssd, (tx - tmu) / tsd

                Xtr, Xte = tz.view(len(sy), -1).numpy(), vz.view(len(ty), -1).numpy()
                lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42).fit(Xtr, sy.numpy())
                plr = lr.predict_proba(Xte)[:, 1]
                rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(Xtr, sy.numpy())
                prf = rf.predict_proba(Xte)[:, 1]
                pgcn, pred_gcn = train_eval_gcn(tz, src["adj"], sy, vz, tgt["adj"], epochs=40, seed=42)

                records.append({
                    "experiment": "Calibration_Strategy", "source": src_n, "target": tgt_n,
                    "calibration": calib, "gain_k": "1.0", "representation": rep,
                    "auc_lr": roc_auc_score(ty, plr), "bacc_lr": balanced_accuracy_score(ty, (plr >= 0.5).astype(int)), "brier_lr": brier_score_loss(ty, plr),
                    "auc_rf": roc_auc_score(ty, prf), "bacc_rf": balanced_accuracy_score(ty, (prf >= 0.5).astype(int)),
                    "auc_gcn": roc_auc_score(ty, pgcn), "bacc_gcn": balanced_accuracy_score(ty, pred_gcn),
                    "feature_abs_drift": 0.0, "prob_abs_drift_lr": 0.0,
                })

    # Gain perturbation under Uncalibrated_Raw
    gain_settings = [0.5, 1.0, 2.0, 5.0, "Heterogeneous_0.5x_5.0x"]
    scaled_tensors = {k: (base_tensors if k == 1.0 else build_site_tensors(k)) for k in gain_settings}
    for src_n, tgt_n in pairs:
        src, tgt_b = base_tensors[src_n], base_tensors[tgt_n]
        sy, ty = src["y"], tgt_b["y_np"]
        for rep in rep_names:
            Xtr = src[rep].view(len(sy), -1).numpy()
            Xb = tgt_b[rep].view(len(ty), -1).numpy()
            lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42).fit(Xtr, sy.numpy())
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(Xtr, sy.numpy())
            p_base = lr.predict_proba(Xb)[:, 1]
            for k in gain_settings:
                Xk = scaled_tensors[k][tgt_n][rep].view(len(ty), -1).numpy()
                pk = lr.predict_proba(Xk)[:, 1]
                prfk = rf.predict_proba(Xk)[:, 1]
                records.append({
                    "experiment": "Gain_Perturbation", "source": src_n, "target": tgt_n,
                    "calibration": "Uncalibrated_Raw", "gain_k": str(k), "representation": rep,
                    "auc_lr": roc_auc_score(ty, pk), "bacc_lr": balanced_accuracy_score(ty, (pk >= 0.5).astype(int)), "brier_lr": brier_score_loss(ty, pk),
                    "auc_rf": roc_auc_score(ty, prfk), "bacc_rf": balanced_accuracy_score(ty, (prfk >= 0.5).astype(int)),
                    "auc_gcn": np.nan, "bacc_gcn": np.nan,
                    "feature_abs_drift": float(np.mean(np.abs(Xk - Xb))),
                    "prob_abs_drift_lr": float(np.mean(np.abs(pk - p_base))),
                })

    df_tr = pd.DataFrame(records)
    tr_csv = REPORTS_DIR / "ablation_specparam_vs_ols_tricenter_transfer.csv"
    df_tr.to_csv(tr_csv, index=False)

    print("\n--- Tri-Center Mean AUC (6 Directed Pairs, Calibration at k=1.0) ---")
    sub_c = df_tr[df_tr["experiment"] == "Calibration_Strategy"].groupby(["calibration", "representation"])[["auc_lr", "auc_rf", "auc_gcn"]].mean()
    print(sub_c.round(4).to_string())

    print("\n--- Tri-Center Heterogeneous Gain Perturbation (0.5x / 5.0x) ---")
    sub_h = df_tr[(df_tr["experiment"] == "Gain_Perturbation") & (df_tr["gain_k"] == "Heterogeneous_0.5x_5.0x")].groupby("representation")[["feature_abs_drift", "prob_abs_drift_lr", "auc_lr", "bacc_lr", "auc_rf", "bacc_rf"]].mean()
    print(sub_h.round(4).to_string())
    print(f"✅ Saved {tr_csv.name}")


def main():
    t0 = time.time()
    sp_path = DATA_DIR / "specparam_vs_masked_ols_7cohorts.pt"
    psd_path = DATA_DIR / "all_7cohorts_welch_psds.pt"
    if not sp_path.exists() or not psd_path.exists():
        raise FileNotFoundError("Run scripts/39_extract_all_welch_psds_and_run_specparam.py first.")

    data = torch.load(sp_path, weights_only=False)
    all_psds = torch.load(psd_path, weights_only=False)

    run_biophysics_and_meta(data)
    eval_12cohort_cv(data)
    eval_tricenter_transfer(all_psds)
    print(f"\n✨ Step 40 ablation completed in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
