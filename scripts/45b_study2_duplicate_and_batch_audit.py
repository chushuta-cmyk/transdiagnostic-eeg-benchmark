#!/usr/bin/env python3
"""
scripts/45b_study2_duplicate_and_batch_audit.py
------------------------------------------------
Forensic audit of Figshare 4244171 (Study 2: Outpatient MDD):
1. Identifies exact SHA-256 duplicate EDF files across raw Figshare release.
2. Evaluates biophysics & 10x5 CV ROC-AUC before vs. after deduplicating
   byte-identical pairs (H S27 == H S30, MDD S33 == MDD S34) across EC and EO.
3. Quantifies year/month acquisition-batch & checkerboard phase-artifact effects
   inside Study 2 HC (2011 vs. Jan-2013 vs. Apr-Jul 2013) and date-restricted MDD vs. HC.
"""

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = BENCHMARK_ROOT / "src"
sys.path.append(str(SRC_DIR))
from harmonizer_and_graph import CHANNELS_19, compute_fold_normative_z

DATA_DIR = BENCHMARK_ROOT / "data" / "derivatives"
REPORTS_DIR = BENCHMARK_ROOT / "reports"
MUMTAZ_DIR = Path("/Users/tan/Downloads/4244171")

FOCAL_5 = [CHANNELS_19.index(c) for c in ["Fp1", "Fp2", "F3", "F4", "Fz"]]
POSTERIOR_5 = [CHANNELS_19.index(c) for c in ["P3", "P4", "Pz", "O1", "O2"]]
REPEATS_10 = [42, 123, 456, 789, 2026, 101, 202, 303, 404, 505]


def get_stratified_splits(y, n_splits=5, seed=42):
    rng = np.random.RandomState(seed)
    idx0 = np.where(y == 0)[0].copy()
    idx1 = np.where(y == 1)[0].copy()
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    folds = [[] for _ in range(n_splits)]
    for k, i in enumerate(idx0):
        folds[k % n_splits].append(i)
    for k, i in enumerate(idx1):
        folds[k % n_splits].append(i)
    all_idx = np.arange(len(y))
    splits = []
    for k in range(n_splits):
        va = np.array(sorted(folds[k]), dtype=int)
        tr = np.setdiff1d(all_idx, va)
        splits.append((tr, va))
    return splits


def eval_10x5_tabular(glist):
    vg = [g for g in glist if int(g["y"]) in (0, 1)]
    X = torch.stack([g["x_specparam_resid"] for g in vg])
    y = np.array([int(g["y"]) for g in vg])
    aucs_lr, aucs_rf = [], []
    for s in REPEATS_10:
        splits = get_stratified_splits(y, n_splits=5, seed=s)
        p_lr, p_rf = np.zeros(len(vg)), np.zeros(len(vg))
        for fold, (tr, va) in enumerate(splits):
            tz, vz, _, _ = compute_fold_normative_z(X[tr], torch.tensor(y[tr], dtype=torch.long), X[va])
            Xtr, Xva, ytr = tz.view(len(tr), -1).numpy(), vz.view(len(va), -1).numpy(), y[tr]
            icv = get_stratified_splits(ytr, n_splits=3, seed=s + fold)
            lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=s), {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}, cv=icv, scoring="roc_auc").fit(Xtr, ytr)
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=s).fit(Xtr, ytr)
            p_lr[va] = lr.predict_proba(Xva)[:, 1]
            p_rf[va] = rf.predict_proba(Xva)[:, 1]
        aucs_lr.append(float(roc_auc_score(y, p_lr)))
        aucs_rf.append(float(roc_auc_score(y, p_rf)))
    return float(np.mean(aucs_lr)), float(np.std(aucs_lr, ddof=1)), float(np.mean(aucs_rf)), float(np.std(aucs_rf, ddof=1))


def cohens_d_welch(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    sp = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2)) + 1e-12
    d = float((a.mean() - b.mean()) / sp)
    _, p = stats.ttest_ind(a, b, equal_var=False)
    return d, float(p)


def main():
    by_hash = {}
    for p in sorted(MUMTAZ_DIR.glob("*.edf")):
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        by_hash.setdefault(h, []).append(p.name)
    dup_rows = [{"sha256_prefix": h, "n_files": len(ns), "filenames": "; ".join(ns)} for h, ns in sorted(by_hash.items(), key=lambda kv: kv[1][0]) if len(ns) > 1]
    pd.DataFrame(dup_rows).to_csv(REPORTS_DIR / "mumtaz_duplicate_edf_hashes.csv", index=False)

    sp = torch.load(DATA_DIR / "specparam_vs_masked_ols_7cohorts.pt", weights_only=False)
    diag = pd.read_csv(REPORTS_DIR / "mumtaz_batch_and_pli_diagnostic.csv").set_index("subject_id")

    ec_all = sp["mumtaz_ec"]
    eo_all = sp["mumtaz_eo"]
    dup_drop = {"mumtaz_hc_030", "mumtaz_mdd_034"}

    ec_dedup = [g for g in ec_all if g["subject_id"] not in dup_drop]
    eo_dedup = [g for g in eo_all if g["subject_id"] not in dup_drop]

    # Exclude Jan-2013 HC phase-artifact cluster + indefinite subjects cleanly
    ec_psd_pli_both = [g for g in ec_dedup if g["subject_id"] in diag.index and not bool(diag.loc[g["subject_id"], "is_indefinite"])]
    ec_vs_hc2011 = [g for g in ec_dedup if g["y"] == 1 or (g["subject_id"] in diag.index and str(diag.loc[g["subject_id"], "meas_date"]).startswith("2011"))]
    ec_vs_hc_no_jan13 = [g for g in ec_dedup if g["y"] == 1 or (g["subject_id"] in diag.index and not str(diag.loc[g["subject_id"], "meas_date"]).startswith("2013-01"))]

    rows = []
    for lbl, glist in [
        ("Study 2a EC: Public Manifest (30 MDD vs 28 HC, N=58)", ec_all),
        ("Study 2a EC: Deduplicated Unique (29 MDD vs 27 HC, N=56)", ec_dedup),
        ("Study 2a EC: Deduplicated Excl. Jan-2013 HC Batch (29 MDD vs 13 HC, N=42)", ec_vs_hc_no_jan13),
        ("Study 2a EC: Deduplicated vs 2011 HC Batch Only (29 MDD vs 9 HC, N=38)", ec_vs_hc2011),
        ("Study 2a EC: Deduplicated + Strictly PSD PLI Both Groups (22 MDD vs 12 HC, N=34)", ec_psd_pli_both),
        ("Study 2b EO: Public Manifest (32 MDD vs 28 HC, N=60)", eo_all),
        ("Study 2b EO: Deduplicated Unique (31 MDD vs 27 HC, N=58)", eo_dedup),
    ]:
        cases = [g for g in glist if g["y"] == 1]
        ctrls = [g for g in glist if g["y"] == 0]
        th_d, th_p = cohens_d_welch([g["x_specparam_resid"].numpy()[FOCAL_5, 0].mean() for g in cases], [g["x_specparam_resid"].numpy()[FOCAL_5, 0].mean() for g in ctrls])
        ch_d, ch_p = cohens_d_welch([g["x_specparam_resid"].numpy()[:, 3].mean() for g in cases], [g["x_specparam_resid"].numpy()[:, 3].mean() for g in ctrls])
        lr_m, lr_s, rf_m, rf_s = eval_10x5_tabular(glist)
        rows.append({
            "subset": lbl, "n_total": len(glist), "n_case": len(cases), "n_ctrl": len(ctrls),
            "theta_d": th_d, "theta_p": th_p, "chi_d": ch_d, "chi_p": ch_p,
            "lr_10x5_mean": lr_m, "lr_10x5_sd": lr_s, "rf_10x5_mean": rf_m, "rf_10x5_sd": rf_s,
        })

    # HC-only batch check (2011 HCs vs Jan-2013 artifact HCs vs Apr-Jul 2013 HCs)
    hc_2011 = [g for g in ec_dedup if g["y"] == 0 and str(diag.loc[g["subject_id"], "meas_date"]).startswith("2011")]
    hc_jan13 = [g for g in ec_dedup if g["y"] == 0 and str(diag.loc[g["subject_id"], "meas_date"]).startswith("2013-01")]
    hc_late13 = [g for g in ec_dedup if g["y"] == 0 and str(diag.loc[g["subject_id"], "meas_date"]).startswith(("2013-04", "2013-05", "2013-06", "2013-07"))]
    mdd_ec_u = [g for g in ec_dedup if g["y"] == 1]
    th_hc_d, th_hc_p = cohens_d_welch([g["x_specparam_resid"].numpy()[FOCAL_5, 0].mean() for g in hc_jan13], [g["x_specparam_resid"].numpy()[FOCAL_5, 0].mean() for g in hc_2011])
    ch_hc_d, ch_hc_p = cohens_d_welch([g["x_specparam_resid"].numpy()[:, 3].mean() for g in hc_jan13], [g["x_specparam_resid"].numpy()[:, 3].mean() for g in hc_2011])
    th_sub_d, th_sub_p = cohens_d_welch([g["x_specparam_resid"].numpy()[FOCAL_5, 0].mean() for g in mdd_ec_u], [g["x_specparam_resid"].numpy()[FOCAL_5, 0].mean() for g in hc_2011])
    print(f"Batch counts (unique HCs): 2011={len(hc_2011)}, Jan-2013={len(hc_jan13)}, Apr-Jul-2013={len(hc_late13)}")
    print(f"HC Jan-2013 vs HC 2011: theta d={th_hc_d:+.3f} (p={th_hc_p:.4f}), chi d={ch_hc_d:+.3f} (p={ch_hc_p:.4f})")
    print(f"MDD (2013) vs HC (2011 only): theta d={th_sub_d:+.3f} (p={th_sub_p:.4f})")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(REPORTS_DIR / "study2_duplicate_and_batch_sensitivity.csv", index=False)
    print("\n=== Study 2 Duplicate & Batch Forensic Sensitivity ===")
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
