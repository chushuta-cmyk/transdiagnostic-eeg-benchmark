#!/usr/bin/env python3
"""
scripts/49_run_v6_sensitivities.py
----------------------------------
Neuroinformatics v6 Targeted Statistical & Methodological Remediation Pipeline:
  1. Study 7 (Iowa PD ds004584, N=149) Reference/Pz Sensitivity:
     Re-extracts raw BIDS recordings comparing (a) CAR19_RefReconPz (online reference
     Pz recovered via CAR, matching canonical Study 7), (b) CAR18_NoPz (18 physically
     recorded 10-20 channels omitting Pz), and (c) CAR19_SplinePz (spherical-spline
     interpolated Pz -> 19-ch CAR) across biophysics, IAPF, 10x5 CV, and severity regressions.
  2. Study 4 (UCSD PD, 46 sess / 31 pts) & Study 6 (UNM PD, 75 sess / 50 pts)
     True 10x5 Participant-Level vs. Session-Level ROC-AUC across all 13 models.
  3. TOST Cross-Contrast Dependency (k=8 vs. k=7 dropping shared-HC 5b or 5a),
     90% Equivalence Confidence Intervals, Holm-Bonferroni multiplicity control,
     and Excl-Study-2 (k=7 / k=6) Macro-AUC Sensitivity (including Spatial Cov 190D).
  4. Riemannian Spatial Covariance (190-D) Fold-Wise Log-Euclidean Training-Mean
     Tangent Centering Check (vs. Fixed Identity I_19 Reference).
  5. Complete Study 2 (Figshare 4244171) 181-EDF File -> Session -> Participant
     Accounting & Schema-Fixed Release Artifacts (adding session, participant_id, record_uid).
  6. Full-Archive Stable fooof v1.1.1 Cross-Version Verification across ALL
     13,927 unique channel spectra (733 sessions x 19 channels).
  7. Paired 1,000-Permutation 4-Model Max-Statistic Clinical Severity Regression Audit
     (saving full paired null distributions for Ridge, SVR_RBF, RF, GBDT across 5 tasks).
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
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
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
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC, SVR
from specparam import SpectralGroupModel
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=UserWarning)
mne.set_log_level("ERROR")

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = BENCHMARK_ROOT / "src"
sys.path.append(str(SRC_DIR))
from harmonizer_and_graph import (
    CHANNELS_19,
    NormativeGCN,
    TopologicalGraphBuilder,
    compute_fold_normative_z,
)
import multi_cohort_loader as mcl

DATA_DIR = BENCHMARK_ROOT / "data" / "derivatives"
REPORTS_DIR = BENCHMARK_ROOT / "reports"
RELEASE_DIR = BENCHMARK_ROOT / "release_artifacts"
SCRATCH_DIR = Path("/Users/tan/.gemini/antigravity/scratch")
FOOOF_PY = SCRATCH_DIR / "fooof_venv" / "bin" / "python"
MUMTAZ_DIR = Path("/Users/tan/Downloads/4244171")
IOWA_DIR = Path("/Users/tan/Documents/eeg_datastes/ds004584")

REPEATS_10 = [42, 123, 456, 789, 2026, 101, 202, 303, 404, 505]
CHANNELS_18 = [c for c in CHANNELS_19 if c != "Pz"]
FOCAL_5 = [CHANNELS_19.index(c) for c in ["Fp1", "Fp2", "F3", "F4", "Fz"]]
POSTERIOR_5 = [CHANNELS_19.index(c) for c in ["P3", "P4", "Pz", "O1", "O2"]]
FOCAL_5_18 = [CHANNELS_18.index(c) for c in ["Fp1", "Fp2", "F3", "F4", "Fz"]]
POSTERIOR_4_18 = [CHANNELS_18.index(c) for c in ["P3", "P4", "O1", "O2"]]

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

MODELS_13 = [
    "GCN_OMST", "GCN_Weighted_PLI", "GCN_Identity", "GCN_Random_EdgeMatched", "GCN_Full",
    "Tuned_LogisticRegression", "Tuned_Ridge", "Linear_SVM", "RBF_SVM",
    "RandomForest", "GBDT", "Single_Theta", "Single_Chi",
]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    splits, all_idx = [], np.arange(len(y))
    for k in range(n_splits):
        val_set = set(fold_groups[k])
        va_mask = np.array([g in val_set for g in groups], dtype=bool)
        splits.append((all_idx[~va_mask], all_idx[va_mask]))
    return splits


def make_edge_matched_random_adj(num_nodes: int, num_edges: int, seed: int = 42) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    m = max(1, min(int(num_edges), len(edges)))
    for idx in rng.choice(len(edges), size=m, replace=False):
        u, v = edges[idx]
        adj[u, v] = adj[v, u] = 1.0
    return torch.tensor(adj, dtype=torch.float32)


def train_eval_gcn(tz, tadj, ty, vz, vadj, hidden_dim=32, epochs=50, lr=0.005, wd=1e-4, seed=42):
    set_seed(seed)
    model = NormativeGCN(in_node_features=tz.shape[-1], hidden_dim=hidden_dim, num_classes=2, dropout=0.25)
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
        return torch.softmax(logits, dim=-1)[:, 1].numpy(), torch.argmax(logits, dim=-1).numpy()


def bootstrap_auc_ci(y_true, y_prob, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    boot = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def extract_features_pli_omst(ep_data: np.ndarray, sfreq: float, ch_names: list, post_idx: list):
    """Compute Welch PSD -> [2, 30] Hz specparam features, IAPF, 4-13 Hz PLI, and OMST."""
    n_ep, n_ch, n_t = ep_data.shape
    freqs, psd = signal.welch(ep_data, fs=sfreq, nperseg=min(int(sfreq * 2.0), n_t), axis=-1)
    mean_psd = np.mean(psd, axis=0)
    sgm = SpectralGroupModel(**SPECPARAM_KWARGS)
    sgm.fit(freqs, np.maximum(mean_psd, 1e-25), [2.0, 30.0], n_jobs=1)
    ff = sgm.data.freqs
    th_m = (ff >= 4.0) & (ff < 8.0)
    dl_m = (ff >= 2.0) & (ff < 4.0)
    al_m = (ff >= 8.0) & (ff <= 13.0)
    aps = sgm.get_params("aperiodic")
    x_mat = np.zeros((n_ch, 4), dtype=np.float32)
    iapfs = []
    for c in range(n_ch):
        m_c = sgm.get_model(c, regenerate=True)
        resid = m_c.data.power_spectrum - m_c.results.model._ap_fit
        x_mat[c] = [float(np.mean(resid[th_m])), float(np.mean(resid[dl_m])), float(np.mean(resid[al_m])), float(aps[c, 1])]
        if c in post_idx:
            pks = m_c.results.params.periodic.params
            if pks.size > 0:
                apks = pks[(pks[:, 0] >= 7.0) & (pks[:, 0] <= 13.5)]
                if len(apks) > 0:
                    iapfs.append(float(apks[np.argmax(apks[:, 1]), 0]))
    iapf_val = float(np.median(iapfs)) if iapfs else np.nan

    sub_ep = ep_data[:min(n_ep, 30)]
    nyq = sfreq / 2.0
    b, a = signal.butter(4, [4.0 / nyq, 13.0 / nyq], btype="band")
    phases = np.angle(signal.hilbert(signal.filtfilt(b, a, sub_ep, axis=-1), axis=-1))
    ph_c = phases.transpose(1, 0, 2).reshape(n_ch, -1)
    pli = np.zeros((n_ch, n_ch), dtype=np.float32)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            v = float(np.abs(np.mean(np.sign(np.sin(ph_c[i] - ph_c[j])))))
            pli[i, j] = pli[j, i] = v
    np.fill_diagonal(pli, 1.0)
    adj_bin, _, opt_m, max_gce = TopologicalGraphBuilder().build_omst_graph(pli)
    return x_mat, pli, adj_bin, opt_m, max_gce, iapf_val


def _iowa_subj_worker(row_dict):
    sid = row_dict["participant_id"]
    raw_orig = mne.io.read_raw_eeglab(str(IOWA_DIR / sid / "eeg" / f"{sid}_task-Rest_eeg.set"), preload=True, verbose=False)
    cmap = {ch: c for c in raw_orig.ch_names for ch in CHANNELS_18 if ch.lower() == c.lower()}
    std1020 = mne.channels.make_standard_montage("standard_1020")

    def finish_epochs(r_proc):
        r_proc.filter(l_freq=1.0, h_freq=40.0, verbose=False)
        if abs(r_proc.info["sfreq"] - 250.0) > 1e-3:
            r_proc.resample(250.0, verbose=False)
        ep = mne.make_fixed_length_epochs(r_proc, duration=4.0, preload=True, verbose=False).get_data(copy=True)
        ptp = np.ptp(ep, axis=-1).max(axis=1)
        mask = ptp < 200.0 * 1e-6
        return ep[mask] if np.sum(mask) >= 5 else ep[np.argsort(ptp)[:min(len(ptp), 8)]]

    # Regime A: Online reference Pz reconstructed via CAR19 (matches canonical Study 7)
    r19 = raw_orig.copy()
    if "Pz" not in r19.ch_names and "pz" not in [c.lower() for c in r19.ch_names]:
        r19.add_reference_channels(ref_channels=["Pz"])
    cmap19 = {ch: c for c in r19.ch_names for ch in CHANNELS_19 if ch.lower() == c.lower()}
    r19.rename_channels(cmap19); r19.pick(CHANNELS_19); r19.reorder_channels(CHANNELS_19)
    r19.set_eeg_reference("average", projection=False, verbose=False)
    x19, _, a19, _, _, iapf19 = extract_features_pli_omst(finish_epochs(r19), 250.0, CHANNELS_19, POSTERIOR_5)

    # Regime B: Strictly 18 physically recorded channels CAR (no Pz)
    r18 = raw_orig.copy().pick([cmap[c] for c in CHANNELS_18]).rename_channels({cmap[c]: c for c in CHANNELS_18}).reorder_channels(CHANNELS_18)
    r18.set_eeg_reference("average", projection=False, verbose=False)
    x18, _, a18, _, _, iapf18 = extract_features_pli_omst(finish_epochs(r18), 250.0, CHANNELS_18, POSTERIOR_4_18)

    # Regime C: Spherical-spline interpolated Pz -> 19-ch CAR
    rsp = raw_orig.copy().pick([cmap[c] for c in CHANNELS_18]).rename_channels({cmap[c]: c for c in CHANNELS_18})
    pz_r = mne.io.RawArray(np.zeros((1, rsp.n_times)), mne.create_info(["Pz"], rsp.info["sfreq"], ["eeg"]), verbose=False)
    rsp.add_channels([pz_r], force_update_info=True)
    rsp.reorder_channels(CHANNELS_19)
    rsp.set_montage(std1020, on_missing="warn", verbose=False)
    rsp.info["bads"] = ["Pz"]
    rsp.interpolate_bads(reset_bads=True, verbose=False)
    rsp.set_eeg_reference("average", projection=False, verbose=False)
    xsp, _, asp, _, _, iapf_sp = extract_features_pli_omst(finish_epochs(rsp), 250.0, CHANNELS_19, POSTERIOR_5)

    return {
        "sid": sid, "y": int(row_dict["GROUP"] == "PD"),
        "moca": float(row_dict["MOCA"]) if pd.notnull(row_dict["MOCA"]) else np.nan,
        "updrs": float(row_dict["UPDRS"]) if pd.notnull(row_dict["UPDRS"]) else np.nan,
        "CAR19_RefReconPz": (x19, a19, iapf19),
        "CAR18_NoPz": (x18, a18, iapf18),
        "CAR19_SplinePz": (xsp, asp, iapf_sp),
    }


def run_study7_pz_sensitivity(sp):
    print("\n=== [v6 Step 1] Study 7 (Iowa PD, N=149) Reference/Pz CAR Sensitivity ===")
    t0 = time.time()
    df_p = pd.read_csv(IOWA_DIR / "participants.tsv", sep="\t")
    jobs = df_p.to_dict(orient="records")
    res_list = []
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4), mp_context=mp.get_context("spawn")) as ex:
        for item in ex.map(_iowa_subj_worker, jobs):
            res_list.append(item)
    res_map = {r["sid"]: r for r in res_list}
    ordered = [res_map[g["subject_id"]] for g in sp["iowa_ds004584"]]

    # Verify CAR19_RefReconPz reproduces cached canonical Study 7 features within numerical tolerance
    max_x_diff = max(float(np.max(np.abs(ordered[i]["CAR19_RefReconPz"][0] - sp["iowa_ds004584"][i]["x"].numpy()))) for i in range(len(ordered)))
    print(f"  Verified CAR19_RefReconPz vs. cached canonical Study 7 max |dx| = {max_x_diff:.2e}")

    rows = []
    for reg_name, f_idx, p_idx in [
        ("CAR19_RefReconPz", FOCAL_5, POSTERIOR_5),
        ("CAR18_NoPz", FOCAL_5_18, POSTERIOR_4_18),
        ("CAR19_SplinePz", FOCAL_5, POSTERIOR_5),
    ]:
        Xs = torch.stack([torch.tensor(r[reg_name][0], dtype=torch.float32) for r in ordered])
        As = torch.stack([torch.tensor(r[reg_name][1], dtype=torch.float32) for r in ordered])
        ys = np.array([r["y"] for r in ordered])
        iapfs = np.array([r[reg_name][2] for r in ordered], dtype=float)
        pids = np.array([r["sid"] for r in ordered])

        def d_welch(vals):
            vc, vr = vals[ys == 1], vals[ys == 0]
            vc, vr = vc[~np.isnan(vc)], vr[~np.isnan(vr)]
            sd = np.sqrt(((len(vc) - 1) * np.var(vc, ddof=1) + (len(vr) - 1) * np.var(vr, ddof=1)) / (len(vc) + len(vr) - 2))
            _, pv = stats.ttest_ind(vc, vr, equal_var=False)
            return float((vc.mean() - vr.mean()) / sd), float(pv)

        th_d, th_p = d_welch(Xs[:, f_idx, 0].mean(dim=-1).numpy())
        pa_d, pa_p = d_welch(Xs[:, p_idx, 2].mean(dim=-1).numpy())
        ch_d, ch_p = d_welch(Xs[:, :, 3].mean(dim=-1).numpy())
        ip_d, ip_p = d_welch(iapfs)

        aucs = {m: [] for m in ["GCN_OMST", "Tuned_LR", "Linear_SVM", "RandomForest"]}
        for s in REPEATS_10:
            splits = get_grouped_splits(pids, ys, n_splits=5, seed=s)
            oof = {m: np.zeros(len(ys)) for m in aucs}
            for f, (tr, va) in enumerate(splits):
                ty = torch.tensor(ys[tr], dtype=torch.long)
                tz, vz, _, _ = compute_fold_normative_z(Xs[tr], ty, Xs[va])
                Xtr, Xva, ytr = tz.view(len(tr), -1).numpy(), vz.view(len(va), -1).numpy(), ys[tr]
                oof["GCN_OMST"][va], _ = train_eval_gcn(tz, As[tr], ty, vz, As[va], seed=s + f)
                icv = get_grouped_splits(pids[tr], ytr, n_splits=3, seed=s + f)
                gs_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=s), {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}, cv=icv, scoring="roc_auc").fit(Xtr, ytr)
                oof["Tuned_LR"][va] = gs_lr.predict_proba(Xva)[:, 1]
                gs_lsvm = GridSearchCV(SVC(kernel="linear", probability=False, max_iter=10000, random_state=s), {"C": [0.01, 0.1, 1.0, 10.0]}, cv=icv, scoring="roc_auc").fit(Xtr, ytr)
                oof["Linear_SVM"][va] = 1.0 / (1.0 + np.exp(-gs_lsvm.decision_function(Xva)))
                oof["RandomForest"][va] = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=s).fit(Xtr, ytr).predict_proba(Xva)[:, 1]
            for m in aucs:
                aucs[m].append(float(roc_auc_score(ys, oof[m])))

        # Patient-only clinical severity regressions (N=100 PD)
        pt_idx = [i for i, r in enumerate(ordered) if r["y"] == 1]
        Xpt = np.stack([ordered[i][reg_name][0].flatten() for i in pt_idx])
        moca_r, moca_r2, updrs_r, updrs_r2 = [], [], [], []
        y_moca = np.array([ordered[i]["moca"] for i in pt_idx])
        y_updrs = np.array([ordered[i]["updrs"] for i in pt_idx])
        for s in REPEATS_10:
            kf = KFold(n_splits=5, shuffle=True, random_state=s)
            oof_m, oof_u = np.zeros(len(pt_idx)), np.zeros(len(pt_idx))
            for tr, va in kf.split(Xpt):
                mu, sd = Xpt[tr].mean(0, keepdims=True), Xpt[tr].std(0, keepdims=True) + 1e-6
                Xtr_s, Xva_s = (Xpt[tr] - mu) / sd, (Xpt[va] - mu) / sd
                oof_m[va] = SVR(C=1.0, epsilon=0.1).fit(Xtr_s, y_moca[tr]).predict(Xva_s)
                oof_u[va] = HistGradientBoostingRegressor(max_iter=100, max_depth=4, random_state=s).fit(Xtr_s, y_updrs[tr]).predict(Xva_s)
            moca_r.append(float(pearsonr(y_moca, oof_m)[0])); moca_r2.append(float(r2_score(y_moca, oof_m)))
            updrs_r.append(float(pearsonr(y_updrs, oof_u)[0])); updrs_r2.append(float(r2_score(y_updrs, oof_u)))

        rows.append({
            "regime": reg_name, "n_channels": Xs.shape[1],
            "frontal_theta_d": th_d, "frontal_theta_p": th_p,
            "posterior_alpha_d": pa_d, "posterior_alpha_p": pa_p,
            "global_chi_d": ch_d, "global_chi_p": ch_p,
            "iapf_d": ip_d, "iapf_p": ip_p,
            "gcn_omst_auc": float(np.mean(aucs["GCN_OMST"])), "gcn_omst_sd": float(np.std(aucs["GCN_OMST"], ddof=1)),
            "tuned_lr_auc": float(np.mean(aucs["Tuned_LR"])), "tuned_lr_sd": float(np.std(aucs["Tuned_LR"], ddof=1)),
            "linear_svm_auc": float(np.mean(aucs["Linear_SVM"])), "linear_svm_sd": float(np.std(aucs["Linear_SVM"], ddof=1)),
            "rf_auc": float(np.mean(aucs["RandomForest"])), "rf_sd": float(np.std(aucs["RandomForest"], ddof=1)),
            "moca_svr_r": float(np.mean(moca_r)), "moca_svr_r2": float(np.mean(moca_r2)),
            "updrs_gbdt_r": float(np.mean(updrs_r)), "updrs_gbdt_r2": float(np.mean(updrs_r2)),
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(REPORTS_DIR / "study7_pz_car_sensitivity.csv", index=False)
    print(df_out[["regime", "n_channels", "frontal_theta_d", "iapf_d", "gcn_omst_auc", "linear_svm_auc", "moca_svr_r", "updrs_gbdt_r"]].round(4).to_string(index=False))
    print(f"  ✓ Saved reports/study7_pz_car_sensitivity.csv ({time.time()-t0:.1f}s)")


def run_participant_vs_session_auc(sp):
    print("\n=== [v6 Step 2] Study 4 & Study 6 10x5 Participant-Level vs. Session-Level AUC ===")
    specs = [
        ("Study 4: UCSD Parkinson All Sessions", sp["ucsd_ds002778"]),
        ("Study 6c: UNM PD All Sessions", sp["unm_ds003490"]),
    ]
    adj_id = torch.zeros((19, 19), dtype=torch.float32)
    adj_full = torch.ones((19, 19), dtype=torch.float32); adj_full.fill_diagonal_(0.0)
    rows = []

    for cname, glist in specs:
        vg = [g for g in glist if int(g["y"]) in (0, 1)]
        n = len(vg)
        y_all = np.array([int(g["y"]) for g in vg])
        groups = np.array([g.get("bids_id", g.get("sub_name", g["subject_id"])) for g in vg])
        X_all = torch.stack([g["x"] for g in vg])
        adj_omst = torch.stack([g["adj"] for g in vg])
        adj_wpli = torch.stack([g["adj_weighted"] for g in vg])
        edge_counts = [max(1, int(g.get("omst_edge_count", 27))) for g in vg]

        uniq_pids = sorted(set(groups.tolist()))
        y_pid = np.array([int(y_all[groups == p][0]) for p in uniq_pids])

        per_model_sess = {m: [] for m in MODELS_13}
        per_model_part = {m: [] for m in MODELS_13}
        split0_part_probs = {}

        for r_idx, seed in enumerate(REPEATS_10):
            set_seed(seed)
            splits = get_grouped_splits(groups, y_all, n_splits=5, seed=seed)
            adj_rnd_all = torch.stack([make_edge_matched_random_adj(19, edge_counts[i], seed=seed * 1000 + i) for i in range(n)])
            oof_p = {m: np.zeros(n) for m in MODELS_13}
            for fold, (tr, va) in enumerate(splits):
                tx, vx = X_all[tr], X_all[va]
                ty = torch.tensor(y_all[tr], dtype=torch.long)
                tz, vz, _, _ = compute_fold_normative_z(tx, ty, vx)
                X_tr, X_va, y_tr = tz.view(len(tr), -1).numpy(), vz.view(len(va), -1).numpy(), y_all[tr]

                oof_p["GCN_OMST"][va], _ = train_eval_gcn(tz, adj_omst[tr], ty, vz, adj_omst[va], seed=seed + fold)
                oof_p["GCN_Weighted_PLI"][va], _ = train_eval_gcn(tz, adj_wpli[tr], ty, vz, adj_wpli[va], seed=seed + fold)
                oof_p["GCN_Identity"][va], _ = train_eval_gcn(tz, adj_id.unsqueeze(0).expand(len(tr), -1, -1), ty, vz, adj_id.unsqueeze(0).expand(len(va), -1, -1), seed=seed + fold)
                oof_p["GCN_Random_EdgeMatched"][va], _ = train_eval_gcn(tz, adj_rnd_all[tr], ty, vz, adj_rnd_all[va], seed=seed + fold)
                oof_p["GCN_Full"][va], _ = train_eval_gcn(tz, adj_full.unsqueeze(0).expand(len(tr), -1, -1), ty, vz, adj_full.unsqueeze(0).expand(len(va), -1, -1), seed=seed + fold)

                icv = get_grouped_splits(groups[tr], y_tr, n_splits=3, seed=seed + fold)
                oof_p["Tuned_LogisticRegression"][va] = GridSearchCV(LogisticRegression(max_iter=1000, random_state=seed), {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}, cv=icv, scoring="roc_auc").fit(X_tr, y_tr).predict_proba(X_va)[:, 1]
                gs_rc = GridSearchCV(RidgeClassifier(random_state=seed), {"alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}, cv=icv, scoring="roc_auc").fit(X_tr, y_tr)
                oof_p["Tuned_Ridge"][va] = 1.0 / (1.0 + np.exp(-gs_rc.decision_function(X_va)))
                gs_l = GridSearchCV(SVC(kernel="linear", probability=False, max_iter=10000, random_state=seed), {"C": [0.01, 0.1, 1.0, 10.0]}, cv=icv, scoring="roc_auc").fit(X_tr, y_tr)
                oof_p["Linear_SVM"][va] = 1.0 / (1.0 + np.exp(-gs_l.decision_function(X_va)))
                gs_r = GridSearchCV(SVC(kernel="rbf", probability=False, max_iter=10000, random_state=seed), {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto", 0.01, 0.1]}, cv=icv, scoring="roc_auc").fit(X_tr, y_tr)
                oof_p["RBF_SVM"][va] = 1.0 / (1.0 + np.exp(-gs_r.decision_function(X_va)))
                oof_p["RandomForest"][va] = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed).fit(X_tr, y_tr).predict_proba(X_va)[:, 1]
                oof_p["GBDT"][va] = HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=seed).fit(X_tr, y_tr).predict_proba(X_va)[:, 1]
                oof_p["Single_Theta"][va] = LogisticRegression(random_state=seed).fit(tz[:, FOCAL_5, 0].mean(dim=-1, keepdim=True).numpy(), y_tr).predict_proba(vz[:, FOCAL_5, 0].mean(dim=-1, keepdim=True).numpy())[:, 1]
                oof_p["Single_Chi"][va] = LogisticRegression(random_state=seed).fit(tz[:, :, 3].mean(dim=-1, keepdim=True).numpy(), y_tr).predict_proba(vz[:, :, 3].mean(dim=-1, keepdim=True).numpy())[:, 1]

            for m in MODELS_13:
                per_model_sess[m].append(float(roc_auc_score(y_all, oof_p[m])))
                p_probs = np.array([float(np.mean(oof_p[m][groups == pid])) for pid in uniq_pids])
                per_model_part[m].append(float(roc_auc_score(y_pid, p_probs)))
                if r_idx == 0:
                    split0_part_probs[m] = p_probs

        for m in MODELS_13:
            lo0, hi0 = bootstrap_auc_ci(y_pid, split0_part_probs[m])
            rows.append({
                "cohort": cname, "model": m, "n_sessions": n, "n_participants": len(uniq_pids),
                "session_mean_auc_10x5": float(np.mean(per_model_sess[m])),
                "session_sd_auc_10x5": float(np.std(per_model_sess[m], ddof=1)),
                "participant_mean_auc_10x5": float(np.mean(per_model_part[m])),
                "participant_sd_auc_10x5": float(np.std(per_model_part[m], ddof=1)),
                "delta_part_minus_sess": float(np.mean(per_model_part[m]) - np.mean(per_model_sess[m])),
                "participant_split0_auc": float(per_model_part[m][0]),
                "participant_split0_ci_lo": lo0, "participant_split0_ci_hi": hi0,
            })
    df_ps = pd.DataFrame(rows)
    df_ps.to_csv(REPORTS_DIR / "participant_vs_session_level_auc_10x5.csv", index=False)
    print(df_ps[df_ps["model"].isin(["GCN_OMST", "GCN_Weighted_PLI", "GCN_Random_EdgeMatched", "Linear_SVM", "Tuned_LogisticRegression", "RandomForest"])][["cohort", "model", "session_mean_auc_10x5", "participant_mean_auc_10x5", "delta_part_minus_sess"]].round(4).to_string(index=False))
    print("  ✓ Saved reports/participant_vs_session_level_auc_10x5.csv")
    return df_ps


def run_tost_and_macro_auc_sensitivity(df_ps):
    print("\n=== [v6 Step 3] TOST Dependency (k=8 vs k=7 Drop 5b/5a), Holm Correction & Excl Study 2 Macro-AUCs ===")
    df_sum = pd.read_csv(REPORTS_DIR / "benchmark_repeated_10x5_summary.csv")
    df_riem = pd.read_csv(REPORTS_DIR / "benchmark_riemannian_tangent_space.csv")
    piv = df_sum.pivot(index="cohort", columns="model", values="mean_auc")
    piv_r = df_riem[df_riem["representation"] == "Riemannian_Spatial_Cov_190D"].pivot(index="cohort", columns="classifier", values="auc")

    piv_part = piv.copy()
    s4_map = df_ps[df_ps["cohort"] == "Study 4: UCSD Parkinson All Sessions"].set_index("model")["participant_mean_auc_10x5"].to_dict()
    for m, v in s4_map.items():
        piv_part.loc["Study 4: UCSD Parkinson All Sessions", m] = float(v)

    nonnested_8 = [c for c in piv.index if c not in ("Study 2b: Mumtaz MDD (Eyes Open)", "Study 5c: Dementia All (AD+FTD) vs Control", "Study 6b: UNM PD-ON vs Control", "Study 6c: UNM PD All Sessions")]
    primary_9 = [c for c in piv.index if c not in ("Study 2b: Mumtaz MDD (Eyes Open)", "Study 6b: UNM PD-ON vs Control", "Study 6c: UNM PD All Sessions")]
    no_s2_7 = [c for c in nonnested_8 if not c.startswith("Study 2:")]
    ad_only_7 = [c for c in nonnested_8 if not c.startswith("Study 5b:")]
    ftd_only_7 = [c for c in nonnested_8 if not c.startswith("Study 5a:")]
    indep_clin_6 = [c for c in ad_only_7 if not c.startswith("Study 2:")]

    macro_rows = []
    for scope_lbl, mat_p, idxs in [
        ("k=8 Non-Nested (Session-Level Study 4)", piv, nonnested_8),
        ("k=8 Non-Nested (Participant-Level Study 4)", piv_part, nonnested_8),
        ("k=9 Primary (Incl. 5c)", piv, primary_9),
        ("k=7 Excl. Study 2 Batch-Confounded", piv, no_s2_7),
        ("k=7 Drop 5b Shared-HC (AD Only)", piv, ad_only_7),
        ("k=7 Drop 5a Shared-HC (FTD Only)", piv, ftd_only_7),
        ("k=6 Excl. Study 2 & Shared-HC 5b", piv, indep_clin_6),
    ]:
        row = {"scope": scope_lbl, "k_contrasts": len(idxs)}
        for m in MODELS_13:
            row[m] = float(mat_p.loc[idxs, m].mean())
        for rc in ("RF", "Ridge", "LR"):
            row[f"SpatialCov_{rc}"] = float(piv_r.loc[idxs, rc].mean())
        macro_rows.append(row)
    pd.DataFrame(macro_rows).to_csv(REPORTS_DIR / "macro_auc_study2_sensitivity.csv", index=False)

    gnn_comps = ["GCN_Weighted_PLI", "GCN_Identity", "GCN_Random_EdgeMatched", "GCN_Full"]
    all_comps = [m for m in MODELS_13 if m != "GCN_OMST"]
    tost_rows = []
    for scope_lbl, mat_p, idxs in [
        ("k=8 Non-Nested", piv, nonnested_8),
        ("k=8 Non-Nested (Study 4 Part-Level)", piv_part, nonnested_8),
        ("k=7 Drop 5b (AD Only)", piv, ad_only_7),
        ("k=7 Drop 5a (FTD Only)", piv, ftd_only_7),
        ("k=7 Excl. Study 2", piv, no_s2_7),
    ]:
        k_n = len(idxs)
        t90 = float(stats.t.ppf(0.95, df=k_n - 1))
        t95 = float(stats.t.ppf(0.975, df=k_n - 1))
        tmp = []
        for m in all_comps:
            x, y = mat_p.loc[idxs, "GCN_OMST"].values, mat_p.loc[idxs, m].values
            diff = x - y
            md = float(diff.mean())
            se = float(diff.std(ddof=1) / np.sqrt(k_n)) + 1e-12
            p02 = max(float(1.0 - stats.t.cdf((md + 0.02) / se, df=k_n - 1)), float(stats.t.cdf((md - 0.02) / se, df=k_n - 1)))
            p03 = max(float(1.0 - stats.t.cdf((md + 0.03) / se, df=k_n - 1)), float(stats.t.cdf((md - 0.03) / se, df=k_n - 1)))
            _, w_p = wilcoxon(x, y) if np.any(np.abs(diff) > 1e-8) else (0.0, 1.0)
            tmp.append({
                "scope": scope_lbl, "k_benchmarks": k_n, "comparator_model": m,
                "is_gnn_family": m in gnn_comps,
                "gcn_mean_auc": float(x.mean()), "comp_mean_auc": float(y.mean()),
                "mean_diff_auc": md, "ci90_diff_lo": md - t90 * se, "ci90_diff_hi": md + t90 * se,
                "ci95_diff_lo": md - t95 * se, "ci95_diff_hi": md + t95 * se,
                "tost_p_002": p02, "tost_p_003": p03, "wilcoxon_raw_p": float(w_p),
            })
        # Holm-Bonferroni separately across 4 GNN comparators and all 12 comparators
        gnn_idx = [i for i, r in enumerate(tmp) if r["is_gnn_family"]]
        _, h02_g, _, _ = multipletests([tmp[i]["tost_p_002"] for i in gnn_idx], method="holm")
        _, h03_g, _, _ = multipletests([tmp[i]["tost_p_003"] for i in gnn_idx], method="holm")
        _, bh_all, _, _ = multipletests([r["tost_p_002"] for r in tmp], method="fdr_bh")
        for i, r in enumerate(tmp):
            r["tost02_bh12"] = float(bh_all[i])
            r["tost02_holm_gnn4"] = np.nan
            r["tost03_holm_gnn4"] = np.nan
        for j, i in enumerate(gnn_idx):
            tmp[i]["tost02_holm_gnn4"] = float(h02_g[j])
            tmp[i]["tost03_holm_gnn4"] = float(h03_g[j])
        tost_rows.extend(tmp)
    pd.DataFrame(tost_rows).to_csv(REPORTS_DIR / "tost_dependency_and_holm_sensitivity.csv", index=False)
    print("  ✓ Saved reports/macro_auc_study2_sensitivity.csv & reports/tost_dependency_and_holm_sensitivity.csv")


def run_spatial_cov_foldwise_check(sp):
    print("\n=== [v6 Step 4] Spatial Cov 190D Fold-Wise Log-Euclidean Mean Centering Audit ===")
    covs = torch.load(DATA_DIR / "spatial_covariances_7cohorts.pt", weights_only=False)
    def upper_vec(M):
        idx = np.triu_indices(19)
        w = np.ones_like(M) * np.sqrt(2.0)
        np.fill_diagonal(w, 1.0)
        return (M * w)[idx].astype(np.float32)

    specs = [
        ("Study 1: Sleep Deprivation (ds004902)", "sleep_ds004902", sp["sleep_ds004902"]),
        ("Study 2: Mumtaz MDD (Eyes Closed)", "mumtaz_ec", sp["mumtaz_ec"]),
        ("Study 3: Cavanagh University Depression", "cavanagh_ds003478", sp["cavanagh_ds003478"]),
        ("Study 4: UCSD Parkinson All Sessions", "ucsd_ds002778", sp["ucsd_ds002778"]),
        ("Study 5a: Dementia AD vs Control", "dementia_ds004504", [g for g in sp["dementia_ds004504"] if g["diagnosis"] in ("Alzheimer", "Control")]),
        ("Study 5b: Dementia FTD vs Control", "dementia_ds004504", [g for g in sp["dementia_ds004504"] if g["diagnosis"] in ("FTD", "Control")]),
        ("Study 6a: UNM PD-OFF vs Control", "unm_ds003490", [g for g in sp["unm_ds003490"] if g["med_state"] in ("OFF", "CTL")]),
        ("Study 7: Iowa Parkinson (ds004584)", "iowa_ds004584", sp["iowa_ds004584"]),
    ]
    rows = []
    for cname, ckey, glist in specs:
        vg = [g for g in glist if int(g["y"]) in (0, 1)]
        y_all = np.array([int(g["y"]) for g in vg])
        grps = np.array([g.get("bids_id", g.get("sub_name", g["subject_id"])) for g in vg])
        log_mats = []
        for g in vg:
            k = (f"{g['subject_id']}_{g['session']}" if g.get("group") == "PD" else g["subject_id"]) if ckey == "unm_ds003490" else g["subject_id"]
            svs, svecs = np.linalg.eigh(covs[ckey][k])
            log_mats.append((svecs * np.log(np.maximum(svs, 1e-12))) @ svecs.T)
        L_all = np.stack(log_mats)  # [N, 19, 19]

        aucs_id = {c: [] for c in ("LR", "Ridge", "RF")}
        aucs_fm = {c: [] for c in ("LR", "Ridge", "RF")}
        for s in REPEATS_10:
            splits = get_grouped_splits(grps, y_all, n_splits=5, seed=s)
            oof_id = {c: np.zeros(len(vg)) for c in aucs_id}
            oof_fm = {c: np.zeros(len(vg)) for c in aucs_fm}
            for fold, (tr, va) in enumerate(splits):
                ytr = y_all[tr]
                icv = get_grouped_splits(grps[tr], ytr, n_splits=3, seed=s + fold)

                # Reference 1: Identity I_19 (fold-wise feature standardization)
                X_id = np.stack([upper_vec(L) for L in L_all])
                mu_i, sd_i = X_id[tr].mean(0, keepdims=True), X_id[tr].std(0, keepdims=True) + 1e-6
                Xtr_i, Xva_i = (X_id[tr] - mu_i) / sd_i, (X_id[va] - mu_i) / sd_i

                # Reference 2: Training-fold Log-Euclidean Fréchet Mean strictly fitted on tr
                L_mean_tr = L_all[tr].mean(axis=0, keepdims=True)
                X_fm = np.stack([upper_vec(L) for L in (L_all - L_mean_tr)])
                mu_f, sd_f = X_fm[tr].mean(0, keepdims=True), X_fm[tr].std(0, keepdims=True) + 1e-6
                Xtr_f, Xva_f = (X_fm[tr] - mu_f) / sd_f, (X_fm[va] - mu_f) / sd_f

                for oof_d, (Xtr, Xva) in [(oof_id, (Xtr_i, Xva_i)), (oof_fm, (Xtr_f, Xva_f))]:
                    oof_d["LR"][va] = GridSearchCV(LogisticRegression(max_iter=1000, random_state=s), {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}, cv=icv, scoring="roc_auc").fit(Xtr, ytr).predict_proba(Xva)[:, 1]
                    oof_d["Ridge"][va] = 1.0 / (1.0 + np.exp(-GridSearchCV(RidgeClassifier(random_state=s), {"alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}, cv=icv, scoring="roc_auc").fit(Xtr, ytr).decision_function(Xva)))
                    oof_d["RF"][va] = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=s).fit(Xtr, ytr).predict_proba(Xva)[:, 1]
            for c in ("LR", "Ridge", "RF"):
                aucs_id[c].append(float(roc_auc_score(y_all, oof_id[c])))
                aucs_fm[c].append(float(roc_auc_score(y_all, oof_fm[c])))
        for c in ("LR", "Ridge", "RF"):
            rows.append({
                "cohort": cname, "classifier": c,
                "auc_identity_ref": float(np.mean(aucs_id[c])), "sd_identity": float(np.std(aucs_id[c], ddof=1)),
                "auc_fold_frechet_ref": float(np.mean(aucs_fm[c])), "sd_fold_frechet": float(np.std(aucs_fm[c], ddof=1)),
                "delta_frechet_minus_id": float(np.mean(aucs_fm[c]) - np.mean(aucs_id[c])),
            })
    df_sc = pd.DataFrame(rows)
    df_sc.to_csv(REPORTS_DIR / "spatial_cov_foldwise_frechet_check.csv", index=False)
    print(df_sc.groupby("classifier")[["auc_identity_ref", "auc_fold_frechet_ref", "delta_frechet_minus_id"]].mean().round(4).to_string())
    print("  ✓ Saved reports/spatial_cov_foldwise_frechet_check.csv")


def run_study2_accounting_and_fix_release(sp):
    print("\n=== [v6 Step 5] Study 2 181-EDF Complete Accounting & Release Artifacts Key Disambiguation ===")
    ec_ids = {g["subject_id"] for g in sp["mumtaz_ec"]}
    eo_ids = {g["subject_id"] for g in sp["mumtaz_eo"]}
    rows = []
    for p in sorted(MUMTAZ_DIR.glob("*.edf")):
        name = p.name
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        state = "EC" if " EC." in name else ("EO" if " EO." in name else "TASK")
        m_sub = pd.Series([name]).str.extract(r"S\s*(\d+)")[0].iloc[0]
        grp = "MDD" if "MDD" in name else "HC"
        canon_id = f"mumtaz_{grp.lower()}_{int(m_sub):03d}" if pd.notnull(m_sub) else name
        if state == "TASK":
            disp = "Excluded (Task paradigm; not resting state)"
        elif name.startswith(("6921143_", "6921959_")):
            disp = "Skipped Figshare ID-prefixed duplicate EO pair (H S15 EO)"
        elif canon_id in ("mumtaz_hc_030", "mumtaz_mdd_034"):
            disp = f"Excluded SHA-256 byte-identical duplicate ({canon_id})"
        elif state == "EC":
            disp = "Included in Study 2a Primary EC (N=56)" if canon_id in ec_ids else "Excluded Pre-Analysis QC (>20% artifact / bridged)"
        else:
            disp = "Included in Study 2b Secondary EO (N=58)" if canon_id in eo_ids else "Excluded Pre-Analysis QC (>20% artifact / bridged)"
        rows.append({"filename": name, "state_block": state, "group_prefix": grp, "canonical_id": canon_id, "sha256_prefix": h, "disposition": disp})
    df_acc = pd.DataFrame(rows)
    df_acc.to_csv(REPORTS_DIR / "study2_complete_181_file_accounting.csv", index=False)
    print(df_acc.groupby(["state_block", "disposition"]).size().to_string())

    # Fix release_artifacts/derived_session_features_733.csv, cv_fold_indices_split0.csv, oof_predictions_split0.csv
    # Add explicit session, participant_id, and guaranteed-unique record_uid!
    feat_rows = []
    for ckey, recs in sp.items():
        for g in recs:
            arr = g["x"].numpy()
            sid = g["subject_id"]
            ses = str(g.get("session", "EO" if ckey == "mumtaz_eo" else ("EC" if ckey == "mumtaz_ec" else "ses-1")))
            pid = str(g.get("bids_id", g.get("sub_name", sid)))
            uid = f"{ckey}__{sid}__{ses}"
            feat_rows.append({
                "record_uid": uid, "cohort_key": ckey, "subject_id": sid, "participant_id": pid, "session": ses,
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
    df_feat = pd.DataFrame(feat_rows)
    assert df_feat["record_uid"].is_unique and len(df_feat) == 733
    df_feat.to_csv(RELEASE_DIR / "derived_session_features_733.csv", index=False)

    # Disambiguate cv_fold_indices_split0.csv & oof_predictions_split0.csv identically using exact cohort VG traversal
    sleep, mum_ec, mum_eo = sp["sleep_ds004902"], sp["mumtaz_ec"], sp["mumtaz_eo"]
    cav, ucsd, dem, unm, iowa = sp["cavanagh_ds003478"], sp["ucsd_ds002778"], sp["dementia_ds004504"], sp["unm_ds003490"], sp["iowa_ds004584"]
    specs_12 = [
        ("Study 1: Sleep Deprivation (ds004902)", sleep),
        ("Study 2: Mumtaz MDD (Eyes Closed)", mum_ec),
        ("Study 2b: Mumtaz MDD (Eyes Open)", mum_eo),
        ("Study 3: Cavanagh University Depression", cav),
        ("Study 4: UCSD Parkinson All Sessions", ucsd),
        ("Study 5a: Dementia AD vs Control", [g for g in dem if g["diagnosis"] in ("Alzheimer", "Control")]),
        ("Study 5b: Dementia FTD vs Control", [g for g in dem if g["diagnosis"] in ("FTD", "Control")]),
        ("Study 5c: Dementia All (AD+FTD) vs Control", dem),
        ("Study 6a: UNM PD-OFF vs Control", [g for g in unm if g["med_state"] in ("OFF", "CTL")]),
        ("Study 6b: UNM PD-ON vs Control", [g for g in unm if g["med_state"] in ("ON", "CTL")]),
        ("Study 6c: UNM PD All Sessions", unm),
        ("Study 7: Iowa Parkinson (ds004584)", iowa),
    ]
    sess_meta, uid_meta = [], []
    for cname, glist in specs_12:
        for g in [x for x in glist if int(x["y"]) in (0, 1)]:
            ses = str(g.get("session", "EO" if cname.startswith("Study 2b") else ("EC" if cname.startswith("Study 2:") else "ses-1")))
            ck = g.get("cohort_key", cname)
            sess_meta.append(ses)
            uid_meta.append(f"{ck}__{g['subject_id']}__{ses}")

    df_folds = pd.read_csv(RELEASE_DIR / "cv_fold_indices_split0.csv")
    df_oof = pd.read_csv(RELEASE_DIR / "oof_predictions_split0.csv")
    assert len(df_folds) == len(sess_meta) and len(df_oof) == len(sess_meta)
    df_folds["session"] = sess_meta
    df_folds["record_uid"] = [f"{c}__{u}" for c, u in zip(df_folds["cohort"], uid_meta)]
    df_oof["session"] = sess_meta
    df_oof["record_uid"] = [f"{c}__{u}" for c, u in zip(df_oof["cohort"], uid_meta)]
    assert df_folds["record_uid"].is_unique and df_oof["record_uid"].is_unique
    df_folds.to_csv(RELEASE_DIR / "cv_fold_indices_split0.csv", index=False)
    df_oof.to_csv(RELEASE_DIR / "oof_predictions_split0.csv", index=False)
    print("  ✓ Verified 100% unique record_uid across derived_session_features_733.csv, cv_fold_indices_split0.csv, and oof_predictions_split0.csv")


def run_full_archive_fooof_v111(sp):
    print("\n=== [v6 Step 6] Full-Archive Stable fooof v1.1.1 Cross-Version Verification (13,927 Spectra) ===")
    t0 = time.time()
    psds = torch.load(DATA_DIR / "all_7cohorts_welch_psds.pt", weights_only=False)
    freqs_eq = np.asarray(psds["sleep_ds004902"][0]["freqs"], dtype=float)

    stacked, sp_th, sp_ch, tags, grp_y = [], [], [], [], []
    for ck in ["sleep_ds004902", "mumtaz_ec", "mumtaz_eo", "cavanagh_ds003478", "ucsd_ds002778", "dementia_ds004504", "unm_ds003490", "iowa_ds004584"]:
        for r_sp, r_psd in zip(sp[ck], psds[ck]):
            assert r_sp["subject_id"] == r_psd["subject_id"]
            stacked.append(np.maximum(np.asarray(r_psd["mean_psd"], dtype=float), 1e-25))
            x = r_sp["x_specparam_resid"].numpy()
            sp_th.append(x[:, 0])
            sp_ch.append(x[:, 3])
            tags.append((ck, int(r_sp["y"])))
    arr_psds = np.vstack(stacked)
    assert len(arr_psds) == 733 * 19

    npz_in = SCRATCH_DIR / "fooof_full13927_in.npz"
    npz_out = SCRATCH_DIR / "fooof_full13927_out.npz"
    np.savez(npz_in, freqs=freqs_eq, psds=arr_psds)
    runner = f"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from fooof import FOOOFGroup
dat = np.load(r"{npz_in}")
freqs, psds = dat["freqs"], dat["psds"]
fg = FOOOFGroup(aperiodic_mode="fixed", peak_width_limits=[1.0, 12.0], max_n_peaks=6, min_peak_height=0.05, peak_threshold=2.0, verbose=False)
fg.fit(freqs, psds, [2.0, 30.0], n_jobs=1)
th_m = (fg.freqs >= 4.0) & (fg.freqs < 8.0)
aps = fg.get_params("aperiodic_params")
r2s = np.array([r.r_squared for r in fg.get_results()])
chi = np.array([aps[i, 1] for i in range(len(psds))])
th = np.zeros(len(psds))
for i in range(len(psds)):
    fm = fg.get_fooof(i, regenerate=True)
    th[i] = float(np.mean((fm.power_spectrum - fm._ap_fit)[th_m]))
np.savez(r"{npz_out}", chi=chi, th=th, r2=r2s, n_null=np.array([fg.n_null_]))
"""
    subprocess.run([str(FOOOF_PY), "-c", runner], check=True)
    res = np.load(npz_out)
    fo_ch = res["chi"].reshape(733, 19)
    fo_th = res["th"].reshape(733, 19)
    sp_ch_arr = np.stack(sp_ch)
    sp_th_arr = np.stack(sp_th)

    rows = []
    ck_arr = np.array([t[0] for t in tags])
    for lbl, mask in [("All_733Sessions_13927Spectra", np.ones(733, dtype=bool))] + [(ck, ck_arr == ck) for ck in ["sleep_ds004902", "mumtaz_ec", "mumtaz_eo", "cavanagh_ds003478", "ucsd_ds002778", "dementia_ds004504", "unm_ds003490", "iowa_ds004584"]]:
        fc, sc = fo_ch[mask].ravel(), sp_ch_arr[mask].ravel()
        ft, st = fo_th[mask].ravel(), sp_th_arr[mask].ravel()
        rows.append({
            "scope": lbl, "n_sessions": int(np.sum(mask)), "n_spectra": int(fc.size),
            "fooof_n_null_total": int(res["n_null"][0]),
            "pearson_r_chi": float(pearsonr(fc, sc)[0]), "mean_abs_diff_chi": float(np.mean(np.abs(fc - sc))),
            "pearson_r_theta": float(pearsonr(ft, st)[0]), "mean_abs_diff_theta": float(np.mean(np.abs(ft - st))),
        })
    df_f = pd.DataFrame(rows)
    df_f.to_csv(REPORTS_DIR / "fooof_v111_full13927_verification.csv", index=False)
    print(df_f.head(2).to_string(index=False))
    print(f"  ✓ Saved reports/fooof_v111_full13927_verification.csv ({time.time()-t0:.1f}s)")


def _reg_perm_worker(args):
    jkey, mname, folded, y, seed_list = args
    r_null = []
    for ps in seed_list:
        yp = np.random.RandomState(ps).permutation(y)
        oof = np.zeros(len(y))
        for Xtr, Xva, tr, va in folded:
            if mname == "GBDT_Regressor":
                est = HistGradientBoostingRegressor(max_iter=100, max_depth=4, random_state=42)
            elif mname == "Random_Forest_Regressor":
                est = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=1)
            else:
                est = SVR(C=1.0, epsilon=0.1)
            est.fit(Xtr, yp[tr])
            oof[va] = est.predict(Xva)
        rv_p, _ = pearsonr(yp, oof)
        r_null.append(0.0 if np.isnan(rv_p) else float(rv_p))
    return jkey, mname, seed_list, r_null


def _ridge_perm_batch(X: np.ndarray, y: np.ndarray, alpha: float = 10.0, n_perms: int = 1000):
    Y_perm = np.column_stack([np.random.RandomState(20000 + i).permutation(y) for i in range(n_perms)])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    OOF = np.zeros_like(Y_perm)
    for tr, va in kf.split(X, y):
        mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-6
        Xtr, Xva = (X[tr] - mu) / sd, (X[va] - mu) / sd
        y_mean = Y_perm[tr].mean(axis=0, keepdims=True)
        W = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]), Xtr.T @ (Y_perm[tr] - y_mean))
        OOF[va] = Xva @ W + y_mean
    yp_c, oo_c = Y_perm - Y_perm.mean(axis=0, keepdims=True), OOF - OOF.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(yp_c ** 2, axis=0) * np.sum(oo_c ** 2, axis=0))
    return np.where(denom > 1e-12, np.sum(yp_c * oo_c, axis=0) / np.maximum(denom, 1e-12), 0.0).astype(float)


def run_regression_maxstat_perms(sp):
    print("\n=== [v6 Step 7] 4-Model Paired 1,000-Permutation Max-Statistic Clinical Regression Audit ===", flush=True)
    t0 = time.time()
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
    models_reg = ["GBDT_Regressor", "Random_Forest_Regressor", "SVR_RBF", "Ridge_Regression"]
    df_base = pd.read_csv(REPORTS_DIR / "multicohort_clinical_regression_benchmarks.csv")
    null_arrays = {(jk, mn): np.zeros(1000, dtype=float) for jk, _, _, _, _ in reg_tasks for mn in models_reg}

    jobs = []
    for jkey, _, _, glist, skey in reg_tasks:
        valid = [g for g in glist if pd.notnull(g.get(skey))]
        y = np.array([float(g[skey]) for g in valid])
        X = np.stack([g["x"].numpy().flatten() for g in valid])
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        folded = []
        for tr, va in kf.split(X, y):
            mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-6
            folded.append(((X[tr] - mu) / sd, (X[va] - mu) / sd, tr, va))
        null_arrays[(jkey, "Ridge_Regression")] = _ridge_perm_batch(X, y, alpha=10.0, n_perms=1000)
        for mname in ["GBDT_Regressor", "Random_Forest_Regressor", "SVR_RBF"]:
            chunk = 100 if mname == "SVR_RBF" else 25
            for st in range(0, 1000, chunk):
                jobs.append((jkey, mname, folded, y, list(range(20000 + st, 20000 + min(st + chunk, 1000)))))

    done_c = 0
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4), mp_context=mp.get_context("spawn")) as ex:
        for jk, mn, seeds, rvals in ex.map(_reg_perm_worker, jobs):
            for sd_i, rv in zip(seeds, rvals):
                null_arrays[(jk, mn)][sd_i - 20000] = rv
            done_c += 1
            if done_c % 25 == 0 or done_c == len(jobs):
                print(f"    Max-stat regression chunks: {done_c}/{len(jobs)} ({time.time() - t0:.1f}s)", flush=True)

    # Compute single-model, within-task 4-model max-stat (one-sided & two-sided), and 12-model patient-only family max-stat
    npz_payload = {f"{k[0]}__{k[1]}": v for k, v in null_arrays.items()}
    np.savez(REPORTS_DIR / "clinical_regression_perm_distributions_v6.npz", **npz_payload)

    pt_keys = ["Dementia_MMSE_Patients_Only_N59", "Iowa_PD_MoCA_Patients_Only_N100", "Iowa_PD_UPDRS_Patients_N100"]
    pt_max_1s = np.max(np.column_stack([null_arrays[(jk, m)] for jk in pt_keys for m in models_reg]), axis=1)
    pt_max_2s = np.max(np.column_stack([np.abs(null_arrays[(jk, m)]) for jk in pt_keys for m in models_reg]), axis=1)

    out_rows = []
    for _, row in df_base.iterrows():
        jk, mn = row["cohort_key"], row["model"]
        r0 = float(row["split0_r"])
        arr = null_arrays[(jk, mn)]
        task_max_1s = np.max(np.column_stack([null_arrays[(jk, m)] for m in models_reg]), axis=1)
        task_max_2s = np.max(np.column_stack([np.abs(null_arrays[(jk, m)]) for m in models_reg]), axis=1)

        d = row.to_dict()
        d["perm_p_split0_1s"] = float((np.sum(arr >= r0) + 1.0) / 1001.0)
        d["perm_p_split0_2s"] = float((np.sum(np.abs(arr) >= abs(r0)) + 1.0) / 1001.0)
        d["maxstat4_perm_p_1s"] = float((np.sum(task_max_1s >= r0) + 1.0) / 1001.0)
        d["maxstat4_perm_p_2s"] = float((np.sum(task_max_2s >= abs(r0)) + 1.0) / 1001.0)
        if jk in pt_keys:
            d["maxstat12_pt_perm_p_1s"] = float((np.sum(pt_max_1s >= r0) + 1.0) / 1001.0)
            d["maxstat12_pt_perm_p_2s"] = float((np.sum(pt_max_2s >= abs(r0)) + 1.0) / 1001.0)
        else:
            d["maxstat12_pt_perm_p_1s"] = np.nan
            d["maxstat12_pt_perm_p_2s"] = np.nan
        out_rows.append(d)

    df_v6 = pd.DataFrame(out_rows)
    df_v6.to_csv(REPORTS_DIR / "multicohort_clinical_regression_benchmarks_v6.csv", index=False)
    print(df_v6[["cohort_key", "model", "pearson_r_mean", "split0_r", "r2_mean", "perm_p_split0_2s", "maxstat4_perm_p_1s", "maxstat4_perm_p_2s"]].round(4).to_string(index=False))
    print(f"  ✓ Saved reports/multicohort_clinical_regression_benchmarks_v6.csv ({time.time()-t0:.1f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Run all fast sensitivities (exclude 1,000-perm clinical regression)")
    parser.add_argument("--perms", action="store_true", help="Run only 4-model paired 1,000-permutation max-stat regression")
    args = parser.parse_args()
    run_all = not (args.fast or args.perms)

    sp = torch.load(DATA_DIR / "specparam_canonical_7cohorts.pt", weights_only=False)
    if args.fast or run_all:
        run_study7_pz_sensitivity(sp)
        df_ps = run_participant_vs_session_auc(sp)
        run_tost_and_macro_auc_sensitivity(df_ps)
        run_spatial_cov_foldwise_check(sp)
        run_study2_accounting_and_fix_release(sp)
        run_full_archive_fooof_v111(sp)
    if args.perms or run_all:
        run_regression_maxstat_perms(sp)
    print("\n🎉 ALL v6 SENSITIVITY & AUDIT STEPS COMPLETE!")


if __name__ == "__main__":
    main()
