#!/usr/bin/env python3
"""
Multi-Seed Cross-Validation, PLI Definiteness Audit, and Meta-Analysis.

This script executes:
1. Multi-seed repeated 5-fold cross-validation across 5 random seeds to evaluate model ranking stability.
2. Empirical audit of Phase Lag Index (PLI) matrix eigenvalue spectra and positive semi-definiteness.
3. Riemannian tangent space manifold projections vs Euclidean connectivity baselines.
4. Random-effects meta-analysis (DerSimonian-Laird & Hartung-Knapp-Sidik-Jonkman) of resting frontal periodic theta polarity.
"""

import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.linalg import eigh
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GridSearchCV

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = BENCHMARK_ROOT / "src"
sys.path.append(str(SRC_DIR))
from harmonizer_and_graph import NormativeGCN, compute_fold_normative_z

DATA_DIR = BENCHMARK_ROOT / "data" / "derivatives"
REPORTS_DIR = BENCHMARK_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    """Set random seeds for Python, NumPy, and PyTorch reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def bootstrap_auc_ci(y_true, y_prob, n_boot: int = 1000, seed: int = 42):
    """Compute non-parametric bootstrap 95% confidence intervals for ROC-AUC."""
    n = len(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan
    base_auc = roc_auc_score(y_true, y_prob)
    rng = np.random.RandomState(seed)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not boot_aucs:
        return float(base_auc), float(base_auc), float(base_auc)
    return float(base_auc), float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))


def make_random_adj(num_nodes: int = 19, num_edges: int = 18, seed: int = 42):
    """Generate symmetric density-matched random graph with self-loops and degree normalization."""
    rng = np.random.RandomState(seed)
    adj = np.eye(num_nodes, dtype=np.float32)
    possible_edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    chosen = rng.choice(len(possible_edges), size=num_edges, replace=False)
    for idx in chosen:
        u, v = possible_edges[idx]
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    d_mat = np.diag(deg_inv_sqrt)
    norm_adj = d_mat @ adj @ d_mat
    return torch.tensor(norm_adj, dtype=torch.float32)


def train_eval_gcn(train_z, train_adj, train_y, val_z, val_adj, epochs=50, lr=0.005, weight_decay=1e-4, seed=42):
    """Train and evaluate NormativeGCN model on fold features and graph adjacency."""
    set_seed(seed)
    model = NormativeGCN(in_node_features=4, hidden_dim=32, num_classes=2, dropout=0.25)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(train_z, train_adj)
        loss = criterion(out, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(val_z, val_adj)
        probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
        preds = torch.argmax(logits, dim=-1).numpy()
    return probs, preds


def load_all_benchmark_cohorts():
    """Load preprocessed graph datasets for all primary and sub-contrast benchmark cohorts."""
    sleep_graphs = torch.load(DATA_DIR / "cohort_sleep_ds004902_graphs.pt")
    mumtaz_ec = torch.load(DATA_DIR / "cohort_mumtaz_mdd_graphs.pt")
    cavanagh = torch.load(DATA_DIR / "cohort_cavanagh_mdd_graphs.pt")
    ucsd_pd = torch.load(DATA_DIR / "cohort_ucsd_parkinson_graphs.pt")
    dementia = torch.load(DATA_DIR / "cohort_dementia_ds004504_graphs.pt")
    unm_pd = torch.load(DATA_DIR / "cohort_unm_pd_ds003490_graphs.pt")
    iowa_pd = torch.load(DATA_DIR / "cohort_iowa_pd_ds004584_graphs.pt")

    ad_ctl = [g for g in dementia if g.get("diagnosis") in ("Alzheimer", "Control")]
    ftd_ctl = [g for g in dementia if g.get("diagnosis") in ("FTD", "Control")]
    unm_off_ctl = [g for g in unm_pd if g.get("med_state") in ("OFF", "CTL")]
    unm_on_ctl = [g for g in unm_pd if g.get("med_state") in ("ON", "CTL")]

    return [
        ("Study 1: Sleep Deprivation", [g for g in sleep_graphs if int(g["y"]) in (0, 1)], True, lambda g: g.get("sub_name", g.get("subject_id"))),
        ("Study 2: Mumtaz MDD (EC)", [g for g in mumtaz_ec if int(g["y"]) in (0, 1)], False, None),
        ("Study 3: Cavanagh Depression", [g for g in cavanagh if int(g["y"]) in (0, 1)], False, None),
        ("Study 4: UCSD Parkinson All", [g for g in ucsd_pd if int(g["y"]) in (0, 1)], True, lambda g: g.get("bids_id", g.get("subject_id"))),
        ("Study 5a: Dementia AD vs CTL", [g for g in ad_ctl if int(g["y"]) in (0, 1)], False, None),
        ("Study 5b: Dementia FTD vs CTL", [g for g in ftd_ctl if int(g["y"]) in (0, 1)], False, None),
        ("Study 5c: Dementia All vs CTL", [g for g in dementia if int(g["y"]) in (0, 1)], False, None),
        ("Study 6a: UNM PD-OFF vs CTL", [g for g in unm_off_ctl if int(g["y"]) in (0, 1)], False, None),
        ("Study 6b: UNM PD-ON vs CTL", [g for g in unm_on_ctl if int(g["y"]) in (0, 1)], False, None),
        ("Study 7: Iowa Parkinson", [g for g in iowa_pd if int(g["y"]) in (0, 1)], False, None),
    ]


# ==============================================================================
# 1. MULTI-SEED REPEATED 5-FOLD CROSS-VALIDATION (TABLE S21)
# ==============================================================================

def run_multiseed_repeated_cv():
    """Evaluate model ranking stability across 5 random seeds using stratified (group) cross-validation."""
    print("\n" + "=" * 75)
    print("🎲 [Part 1] Running Multi-Seed Repeated Cross-Validation (5 Seeds, Table S21)...")
    print("=" * 75)

    cohort_benchmarks = load_all_benchmark_cohorts()
    seeds = [42, 123, 456, 789, 2026]
    models = [
        "GCN_OMST",
        "GCN_Random",
        "GCN_Identity",
        "Tuned_LogisticRegression",
        "Tuned_Ridge",
        "RandomForest",
        "Single_Theta",
    ]

    adj_identity = torch.eye(19, dtype=torch.float32)
    frontal_nodes = [0, 1, 2, 3, 16]  # Fp1, Fp2, F3, F4, Fz
    param_grid_lr = {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}
    param_grid_rc = {"alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}

    records = []

    for cohort_label, g_list, is_grouped, group_fn in cohort_benchmarks:
        n_samples = len(g_list)
        y_all = np.array([int(g["y"]) for g in g_list])
        groups = [group_fn(g) for g in g_list] if is_grouped else [str(i) for i in range(n_samples)]
        n_unique_subj = len(set(groups))

        print(f"\n--- Cohort: {cohort_label} (N={n_samples} sessions, {n_unique_subj} unique subjects) ---")

        for seed in seeds:
            set_seed(seed)
            adj_random = make_random_adj(num_nodes=19, num_edges=18, seed=seed)

            if is_grouped:
                splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                splits = list(splitter.split(g_list, y_all, groups=groups))
            else:
                splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                splits = list(splitter.split(g_list, y_all))

            oof_probs = {m: np.zeros(n_samples) for m in models}
            oof_preds = {m: np.zeros(n_samples) for m in models}

            for fold, (train_idx, val_idx) in enumerate(splits):
                train_x = torch.stack([g_list[i]["x"] for i in train_idx])
                train_adj = torch.stack([g_list[i]["adj"] for i in train_idx])
                train_y = torch.tensor([g_list[i]["y"] for i in train_idx], dtype=torch.long)

                val_x = torch.stack([g_list[i]["x"] for i in val_idx])
                val_adj = torch.stack([g_list[i]["adj"] for i in val_idx])

                train_z, val_z, _, _ = compute_fold_normative_z(train_x, train_y, val_x)

                X_tr = train_z.view(len(train_idx), -1).numpy()
                X_va = val_z.view(len(val_idx), -1).numpy()
                y_tr = train_y.numpy()

                # 1. GCN-OMST
                p_omst, y_omst = train_eval_gcn(train_z, train_adj, train_y, val_z, val_adj, epochs=50, seed=seed + fold)
                oof_probs["GCN_OMST"][val_idx] = p_omst
                oof_preds["GCN_OMST"][val_idx] = y_omst

                # 2. GCN-Random
                val_adj_rnd = adj_random.unsqueeze(0).expand(len(val_idx), -1, -1)
                train_adj_rnd = adj_random.unsqueeze(0).expand(len(train_idx), -1, -1)
                p_rnd, y_rnd = train_eval_gcn(train_z, train_adj_rnd, train_y, val_z, val_adj_rnd, epochs=50, seed=seed + fold)
                oof_probs["GCN_Random"][val_idx] = p_rnd
                oof_preds["GCN_Random"][val_idx] = y_rnd

                # 3. GCN-Identity
                val_adj_id = adj_identity.unsqueeze(0).expand(len(val_idx), -1, -1)
                train_adj_id = adj_identity.unsqueeze(0).expand(len(train_idx), -1, -1)
                p_id, y_id = train_eval_gcn(train_z, train_adj_id, train_y, val_z, val_adj_id, epochs=50, seed=seed + fold)
                oof_probs["GCN_Identity"][val_idx] = p_id
                oof_preds["GCN_Identity"][val_idx] = y_id

                # 4. Tuned Logistic Regression (exact 7-C grid from scripts/17; participant-grouped inner CV in repeated-session cohorts)
                if is_grouped:
                    inner_groups = [groups[i] for i in train_idx]
                    inner_cv = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed).split(X_tr, y_tr, groups=inner_groups))
                else:
                    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
                lr = GridSearchCV(
                    LogisticRegression(max_iter=1000, random_state=seed),
                    param_grid_lr,
                    cv=inner_cv,
                    scoring="roc_auc",
                ).fit(X_tr, y_tr)
                oof_probs["Tuned_LogisticRegression"][val_idx] = lr.predict_proba(X_va)[:, 1]
                oof_preds["Tuned_LogisticRegression"][val_idx] = lr.predict(X_va)

                # 5. Tuned Ridge Classifier (exact 6-alpha grid from scripts/17; participant-grouped inner CV in repeated-session cohorts)
                rc = GridSearchCV(
                    RidgeClassifier(random_state=seed),
                    param_grid_rc,
                    cv=inner_cv,
                    scoring="roc_auc",
                ).fit(X_tr, y_tr)
                dec = rc.decision_function(X_va)
                oof_probs["Tuned_Ridge"][val_idx] = 1.0 / (1.0 + np.exp(-dec))
                oof_preds["Tuned_Ridge"][val_idx] = rc.predict(X_va)

                # 6. Random Forest
                rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed).fit(X_tr, y_tr)
                oof_probs["RandomForest"][val_idx] = rf.predict_proba(X_va)[:, 1]
                oof_preds["RandomForest"][val_idx] = rf.predict(X_va)

                # 7. Single Feature: Frontal Theta alone
                theta_tr = train_z[:, frontal_nodes, 0].mean(dim=-1).unsqueeze(-1).numpy()
                theta_va = val_z[:, frontal_nodes, 0].mean(dim=-1).unsqueeze(-1).numpy()
                lr_th = LogisticRegression(random_state=seed).fit(theta_tr, y_tr)
                oof_probs["Single_Theta"][val_idx] = lr_th.predict_proba(theta_va)[:, 1]
                oof_preds["Single_Theta"][val_idx] = lr_th.predict(theta_va)

            omst_auc_seed = roc_auc_score(y_all, oof_probs["GCN_OMST"])
            for m in models:
                auc = roc_auc_score(y_all, oof_probs[m])
                acc = accuracy_score(y_all, oof_preds[m])
                bal_acc = balanced_accuracy_score(y_all, oof_preds[m])
                records.append({
                    "cohort": cohort_label,
                    "n_sessions": n_samples,
                    "n_participants": n_unique_subj,
                    "seed": seed,
                    "model": m,
                    "auc": auc,
                    "delta_auc_vs_omst": auc - omst_auc_seed,
                    "accuracy": acc,
                    "balanced_accuracy": bal_acc,
                })

    df_multiseed = pd.DataFrame(records)

    summary_records = []
    for (cohort, model), grp in df_multiseed.groupby(["cohort", "model"], sort=False):
        aucs = grp["auc"].values
        deltas = grp["delta_auc_vs_omst"].values
        seed42_auc = float(grp[grp["seed"] == 42]["auc"].values[0])
        summary_records.append({
            "cohort": cohort,
            "n_sessions": int(grp["n_sessions"].iloc[0]),
            "n_participants": int(grp["n_participants"].iloc[0]),
            "model": model,
            "seed42_auc": seed42_auc,
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs, ddof=1)),
            "min_auc": float(np.min(aucs)),
            "max_auc": float(np.max(aucs)),
            "mean_delta_vs_omst": float(np.mean(deltas)),
            "std_delta_vs_omst": float(np.std(deltas, ddof=1)),
            "mean_acc": float(np.mean(grp["accuracy"].values)),
            "mean_bacc": float(np.mean(grp["balanced_accuracy"].values)),
        })
    df_summary = pd.DataFrame(summary_records)

    out_csv = REPORTS_DIR / "benchmark_multiseed_repeated_cv.csv"
    df_multiseed.to_csv(out_csv, index=False)
    out_summary_csv = REPORTS_DIR / "benchmark_multiseed_summary.csv"
    df_summary.to_csv(out_summary_csv, index=False)
    print(f"\n✅ Saved multi-seed CV raw records to: {out_csv}")
    print(f"✅ Saved multi-seed CV summary to: {out_summary_csv}")
    return df_multiseed, df_summary


# ==============================================================================
# 2. PLI EIGENVALUE SPECTRUM AUDIT & EUCLIDEAN VS RIEMANNIAN CONNECTIVITY (TABLE S22)
# ==============================================================================

def audit_pli_eigenvalue_spectrum():
    """Empirically audit minimum eigenvalues of sensor-level 19x19 PLI matrices across 677 primary sessions."""
    print("\n" + "=" * 75)
    print("🔍 [Part 2A] Auditing 19x19 PLI Matrix Eigenvalue Spectrum Across All 677 Primary Sessions...")
    print("=" * 75)

    cohort_files = [
        ("Study 1: Sleep Deprivation (ds004902)", "cohort_sleep_ds004902_graphs.pt"),
        ("Study 2: Outpatient MDD (EC)", "cohort_mumtaz_mdd_graphs.pt"),
        ("Study 3: Cavanagh Depression (ds003478)", "cohort_cavanagh_mdd_graphs.pt"),
        ("Study 4: UCSD Parkinson (ds002778)", "cohort_ucsd_parkinson_graphs.pt"),
        ("Study 5: Clinical Dementia (ds004504)", "cohort_dementia_ds004504_graphs.pt"),
        ("Study 6: UNM Parkinson (ds003490)", "cohort_unm_pd_ds003490_graphs.pt"),
        ("Study 7: Iowa Parkinson (ds004584)", "cohort_iowa_pd_ds004584_graphs.pt"),
    ]

    audit_rows = []
    all_min_evals = []
    total_non_spd = 0
    total_non_spd_after_1e4 = 0
    total_sessions = 0

    for c_label, fname in cohort_files:
        d = torch.load(DATA_DIR / fname)
        min_evals = []
        for g in d:
            c_mat = 0.5 * (g["pli"].numpy() + g["pli"].numpy().T)
            ev = eigh(c_mat, eigvals_only=True)
            min_evals.append(float(ev.min()))
        min_evals = np.array(min_evals)
        all_min_evals.extend(min_evals)

        n_sess = len(min_evals)
        n_neg = int(np.sum(min_evals <= 0.0))
        n_neg_1e4 = int(np.sum(min_evals + 1e-4 <= 0.0))
        total_sessions += n_sess
        total_non_spd += n_neg
        total_non_spd_after_1e4 += n_neg_1e4

        audit_rows.append({
            "cohort": c_label,
            "n_sessions": n_sess,
            "min_eigenvalue": float(min_evals.min()),
            "median_min_eigenvalue": float(np.median(min_evals)),
            "max_min_eigenvalue": float(min_evals.max()),
            "n_indefinite_raw": n_neg,
            "pct_indefinite_raw": 100.0 * n_neg / n_sess,
            "n_indefinite_after_1e4_shift": n_neg_1e4,
        })
        print(
            f"  {c_label:<40} | N={n_sess:<3} | lambda_min in [{min_evals.min():+.4f}, {min_evals.max():+.4f}] "
            f"| Indefinite: {n_neg}/{n_sess} ({100.0*n_neg/n_sess:.1f}%)"
        )

    all_min_evals = np.array(all_min_evals)
    audit_rows.append({
        "cohort": "Total Across All 7 Datasets (Primary Sessions)",
        "n_sessions": total_sessions,
        "min_eigenvalue": float(all_min_evals.min()),
        "median_min_eigenvalue": float(np.median(all_min_evals)),
        "max_min_eigenvalue": float(all_min_evals.max()),
        "n_indefinite_raw": total_non_spd,
        "pct_indefinite_raw": 100.0 * total_non_spd / total_sessions,
        "n_indefinite_after_1e4_shift": total_non_spd_after_1e4,
    })

    df_audit = pd.DataFrame(audit_rows)
    out_csv = REPORTS_DIR / "pli_spd_eigenvalue_audit.csv"
    df_audit.to_csv(out_csv, index=False)
    print(f"\n✅ Saved PLI eigenvalue audit to: {out_csv} (Total indefinite: {total_non_spd}/{total_sessions} = {100.0*total_non_spd/total_sessions:.2f}%)")
    return df_audit


def extract_connectivity_features(pli_tensor: torch.Tensor, mode: str) -> np.ndarray:
    """
    Extracts functional connectivity feature vectors from a batch of [N, 19, 19] PLI matrices:
      - mode == 'Euclidean_UpperTri_171D':
          Strictly upper-triangular off-diagonal PLI weights (19*18/2 = 171 features).
          Requires zero SPD assumptions.
      - mode == 'Riemannian_Gershgorin_SPD_190D':
          Adaptive diagonal shift C_spd = C + (max(0, -lambda_min) + 0.01) * I so every matrix
          is strictly SPD (all eigenvalues >= 0.01) without clipping relative eigenvalue spacing,
          followed by exact matrix logarithm logm(C_spd) and sqrt(2) Frobenius off-diagonal scaling (190 features).
      - mode == 'Riemannian_Higham_SPD_190D':
          Higham (1988) Frobenius-nearest SPD spectral floor projection: eigenvalues clipped at eps = 1e-4,
          followed by logm and sqrt(2) Frobenius scaling (190 features).
    """
    n_graphs = pli_tensor.shape[0]
    dim = 19

    if mode == "Euclidean_UpperTri_171D":
        triu_off = np.triu_indices(dim, k=1)  # 171 off-diagonal pairs
        out = np.zeros((n_graphs, len(triu_off[0])), dtype=np.float32)
        for k in range(n_graphs):
            c_mat = 0.5 * (pli_tensor[k] + pli_tensor[k].T).numpy()
            out[k] = c_mat[triu_off]
        return out

    triu_all = np.triu_indices(dim, k=0)  # 190 elements
    weights = np.ones(len(triu_all[0]), dtype=np.float32)
    for idx, (i, j) in enumerate(zip(triu_all[0], triu_all[1])):
        if i != j:
            weights[idx] = np.sqrt(2.0)

    out = np.zeros((n_graphs, len(triu_all[0])), dtype=np.float32)
    for k in range(n_graphs):
        c_mat = 0.5 * (pli_tensor[k] + pli_tensor[k].T).numpy()
        evals, evecs = eigh(c_mat)
        if mode == "Riemannian_Gershgorin_SPD_190D":
            # Shift entire spectrum upward if min eigenvalue < 0.01, preserving all eigenvalue gaps
            min_ev = float(evals.min())
            shift = max(0.0, 0.01 - min_ev)
            evals_spd = evals + shift
        elif mode == "Riemannian_Higham_SPD_190D":
            # Clip negative/near-zero eigenvalues at 1e-4 (Higham Frobenius projection)
            evals_spd = np.maximum(evals, 1e-4)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        log_c = evecs @ np.diag(np.log(evals_spd)) @ evecs.T
        out[k] = log_c[triu_all] * weights

    return out


def run_riemannian_and_euclidean_connectivity_benchmark():
    """Benchmark Euclidean upper-triangular PLI vectors vs Riemannian Log-Euclidean tangent space projections."""
    print("\n" + "=" * 75)
    print("🌐 [Part 2B] Running Euclidean (171-D) vs SPD-Conditioned Riemannian (190-D) Connectivity Benchmark (Table S22)...")
    print("=" * 75)

    audit_pli_eigenvalue_spectrum()
    cohort_benchmarks = load_all_benchmark_cohorts()

    modes = [
        "Euclidean_UpperTri_171D",
        "Riemannian_Gershgorin_SPD_190D",
        "Riemannian_Higham_SPD_190D",
    ]
    param_grid_lr = {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}
    param_grid_rc = {"alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}

    records = []

    for cohort_label, g_list, is_grouped, group_fn in cohort_benchmarks:
        n_samples = len(g_list)
        y_all = np.array([int(g["y"]) for g in g_list])
        groups = [group_fn(g) for g in g_list] if is_grouped else [str(i) for i in range(n_samples)]
        pli_stack = torch.stack([g["pli"] for g in g_list])

        if is_grouped:
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            splits = list(splitter.split(g_list, y_all, groups=groups))
        else:
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            splits = list(splitter.split(g_list, y_all))

        for mode in modes:
            X_conn = extract_connectivity_features(pli_stack, mode=mode)
            oof_probs = {"LR": np.zeros(n_samples), "Ridge": np.zeros(n_samples), "RF": np.zeros(n_samples)}
            oof_preds = {"LR": np.zeros(n_samples), "Ridge": np.zeros(n_samples), "RF": np.zeros(n_samples)}

            for train_idx, val_idx in splits:
                X_tr, y_tr = X_conn[train_idx], y_all[train_idx]
                X_va = X_conn[val_idx]

                # Fold-nested standardization relative to training healthy controls (y == 0)
                hc_mask = y_tr == 0
                mu = X_tr[hc_mask].mean(axis=0, keepdims=True)
                std = X_tr[hc_mask].std(axis=0, keepdims=True) + 1e-6
                X_tr_z = (X_tr - mu) / std
                X_va_z = (X_va - mu) / std

                # 1. Tuned LR (participant-grouped inner CV in repeated-session cohorts)
                if is_grouped:
                    inner_groups = [groups[i] for i in train_idx]
                    inner_cv = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(X_tr_z, y_tr, groups=inner_groups))
                else:
                    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                lr = GridSearchCV(
                    LogisticRegression(max_iter=1000, random_state=42),
                    param_grid_lr,
                    cv=inner_cv,
                    scoring="roc_auc",
                ).fit(X_tr_z, y_tr)
                oof_probs["LR"][val_idx] = lr.predict_proba(X_va_z)[:, 1]
                oof_preds["LR"][val_idx] = lr.predict(X_va_z)

                # 2. Tuned Ridge
                ridge = GridSearchCV(
                    RidgeClassifier(random_state=42),
                    param_grid_rc,
                    cv=inner_cv,
                    scoring="roc_auc",
                ).fit(X_tr_z, y_tr)
                dec = ridge.decision_function(X_va_z)
                oof_probs["Ridge"][val_idx] = 1.0 / (1.0 + np.exp(-dec))
                oof_preds["Ridge"][val_idx] = ridge.predict(X_va_z)

                # 3. Random Forest
                rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_tr_z, y_tr)
                oof_probs["RF"][val_idx] = rf.predict_proba(X_va_z)[:, 1]
                oof_preds["RF"][val_idx] = rf.predict(X_va_z)

            for clf_key in ["LR", "Ridge", "RF"]:
                m_full = f"{mode}_{clf_key}"
                auc, ci_lo, ci_hi = bootstrap_auc_ci(y_all, oof_probs[clf_key])
                acc = accuracy_score(y_all, oof_preds[clf_key])
                bacc = balanced_accuracy_score(y_all, oof_preds[clf_key])
                records.append({
                    "cohort": cohort_label,
                    "representation": mode,
                    "classifier": clf_key,
                    "model": m_full,
                    "n_samples": n_samples,
                    "auc": auc,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                    "accuracy": acc,
                    "balanced_accuracy": bacc,
                })

    df_riemann = pd.DataFrame(records)
    out_csv = REPORTS_DIR / "benchmark_riemannian_tangent_space.csv"
    df_riemann.to_csv(out_csv, index=False)
    print(f"\n✅ Saved Euclidean & Riemannian connectivity benchmark to: {out_csv}")
    return df_riemann


# ==============================================================================
# 3. RANDOM-EFFECTS META-ANALYSIS OF FRONTAL THETA POLARITY (TABLE S23)
# ==============================================================================

def dersimonian_laird_and_hksj(d_vals, var_vals):
    """Compute DerSimonian-Laird random-effects pooled effect size and Hartung-Knapp-Sidik-Jonkman adjustment."""
    d_vals = np.asarray(d_vals, dtype=float)
    var_vals = np.asarray(var_vals, dtype=float)
    k = len(d_vals)
    w_fe = 1.0 / var_vals
    d_fe = float(np.sum(w_fe * d_vals) / np.sum(w_fe))
    Q = float(np.sum(w_fe * (d_vals - d_fe) ** 2))
    df = k - 1
    p_q = float(1.0 - stats.chi2.cdf(Q, df)) if df > 0 else 1.0
    I2 = max(0.0, 100.0 * (Q - df) / Q) if Q > 0 else 0.0

    denom = float(np.sum(w_fe) - (np.sum(w_fe ** 2) / np.sum(w_fe)))
    tau2 = max(0.0, (Q - df) / denom) if denom > 0 else 0.0
    tau = float(np.sqrt(tau2))

    w_re = 1.0 / (var_vals + tau2)
    d_re = float(np.sum(w_re * d_vals) / np.sum(w_re))
    se_re = float(1.0 / np.sqrt(np.sum(w_re)))
    ci_lo = d_re - 1.959963984540054 * se_re
    ci_hi = d_re + 1.959963984540054 * se_re
    z = d_re / se_re
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))

    # Hartung-Knapp-Sidik-Jonkman (HKSJ) small-sample adjustment (with standard max(1, q_hk) modification)
    if k > 1:
        q_hk = float(np.sum(w_re * (d_vals - d_re) ** 2) / (k - 1))
        se_hk = se_re * np.sqrt(max(1.0, q_hk))
        t_crit = float(stats.t.ppf(0.975, df))
        hksj_lo = d_re - t_crit * se_hk
        hksj_hi = d_re + t_crit * se_hk
        p_hksj = float(2.0 * (1.0 - stats.t.cdf(abs(d_re / se_hk), df)))
    else:
        hksj_lo, hksj_hi, p_hksj = ci_lo, ci_hi, p_val

    return {
        "k_studies": k,
        "pooled_d": d_re,
        "se_d": se_re,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "hksj_ci_lower": hksj_lo,
        "hksj_ci_upper": hksj_hi,
        "z_score": z,
        "p_value": p_val,
        "p_hksj": p_hksj,
        "Q_stat": Q,
        "Q_df": df,
        "Q_p_value": p_q,
        "tau2": tau2,
        "tau": tau,
        "I2_percent": I2,
    }


def run_random_effects_meta_analysis():
    """Conduct random-effects meta-analysis and subgroup polarity testing across mutually exclusive cohorts."""
    print("\n" + "=" * 75)
    print("📊 [Part 3] Running Independent-Cohort Random-Effects Meta-Analysis & Sensitivity Suite (Table S23)...")
    print("=" * 75)

    with open(REPORTS_DIR / "master_results.json") as f:
        bp = json.load(f)["biophysics"]

    # Add strictly participant-independent UCSD subject-averaged contrast (15 PD patients [mean of ON & OFF] vs. 16 HCs)
    # and Morris-DeShon / Borenstein independent-groups equivalent d_IG for Study 1 crossover sleep deprivation (N=71 pairs, r=0.5709)
    custom_stats = {
        "UCSD_PD_SubjMean_vs_HC": (0.917791, 0.377815 ** 2),
        "Sleep_ds004902_SD_vs_NS_dIG": (0.267367, 0.112147 ** 2),
    }

    def get_study_stat(key):
        if key in custom_stats:
            return custom_stats[key]
        th = bp[key]["frontal_theta_5ch"]
        d = float(th["cohens_d"])
        se = float((th["ci_upper"] - th["ci_lower"]) / (2.0 * 1.959963984540054))
        return d, se ** 2

    # 1. Primary Strictly Independent Between-Subject Analysis (K=6 mutually exclusive cohorts, zero shared subjects/controls)
    dep_primary = ["Mumtaz_MDD_EC_vs_HC", "Cavanagh_DEP_vs_HC"]
    neu_primary_off = ["UCSD_PD_OFF_vs_HC", "Dementia_All_vs_HC", "UNM_PD_OFF_vs_HC", "Iowa_PD_vs_HC"]

    # 2. Sensitivity Configurations (Strictly Independent K=6 or K=7)
    neu_sens_subj_mean_pd = ["UCSD_PD_SubjMean_vs_HC", "Dementia_All_vs_HC", "UNM_PD_OFF_vs_HC", "Iowa_PD_vs_HC"]
    neu_sens_on_pd = ["UCSD_PD_ON_vs_HC", "Dementia_All_vs_HC", "UNM_PD_ON_vs_HC", "Iowa_PD_vs_HC"]
    neu_sens_ad_only = ["UCSD_PD_OFF_vs_HC", "Dementia_AD_vs_HC", "UNM_PD_OFF_vs_HC", "Iowa_PD_vs_HC"]
    neu_sens_ftd_only = ["UCSD_PD_OFF_vs_HC", "Dementia_FTD_vs_HC", "UNM_PD_OFF_vs_HC", "Iowa_PD_vs_HC"]
    slowing_plus_sleep = ["Sleep_ds004902_SD_vs_NS_dIG"] + neu_primary_off

    meta_rows = []

    def evaluate_two_group_partition(analysis_label, group1_name, group1_keys, group2_name, group2_keys):
        d1, v1 = zip(*[get_study_stat(k) for k in group1_keys])
        d2, v2 = zip(*[get_study_stat(k) for k in group2_keys])
        r1 = dersimonian_laird_and_hksj(d1, v1)
        r2 = dersimonian_laird_and_hksj(d2, v2)
        r_tot = dersimonian_laird_and_hksj(list(d1) + list(d2), list(v1) + list(v2))
        q_bet = max(0.0, r_tot["Q_stat"] - (r1["Q_stat"] + r2["Q_stat"]))
        p_bet = float(1.0 - stats.chi2.cdf(q_bet, 1))

        for g_name, res in [(group1_name, r1), (group2_name, r2), ("Overall_Pooled", r_tot)]:
            row = {"analysis_set": analysis_label, "domain_group": g_name}
            row.update(res)
            row["Q_between"] = q_bet
            row["p_between"] = p_bet
            meta_rows.append(row)

        print(
            f"  [{analysis_label}] {group1_name}: d={r1['pooled_d']:+.3f} [{r1['ci_lower']:+.3f}, {r1['ci_upper']:+.3f}] | "
            f"{group2_name}: d={r2['pooled_d']:+.3f} [{r2['ci_lower']:+.3f}, {r2['ci_upper']:+.3f}] | "
            f"Q_between={q_bet:.2f} (p={p_bet:.3e})"
        )

    # Run Primary K=6 Independent Benchmark
    evaluate_two_group_partition(
        "Primary_Independent_K6 (PD-OFF & Dementia-All)",
        "Depressive_HypoOscillatory_Domain",
        dep_primary,
        "Neurodegenerative_Slowing_Domain",
        neu_primary_off,
    )

    # Run Sensitivity 1: Participant-Averaged UCSD (15 PD vs 16 HC, zero repeated-session pooling)
    evaluate_two_group_partition(
        "Sens1_Independent_K6 (UCSD Subj-Mean 15v16 & UNM-OFF & Dementia-All)",
        "Depressive_HypoOscillatory_Domain",
        dep_primary,
        "Neurodegenerative_Slowing_Domain",
        neu_sens_subj_mean_pd,
    )

    # Run Sensitivity 2: Dopaminergic ON-Medication States (UCSD-ON & UNM-ON)
    evaluate_two_group_partition(
        "Sens2_Independent_K6 (PD-ON Medication States)",
        "Depressive_HypoOscillatory_Domain",
        dep_primary,
        "Neurodegenerative_Slowing_Domain",
        neu_sens_on_pd,
    )

    # Run Sensitivity 3: Alzheimer's Disease Only (No Shared Dementia Controls)
    evaluate_two_group_partition(
        "Sens3_Independent_K6 (AD-Only Dementia Contrast)",
        "Depressive_HypoOscillatory_Domain",
        dep_primary,
        "Neurodegenerative_Slowing_Domain",
        neu_sens_ad_only,
    )

    # Run Sensitivity 4: Frontotemporal Dementia Only (No Shared Dementia Controls)
    evaluate_two_group_partition(
        "Sens4_Independent_K6 (FTD-Only Dementia Contrast)",
        "Depressive_HypoOscillatory_Domain",
        dep_primary,
        "Neurodegenerative_Slowing_Domain",
        neu_sens_ftd_only,
    )

    # Run Sensitivity 5: Including Study 1 Sleep Deprivation on Independent-Groups Equivalent Scale (d_IG = +0.267, K=7)
    evaluate_two_group_partition(
        "Sens5_All7_Independent_Datasets (Slowing + Sleep Pressure d_IG)",
        "Depressive_HypoOscillatory_Domain",
        dep_primary,
        "Pathological_and_SleepPressure_Slowing_Domain",
        slowing_plus_sleep,
    )

    # Run Leave-One-Cohort-Out Sensitivity on Neurodegenerative Domain (against Depressive Domain)
    for drop_key in neu_primary_off:
        sub_neu = [k for k in neu_primary_off if k != drop_key]
        evaluate_two_group_partition(
            f"LeaveOneOut_Omit_{drop_key}",
            "Depressive_HypoOscillatory_Domain",
            dep_primary,
            "Neurodegenerative_Slowing_Domain",
            sub_neu,
        )

    df_meta = pd.DataFrame(meta_rows)
    out_csv = REPORTS_DIR / "meta_analysis_frontal_theta_polarity.csv"
    df_meta.to_csv(out_csv, index=False)
    print(f"\n✅ Saved audited random-effects meta-analysis suite to: {out_csv}")
    return df_meta


def main():
    """Execute multi-seed cross-validation, PLI eigenvalue audit, and meta-analysis benchmarks."""
    t0 = time.time()
    print("🚀 Starting Audited Step 34 Benchmarks...")
    run_multiseed_repeated_cv()
    run_riemannian_and_euclidean_connectivity_benchmark()
    run_random_effects_meta_analysis()
    print(f"\n✨ Step 34 completed successfully in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
