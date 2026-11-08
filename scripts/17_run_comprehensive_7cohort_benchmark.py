#!/usr/bin/env python3
"""
Step 17: Comprehensive 7-Cohort Extended Model Benchmark & Clinical Regression.
Evaluates:
  1. Models:
     - GCN-OMST (Proposed topological graph filtering)
     - GCN-Identity (A = I, no message passing)
     - GCN-Random (Random 18-edge topology)
     - GCN-Full (A = 1, dense topology)
     - Tuned L2-Logistic Regression
     - Tuned Ridge Classifier
     - Linear SVM (Tuned C)
     - RBF SVM (Tuned C, gamma)
     - Random Forest (100 trees)
     - GBDT (HistGradientBoostingClassifier)
     - Single-Feature Theta alone
     - Single-Feature Chi alone
  2. Cohorts Evaluated:
     - Study 1: Sleep Debt Single-Sess (ds004902, N=36)
     - Study 2: Mumtaz Outpatient MDD (EC, N=64)
     - Study 2b: Mumtaz Outpatient MDD (EO, N=64)
     - Study 3: Cavanagh University Depression (N=121)
     - Study 4: UCSD Parkinson All Sessions (ds002778, N=47)
     - Study 5a: Dementia AD vs CTL (ds004504, N=65)
     - Study 5b: Dementia FTD vs CTL (ds004504, N=52)
     - Study 5c: Dementia All (AD+FTD) vs CTL (ds004504, N=88)
     - Study 6a: UNM PD-OFF vs CTL (ds003490, N=50)
     - Study 6b: UNM PD-ON vs CTL (ds003490, N=50)
     - Study 6c: UNM PD All Sessions (ds003490, N=75)
     - Study 7: Iowa PD vs Control (ds004584, N=149)
  3. Clinical Continuous Scale Regression:
     - Dementia MMSE score (N=88)
     - Iowa PD MoCA score (N=149)
     - Iowa PD UPDRS Part-III motor score (N=100)

Outputs:
  reports/multicohort_7cohort_extended_model_comparison.csv
  reports/multicohort_clinical_regression_benchmarks.csv
"""

import random
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, r2_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, KFold, GridSearchCV

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = BENCHMARK_ROOT / "src"
sys.path.append(str(SRC_DIR))
from harmonizer_and_graph import NormativeGCN, compute_fold_normative_z, PurePyTorchGCNLayer

DATA_DIR = BENCHMARK_ROOT / "data" / "derivatives"
REPORTS_DIR = BENCHMARK_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def bootstrap_auc_ci(y_true, y_prob, n_boot: int = 1000):
    n = len(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan
    base_auc = roc_auc_score(y_true, y_prob)
    rng = np.random.RandomState(42)
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


def evaluate_cohort_all_models(graph_list, cohort_name: str, use_group_kfold: bool = False, n_splits: int = 5):
    print(f"\n=======================================================")
    print(f"📊 Benchmarking all models on: {cohort_name} (N={len(graph_list)})")
    set_seed(42)

    valid_graphs = [g for g in graph_list if int(g['y']) in (0, 1)]
    y_all = np.array([int(g['y']) for g in valid_graphs])
    n_samples = len(valid_graphs)
    groups = [g.get('bids_id', g.get('sub_name', g.get('subject_id', str(i)))) for i, g in enumerate(valid_graphs)]

    if use_group_kfold:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_gen = list(splitter.split(valid_graphs, y_all, groups=groups))
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_gen = list(splitter.split(valid_graphs, y_all))

    models = [
        "GCN_OMST", "GCN_Identity", "GCN_Random", "GCN_Full",
        "Tuned_LogisticRegression", "Tuned_Ridge", "Linear_SVM", "RBF_SVM",
        "RandomForest", "GBDT", "Single_Theta", "Single_Chi"
    ]
    oof_probs = {m: np.zeros(n_samples) for m in models}
    oof_preds = {m: np.zeros(n_samples) for m in models}

    adj_identity = torch.eye(19, dtype=torch.float32)
    adj_random = make_random_adj(num_nodes=19, num_edges=18, seed=42)
    adj_full = torch.ones((19, 19), dtype=torch.float32) / 19.0

    for fold, (train_idx, val_idx) in enumerate(split_gen):
        train_x = torch.stack([valid_graphs[i]['x'] for i in train_idx])
        train_adj_omst = torch.stack([valid_graphs[i]['adj'] for i in train_idx])
        train_y = torch.tensor([valid_graphs[i]['y'] for i in train_idx], dtype=torch.long)

        val_x = torch.stack([valid_graphs[i]['x'] for i in val_idx])
        val_adj_omst = torch.stack([valid_graphs[i]['adj'] for i in val_idx])
        val_y = torch.tensor([valid_graphs[i]['y'] for i in val_idx], dtype=torch.long)

        train_z, val_z, _, _ = compute_fold_normative_z(train_x, train_y, val_x)

        X_tr = train_z.view(len(train_idx), -1).numpy()
        X_va = val_z.view(len(val_idx), -1).numpy()
        y_tr = train_y.numpy()

        # 1. GCN_OMST
        probs, preds = train_eval_gcn(train_z, train_adj_omst, train_y, val_z, val_adj_omst, seed=42 + fold)
        oof_probs["GCN_OMST"][val_idx] = probs
        oof_preds["GCN_OMST"][val_idx] = preds

        # 2. GCN_Identity (A = I)
        val_adj_id = adj_identity.unsqueeze(0).expand(len(val_idx), -1, -1)
        train_adj_id = adj_identity.unsqueeze(0).expand(len(train_idx), -1, -1)
        probs, preds = train_eval_gcn(train_z, train_adj_id, train_y, val_z, val_adj_id, seed=42 + fold)
        oof_probs["GCN_Identity"][val_idx] = probs
        oof_preds["GCN_Identity"][val_idx] = preds

        # 3. GCN_Random
        val_adj_rnd = adj_random.unsqueeze(0).expand(len(val_idx), -1, -1)
        train_adj_rnd = adj_random.unsqueeze(0).expand(len(train_idx), -1, -1)
        probs, preds = train_eval_gcn(train_z, train_adj_rnd, train_y, val_z, val_adj_rnd, seed=42 + fold)
        oof_probs["GCN_Random"][val_idx] = probs
        oof_preds["GCN_Random"][val_idx] = preds

        # 4. GCN_Full (A = 1/N)
        val_adj_fl = adj_full.unsqueeze(0).expand(len(val_idx), -1, -1)
        train_adj_fl = adj_full.unsqueeze(0).expand(len(train_idx), -1, -1)
        probs, preds = train_eval_gcn(train_z, train_adj_fl, train_y, val_z, val_adj_fl, seed=42 + fold)
        oof_probs["GCN_Full"][val_idx] = probs
        oof_preds["GCN_Full"][val_idx] = preds

        # 5. Tuned Logistic Regression
        param_grid_lr = {'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}
        grid_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_grid_lr, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), scoring='roc_auc')
        grid_lr.fit(X_tr, y_tr)
        oof_probs["Tuned_LogisticRegression"][val_idx] = grid_lr.predict_proba(X_va)[:, 1]
        oof_preds["Tuned_LogisticRegression"][val_idx] = grid_lr.predict(X_va)

        # 6. Tuned Ridge Classifier
        param_grid_rc = {'alpha': [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}
        grid_rc = GridSearchCV(RidgeClassifier(random_state=42), param_grid_rc, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), scoring='roc_auc')
        grid_rc.fit(X_tr, y_tr)
        dec = grid_rc.decision_function(X_va)
        oof_probs["Tuned_Ridge"][val_idx] = 1.0 / (1.0 + np.exp(-dec))
        oof_preds["Tuned_Ridge"][val_idx] = grid_rc.predict(X_va)

        # 7. Linear SVM
        param_grid_svm = {'C': [0.01, 0.1, 1.0, 10.0]}
        grid_lsvm = GridSearchCV(SVC(kernel='linear', probability=True, random_state=42), param_grid_svm, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), scoring='roc_auc')
        grid_lsvm.fit(X_tr, y_tr)
        oof_probs["Linear_SVM"][val_idx] = grid_lsvm.predict_proba(X_va)[:, 1]
        oof_preds["Linear_SVM"][val_idx] = grid_lsvm.predict(X_va)

        # 8. RBF SVM
        param_grid_rbf = {'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto', 0.01, 0.1]}
        grid_rbf = GridSearchCV(SVC(kernel='rbf', probability=True, random_state=42), param_grid_rbf, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), scoring='roc_auc')
        grid_rbf.fit(X_tr, y_tr)
        oof_probs["RBF_SVM"][val_idx] = grid_rbf.predict_proba(X_va)[:, 1]
        oof_preds["RBF_SVM"][val_idx] = grid_rbf.predict(X_va)

        # 9. Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_tr, y_tr)
        oof_probs["RandomForest"][val_idx] = rf.predict_proba(X_va)[:, 1]
        oof_preds["RandomForest"][val_idx] = rf.predict(X_va)

        # 10. GBDT
        gbdt = HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=42)
        gbdt.fit(X_tr, y_tr)
        oof_probs["GBDT"][val_idx] = gbdt.predict_proba(X_va)[:, 1]
        oof_preds["GBDT"][val_idx] = gbdt.predict(X_va)

        # 11. Single Feature: Frontal Theta alone
        frontal_nodes = [0, 1, 2, 3, 16]  # Fp1, Fp2, F3, F4, Fz
        theta_tr = train_z[:, frontal_nodes, 0].mean(dim=-1).unsqueeze(-1).numpy()
        theta_va = val_z[:, frontal_nodes, 0].mean(dim=-1).unsqueeze(-1).numpy()
        lr_theta = LogisticRegression(random_state=42).fit(theta_tr, y_tr)
        oof_probs["Single_Theta"][val_idx] = lr_theta.predict_proba(theta_va)[:, 1]
        oof_preds["Single_Theta"][val_idx] = lr_theta.predict(theta_va)

        # 12. Single Feature: Global Chi alone
        chi_tr = train_z[:, :, 3].mean(dim=-1).unsqueeze(-1).numpy()
        chi_va = val_z[:, :, 3].mean(dim=-1).unsqueeze(-1).numpy()
        lr_chi = LogisticRegression(random_state=42).fit(chi_tr, y_tr)
        oof_probs["Single_Chi"][val_idx] = lr_chi.predict_proba(chi_va)[:, 1]
        oof_preds["Single_Chi"][val_idx] = lr_chi.predict(chi_va)

    results = []
    print(f"{'Model':<25} | {'AUC':<7} | {'95% CI':<16} | {'Acc':<7} | {'Bal Acc':<7}")
    print("-" * 72)
    for m in models:
        auc, ci_low, ci_high = bootstrap_auc_ci(y_all, oof_probs[m])
        acc = accuracy_score(y_all, oof_preds[m])
        bal_acc = balanced_accuracy_score(y_all, oof_preds[m])
        print(f"{m:<25} | {auc:6.3f}  | [{ci_low:5.3f}, {ci_high:5.3f}]  | {acc:6.3f}  | {bal_acc:6.3f}")
        results.append({
            'cohort': cohort_name,
            'model': m,
            'auc': auc,
            'ci_lower': ci_low,
            'ci_upper': ci_high,
            'accuracy': acc,
            'balanced_accuracy': bal_acc
        })
    return results


def run_clinical_scale_regression(graph_list, scale_name: str, cohort_label: str):
    print(f"\n=======================================================")
    print(f"🎯 Evaluating Clinical Continuous Scale Regression: {cohort_label} ({scale_name})")
    
    # Filter valid target values
    valid_items = [g for g in graph_list if pd.notnull(g.get(scale_name))]
    if len(valid_items) < 20:
        print(f"Too few valid samples ({len(valid_items)}), skipping.")
        return []

    y_target = np.array([float(g[scale_name]) for g in valid_items])
    n_samples = len(valid_items)
    print(f"Sample size: N={n_samples}, target mean={y_target.mean():.2f} ± {y_target.std():.2f} (range [{y_target.min():.1f}, {y_target.max():.1f}])")

    # Features: flatten 19 x 4 = 76 electrophysiological features
    X_raw = np.stack([g['x'].numpy().flatten() for g in valid_items])  # [N, 76]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    reg_models = {
        "Ridge_Regression": Ridge(alpha=10.0),
        "SVR_RBF": SVR(C=1.0, epsilon=0.1),
        "Random_Forest_Regressor": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        "GBDT_Regressor": HistGradientBoostingRegressor(max_iter=100, max_depth=4, random_state=42)
    }

    oof_preds = {m: np.zeros(n_samples) for m in reg_models}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_raw, y_target)):
        X_tr, y_tr = X_raw[train_idx], y_target[train_idx]
        X_va, y_va = X_raw[val_idx], y_target[val_idx]

        # Standardize within fold
        mean = np.mean(X_tr, axis=0, keepdims=True)
        std = np.std(X_tr, axis=0, keepdims=True) + 1e-6
        X_tr_norm = (X_tr - mean) / std
        X_va_norm = (X_va - mean) / std

        for m_name, model in reg_models.items():
            model.fit(X_tr_norm, y_tr)
            oof_preds[m_name][val_idx] = model.predict(X_va_norm)

    reg_results = []
    print(f"{'Model':<26} | {'Pearson r':<10} | {'p-value':<10} | {'R2':<8} | {'MAE':<8}")
    print("-" * 72)
    from scipy.stats import pearsonr
    for m_name in reg_models:
        preds = oof_preds[m_name]
        r, p = pearsonr(y_target, preds)
        r2 = r2_score(y_target, preds)
        mae = mean_absolute_error(y_target, preds)
        print(f"{m_name:<26} | {r:+8.4f}   | {p:.4e} | {r2:7.3f}  | {mae:7.3f}")
        reg_results.append({
            'cohort': cohort_label,
            'scale': scale_name,
            'n_samples': n_samples,
            'model': m_name,
            'pearson_r': r,
            'p_value': p,
            'r2_score': r2,
            'mae': mae
        })
    return reg_results


def main():
    print("🚀 Running Comprehensive 7-Cohort Extended Benchmark...")
    t_start = time.time()

    # Load All Datasets
    print("Loading derivative graph datasets...")
    sleep_graphs = torch.load(DATA_DIR / "cohort_sleep_ds004902_graphs.pt")
    mumtaz_ec_graphs = torch.load(DATA_DIR / "cohort_mumtaz_mdd_graphs.pt")
    cavanagh_graphs = torch.load(DATA_DIR / "cohort_cavanagh_mdd_graphs.pt")
    ucsd_pd_graphs = torch.load(DATA_DIR / "cohort_ucsd_parkinson_graphs.pt")
    dementia_graphs = torch.load(DATA_DIR / "cohort_dementia_ds004504_graphs.pt")
    unm_pd_graphs = torch.load(DATA_DIR / "cohort_unm_pd_ds003490_graphs.pt")
    iowa_pd_graphs = torch.load(DATA_DIR / "cohort_iowa_pd_ds004584_graphs.pt")

    # Check for Mumtaz EO if available
    eo_path = DATA_DIR / "cohort_mumtaz_eo_mdd_graphs.pt"
    mumtaz_eo_graphs = torch.load(eo_path) if eo_path.exists() else None

    # Dementia Subsets: AD vs CTL, FTD vs CTL, All vs CTL
    ad_ctl = [g for g in dementia_graphs if g.get('diagnosis') in ('Alzheimer', 'Control')]
    ftd_ctl = [g for g in dementia_graphs if g.get('diagnosis') in ('FTD', 'Control')]

    # UNM PD Subsets: PD-OFF vs CTL, PD-ON vs CTL
    unm_off_ctl = [g for g in unm_pd_graphs if g['med_state'] in ('OFF', 'CTL')]
    unm_on_ctl = [g for g in unm_pd_graphs if g['med_state'] in ('ON', 'CTL')]

    all_benchmarks = []

    # 1. Study 1: Sleep Deprivation Single-Session
    all_benchmarks.extend(evaluate_cohort_all_models(sleep_graphs, "Study 1: Sleep Deprivation (ds004902)", use_group_kfold=True))

    # 2. Study 2: Mumtaz Outpatient MDD (EC)
    all_benchmarks.extend(evaluate_cohort_all_models(mumtaz_ec_graphs, "Study 2: Mumtaz MDD (Eyes Closed)", use_group_kfold=False))

    # 2b. Study 2b: Mumtaz Outpatient MDD (EO)
    if mumtaz_eo_graphs:
        all_benchmarks.extend(evaluate_cohort_all_models(mumtaz_eo_graphs, "Study 2b: Mumtaz MDD (Eyes Open)", use_group_kfold=False))

    # 3. Study 3: Cavanagh University Depression
    all_benchmarks.extend(evaluate_cohort_all_models(cavanagh_graphs, "Study 3: Cavanagh University Depression", use_group_kfold=False))

    # 4. Study 4: UCSD Parkinson All Sessions
    all_benchmarks.extend(evaluate_cohort_all_models(ucsd_pd_graphs, "Study 4: UCSD Parkinson All Sessions", use_group_kfold=True))

    # 5. Study 5: Dementia Clinical Cohort (ds004504)
    all_benchmarks.extend(evaluate_cohort_all_models(ad_ctl, "Study 5a: Dementia AD vs Control", use_group_kfold=False))
    all_benchmarks.extend(evaluate_cohort_all_models(ftd_ctl, "Study 5b: Dementia FTD vs Control", use_group_kfold=False))
    all_benchmarks.extend(evaluate_cohort_all_models(dementia_graphs, "Study 5c: Dementia All (AD+FTD) vs Control", use_group_kfold=False))

    # 6. Study 6: UNM Parkinson Cohort (ds003490)
    all_benchmarks.extend(evaluate_cohort_all_models(unm_off_ctl, "Study 6a: UNM PD-OFF vs Control", use_group_kfold=False))
    all_benchmarks.extend(evaluate_cohort_all_models(unm_on_ctl, "Study 6b: UNM PD-ON vs Control", use_group_kfold=False))
    all_benchmarks.extend(evaluate_cohort_all_models(unm_pd_graphs, "Study 6c: UNM PD All Sessions", use_group_kfold=True))

    # 7. Study 7: Iowa Parkinson Cohort (ds004584)
    all_benchmarks.extend(evaluate_cohort_all_models(iowa_pd_graphs, "Study 7: Iowa Parkinson (ds004584)", use_group_kfold=False))

    # Save Classification Benchmarks
    df_models = pd.DataFrame(all_benchmarks)
    out_models_csv = REPORTS_DIR / "multicohort_7cohort_extended_model_comparison.csv"
    df_models.to_csv(out_models_csv, index=False)
    print(f"\n✅ Saved 7-Cohort model comparison to: {out_models_csv}")

    # Clinical Continuous Scale Regression
    print("\n" + "=" * 75)
    print("📈 Running Clinical Scale Regression Benchmarks...")
    reg_results = []
    # MMSE in Dementia
    reg_results.extend(run_clinical_scale_regression(dementia_graphs, "mmse", "Dementia_ds004504"))
    # MoCA in Iowa PD
    reg_results.extend(run_clinical_scale_regression(iowa_pd_graphs, "moca", "Iowa_PD_ds004584"))
    # UPDRS in Iowa PD
    reg_results.extend(run_clinical_scale_regression([g for g in iowa_pd_graphs if g['group'] == 'PD'], "updrs", "Iowa_PD_ds004584"))

    df_reg = pd.DataFrame(reg_results)
    out_reg_csv = REPORTS_DIR / "multicohort_clinical_regression_benchmarks.csv"
    df_reg.to_csv(out_reg_csv, index=False)
    print(f"✅ Saved clinical regression benchmarks to: {out_reg_csv}")

    elapsed = time.time() - t_start
    print(f"\n🎉 All 7-Cohort Benchmarks completed in {elapsed:.1f}s!")


if __name__ == "__main__":
    main()
