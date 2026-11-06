#!/usr/bin/env python3
"""
Step 08: Multi-Cohort Extended Model Benchmark & Graph Ablations.
Evaluates:
  1. Graph Adjacency Ablations:
     - GCN-OMST (Proposed biologically filtered topology)
     - GCN-Identity (A = I, isolated channels, zero message passing)
     - GCN-Random (Random connected 18-edge graph)
     - GCN-FullyConnected (A = 1, dense all-to-all connectivity)
  2. Tuned Classical Machine Learning Baselines (Nested within training folds):
     - Tuned L2-Logistic Regression (Grid search C in [0.001, 0.01, 0.1, 1.0, 10.0])
     - Tuned Ridge Classifier (Grid search alpha in [0.01, 0.1, 1.0, 10.0, 100.0])
     - Linear SVM (Tuned C)
     - RBF-Kernel SVM (Tuned C, gamma)
     - Random Forest (100 trees)
     - Single-Feature Theta alone (Frontal periodic theta)
     - Single-Feature Chi alone (Global aperiodic exponent)
  3. Full 4x4 Zero-Shot Cross-Cohort Transfer Matrix across all directed cohort pairs.

Outputs:
  reports/multicohort_extended_model_comparison.csv
  reports/multicohort_full_transfer_matrix.csv
"""

import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GridSearchCV

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
        return base_auc, base_auc, base_auc
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
    return torch.tensor(adj, dtype=torch.float32)


def train_eval_gcn(train_z, train_adj, train_y, val_z, val_adj, epochs=35, lr=0.005, weight_decay=1e-4, seed=42):
    torch.manual_seed(seed)
    model = NormativeGCN(in_node_features=4, hidden_dim=32, num_classes=2, dropout=0.25)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

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
    print(f"📊 Benchmarking all models on: {cohort_name}")
    set_seed(42)

    valid_graphs = [g for g in graph_list if int(g['y']) in (0, 1)]
    y_all = np.array([int(g['y']) for g in valid_graphs])
    n_samples = len(valid_graphs)
    groups = [g.get('bids_id', g.get('sub_name', g['subject_id'])) for g in valid_graphs]

    if use_group_kfold:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_gen = list(splitter.split(valid_graphs, y_all, groups=groups))
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_gen = list(splitter.split(valid_graphs, y_all))

    # Prediction containers
    models = [
        "GCN_OMST", "GCN_Identity", "GCN_Random", "GCN_Full",
        "Tuned_LogisticRegression", "Tuned_Ridge", "Linear_SVM", "RBF_SVM",
        "RandomForest", "Single_Theta", "Single_Chi"
    ]
    oof_probs = {m: np.zeros(n_samples) for m in models}
    oof_preds = {m: np.zeros(n_samples) for m in models}

    # Static random & full adjacency matrices
    adj_identity = torch.eye(19, dtype=torch.float32)
    adj_random = make_random_adj(num_nodes=19, num_edges=18, seed=42)
    adj_full = torch.ones((19, 19), dtype=torch.float32)

    for fold, (train_idx, val_idx) in enumerate(split_gen):
        train_x = torch.stack([valid_graphs[i]['x'] for i in train_idx])
        train_adj_omst = torch.stack([valid_graphs[i]['adj'] for i in train_idx])
        train_y = torch.tensor([valid_graphs[i]['y'] for i in train_idx], dtype=torch.long)

        val_x = torch.stack([valid_graphs[i]['x'] for i in val_idx])
        val_adj_omst = torch.stack([valid_graphs[i]['adj'] for i in val_idx])
        val_y = torch.tensor([valid_graphs[i]['y'] for i in val_idx], dtype=torch.long)

        train_z, val_z, _, _ = compute_fold_normative_z(train_x, train_y, val_x)

        # Tabular arrays
        X_tr = train_z.view(len(train_idx), -1).numpy()
        X_va = val_z.view(len(val_idx), -1).numpy()
        y_tr = train_y.numpy()

        # 1. GCN_OMST
        probs, preds = train_eval_gcn(train_z, train_adj_omst, train_y, val_z, val_adj_omst, seed=42+fold)
        oof_probs["GCN_OMST"][val_idx] = probs
        oof_preds["GCN_OMST"][val_idx] = preds

        # 2. GCN_Identity (A = I)
        val_adj_id = adj_identity.unsqueeze(0).expand(len(val_idx), -1, -1)
        train_adj_id = adj_identity.unsqueeze(0).expand(len(train_idx), -1, -1)
        probs, preds = train_eval_gcn(train_z, train_adj_id, train_y, val_z, val_adj_id, seed=42+fold)
        oof_probs["GCN_Identity"][val_idx] = probs
        oof_preds["GCN_Identity"][val_idx] = preds

        # 3. GCN_Random
        val_adj_rnd = adj_random.unsqueeze(0).expand(len(val_idx), -1, -1)
        train_adj_rnd = adj_random.unsqueeze(0).expand(len(train_idx), -1, -1)
        probs, preds = train_eval_gcn(train_z, train_adj_rnd, train_y, val_z, val_adj_rnd, seed=42+fold)
        oof_probs["GCN_Random"][val_idx] = probs
        oof_preds["GCN_Random"][val_idx] = preds

        # 4. GCN_Full (A = 1)
        val_adj_fl = adj_full.unsqueeze(0).expand(len(val_idx), -1, -1)
        train_adj_fl = adj_full.unsqueeze(0).expand(len(train_idx), -1, -1)
        probs, preds = train_eval_gcn(train_z, train_adj_fl, train_y, val_z, val_adj_fl, seed=42+fold)
        oof_probs["GCN_Full"][val_idx] = probs
        oof_preds["GCN_Full"][val_idx] = preds

        # 5. Tuned L2-Logistic Regression
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

        # 10. Single Feature: Frontal Theta alone (feature index 0 on frontal channels Fp1, Fp2, F3, F4, Fz: nodes 0,1,2,3,16)
        frontal_nodes = [0, 1, 2, 3, 16]
        theta_tr = train_z[:, frontal_nodes, 0].mean(dim=-1).unsqueeze(-1).numpy()
        theta_va = val_z[:, frontal_nodes, 0].mean(dim=-1).unsqueeze(-1).numpy()
        lr_theta = LogisticRegression(random_state=42).fit(theta_tr, y_tr)
        oof_probs["Single_Theta"][val_idx] = lr_theta.predict_proba(theta_va)[:, 1]
        oof_preds["Single_Theta"][val_idx] = lr_theta.predict(theta_va)

        # 11. Single Feature: Global Chi alone (feature index 3 across all 19 channels)
        chi_tr = train_z[:, :, 3].mean(dim=-1).unsqueeze(-1).numpy()
        chi_va = val_z[:, :, 3].mean(dim=-1).unsqueeze(-1).numpy()
        lr_chi = LogisticRegression(random_state=42).fit(chi_tr, y_tr)
        oof_probs["Single_Chi"][val_idx] = lr_chi.predict_proba(chi_va)[:, 1]
        oof_preds["Single_Chi"][val_idx] = lr_chi.predict(chi_va)

    results = []
    for m in models:
        auc, ci_lo, ci_hi = bootstrap_auc_ci(y_all, oof_probs[m])
        bacc = balanced_accuracy_score(y_all, oof_preds[m])
        acc = accuracy_score(y_all, oof_preds[m])
        results.append({
            'cohort': cohort_name,
            'model': m,
            'n_samples': n_samples,
            'auc': auc,
            'ci_lower': ci_lo,
            'ci_upper': ci_hi,
            'balanced_acc': bacc,
            'accuracy': acc
        })
        print(f"  {m:25s} | AUC={auc:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] | BalAcc={bacc*100:.1f}%")

    return results


def run_directed_transfer(src_graphs, tgt_graphs, src_name: str, tgt_name: str):
    set_seed(42)
    src_valid = [g for g in src_graphs if int(g['y']) in (0, 1)]
    tgt_valid = [g for g in tgt_graphs if int(g['y']) in (0, 1)]

    train_x = torch.stack([g['x'] for g in src_valid])
    train_adj = torch.stack([g['adj'] for g in src_valid])
    train_y = torch.tensor([g['y'] for g in src_valid], dtype=torch.long)

    test_x = torch.stack([g['x'] for g in tgt_valid])
    test_adj = torch.stack([g['adj'] for g in tgt_valid])
    test_y = np.array([int(g['y']) for g in tgt_valid])

    train_z, test_z, _, _ = compute_fold_normative_z(train_x, train_y, test_x)

    # Train GCN-OMST
    probs_gcn, preds_gcn = train_eval_gcn(train_z, train_adj, train_y, test_z, test_adj, epochs=40, seed=42)
    auc_gcn, ci_lo_gcn, ci_hi_gcn = bootstrap_auc_ci(test_y, probs_gcn)
    bacc_gcn = balanced_accuracy_score(test_y, preds_gcn)

    # Train Tuned LR
    X_tr = train_z.view(len(src_valid), -1).numpy()
    X_te = test_z.view(len(tgt_valid), -1).numpy()
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42).fit(X_tr, train_y.numpy())
    probs_lr = lr.predict_proba(X_te)[:, 1]
    preds_lr = lr.predict(X_te)
    auc_lr, ci_lo_lr, ci_hi_lr = bootstrap_auc_ci(test_y, probs_lr)
    bacc_lr = balanced_accuracy_score(test_y, preds_lr)

    return {
        'source_cohort': src_name,
        'target_cohort': tgt_name,
        'n_source': len(src_valid),
        'n_target': len(tgt_valid),
        'gcn_auc': auc_gcn,
        'gcn_ci_lower': ci_lo_gcn,
        'gcn_ci_upper': ci_hi_gcn,
        'gcn_bal_acc': bacc_gcn,
        'lr_auc': auc_lr,
        'lr_ci_lower': ci_lo_lr,
        'lr_ci_upper': ci_hi_lr,
        'lr_bal_acc': bacc_lr
    }


def main():
    print("🚀 Running Extended Multi-Cohort Model Benchmark & Graph Ablations...")
    # Load 4 cohorts
    mumtaz_graphs = torch.load(DATA_DIR / "cohort_mumtaz_mdd_graphs.pt")
    cavanagh_graphs = torch.load(DATA_DIR / "cohort_cavanagh_mdd_graphs.pt")
    parkinson_graphs = torch.load(DATA_DIR / "cohort_ucsd_parkinson_graphs.pt")
    sleep_graphs = torch.load(DATA_DIR / "cohort_sleep_ds004902_graphs.pt")

    # Parkinson subsets
    hc_graphs = [g for g in parkinson_graphs if g['group'] == 'HC']
    on_graphs = [g for g in parkinson_graphs if g['group'] == 'PD_ON']
    off_graphs = [g for g in parkinson_graphs if g['group'] == 'PD_OFF']
    pd_on_graphs = hc_graphs + on_graphs
    pd_off_graphs = hc_graphs + off_graphs

    all_benchmarks = []
    # 1. Mumtaz MDD
    all_benchmarks.extend(evaluate_cohort_all_models(mumtaz_graphs, "Study 2: Mumtaz Outpatient MDD", use_group_kfold=False))
    # 2. Cavanagh Depression
    all_benchmarks.extend(evaluate_cohort_all_models(cavanagh_graphs, "Study 3: Cavanagh University Depression", use_group_kfold=False))
    # 3. Parkinson All Sessions
    all_benchmarks.extend(evaluate_cohort_all_models(parkinson_graphs, "Study 4: Parkinson All Sessions", use_group_kfold=True))
    # 4. Parkinson ON vs HC
    all_benchmarks.extend(evaluate_cohort_all_models(pd_on_graphs, "Study 4b: Parkinson ON vs HC", use_group_kfold=False))
    # 5. Parkinson OFF vs HC
    all_benchmarks.extend(evaluate_cohort_all_models(pd_off_graphs, "Study 4a: Parkinson OFF vs HC", use_group_kfold=False))
    # 6. Sleep Single Session
    all_benchmarks.extend(evaluate_cohort_all_models(sleep_graphs, "Study 1: Sleep Debt Single-Sess", use_group_kfold=True))

    df_models = pd.DataFrame(all_benchmarks)
    out_models_csv = REPORTS_DIR / "multicohort_extended_model_comparison.csv"
    df_models.to_csv(out_models_csv, index=False)
    print(f"\n✓ Saved extended model benchmark to: {out_models_csv}")

    # Full 4x4 Zero-Shot Cross-Cohort Transfer Matrix
    cohort_dict = {
        'Sleep_Debt': sleep_graphs,
        'Mumtaz_MDD': mumtaz_graphs,
        'Cavanagh_MDD': cavanagh_graphs,
        'Parkinson': parkinson_graphs
    }
    cohort_names = list(cohort_dict.keys())

    print("\n" + "=" * 70)
    print("🌐 Computing Full 4x4 Zero-Shot Cross-Cohort Transfer Matrix...")
    transfer_results = []
    for src in cohort_names:
        for tgt in cohort_names:
            if src == tgt:
                continue
            res = run_directed_transfer(cohort_dict[src], cohort_dict[tgt], src, tgt)
            transfer_results.append(res)
            print(f"  Transfer: {src:12s} -> {tgt:12s} | GCN AUC = {res['gcn_auc']:.3f} [{res['gcn_ci_lower']:.3f}, {res['gcn_ci_upper']:.3f}] | LR AUC = {res['lr_auc']:.3f}")

    df_trans = pd.DataFrame(transfer_results)
    out_trans_csv = REPORTS_DIR / "multicohort_full_transfer_matrix.csv"
    df_trans.to_csv(out_trans_csv, index=False)
    print(f"✓ Saved full transfer matrix to: {out_trans_csv}")


if __name__ == "__main__":
    main()
