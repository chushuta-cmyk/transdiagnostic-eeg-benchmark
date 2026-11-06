#!/usr/bin/env python3
"""
Step 18: Cross-Site Parkinson's Generalization & Comprehensive Transdiagnostic Transfer Matrix.
Evaluates:
  1. Tri-Center Parkinson's Generalization (Within-Disorder Cross-Site Transfer):
     - UCSD PD (ds002778, N=47)
     - UNM PD (ds003490 OFF, N=50)
     - Iowa PD (ds004584, N=149)
     All 6 directed combinations: UCSD->UNM, UCSD->Iowa, UNM->UCSD, UNM->Iowa, Iowa->UCSD, Iowa->UNM.
  2. Cross-Cohort Depression Generalization (Within-Disorder):
     - Mumtaz MDD (Clinical Outpatients, N=58) <-> Cavanagh MDD (University Cohort, N=119)
  3. Full Transdiagnostic Cross-Disorder Transfer Matrix:
     - Sleep Deprivation (ds004902)
     - Depression (Mumtaz MDD)
     - Parkinson's Disease (Iowa PD)
     - Clinical Dementia (ds004504 All)
  4. Multi-Model Transfer Evaluation:
     - GCN-OMST (Proposed topological inductive bias)
     - GCN-Identity (A = I, no message passing)
     - Tuned L2-Logistic Regression
     - Random Forest
     - GBDT (HistGradientBoosting)

Outputs:
  reports/cross_site_pd_transfer_benchmarks.csv
  reports/comprehensive_transdiagnostic_transfer_matrix.csv
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = BENCHMARK_ROOT / "src"
sys.path.append(str(SRC_DIR))
from harmonizer_and_graph import NormativeGCN, compute_fold_normative_z

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


def train_eval_gcn(train_z, train_adj, train_y, val_z, val_adj, epochs=40, lr=0.005, weight_decay=1e-4, seed=42):
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


def evaluate_pair_transfer(src_graphs, tgt_graphs, src_name: str, tgt_name: str, transfer_type: str):
    set_seed(42)
    src_valid = [g for g in src_graphs if int(g['y']) in (0, 1)]
    tgt_valid = [g for g in tgt_graphs if int(g['y']) in (0, 1)]

    train_x = torch.stack([g['x'] for g in src_valid])
    train_adj = torch.stack([g['adj'] for g in src_valid])
    train_y = torch.tensor([g['y'] for g in src_valid], dtype=torch.long)

    test_x = torch.stack([g['x'] for g in tgt_valid])
    test_adj = torch.stack([g['adj'] for g in tgt_valid])
    test_y = np.array([int(g['y']) for g in tgt_valid])

    # Normative z-scoring relative to healthy controls of source cohort
    train_z, test_z, _, _ = compute_fold_normative_z(train_x, train_y, test_x)

    # Adjacency for identity ablation
    adj_identity_tr = torch.eye(19, dtype=torch.float32).unsqueeze(0).expand(len(src_valid), -1, -1)
    adj_identity_te = torch.eye(19, dtype=torch.float32).unsqueeze(0).expand(len(tgt_valid), -1, -1)

    # 1. GCN-OMST
    p_gcn, y_gcn = train_eval_gcn(train_z, train_adj, train_y, test_z, test_adj, epochs=40, seed=42)

    # 2. GCN-Identity (A = I)
    p_id, y_id = train_eval_gcn(train_z, adj_identity_tr, train_y, test_z, adj_identity_te, epochs=40, seed=42)

    # 3. Tabular features for LR, RF, GBDT
    X_tr = train_z.view(len(src_valid), -1).numpy()
    X_te = test_z.view(len(tgt_valid), -1).numpy()
    y_tr = train_y.numpy()

    # Tuned LR
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42).fit(X_tr, y_tr)
    p_lr = lr.predict_proba(X_te)[:, 1]
    y_lr = lr.predict(X_te)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, y_tr)
    p_rf = rf.predict_proba(X_te)[:, 1]
    y_rf = rf.predict(X_te)

    # GBDT
    gbdt = HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=42).fit(X_tr, y_tr)
    p_gbdt = gbdt.predict_proba(X_te)[:, 1]
    y_gbdt = gbdt.predict(X_te)

    model_preds = {
        "GCN_OMST": (p_gcn, y_gcn),
        "GCN_Identity": (p_id, y_id),
        "Tuned_LR": (p_lr, y_lr),
        "RandomForest": (p_rf, y_rf),
        "GBDT": (p_gbdt, y_gbdt)
    }

    results = []
    for m_name, (probs, preds) in model_preds.items():
        auc, ci_lo, ci_hi = bootstrap_auc_ci(test_y, probs)
        acc = accuracy_score(test_y, preds)
        bal_acc = balanced_accuracy_score(test_y, preds)
        results.append({
            'transfer_type': transfer_type,
            'source_cohort': src_name,
            'target_cohort': tgt_name,
            'n_source': len(src_valid),
            'n_target': len(tgt_valid),
            'model': m_name,
            'auc': auc,
            'ci_lower': ci_lo,
            'ci_upper': ci_hi,
            'accuracy': acc,
            'balanced_accuracy': bal_acc
        })
    return results


def main():
    print("=" * 75)
    print("🚀 Running Cross-Site Parkinson's & Transdiagnostic Transfer Benchmark...")
    print("=" * 75)
    t0 = time.time()

    # Load All Derivative Datasets
    print("Loading derivative datasets...")
    sleep_graphs = torch.load(DATA_DIR / "cohort_sleep_ds004902_graphs.pt")
    mumtaz_graphs = torch.load(DATA_DIR / "cohort_mumtaz_mdd_graphs.pt")
    cavanagh_graphs = torch.load(DATA_DIR / "cohort_cavanagh_mdd_graphs.pt")
    ucsd_pd_graphs = torch.load(DATA_DIR / "cohort_ucsd_parkinson_graphs.pt")
    unm_pd_graphs = torch.load(DATA_DIR / "cohort_unm_pd_ds003490_graphs.pt")
    iowa_pd_graphs = torch.load(DATA_DIR / "cohort_iowa_pd_ds004584_graphs.pt")
    dementia_graphs = torch.load(DATA_DIR / "cohort_dementia_ds004504_graphs.pt")

    # Filter subsets
    unm_off_ctl = [g for g in unm_pd_graphs if g['med_state'] in ('OFF', 'CTL')]
    unm_on_ctl = [g for g in unm_pd_graphs if g['med_state'] in ('ON', 'CTL')]

    # -------------------------------------------------------------
    # 1. Tri-Center Parkinson's Cross-Site Generalization (6 Pairs)
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("🔬 [Part 1] Tri-Center Parkinson's Generalization (Within-Disorder Cross-Site):")
    print("=" * 75)

    pd_cohorts = {
        "UCSD_PD": ucsd_pd_graphs,
        "UNM_PD_OFF": unm_off_ctl,
        "Iowa_PD": iowa_pd_graphs
    }
    pd_names = list(pd_cohorts.keys())

    pd_transfer_results = []
    for src in pd_names:
        for tgt in pd_names:
            if src == tgt:
                continue
            res_list = evaluate_pair_transfer(
                pd_cohorts[src], pd_cohorts[tgt], src, tgt, transfer_type="Within_Disorder_PD_Cross_Site"
            )
            pd_transfer_results.extend(res_list)
            # Print GCN vs LR vs RF
            gcn_res = [r for r in res_list if r['model'] == 'GCN_OMST'][0]
            lr_res = [r for r in res_list if r['model'] == 'Tuned_LR'][0]
            rf_res = [r for r in res_list if r['model'] == 'RandomForest'][0]
            print(f"  {src:12s} -> {tgt:12s} | GCN AUC={gcn_res['auc']:.3f} [{gcn_res['ci_lower']:.3f}, {gcn_res['ci_upper']:.3f}] | LR={lr_res['auc']:.3f} | RF={rf_res['auc']:.3f}")

    df_pd_trans = pd.DataFrame(pd_transfer_results)
    out_pd_csv = REPORTS_DIR / "cross_site_pd_transfer_benchmarks.csv"
    df_pd_trans.to_csv(out_pd_csv, index=False)
    print(f"\n✅ Saved Tri-Center Parkinson's transfer report to: {out_pd_csv}")

    # -------------------------------------------------------------
    # 2. Depression Cross-Cohort Generalization (2 Pairs)
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("🔬 [Part 2] Depression Cross-Cohort Generalization (Mumtaz <-> Cavanagh):")
    print("=" * 75)

    mdd_transfer_results = []
    for src, tgt, s_name, t_name in [
        (mumtaz_graphs, cavanagh_graphs, "Mumtaz_MDD", "Cavanagh_MDD"),
        (cavanagh_graphs, mumtaz_graphs, "Cavanagh_MDD", "Mumtaz_MDD")
    ]:
        res_list = evaluate_pair_transfer(src, tgt, s_name, t_name, transfer_type="Within_Disorder_MDD_Cross_Site")
        mdd_transfer_results.extend(res_list)
        gcn_res = [r for r in res_list if r['model'] == 'GCN_OMST'][0]
        lr_res = [r for r in res_list if r['model'] == 'Tuned_LR'][0]
        rf_res = [r for r in res_list if r['model'] == 'RandomForest'][0]
        print(f"  {s_name:12s} -> {t_name:12s} | GCN AUC={gcn_res['auc']:.3f} [{gcn_res['ci_lower']:.3f}, {gcn_res['ci_upper']:.3f}] | LR={lr_res['auc']:.3f} | RF={rf_res['auc']:.3f}")

    # -------------------------------------------------------------
    # 3. Comprehensive Transdiagnostic Cross-Disorder Transfer Matrix
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("🌐 [Part 3] Comprehensive Transdiagnostic Transfer Matrix (Cross-Disorder):")
    print("=" * 75)

    trans_cohorts = {
        "Sleep_Debt": sleep_graphs,
        "Mumtaz_MDD": mumtaz_graphs,
        "Iowa_PD": iowa_pd_graphs,
        "Dementia_All": dementia_graphs
    }
    trans_names = list(trans_cohorts.keys())

    all_trans_results = []
    all_trans_results.extend(pd_transfer_results)
    all_trans_results.extend(mdd_transfer_results)

    for src in trans_names:
        for tgt in trans_names:
            if src == tgt:
                continue
            # Check if already computed
            already = any(r['source_cohort'] == src and r['target_cohort'] == tgt for r in all_trans_results)
            if already:
                continue
            res_list = evaluate_pair_transfer(
                trans_cohorts[src], trans_cohorts[tgt], src, tgt, transfer_type="Cross_Disorder_Transdiagnostic"
            )
            all_trans_results.extend(res_list)
            gcn_res = [r for r in res_list if r['model'] == 'GCN_OMST'][0]
            lr_res = [r for r in res_list if r['model'] == 'Tuned_LR'][0]
            rf_res = [r for r in res_list if r['model'] == 'RandomForest'][0]
            print(f"  {src:12s} -> {tgt:12s} | GCN AUC={gcn_res['auc']:.3f} [{gcn_res['ci_lower']:.3f}, {gcn_res['ci_upper']:.3f}] | LR={lr_res['auc']:.3f} | RF={rf_res['auc']:.3f}")

    df_all_trans = pd.DataFrame(all_trans_results)
    out_all_csv = REPORTS_DIR / "comprehensive_transdiagnostic_transfer_matrix.csv"
    df_all_trans.to_csv(out_all_csv, index=False)
    print(f"\n✅ Saved comprehensive transdiagnostic transfer matrix to: {out_all_csv}")

    total_time = time.time() - t0
    print(f"\n🎉 Finished in {total_time:.1f}s!")


if __name__ == "__main__":
    main()
