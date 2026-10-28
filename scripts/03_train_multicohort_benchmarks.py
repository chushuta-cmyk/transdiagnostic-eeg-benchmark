#!/usr/bin/env python3
"""
Step 03: Multi-Cohort NormativeGCN Benchmarking & Cross-Cohort Transfer.
Evaluates Fold-Nested Normative Graph Convolutional Networks (NormativeGCN)
under strict leak-free standards:
  1. Within-Cohort Evaluation (Stratified 5-Fold / LOSO CV):
     - Mumtaz Clinical MDD (N=58)
     - Cavanagh Depression (N=119 evaluable)
     - UCSD Parkinson's (N=31 subjects / 46 sessions)
     - Sleep Deprivation (N=71 pairs / 142 sessions)
  2. Cross-Cohort Transfer Evaluation:
     - Can a normative model trained on Acute Sleep Deprivation transfer to Clinical MDD?
     - Can a model trained on Mumtaz MDD transfer to Cavanagh Depression?
  3. Outputs clean performance metrics:
     - Balanced Accuracy, ROC-AUC with 1,000-bootstrap 95% CI.

Saves:
  reports/multicohort_classification_benchmark.csv
"""

import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = BENCHMARK_ROOT / "src"
sys.path.append(str(SRC_DIR))
from harmonizer_and_graph import NormativeGCN, compute_fold_normative_z, CHANNELS_19

DATA_DIR = BENCHMARK_ROOT / "data" / "derivatives"
REPORTS_DIR = BENCHMARK_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def bootstrap_auc_ci(y_true, y_prob, n_boot: int = 1000):
    """Computes non-parametric 95% bootstrap confidence interval for ROC-AUC."""
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


def run_within_cohort_benchmark(graph_list, cohort_name: str, n_splits: int = 5):
    """Executes leak-free Stratified K-Fold CV with fold-nested normative calibration."""
    set_seed(42)
    device = get_device()

    # Filter binary classification (0 vs 1)
    valid_graphs = [g for g in graph_list if int(g['y']) in (0, 1)]
    if len(valid_graphs) < 10:
        print(f"⚠️ Cohort {cohort_name} has too few samples ({len(valid_graphs)}) for CV.")
        return None

    y_all = np.array([int(g['y']) for g in valid_graphs])
    n_samples = len(valid_graphs)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(n_samples)
    oof_probs = np.zeros(n_samples)

    for fold, (train_idx, val_idx) in enumerate(skf.split(valid_graphs, y_all)):
        # Extract fold tensors
        train_x = torch.stack([valid_graphs[i]['x'] for i in train_idx])      # [N_train, 19, 4]
        train_adj = torch.stack([valid_graphs[i]['adj'] for i in train_idx])  # [N_train, 19, 19]
        train_y = torch.tensor([valid_graphs[i]['y'] for i in train_idx], dtype=torch.long)

        val_x = torch.stack([valid_graphs[i]['x'] for i in val_idx])
        val_adj = torch.stack([valid_graphs[i]['adj'] for i in val_idx])
        val_y = torch.tensor([valid_graphs[i]['y'] for i in val_idx], dtype=torch.long)

        # Strict Fold-Nested Normative Z-score Calibration (fit strictly on y=0 controls)
        train_z, val_z, _, _ = compute_fold_normative_z(train_x, train_y, val_x)

        # Move to compute device
        train_z, train_adj, train_y = train_z.to(device), train_adj.to(device), train_y.to(device)
        val_z, val_adj = val_z.to(device), val_adj.to(device)

        model = NormativeGCN(in_node_features=4, hidden_dim=32, num_classes=2, dropout=0.25).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(35):
            optimizer.zero_grad()
            out = model(train_z, train_adj)
            loss = criterion(out, train_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(val_z, val_adj)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

        oof_probs[val_idx] = probs
        oof_preds[val_idx] = preds

    acc = accuracy_score(y_all, oof_preds)
    b_acc = balanced_accuracy_score(y_all, oof_preds)
    auc_val, ci_lo, ci_hi = bootstrap_auc_ci(y_all, oof_probs)

    print(f"  ✓ {cohort_name:40s} (N={n_samples}): Bal Acc = {b_acc * 100:.1f}%, AUC = {auc_val:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

    return {
        'cohort': cohort_name,
        'evaluation_type': f'Within-Cohort {n_splits}-Fold CV',
        'n_samples': n_samples,
        'n_cases': int(np.sum(y_all == 1)),
        'n_controls': int(np.sum(y_all == 0)),
        'accuracy': acc,
        'balanced_accuracy': b_acc,
        'roc_auc': auc_val,
        'auc_ci_lower': ci_lo,
        'auc_ci_upper': ci_hi
    }


def run_cross_cohort_transfer(train_graphs, test_graphs, train_name: str, test_name: str):
    """Trains on one cohort and evaluates zero-shot transfer on another cohort."""
    set_seed(42)
    device = get_device()

    train_valid = [g for g in train_graphs if int(g['y']) in (0, 1)]
    test_valid = [g for g in test_graphs if int(g['y']) in (0, 1)]

    train_x = torch.stack([g['x'] for g in train_valid])
    train_adj = torch.stack([g['adj'] for g in train_valid])
    train_y = torch.tensor([g['y'] for g in train_valid], dtype=torch.long)

    test_x = torch.stack([g['x'] for g in test_valid])
    test_adj = torch.stack([g['adj'] for g in test_valid])
    test_y = np.array([int(g['y']) for g in test_valid])

    # Calibrate test features using train reference controls
    train_z, test_z, _, _ = compute_fold_normative_z(train_x, train_y, test_x)

    train_z, train_adj, train_y = train_z.to(device), train_adj.to(device), train_y.to(device)
    test_z, test_adj = test_z.to(device), test_adj.to(device)

    model = NormativeGCN(in_node_features=4, hidden_dim=32, num_classes=2, dropout=0.25).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(40):
        optimizer.zero_grad()
        out = model(train_z, train_adj)
        loss = criterion(out, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(test_z, test_adj)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

    acc = accuracy_score(test_y, preds)
    b_acc = balanced_accuracy_score(test_y, preds)
    auc_val, ci_lo, ci_hi = bootstrap_auc_ci(test_y, probs)

    print(f"  ⚡ Transfer [{train_name} -> {test_name}] (N={len(test_valid)}): Bal Acc = {b_acc * 100:.1f}%, AUC = {auc_val:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

    return {
        'cohort': f"{train_name} -> {test_name}",
        'evaluation_type': 'Zero-Shot Cross-Cohort Transfer',
        'n_samples': len(test_valid),
        'n_cases': int(np.sum(test_y == 1)),
        'n_controls': int(np.sum(test_y == 0)),
        'accuracy': acc,
        'balanced_accuracy': b_acc,
        'roc_auc': auc_val,
        'auc_ci_lower': ci_lo,
        'auc_ci_upper': ci_hi
    }


def main():
    print("🚀 Running Multi-Cohort NormativeGCN Benchmark (Within-Cohort & Cross-Cohort Transfer)...")

    cohort_files = {
        'Sleep_ds004902': DATA_DIR / "cohort_sleep_ds004902_graphs.pt",
        'Mumtaz_MDD_4244171': DATA_DIR / "cohort_mumtaz_mdd_graphs.pt",
        'Cavanagh_ds003478': DATA_DIR / "cohort_cavanagh_mdd_graphs.pt",
        'UCSD_Parkinson_ds002778': DATA_DIR / "cohort_ucsd_parkinson_graphs.pt",
    }

    loaded_cohorts = {}
    for name, fpath in cohort_files.items():
        if fpath.exists():
            loaded_cohorts[name] = torch.load(fpath, weights_only=False)
            print(f"Loaded {name}: {len(loaded_cohorts[name])} graph instances.")
        else:
            print(f"⚠️ Missing {name}: {fpath.name} not found.")

    results = []

    # 1. Within-Cohort Evaluations
    print("\n" + "=" * 70)
    print("📌 [Part 1] Within-Cohort NormativeGCN Benchmarks (5-Fold Fold-Nested CV):")
    for name, graphs in loaded_cohorts.items():
        res = run_within_cohort_benchmark(graphs, name, n_splits=5)
        if res:
            results.append(res)

    # 2. Cross-Cohort Transfer Evaluations
    print("\n" + "=" * 70)
    print("📌 [Part 2] Zero-Shot Cross-Cohort Model Transfer:")
    # Transfer 1: Mumtaz MDD -> Cavanagh MDD (Depression to Depression cross-center validation!)
    if 'Mumtaz_MDD_4244171' in loaded_cohorts and 'Cavanagh_ds003478' in loaded_cohorts:
        res = run_cross_cohort_transfer(loaded_cohorts['Mumtaz_MDD_4244171'], loaded_cohorts['Cavanagh_ds003478'],
                                        'Mumtaz_MDD', 'Cavanagh_MDD')
        results.append(res)

    # Transfer 2: Cavanagh MDD -> Mumtaz MDD
    if 'Cavanagh_ds003478' in loaded_cohorts and 'Mumtaz_MDD_4244171' in loaded_cohorts:
        res = run_cross_cohort_transfer(loaded_cohorts['Cavanagh_ds003478'], loaded_cohorts['Mumtaz_MDD_4244171'],
                                        'Cavanagh_MDD', 'Mumtaz_MDD')
        results.append(res)

    # Transfer 3: Sleep Deprivation -> Mumtaz MDD (Testing Shared vs Distinct Pathophysiology)
    if 'Sleep_ds004902' in loaded_cohorts and 'Mumtaz_MDD_4244171' in loaded_cohorts:
        res = run_cross_cohort_transfer(loaded_cohorts['Sleep_ds004902'], loaded_cohorts['Mumtaz_MDD_4244171'],
                                        'Sleep_Deprivation', 'Mumtaz_MDD')
        results.append(res)

    # Transfer 4: Sleep Deprivation -> UCSD Parkinson
    if 'Sleep_ds004902' in loaded_cohorts and 'UCSD_Parkinson_ds002778' in loaded_cohorts:
        res = run_cross_cohort_transfer(loaded_cohorts['Sleep_ds004902'], loaded_cohorts['UCSD_Parkinson_ds002778'],
                                        'Sleep_Deprivation', 'UCSD_Parkinson')
        results.append(res)

    df_res = pd.DataFrame(results)
    out_csv = REPORTS_DIR / "multicohort_classification_benchmark.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\n✓ Saved multi-cohort benchmark results to {out_csv}")


if __name__ == "__main__":
    main()
