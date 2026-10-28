#!/usr/bin/env python3
"""
Step 03: Deterministic Local Training & Statistical Evaluation of NormativeGCN (Study 1).
Performs 20-Fold Leave-One-Subject-Out (LOSO) cross-validation on REAL ds004902 EEG graphs:
  - Strict Fold-Nested Normative Calibration: reference mean/std estimated EXCLUSIVELY
    from Normal Sleep (y=0) training subjects inside each fold (Rutherford et al., 2022).
  - Deterministic seeding (seed=42) for 100% reproducibility across runs.
  - 1,000-iteration subject-level bootstrap 95% Confidence Intervals and permutation p-value.
"""

import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = BENCHMARK_ROOT / "src"
sys.path.append(str(SRC_DIR))
from harmonizer_and_graph import NormativeGCN, compute_fold_normative_z

DATASET_PATH = BENCHMARK_ROOT / "data" / "derivatives" / "pilot_graph_dataset.pt"
MODEL_SAVE_PATH = BENCHMARK_ROOT / "data" / "derivatives" / "pilot_gnn_model.pth"


def set_deterministic_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_single_fold(train_data, test_data, device, fold_seed=42, epochs=35, lr=0.008):
    set_deterministic_seed(fold_seed)
    model = NormativeGCN(in_node_features=4, hidden_dim=32, num_classes=2, dropout=0.25).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    x_train = torch.stack([d['x'] for d in train_data]).to(device)
    adj_train = torch.stack([d['adj'] for d in train_data]).to(device)
    y_train = torch.stack([d['y'] for d in train_data]).to(device)

    x_test = torch.stack([d['x'] for d in test_data]).to(device)
    adj_test = torch.stack([d['adj'] for d in test_data]).to(device)
    y_test = torch.stack([d['y'] for d in test_data]).to(device)

    # True Fold-Nested Normative Calibration (exclusively using y==0 reference baseline)
    z_train, z_test, _, _ = compute_fold_normative_z(x_train, y_train, x_test)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(z_train, adj_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(z_test, adj_test)
        probs = torch.softmax(test_logits, dim=-1)[:, 1].cpu().numpy()
        preds = torch.argmax(test_logits, dim=-1).cpu().numpy()
        targets = y_test.cpu().numpy()

    return probs, preds, targets, model


def compute_bootstrap_ci_and_perm_p(targets, probs, preds, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    probs = np.asarray(probs)
    preds = np.asarray(preds)
    n = len(targets)

    obs_acc = accuracy_score(targets, preds)
    obs_auc = roc_auc_score(targets, probs)

    boot_accs, boot_aucs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(targets[idx])) < 2:
            continue
        boot_accs.append(accuracy_score(targets[idx], preds[idx]))
        boot_aucs.append(roc_auc_score(targets[idx], probs[idx]))

    acc_ci = (np.percentile(boot_accs, 2.5), np.percentile(boot_accs, 97.5))
    auc_ci = (np.percentile(boot_aucs, 2.5), np.percentile(boot_aucs, 97.5))

    # Label permutation test for ROC-AUC
    null_aucs = []
    for _ in range(n_boot):
        perm_y = rng.permutation(targets)
        null_aucs.append(roc_auc_score(perm_y, probs))
    p_perm = (np.sum(np.asarray(null_aucs) >= obs_auc) + 1.0) / (n_boot + 1.0)

    return obs_acc, acc_ci, obs_auc, auc_ci, p_perm


def main():
    set_deterministic_seed(42)
    device = get_device()
    print(f"🚀 Running Deterministic NormativeGCN Pilot on Device: {device.type.upper()}")
    dataset = torch.load(DATASET_PATH, weights_only=False)

    subjects = sorted(list(set(d['subject_id'] for d in dataset)))
    print(f"Loaded {len(dataset)} REAL EEG OMST graphs across {len(subjects)} paired subjects.")

    all_targets, all_probs, all_preds = [], [], []
    best_model = None

    for idx, test_sub in enumerate(subjects):
        train_data = [d for d in dataset if d['subject_id'] != test_sub]
        test_data = [d for d in dataset if d['subject_id'] == test_sub]

        probs, preds, targets, model = train_single_fold(
            train_data, test_data, device, fold_seed=42 + idx, epochs=35, lr=0.008
        )
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_targets.extend(targets)
        if idx == 0:
            best_model = model

    acc, acc_ci, auc_val, auc_ci, p_perm = compute_bootstrap_ci_and_perm_p(
        all_targets, all_probs, all_preds, n_boot=1000, seed=42
    )

    print("\n" + "=" * 68)
    print("📊 REAL-DATA STUDY 1 LOSO BENCHMARK RESULTS (N=40 Sessions, 20 Pairs)")
    print("=" * 68)
    print(f"Out-of-Fold Accuracy : {acc*100:.1f}%  (95% Bootstrap CI: [{acc_ci[0]*100:.1f}%, {acc_ci[1]*100:.1f}%])")
    print(f"Out-of-Fold ROC-AUC  : {auc_val:.3f}   (95% Bootstrap CI: [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}])")
    print(f"Permutation p-value  : p = {p_perm:.4f} (1,000 label permutations)")
    print("=" * 68)

    if best_model is not None:
        torch.save(best_model.state_dict(), MODEL_SAVE_PATH)
        print(f"✓ Saved deterministic model weights to: {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()
