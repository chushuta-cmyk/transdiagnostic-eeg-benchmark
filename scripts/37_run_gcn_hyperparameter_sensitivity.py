#!/usr/bin/env python3
"""
Step 37: GCN Hyperparameter Sensitivity Analysis across Representative Cohorts.
Tests whether NormativeGCN's lack of diagnostic superiority over classical baselines
is an artifact of fixed hyperparameters (lr=0.005, epochs=50, hidden_dim=32)
or represents a robust property across learning rates, training epochs, and network capacities.
"""

import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

import sys
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(SRC_DIR))
from harmonizer_and_graph import NormativeGCN, compute_fold_normative_z

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = BENCHMARK_ROOT / "data" / "derivatives"
REPORTS_DIR = BENCHMARK_ROOT / "reports"


def run_gcn_grid_sensitivity():
    """Evaluate NormativeGCN across learning rates, training epochs, and hidden dimensions."""
    print("🚀 Running NormativeGCN Hyperparameter Sensitivity Grid...")
    
    # Load representative cohorts
    mumtaz = torch.load(DATA_DIR / "cohort_mumtaz_mdd_graphs.pt", weights_only=False)
    cav = torch.load(DATA_DIR / "cohort_cavanagh_mdd_graphs.pt", weights_only=False)
    ucsd = torch.load(DATA_DIR / "cohort_ucsd_parkinson_graphs.pt", weights_only=False)
    unm = torch.load(DATA_DIR / "cohort_unm_pd_ds003490_graphs.pt", weights_only=False)
    unm_off = [g for g in unm if g["med_state"] in ("OFF", "CTL")]
    iowa = torch.load(DATA_DIR / "cohort_iowa_pd_ds004584_graphs.pt", weights_only=False)
    
    cohort_list = [
        ("Study 2: Mumtaz Outpatient MDD (EC)", mumtaz, False),
        ("Study 3: Cavanagh Univ. Depression (EO)", cav, False),
        ("Study 4: UCSD Parkinson All", ucsd, True),
        ("Study 6a: UNM PD-OFF vs CTL", unm_off, False),
        ("Study 7: Iowa Parkinson", iowa, False),
    ]
    
    grid = [
        # (lr, epochs, hidden_dim, label)
        (0.005, 50, 32, "Baseline (lr=0.005, ep=50, h=32)"),
        (0.001, 50, 32, "Lower LR (lr=0.001, ep=50, h=32)"),
        (0.010, 50, 32, "Higher LR (lr=0.010, ep=50, h=32)"),
        (0.005, 30, 32, "Fewer Epochs (lr=0.005, ep=30, h=32)"),
        (0.005, 100, 32, "More Epochs (lr=0.005, ep=100, h=32)"),
        (0.005, 50, 16, "Smaller Capacity (lr=0.005, ep=50, h=16)"),
        (0.005, 50, 64, "Larger Capacity (lr=0.005, ep=50, h=64)"),
    ]
    
    records = []
    
    for c_name, graphs, is_grouped in cohort_list:
        graphs = [g for g in graphs if int(g["y"]) in (0, 1)]
        y = np.array([int(g["y"]) for g in graphs])
        subjs = np.array([g.get("bids_id", g.get("sub_name", g.get("subject_id", str(i)))) for i, g in enumerate(graphs)])

        # Identical splitter to the primary benchmark (scripts/17): shuffled, random_state = 42
        if is_grouped:
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(cv.split(graphs, y, groups=subjs if is_grouped else None))
        
        for lr, epochs, h_dim, config_label in grid:
            oof_probs = np.zeros(len(graphs))
            
            for fold, (tr_idx, val_idx) in enumerate(splits):
                tx = torch.stack([graphs[i]["x"] for i in tr_idx])
                vx = torch.stack([graphs[i]["x"] for i in val_idx])
                ty = torch.tensor([int(graphs[i]["y"]) for i in tr_idx], dtype=torch.long)
                tadj = torch.stack([graphs[i]["adj"] for i in tr_idx])
                vadj = torch.stack([graphs[i]["adj"] for i in val_idx])
                
                tz, vz, _, _ = compute_fold_normative_z(tx, ty, vx)
                
                # Identical seeding to the primary benchmark: seed = 42 + fold
                random.seed(42 + fold)
                np.random.seed(42 + fold)
                torch.manual_seed(42 + fold)
                model = NormativeGCN(in_node_features=4, hidden_dim=h_dim, num_classes=2, dropout=0.25)
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
                criterion = nn.CrossEntropyLoss()
                
                model.train()
                for _ in range(epochs):
                    optimizer.zero_grad()
                    out = model(tz, tadj)
                    loss = criterion(out, ty)
                    loss.backward()
                    optimizer.step()
                    
                model.eval()
                with torch.no_grad():
                    logits = model(vz, vadj)
                    oof_probs[val_idx] = torch.softmax(logits, dim=-1)[:, 1].numpy()
                    
            auc = roc_auc_score(y, oof_probs)
            records.append({
                "cohort": c_name,
                "n_samples": len(graphs),
                "config_label": config_label,
                "learning_rate": lr,
                "epochs": epochs,
                "hidden_dim": h_dim,
                "auc": auc
            })
            print(f"  {c_name[:25]:25s} | {config_label:38s} -> AUC = {auc:.4f}")

    df = pd.DataFrame(records)
    out_csv = REPORTS_DIR / "benchmark_gcn_hyperparameter_sensitivity.csv"
    df.to_csv(out_csv, index=False)
    print(f"✓ Saved GCN hyperparameter sensitivity benchmark to {out_csv}")
    
    # Print summary table
    pivot = df.pivot(index="config_label", columns="cohort", values="auc")
    print("\nSummary Pivot Table:")
    print(pivot.round(3).to_string())
    return df


if __name__ == "__main__":
    run_gcn_grid_sensitivity()
