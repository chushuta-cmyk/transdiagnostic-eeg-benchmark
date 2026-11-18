#!/usr/bin/env python3
"""
scripts/45c_auxiliary_tables.py
-------------------------------
Recomputes canonical Table S1 OMST empirical channel metrics (degree,
betweenness centrality, clustering across 677 remediated primary graphs)
and canonical Table S24 Panel B heuristic slowing vs. Single_Theta benchmarks
directly from data/derivatives/specparam_canonical_7cohorts.pt.
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = BENCHMARK_ROOT / "src"
sys.path.append(str(SRC_DIR))
from harmonizer_and_graph import CHANNELS_19, compute_fold_normative_z

DATA_DIR = BENCHMARK_ROOT / "data" / "derivatives"
REPORTS_DIR = BENCHMARK_ROOT / "reports"

FOCAL_5 = [CHANNELS_19.index(c) for c in ["Fp1", "Fp2", "F3", "F4", "Fz"]]
REPEATS_10 = [42, 123, 456, 789, 2026, 101, 202, 303, 404, 505]

CH_COORDS = {
    "Fp1": ("Prefrontal (Left)", -0.28, +0.78),
    "Fp2": ("Prefrontal (Right)", +0.28, +0.78),
    "F7": ("Inferior Frontal (Left)", -0.72, +0.44),
    "F3": ("Dorsal Frontal (Left)", -0.35, +0.40),
    "Fz": ("Frontal Midline", 0.00, +0.40),
    "F4": ("Dorsal Frontal (Right)", +0.35, +0.40),
    "F8": ("Inferior Frontal (Right)", +0.72, +0.44),
    "T7": ("Mid-Temporal (Left)", -0.84, 0.00),
    "C3": ("Central (Left)", -0.42, 0.00),
    "Cz": ("Central Vertex", 0.00, 0.00),
    "C4": ("Central (Right)", +0.42, 0.00),
    "T8": ("Mid-Temporal (Right)", +0.84, 0.00),
    "P7": ("Post-Temporal (Left)", -0.72, -0.44),
    "P3": ("Parietal (Left)", -0.35, -0.40),
    "Pz": ("Parietal Midline", 0.00, -0.40),
    "P4": ("Parietal (Right)", +0.35, -0.40),
    "P8": ("Post-Temporal (Right)", +0.72, -0.44),
    "O1": ("Occipital (Left)", -0.28, -0.78),
    "O2": ("Occipital (Right)", +0.28, -0.78),
}

T1_EDGES = [("Fp1", "F3"), ("Fp2", "F4"), ("F7", "F3"), ("F8", "F4"), ("F7", "T7"), ("F8", "T8"), ("F3", "Fz"),
            ("F4", "Fz"), ("Fz", "Cz"), ("C3", "Cz"), ("C4", "Cz"), ("Cz", "Pz"), ("Pz", "P3"), ("Pz", "P4"),
            ("T7", "P7"), ("T8", "P8"), ("P3", "O1"), ("P4", "O2")]
T2_EDGES = [("Fp1", "Fp2"), ("Fp1", "F7"), ("Fp2", "F8"), ("F3", "C3"), ("F4", "C4"), ("T7", "C3"), ("T8", "C4"),
            ("C3", "P3"), ("C4", "P4"), ("P7", "P3"), ("P8", "P4"), ("P7", "O1"), ("P8", "O2"), ("O1", "O2"),
            ("Fz", "Pz"), ("F3", "F4"), ("C3", "C4"), ("Cz", "O1")]


def get_grouped_splits(groups, y, n_splits=5, seed=42):
    groups = np.asarray(groups)
    y = np.asarray(y, dtype=int)
    rng = np.random.RandomState(seed)
    uniq_g, seen = [], set()
    for g in groups:
        if g not in seen:
            seen.add(g)
            uniq_g.append(g)
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
        va_mask = np.array([g in set(fold_groups[k]) for g in groups], dtype=bool)
        splits.append((all_idx[~va_mask], all_idx[va_mask]))
    return splits


def main():
    sp = torch.load(DATA_DIR / "specparam_canonical_7cohorts.pt", weights_only=False)
    primary_keys = ["sleep_ds004902", "mumtaz_ec", "cavanagh_ds003478", "ucsd_ds002778", "dementia_ds004504", "unm_ds003490", "iowa_ds004584"]
    all_primary = [g for k in primary_keys for g in sp[k]]
    assert len(all_primary) == 677

    # 1. Table S1 empirical channel metrics across all 677 canonical graphs
    degs = np.zeros((len(all_primary), 19))
    betw = np.zeros((len(all_primary), 19))
    clus = np.zeros((len(all_primary), 19))
    for i, g in enumerate(all_primary):
        A = (g["adj"].numpy() > 0.5).astype(int)
        np.fill_diagonal(A, 0)
        G = nx.from_numpy_array(A)
        dmap = dict(G.degree())
        bmap = nx.betweenness_centrality(G)
        cmap = nx.clustering(G)
        for j in range(19):
            degs[i, j] = dmap[j]
            betw[i, j] = bmap[j]
            clus[i, j] = cmap[j]

    t1_deg = {c: 0 for c in CHANNELS_19}
    t2_deg = {c: 0 for c in CHANNELS_19}
    for u, v in T1_EDGES:
        t1_deg[u] += 1; t1_deg[v] += 1
    for u, v in T2_EDGES:
        t2_deg[u] += 1; t2_deg[v] += 1

    s1_rows = []
    for j, ch in enumerate(CHANNELS_19):
        reg, px, py = CH_COORDS[ch]
        s1_rows.append({
            "channel": ch, "region": reg, "proj_x": px, "proj_y": py,
            "tree1_deg": t1_deg[ch], "tree2_deg": t2_deg[ch], "combined_deg": t1_deg[ch] + t2_deg[ch],
            "emp_deg_mean": float(degs[:, j].mean()), "emp_deg_sd": float(degs[:, j].std(ddof=1)),
            "betw_mean": float(betw[:, j].mean()), "betw_sd": float(betw[:, j].std(ddof=1)),
            "clus_mean": float(clus[:, j].mean()), "clus_sd": float(clus[:, j].std(ddof=1)),
        })
    pd.DataFrame(s1_rows).to_csv(REPORTS_DIR / "table_s1_omst_channel_metrics.csv", index=False)

    # 2. Heuristic Frontal (Theta - Alpha) vs Parameterized Single_Theta (10x5 repeated grouped CV!)
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
    hrows = []
    for cname, glist in specs_12:
        vg = [g for g in glist if int(g["y"]) in (0, 1)]
        y_all = np.array([int(g["y"]) for g in vg])
        grps = np.array([g.get("bids_id", g.get("sub_name", g["subject_id"])) for g in vg])
        X_all = torch.stack([g["x"] for g in vg])

        aucs_raw, aucs_ori, baccs, aucs_th = [], [], [], []
        for s in REPEATS_10:
            splits = get_grouped_splits(grps, y_all, n_splits=5, seed=s)
            oof_h = np.zeros(len(vg))
            oof_t = np.zeros(len(vg))
            oof_y = np.zeros(len(vg))
            for fold, (tr, va) in enumerate(splits):
                ty = torch.tensor(y_all[tr], dtype=torch.long)
                tz, vz, _, _ = compute_fold_normative_z(X_all[tr], ty, X_all[va])
                # Heuristic (frontal theta - frontal alpha) in z-space
                h_tr = (tz[:, FOCAL_5, 0].mean(dim=-1) - tz[:, FOCAL_5, 2].mean(dim=-1)).numpy()
                h_va = (vz[:, FOCAL_5, 0].mean(dim=-1) - vz[:, FOCAL_5, 2].mean(dim=-1)).numpy()
                oof_h[va] = h_va
                lr_h = LogisticRegression(random_state=s).fit(h_tr.reshape(-1, 1), y_all[tr])
                oof_y[va] = lr_h.predict(h_va.reshape(-1, 1))
                # Parameterized Single_Theta
                t_tr = tz[:, FOCAL_5, 0].mean(dim=-1, keepdim=True).numpy()
                t_va = vz[:, FOCAL_5, 0].mean(dim=-1, keepdim=True).numpy()
                oof_t[va] = LogisticRegression(random_state=s).fit(t_tr, y_all[tr]).predict_proba(t_va)[:, 1]
            ar = float(roc_auc_score(y_all, oof_h))
            aucs_raw.append(ar)
            aucs_ori.append(max(ar, 1.0 - ar))
            baccs.append(float(balanced_accuracy_score(y_all, oof_y)))
            at = float(roc_auc_score(y_all, oof_t))
            aucs_th.append(max(at, 1.0 - at))
        hrows.append({
            "cohort": cname, "n_samples": len(vg), "heuristic_rule": "Frontal_Theta_Alpha_Difference",
            "auc_raw": float(np.mean(aucs_raw)), "auc_oriented": float(np.mean(aucs_ori)),
            "balanced_accuracy": float(np.mean(baccs)), "single_theta_auc_oriented": float(np.mean(aucs_th)),
        })
    df_h = pd.DataFrame(hrows)
    df_h.to_csv(REPORTS_DIR / "benchmark_heuristic_slowing.csv", index=False)
    print("✓ Updated table_s1_omst_channel_metrics.csv and benchmark_heuristic_slowing.csv cleanly.")


if __name__ == "__main__":
    main()
