#!/usr/bin/env python3
"""
Build and Evaluate Mumtaz Eyes-Open (EO) Dataset.
Directly tests whether matching recording state (Eyes-Open vs Eyes-Open) resolves
cross-cohort transfer differences between Mumtaz MDD and Cavanagh Depression.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import mne

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = BENCHMARK_ROOT / "src"
sys.path.append(str(SRC_DIR))

from harmonizer_and_graph import (
    TopologicalGraphBuilder,
    extract_real_19ch_features_and_pli,
    CHANNELS_19
)
import multi_cohort_loader as mcl
import importlib
sys.path.append(str(BENCHMARK_ROOT / "scripts"))
bench_module = importlib.import_module("08_multicohort_extended_model_benchmark")
evaluate_cohort_all_models = bench_module.evaluate_cohort_all_models
run_directed_transfer = bench_module.run_directed_transfer

OUTPUT_DIR = BENCHMARK_ROOT / "data" / "derivatives"
REPORTS_DIR = BENCHMARK_ROOT / "reports"
MUMTAZ_DIR = Path("/Users/tan/Downloads/4244171")


def build_mumtaz_eo():
    adapter = mcl.Universal19ChannelAdapter()
    graph_builder = TopologicalGraphBuilder()

    print("\n" + "=" * 70)
    print("🧠 Processing Mumtaz Eyes-Open (EO) Dataset...")
    manifest = mcl.get_mumtaz_mdd_manifest(MUMTAZ_DIR, state='EO')
    print(f"Loaded manifest: {len(manifest)} EO recordings ({sum(1 for x in manifest if x['label']==0)} HC, {sum(1 for x in manifest if x['label']==1)} MDD)")

    dataset = []
    t0 = time.time()
    for idx, item in enumerate(manifest):
        try:
            raw = mne.io.read_raw_edf(str(item['path']), preload=True, verbose=False)
            data, sfreq = adapter.standardize_raw_to_epochs(raw)
            x_mat, pli_matrix = extract_real_19ch_features_and_pli(data, sfreq)
            adj_bin, adj_weighted, opt_m, max_gce = graph_builder.build_omst_graph(pli_matrix)

            dataset.append({
                'x': torch.tensor(x_mat, dtype=torch.float32),
                'adj': torch.tensor(adj_bin, dtype=torch.float32),
                'adj_weighted': torch.tensor(adj_weighted, dtype=torch.float32),
                'pli': torch.tensor(pli_matrix, dtype=torch.float32),
                'y': torch.tensor(item['label'], dtype=torch.long),
                'subject_id': item['subject_id'],
                'group': item['group'],
                'cohort': 'Mumtaz_MDD_EO',
                'domain': item['diagnosis_domain'],
                'n_clean_epochs': data.shape[0],
                'omst_trees': opt_m,
                'gce': max_gce
            })

            if (idx + 1) % 10 == 0 or (idx + 1) == len(manifest):
                print(f"  [{idx + 1}/{len(manifest)}] {item['subject_id']} ({item['group']}): {data.shape[0]} clean epochs, OMST m*={opt_m}")
        except Exception as e:
            print(f"  ⚠️ Error processing {item['subject_id']} ({item['path'].name}): {e}")

    out_path = OUTPUT_DIR / "cohort_mumtaz_eo_graphs.pt"
    torch.save(dataset, out_path)
    print(f"✓ Saved {len(dataset)} Mumtaz EO graphs to {out_path} ({time.time() - t0:.1f}s)")
    return dataset


def main():
    eo_path = OUTPUT_DIR / "cohort_mumtaz_eo_graphs.pt"
    if eo_path.exists():
        print(f"Loading existing Mumtaz EO graphs from {eo_path}...")
        eo_graphs = torch.load(eo_path)
    else:
        eo_graphs = build_mumtaz_eo()

    # Load comparison cohorts
    ec_graphs = torch.load(OUTPUT_DIR / "cohort_mumtaz_mdd_graphs.pt")
    cavanagh_graphs = torch.load(OUTPUT_DIR / "cohort_cavanagh_mdd_graphs.pt")
    sleep_graphs = torch.load(OUTPUT_DIR / "cohort_sleep_ds004902_graphs.pt")

    print("\n" + "=" * 70)
    print("📊 Benchmarking all models on: Mumtaz Eyes-Open (EO)")
    eo_benchmarks = evaluate_cohort_all_models(eo_graphs, "Study 2b: Mumtaz Outpatient MDD (Eyes-Open)", use_group_kfold=False)

    df_eo = pd.DataFrame(eo_benchmarks)
    df_eo.to_csv(REPORTS_DIR / "mumtaz_eo_model_comparison.csv", index=False)
    print(f"✓ Saved Mumtaz EO benchmark results to {REPORTS_DIR / 'mumtaz_eo_model_comparison.csv'}")

    print("\n" + "=" * 70)
    print("🌐 Comparing State-Matched vs State-Mismatched Transfer:")
    transfers = [
        ("Mumtaz_EC (Eyes-Closed)", "Cavanagh_EO (Eyes-Open)", ec_graphs, cavanagh_graphs),
        ("Mumtaz_EO (Eyes-Open)", "Cavanagh_EO (Eyes-Open)", eo_graphs, cavanagh_graphs),
        ("Cavanagh_EO (Eyes-Open)", "Mumtaz_EC (Eyes-Closed)", cavanagh_graphs, ec_graphs),
        ("Cavanagh_EO (Eyes-Open)", "Mumtaz_EO (Eyes-Open)", cavanagh_graphs, eo_graphs),
        ("Mumtaz_EO (Eyes-Open)", "Mumtaz_EC (Eyes-Closed)", eo_graphs, ec_graphs),
        ("Mumtaz_EC (Eyes-Closed)", "Mumtaz_EO (Eyes-Open)", ec_graphs, eo_graphs),
    ]

    results = []
    for src_name, tgt_name, src_g, tgt_g in transfers:
        res = run_directed_transfer(src_g, tgt_g, src_name, tgt_name)
        results.append(res)
        print(f"  Transfer: {src_name} -> {tgt_name}")
        print(f"    GCN AUC = {res['gcn_auc']:.3f} [{res['gcn_ci_lower']:.3f}, {res['gcn_ci_upper']:.3f}] | BalAcc = {res['gcn_bal_acc']*100:.1f}%")
        print(f"    LR  AUC = {res['lr_auc']:.3f} [{res['lr_ci_lower']:.3f}, {res['lr_ci_upper']:.3f}] | BalAcc = {res['lr_bal_acc']*100:.1f}%")

    df_trans = pd.DataFrame(results)
    out_trans = REPORTS_DIR / "mumtaz_eo_state_matched_transfer.csv"
    df_trans.to_csv(out_trans, index=False)
    print(f"✓ Saved state-matched transfer results to {out_trans}")


if __name__ == "__main__":
    main()
