#!/usr/bin/env python3
"""
Step 01: Multi-Cohort Universal 19-Channel OMST Graph Builder.
Extracts real spectral features, aperiodic exponents, and data-driven OMST brain graphs
across all 4 real electrophysiological cohorts:
  1. Mumtaz Clinical MDD (Figshare 4244171, N=58 subjects: 28 HC, 30 MDD)
  2. Cavanagh Depression Rest (OpenNeuro ds003478, N=121 subjects: 75 Control, 44 Depressed, 2 Subclinical)
  3. UCSD Parkinson's Disease (OpenNeuro ds002778, N=46 sessions: 16 HC, 15 PD-OFF, 15 PD-ON)
  4. Sleep Deprivation (OpenNeuro ds004902, N=142 sessions: 71 Normal Sleep, 71 Sleep Deprived)

Saves clean PyTorch graph datasets in data/derivatives/:
  - cohort_mumtaz_mdd_graphs.pt
  - cohort_cavanagh_mdd_graphs.pt
  - cohort_ucsd_parkinson_graphs.pt
  - cohort_sleep_ds004902_graphs.pt
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
import mne

# Add src directory to path
BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = BENCHMARK_ROOT / "src"
sys.path.append(str(SRC_DIR))

from harmonizer_and_graph import (
    TopologicalGraphBuilder,
    extract_real_19ch_features_and_pli,
    CHANNELS_19
)
import multi_cohort_loader as mcl

OUTPUT_DIR = BENCHMARK_ROOT / "data" / "derivatives"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset paths
MUMTAZ_DIR = Path("/Users/tan/Downloads/4244171")
CAVANAGH_DIR = Path("/Users/tan/Documents/eeg_datastes/depression_new")
PARKINSON_DIR = Path("/Users/tan/Documents/eeg_datastes")
SLEEP_FIF_DIR = Path("/Users/tan/Documents/qwen-agent/aCAgqOVe0x/default/sleep_deprivation_eeg/data/derivatives/preprocessed_v3_conservative")


def build_cohort_mumtaz(adapter: mcl.Universal19ChannelAdapter, graph_builder: TopologicalGraphBuilder):
    print("\n" + "=" * 70)
    print("🧠 [Cohort 1/4] Processing Mumtaz Clinical MDD Dataset (Figshare 4244171)...")
    manifest = mcl.get_mumtaz_mdd_manifest(MUMTAZ_DIR, state='EC')
    print(f"Loaded manifest: {len(manifest)} Eyes-Closed recordings ({sum(1 for x in manifest if x['label']==0)} HC, {sum(1 for x in manifest if x['label']==1)} MDD)")

    dataset = []
    omst_counts = []
    gce_scores = []
    t0 = time.time()

    for idx, item in enumerate(manifest):
        try:
            raw = mne.io.read_raw_edf(str(item['path']), preload=True, verbose=False)
            data, sfreq = adapter.standardize_raw_to_epochs(raw)
            x_mat, pli_matrix = extract_real_19ch_features_and_pli(data, sfreq)
            adj_bin, adj_weighted, opt_m, max_gce = graph_builder.build_omst_graph(pli_matrix)

            omst_counts.append(opt_m)
            gce_scores.append(max_gce)

            dataset.append({
                'x': torch.tensor(x_mat, dtype=torch.float32),
                'adj': torch.tensor(adj_bin, dtype=torch.float32),
                'adj_weighted': torch.tensor(adj_weighted, dtype=torch.float32),
                'pli': torch.tensor(pli_matrix, dtype=torch.float32),
                'y': torch.tensor(item['label'], dtype=torch.long),
                'subject_id': item['subject_id'],
                'group': item['group'],
                'cohort': item['cohort'],
                'domain': item['diagnosis_domain'],
                'n_clean_epochs': data.shape[0],
                'omst_trees': opt_m,
                'gce': max_gce
            })

            if (idx + 1) % 10 == 0 or (idx + 1) == len(manifest):
                print(f"  [{idx + 1}/{len(manifest)}] {item['subject_id']} ({item['group']}): {data.shape[0]} clean epochs, OMST m*={opt_m}, GCE={max_gce:.3f}")
        except Exception as e:
            print(f"  ⚠️ Error processing {item['subject_id']} ({item['path'].name}): {e}")

    out_path = OUTPUT_DIR / "cohort_mumtaz_mdd_graphs.pt"
    torch.save(dataset, out_path)
    print(f"✓ Saved {len(dataset)} graphs to {out_path} ({time.time() - t0:.1f}s)")
    return dataset


def build_cohort_cavanagh(adapter: mcl.Universal19ChannelAdapter, graph_builder: TopologicalGraphBuilder):
    print("\n" + "=" * 70)
    print("🧠 [Cohort 2/4] Processing Cavanagh Depression Rest Dataset (OpenNeuro ds003478)...")
    manifest = mcl.get_cavanagh_depression_manifest(CAVANAGH_DIR)
    # Filter out subclinical (keep Control label 0 and Depressed label 1)
    eval_manifest = [x for x in manifest if x['label'] in (0, 1)]
    print(f"Loaded manifest: {len(eval_manifest)} evaluable subjects ({sum(1 for x in eval_manifest if x['label']==0)} Control, {sum(1 for x in eval_manifest if x['label']==1)} Depressed)")

    dataset = []
    omst_counts = []
    gce_scores = []
    t0 = time.time()

    for idx, item in enumerate(eval_manifest):
        try:
            raw = mne.io.read_raw_eeglab(str(item['path']), preload=True, verbose=False)
            data, sfreq = adapter.standardize_raw_to_epochs(raw)
            x_mat, pli_matrix = extract_real_19ch_features_and_pli(data, sfreq)
            adj_bin, adj_weighted, opt_m, max_gce = graph_builder.build_omst_graph(pli_matrix)

            omst_counts.append(opt_m)
            gce_scores.append(max_gce)

            dataset.append({
                'x': torch.tensor(x_mat, dtype=torch.float32),
                'adj': torch.tensor(adj_bin, dtype=torch.float32),
                'adj_weighted': torch.tensor(adj_weighted, dtype=torch.float32),
                'pli': torch.tensor(pli_matrix, dtype=torch.float32),
                'y': torch.tensor(item['label'], dtype=torch.long),
                'subject_id': item['subject_id'],
                'bdi': item['bdi'],
                'group': item['group'],
                'cohort': item['cohort'],
                'domain': item['diagnosis_domain'],
                'n_clean_epochs': data.shape[0],
                'omst_trees': opt_m,
                'gce': max_gce
            })

            if (idx + 1) % 20 == 0 or (idx + 1) == len(eval_manifest):
                print(f"  [{idx + 1}/{len(eval_manifest)}] {item['subject_id']} ({item['group']}, BDI={item['bdi']}): {data.shape[0]} epochs, OMST m*={opt_m}, GCE={max_gce:.3f}")
        except Exception as e:
            print(f"  ⚠️ Error processing {item['subject_id']}: {e}")

    out_path = OUTPUT_DIR / "cohort_cavanagh_mdd_graphs.pt"
    torch.save(dataset, out_path)
    print(f"✓ Saved {len(dataset)} graphs to {out_path} ({time.time() - t0:.1f}s)")
    return dataset


def build_cohort_parkinson(adapter: mcl.Universal19ChannelAdapter, graph_builder: TopologicalGraphBuilder):
    print("\n" + "=" * 70)
    print("🧠 [Cohort 3/4] Processing UCSD Parkinson's Disease Dataset (OpenNeuro ds002778)...")
    manifest = mcl.get_ucsd_parkinson_manifest(PARKINSON_DIR)
    print(f"Loaded manifest: {len(manifest)} sessions ({sum(1 for x in manifest if x['group']=='HC')} HC, {sum(1 for x in manifest if x['group']=='PD_OFF')} PD-OFF, {sum(1 for x in manifest if x['group']=='PD_ON')} PD-ON)")

    dataset = []
    omst_counts = []
    gce_scores = []
    t0 = time.time()

    for idx, item in enumerate(manifest):
        try:
            raw = mne.io.read_raw_bdf(str(item['path']), preload=True, verbose=False)
            data, sfreq = adapter.standardize_raw_to_epochs(raw)
            x_mat, pli_matrix = extract_real_19ch_features_and_pli(data, sfreq)
            adj_bin, adj_weighted, opt_m, max_gce = graph_builder.build_omst_graph(pli_matrix)

            omst_counts.append(opt_m)
            gce_scores.append(max_gce)

            dataset.append({
                'x': torch.tensor(x_mat, dtype=torch.float32),
                'adj': torch.tensor(adj_bin, dtype=torch.float32),
                'adj_weighted': torch.tensor(adj_weighted, dtype=torch.float32),
                'pli': torch.tensor(pli_matrix, dtype=torch.float32),
                'y': torch.tensor(item['label'], dtype=torch.long),
                'subject_id': item['subject_id'],
                'bids_id': item['bids_id'],
                'session': item['session'],
                'group': item['group'],
                'condition': item['condition'],
                'cohort': item['cohort'],
                'domain': item['diagnosis_domain'],
                'n_clean_epochs': data.shape[0],
                'omst_trees': opt_m,
                'gce': max_gce
            })

            if (idx + 1) % 10 == 0 or (idx + 1) == len(manifest):
                print(f"  [{idx + 1}/{len(manifest)}] {item['subject_id']} ({item['group']}): {data.shape[0]} clean epochs, OMST m*={opt_m}, GCE={max_gce:.3f}")
        except Exception as e:
            print(f"  ⚠️ Error processing {item['subject_id']}: {e}")

    out_path = OUTPUT_DIR / "cohort_ucsd_parkinson_graphs.pt"
    torch.save(dataset, out_path)
    print(f"✓ Saved {len(dataset)} graphs to {out_path} ({time.time() - t0:.1f}s)")
    return dataset


def build_cohort_sleep(graph_builder: TopologicalGraphBuilder):
    print("\n" + "=" * 70)
    print("🧠 [Cohort 4/4] Processing Sleep Deprivation Dataset (OpenNeuro ds004902)...")
    manifest = mcl.get_sleep_deprivation_manifest(SLEEP_FIF_DIR)
    print(f"Loaded manifest: {len(manifest)} sessions ({sum(1 for x in manifest if x['label']==0)} Normal Sleep, {sum(1 for x in manifest if x['label']==1)} Sleep Deprived)")

    dataset = []
    omst_counts = []
    gce_scores = []
    t0 = time.time()

    for idx, item in enumerate(manifest):
        try:
            epochs = mne.read_epochs(str(item['path']), preload=True, verbose=False)
            epochs_19 = epochs.pick(CHANNELS_19)
            data = epochs_19.get_data(copy=True)
            sfreq = float(epochs_19.info['sfreq'])

            x_mat, pli_matrix = extract_real_19ch_features_and_pli(data, sfreq)
            adj_bin, adj_weighted, opt_m, max_gce = graph_builder.build_omst_graph(pli_matrix)

            omst_counts.append(opt_m)
            gce_scores.append(max_gce)

            dataset.append({
                'x': torch.tensor(x_mat, dtype=torch.float32),
                'adj': torch.tensor(adj_bin, dtype=torch.float32),
                'adj_weighted': torch.tensor(adj_weighted, dtype=torch.float32),
                'pli': torch.tensor(pli_matrix, dtype=torch.float32),
                'y': torch.tensor(item['label'], dtype=torch.long),
                'subject_id': item['subject_id'],
                'sub_name': item['sub_name'],
                'session': item['session'],
                'group': item['group'],
                'condition': item['condition'],
                'cohort': item['cohort'],
                'domain': item['diagnosis_domain'],
                'n_clean_epochs': data.shape[0],
                'omst_trees': opt_m,
                'gce': max_gce
            })

            if (idx + 1) % 25 == 0 or (idx + 1) == len(manifest):
                print(f"  [{idx + 1}/{len(manifest)}] {item['subject_id']} ({item['condition']}): {data.shape[0]} clean epochs, OMST m*={opt_m}, GCE={max_gce:.3f}")
        except Exception as e:
            print(f"  ⚠️ Error processing {item['subject_id']}: {e}")

    out_path = OUTPUT_DIR / "cohort_sleep_ds004902_graphs.pt"
    torch.save(dataset, out_path)
    print(f"✓ Saved {len(dataset)} graphs to {out_path} ({time.time() - t0:.1f}s)")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Multi-Cohort 19-Channel Graph Builder")
    parser.add_argument("--cohort", choices=["mumtaz", "cavanagh", "parkinson", "sleep", "all"], default="all",
                        help="Which cohort to process (default: all)")
    args = parser.parse_args()

    adapter = mcl.Universal19ChannelAdapter(target_sfreq=250.0, l_freq=1.0, h_freq=40.0)
    graph_builder = TopologicalGraphBuilder(mode='omst', max_msts=6)

    print("🚀 Initializing Multi-Cohort 19-Channel OMST Graph Construction...")

    if args.cohort in ("mumtaz", "all"):
        build_cohort_mumtaz(adapter, graph_builder)

    if args.cohort in ("cavanagh", "all"):
        build_cohort_cavanagh(adapter, graph_builder)

    if args.cohort in ("parkinson", "all"):
        build_cohort_parkinson(adapter, graph_builder)

    if args.cohort in ("sleep", "all"):
        build_cohort_sleep(graph_builder)

    print("\n" + "=" * 70)
    print("✨ All requested multi-cohort graph datasets have been successfully processed!")


if __name__ == "__main__":
    main()
