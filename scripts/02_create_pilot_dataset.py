#!/usr/bin/env python3
"""
Step 02: Extract 100% REAL 19-Channel Graph Dataset from Study 1 Cleaned Epochs (ds004902).
Reads actual MNE .fif epoch files (sub-XX_ses-Y_cleaned-epo.fif) for 20 paired subjects (40 sessions).
Zero synthetic/random features:
  - Picks the exact 19 canonical 10-20 channels from the 61-channel montage.
  - Computes real per-channel Welch PSD, log-log aperiodic exponent chi, and periodic
    residual bandpowers (Theta, Delta, Alpha) via extract_real_19ch_features_and_pli().
  - Computes real Phase Lag Index (PLI) matrices from Hilbert phase series (4-13 Hz).
  - Extracts true Orthogonal Minimal Spanning Trees (OMST) maximizing Global Cost
    Efficiency (GCE, Dimitriadis et al., 2017).
"""

import sys
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

PROJECT_DIR = Path("/Users/tan/Documents/qwen-agent/aCAgqOVe0x/default/sleep_deprivation_eeg")
FEATURES_PATH = PROJECT_DIR / "data" / "derivatives" / "features" / "v3_conservative" / "eeg_session_features.parquet"
PREPROC_DIR = PROJECT_DIR / "data" / "derivatives" / "preprocessed_v3_conservative"
OUTPUT_PATH = BENCHMARK_ROOT / "data" / "derivatives" / "pilot_graph_dataset.pt"


def main():
    print("🔄 Loading session metadata & reading REAL cleaned-epo.fif files (Study 1: ds004902)...")
    df = pd.read_parquet(FEATURES_PATH)

    # Identify subjects with both NS and SD sessions and existing .fif files
    counts = df.groupby('subject')['condition'].nunique()
    paired_subs = sorted(counts[counts == 2].index.tolist())[:20]
    df_pilot = df[df['subject'].isin(paired_subs)].sort_values(['subject', 'session']).copy()
    print(f"Selected {len(paired_subs)} paired subjects ({len(df_pilot)} real .fif sessions).")

    graph_builder = TopologicalGraphBuilder(mode='omst', max_msts=6)
    dataset = []
    omst_counts = []
    gce_scores = []

    for idx, (_, row) in enumerate(df_pilot.iterrows()):
        sub = row['subject']
        ses = f"ses-{row['session']}" if not str(row['session']).startswith('ses-') else str(row['session'])
        cond = row['condition']
        sub_id = int(sub.replace('sub-', ''))
        label = 0 if cond == 'NS' else 1

        fif_path = PREPROC_DIR / sub / ses / f"{sub}_{ses}_cleaned-epo.fif"
        if not fif_path.exists():
            raise FileNotFoundError(f"Missing real .fif file: {fif_path}")

        epochs = mne.read_epochs(str(fif_path), preload=True, verbose=False)
        epochs_19 = epochs.pick(CHANNELS_19)
        data = epochs_19.get_data(copy=True)  # [n_epochs, 19, n_times]
        sfreq = float(epochs_19.info['sfreq'])

        # Extract 100% real spectral/aperiodic node features & real Hilbert PLI matrix
        x_mat, pli_matrix = extract_real_19ch_features_and_pli(data, sfreq)

        # Build true OMST graph via Global Cost Efficiency (GCE) optimization
        adj_bin, adj_weighted, opt_m, max_gce = graph_builder.build_omst_graph(pli_matrix)
        omst_counts.append(opt_m)
        gce_scores.append(max_gce)

        dataset.append({
            'x': torch.tensor(x_mat, dtype=torch.float32),
            'adj': torch.tensor(adj_bin, dtype=torch.float32),
            'adj_weighted': torch.tensor(adj_weighted, dtype=torch.float32),
            'pli': torch.tensor(pli_matrix, dtype=torch.float32),
            'y': torch.tensor(label, dtype=torch.long),
            'subject_id': sub_id,
            'cohort': 'ds004902_SleepDeprivation',
            'condition': cond,
            'omst_trees': opt_m,
            'gce': max_gce
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(df_pilot):
            print(f"  Processed {idx + 1}/{len(df_pilot)} real .fif sessions ({sub} {ses} [{cond}]: OMST m*={opt_m}, GCE={max_gce:.3f})")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, OUTPUT_PATH)
    densities = [float(torch.mean(d['adj'])) for d in dataset]
    print(f"✓ Saved {len(dataset)} REAL EEG graph instances to {OUTPUT_PATH}")
    print(f"  - Node feature shape: {dataset[0]['x'].shape} (Real Periodic Theta, Delta, Alpha, Aperiodic chi)")
    print(f"  - True OMST optimal trees m*: mean={np.mean(omst_counts):.2f} ± {np.std(omst_counts):.2f}")
    print(f"  - Global Cost Efficiency (GCE): mean={np.mean(gce_scores):.3f} ± {np.std(gce_scores):.3f}")
    print(f"  - Resulting data-driven graph density: mean={np.mean(densities):.3f} ± {np.std(densities):.3f}")


if __name__ == '__main__':
    main()
