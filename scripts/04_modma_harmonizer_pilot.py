#!/usr/bin/env python3
"""
Step 04: Real Clinical Depression EEG Harmonizer & OMST Graph Generator (Study 2).
Reads 24 REAL clinical resting-state Eyes-Closed .edf recordings (12 Healthy Controls,
12 Major Depressive Disorder patients) from the public MPHCE dataset (Mumtaz et al., 2016;
benchmark dataset in Lu et al., 2024, IEEE Trans. Affective Computing, Figshare 4244171)
stored in data/raw/mphce_mdd/.

Zero synthetic/simulated data:
  1. Harmonizes legacy 10-20 EDF channel labels (EEG Fp1-LE, T3->T7, T4->T8, T5->P7, T6->P8)
     into the canonical 19-channel 10-20 montage (CHANNELS_19).
  2. Applies 1-40 Hz bandpass filtering, average reference, and 4-second epoching with
     artifact rejection (>150 uV peak-to-peak).
  3. Extracts real per-channel periodic residual bandpowers (Theta, Delta, Alpha) and
     aperiodic exponent chi, plus real 4-13 Hz Phase Lag Index (PLI) matrices.
  4. Constructs true Orthogonal Minimal Spanning Trees (OMST) via Global Cost Efficiency (GCE).
"""

import re
import sys
from pathlib import Path
import numpy as np
import torch
import mne

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = BENCHMARK_ROOT / "src"
sys.path.append(str(SRC_DIR))
from harmonizer_and_graph import (
    TopologicalGraphBuilder,
    extract_real_19ch_features_and_pli,
    CHANNELS_19,
    LEGACY_1020_MAP
)

RAW_MDD_DIR = BENCHMARK_ROOT / "data" / "raw" / "mphce_mdd"
OUTPUT_PATH = BENCHMARK_ROOT / "data" / "derivatives" / "modma_pilot_graphs.pt"


def harmonize_edf_to_19ch(edf_path: Path):
    """Loads a real clinical EDF file and harmonizes channels to standard CHANNELS_19."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    # Map 'EEG Fp1-LE' -> 'Fp1' and legacy T3/T4/T5/T6 -> T7/T8/P7/P8
    rename_dict = {}
    for ch in raw.ch_names:
        clean = ch.replace('EEG ', '').replace('-LE', '').replace('-REF', '').strip()
        clean = LEGACY_1020_MAP.get(clean, clean)
        if clean in CHANNELS_19:
            rename_dict[ch] = clean

    raw.rename_channels(rename_dict)
    missing = [c for c in CHANNELS_19 if c not in raw.ch_names]
    if missing:
        raise ValueError(f"File {edf_path.name} missing required 10-20 channels: {missing}")

    raw.pick(CHANNELS_19)
    raw.reorder_channels(CHANNELS_19)

    # Standard preprocessing: 1-40 Hz bandpass + Common Average Reference
    raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)
    raw.set_eeg_reference('average', projection=False, verbose=False)

    # Segment into 4.0s non-overlapping epochs
    epochs = mne.make_fixed_length_epochs(raw, duration=4.0, preload=True, verbose=False)
    data = epochs.get_data(copy=True)  # [n_epochs, 19, n_times] in Volts

    # Reject high-amplitude artifact epochs (> 150 uV peak-to-peak)
    ptp = np.ptp(data, axis=-1).max(axis=1)
    clean_mask = ptp < 150e-6
    if np.sum(clean_mask) >= 5:
        data = data[clean_mask]

    return data, float(raw.info['sfreq'])


def main():
    print("🧠 Processing REAL Clinical Depression & Healthy Control EDF Recordings (MPHCE / Figshare 4244171)...")
    hc_files = sorted(RAW_MDD_DIR.glob("H_*.edf"))[:12]
    mdd_files = sorted(RAW_MDD_DIR.glob("MDD_*.edf"))[:12]

    if len(hc_files) < 12 or len(mdd_files) < 12:
        raise RuntimeError(f"Expected 12 HC and 12 MDD EDF files in {RAW_MDD_DIR}, found {len(hc_files)} HC, {len(mdd_files)} MDD.")

    graph_builder = TopologicalGraphBuilder(mode='omst', max_msts=6)
    dataset = []
    omst_counts = []
    gce_scores = []

    all_records = [(f, 0, 'HC') for f in hc_files] + [(f, 1, 'MDD') for f in mdd_files]

    for idx, (edf_path, label, cond) in enumerate(all_records):
        # Parse subject number from filename
        m = re.search(r'S(\d+)', edf_path.name)
        sub_num = int(m.group(1)) if m else (idx + 1)
        unique_sub_id = sub_num if label == 0 else (100 + sub_num)

        data, sfreq = harmonize_edf_to_19ch(edf_path)
        x_mat, pli_matrix = extract_real_19ch_features_and_pli(data, sfreq)
        adj_bin, adj_weighted, opt_m, max_gce = graph_builder.build_omst_graph(pli_matrix)

        omst_counts.append(opt_m)
        gce_scores.append(max_gce)

        dataset.append({
            'x': torch.tensor(x_mat, dtype=torch.float32),
            'adj': torch.tensor(adj_bin, dtype=torch.float32),
            'adj_weighted': torch.tensor(adj_weighted, dtype=torch.float32),
            'pli': torch.tensor(pli_matrix, dtype=torch.float32),
            'y': torch.tensor(label, dtype=torch.long),
            'subject_id': unique_sub_id,
            'filename': edf_path.name,
            'cohort': 'MPHCE_Clinical_MDD',
            'condition': cond,
            'omst_trees': opt_m,
            'gce': max_gce
        })

        if (idx + 1) % 6 == 0 or (idx + 1) == len(all_records):
            print(f"  Processed {idx + 1}/{len(all_records)} real EDF files ({edf_path.name} [{cond}]: {data.shape[0]} clean epochs, OMST m*={opt_m}, GCE={max_gce:.3f})")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, OUTPUT_PATH)
    densities = [float(torch.mean(d['adj'])) for d in dataset]
    print(f"✓ Saved {len(dataset)} REAL clinical depression graph instances to {OUTPUT_PATH}")
    print(f"  - Cohort breakdown: {len(hc_files)} Healthy Controls, {len(mdd_files)} Major Depressive Disorder")
    print(f"  - True OMST optimal trees m*: mean={np.mean(omst_counts):.2f} ± {np.std(omst_counts):.2f}")
    print(f"  - Global Cost Efficiency (GCE): mean={np.mean(gce_scores):.3f} ± {np.std(gce_scores):.3f}")
    print(f"  - Resulting data-driven graph density: mean={np.mean(densities):.3f} ± {np.std(densities):.3f}")


if __name__ == '__main__':
    main()
