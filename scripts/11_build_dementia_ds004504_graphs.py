#!/usr/bin/env python3
"""
Step 11: Build Universal 19-Channel OMST Brain Graphs for AHEPA Clinical Dementia Cohort (OpenNeuro ds004504).
Dataset characteristics:
  - 88 participants: 36 Alzheimer's Disease (AD), 23 Frontotemporal Dementia (FTD), 29 Healthy Controls (HC)
  - Native 19-channel standard 10-20 montage
  - Eyes-closed resting state EEG
  - Rich clinical cognitive scores: Mini-Mental State Examination (MMSE)

Saves clean PyTorch graph dataset in data/derivatives/cohort_dementia_ds004504_graphs.pt
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

from harmonizer_and_graph import TopologicalGraphBuilder, extract_real_19ch_features_and_pli
import multi_cohort_loader as mcl

OUTPUT_DIR = BENCHMARK_ROOT / "data" / "derivatives"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DS_DIR = Path("/Users/tan/Documents/eeg_datastes/ds004504")


def build_cohort_dementia():
    print("=" * 75)
    print("🧠 [Cohort 5] Processing AHEPA Clinical Dementia Cohort (OpenNeuro ds004504)...")
    print("=" * 75)

    participants_tsv = DS_DIR / "participants.tsv"
    if not participants_tsv.exists():
        raise FileNotFoundError(f"Participants file not found at {participants_tsv}")

    df_part = pd.read_csv(participants_tsv, sep="\t")
    print(f"Loaded participants.tsv: {len(df_part)} total records")
    print(f"Group distribution: {dict(df_part['Group'].value_counts())}")

    adapter = mcl.Universal19ChannelAdapter(
        target_sfreq=250.0,
        l_freq=1.0,
        h_freq=40.0,
        epoch_duration=4.0,
        ptp_threshold_uv=150.0
    )
    graph_builder = TopologicalGraphBuilder()

    dataset = []
    omst_counts = []
    gce_scores = []
    skipped = []

    t0 = time.time()

    for idx, row in df_part.iterrows():
        sub_id = row["participant_id"]
        group_code = row["Group"]  # A = AD, F = FTD, C = Control
        age = row["Age"]
        gender = row["Gender"]
        mmse = row["MMSE"]

        # Binary label: 1 for Dementia (AD or FTD), 0 for Control
        y_binary = 1 if group_code in ["A", "F"] else 0
        diag_str = "Alzheimer" if group_code == "A" else ("FTD" if group_code == "F" else "Control")

        # Find eyesclosed set file
        set_files = list((DS_DIR / sub_id / "eeg").glob("*task-eyesclosed*_eeg.set"))
        if not set_files:
            # Fallback to any set file in subject eeg directory
            set_files = list((DS_DIR / sub_id / "eeg").glob("*.set"))

        if not set_files:
            print(f"  ⚠️ No .set file found for {sub_id}")
            skipped.append(sub_id)
            continue

        set_path = set_files[0]

        try:
            raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
            data, sfreq = adapter.standardize_raw_to_epochs(raw)

            if data.shape[0] < 5:
                print(f"  ⚠️ {sub_id}: Too few clean epochs ({data.shape[0]}), skipping")
                skipped.append(sub_id)
                continue

            x_mat, pli_matrix = extract_real_19ch_features_and_pli(data, sfreq)
            adj_bin, adj_weighted, opt_m, max_gce = graph_builder.build_omst_graph(pli_matrix)

            omst_counts.append(opt_m)
            gce_scores.append(max_gce)

            dataset.append({
                "x": torch.tensor(x_mat, dtype=torch.float32),
                "adj": torch.tensor(adj_bin, dtype=torch.float32),
                "adj_weighted": torch.tensor(adj_weighted, dtype=torch.float32),
                "pli": torch.tensor(pli_matrix, dtype=torch.float32),
                "y": torch.tensor(y_binary, dtype=torch.long),
                "group_code": group_code,
                "diagnosis": diag_str,
                "mmse": float(mmse) if pd.notnull(mmse) else np.nan,
                "age": float(age) if pd.notnull(age) else np.nan,
                "gender": gender,
                "subject_id": sub_id,
                "cohort": "AHEPA_Dementia_ds004504",
                "domain": "Neurodegenerative_Dementia",
                "n_clean_epochs": data.shape[0],
                "clean_duration_s": float(data.shape[0] * 4.0),
                "omst_trees": opt_m,
                "gce": max_gce
            })

            if (len(dataset)) % 10 == 0 or idx == len(df_part) - 1:
                elapsed = time.time() - t0
                print(f"  [{len(dataset)}/{len(df_part)}] {sub_id} ({diag_str}, MMSE={mmse}): {data.shape[0]} epochs ({data.shape[0]*4.0:.1f}s), OMST m*={opt_m}, GCE={max_gce:.3f} (elapsed: {elapsed:.1f}s)")

        except Exception as e:
            print(f"  ❌ Error processing {sub_id} ({set_path.name}): {e}")
            skipped.append(sub_id)

    total_time = time.time() - t0
    out_path = OUTPUT_DIR / "cohort_dementia_ds004504_graphs.pt"
    torch.save(dataset, out_path)

    print("\n" + "=" * 75)
    print(f"✅ Successfully processed {len(dataset)}/{len(df_part)} subjects in {total_time:.2f}s!")
    print(f"   - Saved graph dataset to: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"   - Clinical breakdown: {sum(1 for d in dataset if d['group_code']=='A')} AD, {sum(1 for d in dataset if d['group_code']=='F')} FTD, {sum(1 for d in dataset if d['group_code']=='C')} Controls")
    print(f"   - Mean clean recording duration: {np.mean([d['clean_duration_s'] for d in dataset]):.1f} ± {np.std([d['clean_duration_s'] for d in dataset]):.1f} s")
    print(f"   - Mean OMST optimal trees m*: {np.mean(omst_counts):.2f} ± {np.std(omst_counts):.2f}")
    print(f"   - Mean Global Cost Efficiency (GCE): {np.mean(gce_scores):.4f} ± {np.std(gce_scores):.4f}")
    if skipped:
        print(f"   - Skipped subjects ({len(skipped)}): {skipped}")
    print("=" * 75)
    return dataset


if __name__ == "__main__":
    build_cohort_dementia()
