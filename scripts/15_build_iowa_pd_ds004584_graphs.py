#!/usr/bin/env python3
"""
Step 15: Build Universal 19-Channel OMST Brain Graphs for Iowa Parkinson's Disease Cohort (OpenNeuro ds004584).
Dataset characteristics:
  - 149 participants: 100 Parkinson's Disease (PD), 49 Healthy Controls (Control)
  - 64-channel 500 Hz continuous resting EEG
  - Rich clinical scales: MoCA (all 149), UPDRS Part-III motor score (all 100 PD)
  - Pz is the recording reference electrode -> added and reconstructed via CAR

Saves clean PyTorch graph dataset in data/derivatives/cohort_iowa_pd_ds004584_graphs.pt
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

DS_DIR = Path("/Users/tan/Documents/eeg_datastes/ds004584")


def build_cohort_iowa_pd():
    print("=" * 75)
    print("🧠 [Cohort 7] Processing Iowa Parkinson's Disease Cohort (OpenNeuro ds004584)...")
    print("=" * 75)

    participants_tsv = DS_DIR / "participants.tsv"
    df_part = pd.read_csv(participants_tsv, sep="\t")
    print(f"Loaded participants.tsv: {len(df_part)} participants (PD: {sum(df_part['GROUP']=='PD')}, Control: {sum(df_part['GROUP']=='Control')})")

    graph_builder = TopologicalGraphBuilder()

    dataset = []
    omst_counts = []
    gce_scores = []
    skipped = []

    t0 = time.time()

    for idx, row in df_part.iterrows():
        sub_id = row["participant_id"]
        group = row["GROUP"]
        age = row["AGE"]
        gender = row["GENDER"]
        moca = row["MOCA"]
        updrs = row["UPDRS"]
        y_val = 1 if group == "PD" else 0

        set_file = DS_DIR / sub_id / "eeg" / f"{sub_id}_task-Rest_eeg.set"
        if not set_file.exists():
            print(f"  ⚠️ {sub_id}: File not found ({set_file}), skipping")
            skipped.append(sub_id)
            continue

        try:
            # Read EEGLAB .set
            raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)

            # Reconstruct Pz reference channel
            if "Pz" not in raw.ch_names and "pz" not in [c.lower() for c in raw.ch_names]:
                raw.add_reference_channels(ref_channels=["Pz"])

            # Map available channels to standard 19
            rename_dict = {}
            for ch in raw.ch_names:
                std_ch = mcl.Universal19ChannelAdapter.normalize_channel_name(ch)
                if std_ch is not None and std_ch not in rename_dict.values():
                    rename_dict[ch] = std_ch

            missing = [c for c in mcl.CHANNELS_19 if c not in rename_dict.values()]
            if missing:
                print(f"  ⚠️ {sub_id}: Missing required 10-20 channels: {missing}, skipping")
                skipped.append(sub_id)
                continue

            raw.rename_channels(rename_dict)
            raw.pick(mcl.CHANNELS_19)
            raw.reorder_channels(mcl.CHANNELS_19)

            # Preprocessing: Common Average Reference + 1.0–40.0 Hz bandpass
            raw.set_eeg_reference("average", projection=False, verbose=False)
            raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)

            # Resample to 250 Hz
            if abs(raw.info["sfreq"] - 250.0) > 1e-3:
                raw.resample(250.0, verbose=False)

            # 4.0s epochs
            epochs = mne.make_fixed_length_epochs(raw, duration=4.0, preload=True, verbose=False)
            data = epochs.get_data(copy=True)  # [n_epochs, 19, n_times]

            # Artifact rejection: clinical threshold 200 uV with fallback to best 8 epochs
            ptp = np.ptp(data, axis=-1).max(axis=1)  # V
            clean_mask = (ptp < 200.0 * 1e-6)
            if np.sum(clean_mask) >= 5:
                clean_data = data[clean_mask]
            else:
                ranked_idx = np.argsort(ptp)[:min(len(ptp), 8)]
                clean_data = data[ranked_idx]

            if clean_data.shape[0] < 5:
                print(f"  ⚠️ {sub_id}: Too few clean epochs ({clean_data.shape[0]}), skipping")
                skipped.append(sub_id)
                continue

            x_mat, pli_matrix = extract_real_19ch_features_and_pli(clean_data, 250.0)
            adj_bin, adj_weighted, opt_m, max_gce = graph_builder.build_omst_graph(pli_matrix)

            omst_counts.append(opt_m)
            gce_scores.append(max_gce)

            dataset.append({
                "x": torch.tensor(x_mat, dtype=torch.float32),
                "adj": torch.tensor(adj_bin, dtype=torch.float32),
                "adj_weighted": torch.tensor(adj_weighted, dtype=torch.float32),
                "pli": torch.tensor(pli_matrix, dtype=torch.float32),
                "y": torch.tensor(y_val, dtype=torch.long),
                "group": group,
                "age": float(age) if pd.notnull(age) else np.nan,
                "gender": gender,
                "moca": float(moca) if pd.notnull(moca) else np.nan,
                "updrs": float(updrs) if pd.notnull(updrs) else np.nan,
                "subject_id": sub_id,
                "cohort": "Iowa_PD_ds004584",
                "domain": "Movement_Disorders_PD",
                "n_clean_epochs": clean_data.shape[0],
                "clean_duration_s": float(clean_data.shape[0] * 4.0),
                "omst_trees": opt_m,
                "gce": max_gce
            })

            if len(dataset) % 20 == 0 or idx == len(df_part) - 1:
                elapsed = time.time() - t0
                print(f"  [{len(dataset)}/{len(df_part)}] {sub_id} ({group}): {clean_data.shape[0]} epochs, OMST m*={opt_m}, GCE={max_gce:.3f} (elapsed: {elapsed:.1f}s)")

        except Exception as e:
            print(f"  ❌ Error processing {sub_id}: {e}")
            skipped.append(sub_id)

    total_time = time.time() - t0
    out_path = OUTPUT_DIR / "cohort_iowa_pd_ds004584_graphs.pt"
    torch.save(dataset, out_path)

    print("\n" + "=" * 75)
    print(f"✅ Successfully processed {len(dataset)}/{len(df_part)} participants in {total_time:.2f}s!")
    print(f"   - Saved graph dataset to: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"   - Diagnostic breakdown: {sum(1 for d in dataset if d['group']=='PD')} PD, {sum(1 for d in dataset if d['group']=='Control')} Control")
    print(f"   - Mean clean recording duration: {np.mean([d['clean_duration_s'] for d in dataset]):.1f} ± {np.std([d['clean_duration_s'] for d in dataset]):.1f} s")
    print(f"   - Mean OMST optimal trees m*: {np.mean(omst_counts):.2f} ± {np.std(omst_counts):.2f}")
    print(f"   - Mean Global Cost Efficiency (GCE): {np.mean(gce_scores):.4f} ± {np.std(gce_scores):.4f}")
    if skipped:
        print(f"   - Skipped ({len(skipped)}): {skipped}")
    print("=" * 75)
    return dataset


if __name__ == "__main__":
    build_cohort_iowa_pd()
