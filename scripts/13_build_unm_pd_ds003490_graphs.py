#!/usr/bin/env python3
"""
Step 13: Build Universal 19-Channel OMST Brain Graphs for UNM Parkinson's Disease Cohort (OpenNeuro ds003490).
Dataset characteristics:
  - 50 participants: 25 Parkinson's Disease (PD), 25 Age-Matched Healthy Controls (CTL)
  - 75 total resting EEG sessions:
      * PD: 25 OFF-medication (washout >= 15h) + 25 ON-medication (counterbalanced across ses-01 and ses-02)
      * CTL: 25 ses-01 recordings
  - 64-channel 500 Hz EEG standardized to universal 10-20 19-channel format

Saves clean PyTorch graph dataset in data/derivatives/cohort_unm_pd_ds003490_graphs.pt
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

DS_DIR = Path("/Users/tan/Documents/eeg_datastes/ds003490")


def build_cohort_unm_pd():
    print("=" * 75)
    print("🧠 [Cohort 6] Processing UNM Parkinson's Disease Cohort (OpenNeuro ds003490)...")
    print("=" * 75)

    participants_tsv = DS_DIR / "participants.tsv"
    df_part = pd.read_csv(participants_tsv, sep="\t")
    print(f"Loaded participants.tsv: {len(df_part)} participants (PD: {sum(df_part['Group']=='PD')}, CTL: {sum(df_part['Group']=='CTL')})")

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

    # Build session list
    sessions_to_process = []
    for _, row in df_part.iterrows():
        sub_id = row["participant_id"]
        group = row["Group"]
        sex = row["sex"]
        age = row["age"]

        if group == "PD":
            for ses_name, med_col in [("ses-01", "sess1_Med"), ("ses-02", "sess2_Med")]:
                med_state = str(row[med_col]).strip()
                set_files = list((DS_DIR / sub_id / ses_name / "eeg").glob("*task-Rest*.set"))
                if set_files:
                    sessions_to_process.append({
                        "sub_id": sub_id,
                        "ses_name": ses_name,
                        "group": "PD",
                        "med_state": med_state,
                        "y_pd": 1,
                        "y_med": 1 if med_state == "ON" else 0,
                        "age": age,
                        "sex": sex,
                        "path": set_files[0]
                    })
                else:
                    skipped.append(f"{sub_id}_{ses_name}")
        else: # CTL
            set_files = list((DS_DIR / sub_id / "ses-01" / "eeg").glob("*task-Rest*.set"))
            if set_files:
                sessions_to_process.append({
                    "sub_id": sub_id,
                    "ses_name": "ses-01",
                    "group": "CTL",
                    "med_state": "CTL",
                    "y_pd": 0,
                    "y_med": -1,
                    "age": age,
                    "sex": sex,
                    "path": set_files[0]
                })
            else:
                skipped.append(f"{sub_id}_ses-01")

    print(f"Total resting sessions to process: {len(sessions_to_process)}")

    for idx, item in enumerate(sessions_to_process):
        sub_id = item["sub_id"]
        ses_name = item["ses_name"]
        set_path = item["path"]

        try:
            raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
            data, sfreq = adapter.standardize_raw_to_epochs(raw)

            if data.shape[0] < 5:
                print(f"  ⚠️ {sub_id} {ses_name}: Too few clean epochs ({data.shape[0]}), skipping")
                skipped.append(f"{sub_id}_{ses_name}")
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
                "y": torch.tensor(item["y_pd"], dtype=torch.long),
                "y_med": item["y_med"],
                "group": item["group"],
                "med_state": item["med_state"],
                "session": ses_name,
                "age": float(item["age"]) if pd.notnull(item["age"]) else np.nan,
                "sex": item["sex"],
                "subject_id": sub_id,
                "cohort": "UNM_PD_ds003490",
                "domain": "Movement_Disorders_PD",
                "n_clean_epochs": data.shape[0],
                "clean_duration_s": float(data.shape[0] * 4.0),
                "omst_trees": opt_m,
                "gce": max_gce
            })

            if (len(dataset)) % 15 == 0 or idx == len(sessions_to_process) - 1:
                elapsed = time.time() - t0
                print(f"  [{len(dataset)}/{len(sessions_to_process)}] {sub_id} {ses_name} ({item['group']}-{item['med_state']}): {data.shape[0]} epochs, OMST m*={opt_m}, GCE={max_gce:.3f} (elapsed: {elapsed:.1f}s)")

        except Exception as e:
            print(f"  ❌ Error processing {sub_id} {ses_name}: {e}")
            skipped.append(f"{sub_id}_{ses_name}")

    total_time = time.time() - t0
    out_path = OUTPUT_DIR / "cohort_unm_pd_ds003490_graphs.pt"
    torch.save(dataset, out_path)

    print("\n" + "=" * 75)
    print(f"✅ Successfully processed {len(dataset)}/{len(sessions_to_process)} sessions in {total_time:.2f}s!")
    print(f"   - Saved graph dataset to: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"   - Session breakdown: {sum(1 for d in dataset if d['med_state']=='ON')} PD-ON, {sum(1 for d in dataset if d['med_state']=='OFF')} PD-OFF, {sum(1 for d in dataset if d['med_state']=='CTL')} Controls")
    print(f"   - Mean clean recording duration: {np.mean([d['clean_duration_s'] for d in dataset]):.1f} ± {np.std([d['clean_duration_s'] for d in dataset]):.1f} s")
    print(f"   - Mean OMST optimal trees m*: {np.mean(omst_counts):.2f} ± {np.std(omst_counts):.2f}")
    print(f"   - Mean Global Cost Efficiency (GCE): {np.mean(gce_scores):.4f} ± {np.std(gce_scores):.4f}")
    if skipped:
        print(f"   - Skipped ({len(skipped)}): {skipped}")
    print("=" * 75)
    return dataset


if __name__ == "__main__":
    build_cohort_unm_pd()
