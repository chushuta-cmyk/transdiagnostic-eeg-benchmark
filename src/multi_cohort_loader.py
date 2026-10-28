#!/usr/bin/env python3
"""
Multi-Cohort Universal 19-Channel Harmonizer & Data Loader.
Supports 4 distinct real resting-state electrophysiological cohorts:
  1. Study 1: OpenNeuro ds004902 (Acute Sleep Deprivation, 71 pairs = 142 sessions, 61-ch ActiCHamp)
  2. Study 2: Mumtaz / Figshare 4244171 (Clinical MDD, 58 subjects = 181 sessions, 19-ch standard 10-20 EDF)
  3. Study 3: Cavanagh / OpenNeuro ds003478 (Clinical Depression Rest, 122 subjects, 66-ch BrainVision/EEGLAB .set)
  4. Study 4: UCSD Nicole Swann / OpenNeuro ds002778 (Parkinson's Disease ON/OFF vs. HC, 31 subjects, 41-ch BioSemi .bdf)

Key Capabilities:
  - Universal 19-channel standard 10-20 mapping with case-insensitive matching and legacy label normalization.
  - Standardized preprocessing: 1.0–40.0 Hz FIR zero-phase bandpass, common average reference (CAR),
    resampling to 250 Hz, 4.0-s fixed epoching, and peak-to-peak artifact rejection (150 uV).
  - Standardized metadata extraction (BIDS participants.tsv, clinical scales: BDI, HAMD, MMSE, visit conditions).
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import mne

from harmonizer_and_graph import CHANNELS_19, LEGACY_1020_MAP


class Universal19ChannelAdapter:
    """Standardizes heterogeneous EEG recordings (EDF, EEGLAB .set, BioSemi .bdf, MNE .fif) to 19 channels."""

    def __init__(self, target_sfreq: float = 250.0, l_freq: float = 1.0, h_freq: float = 40.0,
                 epoch_duration: float = 4.0, ptp_threshold_uv: float = 150.0):
        self.target_sfreq = target_sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.epoch_duration = epoch_duration
        self.ptp_threshold_uv = ptp_threshold_uv

    @staticmethod
    def normalize_channel_name(ch: str) -> Optional[str]:
        """Maps diverse recording channel labels to canonical CHANNELS_19."""
        # 1. Clean prefixes and suffixes (e.g. 'EEG Fp1-LE', 'Fp1-REF', 'EEG Cz')
        clean = ch.replace('EEG ', '').replace('-LE', '').replace('-REF', '').replace('.', '').strip()
        # 2. Check legacy 10-20 notation (T3->T7, T4->T8, T5->P7, T6->P8)
        clean = LEGACY_1020_MAP.get(clean.upper(), clean)
        clean = LEGACY_1020_MAP.get(clean, clean)
        # 3. Case-insensitive comparison against canonical 19 channels
        for std_ch in CHANNELS_19:
            if clean.upper() == std_ch.upper():
                return std_ch
        return None

    def standardize_raw_to_epochs(self, raw: mne.io.BaseRaw) -> Tuple[np.ndarray, float]:
        """
        Extracts the 19 standard channels, references to average, filters 1–40 Hz,
        resamples to target_sfreq, segments into 4s epochs, and rejects high-amplitude artifacts.
        Returns:
            epochs_data: np.ndarray of shape [n_clean_epochs, 19, n_samples] in Volts.
            sfreq: float (target sampling frequency).
        """
        # Find and map available channels to standard 19
        rename_dict = {}
        for ch in raw.ch_names:
            std_ch = self.normalize_channel_name(ch)
            if std_ch is not None and std_ch not in rename_dict.values():
                rename_dict[ch] = std_ch

        missing = [c for c in CHANNELS_19 if c not in rename_dict.values()]
        if missing:
            raise ValueError(f"Recording missing required 10-20 channels: {missing} (Found {len(rename_dict)}/19)")

        raw.rename_channels(rename_dict)
        raw.pick(CHANNELS_19)
        raw.reorder_channels(CHANNELS_19)

        # Preprocessing: Common Average Reference + 1.0–40.0 Hz bandpass
        raw.set_eeg_reference('average', projection=False, verbose=False)
        raw.filter(l_freq=self.l_freq, h_freq=self.h_freq, verbose=False)

        # Resample to target_sfreq if different
        if abs(raw.info['sfreq'] - self.target_sfreq) > 1e-3:
            raw.resample(self.target_sfreq, verbose=False)

        # Generate fixed-length epochs (e.g. 4.0s)
        epochs = mne.make_fixed_length_epochs(raw, duration=self.epoch_duration, preload=True, verbose=False)
        data = epochs.get_data(copy=True)  # [n_epochs, 19, n_times] in Volts

        # Peak-to-peak amplitude artifact rejection
        ptp = np.ptp(data, axis=-1).max(axis=1)  # max peak-to-peak across 19 channels per epoch
        threshold_v = self.ptp_threshold_uv * 1e-6
        clean_mask = (ptp < threshold_v)

        if np.sum(clean_mask) >= 5:
            clean_data = data[clean_mask]
        else:
            # Fallback: keep the least noisy 8 epochs to prevent dropping the subject
            ranked_idx = np.argsort(ptp)[:min(len(ptp), 8)]
            clean_data = data[ranked_idx]

        return clean_data, float(self.target_sfreq)


# ==============================================================================
# Cohort Manifest Loaders
# ==============================================================================

def get_mumtaz_mdd_manifest(data_dir: Path, state: str = 'EC') -> List[Dict]:
    """
    Parses Mumtaz et al. (Figshare 4244171) Clinical Depression dataset.
    Default state='EC' selects resting Eyes-Closed recordings (standard in literature).
    """
    manifest = []
    # Match files: e.g. 'H S1 EC.edf', 'H S10 EC.edf', 'MDD S1 EC.edf', 'MDD S11  EC.edf'
    pattern = f"*{state}*.edf" if state else "*.edf"
    all_edfs = sorted(data_dir.glob(pattern))

    for p in all_edfs:
        name = p.name
        is_hc = name.startswith('H ') or name.startswith('H_') or name.startswith('H')
        is_mdd = name.startswith('MDD')

        if not (is_hc or is_mdd):
            continue

        group = 'HC' if is_hc else 'MDD'
        label = 0 if is_hc else 1

        # Extract subject ID number
        m = re.search(r'S\s*(\d+)', name, re.IGNORECASE)
        sub_num = int(m.group(1)) if m else len(manifest) + 1
        unique_id = f"mumtaz_{group.lower()}_{sub_num:03d}"

        manifest.append({
            'subject_id': unique_id,
            'sub_num': sub_num,
            'group': group,
            'label': label,
            'state': state,
            'path': p,
            'file_format': 'edf',
            'cohort': 'Mumtaz_MDD_4244171',
            'diagnosis_domain': 'Psychiatric_Depression'
        })
    return manifest


def get_cavanagh_depression_manifest(bids_dir: Path, bdi_control_max: float = 9.0,
                                     bdi_depressed_min: float = 14.0) -> List[Dict]:
    """
    Parses Cavanagh et al. (OpenNeuro ds003478) Depression Rest dataset.
    Reads participants.tsv to obtain Beck Depression Inventory (BDI) and SCID diagnosis.
    Categorizes:
      - Control (label=0): BDI <= 9.0 and SCID is not Current MDD
      - Depressed (label=1): BDI >= 14.0 or SCID is Current MDD
    """
    tsv_path = bids_dir / "participants.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"Missing participants.tsv in {bids_dir}")

    df_parts = pd.read_csv(tsv_path, sep='\t')
    manifest = []

    for _, row in df_parts.iterrows():
        sub_id = str(row['participant_id'])
        sub_dir = bids_dir / sub_id / "eeg"
        if not sub_dir.exists():
            continue

        # Prefer run-01, fallback to run-02
        set_files = sorted(sub_dir.glob("*task-Rest*_eeg.set"))
        if not set_files:
            continue
        set_path = set_files[0]

        bdi = float(row['BDI']) if pd.notnull(row['BDI']) else np.nan
        scid = str(row['SCID']) if pd.notnull(row['SCID']) else ''
        age = float(row['age']) if pd.notnull(row['age']) else np.nan
        sex = int(row['sex']) if pd.notnull(row['sex']) else np.nan

        # Clinical stratification
        is_depressed = (bdi >= bdi_depressed_min) or ('Current MDD' in scid)
        is_control = (bdi <= bdi_control_max) and ('Current MDD' not in scid) and ('Past MDD' not in scid)

        if is_control:
            label = 0
            group = 'Control'
        elif is_depressed:
            label = 1
            group = 'Depressed'
        else:
            # Sub-threshold / Mild depression (BDI 10-13) or past MDD
            label = -1
            group = 'Subclinical'

        manifest.append({
            'subject_id': f"unm_{sub_id}",
            'bids_id': sub_id,
            'group': group,
            'label': label,
            'bdi': bdi,
            'age': age,
            'sex': sex,
            'scid': scid,
            'path': set_path,
            'file_format': 'eeglab_set',
            'cohort': 'Cavanagh_ds003478',
            'diagnosis_domain': 'Psychiatric_Depression'
        })
    return manifest


def get_ucsd_parkinson_manifest(bids_dir: Path) -> List[Dict]:
    """
    Parses UCSD Nicole Swann (OpenNeuro ds002778) Parkinson's Disease dataset.
    Extracts:
      - Healthy Controls (sub-hc* / ses-hc): label = 0
      - Parkinson Patients OFF medication (sub-pd* / ses-off): label = 1
      - Parkinson Patients ON medication (sub-pd* / ses-on): label = 2 (or 1 for PD general)
    """
    tsv_path = bids_dir / "participants.tsv"
    df_parts = pd.read_csv(tsv_path, sep='\t') if tsv_path.exists() else None

    manifest = []
    # Find all resting .bdf files
    all_bdfs = sorted(bids_dir.glob("sub-*/*/*/*task-rest_eeg.bdf"))

    for p in all_bdfs:
        sub_id = p.parent.parent.parent.name  # e.g. 'sub-hc1', 'sub-pd3'
        ses_id = p.parent.parent.name         # e.g. 'ses-hc', 'ses-off', 'ses-on'

        is_hc = 'hc' in sub_id.lower() or 'hc' in ses_id.lower()
        is_off = 'off' in ses_id.lower()
        is_on = 'on' in ses_id.lower()

        if is_hc:
            group = 'HC'
            label = 0
            condition = 'Healthy_Control'
        elif is_off:
            group = 'PD_OFF'
            label = 1
            condition = 'PD_Off_Medication'
        elif is_on:
            group = 'PD_ON'
            label = 1  # general PD class 1, with condition tag
            condition = 'PD_On_Medication'
        else:
            continue

        meta = {
            'subject_id': f"ucsd_{sub_id}_{ses_id}",
            'bids_id': sub_id,
            'session': ses_id,
            'group': group,
            'condition': condition,
            'label': label,
            'path': p,
            'file_format': 'bdf',
            'cohort': 'UCSD_Parkinson_ds002778',
            'diagnosis_domain': 'Neurological_Degenerative'
        }

        # Match demographics if tsv exists
        if df_parts is not None and sub_id in df_parts['participant_id'].values:
            row = df_parts[df_parts['participant_id'] == sub_id].iloc[0]
            meta['age'] = float(row['age']) if 'age' in row and pd.notnull(row['age']) else np.nan
            meta['gender'] = str(row['gender']) if 'gender' in row else ''
            meta['disease_duration'] = float(row['disease_duration']) if 'disease_duration' in row and pd.notnull(row['disease_duration']) and row['disease_duration'] != 'n/a' else np.nan

        manifest.append(meta)
    return manifest


def get_sleep_deprivation_manifest(fif_dir: Path, max_pairs: Optional[int] = None) -> List[Dict]:
    """
    Parses OpenNeuro ds004902 Sleep Deprivation preprocessed epochs.
    71 paired subjects (142 sessions):
      - Normal Sleep (NS): label = 0
      - Sleep Deprived (SD): label = 1
    """
    manifest = []
    # Load all paired subjects
    all_fif_files = sorted(fif_dir.glob("sub-*/*/*cleaned-epo.fif"))
    sub_sessions = {}
    for f in all_fif_files:
        sub = f.parent.parent.name
        sub_sessions.setdefault(sub, []).append(f)

    sorted_subs = sorted(sub_sessions.keys())
    if max_pairs is not None:
        sorted_subs = sorted_subs[:max_pairs]

    for sub in sorted_subs:
        files = sub_sessions[sub]
        for f in files:
            ses = f.parent.name
            # Parse condition from filename or parent
            is_ns = 'ses-1' in ses or '_ses-1_' in f.name
            cond = 'NS' if is_ns else 'SD'
            label = 0 if is_ns else 1

            manifest.append({
                'subject_id': f"sleep_{sub}_{ses}",
                'sub_name': sub,
                'session': ses,
                'group': cond,
                'condition': cond,
                'label': label,
                'path': f,
                'file_format': 'mne_fif',
                'cohort': 'ds004902_SleepDeprivation',
                'diagnosis_domain': 'Acute_Physiological_Stress'
            })
    return manifest
