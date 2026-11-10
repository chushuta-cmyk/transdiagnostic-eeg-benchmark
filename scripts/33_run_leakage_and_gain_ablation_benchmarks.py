#!/usr/bin/env python3
"""
Validation Leakage, Feature Representation, and Gain Perturbation Benchmarks.

This script executes empirical ablations for:
1. Validation data leakage regimes (Strict Participant Grouping vs Pre-Standardization vs Subject Mixing).
2. Comparison of spectral parameterization against relative and raw band power across cross-site PD transfer.
3. Multiplicative gain perturbation simulations assessing feature drift, probability drift, and classification stability.
4. Heuristic zero-training slowing baseline evaluation.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
import torch
import torch.nn as nn
import torch.optim as optim
import mne
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GridSearchCV

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = BENCHMARK_ROOT / "src"
sys.path.append(str(SRC_DIR))
from harmonizer_and_graph import NormativeGCN, compute_fold_normative_z
import multi_cohort_loader as mcl

DATA_DIR = BENCHMARK_ROOT / "data" / "derivatives"
REPORTS_DIR = BENCHMARK_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

EEG_ROOT = Path("/Users/tan/Documents/eeg_datastes")


def set_seed(seed: int = 42):
    """Set random seeds for NumPy and PyTorch reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def bootstrap_auc_ci(y_true, y_prob, n_boot: int = 1000, seed: int = 42):
    """Compute non-parametric bootstrap 95% confidence intervals for ROC-AUC."""
    n = len(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan
    base_auc = roc_auc_score(y_true, y_prob)
    rng = np.random.RandomState(seed)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not boot_aucs:
        return float(base_auc), float(base_auc), float(base_auc)
    return float(base_auc), float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))


def cluster_bootstrap_auc_ci(y_true, y_prob, groups, n_boot: int = 1000, seed: int = 42):
    """Compute cluster bootstrap 95% CI resampling unique participants with replacement."""
    unique_groups = np.unique(groups)
    n_clusters = len(unique_groups)
    group_to_indices = {g: np.where(groups == g)[0] for g in unique_groups}
    rng = np.random.RandomState(seed)
    boot_aucs = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=n_clusters, replace=True)
        idx = np.concatenate([group_to_indices[g] for g in sampled_groups])
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not boot_aucs:
        base_auc = roc_auc_score(y_true, y_prob)
        return float(base_auc), float(base_auc)
    return float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))


def train_eval_gcn(train_z, train_adj, train_y, val_z, val_adj, epochs=50, lr=0.005, weight_decay=1e-4, seed=42):
    """Train and evaluate NormativeGCN on fold tensors."""
    set_seed(seed)
    model = NormativeGCN(in_node_features=4, hidden_dim=32, num_classes=2, dropout=0.25)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(train_z, train_adj)
        loss = criterion(out, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(val_z, val_adj)
        probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
        preds = torch.argmax(logits, dim=-1).numpy()
    return probs, preds


def run_leakage_ablation():
    """Evaluate diagnostic classifiers under strict leak-free and leaky validation regimes."""
    print("\n" + "=" * 75)
    print("🔬 [Part 1] Running Data Leakage Impact Ablation Benchmark (Table S19)...")
    print("=" * 75)

    cohorts = [
        (
            "Study 1: Sleep Deprivation",
            "Paired_Crossover_Discordant",
            "cohort_sleep_ds004902_graphs.pt",
            lambda g: g.get("sub_name", g.get("subject_id")),
        ),
        (
            "Study 4: UCSD Parkinson",
            "Repeated_Sessions_Concordant",
            "cohort_ucsd_parkinson_graphs.pt",
            lambda g: g.get("bids_id", g.get("subject_id")),
        ),
        (
            "Study 6c: UNM Parkinson All",
            "Repeated_Sessions_Concordant",
            "cohort_unm_pd_ds003490_graphs.pt",
            lambda g: g.get("subject_id"),
        ),
    ]

    regimes = ["Strict_Leak_Free", "Leaky_Pre_Standardization", "Leaky_Subject_Mixing"]
    models = ["Tuned_LogisticRegression", "RandomForest", "GCN_OMST"]
    param_grid_lr = {"C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]}

    records = []

    for cohort_label, design_type, fname, group_fn in cohorts:
        graphs = torch.load(DATA_DIR / fname)
        valid_graphs = [g for g in graphs if int(g["y"]) in (0, 1)]
        y_all = np.array([int(g["y"]) for g in valid_graphs])
        groups = np.array([group_fn(g) for g in valid_graphs])
        n_samples = len(valid_graphs)
        n_groups = len(np.unique(groups))
        print(f"\nEvaluating {cohort_label} ({design_type}): N={n_samples} sessions, {n_groups} unique participants")

        all_x = torch.stack([g["x"] for g in valid_graphs])
        all_y = torch.tensor(y_all, dtype=torch.long)

        # Pre-compute leaky global channel-wise standardization across the entire dataset (train + test pooled)
        # using global cohort channel-wise mean and std before any train/test split
        global_mean_ch = all_x.mean(dim=0, keepdim=True)  # [1, 19, 4]
        global_std_ch = all_x.std(dim=0, keepdim=True) + 1e-6
        global_z_all = (all_x - global_mean_ch) / global_std_ch

        for regime in regimes:
            set_seed(42)
            oof_probs = {m: np.zeros(n_samples) for m in models}
            oof_preds = {m: np.zeros(n_samples) for m in models}

            if regime in ("Strict_Leak_Free", "Leaky_Pre_Standardization"):
                splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                splits = list(splitter.split(valid_graphs, y_all, groups=groups))
            else:
                # Leaky_Subject_Mixing: sessions from the same participant can appear in both train and val folds
                splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                splits = list(splitter.split(valid_graphs, y_all))

            # Count how many validation sessions share a participant ID with the training fold
            n_leaked_sessions = 0
            for train_idx, val_idx in splits:
                tr_subs = set(groups[train_idx])
                n_leaked_sessions += sum(1 for g_id in groups[val_idx] if g_id in tr_subs)
            leak_rate = 100.0 * n_leaked_sessions / n_samples

            for fold, (train_idx, val_idx) in enumerate(splits):
                train_x = torch.stack([valid_graphs[i]["x"] for i in train_idx])
                train_adj = torch.stack([valid_graphs[i]["adj"] for i in train_idx])
                train_y = torch.tensor([valid_graphs[i]["y"] for i in train_idx], dtype=torch.long)

                val_x = torch.stack([valid_graphs[i]["x"] for i in val_idx])
                val_adj = torch.stack([valid_graphs[i]["adj"] for i in val_idx])

                if regime == "Strict_Leak_Free":
                    train_z, val_z, _, _ = compute_fold_normative_z(train_x, train_y, val_x)
                elif regime == "Leaky_Pre_Standardization":
                    train_z = global_z_all[train_idx]
                    val_z = global_z_all[val_idx]
                elif regime == "Leaky_Subject_Mixing":
                    train_z, val_z, _, _ = compute_fold_normative_z(train_x, train_y, val_x)

                X_tr = train_z.view(len(train_idx), -1).numpy()
                X_va = val_z.view(len(val_idx), -1).numpy()
                y_tr = train_y.numpy()

                # 1. Tuned Logistic Regression (exact 7-grid from Table 2; participant-grouped inner CV when outer CV is grouped)
                if regime in ("Strict_Leak_Free", "Leaky_Pre_Standardization"):
                    inner_groups = [groups[i] for i in train_idx]
                    inner_cv = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(X_tr, y_tr, groups=inner_groups))
                else:
                    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                grid_lr = GridSearchCV(
                    LogisticRegression(max_iter=1000, random_state=42),
                    param_grid_lr,
                    cv=inner_cv,
                    scoring="roc_auc",
                )
                grid_lr.fit(X_tr, y_tr)
                oof_probs["Tuned_LogisticRegression"][val_idx] = grid_lr.predict_proba(X_va)[:, 1]
                oof_preds["Tuned_LogisticRegression"][val_idx] = grid_lr.predict(X_va)

                # 2. Random Forest
                rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, y_tr)
                oof_probs["RandomForest"][val_idx] = rf.predict_proba(X_va)[:, 1]
                oof_preds["RandomForest"][val_idx] = rf.predict(X_va)

                # 3. GCN-OMST
                probs_gcn, preds_gcn = train_eval_gcn(train_z, train_adj, train_y, val_z, val_adj, epochs=50, seed=42 + fold)
                oof_probs["GCN_OMST"][val_idx] = probs_gcn
                oof_preds["GCN_OMST"][val_idx] = preds_gcn

            for m in models:
                auc, ci_lo, ci_hi = bootstrap_auc_ci(y_all, oof_probs[m])
                cl_lo, cl_hi = cluster_bootstrap_auc_ci(y_all, oof_probs[m], groups)
                acc = accuracy_score(y_all, oof_preds[m])
                bal_acc = balanced_accuracy_score(y_all, oof_preds[m])
                records.append({
                    "cohort": cohort_label,
                    "design_type": design_type,
                    "regime": regime,
                    "model": m,
                    "n_sessions": n_samples,
                    "n_participants": n_groups,
                    "subject_overlap_pct": leak_rate,
                    "auc": auc,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                    "cluster_ci_lower": cl_lo,
                    "cluster_ci_upper": cl_hi,
                    "accuracy": acc,
                    "balanced_accuracy": bal_acc,
                })
                print(
                    f"  [{regime:<26}] {m:<25}: AUC = {auc:.3f} "
                    f"(Sess CI [{ci_lo:.3f}, {ci_hi:.3f}], Subj CI [{cl_lo:.3f}, {cl_hi:.3f}]), "
                    f"Overlap = {leak_rate:.1f}%"
                )

    df_leak = pd.DataFrame(records)
    strict_map = df_leak[df_leak["regime"] == "Strict_Leak_Free"].set_index(["cohort", "model"])["auc"].to_dict()
    df_leak["delta_auc_vs_strict"] = df_leak.apply(
        lambda r: r["auc"] - strict_map.get((r["cohort"], r["model"]), np.nan), axis=1
    )

    out_csv = REPORTS_DIR / "benchmark_leakage_ablation.csv"
    df_leak.to_csv(out_csv, index=False)
    print(f"\n✅ Saved leakage ablation benchmark to: {out_csv}")
    return df_leak


# ==============================================================================
# 2. REAL WELCH PSD EXTRACTION & THREE SPECTRAL FEATURE REPRESENTATIONS
# ==============================================================================

def compute_three_representations_from_psd(freqs: np.ndarray, mean_psd: np.ndarray):
    """
    Given freqs [n_freqs] and mean_psd [19, n_freqs] (in linear power units),
    computes three distinct [19, 4] float32 feature matrices:
      1. x_specparam: [Periodic Theta, Periodic Delta, Periodic Alpha, Aperiodic Chi]
      2. x_relpower:  [Relative Theta, Relative Delta, Relative Alpha, log10(Theta/Alpha)]
      3. x_rawlog:    [log10(Theta), log10(Delta), log10(Alpha), log10(Total 1-40Hz Power)]
    """
    valid_mask = (freqs >= 1.0) & (freqs <= 40.0)
    f_valid = freqs[valid_mask]
    log_f = np.log10(f_valid)

    aperiodic_fit_mask = (f_valid < 7.0) | (f_valid > 14.0)
    log_f_fit = log_f[aperiodic_fit_mask]

    theta_mask = (f_valid >= 4.0) & (f_valid < 8.0)
    delta_mask = (f_valid >= 1.0) & (f_valid < 4.0)
    alpha_mask = (f_valid >= 8.0) & (f_valid <= 13.0)

    x_specparam = np.zeros((19, 4), dtype=np.float32)
    x_relpower = np.zeros((19, 4), dtype=np.float32)
    x_rawlog = np.zeros((19, 4), dtype=np.float32)

    for c in range(19):
        psd_c = np.maximum(mean_psd[c, valid_mask], 1e-20)
        log_p = np.log10(psd_c)

        # 1. specparam periodic residual + aperiodic exponent chi
        slope, intercept = np.polyfit(log_f_fit, log_p[aperiodic_fit_mask], 1)
        chi = -slope
        aperiodic_line = intercept + slope * log_f
        flattened_spec = log_p - aperiodic_line

        x_specparam[c, 0] = float(np.mean(flattened_spec[theta_mask]))
        x_specparam[c, 1] = float(np.mean(flattened_spec[delta_mask]))
        x_specparam[c, 2] = float(np.mean(flattened_spec[alpha_mask]))
        x_specparam[c, 3] = float(chi)

        # 2. Relative linear band power (normalized by total 1-40 Hz linear power)
        total_lin = float(np.sum(psd_c)) + 1e-25
        lin_theta = float(np.mean(psd_c[theta_mask]))
        lin_delta = float(np.mean(psd_c[delta_mask]))
        lin_alpha = float(np.mean(psd_c[alpha_mask]))

        x_relpower[c, 0] = float(np.sum(psd_c[theta_mask]) / total_lin)
        x_relpower[c, 1] = float(np.sum(psd_c[delta_mask]) / total_lin)
        x_relpower[c, 2] = float(np.sum(psd_c[alpha_mask]) / total_lin)
        x_relpower[c, 3] = float(np.log10((lin_theta + 1e-25) / (lin_alpha + 1e-25)))

        # 3. Unparameterized raw log10 band power
        x_rawlog[c, 0] = float(np.mean(log_p[theta_mask]))
        x_rawlog[c, 1] = float(np.mean(log_p[delta_mask]))
        x_rawlog[c, 2] = float(np.mean(log_p[alpha_mask]))
        x_rawlog[c, 3] = float(np.mean(log_p))

    return x_specparam, x_relpower, x_rawlog


def load_or_extract_pd_tricenter_psds():
    """Load cached or extract raw Welch PSDs for tri-center Parkinson cohorts (UCSD, UNM, Iowa)."""
    cache_path = DATA_DIR / "pd_tricenter_welch_psds.pt"
    if cache_path.exists():
        print(f"Loading cached tri-center Welch PSDs from {cache_path}...")
        return torch.load(cache_path, weights_only=False)

    print("Extracting real Welch PSDs from raw BIDS files for UCSD, UNM_OFF, and Iowa...")
    adapter_std = mcl.Universal19ChannelAdapter(target_sfreq=250.0, l_freq=1.0, h_freq=40.0, epoch_duration=4.0, ptp_threshold_uv=150.0)

    # Load existing graph files to match exact session ordering and OMST adjacency matrices
    ucsd_graphs = [g for g in torch.load(DATA_DIR / "cohort_ucsd_parkinson_graphs.pt") if int(g["y"]) in (0, 1)]
    unm_graphs = [g for g in torch.load(DATA_DIR / "cohort_unm_pd_ds003490_graphs.pt") if g.get("med_state") in ("OFF", "CTL") and int(g["y"]) in (0, 1)]
    iowa_graphs = [g for g in torch.load(DATA_DIR / "cohort_iowa_pd_ds004584_graphs.pt") if int(g["y"]) in (0, 1)]

    # 1. UCSD (ds002778)
    ucsd_manifest = {item["subject_id"]: item for item in mcl.get_ucsd_parkinson_manifest(EEG_ROOT)}
    ucsd_records = []
    for g in ucsd_graphs:
        sid = g["subject_id"]
        item = ucsd_manifest[sid]
        raw = mne.io.read_raw_bdf(str(item["path"]), preload=True, verbose=False)
        data, sfreq = adapter_std.standardize_raw_to_epochs(raw)
        nperseg = min(int(sfreq * 2.0), data.shape[-1])
        freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
        mean_psd = np.mean(psd, axis=0)
        ucsd_records.append({"subject_id": sid, "y": int(g["y"]), "adj": g["adj"], "freqs": freqs, "mean_psd": mean_psd})
    print(f"  ✓ Extracted UCSD PSDs: {len(ucsd_records)} sessions")

    # 2. UNM OFF + CTL (ds003490)
    ds_unm = EEG_ROOT / "ds003490"
    df_unm = pd.read_csv(ds_unm / "participants.tsv", sep="\t")
    unm_path_map = {}
    for _, row in df_unm.iterrows():
        sub_id = row["participant_id"]
        if row["Group"] == "PD":
            for ses_name, med_col in [("ses-01", "sess1_Med"), ("ses-02", "sess2_Med")]:
                if str(row[med_col]).strip() == "OFF":
                    set_files = list((ds_unm / sub_id / ses_name / "eeg").glob("*task-Rest*.set"))
                    if set_files:
                        unm_path_map[sub_id] = set_files[0]
        else:
            set_files = list((ds_unm / sub_id / "ses-01" / "eeg").glob("*task-Rest*.set"))
            if set_files:
                unm_path_map[sub_id] = set_files[0]

    unm_records = []
    for g in unm_graphs:
        sid = g["subject_id"]
        raw = mne.io.read_raw_eeglab(str(unm_path_map[sid]), preload=True, verbose=False)
        data, sfreq = adapter_std.standardize_raw_to_epochs(raw)
        nperseg = min(int(sfreq * 2.0), data.shape[-1])
        freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
        mean_psd = np.mean(psd, axis=0)
        unm_records.append({"subject_id": sid, "y": int(g["y"]), "adj": g["adj"], "freqs": freqs, "mean_psd": mean_psd})
    print(f"  ✓ Extracted UNM_OFF PSDs: {len(unm_records)} sessions")

    # 3. Iowa (ds004584)
    ds_iowa = EEG_ROOT / "ds004584"
    iowa_records = []
    for g in iowa_graphs:
        sid = g["subject_id"]
        set_file = ds_iowa / sid / "eeg" / f"{sid}_task-Rest_eeg.set"
        raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)
        if "Pz" not in raw.ch_names and "pz" not in [c.lower() for c in raw.ch_names]:
            raw.add_reference_channels(ref_channels=["Pz"])
        rename_dict = {}
        for ch in raw.ch_names:
            std_ch = mcl.Universal19ChannelAdapter.normalize_channel_name(ch)
            if std_ch is not None and std_ch not in rename_dict.values():
                rename_dict[ch] = std_ch
        raw.rename_channels(rename_dict)
        raw.pick(mcl.CHANNELS_19)
        raw.reorder_channels(mcl.CHANNELS_19)
        raw.set_eeg_reference("average", projection=False, verbose=False)
        raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)
        if abs(raw.info["sfreq"] - 250.0) > 1e-3:
            raw.resample(250.0, verbose=False)
        epochs = mne.make_fixed_length_epochs(raw, duration=4.0, preload=True, verbose=False)
        data = epochs.get_data(copy=True)
        ptp = np.ptp(data, axis=-1).max(axis=1)
        clean_mask = ptp < 200.0 * 1e-6
        clean_data = data[clean_mask] if np.sum(clean_mask) >= 5 else data[np.argsort(ptp)[: min(len(ptp), 8)]]
        nperseg = min(int(250.0 * 2.0), clean_data.shape[-1])
        freqs, psd = signal.welch(clean_data, fs=250.0, nperseg=nperseg, axis=-1)
        mean_psd = np.mean(psd, axis=0)
        iowa_records.append({"subject_id": sid, "y": int(g["y"]), "adj": g["adj"], "freqs": freqs, "mean_psd": mean_psd})
    print(f"  ✓ Extracted Iowa PSDs: {len(iowa_records)} sessions")

    cache_obj = {"UCSD": ucsd_records, "UNM_OFF": unm_records, "Iowa": iowa_records}
    torch.save(cache_obj, cache_path)
    print(f"  ✓ Saved cached PSDs to {cache_path}")
    return cache_obj


def run_gain_and_calibration_ablation():
    """Evaluate cross-site transfer across three spectral representations, normative calibration, and gain scaling."""
    print("\n" + "=" * 75)
    print("📡 [Part 2] Running Real Spectral Representation, Calibration & Gain Ablation (Table S20)...")
    print("=" * 75)

    psd_cache = load_or_extract_pd_tricenter_psds()

    # Build baseline (k=1.0) feature tensors for all 3 representations across UCSD, UNM_OFF, Iowa
    site_tensors = {}
    for site_name, rec_list in psd_cache.items():
        spec_list, rel_list, raw_list, adj_list, y_list = [], [], [], [], []
        for r in rec_list:
            x_sp, x_rl, x_rw = compute_three_representations_from_psd(r["freqs"], r["mean_psd"])
            spec_list.append(torch.tensor(x_sp, dtype=torch.float32))
            rel_list.append(torch.tensor(x_rl, dtype=torch.float32))
            raw_list.append(torch.tensor(x_rw, dtype=torch.float32))
            adj_list.append(r["adj"])
            y_list.append(r["y"])
        site_tensors[site_name] = {
            "specparam_periodic_residual": torch.stack(spec_list),
            "relative_band_power": torch.stack(rel_list),
            "raw_log_band_power": torch.stack(raw_list),
            "adj": torch.stack(adj_list),
            "y": torch.tensor(y_list, dtype=torch.long),
            "y_np": np.array(y_list),
            "records": rec_list,
        }

    pairs = [
        ("UCSD", "UNM_OFF"),
        ("UCSD", "Iowa"),
        ("UNM_OFF", "UCSD"),
        ("UNM_OFF", "Iowa"),
        ("Iowa", "UCSD"),
        ("Iowa", "UNM_OFF"),
    ]

    calibration_regimes = ["Uncalibrated_Raw", "Global_Cohort_ZScore", "Target_HC_Normative"]
    feature_types = ["specparam_periodic_residual", "relative_band_power", "raw_log_band_power"]

    records = []

    # Experiment A: 6 Directed Pairs x 3 Calibration Strategies x 3 Real Feature Representations
    for src_name, tgt_name in pairs:
        src = site_tensors[src_name]
        tgt = site_tensors[tgt_name]
        src_y = src["y"]
        tgt_y_np = tgt["y_np"]

        for f_type in feature_types:
            src_x = src[f_type]
            tgt_x = tgt[f_type]

            for calib in calibration_regimes:
                set_seed(42)
                if calib == "Uncalibrated_Raw":
                    train_z = src_x
                    test_z = tgt_x
                elif calib == "Global_Cohort_ZScore":
                    # Channel-wise standardization using pooled cohort mean/std (cases + controls)
                    mu_s = src_x.mean(dim=0, keepdim=True)
                    sd_s = src_x.std(dim=0, keepdim=True) + 1e-6
                    mu_t = tgt_x.mean(dim=0, keepdim=True)
                    sd_t = tgt_x.std(dim=0, keepdim=True) + 1e-6
                    train_z = (src_x - mu_s) / sd_s
                    test_z = (tgt_x - mu_t) / sd_t
                elif calib == "Target_HC_Normative":
                    # Channel-wise standardization relative to each site's healthy controls (y == 0)
                    _, _, src_hc_mu, src_hc_sd = compute_fold_normative_z(src_x, src_y, src_x)
                    _, _, tgt_hc_mu, tgt_hc_sd = compute_fold_normative_z(tgt_x, tgt["y"], tgt_x)
                    train_z = (src_x - src_hc_mu) / src_hc_sd
                    test_z = (tgt_x - tgt_hc_mu) / tgt_hc_sd

                X_tr = train_z.view(len(src_y), -1).numpy()
                X_te = test_z.view(len(tgt_y_np), -1).numpy()

                # 1. Logistic Regression
                lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42).fit(X_tr, src_y.numpy())
                p_lr = lr.predict_proba(X_te)[:, 1]
                pred_lr = (p_lr >= 0.5).astype(int)
                auc_lr, ci_lo_lr, ci_hi_lr = bootstrap_auc_ci(tgt_y_np, p_lr)
                bacc_lr = balanced_accuracy_score(tgt_y_np, pred_lr)
                brier_lr = brier_score_loss(tgt_y_np, p_lr)

                # 2. Random Forest
                rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, src_y.numpy())
                p_rf = rf.predict_proba(X_te)[:, 1]
                pred_rf = (p_rf >= 0.5).astype(int)
                auc_rf, ci_lo_rf, ci_hi_rf = bootstrap_auc_ci(tgt_y_np, p_rf)
                bacc_rf = balanced_accuracy_score(tgt_y_np, pred_rf)
                brier_rf = brier_score_loss(tgt_y_np, p_rf)

                # 3. GCN-OMST
                p_gcn, pred_gcn = train_eval_gcn(train_z, src["adj"], src_y, test_z, tgt["adj"], epochs=40, seed=42)
                auc_gcn, ci_lo_gcn, ci_hi_gcn = bootstrap_auc_ci(tgt_y_np, p_gcn)
                bacc_gcn = balanced_accuracy_score(tgt_y_np, pred_gcn)
                brier_gcn = brier_score_loss(tgt_y_np, p_gcn)

                records.append({
                    "experiment": "Calibration_Strategy",
                    "source_cohort": src_name,
                    "target_cohort": tgt_name,
                    "calibration": calib,
                    "gain_multiplier_k": 1.0,
                    "feature_type": f_type,
                    "auc_logistic_regression": auc_lr,
                    "ci_lower_lr": ci_lo_lr,
                    "ci_upper_lr": ci_hi_lr,
                    "bacc_lr": bacc_lr,
                    "brier_lr": brier_lr,
                    "auc_random_forest": auc_rf,
                    "ci_lower_rf": ci_lo_rf,
                    "ci_upper_rf": ci_hi_rf,
                    "bacc_rf": bacc_rf,
                    "brier_rf": brier_rf,
                    "auc_gcn_omst": auc_gcn,
                    "ci_lower_gcn": ci_lo_gcn,
                    "ci_upper_gcn": ci_hi_gcn,
                    "bacc_gcn": bacc_gcn,
                    "brier_gcn": brier_gcn,
                    "feature_abs_drift": 0.0,
                    "prob_abs_drift_lr": 0.0,
                    "prob_abs_drift_rf": 0.0,
                })

    # Experiment B: Explicit PSD Gain Scaling PSD_k(f) = k^2 * PSD(f) under Uncalibrated Transfer
    # Both Uniform Target Gain (k in {0.5, 1.0, 2.0, 5.0}) and Heterogeneous Multi-Amplifier Gain (50% k=0.5, 50% k=5.0)
    gain_settings = [0.5, 1.0, 2.0, 5.0, "Heterogeneous_0.5x_5.0x"]

    # Pre-compute scaled target tensors for each k and each site
    scaled_site_tensors = {k: {} for k in gain_settings}
    for k in gain_settings:
        for site_name, rec_list in psd_cache.items():
            spec_list, rel_list, raw_list = [], [], []
            for idx, r in enumerate(rec_list):
                if k == "Heterogeneous_0.5x_5.0x":
                    k_val = 0.5 if (idx % 2 == 0) else 5.0
                else:
                    k_val = float(k)
                scaled_psd = (k_val ** 2) * r["mean_psd"]
                x_sp, x_rl, x_rw = compute_three_representations_from_psd(r["freqs"], scaled_psd)
                spec_list.append(torch.tensor(x_sp, dtype=torch.float32))
                rel_list.append(torch.tensor(x_rl, dtype=torch.float32))
                raw_list.append(torch.tensor(x_rw, dtype=torch.float32))
            scaled_site_tensors[k][site_name] = {
                "specparam_periodic_residual": torch.stack(spec_list),
                "relative_band_power": torch.stack(rel_list),
                "raw_log_band_power": torch.stack(raw_list),
            }

    for src_name, tgt_name in pairs:
        src = site_tensors[src_name]
        tgt_base = site_tensors[tgt_name]
        src_y = src["y"]
        tgt_y_np = tgt_base["y_np"]

        for f_type in feature_types:
            X_tr = src[f_type].view(len(src_y), -1).numpy()
            X_te_base = tgt_base[f_type].view(len(tgt_y_np), -1).numpy()

            lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42).fit(X_tr, src_y.numpy())
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, src_y.numpy())

            p_lr_base = lr.predict_proba(X_te_base)[:, 1]
            p_rf_base = rf.predict_proba(X_te_base)[:, 1]

            for k in gain_settings:
                X_te_k = scaled_site_tensors[k][tgt_name][f_type].view(len(tgt_y_np), -1).numpy()
                feat_drift = float(np.mean(np.abs(X_te_k - X_te_base)))

                p_lr_k = lr.predict_proba(X_te_k)[:, 1]
                pred_lr_k = (p_lr_k >= 0.5).astype(int)
                auc_lr_k = float(roc_auc_score(tgt_y_np, p_lr_k))
                bacc_lr_k = float(balanced_accuracy_score(tgt_y_np, pred_lr_k))
                brier_lr_k = float(brier_score_loss(tgt_y_np, p_lr_k))
                prob_drift_lr = float(np.mean(np.abs(p_lr_k - p_lr_base)))

                p_rf_k = rf.predict_proba(X_te_k)[:, 1]
                pred_rf_k = (p_rf_k >= 0.5).astype(int)
                auc_rf_k = float(roc_auc_score(tgt_y_np, p_rf_k))
                bacc_rf_k = float(balanced_accuracy_score(tgt_y_np, pred_rf_k))
                brier_rf_k = float(brier_score_loss(tgt_y_np, p_rf_k))
                prob_drift_rf = float(np.mean(np.abs(p_rf_k - p_rf_base)))

                records.append({
                    "experiment": "Amplifier_Gain_Ablation",
                    "source_cohort": src_name,
                    "target_cohort": tgt_name,
                    "calibration": "Uncalibrated_Raw",
                    "gain_multiplier_k": str(k),
                    "feature_type": f_type,
                    "auc_logistic_regression": auc_lr_k,
                    "ci_lower_lr": np.nan,
                    "ci_upper_lr": np.nan,
                    "bacc_lr": bacc_lr_k,
                    "brier_lr": brier_lr_k,
                    "auc_random_forest": auc_rf_k,
                    "ci_lower_rf": np.nan,
                    "ci_upper_rf": np.nan,
                    "bacc_rf": bacc_rf_k,
                    "brier_rf": brier_rf_k,
                    "auc_gcn_omst": np.nan,
                    "ci_lower_gcn": np.nan,
                    "ci_upper_gcn": np.nan,
                    "bacc_gcn": np.nan,
                    "brier_gcn": np.nan,
                    "feature_abs_drift": feat_drift,
                    "prob_abs_drift_lr": prob_drift_lr,
                    "prob_abs_drift_rf": prob_drift_rf,
                })

    df_gain = pd.DataFrame(records)
    out_csv = REPORTS_DIR / "benchmark_gain_and_calibration_ablation.csv"
    df_gain.to_csv(out_csv, index=False)

    # Print summary of Gain Perturbation
    print("\n--- Gain Perturbation Summary (Mean across 6 PD Transfer Pairs) ---")
    sub_g = df_gain[df_gain["experiment"] == "Amplifier_Gain_Ablation"]
    for f_type in feature_types:
        print(f"\nFeature Representation: {f_type}")
        for k in gain_settings:
            rows = sub_g[(sub_g["feature_type"] == f_type) & (sub_g["gain_multiplier_k"] == str(k))]
            print(
                f"  k = {str(k):<22} | |dX| = {rows['feature_abs_drift'].mean():.4f} | "
                f"LR AUC = {rows['auc_logistic_regression'].mean():.3f}, BalAcc = {rows['bacc_lr'].mean():.3f}, Brier = {rows['brier_lr'].mean():.3f}, |dp| = {rows['prob_abs_drift_lr'].mean():.4f} | "
                f"RF AUC = {rows['auc_random_forest'].mean():.3f}, BalAcc = {rows['bacc_rf'].mean():.3f}"
            )

    print(f"\n✅ Saved gain & calibration ablation to: {out_csv}")
    return df_gain


# ==============================================================================
# 3. HEURISTIC ZERO-TRAINING BASELINE
# ==============================================================================

def run_heuristic_slowing_baseline():
    """Evaluate zero-training heuristic slowing rule (frontal theta minus alpha power)."""
    print("\n" + "=" * 75)
    print("📏 [Part 3] Running Heuristic Frontal Slowing Index Baseline (Zero-Training)...")
    print("=" * 75)

    sleep_graphs = torch.load(DATA_DIR / "cohort_sleep_ds004902_graphs.pt")
    mumtaz_ec = torch.load(DATA_DIR / "cohort_mumtaz_mdd_graphs.pt")
    mumtaz_eo_path = DATA_DIR / "cohort_mumtaz_eo_graphs.pt"
    mumtaz_eo = torch.load(mumtaz_eo_path) if mumtaz_eo_path.exists() else None
    cavanagh = torch.load(DATA_DIR / "cohort_cavanagh_mdd_graphs.pt")
    ucsd_pd = torch.load(DATA_DIR / "cohort_ucsd_parkinson_graphs.pt")
    dementia = torch.load(DATA_DIR / "cohort_dementia_ds004504_graphs.pt")
    unm_pd = torch.load(DATA_DIR / "cohort_unm_pd_ds003490_graphs.pt")
    iowa_pd = torch.load(DATA_DIR / "cohort_iowa_pd_ds004584_graphs.pt")

    contrasts = [
        ("Study 1: Sleep Deprivation", [g for g in sleep_graphs if int(g["y"]) in (0, 1)]),
        ("Study 2: Mumtaz MDD (EC)", [g for g in mumtaz_ec if int(g["y"]) in (0, 1)]),
        ("Study 3: Cavanagh Depression", [g for g in cavanagh if int(g["y"]) in (0, 1)]),
        ("Study 4: UCSD Parkinson All", [g for g in ucsd_pd if int(g["y"]) in (0, 1)]),
        ("Study 5a: Dementia AD vs CTL", [g for g in dementia if g.get("diagnosis") in ("Alzheimer", "Control")]),
        ("Study 5b: Dementia FTD vs CTL", [g for g in dementia if g.get("diagnosis") in ("FTD", "Control")]),
        ("Study 5c: Dementia All vs CTL", [g for g in dementia if int(g["y"]) in (0, 1)]),
        ("Study 6a: UNM PD-OFF vs CTL", [g for g in unm_pd if g.get("med_state") in ("OFF", "CTL")]),
        ("Study 6b: UNM PD-ON vs CTL", [g for g in unm_pd if g.get("med_state") in ("ON", "CTL")]),
        ("Study 7: Iowa Parkinson", [g for g in iowa_pd if int(g["y"]) in (0, 1)]),
    ]
    if mumtaz_eo:
        contrasts.insert(2, ("Study 2b: Mumtaz MDD (EO)", [g for g in mumtaz_eo if int(g["y"]) in (0, 1)]))

    frontal_indices = [0, 1, 2, 3, 16]  # Fp1, Fp2, F3, F4, Fz

    records = []
    for c_name, g_list in contrasts:
        y_true = np.array([int(g["y"]) for g in g_list])
        theta_frontal = np.array([g["x"][frontal_indices, 0].mean().item() for g in g_list])
        alpha_frontal = np.array([g["x"][frontal_indices, 2].mean().item() for g in g_list])

        auc_theta_raw = roc_auc_score(y_true, theta_frontal)
        auc_theta_oriented = max(auc_theta_raw, 1.0 - auc_theta_raw)

        slowing_index = theta_frontal - alpha_frontal
        auc_slowing_raw = roc_auc_score(y_true, slowing_index)
        auc_slowing_oriented = max(auc_slowing_raw, 1.0 - auc_slowing_raw)

        best_bacc = 0.5
        for th in np.percentile(slowing_index, np.linspace(10, 90, 81)):
            pred = (slowing_index >= th).astype(int)
            bacc = max(balanced_accuracy_score(y_true, pred), balanced_accuracy_score(y_true, 1 - pred))
            if bacc > best_bacc:
                best_bacc = bacc

        records.append({
            "cohort": c_name,
            "n_samples": len(g_list),
            "heuristic_rule": "Frontal_Theta_Alpha_Difference",
            "auc_raw": auc_slowing_raw,
            "auc_oriented": auc_slowing_oriented,
            "balanced_accuracy": best_bacc,
            "single_theta_auc_oriented": auc_theta_oriented,
        })

    df_heur = pd.DataFrame(records)
    out_csv = REPORTS_DIR / "benchmark_heuristic_slowing.csv"
    df_heur.to_csv(out_csv, index=False)
    print(f"✅ Saved heuristic slowing benchmark to: {out_csv}")
    return df_heur


def main():
    """Execute all leakage, representation, and gain perturbation benchmark routines."""
    t0 = time.time()
    print("🚀 Starting Audited Step 33 Benchmarks...")
    run_leakage_ablation()
    run_gain_and_calibration_ablation()
    run_heuristic_slowing_baseline()
    print(f"\n✨ Step 33 completed successfully in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
