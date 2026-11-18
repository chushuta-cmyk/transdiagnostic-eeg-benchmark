#!/usr/bin/env python3
"""
scripts/44_rebuild_unm_rest_and_diagnostics.py
----------------------------------------------
Phase 1 Data Remediation:
1. Back up legacy derivatives before fixing Study 6 (UNM ds003490).
2. Re-extract Study 6 (UNM ds003490, N=75 sessions) strictly from the contiguous
   60-second resting Eyes Open (EO) block(s) in each task-Rest recording, completely
   excluding the 60s Eyes Closed block and the ~4.5-min Auditory Oddball task
   (Standard/Novel/Target tones at ~303-575 s).
3. Extract a parallel 74-session UNM pure Eyes Closed (EC) resting sensitivity
   dataset (excluding CTL sub-005 who only has 5 s EC).
4. Recompute Welch PSDs, masked-OLS baselines, genuine specparam v2.0.0rc7 across
   BOTH [1.0, 40.0] Hz and interior [2.0, 30.0] Hz fit windows across all 14,003
   spectra (737 sessions x 19 channels), plus cohort epoch balance & Mumtaz phase
   artifact diagnostics.
"""

import shutil
import sys
import time
from pathlib import Path
from typing import List, Tuple

import mne
import numpy as np
import pandas as pd
import torch
from scipy import signal, stats
from specparam import SpectralGroupModel

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = BENCHMARK_ROOT / "src"
sys.path.append(str(SRC_DIR))

from harmonizer_and_graph import (
    CHANNELS_19,
    TopologicalGraphBuilder,
    extract_real_19ch_features_and_pli,
)
import multi_cohort_loader as mcl

DATA_DIR = BENCHMARK_ROOT / "data" / "derivatives"
BACKUP_DIR = DATA_DIR / "legacy_backup_nov14"
REPORTS_DIR = BENCHMARK_ROOT / "reports"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

UNM_DIR = Path("/Users/tan/Documents/eeg_datastes/ds003490")
MUMTAZ_DIR = Path("/Users/tan/Downloads/4244171")

SPECPARAM_KWARGS = {
    "aperiodic_mode": "fixed",
    "periodic_mode": "gaussian",
    "peak_width_limits": [1.0, 12.0],
    "max_n_peaks": 6,
    "min_peak_height": 0.05,
    "peak_threshold": 2.0,
    "metrics": ["error_mae", "gof_rsquared"],
    "verbose": False,
}


def get_contiguous_blocks(df_ev: pd.DataFrame, key: str) -> List[Tuple[float, float]]:
    onsets = (
        df_ev[df_ev["trial_type"].astype(str).str.contains(key, na=False)]["onset"]
        .astype(float)
        .sort_values()
        .values
    )
    if len(onsets) == 0:
        return []
    blocks = []
    s = prev = float(onsets[0])
    for t in onsets[1:]:
        t = float(t)
        if t - prev > 3.0:
            blocks.append((s, prev + 1.0))
            s = t
        prev = t
    blocks.append((s, prev + 1.0))
    return blocks


def compute_masked_ols_from_psd(
    freqs: np.ndarray, mean_psd: np.ndarray, f_low: float = 1.0, f_high: float = 40.0
) -> np.ndarray:
    valid_mask = (freqs >= f_low) & (freqs <= f_high)
    f_valid = freqs[valid_mask]
    log_f = np.log10(f_valid)
    aperiodic_fit_mask = (f_valid < 7.0) | (f_valid > 14.0)
    log_f_fit = log_f[aperiodic_fit_mask]

    theta_mask = (f_valid >= 4.0) & (f_valid < 8.0)
    delta_mask = (f_valid >= f_low) & (f_valid < 4.0)
    alpha_mask = (f_valid >= 8.0) & (f_valid <= 13.0)

    x_mat = np.zeros((19, 4), dtype=np.float32)
    for c in range(19):
        log_p = np.log10(np.maximum(mean_psd[c, valid_mask], 1e-25))
        slope, intercept = np.polyfit(log_f_fit, log_p[aperiodic_fit_mask], 1)
        flattened = log_p - (intercept + slope * log_f)
        x_mat[c, 0] = float(np.mean(flattened[theta_mask]))
        x_mat[c, 1] = float(np.mean(flattened[delta_mask]))
        x_mat[c, 2] = float(np.mean(flattened[alpha_mask]))
        x_mat[c, 3] = float(-slope)
    return x_mat


def fit_specparam_window(
    freqs: np.ndarray, psds_flat: np.ndarray, f_low: float, f_high: float
):
    sgm = SpectralGroupModel(**SPECPARAM_KWARGS)
    sgm.fit(freqs, psds_flat, [f_low, f_high], n_jobs=1)
    f_fit = sgm.data.freqs
    theta_mask = (f_fit >= 4.0) & (f_fit < 8.0)
    delta_mask = (f_fit >= f_low) & (f_fit < 4.0)
    alpha_mask = (f_fit >= 8.0) & (f_fit <= 13.0)

    ap_params = sgm.get_params("aperiodic")
    r2_all = sgm.get_metrics("gof", "rsquared")
    mae_all = sgm.get_metrics("error", "mae")
    null_set = set(sgm.results.null_inds or [])
    return sgm, f_fit, theta_mask, delta_mask, alpha_mask, ap_params, r2_all, mae_all, null_set


def main():
    t0 = time.time()
    for fname in [
        "cohort_unm_pd_ds003490_graphs.pt",
        "all_7cohorts_welch_psds.pt",
        "specparam_vs_masked_ols_7cohorts.pt",
    ]:
        src = DATA_DIR / fname
        dst = BACKUP_DIR / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"Backed up {fname} -> {dst}")

    # ------------------------------------------------------------------
    # Step 1: Rebuild Study 6 (UNM ds003490) purely from resting Eyes Open
    #         (and extract 74-session Eyes Closed resting sensitivity)
    # ------------------------------------------------------------------
    print("\n=== Step 1: Re-extracting Study 6 UNM ds003490 purely from resting blocks ===")
    part = pd.read_csv(UNM_DIR / "participants.tsv", sep="\t").set_index("participant_id")
    adapter = mcl.Universal19ChannelAdapter(
        target_sfreq=250.0, l_freq=1.0, h_freq=40.0, epoch_duration=4.0, ptp_threshold_uv=150.0
    )
    graph_builder = TopologicalGraphBuilder()

    unm_eo_graphs, unm_eo_psds = [], []
    unm_ec_graphs, unm_ec_psds = [], []

    for sub_id in sorted(part.index):
        row = part.loc[sub_id]
        group = str(row["Group"])
        sex = row["sex"]
        age = float(row["age"]) if pd.notnull(row["age"]) else np.nan
        ses_list = (
            [("ses-01", str(row["sess1_Med"]).strip()), ("ses-02", str(row["sess2_Med"]).strip())]
            if group == "PD"
            else [("ses-01", "CTL")]
        )
        y_pd = 1 if group == "PD" else 0

        for ses_name, med_state in ses_list:
            set_files = sorted((UNM_DIR / sub_id / ses_name / "eeg").glob("*task-Rest*.set"))
            ev_files = sorted((UNM_DIR / sub_id / ses_name / "eeg").glob("*events.tsv"))
            if not set_files or not ev_files:
                continue
            df_ev = pd.read_csv(ev_files[0], sep="\t")
            eo_blocks = get_contiguous_blocks(df_ev, "Eyes Open")
            ec_blocks = get_contiguous_blocks(df_ev, "Eyes Closed")

            raw = mne.io.read_raw_eeglab(str(set_files[0]), preload=True, verbose=False)

            # Pure Eyes Open (first 60s contiguous resting block -> balanced 60s across all 75 sessions)
            t_a, t_b = eo_blocks[0]
            raw_eo = raw.copy().crop(tmin=t_a, tmax=min(t_b, float(raw.times[-1])))
            data_eo, sfreq_eo = adapter.standardize_raw_to_epochs(raw_eo)
            nperseg = min(int(sfreq_eo * 2.0), data_eo.shape[-1])
            freqs_eo, psd_eo = signal.welch(data_eo, fs=sfreq_eo, nperseg=nperseg, axis=-1)
            mean_psd_eo = np.mean(psd_eo, axis=0).astype(np.float64)
            x_ols_eo, pli_eo = extract_real_19ch_features_and_pli(data_eo, sfreq_eo)
            adj_b, adj_w, opt_m, max_gce = graph_builder.build_omst_graph(pli_eo)
            rec_id = f"{sub_id}_{ses_name}" if group == "PD" else sub_id

            g_rec = {
                "x": torch.tensor(x_ols_eo, dtype=torch.float32),
                "adj": torch.tensor(adj_b, dtype=torch.float32),
                "adj_weighted": torch.tensor(adj_w, dtype=torch.float32),
                "pli": torch.tensor(pli_eo, dtype=torch.float32),
                "y": torch.tensor(y_pd, dtype=torch.long),
                "y_med": (1 if med_state == "ON" else 0) if group == "PD" else -1,
                "group": group,
                "med_state": med_state,
                "session": ses_name,
                "age": age,
                "sex": sex,
                "subject_id": sub_id,
                "bids_id": sub_id,
                "session_uid": rec_id,
                "cohort": "UNM_PD_ds003490",
                "domain": "Movement_Disorders_PD",
                "n_clean_epochs": int(data_eo.shape[0]),
                "clean_duration_s": float(data_eo.shape[0] * 4.0),
                "omst_trees": int(opt_m),
                "gce": float(max_gce),
                "rest_condition": "Pure_EO_60s",
            }
            unm_eo_graphs.append(g_rec)
            p_rec = {
                "cohort_key": "unm_ds003490",
                "subject_id": sub_id,
                "bids_id": sub_id,
                "session_uid": rec_id,
                "freqs": freqs_eo.astype(np.float64),
                "mean_psd": mean_psd_eo,
                "y": int(y_pd),
                "x_masked_ols": torch.tensor(x_ols_eo, dtype=torch.float32),
                "adj": g_rec["adj"],
                "pli": g_rec["pli"],
                "gce": float(max_gce),
                "omst_trees": int(opt_m),
                "ols_max_abs_diff": float(np.max(np.abs(compute_masked_ols_from_psd(freqs_eo, mean_psd_eo) - x_ols_eo))),
                "session": ses_name,
                "group": group,
                "med_state": med_state,
                "y_med": g_rec["y_med"],
                "age": age,
                "sex": sex,
                "n_clean_epochs": int(data_eo.shape[0]),
                "rest_condition": "Pure_EO_60s",
            }
            unm_eo_psds.append(p_rec)

            # Pure Eyes Closed sensitivity (>= 50s contiguous EC markers; excludes sub-005 5s)
            if ec_blocks and (ec_blocks[0][1] - ec_blocks[0][0]) >= 50.0:
                ec_a, ec_b = ec_blocks[0]
                raw_ec = raw.copy().crop(tmin=ec_a, tmax=min(ec_b, float(raw.times[-1])))
                data_ec, sfreq_ec = adapter.standardize_raw_to_epochs(raw_ec)
                freqs_ec, psd_ec = signal.welch(data_ec, fs=sfreq_ec, nperseg=min(int(sfreq_ec * 2.0), data_ec.shape[-1]), axis=-1)
                mean_psd_ec = np.mean(psd_ec, axis=0).astype(np.float64)
                x_ols_ec, pli_ec = extract_real_19ch_features_and_pli(data_ec, sfreq_ec)
                adj_b_ec, _, opt_m_ec, max_gce_ec = graph_builder.build_omst_graph(pli_ec)
                unm_ec_psds.append({
                    "cohort_key": "unm_ec_sensitivity",
                    "subject_id": sub_id,
                    "bids_id": sub_id,
                    "session_uid": rec_id,
                    "freqs": freqs_ec.astype(np.float64),
                    "mean_psd": mean_psd_ec,
                    "y": int(y_pd),
                    "x_masked_ols": torch.tensor(x_ols_ec, dtype=torch.float32),
                    "adj": torch.tensor(adj_b_ec, dtype=torch.float32),
                    "pli": torch.tensor(pli_ec, dtype=torch.float32),
                    "gce": float(max_gce_ec),
                    "omst_trees": int(opt_m_ec),
                    "session": ses_name,
                    "group": group,
                    "med_state": med_state,
                    "age": age,
                    "sex": sex,
                    "n_clean_epochs": int(data_ec.shape[0]),
                    "rest_condition": "Pure_EC_60s",
                })

    torch.save(unm_eo_graphs, DATA_DIR / "cohort_unm_pd_ds003490_graphs.pt")
    torch.save(unm_ec_psds, DATA_DIR / "unm_ec_sensitivity_psds.pt")

    pd_ep = [r["n_clean_epochs"] for r in unm_eo_psds if r["y"] == 1]
    ctl_ep = [r["n_clean_epochs"] for r in unm_eo_psds if r["y"] == 0]
    print(
        f"  ✓ UNM EO rebuilt (N={len(unm_eo_psds)}): PD={np.mean(pd_ep):.2f}±{np.std(pd_ep,ddof=1):.2f} "
        f"vs CTL={np.mean(ctl_ep):.2f}±{np.std(ctl_ep,ddof=1):.2f} (Welch p={stats.ttest_ind(pd_ep, ctl_ep, equal_var=False)[1]:.4f})"
    )

    # Update canonical all_7cohorts_welch_psds.pt
    all_psds = torch.load(DATA_DIR / "all_7cohorts_welch_psds.pt", weights_only=False)
    all_psds["unm_ds003490"] = unm_eo_psds
    torch.save(all_psds, DATA_DIR / "all_7cohorts_welch_psds.pt")

    # Also ensure sleep & others have n_clean_epochs if present in graph files
    # ------------------------------------------------------------------
    # Step 2: Fit specparam over BOTH [1.0, 40.0] Hz and interior [2.0, 30.0] Hz
    # ------------------------------------------------------------------
    print("\n=== Step 2: Running specparam v2.0.0rc7 on [1.0, 40.0] Hz AND [2.0, 30.0] Hz ===")
    out_1_40 = {}
    out_2_30 = {}
    diag_rows = []

    for c_key, recs in all_psds.items():
        freqs = recs[0]["freqs"]
        psds_flat = np.vstack([np.maximum(r["mean_psd"], 1e-25) for r in recs])

        for label_win, fl, fh, target_dict in [
            ("1.0-40.0", 1.0, 40.0, out_1_40),
            ("2.0-30.0", 2.0, 30.0, out_2_30),
        ]:
            sgm, f_fit, th_m, dl_m, al_m, aps, r2s, maes, null_set = fit_specparam_window(freqs, psds_flat, fl, fh)
            valid_f = (freqs >= fl) & (freqs <= fh)
            cohort_recs = []
            r2_l, mae_l, npk_l, ok_l, chi_ols_l, chi_sp_l = [], [], [], [], [], []
            for s_idx, rec in enumerate(recs):
                x_sp_res = np.zeros((19, 4), dtype=np.float32)
                x_sp_pk = np.zeros((19, 4), dtype=np.float32)
                x_ols_win = compute_masked_ols_from_psd(freqs, rec["mean_psd"], f_low=fl, f_high=fh)
                ch_r2 = np.zeros(19, dtype=np.float32)
                ch_mae = np.zeros(19, dtype=np.float32)
                ch_off = np.zeros(19, dtype=np.float32)
                ch_npk = np.zeros(19, dtype=np.int32)
                ch_ok = np.ones(19, dtype=bool)
                for ch in range(19):
                    g_i = s_idx * 19 + ch
                    if g_i in null_set or np.isnan(aps[g_i, 1]):
                        ch_ok[ch] = False
                        continue
                    m_res = sgm.get_model(g_i, regenerate=True)
                    lp = m_res.data.power_spectrum
                    ap_line = m_res.results.model._ap_fit
                    pk_comp = m_res.results.model._peak_fit
                    resid = lp - ap_line
                    x_sp_res[ch, 0] = float(np.mean(resid[th_m]))
                    x_sp_res[ch, 1] = float(np.mean(resid[dl_m]))
                    x_sp_res[ch, 2] = float(np.mean(resid[al_m]))
                    x_sp_res[ch, 3] = float(aps[g_i, 1])

                    x_sp_pk[ch, 0] = float(np.mean(pk_comp[th_m]))
                    x_sp_pk[ch, 1] = float(np.mean(pk_comp[dl_m]))
                    x_sp_pk[ch, 2] = float(np.mean(pk_comp[al_m]))
                    x_sp_pk[ch, 3] = float(aps[g_i, 1])

                    ch_off[ch] = float(aps[g_i, 0])
                    ch_r2[ch] = float(r2s[g_i])
                    ch_mae[ch] = float(maes[g_i])
                    n_pk = int(m_res.results.n_peaks)
                    ch_npk[ch] = n_pk

                    r2_l.append(float(r2s[g_i]))
                    mae_l.append(float(maes[g_i]))
                    npk_l.append(n_pk)
                    ok_l.append(True)
                    chi_ols_l.append(float(x_ols_win[ch, 3]))
                    chi_sp_l.append(float(aps[g_i, 1]))

                new_rec = {
                    k: v
                    for k, v in rec.items()
                    if k not in ("freqs", "mean_psd")
                }
                new_rec["x_masked_ols_win"] = torch.tensor(x_ols_win, dtype=torch.float32)
                new_rec["x_specparam_resid"] = torch.tensor(x_sp_res, dtype=torch.float32)
                new_rec["x_specparam_peaks"] = torch.tensor(x_sp_pk, dtype=torch.float32)
                new_rec["sp_r2"] = ch_r2
                new_rec["sp_mae"] = ch_mae
                new_rec["sp_offset"] = ch_off
                new_rec["sp_n_peaks"] = ch_npk
                new_rec["sp_fit_ok"] = ch_ok
                cohort_recs.append(new_rec)

            target_dict[c_key] = cohort_recs
            r2_arr = np.asarray(r2_l)
            mae_arr = np.asarray(mae_l)
            diag_rows.append({
                "fit_window_hz": label_win,
                "cohort_key": c_key,
                "n_sessions": len(recs),
                "n_spectra": len(recs) * 19,
                "n_failed": int(len(null_set)),
                "pct_r2_lt_080": float(100.0 * np.mean(r2_arr < 0.80)),
                "r2_p05": float(np.percentile(r2_arr, 5)),
                "r2_median": float(np.median(r2_arr)),
                "r2_mean": float(np.mean(r2_arr)),
                "mae_median": float(np.median(mae_arr)),
                "mae_mean": float(np.mean(mae_arr)),
                "mean_n_peaks": float(np.mean(npk_l)),
                "mean_chi_ols": float(np.mean(chi_ols_l)),
                "mean_chi_specparam": float(np.mean(chi_sp_l)),
            })

    # Fit UNM EC sensitivity across [2.0, 30.0] Hz and [1.0, 40.0] Hz too
    freqs_ec = unm_ec_psds[0]["freqs"]
    psds_ec_flat = np.vstack([np.maximum(r["mean_psd"], 1e-25) for r in unm_ec_psds])
    unm_ec_2_30 = {}
    for label_win, fl, fh in [("1.0-40.0", 1.0, 40.0), ("2.0-30.0", 2.0, 30.0)]:
        sgm, f_fit, th_m, dl_m, al_m, aps, r2s, maes, null_set = fit_specparam_window(freqs_ec, psds_ec_flat, fl, fh)
        valid_f = (freqs_ec >= fl) & (freqs_ec <= fh)
        ec_list = []
        for s_idx, rec in enumerate(unm_ec_psds):
            x_sp_res = np.zeros((19, 4), dtype=np.float32)
            for ch in range(19):
                g_i = s_idx * 19 + ch
                m_res = sgm.get_model(g_i, regenerate=True)
                lp = m_res.data.power_spectrum
                ap_line = m_res.results.model._ap_fit
                resid = lp - ap_line
                x_sp_res[ch, 0] = float(np.mean(resid[th_m]))
                x_sp_res[ch, 1] = float(np.mean(resid[dl_m]))
                x_sp_res[ch, 2] = float(np.mean(resid[al_m]))
                x_sp_res[ch, 3] = float(aps[g_i, 1])
            nr = {k: v for k, v in rec.items() if k not in ("freqs", "mean_psd")}
            nr["x_specparam_resid"] = torch.tensor(x_sp_res, dtype=torch.float32)
            ec_list.append(nr)
        unm_ec_2_30[label_win] = ec_list
    torch.save(unm_ec_2_30, DATA_DIR / "unm_ec_sensitivity_specparam.pt")

    # Save [2.0, 30.0] Hz as primary specparam bundle and preserve [1.0, 40.0] Hz bundle for sensitivity audit!
    torch.save(out_2_30, DATA_DIR / "specparam_vs_masked_ols_7cohorts.pt")
    torch.save(out_1_40, DATA_DIR / "specparam_1to40_sensitivity_7cohorts.pt")
    df_diag = pd.DataFrame(diag_rows)
    df_diag.to_csv(REPORTS_DIR / "specparam_fit_diagnostics_by_cohort.csv", index=False)
    print("✓ Saved [2.0, 30.0] Hz (primary) and [1.0, 40.0] Hz (edge-sensitivity) bundles + diagnostics.")
    print(df_diag.to_string(index=False))

    # ------------------------------------------------------------------
    # Step 3: Epoch-count balance audit across ALL cohorts + Mumtaz batch check
    # ------------------------------------------------------------------
    print("\n=== Step 3: Clean Epoch Balance & Mumtaz Batch / Indefinite PLI Diagnostic ===")
    epoch_rows = []
    for c_lbl, c_key in [
        ("Study 1: Sleep Deprivation (ds004902)", "sleep_ds004902"),
        ("Study 2a: Outpatient MDD EC (4244171)", "mumtaz_ec"),
        ("Study 2b: Outpatient MDD EO (4244171)", "mumtaz_eo"),
        ("Study 3: Univ. Depression (ds003478)", "cavanagh_ds003478"),
        ("Study 4: UCSD PD ON/OFF (ds002778)", "ucsd_ds002778"),
        ("Study 5: AHEPA Dementia (ds004504)", "dementia_ds004504"),
        ("Study 6: UNM PD Pure EO (ds003490)", "unm_ds003490"),
        ("Study 7: Iowa PD (ds004584)", "iowa_ds004584"),
    ]:
        recs = out_2_30[c_key]
        ep0 = [float(r["n_clean_epochs"]) for r in recs if int(r["y"]) == 0]
        ep1 = [float(r["n_clean_epochs"]) for r in recs if int(r["y"]) == 1]
        all_ep = [float(r["n_clean_epochs"]) for r in recs]
        if c_key == "sleep_ds004902":
            _, p_ep = stats.ttest_rel(ep1, ep0)
        else:
            _, p_ep = stats.ttest_ind(ep1, ep0, equal_var=False)
        epoch_rows.append({
            "cohort": c_lbl,
            "n_sessions": len(recs),
            "all_mean": float(np.mean(all_ep)),
            "all_sd": float(np.std(all_ep, ddof=1)),
            "all_min": int(np.min(all_ep)),
            "all_max": int(np.max(all_ep)),
            "case_mean": float(np.mean(ep1)),
            "case_sd": float(np.std(ep1, ddof=1)),
            "ctrl_mean": float(np.mean(ep0)),
            "ctrl_sd": float(np.std(ep0, ddof=1)),
            "case_vs_ctrl_p": float(p_ep),
        })
    df_ep = pd.DataFrame(epoch_rows)
    df_ep.to_csv(REPORTS_DIR / "cohort_epoch_duration_audit.csv", index=False)
    print(df_ep.to_string(index=False))

    # Mumtaz per-subject PLI & recording-date batch breakdown
    mum_rows = []
    for g in out_2_30["mumtaz_ec"]:
        pm = g["pli"].numpy().astype(np.float64)
        C = 0.5 * (pm + pm.T)
        np.fill_diagonal(C, 1.0)
        ev = np.linalg.eigvalsh(C)
        # find matching raw EDF meas_date
        grp_tag = "H" if int(g["y"]) == 0 else "MDD"
        num = int(g["subject_id"].split("_")[-1])
        cands = list(MUMTAZ_DIR.glob(f"{grp_tag} S{num} *EC*.edf"))
        mdate = ""
        if cands:
            raw_m = mne.io.read_raw_edf(str(cands[0]), preload=False, verbose=False)
            mdate = str(raw_m.info["meas_date"])[:10]
        mum_rows.append({
            "subject_id": g["subject_id"],
            "group": grp_tag,
            "meas_date": mdate,
            "n_clean_epochs": int(g["n_clean_epochs"]),
            "min_eig_I_plus_PLI": float(ev.min()),
            "is_indefinite": bool(ev.min() <= 0.0),
            "median_pli": float(np.median(pm[np.triu_indices(19, k=1)])),
        })
    df_mum = pd.DataFrame(mum_rows)
    df_mum.to_csv(REPORTS_DIR / "mumtaz_batch_and_pli_diagnostic.csv", index=False)
    for grp_tag in ("H", "MDD"):
        sub = df_mum[df_mum["group"] == grp_tag]
        print(
            f"  Mumtaz {grp_tag}: {sub['is_indefinite'].sum()}/{len(sub)} indefinite, "
            f"median_PLI={sub['median_pli'].mean():.3f} (max={sub['median_pli'].max():.3f}), "
            f"dates=[{sub['meas_date'].min()} .. {sub['meas_date'].max()}]"
        )

    print(f"\n✅ Phase 1 complete in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
