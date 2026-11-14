#!/usr/bin/env python3
"""
Step 39: Extract Real Welch PSDs Across All 7 Cohorts (N = 677 Primary + 60 Mumtaz EO = 737 Sessions)
and Execute Genuine specparam (v2.0.0rc7) Spectral Parameterization vs. [7, 14] Hz-Masked OLS.

Verifies bit-exact reproducibility (max |diff| < 1e-5) against cached cohort_*_graphs.pt 'x'
tensors when applying [7, 14] Hz-masked log-linear OLS, then runs genuine specparam
(aperiodic_mode='fixed', peak_width_limits=[1.0, 12.0], max_n_peaks=6, min_peak_height=0.05,
peak_threshold=2.0, freq_range=[1.0, 40.0]) on all 737 x 19 = 14,003 channel spectra.

Outputs:
  - data/derivatives/all_7cohorts_welch_psds.pt
  - data/derivatives/specparam_vs_masked_ols_7cohorts.pt
  - reports/specparam_fit_diagnostics_by_cohort.csv
  - reports/specparam_gain_invariance_check.csv
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np
import pandas as pd
import torch
from scipy import signal
import specparam
from specparam import SpectralGroupModel

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = BENCHMARK_ROOT / "src"
sys.path.append(str(SRC_DIR))
from harmonizer_and_graph import CHANNELS_19
import multi_cohort_loader as mcl

DATA_DIR = BENCHMARK_ROOT / "data" / "derivatives"
REPORTS_DIR = BENCHMARK_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MUMTAZ_DIR = Path("/Users/tan/Downloads/4244171")
EEG_ROOT = Path("/Users/tan/Documents/eeg_datastes")
CAVANAGH_DIR = EEG_ROOT / "depression_new"
DEMENTIA_DIR = EEG_ROOT / "ds004504"
UNM_DIR = EEG_ROOT / "ds003490"
IOWA_DIR = EEG_ROOT / "ds004584"
SLEEP_FIF_DIR = Path(
    "/Users/tan/Documents/qwen-agent/aCAgqOVe0x/default/"
    "sleep_deprivation_eeg/data/derivatives/preprocessed_v3_conservative"
)

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


def compute_masked_ols_from_psd(freqs: np.ndarray, mean_psd: np.ndarray) -> np.ndarray:
    """Reproduce exact [7, 14] Hz-masked log-linear OLS baseline from [19, n_freqs] Welch PSD."""
    valid_mask = (freqs >= 1.0) & (freqs <= 40.0)
    f_valid = freqs[valid_mask]
    log_f = np.log10(f_valid)
    aperiodic_fit_mask = (f_valid < 7.0) | (f_valid > 14.0)
    log_f_fit = log_f[aperiodic_fit_mask]

    theta_mask = (f_valid >= 4.0) & (f_valid < 8.0)
    delta_mask = (f_valid >= 1.0) & (f_valid < 4.0)
    alpha_mask = (f_valid >= 8.0) & (f_valid <= 13.0)

    x_mat = np.zeros((19, 4), dtype=np.float32)
    for c in range(19):
        log_p = np.log10(np.maximum(mean_psd[c, valid_mask], 1e-15))
        slope, intercept = np.polyfit(log_f_fit, log_p[aperiodic_fit_mask], 1)
        flattened = log_p - (intercept + slope * log_f)
        x_mat[c, 0] = float(np.mean(flattened[theta_mask]))
        x_mat[c, 1] = float(np.mean(flattened[delta_mask]))
        x_mat[c, 2] = float(np.mean(flattened[alpha_mask]))
        x_mat[c, 3] = float(-slope)
    return x_mat


def welch_from_epochs(epochs_data: np.ndarray, sfreq: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute channel-wise epoch-averaged Welch PSD identical to harmonizer_and_graph."""
    nperseg = min(int(sfreq * 2.0), epochs_data.shape[-1])
    freqs, psd = signal.welch(epochs_data, fs=sfreq, nperseg=nperseg, axis=-1)
    return freqs.astype(np.float64), np.mean(psd, axis=0).astype(np.float64)


def extract_or_load_all_welch_psds(force_extract: bool = False) -> Dict[str, List[Dict]]:
    """Extract (or load cached) Welch PSDs for all 737 sessions with bit-exact OLS audit."""
    cache_path = DATA_DIR / "all_7cohorts_welch_psds.pt"
    if cache_path.exists() and not force_extract:
        print(f"Loading existing full 7-cohort Welch PSD cache: {cache_path}")
        return torch.load(cache_path, weights_only=False)

    t0 = time.time()
    adapter_150 = mcl.Universal19ChannelAdapter(
        target_sfreq=250.0, l_freq=1.0, h_freq=40.0, epoch_duration=4.0, ptp_threshold_uv=150.0
    )

    pd_cache_path = DATA_DIR / "pd_tricenter_welch_psds.pt"
    pd_cache = torch.load(pd_cache_path, weights_only=False) if pd_cache_path.exists() else {}

    sleep_gs = torch.load(DATA_DIR / "cohort_sleep_ds004902_graphs.pt", weights_only=False)
    mum_ec_gs = torch.load(DATA_DIR / "cohort_mumtaz_mdd_graphs.pt", weights_only=False)
    mum_eo_gs = torch.load(DATA_DIR / "cohort_mumtaz_eo_graphs.pt", weights_only=False)
    cav_gs = torch.load(DATA_DIR / "cohort_cavanagh_mdd_graphs.pt", weights_only=False)
    ucsd_gs = torch.load(DATA_DIR / "cohort_ucsd_parkinson_graphs.pt", weights_only=False)
    dem_gs = torch.load(DATA_DIR / "cohort_dementia_ds004504_graphs.pt", weights_only=False)
    unm_gs = torch.load(DATA_DIR / "cohort_unm_pd_ds003490_graphs.pt", weights_only=False)
    iowa_gs = torch.load(DATA_DIR / "cohort_iowa_pd_ds004584_graphs.pt", weights_only=False)

    def make_record(g: dict, freqs: np.ndarray, mean_psd: np.ndarray, cohort_key: str) -> dict:
        x_ols = compute_masked_ols_from_psd(freqs, mean_psd)
        diff = float(np.max(np.abs(x_ols - g["x"].numpy())))
        if diff > 1e-5:
            raise RuntimeError(
                f"OLS mismatch > 1e-5 in {cohort_key} / {g['subject_id']} "
                f"(max |diff| = {diff:.6e})"
            )
        rec = {
            "cohort_key": cohort_key,
            "subject_id": g["subject_id"],
            "freqs": freqs,
            "mean_psd": mean_psd,
            "y": int(g["y"]),
            "x_masked_ols": torch.tensor(x_ols, dtype=torch.float32),
            "adj": g["adj"],
            "pli": g["pli"],
            "gce": float(g.get("gce", 0.0)),
            "omst_trees": int(g.get("omst_trees", 1)),
            "ols_max_abs_diff": diff,
        }
        for k in [
            "sub_name", "bids_id", "session", "group", "condition",
            "diagnosis", "group_code", "med_state", "y_med", "age",
            "gender", "sex", "mmse", "moca", "updrs", "bdi", "n_clean_epochs"
        ]:
            if k in g:
                rec[k] = g[k]
        return rec

    all_psds: Dict[str, List[Dict]] = {}

    # 1. Sleep ds004902 (142 sessions)
    print("\n[1/8] Extracting Sleep ds004902 (N=142)...")
    sleep_map = {m["subject_id"]: m["path"] for m in mcl.get_sleep_deprivation_manifest(SLEEP_FIF_DIR)}
    recs = []
    for g in sleep_gs:
        ep = mne.read_epochs(str(sleep_map[g["subject_id"]]), preload=True, verbose=False).pick(CHANNELS_19)
        freqs, mean_psd = welch_from_epochs(ep.get_data(copy=True), float(ep.info["sfreq"]))
        recs.append(make_record(g, freqs, mean_psd, "sleep_ds004902"))
    all_psds["sleep_ds004902"] = recs
    print(f"  ✓ Sleep ds004902: {len(recs)} sessions, max OLS diff={max(r['ols_max_abs_diff'] for r in recs):.2e}")

    # 2. Mumtaz EC (58 sessions)
    print("[2/8] Extracting Mumtaz MDD EC (N=58)...")
    mum_ec_map = {m["subject_id"]: m["path"] for m in mcl.get_mumtaz_mdd_manifest(MUMTAZ_DIR, state="EC")}
    recs = []
    for g in mum_ec_gs:
        raw = mne.io.read_raw_edf(str(mum_ec_map[g["subject_id"]]), preload=True, verbose=False)
        data, sfreq = adapter_150.standardize_raw_to_epochs(raw)
        freqs, mean_psd = welch_from_epochs(data, sfreq)
        recs.append(make_record(g, freqs, mean_psd, "mumtaz_ec"))
    all_psds["mumtaz_ec"] = recs
    print(f"  ✓ Mumtaz EC: {len(recs)} sessions, max OLS diff={max(r['ols_max_abs_diff'] for r in recs):.2e}")

    # 3. Mumtaz EO (60 sessions)
    print("[3/8] Extracting Mumtaz MDD EO (N=60)...")
    mum_eo_map = {m["subject_id"]: m["path"] for m in mcl.get_mumtaz_mdd_manifest(MUMTAZ_DIR, state="EO")}
    recs = []
    for g in mum_eo_gs:
        raw = mne.io.read_raw_edf(str(mum_eo_map[g["subject_id"]]), preload=True, verbose=False)
        data, sfreq = adapter_150.standardize_raw_to_epochs(raw)
        freqs, mean_psd = welch_from_epochs(data, sfreq)
        recs.append(make_record(g, freqs, mean_psd, "mumtaz_eo"))
    all_psds["mumtaz_eo"] = recs
    print(f"  ✓ Mumtaz EO: {len(recs)} sessions, max OLS diff={max(r['ols_max_abs_diff'] for r in recs):.2e}")

    # 4. Cavanagh ds003478 (119 sessions)
    print("[4/8] Extracting Cavanagh ds003478 (N=119)...")
    cav_map = {m["subject_id"]: m["path"] for m in mcl.get_cavanagh_depression_manifest(CAVANAGH_DIR)}
    recs = []
    for i, g in enumerate(cav_gs):
        raw = mne.io.read_raw_eeglab(str(cav_map[g["subject_id"]]), preload=True, verbose=False)
        data, sfreq = adapter_150.standardize_raw_to_epochs(raw)
        freqs, mean_psd = welch_from_epochs(data, sfreq)
        recs.append(make_record(g, freqs, mean_psd, "cavanagh_ds003478"))
        if (i + 1) % 30 == 0 or (i + 1) == len(cav_gs):
            print(f"    [{i+1}/{len(cav_gs)}] elapsed {time.time()-t0:.1f}s")
    all_psds["cavanagh_ds003478"] = recs
    print(f"  ✓ Cavanagh: {len(recs)} sessions, max OLS diff={max(r['ols_max_abs_diff'] for r in recs):.2e}")

    # 5. UCSD PD ds002778 (46 sessions; reuse verified pd_cache if present)
    print("[5/8] Loading/Verifying UCSD PD ds002778 (N=46)...")
    ucsd_cached = {r["subject_id"]: r for r in pd_cache.get("UCSD", [])}
    ucsd_manifest = {m["subject_id"]: m["path"] for m in mcl.get_ucsd_parkinson_manifest(EEG_ROOT)}
    recs = []
    for g in ucsd_gs:
        sid = g["subject_id"]
        if sid in ucsd_cached:
            freqs, mean_psd = ucsd_cached[sid]["freqs"], ucsd_cached[sid]["mean_psd"]
        else:
            raw = mne.io.read_raw_bdf(str(ucsd_manifest[sid]), preload=True, verbose=False)
            data, sfreq = adapter_150.standardize_raw_to_epochs(raw)
            freqs, mean_psd = welch_from_epochs(data, sfreq)
        recs.append(make_record(g, freqs, mean_psd, "ucsd_ds002778"))
    all_psds["ucsd_ds002778"] = recs
    print(f"  ✓ UCSD: {len(recs)} sessions, max OLS diff={max(r['ols_max_abs_diff'] for r in recs):.2e}")

    # 6. Dementia ds004504 (88 sessions)
    print("[6/8] Extracting Dementia ds004504 (N=88)...")
    recs = []
    for i, g in enumerate(dem_gs):
        sid = g["subject_id"]
        set_files = list((DEMENTIA_DIR / sid / "eeg").glob("*task-eyesclosed*_eeg.set"))
        if not set_files:
            set_files = list((DEMENTIA_DIR / sid / "eeg").glob("*.set"))
        raw = mne.io.read_raw_eeglab(str(set_files[0]), preload=True, verbose=False)
        data, sfreq = adapter_150.standardize_raw_to_epochs(raw)
        freqs, mean_psd = welch_from_epochs(data, sfreq)
        recs.append(make_record(g, freqs, mean_psd, "dementia_ds004504"))
        if (i + 1) % 30 == 0 or (i + 1) == len(dem_gs):
            print(f"    [{i+1}/{len(dem_gs)}] elapsed {time.time()-t0:.1f}s")
    all_psds["dementia_ds004504"] = recs
    print(f"  ✓ Dementia: {len(recs)} sessions, max OLS diff={max(r['ols_max_abs_diff'] for r in recs):.2e}")

    # 7. UNM PD ds003490 (75 sessions: 50 OFF+CTL reused from pd_cache + 25 ON extracted)
    print("[7/8] Extracting/Verifying UNM PD ds003490 (N=75)...")
    unm_off_cached = {r["subject_id"]: r for r in pd_cache.get("UNM_OFF", [])}
    recs = []
    for g in unm_gs:
        sid = g["subject_id"]
        ses = g["session"]
        med = g["med_state"]
        if med in ("OFF", "CTL") and sid in unm_off_cached:
            freqs, mean_psd = unm_off_cached[sid]["freqs"], unm_off_cached[sid]["mean_psd"]
        else:
            set_files = list((UNM_DIR / sid / ses / "eeg").glob("*task-Rest*.set"))
            raw = mne.io.read_raw_eeglab(str(set_files[0]), preload=True, verbose=False)
            data, sfreq = adapter_150.standardize_raw_to_epochs(raw)
            freqs, mean_psd = welch_from_epochs(data, sfreq)
        recs.append(make_record(g, freqs, mean_psd, "unm_ds003490"))
    all_psds["unm_ds003490"] = recs
    print(f"  ✓ UNM PD: {len(recs)} sessions, max OLS diff={max(r['ols_max_abs_diff'] for r in recs):.2e}")

    # 8. Iowa PD ds004584 (149 sessions; reuse verified pd_cache)
    print("[8/8] Loading/Verifying Iowa PD ds004584 (N=149)...")
    iowa_cached = {r["subject_id"]: r for r in pd_cache.get("Iowa", [])}
    recs = []
    for g in iowa_gs:
        sid = g["subject_id"]
        freqs, mean_psd = iowa_cached[sid]["freqs"], iowa_cached[sid]["mean_psd"]
        recs.append(make_record(g, freqs, mean_psd, "iowa_ds004584"))
    all_psds["iowa_ds004584"] = recs
    print(f"  ✓ Iowa PD: {len(recs)} sessions, max OLS diff={max(r['ols_max_abs_diff'] for r in recs):.2e}")

    torch.save(all_psds, cache_path)
    total_n = sum(len(v) for v in all_psds.values())
    print(f"\n✅ Saved {total_n} verified Welch PSD sessions to {cache_path} in {time.time()-t0:.1f}s")
    return all_psds


def run_specparam_across_all_cohorts(all_psds: Dict[str, List[Dict]]):
    """Fit genuine specparam (v2.0.0rc7) across all 737 x 19 = 14,003 channel spectra."""
    print("\n" + "=" * 78)
    print(f"🔬 Running genuine specparam {specparam.__version__} across all 737 sessions (14,003 spectra)...")
    print("=" * 78)
    t0 = time.time()

    out_cohorts = {}
    diag_rows = []
    chi_ols_all, chi_sp_all = [], []
    theta_ols_all, theta_sp_resid_all, theta_sp_peaks_all = [], [], []

    for c_key, recs in all_psds.items():
        # Verify identical frequency grid within cohort
        freqs = recs[0]["freqs"]
        for r in recs:
            if not np.allclose(r["freqs"], freqs, atol=1e-8):
                raise RuntimeError(f"Frequency grid mismatch in {c_key} / {r['subject_id']}")

        n_sess = len(recs)
        psds_flat = np.vstack([np.maximum(r["mean_psd"], 1e-25) for r in recs])  # [N*19, n_freqs]

        sgm = SpectralGroupModel(**SPECPARAM_KWARGS)
        sgm.fit(freqs, psds_flat, [1.0, 40.0], n_jobs=1)

        f_fit = sgm.data.freqs
        theta_mask = (f_fit >= 4.0) & (f_fit < 8.0)
        delta_mask = (f_fit >= 1.0) & (f_fit < 4.0)
        alpha_mask = (f_fit >= 8.0) & (f_fit <= 13.0)

        ap_params = sgm.get_params("aperiodic")  # [N*19, 2] -> (offset, exponent)
        r2_all = sgm.get_metrics("gof", "rsquared")
        mae_all = sgm.get_metrics("error", "mae")

        n_null = int(sgm.results.n_null)
        null_set = set(sgm.results.null_inds or [])

        cohort_records = []
        r2_ch, mae_ch, n_peaks_ch, fit_ok_ch = [], [], [], []

        for s_idx, rec in enumerate(recs):
            x_sp_resid = np.zeros((19, 4), dtype=np.float32)
            x_sp_peaks = np.zeros((19, 4), dtype=np.float32)
            ch_r2 = np.zeros(19, dtype=np.float32)
            ch_mae = np.zeros(19, dtype=np.float32)
            ch_offset = np.zeros(19, dtype=np.float32)
            ch_npeaks = np.zeros(19, dtype=np.int32)
            ch_ok = np.ones(19, dtype=bool)

            for ch in range(19):
                flat_i = s_idx * 19 + ch
                if flat_i in null_set:
                    ch_ok[ch] = False
                    x_sp_resid[ch, :] = np.nan
                    x_sp_peaks[ch, :] = np.nan
                    ch_r2[ch] = np.nan
                    ch_mae[ch] = np.nan
                    ch_offset[ch] = np.nan
                    ch_npeaks[ch] = 0
                else:
                    m_i = sgm.get_model(flat_i)
                    log_p = m_i.data.power_spectrum
                    ap_fit = m_i.results.model._ap_fit
                    pk_fit = m_i.results.model._peak_fit
                    resid = log_p - ap_fit

                    off_val, chi_val = float(ap_params[flat_i, 0]), float(ap_params[flat_i, 1])
                    x_sp_resid[ch, 0] = float(np.mean(resid[theta_mask]))
                    x_sp_resid[ch, 1] = float(np.mean(resid[delta_mask]))
                    x_sp_resid[ch, 2] = float(np.mean(resid[alpha_mask]))
                    x_sp_resid[ch, 3] = chi_val

                    x_sp_peaks[ch, 0] = float(np.mean(pk_fit[theta_mask]))
                    x_sp_peaks[ch, 1] = float(np.mean(pk_fit[delta_mask]))
                    x_sp_peaks[ch, 2] = float(np.mean(pk_fit[alpha_mask]))
                    x_sp_peaks[ch, 3] = chi_val

                    ch_r2[ch] = float(r2_all[flat_i])
                    ch_mae[ch] = float(mae_all[flat_i])
                    ch_offset[ch] = off_val
                    ch_npeaks[ch] = int(m_i.results.n_peaks)

                    r2_ch.append(ch_r2[ch])
                    mae_ch.append(ch_mae[ch])
                    n_peaks_ch.append(ch_npeaks[ch])
                    fit_ok_ch.append(True)

                    chi_ols_all.append(float(rec["x_masked_ols"][ch, 3]))
                    chi_sp_all.append(chi_val)
                    theta_ols_all.append(float(rec["x_masked_ols"][ch, 0]))
                    theta_sp_resid_all.append(float(x_sp_resid[ch, 0]))
                    theta_sp_peaks_all.append(float(x_sp_peaks[ch, 0]))

            out_rec = {k: v for k, v in rec.items() if k not in ("freqs", "mean_psd")}
            out_rec["x_specparam_resid"] = torch.tensor(x_sp_resid, dtype=torch.float32)
            out_rec["x_specparam_peaks"] = torch.tensor(x_sp_peaks, dtype=torch.float32)
            out_rec["sp_r2"] = ch_r2
            out_rec["sp_mae"] = ch_mae
            out_rec["sp_offset"] = ch_offset
            out_rec["sp_n_peaks"] = ch_npeaks
            out_rec["sp_fit_ok"] = ch_ok
            cohort_records.append(out_rec)

        out_cohorts[c_key] = cohort_records
        r2_arr = np.array(r2_ch)
        mae_arr = np.array(mae_ch)
        diag_rows.append({
            "cohort": c_key,
            "n_sessions": n_sess,
            "n_spectra": n_sess * 19,
            "n_failed_fits": n_null,
            "pct_r2_lt_080": float(np.mean(r2_arr < 0.80) * 100.0),
            "r2_min": float(np.min(r2_arr)),
            "r2_p05": float(np.percentile(r2_arr, 5)),
            "r2_median": float(np.median(r2_arr)),
            "r2_mean": float(np.mean(r2_arr)),
            "r2_max": float(np.max(r2_arr)),
            "mae_median": float(np.median(mae_arr)),
            "mae_mean": float(np.mean(mae_arr)),
            "mae_max": float(np.max(mae_arr)),
            "mean_n_peaks": float(np.mean(n_peaks_ch)),
            "chi_ols_mean": float(np.mean([r["x_masked_ols"][:, 3].mean().item() for r in recs])),
            "chi_specparam_mean": float(np.nanmean([r["x_specparam_resid"][:, 3].mean().item() for r in cohort_records])),
            "theta_ols_mean": float(np.mean([r["x_masked_ols"][:, 0].mean().item() for r in recs])),
            "theta_sp_resid_mean": float(np.nanmean([r["x_specparam_resid"][:, 0].mean().item() for r in cohort_records])),
            "theta_sp_peaks_mean": float(np.nanmean([r["x_specparam_peaks"][:, 0].mean().item() for r in cohort_records])),
        })
        print(
            f"  {c_key:<18} | {n_sess*19:>4} spectra | failed={n_null} | "
            f"R2 med={np.median(r2_arr):.3f} (p05={np.percentile(r2_arr,5):.3f}) | "
            f"MAE med={np.median(mae_arr):.3f} | "
            f"chi OLS->SP: {diag_rows[-1]['chi_ols_mean']:.3f}->{diag_rows[-1]['chi_specparam_mean']:.3f} | "
            f"theta OLS->SP: {diag_rows[-1]['theta_ols_mean']:+.4f}->{diag_rows[-1]['theta_sp_resid_mean']:+.4f}"
        )

    out_pt = DATA_DIR / "specparam_vs_masked_ols_7cohorts.pt"
    torch.save(out_cohorts, out_pt)

    df_diag = pd.DataFrame(diag_rows)
    out_csv = REPORTS_DIR / "specparam_fit_diagnostics_by_cohort.csv"
    df_diag.to_csv(out_csv, index=False)

    print(
        f"\nGlobal channel correlations (N={len(chi_ols_all)} channel spectra): "
        f"r(chi_OLS, chi_SP) = {np.corrcoef(chi_ols_all, chi_sp_all)[0,1]:.4f} | "
        f"r(theta_OLS, theta_SP_resid) = {np.corrcoef(theta_ols_all, theta_sp_resid_all)[0,1]:.4f} | "
        f"r(theta_OLS, theta_SP_peaks) = {np.corrcoef(theta_ols_all, theta_sp_peaks_all)[0,1]:.4f}"
    )

    # Gain invariance verification across 50 representative spectra (UCSD + UNM + Iowa)
    print("\n🧪 Verifying real-PSD multiplicative gain invariance k in {0.01, 0.5, 2.0, 5.0, 100.0}...")
    sample_recs = all_psds["ucsd_ds002778"][:5] + all_psds["mumtaz_ec"][:5] + all_psds["iowa_ds004584"][:5]
    freqs0 = sample_recs[0]["freqs"]
    psds0 = np.vstack([np.maximum(r["mean_psd"], 1e-25) for r in sample_recs])
    sgm_base = SpectralGroupModel(**SPECPARAM_KWARGS)
    sgm_base.fit(freqs0, psds0, [1.0, 40.0], n_jobs=1)
    ap_base = sgm_base.get_params("aperiodic")
    f_fit0 = sgm_base.data.freqs
    th_m = (f_fit0 >= 4.0) & (f_fit0 < 8.0)
    resid_base = np.array([sgm_base.get_model(i).data.power_spectrum - sgm_base.get_model(i).results.model._ap_fit for i in range(len(psds0))])
    theta_base = resid_base[:, th_m].mean(axis=1)

    gain_rows = []
    for k_val in [0.01, 0.5, 1.0, 2.0, 5.0, 100.0]:
        sgm_k = SpectralGroupModel(**SPECPARAM_KWARGS)
        sgm_k.fit(freqs0, (k_val ** 2) * psds0, [1.0, 40.0], n_jobs=1)
        ap_k = sgm_k.get_params("aperiodic")
        resid_k = np.array([sgm_k.get_model(i).data.power_spectrum - sgm_k.get_model(i).results.model._ap_fit for i in range(len(psds0))])
        theta_k = resid_k[:, th_m].mean(axis=1)
        gain_rows.append({
            "gain_k": k_val,
            "n_spectra": len(psds0),
            "expected_offset_shift": 2.0 * np.log10(k_val),
            "measured_offset_shift_mean": float(np.mean(ap_k[:, 0] - ap_base[:, 0])),
            "max_abs_diff_chi": float(np.max(np.abs(ap_k[:, 1] - ap_base[:, 1]))),
            "max_abs_diff_theta_resid": float(np.max(np.abs(theta_k - theta_base))),
        })
        print(
            f"  k={k_val:<5} | Δoffset={gain_rows[-1]['measured_offset_shift_mean']:+.6f} "
            f"(expected {2*np.log10(k_val):+.6f}) | "
            f"max|Δχ|={gain_rows[-1]['max_abs_diff_chi']:.2e} | "
            f"max|Δθ_resid|={gain_rows[-1]['max_abs_diff_theta_resid']:.2e}"
        )

    pd.DataFrame(gain_rows).to_csv(REPORTS_DIR / "specparam_gain_invariance_check.csv", index=False)
    print(f"\n✅ Saved specparam tensors ({out_pt}) & diagnostics ({out_csv}) in {time.time()-t0:.1f}s!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-extract", action="store_true", help="Re-extract Welch PSDs from raw files")
    args = parser.parse_args()

    all_psds = extract_or_load_all_welch_psds(force_extract=args.force_extract)
    run_specparam_across_all_cohorts(all_psds)


if __name__ == "__main__":
    main()
