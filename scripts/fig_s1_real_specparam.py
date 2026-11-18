#!/usr/bin/env python3
"""
Generate Publication-Quality Supplementary Figure S1 (fig_S1_specparam_example.png/.pdf)
from a REAL clinical participant PSD (OpenNeuro ds004504 sub-018, Alzheimer's disease, Fz)
comparing genuine `specparam` (v2.0.0rc7) against the deprecated [7, 14] Hz-masked OLS baseline.
"""

from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from specparam import SpectralModel

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = BENCHMARK_ROOT / "src"
sys.path.append(str(SRC_DIR))
from harmonizer_and_graph import CHANNELS_19, compute_masked_ols_19ch_features

DATA_DIR = BENCHMARK_ROOT / "data" / "derivatives"
REPORTS_DIR = BENCHMARK_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_real_figure_s1(
    output_dir: Path = REPORTS_DIR,
    cohort_key: str = "dementia_ds004504",
    subject_id: str = "sub-018",
    channel_name: str = "Fz",
):
    """Render Figure S1 using empirical Welch PSD and genuine specparam decomposition."""
    psd_path = DATA_DIR / "all_7cohorts_welch_psds.pt"
    if not psd_path.exists():
        raise FileNotFoundError(
            f"Missing empirical PSD cache {psd_path}. "
            "Run scripts/39_extract_all_welch_psds_and_run_specparam.py first."
        )

    all_psds = torch.load(psd_path, weights_only=False)
    matches = [r for r in all_psds[cohort_key] if r["subject_id"] == subject_id]
    if not matches:
        raise ValueError(f"Subject {subject_id!r} not found in cohort {cohort_key!r}.")
    rec = matches[0]

    ch_idx = CHANNELS_19.index(channel_name)
    freqs_all = rec["freqs"]
    # Convert Welch PSD from V^2/Hz to standard EEG uV^2/Hz (1 V^2 = 10^12 uV^2)
    psd_uv2 = rec["mean_psd"] * 1e12

    sm = SpectralModel(
        aperiodic_mode="fixed",
        periodic_mode="gaussian",
        peak_width_limits=[1.0, 12.0],
        max_n_peaks=6,
        min_peak_height=0.05,
        peak_threshold=2.0,
        metrics=["error_mae", "gof_rsquared"],
        verbose=False,
    )
    sm.fit(freqs_all, np.maximum(psd_uv2[ch_idx], 1e-12), [2.0, 30.0])

    f_fit = sm.data.freqs
    raw_log = sm.data.power_spectrum
    ap_sp = sm.results.model._ap_fit
    pk_sp = sm.results.model._peak_fit
    full_sp = sm.results.model.modeled_spectrum
    ap_offset, ap_chi = sm.get_params("aperiodic")
    r2_val = float(sm.get_metrics("gof", "rsquared"))
    mae_val = float(sm.get_metrics("error", "mae"))
    peaks = sm.get_params("peak")  # [n_peaks, 3] -> CF, PW, BW

    # Legacy [7, 14] Hz-masked OLS baseline across full [1, 40] Hz & interior [2, 30] Hz context
    valid_40 = (freqs_all >= 1.0) & (freqs_all <= 40.0)
    f_40 = freqs_all[valid_40]
    raw_log_40 = np.log10(np.maximum(psd_uv2[ch_idx, valid_40], 1e-12))
    log_f = np.log10(f_fit)
    ols_mask = (f_fit < 7.0) | (f_fit > 14.0)
    slope_ols, intercept_ols = np.polyfit(log_f[ols_mask], raw_log[ols_mask], 1)
    ap_ols = intercept_ols + slope_ols * log_f
    chi_ols = -slope_ols

    th_mask = (f_fit >= 4.0) & (f_fit < 8.0)
    theta_sp = float(np.mean((raw_log - ap_sp)[th_mask]))
    theta_ols = float(np.mean((raw_log - ap_ols)[th_mask]))

    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.4, 7.6), sharex=True, dpi=300,
        gridspec_kw={"height_ratios": [1.18, 1.0], "hspace": 0.16}
    )

    # Panel A: Empirical log10 PSD, interior [2, 30] Hz specparam fit, and OLS contamination
    ax1.axvspan(1.0, 2.0, color="#FEE2E2", alpha=0.40, zorder=0, label="Excluded passband skirt (<2 / >30 Hz)")
    ax1.axvspan(30.0, 40.0, color="#FEE2E2", alpha=0.40, zorder=0)
    ax1.axvspan(4.0, 7.0, color="#FEF3C7", alpha=0.55, zorder=0, label="Unmasked 4–7 Hz theta (biases OLS fit)")
    ax1.axvspan(7.0, 14.0, color="#E2E8F0", alpha=0.45, zorder=0, label="Masked [7, 14] Hz window (OLS excluded)")
    ax1.plot(f_40, raw_log_40, color="#1E293B", lw=1.6, zorder=3,
             label=rf"Real Welch $\log_{{10}}\mathrm{{PSD}}(f)$ ({subject_id}, AD, Ch {channel_name})")
    ax1.plot(f_fit, full_sp, color="#2563EB", lw=2.0, zorder=4,
             label=rf"specparam [2, 30] Hz fit (Aperiodic + {len(peaks)} Gaussian peaks)")
    ax1.plot(f_fit, ap_sp, color="#DC2626", ls="--", lw=2.0, zorder=5,
             label=rf"specparam aperiodic fit ($b={ap_offset:.3f}$, $\chi={ap_chi:.3f}$)")
    ax1.plot(f_fit, ap_ols, color="#0D9488", ls="-.", lw=1.8, zorder=5,
             label=rf"Legacy [7, 14] Hz-masked OLS ($\chi_{{OLS}}={chi_ols:.3f}$)")

    ax1.set_ylim(-1.60, 1.82)
    ax1.set_ylabel(r"Power $\log_{10}(\mu\mathrm{V}^2/\mathrm{Hz})$ [B]", fontsize=10.2, fontweight="bold")
    ax1.set_title(
        f"A. Real Participant Power Spectrum ({cohort_key}, {subject_id}, Channel {channel_name}): "
        "Interior [2, 30] Hz specparam vs. Masked OLS",
        fontsize=10.4, fontweight="bold", pad=8
    )
    ax1.legend(loc="upper right", fontsize=7.6, frameon=True, framealpha=0.96)
    ax1.grid(True, linestyle=":", alpha=0.45)

    pk_lines = "\n".join(
        f"  Peak {i+1}: {cf:.2f} Hz (PW={pw:.2f} B, BW={bw:.2f} Hz)"
        for i, (cf, pw, bw) in enumerate(peaks)
    )
    info_txt = (
        f"Empirical Fit (specparam v2.0.0rc7)\n"
        f"Subject: {subject_id} (AD, ds004504, Ch {channel_name})\n"
        f"b = {ap_offset:.3f} B | χ = {ap_chi:.3f} (OLS χ = {chi_ols:.3f})\n"
        f"GoF R² = {r2_val:.3f} | MAE = {mae_val:.3f} B\n"
        f"{pk_lines}"
    )
    ax1.text(
        0.02, 0.04, info_txt, transform=ax1.transAxes, fontsize=7.8, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.42", facecolor="#F8FAFC", edgecolor="#94A3B8", lw=0.85, alpha=0.96),
        zorder=6
    )

    # Panel B: Flattened Periodic Residuals
    resid_sp = raw_log - ap_sp
    resid_ols = raw_log - ap_ols
    ax2.axvspan(1.0, 2.0, color="#FEE2E2", alpha=0.35, zorder=0)
    ax2.axvspan(30.0, 40.0, color="#FEE2E2", alpha=0.35, zorder=0)
    ax2.axvspan(4.0, 8.0, color="#FEF3C7", alpha=0.45, zorder=0)
    ax2.axhline(0.0, color="#DC2626", ls="--", lw=1.3, zorder=2, label="specparam aperiodic baseline (0.0 B)")
    ax2.plot(f_fit, resid_sp, color="#475569", lw=1.3, alpha=0.85, zorder=3,
             label="Empirical specparam residual (Raw − specparam ap)")
    ax2.plot(f_fit, pk_sp, color="#2563EB", lw=2.1, zorder=4,
             label="Fitted specparam Gaussian peaks")
    ax2.fill_between(f_fit, 0.0, pk_sp, color="#60A5FA", alpha=0.22, zorder=1)
    ax2.plot(f_fit, resid_ols, color="#0D9488", ls="-.", lw=1.7, zorder=4,
             label="Deprecated [7, 14] Hz OLS residual (theta biased downward)")

    ax2.set_ylim(-0.72, 1.18)
    ax2.text(
        0.58, 0.56,
        f"4–8 Hz Mean Periodic Theta:\n"
        f"• True specparam residual: {theta_sp:+.3f} B\n"
        f"• Masked OLS residual:     {theta_ols:+.3f} B\n"
        f"• OLS Attenuation Bias:    {theta_ols - theta_sp:+.3f} B",
        transform=ax2.transAxes, fontsize=8.2, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.42", facecolor="#FFFBEB", edgecolor="#D97706", lw=0.9, alpha=0.96),
        zorder=6
    )

    ax2.set_xlim(1.0, 40.0)
    ax2.set_xlabel("Frequency (Hz)", fontsize=10.2, fontweight="bold")
    ax2.set_ylabel("Periodic Residual [B]", fontsize=10.2, fontweight="bold")
    ax2.set_title(
        "B. Flattened Periodic Spectra: Exposing Theta-Band Pull-Up Bias in [7, 14] Hz-Masked OLS",
        fontsize=10.6, fontweight="bold", pad=8
    )
    ax2.legend(loc="upper right", fontsize=8.0, frameon=True, framealpha=0.96)
    ax2.grid(True, linestyle=":", alpha=0.45)

    out_pdf = output_dir / "fig_S1_specparam_example.pdf"
    out_png = output_dir / "fig_S1_specparam_example.png"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_png, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"✓ Generated real-participant Figure S1 -> {out_png} & {out_pdf}")
    return {
        "cohort": cohort_key,
        "subject_id": subject_id,
        "channel": channel_name,
        "ap_offset_uV2": float(ap_offset),
        "ap_chi_sp": float(ap_chi),
        "ap_chi_ols": float(chi_ols),
        "r2": r2_val,
        "mae": mae_val,
        "theta_sp_resid": theta_sp,
        "theta_ols_resid": theta_ols,
    }


if __name__ == "__main__":
    stats_out = plot_real_figure_s1()
    print("Real Fig S1 params:", stats_out)
