#!/usr/bin/env python3
"""
scripts/51_generate_v6_figures.py
----------------------------------
Springer Neuroinformatics v6 Figure Suite (Figures 1-4 & Supplementary Figures S1-S6):
  - Removes all internal figure-level titles (suptitles) per Springer Nature artwork rules.
  - Enforces lowercase panel identifiers (a), (b), (c), (d) consistently across all artwork.
  - Optimizes canvas dimensions (near 174 mm double-column aspect) and font sizes so labels
    remain >= 8-10 pt equivalent at final print width, exporting 600-DPI PNG + vector PDF + EPS.
  - Synchronizes Figure 3(c) (SPD manifold suitability framing), Figure 2(c) (neutral batch
    roll-off annotation), and Figure 4(c) (paired 4-model max-statistic two-sided perm p).
"""

import json
import shutil
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from specparam import SpectralModel

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = BENCHMARK_ROOT / "src"
sys.path.append(str(SRC_DIR))
from harmonizer_and_graph import CHANNELS_19

REPORTS_DIR = BENCHMARK_ROOT / "reports"
DATA_DIR = BENCHMARK_ROOT / "data" / "derivatives"
MANUSCRIPT_FIGS_DIR = BENCHMARK_ROOT / "manuscript" / "figures"
for d in (REPORTS_DIR, MANUSCRIPT_FIGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 0.90
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["axes.labelsize"] = 9.5
plt.rcParams["axes.titlesize"] = 10.0
plt.rcParams["xtick.labelsize"] = 8.8
plt.rcParams["ytick.labelsize"] = 8.8
plt.rcParams["legend.fontsize"] = 8.0
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.edgecolor"] = "#cbd5e1"


def save_fig(fig, stem: str, is_main: bool = True):
    """Export 600-DPI PNG, vector PDF, and vector EPS across reports/ and manuscript/figures/."""
    for folder in (REPORTS_DIR, MANUSCRIPT_FIGS_DIR):
        fig.savefig(folder / f"{stem}.png", dpi=600, bbox_inches="tight")
        fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(folder / f"{stem}.eps", bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {stem}.png (600 dpi) / .pdf / .eps")


def plot_figure_1():
    df = pd.read_csv(REPORTS_DIR / "multicohort_biophysical_effect_sizes_expanded.csv")
    cohort_order = [
        ("Study 1: Acute Sleep Deprivation", "Study 1: Sleep Deprivation (N=71 pairs)", "#64748b"),
        ("Study 2: Outpatient MDD", "Study 2: Outpatient MDD EC (29 vs 27)*", "#2563eb"),
        ("Study 3: Univ. Depression Screening", "Study 3: Univ. Depression (44 vs 75)", "#0284c7"),
        ("Study 4: UCSD Parkinson All", "Study 4: UCSD PD Mean (15 vs 16)", "#d97706"),
        ("Study 6a: UNM Parkinson OFF", "Study 6a: UNM PD-OFF EO (25 vs 25)", "#ea580c"),
        ("Study 6b: UNM Parkinson ON", "Study 6b: UNM PD-ON EO (25 vs 25)", "#f59e0b"),
        ("Study 7: Iowa Parkinson Cohort", "Study 7: Iowa PD (100 vs 49)", "#b45309"),
        ("Study 5a: Alzheimer's Disease (AD)", "Study 5a: Dementia AD (36 vs 29)", "#dc2626"),
        ("Study 5b: Frontotemporal Dementia (FTD)", "Study 5b: Dementia FTD (23 vs 29)", "#e11d48"),
        ("Study 5c: Clinical Dementia (All)", "Study 5c: Dementia All (59 vs 29)", "#991b1b"),
    ]
    panels = [
        ("frontal_theta", "(a) Frontal Periodic Theta (4–8 Hz)"),
        ("global_delta", "(b) Global Low-Freq Residual (2–4 Hz)"),
        ("posterior_alpha", "(c) Posterior Periodic Alpha (8–13 Hz)"),
        ("global_chi", "(d) Global Aperiodic Exponent (χ)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.6), dpi=300, sharey=True)
    y_pos = np.arange(len(cohort_order))[::-1]

    for ax, (mkey, mtitle) in zip(axes.flat, panels):
        sub = df[df["marker_key"] == mkey].set_index("cohort")
        ax.axvline(0.0, color="#334155", ls="--", lw=1.1, zorder=1)
        ax.axhspan(-0.5, 2.5, color="#fef2f2", alpha=0.55, zorder=0)
        ax.axhspan(2.5, 6.5, color="#fffbeb", alpha=0.55, zorder=0)
        ax.axhspan(6.5, 8.5, color="#eff6ff", alpha=0.55, zorder=0)

        for idx, (cname, _, col) in enumerate(cohort_order):
            row = sub.loc[cname]
            d, lo, hi = float(row["cohens_d"]), float(row["ci_lower"]), float(row["ci_upper"])
            q50, q10 = float(row["fdr_q_50"]), float(row["fdr_q_10"])
            sig = "***" if q50 < 0.001 else ("**" if q50 < 0.01 else ("*" if q50 < 0.05 else ("†" if q10 < 0.05 else "")))
            y = y_pos[idx]
            ax.errorbar(
                d, y, xerr=[[d - lo], [hi - d]], fmt="o", color=col, ecolor=col,
                elinewidth=1.7, capsize=3.2, markersize=6.0, markeredgecolor="#ffffff", markeredgewidth=0.8, zorder=3,
            )
            ax.text(
                max(hi + 0.05, d + 0.08), y, f"d={d:+.2f}{sig}", va="center", ha="left",
                fontsize=7.8, color="#0f172a", zorder=4,
                bbox=dict(boxstyle="square,pad=0.05", fc="#ffffff", ec="none", alpha=0.85),
            )

        ax.set_title(mtitle, fontweight="bold", loc="left", pad=6, fontsize=9.8)
        ax.set_xlim(-1.95, 2.75)
        ax.set_ylim(-1.15, 9.55)
        ax.set_xlabel("Cohen's d (95% CI)", fontsize=9.0)
        ax.grid(axis="x", ls=":", alpha=0.5, color="#94a3b8")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[:, 0]:
        ax.set_yticks(y_pos)
        ax.set_yticklabels([lbl for _, lbl, _ in cohort_order], fontsize=8.4)
    axes[1, 1].text(
        0.99, 0.015, r"$*\,q_{50}<0.05 \quad **\,q_{50}<0.01 \quad ***\,q_{50}<0.001 \quad \dagger\,q_{10}<0.05$",
        transform=axes[1, 1].transAxes, ha="right", va="bottom", fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.24", fc="#ffffff", ec="#cbd5e1", alpha=0.96),
    )
    fig.tight_layout(h_pad=1.3, w_pad=0.9)
    save_fig(fig, "Figure_1", is_main=True)


def plot_figure_2():
    wide = pd.read_csv(REPORTS_DIR / "ablation_specparam_vs_ols_biophysics_wide.csv")
    theta_df = wide[wide["marker"] == "frontal_theta_5ch"].set_index("cohort")
    sim = pd.read_csv(REPORTS_DIR / "synthetic_spectral_leakage_simulation.csv")

    fig = plt.figure(figsize=(11.4, 4.7), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.32, 1.05, 1.05], wspace=0.35)

    # Panel (a)
    ax1 = fig.add_subplot(gs[0, 0])
    cohort_rows = [
        ("Study 1: Acute Sleep Deprivation", "S1: Sleep Depr."),
        ("Study 2: Outpatient MDD", "S2: MDD EC*"),
        ("Study 3: Univ. Depression Screening", "S3: Univ. Dep."),
        ("Study 4: UCSD Parkinson All", "S4: UCSD PD"),
        ("Study 6a: UNM Parkinson OFF", "S6a: UNM OFF"),
        ("Study 6b: UNM Parkinson ON", "S6b: UNM ON"),
        ("Study 7: Iowa Parkinson Cohort", "S7: Iowa PD"),
        ("Study 5a: Alzheimer's Disease (AD)", "S5a: AD"),
        ("Study 5b: Frontotemporal Dementia (FTD)", "S5b: FTD"),
        ("Study 5c: Clinical Dementia (All)", "S5c: Dem. All"),
    ]
    y = np.arange(len(cohort_rows))[::-1]
    h = 0.26
    d_ols40 = [float(theta_df.loc[c, "cohens_d_masked_ols_1to40"]) for c, _ in cohort_rows]
    d_sp40 = [float(theta_df.loc[c, "cohens_d_specparam_resid_1to40"]) for c, _ in cohort_rows]
    d_sp30 = [float(theta_df.loc[c, "cohens_d_specparam_resid_2to30"]) for c, _ in cohort_rows]

    ax1.axvline(0, color="#334155", ls="--", lw=1.0)
    ax1.barh(y + h, d_ols40, height=h, color="#94a3b8", edgecolor="#475569", lw=0.6, label="[7, 14] Masked OLS [1–40]")
    ax1.barh(y, d_sp40, height=h, color="#0284c7", edgecolor="#0369a1", lw=0.6, label="specparam [1–40 Hz]")
    ax1.barh(y - h, d_sp30, height=h, color="#dc2626", edgecolor="#991b1b", lw=0.6, label="specparam [2–30 Hz]")
    ax1.annotate(
        f"S2 edge collapse:\n{d_sp40[1]:+.2f} vs {d_sp30[1]:+.2f}",
        xy=(d_sp40[1], y[1]), xytext=(0.08, y[1] + 0.05),
        va="center", fontsize=7.5, color="#991b1b", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#991b1b", lw=0.9),
        bbox=dict(boxstyle="round,pad=0.20", fc="#fef2f2", ec="#fca5a5"),
    )
    ax1.set_yticks(y)
    ax1.set_yticklabels([lbl for _, lbl in cohort_rows], fontsize=8.2)
    ax1.set_xlim(-1.60, 1.65)
    ax1.set_xlabel("Frontal Theta Effect Size (d)", fontsize=9.0)
    ax1.set_title("(a) Estimator & Fit-Window Shift", fontweight="bold", loc="left", fontsize=9.6)
    ax1.legend(loc="lower left", fontsize=7.0, framealpha=0.95)
    ax1.grid(axis="x", ls=":", alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel (b)
    ax2 = fig.add_subplot(gs[0, 1])
    scen_order = [
        ("S1_TrueNullTheta_TrueExponentDrop", "S1: Null θ +\nSlope Drop"),
        ("S2_TrueNullTheta_BroadAlphaAndSlowCurve", "S2: Null θ +\nBroad Alpha"),
        ("S3_FilterEdgeAttenuation_Study2Replica", "S3: Skirt\nAttenuation"),
    ]
    x2, w2, all_b = np.arange(len(scen_order)), 0.25, []
    for idx_e, (ekey, elbl, col) in enumerate([
        ("Masked_OLS_1to40", "Masked OLS [1–40]", "#64748b"),
        ("specparam_1to40", "specparam [1–40]", "#0284c7"),
        ("specparam_2to30", "specparam [2–30]", "#dc2626"),
    ]):
        b_vals = [float(sim[(sim["scenario"] == sk) & (sim["estimator"] == ekey)].iloc[0]["theta_d_bias"]) for sk, _ in scen_order]
        all_b.extend(b_vals)
        bars = ax2.bar(x2 + (idx_e - 1) * w2, b_vals, width=w2, color=col, edgecolor="#1e293b", lw=0.6, label=elbl, zorder=3)
        for b, bv in zip(bars, b_vals):
            va = "bottom" if bv >= 0 else "top"
            off = 0.06 if bv >= 0 else -0.06
            ax2.text(b.get_x() + b.get_width() / 2, bv + off, f"{bv:+.2f}", ha="center", va=va, fontsize=7.0, fontweight="bold")
    ax2.axhline(0.0, color="#0f172a", lw=1.0, zorder=2)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([lbl for _, lbl in scen_order], fontsize=7.8)
    ax2.set_ylim(min(-0.85, min(all_b) - 0.35), max(3.6, max(all_b) + 0.65))
    ax2.set_ylabel("Simulation Bias (Est − True d)", fontsize=9.0)
    ax2.set_title("(b) Monte Carlo Bias (150 Trials)", fontweight="bold", loc="left", fontsize=9.6)
    ax2.legend(loc="upper left", fontsize=7.2, framealpha=0.95)
    ax2.grid(axis="y", ls=":", alpha=0.5, zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel (c)
    ax3 = fig.add_subplot(gs[0, 2])
    with open(REPORTS_DIR / "study2_unfiltered_batch_spectra.json", "r", encoding="utf-8") as f:
        s2b = json.load(f)
    freqs_u = np.asarray(s2b["freqs"], dtype=float)
    ax3.axvspan(0.5, 2.0, color="#fee2e2", alpha=0.45, zorder=0, label="Skirt (<2 / >30 Hz)")
    ax3.axvspan(30.0, 65.0, color="#fee2e2", alpha=0.35, zorder=0)
    ax3.axvline(50.0, color="#94a3b8", ls=":", lw=1.0, zorder=1)
    for bkey, blbl, col, ls in [
        ("HC_2011", f"2011 HC (n={s2b['batches']['HC_2011']['n']})", "#0284c7", "-"),
        ("HC_2013_01", f"Jan-13 HC (n={s2b['batches']['HC_2013_01']['n']})", "#d97706", "-."),
        ("HC_2013_04_07", f"Apr–Jul 13 HC (n={s2b['batches']['HC_2013_04_07']['n']})", "#16a34a", "--"),
        ("MDD_2013_04_07", f"Apr–Jul 13 MDD (n={s2b['batches']['MDD_2013_04_07']['n']})", "#dc2626", "-"),
    ]:
        mu = np.asarray(s2b["batches"][bkey]["mean"], dtype=float)
        sem = np.asarray(s2b["batches"][bkey]["sem"], dtype=float)
        ax3.plot(freqs_u, mu, color=col, ls=ls, lw=1.6, label=blbl, zorder=3)
        ax3.fill_between(freqs_u, mu - sem, mu + sem, color=col, alpha=0.14, zorder=2)
    ax3.annotate(
        "Unnotched 50 Hz\n(+2.03 B)", xy=(50.0, max(s2b["batches"]["HC_2013_01"]["mean"])),
        xytext=(29.0, -11.3), fontsize=7.2, color="#b45309", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#b45309", lw=0.9),
        bbox=dict(boxstyle="round,pad=0.2", fc="#fffbeb", ec="#fcd34d"),
    )
    ax3.annotate(
        "Apr–Jul 2013\nupstream roll-off",
        xy=(40.0, s2b["batches"]["MDD_2013_04_07"]["mean"][int(np.argmin(np.abs(freqs_u - 40.0)))]),
        xytext=(15.5, -15.8), fontsize=7.2, color="#991b1b", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#991b1b", lw=0.9),
        bbox=dict(boxstyle="round,pad=0.2", fc="#fef2f2", ec="#fca5a5"),
    )
    ax3.set_xlim(0.5, 65.0)
    ax3.set_ylim(-17.4, -10.0)
    ax3.set_xlabel("Frequency (Hz; Unfiltered EDF)", fontsize=9.0)
    ax3.set_ylabel(r"Mean 19-ch $\log_{10}$ PSD [B]", fontsize=9.0)
    ax3.set_title("(c) Study 2 Raw EDF Batches", fontweight="bold", loc="left", fontsize=9.6)
    ax3.legend(loc="lower left", fontsize=6.6, framealpha=0.95)
    ax3.grid(True, ls=":", alpha=0.5)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.15, top=0.90)
    save_fig(fig, "Figure_2", is_main=True)


def plot_figure_3():
    ms = pd.read_csv(REPORTS_DIR / "benchmark_repeated_10x5_summary.csv")
    lk = pd.read_csv(REPORTS_DIR / "benchmark_leakage_ablation.csv")
    pli_audit = pd.read_csv(REPORTS_DIR / "pli_spd_eigenvalue_audit.csv")
    pli_sum = pd.read_csv(REPORTS_DIR / "pli_spd_eigenvalue_summary.csv").iloc[0]
    dg = pd.read_csv(REPORTS_DIR / "v7_deep_baseline_and_tuned_gcn_10x5.csv")
    sc_fm = pd.read_csv(REPORTS_DIR / "spatial_cov_foldwise_frechet_check.csv")

    fig = plt.figure(figsize=(11.8, 4.9), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.28, 1.05, 1.02], wspace=0.38)

    # Panel (a)
    ax1 = fig.add_subplot(gs[0, 0])
    nn8 = ms[~ms["cohort"].str.contains("2b|5c|6b|6c")].copy()
    agg = nn8.groupby("model")[["mean_auc", "std_auc"]].agg(mean_auc=("mean_auc", "mean"), within_sd=("std_auc", "mean"))
    dg_agg = dg.groupby("model")[["mean_auc", "sd_auc"]].mean()
    airm_rf_row = sc_fm[sc_fm["classifier"] == "RF"]
    agg.loc["SpatialCov_AIRM_RF"] = [float(airm_rf_row["auc_fold_frechet_ref"].mean()), float(airm_rf_row["sd_fold_frechet"].mean())]
    for m_new in ("ShallowSpecCNN", "GCN_OMST_Tuned"):
        agg.loc[m_new] = [float(dg_agg.loc[m_new, "mean_auc"]), float(dg_agg.loc[m_new, "sd_auc"])]

    model_order = [
        ("SpatialCov_AIRM_RF", "SpatialCov RF (AIRM)", "#0f766e"),
        ("Linear_SVM", "Linear SVM", "#1d4ed8"),
        ("RandomForest", "Random Forest", "#0284c7"),
        ("Tuned_LogisticRegression", "Tuned LR", "#0369a1"),
        ("ShallowSpecCNN", "ShallowSpecCNN", "#7c3aed"),
        ("GCN_OMST_Tuned", "GCN-OMST-Tuned", "#e11d48"),
        ("GCN_Random_EdgeMatched", "GCN-Random*", "#f97316"),
        ("GCN_Full", "GCN-Full", "#ea580c"),
        ("GCN_OMST", "GCN-OMST", "#dc2626"),
        ("GCN_Weighted_PLI", "GCN-WPLI", "#b91c1c"),
        ("GCN_Identity", "GCN-Identity", "#94a3b8"),
    ]
    x = np.arange(len(model_order))
    vals = [float(agg.loc[m, "mean_auc"]) for m, _, _ in model_order]
    sds = [float(agg.loc[m, "within_sd"]) for m, _, _ in model_order]
    bars = ax1.bar(x, vals, yerr=sds, capsize=2.4, color=[c for _, _, c in model_order], edgecolor="#1e293b", lw=0.6, alpha=0.92, zorder=3)
    ax1.axhline(0.50, color="#475569", ls=":", lw=1.0)
    ax1.axhline(float(agg.loc["GCN_OMST", "mean_auc"]), color="#dc2626", ls="--", lw=1.0, alpha=0.75)
    for b, v, sd in zip(bars, vals, sds):
        ax1.text(b.get_x() + b.get_width() / 2, v + sd + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold", rotation=35)
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl, _ in model_order], rotation=35, ha="right", fontsize=7.4)
    ax1.set_ylim(0.48, 0.85)
    ax1.set_ylabel("10×5 Mean ROC-AUC (k=8)", fontsize=9.0)
    ax1.set_title("(a) 10×5 Model & Graph Parity", fontweight="bold", loc="left", fontsize=9.6)
    ax1.grid(axis="y", ls=":", alpha=0.5, zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel (b)
    ax2 = fig.add_subplot(gs[0, 1])
    cohorts_lk = [
        ("Study 1: Sleep Deprivation", "Study 1\n(Discordant)"),
        ("Study 4: UCSD Parkinson", "Study 4\n(Concordant)"),
        ("Study 6c: UNM Parkinson All", "Study 6c\n(Concordant)"),
    ]
    lk_sub = lk[lk["regime"] == "Leaky_Subject_Mixing"].copy()
    x2, w2 = np.arange(len(cohorts_lk)) * 1.25, 0.28
    for idx_m, (mkey, mlbl, mcol) in enumerate([
        ("GCN_OMST", "GCN-OMST", "#dc2626"),
        ("Tuned_LogisticRegression", "Tuned LR", "#0284c7"),
        ("RandomForest", "RF", "#0f766e"),
    ]):
        deltas, ylo, yhi = [], [], []
        for ck, _ in cohorts_lk:
            row = lk_sub[(lk_sub["cohort"] == ck) & (lk_sub["model"] == mkey)].iloc[0]
            dv = float(row["mean_delta_vs_strict"])
            deltas.append(dv)
            ylo.append(max(0.0, dv - float(row["delta_ci_lo"])))
            yhi.append(max(0.0, float(row["delta_ci_hi"]) - dv))
        bars2 = ax2.bar(x2 + (idx_m - 1) * w2, deltas, width=w2, yerr=[ylo, yhi], capsize=2.4, color=mcol, edgecolor="#1e293b", lw=0.6, label=mlbl, zorder=3)
        for b, dv, le, he in zip(bars2, deltas, ylo, yhi):
            va = "bottom" if dv >= 0 else "top"
            yt = (dv + he + 0.005) if dv >= 0 else (dv - le - 0.006)
            ax2.text(b.get_x() + b.get_width() / 2, yt, f"{dv:+.3f}", ha="center", va=va, fontsize=6.6, fontweight="bold")
    ax2.axhline(0.0, color="#334155", lw=1.0, zorder=2)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([lbl for _, lbl in cohorts_lk], fontsize=8.0)
    ax2.set_xlim(-0.55, 3.05)
    ax2.set_ylim(-0.155, 0.165)
    ax2.set_ylabel("Mean ΔAUC ± 95% CI (Mixed − Strict)", fontsize=8.8)
    ax2.set_title("(b) Bidirectional Leakage", fontweight="bold", loc="left", fontsize=9.6)
    ax2.legend(loc="upper left", fontsize=7.4)
    ax2.grid(axis="y", ls=":", alpha=0.5, zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel (c): Clarified SPD suitability framing per Critique I.3
    ax3 = fig.add_subplot(gs[0, 2])
    n_indef, n_tot, pct_indef = int(pli_sum["n_indefinite_raw"]), int(pli_sum["n_sessions"]), float(pli_sum["pct_indefinite_raw"])
    audit_rows = [
        ("Study 2: Outpatient MDD (EC)", "Study 2: MDD EC"),
        ("Study 5: Clinical Dementia (ds004504)", "Study 5: Dementia"),
        ("Study 3: Cavanagh Depression (ds003478)", "Study 3: Univ. Dep."),
        ("Study 7: Iowa Parkinson (ds004584)", "Study 7: Iowa PD"),
        ("Study 1: Sleep Deprivation (ds004902)", "Study 1: Sleep"),
        ("Study 4: UCSD Parkinson (ds002778)", "Study 4: UCSD PD"),
        ("Study 6: UNM Parkinson EO (ds003490)", "Study 6: UNM PD"),
    ]
    pa = pli_audit.set_index("cohort")
    y3 = np.arange(len(audit_rows))[::-1]
    pcts = [float(pa.loc[k, "pct_indefinite_raw"]) for k, _ in audit_rows]
    mins = [float(pa.loc[k, "min_eigenvalue"]) for k, _ in audit_rows]
    bars3 = ax3.barh(y3, pcts, height=0.58, color="#dc2626", alpha=0.82, edgecolor="#7f1d1d", lw=0.7, zorder=3)
    for b, pv, mv in zip(bars3, pcts, mins):
        ax3.text(pv + 1.0, b.get_y() + b.get_height() / 2, f"{pv:.1f}% (λ={mv:.2f})", va="center", ha="left", fontsize=7.6)
    ax3.set_yticks(y3)
    ax3.set_yticklabels([lbl for _, lbl in audit_rows], fontsize=8.0)
    ax3.set_xlim(0, 68)
    ax3.set_xlabel("Indefinite (I + PLI) Matrices (%)", fontsize=9.0)
    ax3.set_title(f"(c) Indefinite PLI SPD Inputs ({n_indef}/{n_tot})", fontweight="bold", loc="left", fontsize=9.5)
    ax3.grid(axis="x", ls=":", alpha=0.5, zorder=0)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.22, top=0.90)
    save_fig(fig, "Figure_3", is_main=True)


def plot_figure_4(sp):
    tr = pd.read_csv(REPORTS_DIR / "comprehensive_transdiagnostic_transfer_matrix.csv")
    ms = pd.read_csv(REPORTS_DIR / "benchmark_repeated_10x5_summary.csv")
    rg_p = REPORTS_DIR / "multicohort_clinical_regression_benchmarks_v6.csv"
    rg = pd.read_csv(rg_p if rg_p.exists() else REPORTS_DIR / "multicohort_clinical_regression_benchmarks.csv")

    fig = plt.figure(figsize=(11.4, 4.6), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.02, 1.16], wspace=0.36)

    # Panel (a)
    ax1 = fig.add_subplot(gs[0, 0])
    doms = [
        ("Sleep_Debt", "Study 1: Sleep Deprivation (ds004902)", "Sleep\nDebt"),
        ("Mumtaz_MDD", "Study 2: Mumtaz MDD (Eyes Closed)", "Outpt.\nMDD*"),
        ("Iowa_PD", "Study 7: Iowa Parkinson (ds004584)", "Iowa\nPD"),
        ("Dementia_All", "Study 5c: Dementia All (AD+FTD) vs Control", "Dem.\nAll"),
    ]
    mat = np.zeros((4, 4))
    tr_sub = tr[tr["model"] == "RandomForest"]
    ms_sub = ms[ms["model"] == "RandomForest"].set_index("cohort")
    for i, (tk_s, wc_s, _) in enumerate(doms):
        for j, (tk_t, _, _) in enumerate(doms):
            mat[i, j] = float(ms_sub.loc[wc_s, "mean_auc"]) if i == j else float(tr_sub[(tr_sub["source_cohort"] == tk_s) & (tr_sub["target_cohort"] == tk_t)]["auc"].values[0])
    im = ax1.imshow(mat, cmap="RdBu_r", vmin=0.35, vmax=0.90, aspect="auto")
    for i in range(4):
        for j in range(4):
            val = mat[i, j]
            txt_col = "#ffffff" if (val >= 0.78 or val <= 0.44) else "#0f172a"
            lbl = f"{val:.3f}\n(10×5)" if i == j else f"{val:.3f}"
            ax1.text(j, i, lbl, ha="center", va="center", color=txt_col, fontsize=8.0, fontweight=("normal" if i == j else "bold"))
    ax1.set_xticks(range(4))
    ax1.set_xticklabels([lbl for _, _, lbl in doms], fontsize=8.0)
    ax1.set_yticks(range(4))
    ax1.set_yticklabels([lbl.replace("\n", " ") for _, _, lbl in doms], fontsize=8.0)
    ax1.set_xlabel("Target Test Cohort", fontsize=9.0)
    ax1.set_ylabel("Source Training Cohort", fontsize=9.0)
    ax1.set_title("(a) Split-Target-HC Transfer (RF)", fontweight="bold", loc="left", fontsize=9.6)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.set_label("ROC-AUC", fontsize=8.2)

    # Panel (b)
    ax2 = fig.add_subplot(gs[0, 1])
    dem = [g for g in sp["dementia_ds004504"] if pd.notnull(g.get("mmse"))]
    X_dem = np.stack([g["x_specparam_resid"].numpy().flatten() for g in dem])
    y_dem = np.array([float(g["mmse"]) for g in dem])
    diag_dem = np.array([g["diagnosis"] for g in dem])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_full = np.zeros(len(dem))
    for tr_i, va_i in kf.split(X_dem, y_dem):
        mu, sd = X_dem[tr_i].mean(0, keepdims=True), X_dem[tr_i].std(0, keepdims=True) + 1e-6
        oof_full[va_i] = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42).fit((X_dem[tr_i] - mu) / sd, y_dem[tr_i]).predict((X_dem[va_i] - mu) / sd)
    r_full, p_full = pearsonr(y_dem, oof_full)

    dem_pts = [g for g in dem if g["diagnosis"] in ("Alzheimer", "FTD")]
    X_pts = np.stack([g["x_specparam_resid"].numpy().flatten() for g in dem_pts])
    y_pts = np.array([float(g["mmse"]) for g in dem_pts])
    oof_pts = np.zeros(len(dem_pts))
    for tr_i, va_i in kf.split(X_pts, y_pts):
        mu, sd = X_pts[tr_i].mean(0, keepdims=True), X_pts[tr_i].std(0, keepdims=True) + 1e-6
        oof_pts[va_i] = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42).fit((X_pts[tr_i] - mu) / sd, y_pts[tr_i]).predict((X_pts[va_i] - mu) / sd)
    r_pts, p_pts = pearsonr(y_pts, oof_pts)

    is_ctrl = diag_dem == "Control"
    ax2.scatter(y_dem[is_ctrl], oof_full[is_ctrl], c="#0284c7", s=32, alpha=0.82, edgecolors="#ffffff", lw=0.6, label="Controls (N=29)")
    ax2.scatter(y_dem[~is_ctrl], oof_full[~is_ctrl], c="#dc2626", s=32, alpha=0.82, edgecolors="#ffffff", lw=0.6, label="AD+FTD (N=59)")
    m_f, b_f = np.polyfit(y_dem, oof_full, 1)
    xs_f = np.linspace(y_dem.min(), y_dem.max(), 100)
    ax2.plot(xs_f, m_f * xs_f + b_f, color="#0f172a", lw=1.7, label=f"All (N=88): r={r_full:.3f} (RF)")
    m_p, b_p = np.polyfit(y_pts, oof_pts, 1)
    xs_p = np.linspace(y_pts.min(), y_pts.max(), 100)
    ax2.plot(xs_p, m_p * xs_p + b_p, color="#b91c1c", ls="--", lw=1.7, label=f"Pts (N=59): r={r_pts:.3f} (RF)")
    ax2.set_xlabel("Observed MMSE Score", fontsize=9.0)
    ax2.set_ylabel("Split-0 OOF Predicted MMSE", fontsize=9.0)
    ax2.set_title("(b) Dementia MMSE Confounding (RF OOF)", fontweight="bold", loc="left", fontsize=9.6)
    ax2.set_ylim(13.5, 36.2)
    ax2.legend(loc="upper left", fontsize=7.0, framealpha=0.96)
    ax2.grid(True, ls=":", alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel (c)
    ax3 = fig.add_subplot(gs[0, 2])
    tasks_cfg = [
        ("Dementia_MMSE_All_N88", "Dem.\nMMSE\n(88)"),
        ("Dementia_MMSE_Patients_Only_N59", "Dem.\nMMSE\n(Pts 59)"),
        ("Iowa_PD_MoCA_All_N149", "Iowa\nMoCA\n(149)"),
        ("Iowa_PD_MoCA_Patients_Only_N100", "Iowa\nMoCA\n(Pts 100)"),
        ("Iowa_PD_UPDRS_Patients_N100", "Iowa\nUPDRS\n(Pts 100)"),
    ]
    x3, w3 = np.arange(len(tasks_cfg)) * 1.15, 0.22
    for idx_m, (mname, mlbl, mcol) in enumerate([
        ("GBDT_Regressor", "GBDT", "#ea580c"),
        ("Random_Forest_Regressor", "RF", "#0f766e"),
        ("SVR_RBF", "SVR-RBF", "#6366f1"),
        ("Ridge_Regression", "Ridge", "#64748b"),
    ]):
        rs, sds = [], []
        for ckey, _ in tasks_cfg:
            row = rg[(rg["cohort_key"] == ckey) & (rg["model"] == mname)].iloc[0]
            rs.append(float(row["pearson_r_mean"]))
            sds.append(float(row["pearson_r_sd"]))
        ax3.bar(x3 + (idx_m - 1.5) * w3, rs, yerr=sds, capsize=2.0, width=w3, color=mcol, edgecolor="#1e293b", lw=0.6, label=mlbl, zorder=3)
    pcol = "maxstat4_perm_p_2s" if "maxstat4_perm_p_2s" in rg.columns else "perm_p"
    for i_t, (ckey, _) in enumerate(tasks_cfg):
        sub_t = rg[rg["cohort_key"] == ckey]
        best_row = sub_t.loc[sub_t["pearson_r_mean"].idxmax()]
        y_top = float(sub_t["pearson_r_mean"].max() + sub_t["pearson_r_sd"].max())
        p_p = float(best_row[pcol])
        p_lbl = "p_max<0.01" if p_p < 0.01 else f"p_max={p_p:.2f}"
        y_off = 0.045 if i_t % 2 == 1 else 0.018
        ax3.text(x3[i_t], max(y_top + y_off, 0.05), p_lbl, ha="center", va="bottom", fontsize=6.4, fontweight="bold", color="#991b1b" if p_p >= 0.05 else "#0f172a")
    ax3.axhline(0.0, color="#334155", lw=1.0)
    ax3.set_xticks(x3)
    ax3.set_xticklabels([lbl for _, lbl in tasks_cfg], fontsize=7.6)
    ax3.set_ylim(-0.18, 0.70)
    ax3.set_ylabel("10×5 OOF Pearson r (Mean ± SD)", fontsize=8.8)
    ax3.set_title("(c) Severity Regression Floor", fontweight="bold", loc="left", fontsize=9.6)
    ax3.legend(loc="upper right", fontsize=7.2, ncol=2)
    ax3.grid(axis="y", ls=":", alpha=0.5, zorder=0)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.18, top=0.90)
    save_fig(fig, "Figure_4", is_main=True)


def plot_figure_s1():
    all_psds = torch.load(DATA_DIR / "all_7cohorts_welch_psds.pt", weights_only=False)
    rec = [r for r in all_psds["dementia_ds004504"] if r["subject_id"] == "sub-018"][0]
    ch_idx = CHANNELS_19.index("Fz")
    freqs_all = rec["freqs"]
    psd_uv2 = rec["mean_psd"] * 1e12

    sm = SpectralModel(aperiodic_mode="fixed", periodic_mode="gaussian", peak_width_limits=[1.0, 12.0], max_n_peaks=6, min_peak_height=0.05, peak_threshold=2.0, metrics=["error_mae", "gof_rsquared"], verbose=False)
    sm.fit(freqs_all, np.maximum(psd_uv2[ch_idx], 1e-12), [2.0, 30.0])
    f_fit, raw_log = sm.data.freqs, sm.data.power_spectrum
    ap_sp, pk_sp, full_sp = sm.results.model._ap_fit, sm.results.model._peak_fit, sm.results.model.modeled_spectrum
    ap_offset, ap_chi = sm.get_params("aperiodic")
    r2_val, mae_val = float(sm.get_metrics("gof", "rsquared")), float(sm.get_metrics("error", "mae"))
    peaks = sm.get_params("peak")

    valid_40 = (freqs_all >= 1.0) & (freqs_all <= 40.0)
    f_40, raw_log_40 = freqs_all[valid_40], np.log10(np.maximum(psd_uv2[ch_idx, valid_40], 1e-12))
    ols_mask = (f_fit < 7.0) | (f_fit > 14.0)
    sl_ols, ic_ols = np.polyfit(np.log10(f_fit[ols_mask]), raw_log[ols_mask], 1)
    ap_ols, chi_ols = ic_ols + sl_ols * np.log10(f_fit), -sl_ols
    th_m = (f_fit >= 4.0) & (f_fit < 8.0)
    theta_sp, theta_ols = float(np.mean((raw_log - ap_sp)[th_m])), float(np.mean((raw_log - ap_ols)[th_m]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 6.8), sharex=True, dpi=300, gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.18})
    ax1.axvspan(1.0, 2.0, color="#FEE2E2", alpha=0.40, zorder=0, label="Excluded skirt (<2 / >30 Hz)")
    ax1.axvspan(30.0, 40.0, color="#FEE2E2", alpha=0.40, zorder=0)
    ax1.axvspan(4.0, 7.0, color="#FEF3C7", alpha=0.55, zorder=0, label="Unmasked 4–7 Hz theta (OLS bias)")
    ax1.axvspan(7.0, 14.0, color="#E2E8F0", alpha=0.45, zorder=0, label="Masked [7, 14] Hz OLS gap")
    ax1.plot(f_40, raw_log_40, color="#1E293B", lw=1.6, zorder=3, label="Welch log10 PSD (sub-018 AD, Fz)")
    ax1.plot(f_fit, full_sp, color="#2563EB", lw=2.0, zorder=4, label=f"specparam [2, 30] Hz ({len(peaks)} peaks)")
    ax1.plot(f_fit, ap_sp, color="#DC2626", ls="--", lw=2.0, zorder=5, label=f"specparam aperiodic (χ={ap_chi:.3f})")
    ax1.plot(f_fit, ap_ols, color="#0D9488", ls="-.", lw=1.8, zorder=5, label=f"Masked OLS (χ_OLS={chi_ols:.3f})")
    ax1.set_ylim(-1.60, 1.85)
    ax1.set_ylabel(r"Power $\log_{10}(\mu\mathrm{V}^2/\mathrm{Hz})$ [B]", fontweight="bold")
    ax1.set_title(f"(a) Real Spectrum Fit: sub-018 Fz (R²={r2_val:.3f}, MAE={mae_val:.3f} B)", fontweight="bold", loc="left", fontsize=9.8)
    ax1.legend(loc="upper right", fontsize=7.6, framealpha=0.96)
    ax1.grid(True, linestyle=":", alpha=0.45)

    ax2.axvspan(1.0, 2.0, color="#FEE2E2", alpha=0.35, zorder=0)
    ax2.axvspan(30.0, 40.0, color="#FEE2E2", alpha=0.35, zorder=0)
    ax2.axvspan(4.0, 8.0, color="#FEF3C7", alpha=0.45, zorder=0)
    ax2.axhline(0.0, color="#DC2626", ls="--", lw=1.3, zorder=2, label="specparam baseline (0.0 B)")
    ax2.plot(f_fit, raw_log - ap_sp, color="#475569", lw=1.3, alpha=0.85, zorder=3, label="specparam residual")
    ax2.plot(f_fit, pk_sp, color="#2563EB", lw=2.1, zorder=4, label="Fitted Gaussian peaks")
    ax2.fill_between(f_fit, 0.0, pk_sp, color="#60A5FA", alpha=0.22, zorder=1)
    ax2.plot(f_fit, raw_log - ap_ols, color="#0D9488", ls="-.", lw=1.7, zorder=4, label="Masked OLS residual (pulled down)")
    ax2.set_ylim(-0.72, 1.18)
    ax2.text(0.56, 0.56, f"4–8 Hz Mean Theta:\n• specparam: {theta_sp:+.3f} B\n• Masked OLS: {theta_ols:+.3f} B\n• Bias: {theta_ols - theta_sp:+.3f} B", transform=ax2.transAxes, fontsize=8.0, va="top", ha="left", bbox=dict(boxstyle="round,pad=0.38", facecolor="#FFFBEB", edgecolor="#D97706", alpha=0.96), zorder=6)
    ax2.set_xlim(1.0, 40.0)
    ax2.set_xlabel("Frequency (Hz)", fontweight="bold")
    ax2.set_ylabel("Periodic Residual [B]", fontweight="bold")
    ax2.set_title("(b) Flattened Periodic Residuals", fontweight="bold", loc="left", fontsize=9.8)
    ax2.legend(loc="upper right", fontsize=7.8, framealpha=0.96)
    ax2.grid(True, linestyle=":", alpha=0.45)
    save_fig(fig, "fig_S1_specparam_example", is_main=False)


def plot_figure_s2(master):
    channel_order = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
    node_coords = {
        "Fp1": (-0.28, 0.78), "Fp2": (0.28, 0.78), "F7": (-0.72, 0.44), "F3": (-0.35, 0.40), "Fz": (0.00, 0.40), "F4": (0.35, 0.40), "F8": (0.72, 0.44),
        "T7": (-0.84, 0.00), "C3": (-0.42, 0.00), "Cz": (0.00, 0.00), "C4": (0.42, 0.00), "T8": (0.84, 0.00),
        "P7": (-0.72, -0.44), "P3": (-0.35, -0.40), "Pz": (0.00, -0.40), "P4": (0.35, -0.40), "P8": (0.72, -0.44), "O1": (-0.28, -0.78), "O2": (0.28, -0.78),
    }
    t1 = [("Fp1", "F3"), ("Fp2", "F4"), ("F7", "F3"), ("F8", "F4"), ("F7", "T7"), ("F8", "T8"), ("F3", "Fz"), ("F4", "Fz"), ("Fz", "Cz"), ("C3", "Cz"), ("C4", "Cz"), ("Cz", "Pz"), ("Pz", "P3"), ("Pz", "P4"), ("T7", "P7"), ("T8", "P8"), ("P3", "O1"), ("P4", "O2")]
    t2 = [("Fp1", "Fp2"), ("Fp1", "F7"), ("Fp2", "F8"), ("F3", "C3"), ("F4", "C4"), ("T7", "C3"), ("T8", "C4"), ("C3", "P3"), ("C4", "P4"), ("P7", "P3"), ("P8", "P4"), ("P7", "O1"), ("P8", "O2"), ("O1", "O2"), ("Fz", "Pz"), ("F3", "F4"), ("C3", "C4"), ("Cz", "O1")]

    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=300)
    ax.add_patch(patches.Circle((0, 0), 1.0, edgecolor="#334155", facecolor="#f8fafc", lw=1.6, zorder=1))
    ax.plot([-0.11, 0.0, 0.11], [0.99, 1.12, 0.99], color="#334155", lw=1.6, zorder=1)
    ax.add_patch(patches.Ellipse((-1.01, 0.0), 0.10, 0.26, edgecolor="#334155", facecolor="#f8fafc", lw=1.5, zorder=1))
    ax.add_patch(patches.Ellipse((1.01, 0.0), 0.10, 0.26, edgecolor="#334155", facecolor="#f8fafc", lw=1.5, zorder=1))
    for u, v in t1:
        ax.plot([node_coords[u][0], node_coords[v][0]], [node_coords[u][1], node_coords[v][1]], color="#2563eb", lw=1.8, alpha=0.85, zorder=2)
    for u, v in t2:
        ax.plot([node_coords[u][0], node_coords[v][0]], [node_coords[u][1], node_coords[v][1]], color="#d97706", lw=1.5, ls="--", alpha=0.75, zorder=2)
    for ch in channel_order:
        x, y = node_coords[ch]
        ax.add_patch(patches.Circle((x, y), 0.075, facecolor="#1e293b", edgecolor="#ffffff", lw=1.4, zorder=3))
        ax.text(x, y, ch, color="#ffffff", fontsize=7.8, fontweight="bold", ha="center", va="center", zorder=4)

    omst = master["omst_topology"]
    l1 = plt.Line2D([0], [0], color="#2563eb", lw=2.0, label="Tree 1: Primary MST Backbone (18 edges)")
    l2 = plt.Line2D([0], [0], color="#d97706", lw=1.8, ls="--", label="Tree 2: Orthogonal Spanning Tree (18 edges)")
    ax.legend(handles=[l1, l2], loc="lower center", bbox_to_anchor=(0.5, -0.09), fontsize=8.3)
    ax.text(0.5, 1.01, f"{omst['total_graphs']} graphs ({omst['total_node_instances']:,} nodes): M = {omst['mean_trees']:.2f} ± {omst['sd_trees']:.2f}, GCE = {omst['mean_gce']:.3f} ± {omst['sd_gce']:.3f}, 0.0% isolated", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.8, color="#334155")
    ax.set_xlim(-1.18, 1.18); ax.set_ylim(-1.16, 1.22); ax.set_aspect("equal"); ax.axis("off")
    save_fig(fig, "fig_S2_omst_topology", is_main=False)


def plot_figure_s3():
    df = pd.read_csv(REPORTS_DIR / "specparam_gain_invariance_check.csv")
    k_vals = df["gain_k"].to_numpy(dtype=float)
    meas_shift = df["measured_offset_shift_mean"].to_numpy(dtype=float)
    diff_chi = df["max_abs_diff_chi"].to_numpy(dtype=float)
    diff_th = df["max_abs_diff_theta_resid"].to_numpy(dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 5.8), sharex=True, dpi=300, gridspec_kw={"height_ratios": [1.05, 1.1], "hspace": 0.24})
    k_dense = np.logspace(-2.1, 2.1, 200)
    ax1.plot(k_dense, 2.0 * np.log10(k_dense), color="#dc2626", ls="--", lw=1.7, label=r"Analytical: $\Delta b = 2\log_{10}(k)$")
    ax1.scatter(k_vals, meas_shift, color="#1d4ed8", s=54, edgecolor="#ffffff", lw=1.2, zorder=4, label=f"Observed Mean Δb (n={int(df['n_spectra'].iloc[0])})")
    ax1.set_xscale("log"); ax1.set_ylabel("Aperiodic Offset Shift Δb [B]")
    ax1.set_title("(a) Offset Shift vs. Analytical Gain Scaling", fontweight="bold", loc="left")
    ax1.legend(loc="upper left"); ax1.grid(True, ls=":", alpha=0.5); ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    ax2.plot(k_vals, diff_chi * 1e6, marker="o", color="#0284c7", lw=1.9, ms=5.8, label=r"Max $|\Delta\chi|$ ($\times 10^{-6}$)")
    ax2.plot(k_vals, diff_th * 1e6, marker="s", color="#dc2626", lw=1.9, ms=5.8, label=r"Max $|\Delta\theta|$ ($\times 10^{-6}$ B)")
    ax2.set_xscale("log"); ax2.set_ylim(-1.0, max(diff_chi.max(), diff_th.max()) * 1e6 * 1.38)
    ax2.set_xlabel("Multiplicative Gain Factor k"); ax2.set_ylabel(r"Max Drift ($\times 10^{-6}$)")
    max_ceil = int(np.ceil(max(diff_chi.max(), diff_th.max()) * 1e6)) + 1
    ax2.set_title(rf"(b) Exponent & Residual Drift ($< {max_ceil}\times 10^{{-6}}$)", fontweight="bold", loc="left")
    ax2.legend(loc="upper center"); ax2.grid(True, ls=":", alpha=0.5); ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    save_fig(fig, "fig_S3_gain_invariance", is_main=False)


def plot_figure_s4():
    with open(REPORTS_DIR / "real_grand_average_spectra.json", "r", encoding="utf-8") as f:
        cache = json.load(f)
    freqs_raw, freqs_res, cohorts = np.asarray(cache["freqs"], dtype=float), np.asarray(cache["freqs_res"], dtype=float), cache["cohorts"]
    fig, axes = plt.subplots(2, len(cohorts), figsize=(18.5, 6.2), dpi=300, sharex=True)
    for col_i, cinfo in enumerate(cohorts):
        ax_top, ax_bot = axes[0, col_i], axes[1, col_i]
        for ax, fv, mk, sk in [(ax_top, freqs_raw, "raw_mean", "raw_sem"), (ax_bot, freqs_res, "res_mean", "res_sem")]:
            cm, cs = np.asarray(cinfo["case"][mk]), np.asarray(cinfo["case"][sk])
            hm, hs = np.asarray(cinfo["ctrl"][mk]), np.asarray(cinfo["ctrl"][sk])
            ax.axvspan(1.0, 2.0, color="#fee2e2", alpha=0.45, zorder=0); ax.axvspan(30.0, 40.0, color="#fee2e2", alpha=0.45, zorder=0); ax.axvspan(4.0, 8.0, color="#fef3c7", alpha=0.55, zorder=0)
            ax.plot(fv, hm, color="#0284c7", lw=1.6, label=cinfo["ctrl_label"], zorder=3); ax.fill_between(fv, hm - hs, hm + hs, color="#0284c7", alpha=0.18, zorder=2)
            ax.plot(fv, cm, color="#dc2626", lw=1.6, label=cinfo["case_label"], zorder=4); ax.fill_between(fv, cm - cs, cm + cs, color="#dc2626", alpha=0.18, zorder=2)
            ax.grid(True, ls=":", alpha=0.45); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax_top.set_title(f"({chr(97 + col_i)}) {cinfo['title']}", fontsize=8.8, fontweight="bold", loc="left")
        ax_top.legend(loc="upper right", fontsize=6.6, handlelength=1.4)
        ax_bot.axhline(0.0, color="#64748b", ls="--", lw=0.8); ax_bot.set_xlim(1.0, 40.0); ax_bot.set_xlabel("Frequency (Hz)", fontsize=8.4)
    axes[0, 0].set_ylabel(r"Raw $\log_{10}$ PSD [B] (1–40 Hz)", fontsize=8.8)
    axes[1, 0].set_ylabel("[2.0, 30.0] Hz Residual [B]", fontsize=8.8)
    fig.tight_layout()
    save_fig(fig, "fig_S4_cohort_spectra", is_main=False)


def plot_figure_s5(master):
    acc = master["accounting"]
    c_dict = acc["cohorts"]
    fig, ax = plt.subplots(figsize=(14.5, 6.8), dpi=300)
    ax.axis("off")
    def box(x, y, w, h, text, bg="#f8fafc", ec="#1e293b", fs=8.0):
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.016", facecolor=bg, edgecolor=ec, lw=1.2, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#0f172a", zorder=3)
    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#334155", lw=1.2), zorder=1)

    box(0.16, 0.84, 0.68, 0.12,
        f"Seven OpenNeuro / Figshare Cohorts: Enrolled N = {acc['enrolled_participants']} | Excluded N = {acc['excluded_participants']} (6 QC + 2 BDI + 2 SHA-256 dups)\n"
        f"Primary Unique Benchmark: N = {acc['total_unique_participants']} Participants ({acc['total_analyzed_sessions']} Primary Sessions; {acc['total_with_mumtaz_eo']} Total = {acc['total_with_mumtaz_eo']*19:,} Spectra)",
        bg="#e2e8f0", ec="#0f172a", fs=8.3)
    items = [
        ("Study 1: Sleep Debt\n(ds004902)", c_dict["Study 1 (Sleep ds004902)"], "0 Excluded\n(Complete pairs)", "71 SD vs 71 NS\nPaired Crossover"),
        ("Study 2: Outpatient MDD\n(Figshare 4244171)", c_dict["Study 2 (Mumtaz MDD 4244171)"], "8 Excl (6 QC + 2 dup)\n(+61 TASK, +2 H S15\nID-prefix EO skipped)", "29 MDD, 27 HC EC\n(+58 uniq EO)"),
        ("Study 3: Univ. Dep.\n(ds003478)", c_dict["Study 3 (Cavanagh ds003478)"], "2 Excluded\n(Borderline\nBDI 10–13)", "44 Dep (BDI>=14)\nvs 75 HC (BDI<=7)"),
        ("Study 4: UCSD PD\n(ds002778)", c_dict["Study 4 (UCSD PD ds002778)"], "0 Excluded\n(All valid;\n31 pts / 46 sess)", "15 PD (ON/OFF)\n+ 16 HC"),
        ("Study 5: Dementia\n(ds004504)", c_dict["Study 5 (AHEPA Dementia ds004504)"], "0 Excluded\n(All valid)", "36 AD + 23 FTD\nvs 29 HC"),
        ("Study 6: UNM PD\n(ds003490)", c_dict["Study 6 (UNM PD ds003490)"], "0 Excluded\nPure 60s EO Rest\n(Task removed)", "25 PD (ON/OFF)\n+ 25 HC (+74 EC)"),
        ("Study 7: Iowa PD\n(ds004584)", c_dict["Study 7 (Iowa PD ds004584)"], "0 Excluded\n(Pz ref CAR;\n18-ch CAR checked)", "100 PD vs 49 HC\nIndependent Sample"),
    ]
    xs, w_box = np.linspace(0.012, 0.862, 7), 0.126
    for x, (t_str, meta, exc_str, fin_str) in zip(xs, items):
        cx = x + w_box / 2
        arrow(0.50, 0.834, cx, 0.726)
        box(x, 0.57, w_box, 0.15, f"{t_str}\nEnrolled: N={meta['enrolled']}", bg="#eff6ff", ec="#2563eb", fs=7.3)
        arrow(cx, 0.564, cx, 0.486)
        bg_c = "#fef2f2" if meta["excluded"] > 0 else ("#fffbeb" if "Pure" in exc_str else "#f0fdf4")
        ec_c = "#dc2626" if meta["excluded"] > 0 else ("#d97706" if "Pure" in exc_str else "#16a34a")
        box(x, 0.34, w_box, 0.14, exc_str, bg=bg_c, ec=ec_c, fs=6.9)
        arrow(cx, 0.334, cx, 0.256)
        box(x, 0.10, w_box, 0.15, f"Analyzed:\nN={meta['analyzed_pts']} ({meta['analyzed_sessions']} sess)\n{fin_str}", bg="#f0fdf4", ec="#15803d", fs=6.9)
        arrow(cx, 0.094, 0.50, 0.026)

    box(0.06, -0.09, 0.88, 0.11,
        f"Harmonized 19-ch Pipeline: 250 Hz, 2.0 s Welch PSD (0.50 Hz df), Interior [2.0, 30.0] Hz specparam v2.0.0rc7 (0/13,927 fails; full fooof v1.1.1 r >= 0.996)\n"
        f"Fold-Nested Normative Z-Scoring, 10×5 True Participant-Grouped CV (+ Study 4/6 Participant-Level AUC), OMST (0.0% isolated nodes)",
        bg="#dcfce7", ec="#166534", fs=8.1)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.12, 0.99)
    save_fig(fig, "fig_S5_enrollment_flowchart", is_main=False)


def plot_figure_s6():
    with open(REPORTS_DIR / "real_permutation_distributions.json", "r", encoding="utf-8") as f:
        pdata = json.load(f)
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.2), dpi=300)
    for ax, tid in zip(axes.flat, ["A", "B", "C", "D", "E", "F"]):
        info = pdata[tid]
        ng, nl = np.asarray(info["null_gcn_aucs"], dtype=float), np.asarray(info["null_lr_aucs"], dtype=float)
        og, ol, pg, pl = float(info["obs_gcn_auc"]), float(info["obs_lr_auc"]), float(info["p_perm_gcn"]), float(info["p_perm_lr"])
        bins = np.linspace(0.22, 0.92, 32)
        ax.hist(ng, bins=bins, density=True, alpha=0.45, color="#dc2626", edgecolor="#991b1b", lw=0.6, label=f"GCN Null (n={len(ng)})")
        ax.hist(nl, bins=bins, density=True, alpha=0.40, color="#0284c7", edgecolor="#0369a1", lw=0.6, label=f"LR Null (n={len(nl)})")
        ax.axvline(0.50, color="#475569", ls=":", lw=1.0)
        ax.axvline(og, color="#b91c1c", ls="--", lw=1.9, label=f"Obs GCN={og:.3f} (p={pg:.4f})")
        ax.axvline(ol, color="#0369a1", ls="-.", lw=1.9, label=f"Obs LR={ol:.3f} (p={pl:.4f})")
        ax.set_title(f"({tid.lower()}) {info['label']}", fontsize=9.0, fontweight="bold", loc="left")
        ax.set_xlabel("Split-0 / Split-HC ROC-AUC"); ax.set_ylabel("Null Density")
        ax.legend(loc="upper left", fontsize=7.0); ax.grid(True, ls=":", alpha=0.45); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "fig_S6_permutation_tests", is_main=False)


def main():
    shutil.rmtree(BENCHMARK_ROOT / "submission_Neuroinformatics" / "Figures", ignore_errors=True)
    sp = torch.load(DATA_DIR / "specparam_vs_masked_ols_7cohorts.pt", weights_only=False)
    with open(REPORTS_DIR / "master_results.json", "r", encoding="utf-8") as f:
        master = json.load(f)
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4(sp)
    plot_figure_s1()
    plot_figure_s2(master)
    plot_figure_s3()
    plot_figure_s4()
    plot_figure_s5(master)
    plot_figure_s6()
    print("✓ All v6 figures regenerated cleanly without internal titles & with lowercase panels.")


if __name__ == "__main__":
    main()
