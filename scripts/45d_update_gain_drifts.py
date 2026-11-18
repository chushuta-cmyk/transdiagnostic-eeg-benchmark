#!/usr/bin/env python3
"""
scripts/45d_update_gain_drifts.py
---------------------------------
Enriches benchmark_gain_and_calibration_ablation.csv with BOTH session-mean
(|Δfeat|, |Δp_LR|, |Δp_RF|) AND worst-case max drift across the 6 PD transfer pairs.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "derivatives"
REPORTS_DIR = ROOT / "reports"


def build_gain_feats(recs, ftype, gain_k):
    feats = []
    for r, p_rec in recs:
        psd = p_rec["mean_psd"] * (gain_k ** 2)
        freqs = p_rec["freqs"]
        f30 = (freqs >= 2.0) & (freqs <= 30.0)
        fv = freqs[f30]
        if ftype == "specparam_periodic_residual":
            x = r["x_specparam_resid"].numpy().copy()
        elif ftype == "masked_ols_residual":
            x = r["x_masked_ols_win"].numpy().copy()
        elif ftype == "raw_log_band_power":
            th_m = (fv >= 4.0) & (fv < 8.0)
            dl_m = (fv >= 2.0) & (fv < 4.0)
            al_m = (fv >= 8.0) & (fv <= 13.0)
            lp = np.log10(np.maximum(psd[:, f30], 1e-25))
            x = np.zeros((19, 4), dtype=np.float32)
            x[:, 0] = lp[:, th_m].mean(axis=1)
            x[:, 1] = lp[:, dl_m].mean(axis=1)
            x[:, 2] = lp[:, al_m].mean(axis=1)
            x[:, 3] = r["x_specparam_resid"].numpy()[:, 3]
        elif ftype == "relative_band_power":
            th_m = (fv >= 4.0) & (fv < 8.0)
            dl_m = (fv >= 2.0) & (fv < 4.0)
            al_m = (fv >= 8.0) & (fv <= 13.0)
            tot = np.maximum(psd[:, f30].sum(axis=1, keepdims=True), 1e-25)
            x = np.zeros((19, 4), dtype=np.float32)
            x[:, 0] = np.log10(np.maximum(psd[:, f30][:, th_m].mean(axis=1, keepdims=True) / tot, 1e-12)).ravel()
            x[:, 1] = np.log10(np.maximum(psd[:, f30][:, dl_m].mean(axis=1, keepdims=True) / tot, 1e-12)).ravel()
            x[:, 2] = np.log10(np.maximum(psd[:, f30][:, al_m].mean(axis=1, keepdims=True) / tot, 1e-12)).ravel()
            x[:, 3] = r["x_specparam_resid"].numpy()[:, 3]
        feats.append(x.flatten())
    return np.asarray(feats, dtype=np.float64)


def main():
    sp = torch.load(DATA_DIR / "specparam_vs_masked_ols_7cohorts.pt", weights_only=False)
    psds = torch.load(DATA_DIR / "all_7cohorts_welch_psds.pt", weights_only=False)
    pd_sp_psds = {
        "UCSD": list(zip(sp["ucsd_ds002778"], psds["ucsd_ds002778"])),
        "UNM_OFF": [(r, p) for r, p in zip(sp["unm_ds003490"], psds["unm_ds003490"]) if r["med_state"] in ("OFF", "CTL")],
        "Iowa": list(zip(sp["iowa_ds004584"], psds["iowa_ds004584"])),
    }

    gain_rows = []
    for sn in pd_sp_psds:
        for tn in pd_sp_psds:
            if sn == tn:
                continue
            s_pairs, t_pairs = pd_sp_psds[sn], pd_sp_psds[tn]
            ys = np.array([int(r["y"]) for r, _ in s_pairs])
            yt = np.array([int(r["y"]) for r, _ in t_pairs])
            for ftype in ["specparam_periodic_residual", "masked_ols_residual", "relative_band_power", "raw_log_band_power"]:
                Xs_base = build_gain_feats(s_pairs, ftype, 1.0)
                Xt_base = build_gain_feats(t_pairs, ftype, 1.0)
                lr_u = LogisticRegression(C=0.1, max_iter=1000, random_state=42).fit(Xs_base, ys)
                rf_u = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(Xs_base, ys)
                p_lr_ref = lr_u.predict_proba(Xt_base)[:, 1]
                p_rf_ref = rf_u.predict_proba(Xt_base)[:, 1]

                for k_lbl, k_fn in [
                    ("0.5", lambda n: np.full(n, 0.5)),
                    ("1.0", lambda n: np.ones(n)),
                    ("2.0", lambda n: np.full(n, 2.0)),
                    ("5.0", lambda n: np.full(n, 5.0)),
                    ("Heterogeneous_0.5x_5.0x", lambda n: np.array([0.5 if i % 2 == 0 else 5.0 for i in range(n)])),
                ]:
                    ks = k_fn(len(t_pairs))
                    Xt_k = np.vstack([build_gain_feats([t_pairs[i]], ftype, ks[i]) for i in range(len(t_pairs))])
                    pr_lr = lr_u.predict_proba(Xt_k)[:, 1]
                    pr_rf = rf_u.predict_proba(Xt_k)[:, 1]
                    gain_rows.append({
                        "experiment": "Amplifier_Gain_Ablation", "source": sn, "target": tn, "calibration": "Uncalibrated_Raw",
                        "feature_type": ftype, "gain_k": k_lbl,
                        "lr_auc": float(roc_auc_score(yt, pr_lr)), "lr_bacc": float(balanced_accuracy_score(yt, (pr_lr >= 0.5).astype(int))), "lr_brier": float(brier_score_loss(yt, pr_lr)),
                        "rf_auc": float(roc_auc_score(yt, pr_rf)),
                        "mean_feat_drift": float(np.mean(np.abs(Xt_k - Xt_base))),
                        "max_feat_drift": float(np.max(np.abs(Xt_k - Xt_base))),
                        "lr_mean_prob_drift": float(np.mean(np.abs(pr_lr - p_lr_ref))),
                        "lr_max_prob_drift": float(np.max(np.abs(pr_lr - p_lr_ref))),
                        "rf_mean_prob_drift": float(np.mean(np.abs(pr_rf - p_rf_ref))),
                        "rf_max_prob_drift": float(np.max(np.abs(pr_rf - p_rf_ref))),
                    })
                for cal_lbl in ["Uncalibrated_Raw", "Global_Cohort_ZScore", "Target_HC_Normative"]:
                    if cal_lbl == "Uncalibrated_Raw":
                        Xs_c, Xt_c = Xs_base, Xt_base
                    elif cal_lbl == "Global_Cohort_ZScore":
                        Xs_c = (Xs_base - Xs_base.mean(0, keepdims=True)) / (Xs_base.std(0, keepdims=True) + 1e-6)
                        Xt_c = (Xt_base - Xt_base.mean(0, keepdims=True)) / (Xt_base.std(0, keepdims=True) + 1e-6)
                    else:
                        s_hc = Xs_base[ys == 0]; t_hc = Xt_base[yt == 0]
                        Xs_c = (Xs_base - s_hc.mean(0, keepdims=True)) / (s_hc.std(0, keepdims=True) + 1e-6)
                        Xt_c = (Xt_base - t_hc.mean(0, keepdims=True)) / (t_hc.std(0, keepdims=True) + 1e-6)
                    lr_c = LogisticRegression(C=0.1, max_iter=1000, random_state=42).fit(Xs_c, ys)
                    rf_c = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(Xs_c, ys)
                    pr_lc, pr_rc = lr_c.predict_proba(Xt_c)[:, 1], rf_c.predict_proba(Xt_c)[:, 1]
                    gain_rows.append({
                        "experiment": "Calibration_Strategy", "source": sn, "target": tn, "calibration": cal_lbl,
                        "feature_type": ftype, "gain_k": "1.0",
                        "lr_auc": float(roc_auc_score(yt, pr_lc)), "lr_bacc": float(balanced_accuracy_score(yt, (pr_lc >= 0.5).astype(int))), "lr_brier": float(brier_score_loss(yt, pr_lc)),
                        "rf_auc": float(roc_auc_score(yt, pr_rc)),
                        "mean_feat_drift": 0.0, "max_feat_drift": 0.0,
                        "lr_mean_prob_drift": 0.0, "lr_max_prob_drift": 0.0,
                        "rf_mean_prob_drift": 0.0, "rf_max_prob_drift": 0.0,
                    })
    df = pd.DataFrame(gain_rows)
    df.to_csv(REPORTS_DIR / "benchmark_gain_and_calibration_ablation.csv", index=False)
    print(df[df["feature_type"] == "raw_log_band_power"].groupby(["experiment", "calibration", "gain_k"], as_index=False).agg(
        lr_auc=("lr_auc", "mean"), lr_bacc=("lr_bacc", "mean"), lr_brier=("lr_brier", "mean"),
        rf_auc=("rf_auc", "mean"), mean_p_drift=("lr_mean_prob_drift", "mean"), max_p_drift=("lr_max_prob_drift", "max"),
    ).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
