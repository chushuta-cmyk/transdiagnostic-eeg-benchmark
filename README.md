# Open Benchmarking of Parameterized EEG Spectra and Graph Neural Networks Across Seven Public Cohorts Exposes Spectral Window Confounding and Connectivity Limits

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-orange.svg)](https://pytorch.org/)
[![specparam v2.0.0rc7](https://img.shields.io/badge/specparam-v2.0.0rc7-blue.svg)](https://specparam-tools.github.io/)

An open-source, leak-free computational benchmark evaluating **19-Channel Spatial Harmonization**, **Passband-Interior `[2.0, 30.0] Hz` Spectral Parameterization (`specparam v2.0.0rc7`)**, **Orthogonal Minimum Spanning Trees (OMST)**, and **Graph Neural Networks vs. Regularized Classical & Riemannian Baselines** across seven public resting-state EEG cohorts ($N = 564$ unique primary participants, $675$ primary sessions, $733$ total sessions including eyes-open recordings, $13{,}927$ unique channel spectra).

---

## Multi-Cohort Dataset Manifest

| Cohort | Study & Accession | Analyzed N (Sess / Pts) | Design & Clinical Focus | Native Array & Harmonized Rate |
| :--- | :--- | :---: | :--- | :--- |
| **Study 1** | Sleep Deprivation ([OpenNeuro `ds004902`](https://openneuro.org/datasets/ds004902)) | 142 / 71 | Within-subject crossover (36 h sleep deprivation vs. normal sleep, EO) | 64-ch BrainVision actiCHamp (500 → 250 Hz) |
| **Study 2** | Outpatient MDD ([Figshare `4244171`](https://doi.org/10.6084/m9.figshare.4244171)) | 56 / 56 (+58 EO uniq.; nom. 58/60) | DSM-IV MDD vs. controls; SHA-256 duplicate pairs & Jan-2013 HC batch audited | 19-ch clinical linked-ear (`-LE`) system (256 → 250 Hz) |
| **Study 3** | University Depression ([OpenNeuro `ds003478`](https://openneuro.org/datasets/ds003478)) | 119 / 119 | Student BDI-II ≥ 14 vs. ≤ 7 screening (EO) | 64-ch BioSemi ActiveTwo (500 → 250 Hz) |
| **Study 4** | UCSD Parkinson's Disease ([OpenNeuro `ds002778`](https://openneuro.org/datasets/ds002778)) | 46 / 31 | 15 PD paired ON/OFF vs. 16 HC (EO fixation) | 32-ch BioSemi ActiveTwo (512 → 250 Hz) |
| **Study 5** | Clinical Dementia ([OpenNeuro `ds004504`](https://openneuro.org/datasets/ds004504)) | 88 / 88 | 36 AD, 23 FTD, 29 HC with MMSE (EC) | 19-ch Nihon Kohden (500 → 250 Hz) |
| **Study 6** | UNM Parkinson's Disease ([OpenNeuro `ds003490`](https://openneuro.org/datasets/ds003490)) | 75 / 50 | 25 PD paired ON/OFF vs. 25 HC isolated cleanly from 60 s pure EO rest (+74 EC sens.) | 64-ch BrainVision (500 → 250 Hz) |
| **Study 7** | Iowa Parkinson's Disease ([OpenNeuro `ds004584`](https://openneuro.org/datasets/ds004584)) | 149 / 149 | 100 PD ON-medication vs. 49 HC with MoCA & UPDRS-III (EO) | 64-ch BrainVision, Pz-ref recovered (500 → 250 Hz) |

---

## Core Methodological Audits & Findings

1. **Spectral Fit-Window Stability (`105` 2D Grid Settings) & Raw Archive Forensics:**
   - Fits full-spectrum `specparam v2.0.0rc7` (cross-verified against legacy `fooof v1.1.1`, `r >= 0.992`) inside clean `[2.0, 30.0] Hz` interior bounds across a `105`-configuration 2D grid (`0` failed fits), resolving filter-skirt roll-off collapse (`Study 2` median `R² = 0.897 → 0.975`) while eliminating `[7, 14] Hz`-masked OLS bias (proven via empirical-variance-calibrated 150-trial Monte Carlo simulations and 1,000-bootstrap paired `Δd` contrasts).
   - Exposes exact byte-identical SHA-256 duplicate EDF files (`H S27` = `H S30`, `MDD S33` = `MDD S34`) and unfiltered `0.5–80 Hz` raw EDF batch confounding (`Apr–Jul 2013` MDD + `4` HCs share upstream roll-off; `Jan-2013` HCs show unnotched `50 Hz` artifacts and `θ d = +3.347` vs. `2011` HCs; excluding `Jan-2013` HCs eliminates Study 2 MDD frontal theta `d = -0.082, p = 0.752`), whereas PD/dementia theta slowing replicates reliably (`k = 4 d_RE = +0.787, p_HKSJ = 0.0118`).
2. **True 10×5 Participant-Grouped CV, `TOST` Equivalence & Spatial Covariance Baselines (`13` Models):**
   - Across `k = 8` strictly non-nested contrasts (and `k = 9` primary contrasts), `GCN-OMST` (`mean AUC = 0.705` / `k = 9: 0.717`) is statistically equivalent within `±0.02` AUC (`TOST p <= 0.0115`) to continuous `GCN-Weighted-PLI` (`0.704`), edge-matched `GCN-Random` (`0.709`, Wilcoxon `p = 0.4609`, `TOST p = 0.0017`), and `GCN-Full` (`0.707`, `TOST p = 0.0001`), trailing `Linear SVM` (`0.721`), `Tuned LR` (`0.721`), `Random Forest` (`0.720`), `Tuned Ridge` (`0.720`), and empirical `19 × 19` sample spatial covariance Riemannian tangent features (`0.724` RF).
3. **PLI Indefiniteness, Bidirectional Leakage & Severity Variance Floor:**
   - `79/675` (`11.70%`) symmetrized Phase Lag Index (`I_19 + PLI`) matrices violate positive semi-definiteness (`min λ = -7.370`), breaking PLI Log-Euclidean tangent representations (`0.598–0.604` vs. direct `19 × 19` spatial covariance `0.724` RF).
   - Session-mixed CV shifts 10×5 AUC bidirectionally (`-0.094` in discordant Sleep Deprivation crossover vs. `+0.080` in concordant UCSD PD repeated sessions).
   - Disjoint 50/50 `Split-Target-HC` transfer and `10×5` clinical regressions paired with `1,000`-permutation retraining tests show that removing ceiling controls abolishes dementia MMSE significance (`N = 59`, `perm p >= 0.111`, `OOF R² <= -0.047`), while `N = 100` Iowa PD patients retain weak correlation (`MoCA SVR r = +0.293, perm p = 0.0020`; `UPDRS-III GBDT r = +0.242, perm p = 0.0070`) under a universal `OOF R² <= 0.035` floor (`10/12` patient-only models `R² < 0`).

---

## Canonical Reproduction Pipeline

```bash
pip install -r release_artifacts/requirements_lock.txt

# 1. Isolate UNM ds003490 contiguous pure EO/EC rest, recompute canonical [2, 30] Hz & [1, 40] Hz specparam, audit Study 2 batch phase artifacts
python3 scripts/44_rebuild_unm_rest_and_diagnostics.py

# 2. Run complete v5 forensic & benchmarking pipeline (unfiltered Study 2 EDFs, SHA-256 deduplication, 105-window grid, fooof v1.1.1 check, calibrated MC sims, 13-model 10x5 CV w/ GCN-Weighted-PLI & fixed SVMs, TOST, PLI SPD & Spatial Cov Riemannian, 1,000-perm regressions, release export)
python3 scripts/47_run_v5_forensics_and_refinements.py

# 3. Generate publication Figures 1–4 and S1–S6 (.png & .pdf)
python3 scripts/fig_s1_real_specparam.py
python3 scripts/42_generate_all_specparam_figures.py

# 4. Build Neuroinformatics v5 manuscript, Online Resource 1 (Tables S1–S27), Editorial Manager DOCX package, and run strict verification audits
python3 scripts/48_build_paper2_neuroinformatics_v5.py
```

---

## Zenodo-Ready Release Artifacts

- `release_artifacts/derived_session_features_733.csv` — Derived per-session `[2.0, 30.0] Hz` periodic residuals, exponents, `IAPF`, clean epoch counts, and OMST topology metrics across all 733 unique recordings.
- `release_artifacts/cv_fold_indices_split0.csv` — Participant-grouped Split-0 outer CV fold assignments.
- `release_artifacts/oof_predictions_split0.csv` — Out-of-fold probability predictions (`GCN_OMST`, `Tuned_LR`, `RandomForest`).
- `release_artifacts/requirements_lock.txt` — Exact Python environment freeze.

---

## License

MIT License. See `LICENSE` for details.
