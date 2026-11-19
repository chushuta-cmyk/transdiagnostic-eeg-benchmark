# An Open Multi-Cohort EEG Benchmark for Spectral Parameterization, Sensor Graphs, and Reproducibility Audits

[![License: MIT (Code)](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](LICENSE)
[![License: CC BY 4.0 (Derived Artifacts)](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](release_artifacts/DATA_LICENSES.md)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-orange.svg)](https://pytorch.org/)
[![specparam v2.0.0rc7](https://img.shields.io/badge/specparam-v2.0.0rc7-blue.svg)](https://specparam-tools.github.io/)

An open-source, leak-free computational neuroinformatics benchmark evaluating **19-Channel Spatial Harmonization**, **Passband-Interior `[2.0, 30.0] Hz` Spectral Parameterization (`specparam v2.0.0rc7` & `fooof v1.1.1`)**, **Orthogonal Minimum Spanning Trees (OMST)**, and **Graph Neural Networks vs. Classical & Riemannian Baselines** across seven public resting-state EEG cohorts ($N = 564$ unique primary participants, $675$ primary sessions, $733$ total unique recordings, $13{,}927$ unique channel spectra).

> **Layered Licensing:** Original Python source code (`src/`, `scripts/`) is released under the **MIT License** ([LICENSE](LICENSE)). Derived benchmark CSVs and release manifests (`release_artifacts/`) are released under **CC BY 4.0** ([release_artifacts/DATA_LICENSES.md](release_artifacts/DATA_LICENSES.md)), preserving upstream OpenNeuro **CC0** (`ds004902`, `ds003478`, `ds002778`, `ds004504`, `ds003490`, `ds004584`) and Figshare **CC BY 4.0** (`4244171`) terms. Citation metadata is provided in [CITATION.cff](CITATION.cff).

---

## Multi-Cohort Dataset Manifest

| Cohort | Study & Accession | Upstream License | Analyzed N (Sess / Pts) | Design & Clinical Focus | Native Array & Harmonized Rate |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Study 1** | Sleep Deprivation ([OpenNeuro `ds004902`](https://openneuro.org/datasets/ds004902)) | CC0 | 142 / 71 | Within-subject crossover (36 h sleep deprivation vs. normal sleep, EO) | 64-ch BrainVision actiCHamp (500 → 250 Hz) |
| **Study 2** | Outpatient MDD ([Figshare `4244171`](https://doi.org/10.6084/m9.figshare.4244171)) | CC BY 4.0 | 56 / 56 (+58 EO uniq.; 181 EDFs audited) | DSM-IV MDD vs. controls; SHA-256 duplicates, ID-prefixed EO copies, & batches audited | 19-ch clinical linked-ear (`-LE`) system (256 → 250 Hz) |
| **Study 3** | University Depression ([OpenNeuro `ds003478`](https://openneuro.org/datasets/ds003478)) | CC0 | 119 / 119 | Student BDI-II ≥ 14 vs. ≤ 7 screening (EO) | 64-ch BioSemi ActiveTwo (500 → 250 Hz) |
| **Study 4** | UCSD Parkinson's Disease ([OpenNeuro `ds002778`](https://openneuro.org/datasets/ds002778)) | CC0 | 46 / 31 | 15 PD paired ON/OFF vs. 16 HC (EO fixation) | 32-ch BioSemi ActiveTwo (512 → 250 Hz) |
| **Study 5** | Clinical Dementia ([OpenNeuro `ds004504`](https://openneuro.org/datasets/ds004504)) | CC0 | 88 / 88 | 36 AD, 23 FTD, 29 HC with MMSE (EC) | 19-ch Nihon Kohden (500 → 250 Hz) |
| **Study 6** | UNM Parkinson's Disease ([OpenNeuro `ds003490`](https://openneuro.org/datasets/ds003490)) | CC0 | 75 / 50 | 25 PD paired ON/OFF vs. 25 HC isolated cleanly from 60 s pure EO rest (+74 EC sens.) | 64-ch BrainVision (500 → 250 Hz) |
| **Study 7** | Iowa Parkinson's Disease ([OpenNeuro `ds004584`](https://openneuro.org/datasets/ds004584)) | CC0 | 149 / 149 | 100 PD ON-medication vs. 49 HC with MoCA & UPDRS-III (EO); `Pz` ref recovered & 18-ch verified | 64-ch BrainVision, online `Pz` ref (500 → 250 Hz) |

---

## Core Methodological Audits & `v6` Findings

1. **Spectral Fit-Window Stability (`105` 2D Grid Settings), Full `fooof v1.1.1` Verification & Archive Forensics:**
   - Fits full-spectrum `specparam v2.0.0rc7` over clean interior `[2.0, 30.0] Hz` bounds across a `105`-setting 2D grid (`0` failed fits) and verifies exact convergence against stable `fooof v1.1.1` across **all `13,927` unique spectra** (`0` null fits, `r_chi = 0.9958`, `mean Abs(Δχ) = 0.0044`, `r_theta = 0.9994`, `mean Abs(Δθ) = 0.00039 B`).
   - Accounts completely for all `181` Figshare EDFs (`58` EC, `62` EO including `2` SHA-256 pairs and `2` ID-prefixed `H S15 EO` copies skipped, `61` TASK excluded) and uncovers `Apr–Jul 2013` upstream spectral roll-off batch confounding (`Jan-2013` HCs excluded: Study 2 theta `d = -0.082, p = 0.752`), whereas PD/dementia periodic theta elevation replicates reliably (`k = 4 d_RE = +0.787, p_HKSJ = 0.0118`) and remains invariant in Study 7 under strictly `18`-channel physical CAR (`d = +0.882` vs. `+0.867`) and spherical-spline `Pz` (`+0.861`).
2. **True `10×5` Participant-Grouped CV, Holm-Corrected `TOST` Equivalence, & Spatial Covariance Baselines (`13` Models):**
   - Across `k = 8` non-nested contrasts (`GCN-OMST = 0.705`; `k = 7 Excl. Study 2 = 0.684`), `GCN-OMST` is **Holm–Bonferroni `±0.02` `TOST`-equivalent** (`Holm p <= 0.0230`) to edge-matched `GCN-Random` (`0.709`, `Holm p = 0.0052`), `GCN-Full` (`0.707`, `Holm p = 0.0004`), and `GCN-Weighted-PLI` (`0.704`, `Holm p = 0.0230`), persisting when dropping shared-control `Study 5a` or `5b` (`k = 7`, `Holm p <= 0.0488`) and matching participant-averaged `10×5` AUCs in Studies `4` (`0.607` vs. `0.600`) and `6c` (`0.613` vs. `0.614`).
   - Classical tuned models (`Linear SVM = 0.721`, `Tuned LR = 0.721`, `RF = 0.720`; `0.702–0.703` excl. Study 2) and direct `19 × 19` sample spatial covariance Riemannian tangent features (`RF = 0.724` `k = 8`, `0.710` excl. Study 2; fold-wise Log-Euclidean training Fréchet mean identity-exact `Δ = 0.0000`) match or outperform all graph networks.
3. **PLI Indefiniteness, Bidirectional Leakage & Paired 4-Model / 12-Model `Max-Stat` Clinical Regressions:**
   - `79/675` (`11.70%`) symmetrized `I_19 + PLI` matrices fail positive semi-definiteness (`min λ = -7.370`), degrading PLI tangent representations (`0.598–0.604`).
   - Session-mixed CV shifts `10×5` AUC bidirectionally (`-0.094` in discordant Sleep Deprivation crossover vs. `+0.080` in concordant UCSD PD repeated sessions).
   - Paired `1,000`-permutation **4-model and 12-model patient-only `max-statistic` tests** abolish MMSE significance when healthy ceiling controls are removed (`N = 59`, `p_max4 >= 0.854`, `p_max12 >= 0.994`), while weak patient-only `Iowa MoCA SVR` (`r = +0.293`, `p_max4 = 0.041`, `p_max12 = 0.224`) and `UPDRS-III GBDT` (`r = +0.242`, `p_max4 = 0.085`, `p_max12 = 0.335`) share a universal `Mean OOF R² <= +0.035` floor (`10/12` negative).

---

## Canonical `v6` Reproduction Pipeline

```bash
pip install -r release_artifacts/requirements_lock.txt

# 1. Isolate UNM ds003490 pure EO/EC rest & primary Study 1–7 extracted features
python3 scripts/44_rebuild_unm_rest_and_diagnostics.py

# 2. Run v5 core forensic, 105-window, 13-model 10x5 CV, leakage, & SPD benchmarks
python3 scripts/47_run_v5_forensics_and_refinements.py

# 3. Run v6 sensitivity & max-stat audits (Study 7 18-ch/Spline, Study 4/6 part-level,
#    Holm TOST & Excl-S2, fold-wise Frechet check, 181-EDF log, full 13,927 fooof v1.1.1,
#    paired 4-model max-stat permutations, & release ID disambiguation)
python3 scripts/49_run_v6_sensitivities.py --all

# 4. Generate publication Figures 1–4 and S1–S6 (.png, .pdf, .eps)
python3 scripts/51_generate_v6_figures.py

# 5. Build Springer Neuroinformatics v6 Manuscript, Online Resource 1 (PDF + DOCX),
#    Online Resource 2 (XLSX), Editorial Manager attachments, and strict XML/table audits
python3 scripts/50_build_paper2_neuroinformatics_v6.py
```

---

## Release Artifacts & Submission Package

- `submission_Neuroinformatics/` — Complete Springer *Neuroinformatics* Editorial Manager submission package (`01_Cover_Letter`, `02_Research_Highlights`, `03_Title_Page`, `04_Main_Manuscript`, `05_Online_Resource_1_Neuroinformatics.pdf` + `.docx`, `05b_Online_Resource_2_Benchmark_Tables.xlsx`, `06_Declarations`, `07_Suggested_Reviewers`, `08_Submission_Checklist`, `Figures/`).
- `release_artifacts/derived_session_features_733.csv` — Derived `[2.0, 30.0] Hz` features across `733` unique recordings with explicit `session`, `participant_id`, and unique `record_uid`.
- `release_artifacts/cv_fold_indices_split0.csv` — Participant-grouped Split-0 outer CV fold assignments with unique `record_uid`.
- `release_artifacts/oof_predictions_split0.csv` — Out-of-fold probabilities (`GCN_OMST`, `Tuned_LR`, `RandomForest`) with unique `record_uid`.
- `release_artifacts/DATA_LICENSES.md` — Layered license manifest (`MIT` code / `CC BY 4.0` derivatives / upstream `CC0` & `CC BY 4.0`).
- `release_artifacts/requirements_lock.txt` — Exact Python environment freeze.
- `CITATION.cff` — Machine-readable citation metadata (`v6.0.0`).
