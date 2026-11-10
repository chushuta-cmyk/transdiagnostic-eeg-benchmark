#!/usr/bin/env python3
"""
Step 32: Build Clean, Substantively Revised Main Manuscript & Supplementary Material
(.md and .docx) with Verified Ground-Truth Statistics, Fixed Cross-References,
Sequential References with Hanging Indents, Validated OOXML (xmlns:m), and Automated Audits.

Author: Chushu Tan
China University of Political Science and Law
"""

import importlib
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.shared import Inches, Pt, RGBColor

BENCHMARK_ROOT = Path("/Users/tan/Documents/qwen-agent/aCAgqOVe0x/default/transdiagnostic_eeg_benchmark")
SCRIPTS_DIR = BENCHMARK_ROOT / "scripts"
MANUSCRIPT_DIR = BENCHMARK_ROOT / "manuscript"
REPORTS_DIR = BENCHMARK_ROOT / "reports"
ARTIFACT_DIR = Path("/Users/tan/.gemini/antigravity/brain/874c84f2-3204-455c-a632-daaf2336a273")

sys.path.insert(0, str(SCRIPTS_DIR))
tbl = importlib.import_module("32a_manuscript_tables")

V2_MD = MANUSCRIPT_DIR / "PAPER_2_MULTI_COHORT_TRANSDIAGNOSTIC_EEG_BENCHMARK_v2.md"
V2_DOCX = MANUSCRIPT_DIR / "PAPER_2_MULTI_COHORT_TRANSDIAGNOSTIC_EEG_BENCHMARK_v2.docx"
SUPP_MD = MANUSCRIPT_DIR / "PAPER_2_SUPPLEMENTARY_MATERIAL.md"
SUPP_DOCX = MANUSCRIPT_DIR / "PAPER_2_SUPPLEMENTARY_MATERIAL.docx"
SUPP_MD_COPY = MANUSCRIPT_DIR / "Supplementary_Material.md"
SUPP_DOCX_COPY = MANUSCRIPT_DIR / "Supplementary_Material.docx"

EXACT_TITLE = (
    "Participant-Level Benchmarking of Frontal Periodic Theta and Graph Neural Network Utility "
    "Across Seven Public 19-Channel Clinical EEG Cohorts"
)


def build_main_manuscript_markdown() -> str:
    t1 = tbl.build_main_table_1()
    t2 = tbl.build_main_table_2()
    t3 = tbl.build_main_table_3()
    t4 = tbl.build_main_table_4()

    body = rf"""# {EXACT_TITLE}

**Chushu Tan**¹*

¹ School of Sociology, China University of Political Science and Law, Beijing 102249, China  
* Correspondence: Guangzhou, Guangdong 510006, China; Email: chushuta@gmail.com; ORCID: 0009-0007-3093-0160

---

## Abstract

**Background:** Quantitative EEG (qEEG) cross-study comparisons are hindered by montage heterogeneity, periodic–aperiodic conflation, and repeated-session validation leakage. Whether graph neural networks (GNNs) outperform regularized classical models on standard 19-channel montages remains unestablished under leak-free protocols.

**Methods:** I analyzed seven public EEG datasets (566 participants in the primary recording state; 677 primary sessions) spanning acute sleep deprivation, depression, Parkinson's disease (PD), and dementia using a 19-channel pipeline: `specparam` decomposition, Phase Lag Index orthogonal minimum spanning trees (OMST), participant-grouped cross-validation, and fold-nested control standardization. Twelve model configurations, including four graph convolutional network (GCN) variants, were compared.

**Results:** Frontal periodic theta was lower than in controls in outpatient MDD (d = -0.826) and depression screening (d = -0.676) and higher in PD (d = +0.803 to +0.918) and dementia (d = +0.903 to +1.279). Because cohorts differ in hardware, eye state, and medication, opposite directions represent cohort-level observations. Across nine contrasts, `GCN-OMST` (mean AUC = 0.738; 5-seed mean 0.731) did not outperform `Random Forest` (0.752), `Tuned Ridge` (0.749), or random-graph GCNs (0.754). Session-mixed splits inflated AUC by up to 0.102. In 12.56% (85/677) of sessions, PLI matrices were indefinite (λ_min ≤ -7.370), requiring eigenvalue-adaptive Riemannian regularization (Adaptive SPD mean AUC = 0.743). Cross-site PD transfer AUCs ranged from 0.590 to 0.811; MoCA prediction within PD was weak (R² = 0.035).

**Conclusions:** In these cohorts, frontal periodic theta direction differed between depression and neurodegenerative groups, and a compact GCN showed no detectable advantage over regularized classical baselines under participant-level validation.

**Keywords:** Quantitative EEG, Spectral Parameterization, Graph Neural Networks, Frontal Theta, Parkinson's Disease, Depression, Benchmarking, Participant-Level Cross-Validation.

---

## 1. Introduction

Quantitative electroencephalography (qEEG) measures macro-scale post-synaptic cortical synchronization with millisecond temporal resolution {{ref:Nunez2006,LopesDaSilva2013}}, linking rhythmic neural activity to sleep regulation, cognition, and neuropsychiatric syndromes {{ref:Buzsaki2006,Basar2006,Fries2005,BuzsakiDraguhn2004,Cohen2014}}. Despite growing interest in automated network and machine learning biomarkers {{ref:Stam2014}}, single-cohort EEG classifiers frequently fail to generalize across independent recording sites {{ref:Kapur2012,Billeci2020}}. Practical comparability is hindered by heterogeneous electrode montages (ranging from 19-channel clinical 10–20 layouts to high-density research caps), multiplicative amplifier gain differences across hardware platforms, and optimistic validation bias introduced when repeated sessions cross fold boundaries or when normative standardization precedes data splitting {{ref:Scheinost2019,Varoquaux2018,Wolfers2018,Rutherford2022}}.

At the physiological level, conventional narrowband filtering conflates resonant periodic oscillations with broadband 1/f^χ aperiodic activity {{ref:Donoghue2020,Voytek2015,ColeVoytek2017}}, an electrophysiological slope associated in computational models with microcircuit excitation/inhibition (E/I) balance and population firing statistics {{ref:Gao2017,Donoghue2022}}. This periodic–aperiodic conflation complicates the interpretation of resting frontal theta activity (4–8 Hz) {{ref:CavanaghFrank2014,Mitchell2008}}. Frontal theta increases during prolonged wakefulness as an index of homeostatic sleep pressure {{ref:DeGennaro2001}}, yet resting-state studies in major depressive disorder (MDD) have reported both elevated frontal theta {{ref:Olbrich2013,Knott2001}} and reduced frontal theta {{ref:Newson2019,Roh2017}}. Meanwhile, low-frequency slowing is well established in Parkinson's disease (PD) and Alzheimer's disease (AD) {{ref:Stoffers2007,Bosboom2006,Stam2010}}, leaving open how parameterized periodic theta compares across affective, sleep-manipulated, and neurodegenerative datasets processed through an identical pipeline.

A parallel methodological question concerns graph representation learning on low-density scalp montages {{ref:BullmoreSporns2009,RubinovSporns2010}}. Functional connectivity measures that suppress zero-lag volume conduction, such as the Phase Lag Index (PLI) {{ref:Stam2007,Vinck2011}}, paired with parameter-free Orthogonal Minimum Spanning Trees (OMST) {{ref:Dimitriadis2015,Dimitriadis2017}}, avoid arbitrary edge-density thresholds and isolated nodes. Building on such graphs, Graph Convolutional Networks (GCNs) {{ref:KipfWelling2017}} are widely assumed to confer relational inductive bias for EEG decoding. However, sensor-level phase synchrony metrics such as PLI violate metric transitivity and can exhibit non-positive definiteness, which complicates geometric projection onto Riemannian manifolds. Furthermore, on coarse 19-channel arrays with limited spatial degrees of freedom, whether topological message passing provides demonstrable advantages over isolated-node graphs, regularized linear models, and tree ensembles under leak-free cross-validation remains unestablished.

To address these gaps, I assembled an open 19-channel EEG benchmarking pipeline and evaluated it across seven independent public datasets (N = 566 participants in the primary recording state, 677 primary sessions) spanning experimental acute sleep deprivation {{ref:Xiang2024}}, outpatient MDD {{ref:Mumtaz2017}}, university depression screening {{ref:Cavanagh2019}}, three clinical PD cohorts recorded under dopaminergic manipulation or multi-scale clinical assessment {{ref:George2013,Cavanagh2018,Singh2023}}, and clinical dementia {{ref:Miltiadous2023}}. Rather than proposing another architectural variant on an isolated cohort, the goal of this benchmark is to estimate how much reported separation depends on model class versus feature content, to audit foundational assumptions (validation leakage, PLI positive semi-definiteness, and amplifier-gain handling), and to describe the direction of frontal periodic theta effects across standardized 19-channel clinical montages. Using participant-grouped cross-validation and fold-nested healthy-control standardization, I address three research questions:
1. **Question 1 (Electrophysiological Directionality):** Once broadband 1/f^χ background activity is separated from narrowband peaks, how does resting frontal periodic theta differ across depression, acute sleep deprivation, Parkinson's disease, and dementia comparisons?
2. **Question 2 (Graph Inductive Bias and Topological Utility):** Does a compact `NormativeGCN` operating on PLI-OMST topology outperform graph adjacency ablations, Riemannian tangent-space projections, and classical tabular classifiers on 19-channel resting EEG, and are sensor-level PLI matrices mathematically well-conditioned?
3. **Question 3 (Exploratory Cross-Center Transfer and Score Prediction):** To what extent do 19-channel features transfer across three independent PD centers and across diagnostic domains under uncalibrated, global-z, and target-control standardization regimes, and how accurately do they predict cross-sectional cognitive and motor scale scores?

---

## 2. Methods

### 2.1 Public EEG Datasets, Study Designs, and Participant Accounting
I conducted a secondary computational analysis of seven de-identified, publicly available resting-state EEG datasets retrieved from OpenNeuro and Figshare (repository accessions and software specifications in Supplementary Table S9). Because the archive combines an experimental sleep-deprivation crossover study, a university symptom-screening sample, and five clinical neurological or psychiatric case-control cohorts (two incorporating within-patient dopaminergic ON/OFF sessions), each dataset was analyzed according to its native experimental or clinical design rather than treated as a uniform hospital registry. Across all seven public releases, N = 574 participants (685 primary sessions; 745 including Study 2 eyes-open recordings) were enrolled. Applying pre-specified automated quality and diagnostic screening rules excluded 8 participants (1.4%: 6 in Study 2 with >20% artifact-rejected epochs or channel bridging; 2 in Study 3 with borderline screening scores 10 ≤ BDI-II ≤ 13; 0 exclusions in the remaining five datasets), yielding N = 566 participants analyzed in each cohort's primary recording state and 677 primary sessions (737 total sessions including Study 2 eyes-open; the 566 count refers to the primary recording state, so the 6 Study 2 participants excluded from the primary eyes-closed analysis are not counted even though 5 of them have a retained eyes-open recording; participant flow in Supplementary Figure S5; per-cohort demographics, hardware systems, and native sampling rates in Supplementary Table S3):

- **Study 1 — Experimental Acute Sleep Deprivation (`OpenNeuro ds004902`, N = 71 participants, 142 paired sessions)** {{ref:Xiang2024}}: Healthy young adults (age 20.0 ± 1.4 years, range 17–23; 47.9% female) completed counterbalanced eyes-open resting EEG under normal sleep (NS) and 36 h monitored total sleep deprivation (SD).
- **Study 2 — Outpatient Major Depressive Disorder (`Figshare 4244171`, N = 58 primary eyes-closed; N = 60 secondary eyes-open)** {{ref:Mumtaz2017}}: DSM-IV outpatient MDD patients and healthy controls recorded on a native 19-channel 10–20 system. In the primary eyes-closed state, 6 participants were excluded due to excessive movement or bridging artifacts (>20% rejected epochs), leaving N = 58 analyzed participants (30 MDD, 28 HC). In the secondary eyes-open state, 4 participants were excluded, leaving N = 60 analyzed participants (32 MDD, 28 HC; 55 participants contributed both clean EC and EO sessions, yielding 63 unique Study 2 participants). Note that participant-level chronological age was not released in the public archive for this cohort, precluding chronological age ANCOVA for Study 2.
- **Study 3 — University Depression-Screening Cohort (`OpenNeuro ds003478`, N = 119 participants)** {{ref:Cavanagh2019}}: Undergraduate students grouped into a depressive-symptom screening group (BDI-II ≥ 14, N = 44) and low-symptom controls (BDI-II ≤ 7, N = 75) under eyes-open resting EEG. Based on accompanying Structured Clinical Interview for DSM (SCID) records, the 44 BDI-II ≥ 14 participants comprised 11 with current MDD, 11 with past MDD, 8 who did not meet criteria for current or past MDD, and 14 who were not interviewed; this cohort is therefore designated throughout as a university depression-screening sample rather than clinical MDD.
- **Study 4 — UCSD Parkinson's Disease ON/OFF Cohort (`OpenNeuro ds002778`, N = 31 participants, 46 sessions)** {{ref:George2013}}: Idiopathic PD patients (N = 15) recorded during eyes-open resting EEG (with a visual fixation cross) in counterbalanced ON-medication and overnight OFF-withdrawal (≥ 12 h) sessions alongside healthy controls (N = 16).
- **Study 5 — AHEPA Clinical Dementia Cohort (`OpenNeuro ds004504`, N = 88 participants)** {{ref:Miltiadous2023}}: Eyes-closed 19-channel clinical recordings from patients with Alzheimer's disease (AD, N = 36), frontotemporal dementia (FTD, N = 23), and cognitively unimpaired elderly controls (N = 29), with Mini-Mental State Examination (MMSE) scores.
- **Study 6 — UNM Parkinson's Disease ON/OFF Washout Cohort (`OpenNeuro ds003490`, N = 50 participants, 75 sessions)** {{ref:Cavanagh2018}}: Idiopathic PD patients (N = 25) evaluated during alternating 1 min eyes-open and 1 min eyes-closed resting EEG in counterbalanced ON-medication and standardized ≥ 15 h overnight OFF-washout sessions alongside matched controls (N = 25).
- **Study 7 — Iowa Parkinson's Multi-Scale Cohort (`OpenNeuro ds004584`, N = 149 participants)** {{ref:Singh2023}}: Eyes-open resting recordings from idiopathic PD patients (N = 100, all evaluated ON dopaminergic medication) and elderly controls (N = 49) with Montreal Cognitive Assessment (MoCA) and UPDRS Part-III motor ratings.

All primary studies obtained institutional review board approval and written informed consent. Secondary computational analysis of these de-identified public archives required no additional human-subjects intervention and was exempt under 45 CFR 46.104(d)(4) and institutional guidelines.

### 2.2 Signal Preprocessing and Universal 19-Channel Montage Harmonization
All recordings were mapped onto the canonical 19-channel international 10–20 montage:
$$\\mathcal{{C}}_{{19}} = \\{{\\text{{Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, P7, P3, Pz, P4, P8, O1, O2}}\\}} \\qquad (1)$$
Legacy labels (`T3, T4, T5, T6`) were mapped to modern nomenclature (`T7, T8, P7, P8`; channel mapping in Supplementary Table S5). Where `Pz` served as the native physical recording reference (Study 7, `ds004584`), an initial zero-valued `Pz` channel was appended prior to Common Average Referencing (CAR), algebraically recovering its voltage relative to the 19-channel spatial mean. Signals were zero-phase FIR bandpass filtered (1.0–40.0 Hz, Hamming window), resampled to 250 Hz, CAR re-referenced, and segmented into non-overlapping 4.0 s epochs (1,000 samples). Epochs exceeding 150 μV peak-to-peak amplitude were rejected; to avoid selective attrition in geriatric tremor recordings, an adaptive fallback retained the cleanest available epochs, yielding cohort-minimum retained clean epochs ranging from 5 to 12 epochs (20–48 s; Table S8b), while >95% of all recordings retained ≥ 15 clean epochs (60 s; sensitivity analyses excluding recordings with <8 clean epochs confirmed identical diagnostic patterns; Supplementary Tables S8 and S8b).

### 2.3 Spectral Parameterization and Multiplicative Gain Handling
Channel power spectral densities (PSDs) were estimated via Welch's method (2.0 s Hamming sub-windows, 50% overlap, Δ f = 0.50 Hz) and decomposed in log-log space across 1.0–40.0 Hz using `specparam` (v1.1.0; Eq. 2; `aperiodic_mode='fixed'`, `peak_width_limits=[1.0, 12.0] Hz`, `max_n_peaks=6`, `min_peak_height=0.05 B`, `peak_threshold=2.0 SD`) {{ref:Donoghue2020}} (single-subject example in Supplementary Figure S1):
$$\\log_{{10}} P(f) = L(f) + \\sum_{{k=1}}^K G_k(f), \\quad L(f) = b - \\chi \\log_{{10}}(f) \\qquad (2)$$
where b is the broadband offset, χ is the aperiodic exponent, and G_k(f) denotes Gaussian oscillatory peaks. Under any positive multiplicative amplifier scaling s'(t) = γ s(t) (γ > 0), the PSD scales as P'(f) = γ² P(f), shifting log₁₀ P'(f) = log₁₀ P(f) + 2 log₁₀ γ. Because 2 log₁₀ γ is frequency-independent, it is absorbed entirely into the offset (b' = b + 2 log₁₀ γ), leaving both the aperiodic exponent (χ' ≡ χ) and the flattened periodic residual spectrum (P̃'(f) = log₁₀ P'(f) - L'(f) ≡ P̃(f)) algebraically invariant (analytical derivation and non-multiplicative hardware boundary conditions in Supplementary Sections S1.3 and S1.3.1; numerical verification across γ ∈ [10^-2, 10²] in Supplementary Table S2 and Supplementary Figure S3; empirical comparisons against relative band power and unparameterized log₁₀ band power under uniform and heterogeneous multi-amplifier gain perturbations in Supplementary Table S20). This invariance is an exact property of the downstream feature definitions (`specparam` and PLI) and equally holds for relative band power, so it is a property of the pipeline and not a source of added discriminative information; on unperturbed recordings, unparameterized log₁₀ band power transferred as well as or better than `specparam` residuals (Supplementary Table S20). Initial preprocessing artifact rejection thresholds (e.g., ± 150 μV) are amplitude-dependent and were applied in native calibrated microvolt units.

### 2.4 Functional Connectomics via Phase Lag Index and OMST
Pairwise phase synchronization was estimated in the combined theta–alpha band (4–13 Hz) using the Phase Lag Index (PLI; Eq. 3) {{ref:Stam2007}}:
$$\\text{{PLI}}_{{jk}} = \\left| \\langle \\operatorname{{sign}}[\\sin(\\Delta \\phi_{{jk}}(t))] \\rangle \\right| \\qquad (3)$$
which discards zero-lag coupling and is invariant to positive amplitude scaling. Binary undirected graphs were extracted without arbitrary density thresholds via Orthogonal Minimum Spanning Trees (OMST) {{ref:Dimitriadis2017}}, iteratively uniting disjoint Kruskal spanning trees (18 edges per tree) to maximize Global Cost Efficiency (Eq. 4):
$$\\text{{GCE}}(m) = \\frac{{E_{{\\text{{global}}}}(G_m)}}{{E_{{\\text{{global}}}}(G_{{\\text{{full}}}})}} - \\text{{Cost}}(G_m) \qquad (4)$$
Selecting m^* = argmax_m GCE(m) guarantees a connected 19-node topology with zero isolated vertices for every session (Supplementary Section S1.2).

### 2.5 Participant-Grouped Cross-Validation and Fold-Nested Normative Calibration
To standardize features relative to healthy reference variation while keeping test-fold information out of the reference statistics {{ref:Scheinost2019,Wolfers2018}}, channel-wise z-scores were computed strictly from healthy control (or rested reference) sessions inside each training fold (Eq. 5):
$$Z_{{i,c,k}} = \\frac{{X_{{i,c,k}} - \\hat{{\\mu}}_{{\\text{{HC}},c,k}}^{{\\text{{train}}}}}}{{\\hat{{\\sigma}}_{{\\text{{HC}},c,k}}^{{\\text{{train}}}} + \\epsilon}} \\qquad (5)$$
Validation and test folds were scaled solely using (μ_HC^train, σ_HC^train) (a Leave-One-Control-Out [LOCO] jackknife sensitivity check—where each training-fold control is standardized using the remaining N_HC,train - 1 training controls to eliminate finite-sample self-inclusion shrinkage—confirmed consistent out-of-fold ROC-AUC across repeated-session cohorts: `Tuned Logistic Regression` 0.608 vs. 0.612 in Study 1, 0.613 vs. 0.656 in Study 4, and 0.715 vs. 0.723 in Study 6c; Supplementary Table S19). All within-cohort classification benchmarks used 5-fold cross-validation; in datasets with repeated sessions per participant (Studies 1, 4, and 6c), `StratifiedGroupKFold(n_splits=5)` grouped by participant ID in the outer loop and `StratifiedGroupKFold(n_splits=3)` grouped by participant ID in the inner hyperparameter loop ensured that all sessions from a given individual resided in the same fold (0.0% participant overlap across both outer and inner splits). Leakage ablations contrasted this strict participant-grouped protocol against global pre-standardization and session-mixing splits (`StratifiedKFold`, 52.2%–70.4% participant overlap), reporting both session-level and participant-cluster bootstrap 95% confidence intervals (Supplementary Table S19); session mixing changed AUC by -0.069 to +0.102, whereas global pre-standardization changed it by only -0.050 to +0.002, so session mixing, not pre-standardization, was the main source of optimistic bias. For cross-site and cross-disorder transfer, *target-reference (transductive healthy-control) normative calibration* standardized target-cohort sessions using the healthy control recordings of the target site; it requires no target patient data and no model retraining, but it does require a labeled healthy reference sample at the target site, and those controls are also part of the evaluation set except in the disjoint Split-Target-HC holdout check. It was benchmarked against uncalibrated raw transfer, global cohort z-scoring, and a disjoint Split-Target-HC holdout calibration check across all six directed PD transfer pairs (Supplementary Table S20).

### 2.6 Benchmarking Models, Hyperparameters, and Statistical Testing

#### 2.6.1 Model Architecture Tiers (Tiers 1–4)
Twelve primary classification configurations spanning three tiers were evaluated identically across cohorts (architectural details in Supplementary Section S1.4 and Table S7; classical estimator settings and inner participant-grouped tuning verification in Supplementary Table S7b; TRIPOD+AI checklist in Supplementary Table S10):
1. **Graph Neural Networks (4 variants):** `NormativeGCN` comprises two spectral graph convolution layers (`4 -> 32 -> 32`, ReLU, dropout p = 0.25) with symmetric normalized adjacency D̃^-1/2(A + I_19)D̃^-1/2, concatenated global mean and max pooling (64-D), and a two-layer classification head (`Linear(64, 32) -> ReLU -> Dropout(0.25) -> Linear(32, 2)`), totaling 3,362 trainable parameters. All GCN variants (`GCN-OMST`, isolated-node `GCN-Identity` [A = I], random 18-edge `GCN-Random`, and dense `GCN-Full` [A = 1]) were trained for a fixed 50 epochs using PyTorch `Adam` (η = 0.005, weight decay 10^-4) with zero test-fold early stopping.
2. **Classical Tabular Baselines (6 models on flattened 76-D node features):** L2 `Tuned Logistic Regression` (C ∈ {0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0}), `Tuned Ridge Classifier` (α ∈ {0.01, 0.1, 1.0, 10.0, 50.0, 100.0}), `Linear SVM` (C ∈ {0.01, 0.1, 1.0, 10.0}), and `RBF SVM` (C ∈ {0.1, 1.0, 10.0}, γ ∈ {'scale', 'auto'})—each tuned via nested 3-fold cross-validation inside the outer training split using `StratifiedKFold(n_splits=3)` in single-session cohorts and participant-grouped `StratifiedGroupKFold(n_splits=3)` in repeated-session cohorts (enforcing participant grouping in the inner loop improved `Tuned Logistic Regression` out-of-fold ROC-AUC relative to ungrouped inner `StratifiedKFold(3)` from 0.609 → 0.612 in Study 1, 0.627 → 0.656 in Study 4, and 0.704 → 0.723 in Study 6c; Supplementary Table S7b)—plus fixed `Random Forest` (`n_estimators=100`, `max_depth=5`) and `GBDT` (`HistGradientBoostingClassifier`, `max_iter=100`, `max_depth=4`).
3. **Univariate Physiological Baselines (2 models):** L2 logistic regression on 5-channel canonical frontal periodic theta (`Fp1, Fp2, F3, F4, Fz`) alone (`Single θ`) and on 19-channel global mean aperiodic exponent χ alone (`Single χ`). Leave-one-feature-out 3-feature `NormativeGCN` ablations and additional single-marker baselines were also evaluated (Supplementary Table S6).
4. **Multi-Seed Stability and Functional Connectivity Geometry Baselines:** To quantify partition and initialization sensitivity, 5-fold cross-validation was repeated across five independent random seeds (`42, 123, 456, 789, 2026`) for seven configurations (`GCN-OMST`, `GCN-Random`, `GCN-Identity`, `Tuned Logistic`, `Tuned Ridge`, `Random Forest`, `Single θ`) on ten contrasts, reporting both the reference `seed = 42` split (Table 2) and the 5-seed distribution (`Mean ± SD`, `[Min, Max]`, and paired ΔAUC vs. `GCN-OMST`; Supplementary Table S21). Additionally, because the Phase Lag Index is a non-linear sign statistic rather than a Gram covariance matrix, I audited the exact eigenvalue spectrum of all 677 symmetrized 19 × 19 PLI matrices (C = I_19 + PLI) and benchmarked direct 171-D Euclidean upper-triangular PLI vectors against two strictly symmetric positive-definite (SPD) Log-Euclidean tangent-space projections (190-D eigenvalue-adaptive diagonal loading λ_min ≥ 0.01 and 190-D Higham spectral-floor projection λ_i ≥ 10^-4) across `Tuned Logistic Regression`, `Tuned Ridge`, and `Random Forest` (Supplementary Table S22).

#### 2.6.2 Statistical Testing and Model Rankings
Group biophysical contrasts were quantified via paired Cohen's d_z (paired t-test) in Study 1 and Cohen's d with Welch's unequal-variance t-tests in Studies 2–7, controlling the false discovery rate via Benjamini–Hochberg adjustment across the 10 primary frontal theta contrasts (q_10; Table 1) and conservatively across all 50 marker-by-cohort tests (q_50; Supplementary Table S4). Out-of-fold ROC-AUCs are reported with 1,000-bootstrap 95% confidence intervals (and participant-cluster bootstrap intervals in repeated-session cohorts; Supplementary Table S19). Head-to-head model rankings across contrasts were evaluated across nine primary case-control benchmark contrasts (Studies 1, 2, 3, 4, 5a, 5b, 5c, 6a, 7; incorporating shared healthy controls for the nested dementia subgroups in Study 5: AD vs. HC, FTD vs. HC, and combined dementia vs. HC) using exact two-sided paired Wilcoxon signed-rank tests; to prevent control group re-use and preserve exchangeability under the signed-rank test, repeated medication contrasts sharing controls (Study 6b UNM PD-ON and Study 6c UNM All share the same N = 25 healthy controls with Study 6a) were excluded from the primary omnibus ranking and evaluated as a complete 11-contrast sensitivity check (Supplementary Table S11b).

#### 2.6.3 Cross-Dataset Transfer and Calibration Protocols
For cross-center and cross-disorder generalization (RQ3), models trained on one dataset were evaluated directly on another without fine-tuning or target patient labels, testing generalization under unsupervised *target-reference (transductive HC) normative calibration* (standardizing target recordings using the mean and standard deviation of target-site healthy controls; Eq. 5). To determine whether `specparam` periodic residuals confer empirical robustness over standard spectral features, cross-site transfer across the three Parkinson's datasets was re-evaluated under experimental amplifier gain scaling (PSD_k(f) = k² PSD(f)) and compared against relative band power and unparameterized log₁₀ power (Supplementary Table S20).

### 2.7 Exploratory Cross-Sectional Clinical Scale Regression
As an exploratory evaluation of inter-individual symptom score prediction from resting-state EEG, four regressors (`Ridge` with α = 10, RBF `SVR`, `Random Forest Regressor` [`max_depth=5`], and `GBDT Regressor` [`max_depth=4`]) were evaluated via 5-fold cross-validation on training-fold-standardized 76-D features for MMSE in Study 5 (N = 88), MoCA in Study 7 (N = 149), and UPDRS-III motor scores in Study 7 (N = 100 PD patients). To test whether cognitive associations reflected diagnostic group separation (e.g., ceiling MMSE = 30.0 in healthy controls) versus within-patient score variation, sensitivity regressions were repeated exclusively within dementia patients (N = 59) and within PD patients (N = 100).

---

## 3. Results

### 3.1 Harmonized 19-Channel Pipeline, Numerical Gain Verification, and OMST Topology
Across all seven datasets and 677 primary sessions, the harmonization adapter recovered all 19 canonical 10–20 channels, including algebraic reconstruction of `Pz` under Common Average Referencing in Study 7. Numerical gain scaling across four orders of magnitude (γ ∈ [10^-2, 10²]) yielded R² = 1.000000 and maximum absolute error < 10^-14 for periodic band residuals and aperiodic χ, while shifting only the unadjusted offset b by 2 log₁₀ γ (Supplementary Table S2 and Supplementary Figure S3). Across all 677 primary PLI-OMST connectomes (12,863 node instances), GCE optimization selected M = 1.52 ± 0.50 orthogonal trees (48.15% M = 1, 51.85% M = 2), corresponding to E = 27.33 ± 9.00 edges, mean node degree mean degree k̄ = 2.88 ± 0.95 (degree sum 54.66), edge density 15.98% ± 5.26%, mean GCE = 0.4939 ± 0.0438, and 0.0000% isolated nodes (Supplementary Section S1.2, Supplementary Table S1, and Supplementary Figure S2).

---

### 3.2 Multi-Cohort Electrophysiological Comparisons and Sensitivity Analyses
**Addressing Research Question 1 (Cross-Cohort Electrophysiological Invariance):** Effect sizes across the ten primary group contrasts are presented in Figure 1 and Table 1, with the pooled 50-test statistical table in Supplementary Table S4 and representative cohort spectra in Supplementary Figure S4.

![](../reports/Figure_1_Multicohort_Biophysical_Effect_Sizes.png){{width=6.5in}}
**Figure 1. Standardized Electrophysiological Effect Sizes Across Seven Public EEG Datasets.** Forest plots show independent-groups Cohen's d (or paired d_z in Study 1) with 95% confidence intervals across ten contrasts. (A) Frontal Periodic Theta (4–8 Hz) is lower in both depression cohorts and higher across Parkinson's disease and Alzheimer's/combined dementia contrasts. (B) Global Periodic Delta (1–4 Hz). (C) Posterior Periodic Alpha (8–13 Hz) is selectively lower in Alzheimer's disease and frontotemporal dementia. (D) Global Aperiodic Exponent (χ) is higher (steeper 1/f slope) in Alzheimer's and combined dementia and lower (flatter) in outpatient MDD. Square markers indicate uncorrected p < 0.05 (\\* p < 0.05, \\*\\* p < 0.01, \\*\\*\\* p < 0.001).

{t1}

Four primary electrophysiological findings emerged:
1. **Opposing Frontal Periodic Theta Directions:** Resting frontal periodic theta was significantly lower than controls in outpatient MDD (Study 2: d = -0.826, p = 0.0027, q_10 = 0.0053, q_50 = 0.0121; eyes-open vs. eyes-closed sensitivity in Supplementary Table S13) and in the university depression-screening group (Study 3: d = -0.676, p = 4.96 × 10^-4, q_10 = 0.0012, q_50 = 0.0035). In Study 2, repeating the group contrast under eyes-open recording (Study 2b, N = 60: 32 MDD, 28 HC; Supplementary Table S13 Panel A) yielded an essentially identical effect size (d = -0.872 [-1.405, -0.340], Welch t = -3.338, df = 54.3, p = 0.0015; within-subject paired theta correlation across states r = +0.742, p < 10^-9), indicating that the Study 2 effect does not depend on eye state. Conversely, frontal periodic theta was significantly higher than controls across clinical PD and dementia cohorts: Study 4 UCSD PD participant-averaged comparison (N = 15 unique PD patients [mean of ON and OFF] vs. N = 16 HCs: d = +0.918, p = 0.0159, q_10 = 0.0199, q_50 = 0.0468; single-session Study 4a PD-OFF d = +0.559, p = 0.1303, ns; Study 4b PD-ON d = +1.150, p = 0.0035; pooled sessions 30/16: d = +0.837, p = 0.0090), Study 5a AD (d = +1.279, p = 1.41 × 10^-6, q_10 = 7.05 × 10^-6), Study 5c combined dementia (d = +0.903, p = 1.30 × 10^-5, q_10 = 4.33 × 10^-5; Study 5b FTD alone d = +0.546, p = 0.0720, ns), Study 6a UNM PD-OFF (d = +0.865, p = 0.0037, q_10 = 0.0061), Study 6b UNM PD-ON (d = +0.803, p = 0.0066, q_10 = 0.0095), and Study 7 Iowa PD (d = +0.899, p = 7.16 × 10^-8, q_10 = 7.16 × 10^-7). To synthesize these effects without shared controls or repeated sessions, a random-effects meta-analysis restricted strictly to K = 6 mutually exclusive between-subject cohorts (N = 495 participants; Supplementary Table S23) gave opposite-signed group means: pooled d_RE = -0.726 (DerSimonian–Laird 95% CI [-1.037, -0.415], p = 4.77 × 10^-6; note that at k = 2 studies, the ad-hoc variance-floored Hartung–Knapp–Sidik–Jonkman t_df=1 interval [-2.744, +1.291] is wide by construction, with unfloored HKSJ [-1.622, +0.170]) in the depressive hypo-oscillatory domain versus d_RE = +0.856 (DL 95% CI [+0.617, +1.096], p = 2.48 × 10^-12; ad-hoc variance-floored HKSJ 95% CI [+0.467, +1.245], p_HKSJ = 0.0060, unfloored HKSJ [+0.662, +1.051]) in the neurodegenerative slowing domain (k = 4: UCSD PD-OFF, Combined Dementia, UNM PD-OFF, and Iowa PD). The between-group Q statistic was large (Q_between = 62.38, df = 1, p = 2.89 × 10^-15), as is expected when two groups of effect sizes have opposite signs, and the sign pattern held across all five sensitivity specifications—substituting participant-averaged UCSD (15 vs. 16, d = +0.918), PD-ON states, AD-only, or FTD-only, or incorporating Study 1 Acute Sleep Deprivation on the Morris–DeShon independent-groups equivalent scale (d_IG = +0.267 ± 0.112, K = 7)—as well as all Leave-One-Cohort-Out deletions (Q_between = 45.67–66.82, all p ≤ 1.40 × 10^-11; Supplementary Table S23). Because the two groups differ simultaneously in diagnosis, age, recording hardware, eye state, medication, and site, because the depression group contains only k = 2 cohorts (variance-floored HKSJ interval including zero), and because only about half of the Study 3 screening-positive participants had SCID-confirmed current or past MDD, these results describe opposite directions in these cohorts and not a disease-intrinsic polarity.
2. **Posterior Periodic Alpha Attenuation and Aperiodic Steepening in Dementia:** Posterior periodic alpha was substantially lower in AD (d = -1.059, p = 4.24 × 10^-5, q_50 = 3.54 × 10^-4), FTD (d = -0.949, p = 0.0018, q_50 = 0.0091), and combined dementia (d = -0.981, p = 9.31 × 10^-6, q_50 = 1.55 × 10^-4), whereas PD and depression cohorts did not show comparable posterior alpha loss. Concurrently, global χ was higher (steeper spectral slope) in AD (d = +1.096, p = 2.18 × 10^-5, q_50 = 2.18 × 10^-4) and lower (flatter slope) in outpatient MDD (d = -0.744, p = 0.0062, q_50 = 0.0237).
3. **Small but Detectable Within-Subject Theta Shift Under Acute Sleep Deprivation:** In Study 1 (N = 71 pairs), 36 h total wakefulness produced a small but detectable within-subject increase in frontal periodic theta (d_z = +0.284 [+0.047, +0.521], paired p = 0.0194, surviving 10-test primary contrast FDR correction at q_10 = 0.0216, but not the more conservative 50-test benchmark-wide correction at q_50 = 0.0540; independent-equivalent d_IG = +0.267; global theta d_z = +0.568, p = 9.05 × 10^-6), qualifying this as a subtle homeostatic effect. Meanwhile, global χ (p = 0.1269) and posterior alpha (p = 0.4198) did not differ significantly.
4. **Absence of Group Differences in Scalar OMST Global Cost Efficiency:** Across all ten contrasts, scalar OMST GCE did not differ significantly between case and reference groups (all raw p ≥ 0.0728, all q_50 ≥ 0.1583), indicating that diagnostic separation in these 19-channel resting recordings is concentrated in regional spectral parameters rather than global binary wiring efficiency.

**Sensitivity Analyses:** Four supplementary checks assessed potential artifacts and confounders. First, *frontopolar channel exclusion* (`F3, Fz, F4` only, omitting `Fp1` and `Fp2`) preserved frontal periodic theta effect sizes across Study 2 MDD (d = -0.831, p = 0.0024), Study 3 university depression (d = -0.653, p = 8.25 × 10^-4), Study 5c dementia (d = +0.935, p = 6.81 × 10^-6), Study 6a UNM PD-OFF (d = +0.922, p = 0.0021), and Study 7 Iowa PD (d = +0.944, p = 1.75 × 10^-8), while Study 1 sleep deprivation attenuated slightly (d_z = +0.237, p = 0.0501). Second, *within-subject dopaminergic ON vs. OFF comparisons* differed by clinical protocol (Figure 2B and Supplementary Table S14): in Study 6 UNM (N = 25 pairs following ≥ 15 h washout), no statistically significant ON-OFF differences were detected across any marker (frontal periodic theta d_z = -0.092, paired p = 0.6490; all markers p ≥ 0.1579), whereas in Study 4 UCSD (N = 15 pairs), ON sessions showed higher frontal periodic theta (d_z = +0.660, p = 0.0228) and lower global delta (d_z = -0.753, p = 0.0113). Third, *chronological age ANCOVA* (`Marker ~ Group + Age` across 7 between-subject contrasts, 28 models; Supplementary Table S18) confirmed that every unadjusted significant group difference remained significant after age adjustment (frontal periodic theta p ≤ 0.0073 across Study 3, Study 4b UCSD ON, Study 5, Study 6a/6b, and Study 7; Study 5 posterior alpha p = 5.49 × 10^-5 and χ p = 0.0075), while unadjusted non-significant contrasts (such as Study 4a UCSD OFF frontal theta, p = 0.1367) remained non-significant and no age covariate reached p < 0.05 (minimum age p = 0.0609). Fourth, *posterior Individual Alpha Peak Frequency (`IAPF`, 6–13 Hz periodic center of gravity) analysis* across the three PD cohorts showed significant posterior alpha peak slowing in Study 7 Iowa PD (8.84 ± 1.32 Hz vs. 9.76 ± 1.65 Hz, Δ = -0.92 Hz, d = -0.638, p = 0.0011) and smaller non-significant downward shifts in Study 6a UNM PD-OFF (9.41 ± 0.87 Hz vs. 9.78 ± 0.90 Hz, Δ = -0.37 Hz, d = -0.421, p = 0.1435) and participant-averaged Study 4 UCSD PD (9.65 ± 0.93 Hz vs. 9.92 ± 1.42 Hz, Δ = -0.27 Hz, d = -0.241, p = 0.4980); in `Frontal_Theta ~ PD + IAPF` ANCOVA models, lower posterior `IAPF` strongly predicted higher 4–8 Hz periodic power (`IAPF` t = -5.807, p < 0.0001 in UNM; t = -4.987, p < 0.0001 in Iowa), yet the PD group elevation in frontal periodic theta remained statistically significant after controlling for `IAPF` in all three cohorts (participant-averaged UCSD t = +2.427, p = 0.0219; UNM PD-OFF t = +2.659, p = 0.0107; Iowa PD t = +3.879, p = 1.58 × 10^-4).

---

### 3.3 Within-Cohort Classification Benchmarks and Graph Ablations
Table 2 summarizes 5-fold cross-validated ROC-AUC across the 9 primary diagnostic contrasts (9 × 12 = 108 evaluations); the complete 11 × 12 = 132 within-cohort benchmark (adding Study 6b UNM PD-ON and Study 6c UNM All) is reported in Supplementary Table S11, descriptive across-contrast Wilcoxon comparisons in Supplementary Table S11b, single-marker and leave-one-feature-out ablations in Supplementary Table S6, 5-seed stability in Supplementary Table S21, functional connectivity geometry baselines in Supplementary Table S22, and label-permutation null distributions in Supplementary Figure S6.

{t2}

Across the 9 primary contrasts, `GCN-OMST` achieved a mean ROC-AUC of 0.738 (range: 0.529 in sleep deprivation and 0.647 in university depression screening to 0.842 in combined dementia and 0.902 in outpatient MDD). No statistically significant performance advantage of `GCN-OMST` was detected over classical regularized classifiers or graph topology ablations (Supplementary Table S11b). Across the 9 primary contrasts, `GCN-OMST` recorded 4 wins and 5 losses against `Random Forest` (mean AUC = 0.752, Wilcoxon W = 17.0, p = 0.5703), 4 wins and 5 losses against `Tuned Ridge` (0.749, W = 17.0, p = 0.5703), and 5 wins and 4 losses against `Tuned Logistic Regression` (0.743, W = 22.0, p = 1.0000). Similarly, `GCN-OMST` did not outperform isolated-node `GCN-Identity` (0.742, p = 0.6523), random-topology `GCN-Random` (0.754, p = 0.2500), or dense `GCN-Full` (0.739, p = 1.0000; Figure 3B). Only `RBF SVM` (0.661, raw p = 0.0117, q = 0.0645) and univariate `Single χ` (0.567, raw p = 0.0078, q = 0.0645) yielded lower uncorrected AUCs, neither surviving FDR correction. Tests used unrounded AUCs; recomputing from the three-decimal values printed in Table 2 changes some ranks and ties (for example, `Tuned Ridge` W = 18, p = 0.652, and a Study 5a tie between `GCN-OMST` and `Single θ` at 0.807, which are 0.8075 and 0.8065 unrounded), without altering any conclusion. With nine non-independent contrasts (Studies 5a–5c share controls) these exact tests have low power, so non-significance indicates that no difference was detected, not that the models are equivalent.

In five of the nine primary cohorts—Study 1 Sleep Deprivation (`Single θ` 0.554 vs. `GCN-OMST` 0.529), Study 3 university depression screening (0.683 vs. 0.647), Study 4 UCSD PD (0.738 vs. 0.725), Study 6a UNM PD-OFF (0.768 vs. 0.699), and Study 7 Iowa PD (0.739 vs. 0.685)—univariate 5-channel frontal periodic theta alone matched or exceeded the 3,362-parameter `GCN-OMST` model (4 wins, 5 losses, p = 0.8203). Repeating 5-fold cross-validation across five independent random seeds (`42, 123, 456, 789, 2026`; Supplementary Table S21) confirmed that 5-seed grand-mean ROC-AUC for `GCN-OMST` (0.731) trailed `Random Forest` (0.752), `Tuned Ridge` (0.741), `Tuned Logistic Regression` (0.738), `GCN-Random` (0.736), and `GCN-Identity` (0.736), and revealed substantial partition sensitivity in the smallest repeated-session cohort (Study 4 UCSD PD, N = 31 participants, where `GCN-OMST` spanned [0.463, 0.725], 0.624 ± 0.105, whereas `Single θ` remained stable at 0.719 ± 0.018).

#### 3.3.1 Empirical Audit of PLI Definiteness and Riemannian Tangent Space Benchmarking
In geometric machine learning and functional connectomics, connectivity matrices are frequently mapped onto the Riemannian manifold of Symmetric Positive Definite (SPD) matrices S_++^n using Log-Euclidean tangent space projections. However, sensor-level Phase Lag Index (PLI) is computed from pairwise phase-lead/lag asymmetry sign(Δ φ_ij), which evaluates directional phase locking rather than inner-product covariance and does not satisfy the triangle inequality or metric transitivity. An empirical audit across all 677 primary resting sessions (19 × 19 augmented matrices C = I_19 + PLI; Supplementary Table S22) revealed that 85 / 677 (12.56%) sessions possessed strictly negative minimum eigenvalues, with λ_min dropping as low as -7.370. This empirical indefiniteness prevents direct immersion into the Riemannian SPD cone without mathematical regularisation. To benchmark Riemannian manifold representations against classical models, two SPD regularization projections were evaluated: eigenvalue-adaptive diagonal loading (C_reg = C + (|λ_min| + 0.01) I) and Higham nearest-SPD projection, mapped into 190-dimensional Log-Euclidean tangent space, alongside 171-dimensional upper-triangle Euclidean PLI vectors (Supplementary Table S22). Across the 9 primary diagnostic contrasts, Riemannian tangent-space baselines achieved moderate discriminability (Eigenvalue-Adaptive SPD mean AUC = 0.743, Higham SPD mean AUC = 0.742, Euclidean PLI mean AUC = 0.743), trailing classical node-feature models (`Random Forest` 0.752, `Tuned Ridge` 0.749). Across nine of the ten evaluated contrasts (with only secondary contrast Study 6b UNM PD-ON as an exception, where `Euclidean PLI + Ridge` reached 0.773 vs. 0.746 for `Single θ`), simple univariate parameterized node features matched or exceeded raw connectome projections, indicating that, in these data, sensor-level PLI connectivity did not provide discriminability beyond node features on standard 19-channel montages.

Leave-one-feature-out `NormativeGCN` ablations (Supplementary Table S6) confirmed that classification relies primarily on regional node features: omitting periodic theta (`Ablate θ`) substantially reduced AUC across depression, PD, and FTD/combined dementia (e.g., 0.902 → 0.727 in Study 2 MDD, 0.725 → 0.508 in Study 4 UCSD PD, 0.842 → 0.725 in Study 5c dementia, and 0.685 → 0.550 in Study 7 Iowa PD), whereas in Study 5a AD (0.807 → 0.798) posterior alpha and χ provided strong complementary signal.

---

### 3.4 Addressing Research Question 3: Exploratory Cross-Center Parkinson's Transfer and Cross-Disorder Generalization
External transfer results under target-reference (transductive HC) calibration—standardized using the healthy controls of the target site without model retraining (this requires a labeled healthy reference sample at the target site, and those controls are also part of the evaluation set)—are exploratory benchmarks summarized in Table 3, Figure 2A, and Figure 3A (complete transfer matrices in Supplementary Tables S12, S13, S16, and S17; leakage and calibration/gain ablations in Supplementary Tables S19 and S20).

![](../reports/Figure_2_Tri_Center_Parkinsons_Generalization.png){{width=6.5in}}
**Figure 2. Tri-Center Parkinson's Cross-Site Transfer and Within-Patient Dopaminergic Washout Comparison.** (A) `GCN-OMST` ROC-AUC across Iowa (N = 149), UNM OFF (N = 50), and UCSD (N = 46); diagonal cells show 5-fold cross-validated AUC and off-diagonal cells show site-calibrated cross-center transfer without retraining. (B) Within-participant paired differences (ON - OFF with 95% confidence intervals) across N = 25 UNM PD patients following ≥ 15 h overnight washout; no statistically significant differences were detected across extracted markers (all paired p ≥ 0.16).

{t3}

![](../reports/Figure_3_Transdiagnostic_Transfer_and_Ablations.png){{width=6.5in}}
**Figure 3. Cross-Disorder Transfer Matrix and Graph Topology Ablations.** (A) Four-domain `GCN-OMST` ROC-AUC matrix (diagonal: 5-fold CV; off-diagonal: site-calibrated transfer). Positive transfer occurs among Parkinson's, dementia, and sleep-deprivation slowing regimes, whereas models trained on slowing cohorts and evaluated on outpatient MDD fall below chance (0.274–0.336). (B) Comparison of `GCN-OMST`, isolated-node `GCN-Identity` (A = I), and dense `GCN-Full` (A = 1) across five cohorts.

Two principal external-transfer patterns were observed:
1. **Directional Variability, Spectral Representation, and Calibration in Tri-Center Parkinson's Transfer:** Across the six directed PD transfer pairs on the pre-computed `specparam` node tensors (Table 3 and Supplementary Table S16), `GCN-OMST` ROC-AUC ranged from 0.590 to 0.811. Training on the largest cohort (Study 7 Iowa, N = 149) transferred moderately well to UNM (AUC = 0.811 [0.670, 0.935]) and UCSD (`GCN-OMST` 0.740 [0.583, 0.883]; `Random Forest` 0.804), whereas transfers originating from smaller cohorts were lower (`UCSD -> Iowa` and `UNM -> Iowa`: 0.686–0.694; `UCSD -> UNM`: `GCN-OMST` 0.621, `Tuned Logistic` 0.698; `UNM -> UCSD`: `GCN-OMST` 0.590, `Random Forest` 0.616). To compare spectral representations and amplifier gain perturbations on an identical footing, Supplementary Table S20 re-extracted `specparam` periodic residuals, relative band power, and unparameterized log₁₀ band power directly from cached BIDS Welch PSDs (`pd_tricenter_welch_psds.pt`); note that this direct PSD background-subtraction pipeline differs slightly from the pre-computed specparam node tensors in Table 3/S16, producing ~ 0.05–0.07 single-pair variation (`UNM -> UCSD` GCN 0.590 in Table 3 vs. 0.662 in Table S20; `UCSD -> UNM` GCN 0.621 vs. 0.565; `UCSD -> UNM` `Tuned Logistic` 0.698 vs. 0.578) and underscoring that individual cross-site transfer pairs such as `Iowa -> UNM` (0.811) should not be over-interpreted in isolation. Cross-site transfer performance varies substantially across centers and evaluation directions (0.590 to 0.811), indicating that while cross-center domain transfer is feasible under target-normative standardization, it remains exploratory and is not yet sufficiently stable for clinical deployment without local site calibration. Across all six PD pairs, the direct PSD ablations (Supplementary Table S20 Panel B and Panel C) gave four methodological results: (i) **`specparam` vs. Relative Band Power:** `specparam` periodic residuals achieved higher 6-pair mean ROC-AUC than relative band power across all three calibration regimes for `Random Forest` (`Target-HC` 0.712 vs. 0.693, `Global-Z` 0.718 vs. 0.702, `Uncalibrated` 0.707 vs. 0.689) and `Tuned Logistic Regression` (`Target-HC` 0.683 vs. 0.653, `Global-Z` 0.700 vs. 0.656, `Uncalibrated` 0.708 vs. 0.633); for `GCN-OMST`, `specparam` outperformed relative band power under `Uncalibrated Raw` transfer (0.735 vs. 0.640), whereas under `Global Cohort Z-Score` (0.707 vs. 0.723) and `Target-HC Normative` (0.698 vs. 0.713) relative band power yielded higher 6-pair mean AUC due to single-pair variance in `UNM -> UCSD` (0.662 vs. 0.812, N = 16 target HCs); (ii) **Unperturbed Accuracy vs. Gain-Perturbation Robustness:** on unperturbed empirical recordings (k = 1.0), unparameterized log₁₀ band power actually achieved slightly higher uncalibrated transfer ROC-AUC (`Tuned Logistic` 0.754 vs. 0.708; `Random Forest` 0.730 vs. 0.707) than `specparam` periodic residuals, indicating that the benefit of spectral parameterization here is algebraic invariance to gain mismatch (shared by relative band power) rather than higher baseline accuracy when recordings already share comparable scaling; (iii) **Trade-Offs of Target-HC Calibration:** standardizing against target-site healthy controls (a labeled reference sample) improved fixed-threshold (p = 0.5) Balanced Accuracy for linear `Tuned Logistic` on `specparam` (0.582 → 0.635, closely matched by a disjoint 50/50 Split-Target-HC holdout check at 0.633 and AUC = 0.680 vs. 0.683; `Random Forest` 0.655 → 0.670; unparameterized `Logistic` 0.622 → 0.621) and rescued `GCN-OMST` on unparameterized log₁₀ band power (0.471 → 0.685), but reduced 6-pair mean rank-order ROC-AUC on `specparam` features (`Tuned Logistic` 0.708 → 0.683, `GCN-OMST` 0.735 → 0.698) and increased `Tuned Logistic` Brier scores across representations (`specparam` 0.226 → 0.256) due to finite-sample variance when estimating 76 reference parameters from small control groups (N = 16–25 HCs); and (iv) **Amplifier Gain Perturbation and Leakage:** under explicit PSD gain scaling (PSD_k(f) = k² PSD(f); Supplementary Table S20 Panel C), `specparam` features maintained exact zero drift (|Δ X| = 0.0000), whereas unparameterized log₁₀ power suffered probability drift (|Δ p_LR| = 0.205–0.339), fixed-threshold Balanced Accuracy collapse to chance (0.500), non-linear `Random Forest` AUC collapse (0.730 → 0.446) under uniform scaling, and linear `Logistic Regression` AUC degradation (0.754 → 0.636) under heterogeneous multi-amplifier scaling (50%  0.5×, 50%  5.0×). Concurrently, leakage ablations (Supplementary Table S19) showed that ungrouped session mixing (`StratifiedKFold`) changed ROC-AUC in a model-dependent manner—inflating AUC by up to +0.102 in label-concordant PD cohorts (`GCN-OMST` 0.725 → 0.827 [+0.102], `Random Forest` 0.673 → 0.731 [+0.058], and `Tuned Logistic` 0.656 → 0.708 [+0.052] in UCSD; `Random Forest` 0.716 → 0.807 [+0.091] and `GCN-OMST` 0.681 → 0.762 [+0.081] in UNM, with `Tuned Logistic` 0.723 → 0.710 [-0.014] under participant-grouped inner tuning vs. 0.704 → 0.710 [+0.006] under ungrouped inner tuning)—while depressing non-linear AUC in the label-discordant paired crossover Study 1 cohort (`Random Forest` 0.629 → 0.559, Δ = -0.069), consistent with negative within-subject residual correlation across discordant sessions.
2. **Domain Alignment and Directional Asymmetry Across Disorders:** Within the affective spectrum, training on university depression screening transferred to outpatient MDD (`GCN-OMST` AUC = 0.833 [0.708, 0.936]), whereas the reverse eye-state-mismatched transfer (`Outpatient MDD EC -> Univ. Depression EO`) was near chance for `GCN-OMST` (0.463; `Random Forest` 0.677) and improved modestly when eye state was matched (`Eyes-Open -> Eyes-Open` `GCN-OMST` 0.569, `Logistic` 0.592; Supplementary Table S13). Across slowing conditions, positive transfer occurred for `Iowa PD -> Dementia` (0.792; `GCN-Identity` 0.835), `Sleep Deprivation -> Dementia` (0.840), `Sleep Deprivation -> UCSD PD` (0.802; Supplementary Tables S12 and S17), and `Sleep Deprivation -> Iowa PD` (0.732), with more modest reverse AUCs (0.563–0.647; Supplementary Table S17). Cross-disorder transfers involving outpatient MDD exhibited clear directional asymmetry: classifiers trained on `Iowa PD`, `Dementia`, or `Sleep Deprivation` and evaluated on `Outpatient MDD` yielded sub-chance ROC-AUCs (0.325, 0.336, and 0.274, respectively), as did `Outpatient MDD -> Dementia` (0.338), consistent with, but not independent evidence for, the opposite frontal periodic theta direction reported in Section 3.2 (sub-chance AUC is expected when a classifier trained on one sign of effect is applied to a cohort with the opposite sign, and these cohorts also differ in age, hardware, and eye state). By contrast, `Outpatient MDD -> Iowa PD` (`GCN-OMST` 0.539, though `Logistic`/`RF`/`GBDT` were 0.325–0.330) and `Outpatient MDD -> Sleep Deprivation` (0.490) hovered near chance because multivariate MDD weights combine frontal theta with aperiodic χ differences that are much weaker in PD and sleep loss.

---

### 3.5 Exploratory Cross-Sectional Clinical Scale Prediction
Exploratory out-of-fold 5-fold cross-validation results for continuous clinical scales are reported in Table 4 and Figure 4 (full 20-row regressor breakdown in Supplementary Table S15).

![](../reports/Figure_4_Clinical_Scale_Continuous_Prediction.png){{width=6.5in}}
**Figure 4. Cross-Sectional Clinical Scale Associations and Out-of-Fold Regression.** (A) Dementia cohort MMSE (N = 88) vs. posterior periodic alpha (r = +0.374, p = 3.36 × 10^-4; 5-fold `GBDT` out-of-fold r = +0.472). (B) Dementia MMSE vs. global periodic theta (r = -0.373, p = 3.43 × 10^-4). (C) Iowa PD cohort MoCA (N = 149) vs. global periodic theta (r = -0.343, p = 1.87 × 10^-5; 5-fold `SVR` r = +0.339). (D) Patient-only sensitivity analysis in Iowa PD (N = 100 PD patients excluding controls): out-of-fold `SVR` predictions vs. observed MoCA (r = +0.334, p = 6.74 × 10^-4, R² = 0.035, MAE = 2.841).

{t4}

In Study 5 (N = 88), full-cohort out-of-fold predictions correlated moderately with MMSE (`GBDT`: r = +0.472, p = 3.48 × 10^-6, R² = 0.209, MAE = 4.071). However, because all 29 healthy controls had ceiling scores (MMSE = 30.0), restricting regression strictly to the N = 59 dementia patients attenuated performance substantially (`GBDT`: r = +0.246, p = 0.0607, R² = 0.030, MAE = 3.172; linear and SVR models p ≥ 0.104), indicating that full-cohort MMSE predictability was largely driven by case–control separation. In Study 7, out-of-fold `SVR` predictions correlated modestly with MoCA both across the full sample (N = 149: r = +0.339, p = 2.32 × 10^-5, R² = 0.060, MAE = 2.397) and within PD patients alone (N = 100: r = +0.334, p = 6.74 × 10^-4, R² = 0.035, MAE = 2.841), showing that resting spectral features capture a statistically significant but small fraction (~ 3.5%) of inter-individual cognitive score variance among PD patients. By contrast, UPDRS-III motor severity among the N = 100 PD patients was not significantly predicted by any resting-state regressor (best `GBDT` r = +0.165, p = 0.1008, R² = -0.105, where negative out-of-fold R² indicates that cross-validated predictions had greater mean squared error than predicting the training-fold target mean).

---

## 4. Discussion

### 4.1 Summary of Findings Relative to Research Questions
Evaluating seven heterogeneous public EEG datasets (N = 566 participants in the primary recording state, 677 primary sessions) with a single open 19-channel pipeline and participant-grouped validation yielded three main findings. First, after separating periodic peaks from broadband 1/f^χ activity {{ref:Donoghue2020}}, resting frontal periodic theta had opposite group-level directions in these cohorts: lower values in outpatient MDD and university depression screening (k = 2; d_RE = -0.726, with a variance-floored HKSJ interval that includes zero), and higher values in PD and dementia cohorts (k = 4; d_RE = +0.856), together with a modest within-subject increase under acute sleep deprivation (dz = +0.284, d_IG = +0.267). The between-group Q statistic (62.38, p = 2.89 × 10^-15) is large mainly because the two groups have opposite signs, and the groups differ in diagnosis, age, hardware, eye state, medication, and site, so this is a cohort-level description and not evidence of a disease-intrinsic polarity. Second, no stable, reproducible classification advantage of `NormativeGCN` (`GCN-OMST` reference mean ROC-AUC 0.738, 5-seed mean 0.731) was detected over isolated-node or random-graph ablations (`GCN-Identity` 0.742, `GCN-Random` 0.754) or regularized classical classifiers (`Random Forest` 0.752, `Tuned Ridge` 0.749, `Tuned Logistic` 0.743), including across a one-at-a-time hyperparameter grid (Supplementary Table S7c); this is a statement about one custom GCN family and low-powered tests, not proof of no topological information. Third, cross-center PD transfer varied by training cohort size, feature pipeline, classifier family, and calibration regime, while exploratory cross-sectional cognitive score prediction was modest (R² = 0.035 for MoCA within PD patients) and attenuated in dementia once healthy controls were excluded.

### 4.2 Interpretation of Frontal Periodic Theta and Aperiodic Slope Differences
Prior resting-state EEG studies of depression and neurological slowing often relied on total narrowband power or relative band power, which blend narrowband oscillations with broadband 1/f offset and slope shifts {{ref:Donoghue2020,Donoghue2022}}. After subtracting the parameterized aperiodic floor, the cohort-level results show three patterns that are consistent with, but do not by themselves establish, distinct physiological regimes: (1) *depressive hypo-oscillatory attenuation* in both outpatient MDD (d = -0.826) and university depressive-symptom screening (d = -0.676), consistent with reduced resting frontomedial oscillatory engagement {{ref:CavanaghFrank2014,Newson2019}}; (2) *acute homeostatic sleep-pressure elevation* in healthy crossover sleep deprivation (d_z = +0.284, d_IG = +0.267) {{ref:DeGennaro2001}}; and (3) *chronic neurodegenerative slowing* across PD and AD/combined dementia (d = +0.559 to +1.279; participant-averaged UCSD PD d = +0.918), consistent with thalamocortical and corticocortical slowing {{ref:Stoffers2007,Bosboom2006,Stam2010}}. Importantly, the posterior Individual Alpha Peak Frequency (`IAPF`, 6–13 Hz center of gravity) analysis clarifies the spectral mechanism of this neurodegenerative theta elevation: while posterior `IAPF` shifted downward in PD (significantly in Study 7 Iowa PD by -0.92 Hz, p = 0.0011, with smaller non-significant downward shifts of -0.27 to -0.37 Hz in UCSD and UNM) and correlated negatively with 4–8 Hz periodic power, the PD elevation in frontal periodic theta remained statistically significant after controlling for `IAPF` across all three PD cohorts (p ≤ 0.0219), indicating that frontal periodic theta elevation in PD reflects both posterior alpha peak slowing toward the theta–alpha boundary and an `IAPF`-independent increase in frontal 4–8 Hz periodic activity. In AD, concurrent posterior periodic alpha loss and steeper aperiodic χ diverge from normative adult aging—where χ typically flattens with age {{ref:Voytek2015}}—and may reflect synaptic loss, cholinergic deafferentation, or altered cortical E/I tone {{ref:Gao2017,Stam2010}}.

Several alternative, non-mechanistic explanations and boundary conditions warrant caution. Resting EEG phenotypes in public archives can be influenced by unmeasured differences in vigilance, subclinical drowsiness, chronic antidepressant or anticholinergic exposure, skull conductivity, and eye-closure state. In particular, the three PD cohorts differed in resting eye-state and medication protocols—eyes-open with fixation in Study 4 UCSD (ON and OFF), alternating eyes-open/eyes-closed blocks in Study 6 UNM (ON and OFF), and eyes-open ON-medication in Study 7 Iowa—while Study 2 MDD (primary eyes-closed, secondary eyes-open) and Study 5 dementia were recorded eyes-closed. Within Study 2 itself, comparing the primary eyes-closed protocol (N = 58, d = -0.826, p = 0.0027) directly with the secondary eyes-open protocol (N = 60, d = -0.872, p = 0.0015) revealed nearly identical frontal periodic theta effect sizes and strong intra-individual concordance across overlapping participants (r = +0.742, p < 10^-9; Supplementary Table S13 Panel A), indicating that the attenuation in this cohort does not depend on eye-closure state. Because each group contrast and fold-nested standardization is referenced strictly against healthy controls recorded under the identical site protocol, and because frontal periodic theta elevation is replicated across all three PD centers in participant-averaged and ON-medication comparisons (noting that in Study 4 UCSD the small N = 15 OFF-medication subset alone did not reach significance, d = +0.559, p = 0.1303, and ON sessions exceeded OFF sessions) and survives eye-state matching in Study 2/Study 3 sensitivity checks (Supplementary Table S13 Panel B), eye state alone is therefore unlikely to explain the opposite signs; however, the depression and neurodegenerative groups also differ in age, recording hardware, site, and medication, and these factors cannot be separated with the present design. The finding describes opposite directions in these cohorts and is not evidence that the direction is intrinsic to the disorder; the Q_between statistic and the sub-chance MDD transfer AUCs follow from the opposite signs and are not independent confirmation. Moreover, in Study 3 only half of the BDI-II ≥ 14 university group had SCID-confirmed current or past MDD, meaning that frontal theta reduction in that dataset tracks self-reported depressive symptom burden rather than a uniform clinical diagnosis. Likewise, the difference between Study 6 UNM (where no statistically significant ON-OFF differences were detected after ≥ 15 h washout) and Study 4 UCSD (where ON sessions exhibited higher periodic theta and lower delta) may reflect protocol differences in washout duration, disease stage, or sample size (N = 25 vs. N = 15 pairs); absence of statistical significance in UNM should not be interpreted as formal pharmacological equivalence without pre-specified equivalence bounds.

### 4.3 Why Graph Message Passing Showed No Detectable Advantage on 19-Channel Resting EEG
Under this design, `GCN-OMST` showed no statistically significant or reproducible gain over `GCN-Identity` (A = I), `GCN-Random`, `GCN-Full`, `Tuned Ridge`, or `Random Forest` across 132 within-cohort evaluations and five random seeds (Supplementary Tables S11b and S21), and univariate frontal theta alone matched or exceeded `GCN-OMST` in five of the nine primary cohorts. This must be read within the limits of the design: only one custom 3,362-parameter GCN family was tested and no general-purpose EEG graph model (for example, DGCNN) was included; the smallest cohort (Study 4, N = 31 participants) produced a seed SD of 0.105 for `GCN-OMST`; and the nine-contrast Wilcoxon tests are low-powered and partly share control groups. A one-at-a-time hyperparameter grid (learning rate, epochs, hidden width; 914 to 8,674 parameters; Supplementary Table S7c) moved the 5-cohort mean AUC between 0.709 and 0.738 (per-cohort ranges 0.046–0.134) without yielding a configuration that reproducibly exceeded `Random Forest` (0.740) or univariate `Single θ` in Studies 3, 4, 6a, and 7; the best configurations were within 0.002 of `Random Forest`. Because these configurations were compared on the same test folds, this bounds the benefit of tuning from above and does not exclude gains for other graph architectures.

Three explanations are plausible but were not tested directly. First, a standard 10–20 montage has only 19 coarse scalp nodes (mean OMST degree ≈ 2.9 across sessions), which gives limited neighborhood structure for message passing {{ref:KipfWelling2017}}, and in small repeated-session cohorts partition variance (Study 4: 0.463–0.725 across seeds) exceeds the differences between architectures. Second, sensor-level PLI estimates from short resting recordings are noisy {{ref:Stam2007,Dimitriadis2017}}: scalar OMST GCE showed no case–control differences in any dataset, 12.56% (85/677) of symmetrized 19 × 19 PLI matrices were indefinite because PLI is a sign-based statistic and not a Gram matrix, and both direct 171-D Euclidean PLI and 190-D SPD-conditioned Riemannian tangent-space classifiers trailed the best node-feature models in all nine primary contrasts (Supplementary Table S22); message passing across noisy edges may also dilute localized oscillatory features. Third, leave-one-feature-out ablations (Supplementary Table S6) indicated that diagnostic information in these resting cohorts is concentrated in regional node features, chiefly frontal periodic theta and, in dementia, posterior alpha and χ, which regularized linear models and tree ensembles can use directly from the 76-D feature table.

The scope of this null result is therefore limited: with this GCN, no stable topological advantage was detected for static spectral node features on low-density 19-channel resting EEG, and graph models or Riemannian geometry may still help with high-density arrays, source-reconstructed cortical parcellations, task-evoked dynamic connectivity, or spatial covariance inputs.

### 4.4 Gain Handling, Cross-Site Calibration Boundaries, and Exploratory Score Prediction
The theoretical and empirical analyses (Supplementary Sections S1.3–S1.3.1 and Supplementary Table S20) clarify the exact relationship between amplifier gain invariance and cross-site generalization. Analytically, `specparam` periodic residuals and PLI are strictly invariant to positive multiplicative voltage scaling (v'(t) = k · v(t)), absorbing the +2log₁₀k spectral offset shift into b. Because relative band power computes the ratio of band power to total power (P_band / P_total, where [k² P_band] / [k² P_total] = P_band / P_total), the multiplicative factor k² cancels analytically in both numerator and denominator, resulting in identical zero numerical drift (|Δ X| = 0.0000) under uniform gain perturbation. However, relative power conflates narrowband oscillatory changes with broadband 1/f^χ tilt, yielding substantially lower cross-site transfer AUC than parameterized `specparam` residuals across linear classifiers (0.633 vs. 0.708 for uncalibrated logistic regression; Supplementary Table S20). Gain invariance is therefore a property of the feature definitions (log-spectral additivity; cancellation of k² in relative power), not a performance advantage, and the uncalibrated regime is only a reference condition because any target-site z-scoring cancels a uniform gain. Empirically, on unperturbed cross-site recordings (k = 1.0), unparameterized log₁₀ band power actually achieved slightly higher uncalibrated transfer ROC-AUC (`Tuned Logistic` 0.754 vs. 0.708; `Random Forest` 0.730 vs. 0.707) than `specparam` periodic residuals, showing that spectral parameterization confers *gain-perturbation robustness* rather than higher baseline accuracy when amplifiers already share comparable scaling. Under explicit gain perturbations (Supplementary Table S20 Panel C), while uniform target-site scaling leaves rank-order ROC-AUC of linear classifiers on unparameterized log₁₀ band power unchanged (because a constant additive shift across all test samples shifts linear logits by a constant scalar), unparameterized log₁₀ power suffers severe probability calibration drift (|Δ p_LR| = 0.205–0.339), fixed-threshold (p = 0.5) Balanced Accuracy collapse, non-linear `Random Forest` ROC-AUC collapse (0.730 → 0.446), and linear ROC-AUC degradation (0.754 → 0.636) under heterogeneous multi-amplifier scaling. Furthermore, `specparam` periodic residuals achieved higher 6-pair mean transfer ROC-AUC than gain-invariant relative band power across all calibration regimes for `Random Forest` and `Tuned Logistic Regression` (and for `GCN-OMST` under uncalibrated transfer, though not under global-z or target-HC calibration) because relative power conflates narrowband peaks with 1/f^χ slope. At the same time, multiplicative gain invariance does not eliminate non-multiplicative hardware, filter, reference, or demographic shifts across centers, and `Target-HC Normative` calibration (which requires a labeled healthy reference sample at the target site) involves a clear trade-off: it improves fixed-threshold (p = 0.5) linear Balanced Accuracy (`specparam` 0.582 → 0.635) and rescues `GCN-OMST` on unparameterized log₁₀ power (0.471 → 0.685), but slightly lowers 6-pair mean rank-order ROC-AUC on `specparam` features (`Tuned Logistic` 0.708 → 0.683, `GCN-OMST` 0.735 → 0.698) and increases Brier scores when estimating 76 reference parameters from small control cohorts (N = 16–25 HCs). Similarly, leakage ablations (Supplementary Table S19) show that session mixing (unlike global pre-standardization, which changed AUC by only -0.050 to +0.002) alters AUC in a model-dependent manner by up to +0.102 in label-concordant case–control designs while depressing non-linear AUC by up to -0.069 in label-discordant paired crossover designs. Finally, exploratory cross-sectional regression results underscore the gap between group discrimination and individual severity estimation: patient-only dementia MMSE prediction fell below conventional significance (r = +0.246, p = 0.0607), Iowa PD patient-only MoCA predictions explained only R² = 0.035 (MAE = 2.841, ~ 3.5%) of variance, and UPDRS-III motor scores were not predicted (p = 0.1008).

### 4.5 The Methodological Value of Rigorous Negative Benchmarks in Computational EEG
In computational neuroimaging and machine learning applied to clinical medicine, publication pipelines are historically skewed toward positive findings, novel architectural claims, and high reported classification accuracies {{ref:Kapur2012,Varoquaux2018}}. When emerging architectures such as Graph Neural Networks are introduced, enthusiasm often outpaces methodological rigor: evaluation protocols frequently omit participant-level grouping of repeated measurements, fail to test against regularized linear or tree-based tabular baselines, and omit topological controls such as edge permutations or isolated-node graphs.

The primary contribution of this work is therefore methodological and descriptive. By evaluating an identical, open pipeline with participant-grouped validation across seven heterogeneous datasets spanning four diagnostic domains, the present benchmark found that, once validation leakage and connectome indefiniteness were controlled, a compact GCN gave no detectable advantage over tuned classical models or even univariate regional features on standard 19-channel resting EEG. Such negative findings provide useful guardrails for translational work: they argue for including regularized tabular and univariate baselines before adopting over-parameterized architectures, and they point toward settings where network representations may carry more information, such as high-density sensor arrays, source-space reconstructions, and task-evoked dynamics.

### 4.6 Limitations and Future Directions
Key limitations constrain clinical extrapolation: (1) reducing 32- and 64-channel recordings to 19 channels sacrifices spatial resolution to prioritize compatibility with standard clinical archives; (2) scalp CAR and PLI reduce but do not eliminate volume conduction or residual electromyographic/ocular artifacts, and Study 2 lacked participant-level age records for ANCOVA; (3) target-site normative calibration requires a labeled healthy-control reference sample matched in recording protocol, and those controls overlap with the evaluation set except in the split-holdout check; (4) all clinical scale analyses were cross-sectional rather than longitudinal; (5) the depression and neurodegenerative cohorts differ in age, hardware, eye state, medication, and site, the depression group has only k = 2 cohorts (one a symptom-screening sample in which about half of the screening-positive participants had SCID-confirmed MDD), and cohort-level theta directions therefore cannot be attributed to the disorders themselves; (6) only one custom GCN family (3,362 parameters) and a one-at-a-time hyperparameter grid were evaluated, no general-purpose EEG graph model such as DGCNN was included, the smallest cohort has 31 participants (seed SD 0.105), and the across-contrast Wilcoxon tests are low-powered with partly shared controls; and (7) multi-seed stability was assessed for seven of the twelve configurations. Prospective, multi-wave cohorts with standardized medication documentation, matched control sites, and source-space benchmarking will be required to assess disease specificity and longitudinal prognostic or treatment-monitoring utility.

---

## 5. Conclusions

Across seven heterogeneous public EEG datasets (N = 566 participants in the primary recording state, 677 primary sessions), an open 19-channel pipeline with participant-grouped validation showed opposite resting frontal periodic theta directions in depression cohorts and in PD/dementia cohorts, which are cohort-level observations confounded by age, hardware, eye state, medication, and site. A compact `NormativeGCN` did not show a detectable advantage over classical regularized baselines on low-density montages in this design. Site-calibrated PD transfer varied substantially with training sample size and transfer direction, and cross-sectional cognitive score prediction remained modest. The principal contribution is an open, auditable baseline that documents what standard 19-channel resting EEG can and cannot separate with simple and graph-based models under participant-level validation, and it underscores the need for classical baselines and patient-only sensitivity checks before claiming clinical readiness.

---

## Data and Code Availability

### Data Availability Statement
All seven EEG datasets analyzed in this study are publicly available from open repositories:
1. Study 1 (Acute Sleep Deprivation): OpenNeuro `ds004902` (https://openneuro.org/datasets/ds004902)
2. Study 2 (Outpatient MDD): Figshare `4244171` (https://doi.org/10.6084/m9.figshare.4244171)
3. Study 3 (University Depression Screening): OpenNeuro `ds003478` (https://openneuro.org/datasets/ds003478)
4. Study 4 (UCSD Parkinson's Disease): OpenNeuro `ds002778` (https://openneuro.org/datasets/ds002778)
5. Study 5 (AHEPA Clinical Dementia): OpenNeuro `ds004504` (https://openneuro.org/datasets/ds004504)
6. Study 6 (UNM Parkinson's Disease): OpenNeuro `ds003490` (https://openneuro.org/datasets/ds003490)
7. Study 7 (Iowa Parkinson's Disease): OpenNeuro `ds004584` (https://openneuro.org/datasets/ds004584)

### Code Availability Statement
All preprocessing modules, OMST graph constructors, `NormativeGCN` and baseline implementations, statistical verification scripts, and figure-generation pipelines are openly available at https://github.com/chushuta-cmyk/transdiagnostic-eeg-benchmark.

### Author Contributions
C.T. conceived the study, formulated the analytical framework, implemented the software pipeline, conducted all statistical and machine learning analyses, prepared the figures and tables, and wrote and revised the manuscript.

### Ethics Statement
This work is a secondary computational analysis of de-identified, publicly available EEG datasets encompassing experimental crossover, symptom-screening, and clinical cohorts, and is exempt from additional institutional review under 45 CFR 46.104(d)(4) and institutional guidelines. All primary studies were approved by their respective institutional review boards with written informed consent obtained from participants.

### Acknowledgments
I thank the investigators and study participants who collected and shared the seven open-access EEG datasets via OpenNeuro and Figshare.

### Competing Interests
The author declares no competing financial or non-financial interests.

### Declaration of Generative AI in the Writing Process
During the preparation of this work, the author used Google Gemini for assistance in computational script automation, manuscript drafting, and formatting, and utilized OpenAI ChatGPT, DeepSeek (V3/R1), and Anthropic Claude for independent methodological cross-validation, code auditing, and linguistic refinement. After using these tools, the author reviewed and edited the content as needed and takes full responsibility for the content of the publication.
"""
    return tbl.compile_sequential_citations(body)


def build_supplementary_markdown() -> str:
    ts1 = tbl.build_supp_table_s1()
    ts4 = tbl.build_supp_table_s4()
    ts6 = tbl.build_supp_table_s6()
    ts7_7b = tbl.build_supp_table_s7_and_s7b()
    ts7c = tbl.build_supp_table_s7c()
    ts11_11b = tbl.build_supp_table_s11_and_s11b()
    ts14 = tbl.build_supp_table_s14()
    ts15 = tbl.build_supp_table_s15()
    ts16_17 = tbl.build_supp_table_s16_and_s17()
    ts18 = tbl.build_supp_table_s18()
    ts19_23 = tbl.build_supp_tables_s19_to_s23()

    supp = rf"""# Supplementary Material

**{EXACT_TITLE}**

**Chushu Tan**¹*

¹ School of Sociology, China University of Political Science and Law, Beijing 102249, China  
* Corresponding author: Chushu Tan, Guangzhou, Guangdong 510006, China; Email: chushuta@gmail.com; ORCID: 0009-0007-3093-0160

---

## Table of Contents

### Extended Supplementary Methods
- **Section S1.1:** Parameterized Spectral Decomposition via `specparam`
- **Section S1.2:** Orthogonal Minimum Spanning Tree (OMST) Graph Extraction and Cost Efficiency
- **Section S1.3:** Analytical Derivation of Multiplicative Voltage Gain Invariance
- **Section S1.3.1:** Hardware, Filtering, Referencing, and Demographic Non-Multiplicative Shift Boundaries
- **Section S1.4:** Normative Graph Convolutional Network Architecture and Pure PyTorch Implementation

### Supplementary Figures
- **Figure S1:** Single-Subject `specparam` Parameterization Example
- **Figure S2:** OMST Graph Topology and Cost Optimization Across Cohorts
- **Figure S3:** Multiplicative Gain Invariance Numerical Verification Across Four Orders of Magnitude
- **Figure S4:** Cohort-Averaged Power Spectral Densities Across All Seven Datasets
- **Figure S5:** Study Enrollment, Exclusion, and Participant Flowchart
- **Figure S6:** Empirical Permutation Null Distributions and Model Significance

### Supplementary Tables
- **Table S1:** Canonical 19-Channel 10–20 Electrode Coordinates and Empirical OMST Graph Metrics Across All 677 Primary Sessions
- **Table S2:** Numerical Verification of Feature Invariance Across Four Orders of Magnitude of Amplifier Gain
- **Table S3:** Multi-Cohort Study Designs, Demographics, Hardware Systems, and Clinical Scales (N = 566 Unique Participants, 677 Primary Sessions)
- **Table S4:** Complete Statistical Summary of Electrophysiological Markers Across All Multi-Cohort Comparisons (50 Pooled Hypotheses with Benjamini–Hochberg FDR Control)
- **Table S5:** Canonical 19-Channel 10–20 Label Mapping Across All Seven Public Datasets
- **Table S6:** Single-Biomarker Baselines and Leave-One-Feature-Out NormativeGCN Ablations Across All 11 Within-Cohort Contrasts
- **Table S7:** NormativeGCN Architecture and Training Hyperparameters
- **Table S7b:** Classical Machine Learning and Univariate Baseline Configurations
- **Table S7c:** NormativeGCN Hyperparameter Sensitivity Analysis Across Learning Rates, Training Epochs, and Hidden Capacities
- **Table S8:** Signal Preprocessing and Artifact Control Specifications
- **Table S8b:** Retained Clean Epoch Counts and Usable Recording Duration Across Datasets
- **Table S9:** Computational Environment and Open Repository Specifications
- **Table S10:** TRIPOD+AI Reporting Checklist for Machine Learning Diagnostic Models
- **Table S11:** Complete Within-Cohort Machine Learning Benchmark Across 11 Contrasts and 12 Models (132 Evaluations)
- **Table S11b:** Pairwise Wilcoxon Signed-Rank Comparisons Between `GCN_OMST` and Competing Models
- **Table S12:** Initial Four-Cohort 4 × 4 Cross-Dataset Transfer Matrix (12 Directed Pairs)
- **Table S13:** Recording-State Sensitivity: Within-Cohort Electrophysiological Robustness and Cross-Site Generalization Across Eye States
- **Table S14:** Within-Subject Paired Dopaminergic Medication Comparisons in Study 6 (UNM) and Study 4 (UCSD)
- **Table S15:** Complete Cross-Sectional Clinical Continuous Scale Regression Metrics Across All Four Regressors
- **Table S16:** Tri-Center Parkinson's Disease Site-Calibrated Cross-Center Transfer Matrix
- **Table S17:** Complete Cross-Disorder and Cross-Domain Transfer Matrix (14 Directed Pairs)
- **Table S18:** Age-Adjusted ANCOVA Sensitivity Analysis Across All Five Cohorts with Participant-Level Age Metadata
- **Table S19:** Empirical Quantification of Data Leakage Across Repeated-Session Cohorts
- **Table S20:** Cross-Site Parkinson's Disease Calibration Ablation and Empirical Amplifier Gain Perturbation
- **Table S21:** Multi-Seed Repeated 5-Fold Cross-Validation Benchmark Across Five Independent Random Seeds
- **Table S22:** Mathematical Audit of 19 × 19 PLI Matrix Definiteness and Euclidean vs. SPD-Conditioned Riemannian Connectivity Benchmarks
- **Table S23:** Random-Effects Meta-Analysis and Independence Sensitivity Suite for Resting Frontal Periodic Theta Polarity

---

## Section S1. Extended Supplementary Methods

### S1.1 Parameterized Spectral Decomposition via `specparam`
To separate narrowband neural oscillations from broadband 1/f^χ background activity, resting-state EEG channel power spectra were decomposed using `specparam` (v1.1.0; Donoghue et al., 2020; configured with `aperiodic_mode='fixed'`, `peak_width_limits=[1.0, 12.0] Hz`, `max_n_peaks=6`, `min_peak_height=0.05 B`, `peak_threshold=2.0 SD` across 1.0–40.0 Hz). Continuous 250 Hz signals were segmented into non-overlapping 4.0 s epochs. Power spectral density (PSD) was estimated via Welch's modified periodogram using 2.0 s Hamming sub-windows with 50% overlap, providing a frequency resolution of Δ f = 1 / 2.0 s = 0.50 Hz across 1.0–40.0 Hz. Spectra were modeled in fixed aperiodic mode as:

$$\\log_{{10}} \\mathrm{{PSD}}(f) = L(f) + \\sum_{{n=1}}^N a_n \\exp\\left(-\\frac{{(f - c_n)^2}}{{2 w_n^2}}\\right)$$

where the broadband aperiodic component L(f) is linear in log-log coordinates:

$$L(f) = b - \\chi \\log_{{10}}(f)$$

Here b denotes the broadband offset (in log₁₀ μV²/Hz, or Bell units, B), χ denotes the dimensionless aperiodic exponent, and (c_n, a_n, 2w_n) parameterize Gaussian peak center frequencies, log-power amplitudes above L(f), and bandwidths. Band-specific periodic residual power over [f_1, f_2] was computed as the band-averaged flattened residual spectrum (the empirical log-power minus the fitted aperiodic background L(f), rather than the non-negative Gaussian peak sum ∑_n G_n(f) alone):

$$P_{{\\mathrm{{periodic}}, [f_1, f_2]}} = \\frac{{1}}{{f_2 - f_1}} \\int_{{f_1}}^{{f_2}} \\left[\\log_{{10}} \\mathrm{{PSD}}(f) - L(f)\\right] df$$

evaluated across Periodic Theta (4.0–8.0 Hz, `x[:, 0]`), Periodic Delta (1.0–4.0 Hz, `x[:, 1]`), and Periodic Alpha (8.0–13.0 Hz, `x[:, 2]`), alongside channel-level χ (`x[:, 3]`). Note that because the 1.0–40.0 Hz zero-phase FIR bandpass filter applies a 1.0 Hz high-pass transition roll-off that slightly attenuates empirical power near 1.0–2.0 Hz relative to the linear log-log 1/f^χ background L(f) fitted across the full 1–40 Hz range, band-averaged Periodic Delta residuals (1.0–4.0 Hz) can exhibit small negative mean values (e.g., -0.059 B). Across cohorts, `specparam` fits achieved R² > 0.985 and mean absolute error < 0.045 B.

### S1.2 Data-Driven Connectomics via Orthogonal Minimum Spanning Trees (OMST)
Pairwise functional phase coupling was quantified using the Phase Lag Index (PLI; Stam et al., 2007) in the 4.0–13.0 Hz band to discard zero-lag volume-conducted synchronization. Edge distances were defined as:

$$c_{{ij}} = (1.0 - \\mathrm{{PLI}}_{{ij}}) + 10^{{-4}}$$

where the 10^-4 offset prevents zero-distance degeneracies in sparse graph routines. Starting from the primary Kruskal minimum spanning tree (N - 1 = 18 edges connecting all N = 19 channels), edge-disjoint orthogonal spanning trees were extracted iteratively and cumulative subgraphs G_m were scored by Global Cost Efficiency (Dimitriadis et al., 2017):

$$\\mathrm{{GCE}}(m) = \\frac{{E_{{\\mathrm{{global}}}}(G_m)}}{{E_{{\\mathrm{{global}}}}(G_{{\\mathrm{{full}}}})}} - \\mathrm{{Cost}}(G_m)$$

where E_global is average inverse shortest-path length and Cost(G_m) is normalized wiring weight.

Across all N = 566 analyzed participants and 677 primary recording sessions, optimal tree counts averaged M = 1.52 ± 0.50 (48.15% M = 1, 51.85% M = 2; Study 1 Sleep Deprivation: 1.59 ± 0.49, density 16.75% ± 5.19%; Study 2 Outpatient MDD: 1.33 ± 0.47, density 13.97% ± 4.98%; Study 3 Univ. Depression Screening: 1.45 ± 0.50, density 15.30% ± 5.26%; Study 4 UCSD PD: 1.61 ± 0.49, density 16.93% ± 5.19%; Study 5 Dementia: 1.53 ± 0.50, density 16.15% ± 5.28%; Study 6 UNM PD: 1.63 ± 0.49, density 17.12% ± 5.13%; Study 7 Iowa PD: 1.48 ± 0.50, density 15.61% ± 5.28%). Across all 677 graphs, selected edge counts averaged E = 27.33 ± 9.00, corresponding to mean node degree mean degree k̄ = 2.88 ± 0.95 (degree sum 54.66), edge density 15.98% ± 5.26%, mean GCE = 0.4939 ± 0.0438, and an isolated node rate of 0.0000% across all 12,863 node instances (677 × 19).

### S1.3 Analytical Proof of Multiplicative Amplifier Gain Invariance
Let raw EEG voltage undergo positive multiplicative scaling v'(t) = k · v(t) (k > 0). Its Welch PSD scales quadratically:

$$\\mathrm{{PSD}}'(f) = k^2 \\cdot \\mathrm{{PSD}}(f)$$

producing a uniform additive constant in base-10 logarithmic power:

$$\\log_{{10}} \\mathrm{{PSD}}'(f) = \\log_{{10}} \\mathrm{{PSD}}(f) + 2 \\log_{{10}} k$$

Because ∂ (2 log₁₀ k) / ∂ f = 0, fitting `specparam` to log₁₀ PSD'(f) yields:
1. χ' = χ (aperiodic slope invariance);
2. c'_n = c_n, w'_n = w_n, and a'_n = a_n (peak parameter invariance);
3. b' = b + 2 log₁₀ k (the unadjusted offset absorbs the entire multiplicative scale factor);
4. log₁₀ PSD'(f) - L'(f) = log₁₀ PSD(f) - L(f) (residual periodic spectrum invariance).

Similarly, because positive amplitude scaling preserves instantaneous analytic signal phase (∠(k · v(t)) = ∠ v(t)), pairwise PLI matrices and resulting OMST topologies are invariant to k > 0.

### S1.3.1 Boundary Conditions: Multiplicative Gain vs. Non-Multiplicative Hardware Shifts
Whereas multiplicative digitizer gain (v'(t) = k · v(t)) is cancelled analytically by `specparam` and PLI, real-world multi-center EEG archives also differ in non-multiplicative hardware characteristics: (1) analog high-pass and anti-aliasing filter rolloffs; (2) sensor impedance and thermal noise floors; (3) native physical reference geometry prior to CAR; and (4) environmental shielding and eye-state instructions. To address center-specific baseline offsets and demographic shifts without test-set leakage, multiplicative gain invariance is paired with **fold-nested normative standardization against training-fold (or target-site) healthy controls** (Main Text Section 2.5).

### S1.4 `NormativeGCN` Architecture and Optimization Specification
`NormativeGCN` is implemented in PyTorch (`src/harmonizer_and_graph.py`) and operates on 19-node binary adjacency matrices A ∈ {0, 1}^19 × 19 with 4-dimensional normatively calibrated node features z_i = [z_θ, z_δ, z_α, z_χ]^T ∈ R⁴:
- **Layer 1:** `PurePyTorchGCNLayer(4 -> 32)` with symmetric normalized adjacency Ã = D̃^-1/2(A + I_19)D̃^-1/2, ReLU activation, and Dropout (p = 0.25) (4 × 32 + 32 = 160 parameters).
- **Layer 2:** `PurePyTorchGCNLayer(32 -> 32)` with ReLU activation and Dropout (p = 0.25) (32 × 32 + 32 = 1,056 parameters).
- **Readout:** Concatenated global mean pooling (32-D) and global max pooling (32-D) across the 19 nodes, yielding h_G = [h_mean  ||  h_max] ∈ R^64.
- **Classification Head:** `Linear(64 -> 32)` (2,080 parameters) → ReLU → Dropout (p = 0.25) → `Linear(32 -> 2)` (66 parameters).
- **Parameter Count & Training:** Exactly 3,362 trainable parameters (160 + 1,056 + 2,080 + 66), trained via `torch.optim.Adam` (learning rate η = 0.005, weight decay λ = 1.0 × 10^-4, cross-entropy loss) for a fixed 50 epochs per fold with zero evaluation-fold early stopping (`scripts/17_run_comprehensive_7cohort_benchmark.py`).

---

## Section S2. Supplementary Tables

{ts1}

---

### Table S2. Numerical Verification of Feature Invariance Across Four Orders of Magnitude of Amplifier Gain
| Gain Factor (k) | Theoretical Offset b_theory (B) | Extracted Offset b_sim (B) | Offset Error Δb (B) | Periodic θ Error (B) | Periodic δ Error (B) | Periodic α Error (B) | Aperiodic χ Error |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.01 | -3.3500 | -3.3500 | < 1.0 × 10⁻¹⁵ | 2.1 × 10⁻¹⁵ | 1.8 × 10⁻¹⁵ | 2.5 × 10⁻¹⁵ | 1.2 × 10⁻¹⁵ |
| 0.10 | -1.3500 | -1.3500 | < 1.0 × 10⁻¹⁵ | 1.4 × 10⁻¹⁵ | 1.1 × 10⁻¹⁵ | 1.6 × 10⁻¹⁵ | 8.0 × 10⁻¹⁶ |
| 0.50 | +0.0479 | +0.0479 | < 1.0 × 10⁻¹⁵ | 8.0 × 10⁻¹⁶ | 5.0 × 10⁻¹⁶ | 9.0 × 10⁻¹⁶ | 4.0 × 10⁻¹⁶ |
| 1.00 (Base) | +0.6500 | +0.6500 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2.00 | +1.2521 | +1.2521 | < 1.0 × 10⁻¹⁵ | 9.5 × 10⁻¹⁶ | 7.0 × 10⁻¹⁶ | 1.1 × 10⁻¹⁵ | 5.0 × 10⁻¹⁶ |
| 10.00 | +2.6500 | +2.6500 | < 1.0 × 10⁻¹⁵ | 1.8 × 10⁻¹⁵ | 1.5 × 10⁻¹⁵ | 2.0 × 10⁻¹⁵ | 1.0 × 10⁻¹⁵ |
| 100.00 | +4.6500 | +4.6500 | < 1.0 × 10⁻¹⁵ | 2.7 × 10⁻¹⁵ | 2.3 × 10⁻¹⁵ | 3.1 × 10⁻¹⁵ | 1.6 × 10⁻¹⁵ |

*Note: Evaluated on 19-channel EEG epochs scaled by multiplicative factor k ∈ [0.01, 100] (b_0 = 0.650 B, χ_0 = 1.152). Theoretical offset shift Δ b = 2 log₁₀(k) is confirmed, while periodic residuals and χ exhibit deviations bounded by IEEE-754 double-precision machine epsilon (< 10^-14).*

---

### Table S3. Multi-Cohort Study Designs, Demographics, Hardware Systems, and Clinical Scales (N = 566 Unique Participants, 677 Primary Sessions)
| Study ID | Dataset & Setting | Analyzed N (Participants / Primary Sessions) | Group Breakdown | Age in Years (Mean ± SD) | Female (%) | Native Hardware & Channel Count | Sampling Rate (Native → Harmonized) | Protocol / Clinical Assessments |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- | :---: | :--- |
| **Study 1** | Acute Sleep Deprivation (`ds004902`, Experimental Crossover) | 71 / 142 | SD (71) vs. NS (71) [Paired] | 20.0 ± 1.4 (range 17–23) | 47.9% (34/71) | BrainVision actiCHamp (64 Ch) | 500 Hz → 250 Hz | Monitored 36 h wakefulness vs. normal sleep (EO) |
| **Study 2** | Outpatient Clinical MDD (`Figshare 4244171`, Psychiatric Clinic) | 58 / 58 (+60 EO) | MDD (30) vs. HC (28) [EC] | MDD 40.3 ± 12.9; HC 38.2 ± 15.6 | 37.9% (22/58) | Nihon Kohden (19 Ch native 10–20) | 256 Hz → 250 Hz | DSM-IV MDD; BDI (24.8 ± 8.6 vs. 3.9 ± 3.4) |
| **Study 3** | Univ. Depression Screening (`ds003478`, Student Screening) | 119 / 119 | BDI-II ≥ 14 (44) vs. ≤ 7 (75) | 19.4 ± 2.0 (N=118 w/ age) | 60.5% (72/119) | BioSemi ActiveTwo (64 Ch) | 500 Hz → 250 Hz | BDI-II screening; partial SCID (11 curr., 11 past MDD, 8 non-MDD, 14 uninterviewed) |
| **Study 4** | UCSD Parkinson's (`ds002778`, Neurology Clinic) | 31 / 46 | PD (15 ON + 15 OFF) vs. HC (16) | PD 63.2 ± 8.1; HC 63.5 ± 9.6 | 54.8% (17/31) | BioSemi ActiveTwo (32 Ch) | 512 Hz → 250 Hz | Eyes-open (fixation); counterbalanced ON vs. ≥ 12 h OFF withdrawal |
| **Study 5** | AHEPA Clinical Dementia (`ds004504`, Hospital Neurology) | 88 / 88 | AD (36), FTD (23), HC (29) | AD 72.8 ± 7.9; FTD 68.2 ± 7.3; HC 70.1 ± 6.8 | 55.7% (49/88) | Nihon Kohden (19 Ch native 10–20) | 500 Hz → 250 Hz | Eyes-closed resting; MMSE (AD 17.8 ± 4.5, FTD 22.2 ± 4.8, HC 30.0 ± 0.0) |
| **Study 6** | UNM Parkinson's (`ds003490`, Clinical Research Center) | 50 / 75 | PD (25 ON + 25 OFF) vs. HC (25) | PD 67.4 ± 8.7; HC 69.1 ± 8.2 | 42.0% (21/50) | BrainVision (64 Ch) | 500 Hz → 250 Hz | Alternating EO/EC resting; counterbalanced ON vs. ≥ 15 h overnight washout |
| **Study 7** | Iowa Parkinson's (`ds004584`, Hospital Movement Clinic) | 149 / 149 | PD (100) vs. HC (49) | PD 68.5 ± 8.1; HC 70.9 ± 7.6 | 42.3% (63/149) | BrainVision (64 Ch, native `Pz` ref) | 500 Hz → 250 Hz | Eyes-open (all PD patients ON medication); MoCA (N=149, 25.1 ± 3.6) & UPDRS-III (N=100, 12.5 ± 7.1) |

*Note: Master accounting totals N = 566 unique analyzed participants and 677 primary sessions (142 + 58 + 119 + 46 + 88 + 75 + 149 = 677), or 737 total sessions when including the N = 60 Study 2 eyes-open recordings (32 MDD, 28 HC). In Study 2, only group-level age summary statistics were published in Figshare.*

---

{ts4}

---

### Table S5. Canonical 19-Channel 10–20 Label Mapping Across All Seven Public Datasets
| Canonical 10–20 | Scalp Region | Study 1 Sleep (64 Ch) | Study 2 Outpatient MDD (19 Ch) | Study 3 Univ. Dep. (64 Ch) | Study 4 UCSD PD (32 Ch) | Study 5 Dementia (19 Ch) | Study 6 UNM PD (64 Ch) | Study 7 Iowa PD (64 Ch) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fp1 / Fp2** | Frontopolar | Direct | `EEG Fp1/Fp2-LE` | Direct | Direct | `EEG Fp1/Fp2` | Direct | Direct |
| **F7 / F3 / Fz / F4 / F8** | Frontal | Direct | `EEG F*-LE` | Direct | Direct | `EEG F*` | Direct | Direct |
| **T7 / T8** | Mid-Temporal | Direct (`T7/T8`) | Mapped from `T3/T4` | Direct (`T7/T8`) | Direct (`T7/T8`) | Mapped from `T3/T4` | Direct (`T7/T8`) | Direct (`T7/T8`) |
| **C3 / Cz / C4** | Central | Direct | `EEG C*-LE` | Direct | Direct | `EEG C*` | Direct | Direct |
| **P7 / P8** | Posterior Temporal | Direct (`P7/P8`) | Mapped from `T5/T6` | Direct (`P7/P8`) | Direct (`P7/P8`) | Mapped from `T5/T6` | Direct (`P7/P8`) | Direct (`P7/P8`) |
| **P3 / P4** | Parietal | Direct | `EEG P3/P4-LE` | Direct | Direct | `EEG P3/P4` | Direct | Direct |
| **Pz** | Parietal Midline | Direct | `EEG Pz-LE` | Direct | Direct | `EEG Pz` | Direct | Reconstructed via CAR |
| **O1 / O2** | Occipital | Direct | `EEG O1/O2-LE` | Direct | Direct | `EEG O1/O2` | Direct | Direct |

---

{ts6}

---

{ts7_7b}

---

{ts7c}

---

### Table S8. Signal Preprocessing and Artifact Control Specifications
| Pipeline Stage | Parameter / Setting | Software Implementation | Methodological Purpose |
| :--- | :--- | :--- | :--- |
| **Bandpass Filter** | 1.0–40.0 Hz | Zero-phase FIR (Hamming window, MNE `raw.filter`) | Attenuates slow drift (<1 Hz) and 50/60 Hz mains interference |
| **Resampling** | 250 Hz | Polyphase anti-aliased resampling (`raw.resample`) | Uniform 4 ms sample interval across 256, 500, and 512 Hz native archives |
| **Spatial Reference** | Common Average (CAR) | Instantaneous 19-channel mean subtraction | Harmonizes reference geometry and reconstructs native `Pz` reference in Study 7 |
| **Epoching** | 4.0 s non-overlapping | 1,000 samples/epoch (`mne.make_fixed_length_epochs`) | Supports 2.0 s Welch sub-windows (Δ f = 0.50 Hz) |
| **Artifact Threshold** | 150 μV peak-to-peak | Channel-wise max peak-to-peak screening | Rejects gross ocular blink and electromyographic bursts |
| **Geriatric Fallback** | 200 μV (≥ 8 epochs) | Lowest-amplitude epoch retention in Study 7 | Prevents tremor-related attrition in elderly movement-disorder recordings |

### Table S8b. Retained Clean Epoch Counts and Usable Recording Duration Across Datasets
| Study ID | Dataset | Condition | Analyzed Sessions | Threshold | Clean Epochs (Mean ± SD [Range]) | Clean Duration in Seconds (Mean ± SD [Range]) | Retained Epoch % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Study 1** | Sleep Deprivation (`ds004902`) | Paired SD & NS (EO) | 142 | 150 μV | 135.1 ± 21.2 [50–174] | 540.4 ± 84.8 s [200–696 s] | 90.1% |
| **Study 2a** | Outpatient MDD (`4244171`) | Eyes-Closed (Primary) | 58 | 150 μV | 71.6 ± 7.5 [43–94] | 286.4 ± 30.0 s [172–376 s] | 95.5% |
| **Study 2b** | Outpatient MDD (`4244171`) | Eyes-Open (Secondary) | 60 | 150 μV | 54.6 ± 18.3 [10–78] | 218.4 ± 73.2 s [40–312 s] | 87.4% |
| **Study 3** | Univ. Depression (`ds003478`) | Eyes-Open | 119 | 150 μV | 75.1 ± 21.1 [11–120] | 300.4 ± 84.4 s [44–480 s] | 83.4% |
| **Study 4** | UCSD Parkinson's (`ds002778`) | ON & OFF (EO Fixation) | 46 | 150 μV | 27.3 ± 14.9 [7–49] | 109.2 ± 59.6 s [28–196 s] | 78.0% |
| **Study 5** | AHEPA Dementia (`ds004504`) | Eyes-Closed | 88 | 150 μV | 171.9 ± 44.2 [12–319] | 687.7 ± 176.9 s [48–1,276 s] | 91.4% |
| **Study 6** | UNM Parkinson's (`ds003490`) | ON & OFF (Alternating EO/EC) | 75 | 150 μV | 73.8 ± 43.7 [9–173] | 295.0 ± 174.6 s [36–692 s] | 82.0% |
| **Study 7** | Iowa Parkinson's (`ds004584`) | Eyes-Open (PD ON Med) | 149 | 200 μV | 28.6 ± 12.8 [5–85] | 114.4 ± 51.2 s [20–340 s] | 76.3% |

---

### Table S9. Computational Environment and Open Repository Specifications
| Package / Resource | Version / Identifier | Role in Pipeline | URL |
| :--- | :--- | :--- | :--- |
| Python | 3.10+ | Core execution environment | https://www.python.org |
| MNE-Python | 1.6.1 | Signal filtering, resampling, CAR, epoching | https://mne.tools |
| `specparam` | 1.1.0 | Periodic–aperiodic PSD parameterization | https://specparam-tools.github.io |
| PyTorch | 2.2.1 | `NormativeGCN` training and evaluation | https://pytorch.org |
| `scikit-learn` | 1.4.1 | Classical classifiers, regressors, nested CV | https://scikit-learn.org |
| NetworkX | 3.2.1 | Kruskal spanning trees and OMST metrics | https://networkx.org |
| SciPy / `statsmodels` | 1.12.0 / 0.14+ | Welch/paired tests, Wilcoxon, OLS ANCOVA, BH-FDR | https://scipy.org |
| Benchmark Codebase | Git `main` | Open-source scripts and reproducibility pipeline | https://github.com/chushuta-cmyk/transdiagnostic-eeg-benchmark |

---

### Table S10. TRIPOD+AI Reporting Checklist
| Section / Topic | Item | Checklist Requirement | Manuscript & Supplementary Location |
| :--- | :---: | :--- | :--- |
| **Title** | 1 | Identify study as evaluating multivariable prediction models. | Exact Title: "{EXACT_TITLE}" |
| **Abstract** | 2 | Structured summary of background, methods, results, conclusions. | Main Text Abstract (<250 words) |
| **Background & Aims** | 3a–3b | Clinical/methodological context and explicit research questions. | Main Text §1 (four-paragraph introduction and three research questions) |
| **Data Sources** | 4a–4b | Study designs, public archives, inclusion/exclusion rules. | Main Text §2.1, Table S3, and Figure S5 (574 enrolled → 8 excluded → 566 analyzed) |
| **Predictors** | 5a–5b | Feature definitions and outcome-blind extraction. | Main Text §2.2–§2.4, §S1.1–§S1.3 (automated `specparam` + PLI-OMST extraction blind to labels) |
| **Sample Size** | 6 | Full accounting of participants and recording sessions. | N = 566 analyzed participants (primary recording state), 677 primary sessions (737 with Study 2 EO); Table S3 & Figure S5 |
| **Missing Data** | 7 | Handling of reference channels, artifacts, and missing covariates. | Main Text §2.2, Table S8, Table S18 note (Study 2 lacks individual ages; 1 missing age in Study 3) |
| **Model Specification** | 8a–8b | Full architectures, tuning grids, and fold-nested calibration. | Main Text §2.5–§2.6, §S1.4, Table S7, and Table S7b |
| **Validation** | 9–10 | Internal group-stratified CV, external site-calibrated transfer, metrics. | Main Text §2.5–§2.7; ROC-AUC (1,000-bootstrap 95% CIs), Balanced Accuracy, Wilcoxon tests, r, R², MAE |
| **Results** | 11–15 | Participant flow, baseline comparisons, ablations, permutations. | Main Text Tables 1–4, Figures 1–4; Supplementary Tables S4, S6, S11, S11b, S12–S23; Figures S5–S6 |
| **Interpretation & Scope** | 16–18 | Alternative explanations, 19-channel scope limits, cross-sectional bounds. | Main Text §4.1–§4.6 |
| **Open Science & Ethics** | 19–22 | Persistent dataset IDs, GitHub code URL, IRB exemption, competing interests. | Main Text Data and Code Availability |

---

{ts11_11b}

---

### Table S12. Initial Four-Cohort 4 × 4 Cross-Dataset Transfer Matrix (12 Directed Pairs)
| Source Dataset | Target Dataset | Source N | Target N | `NormativeGCN` AUC [95% CI] | `NormativeGCN` Bal. Acc. | Logistic Regression AUC [95% CI] | Logistic Bal. Acc. |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Study 1: Sleep Deprivation** | Study 2: Outpatient MDD (EC) | 142 | 58 | 0.277 [0.136, 0.422] | 42.3% | 0.376 [0.233, 0.541] | 46.9% |
| **Study 1: Sleep Deprivation** | Study 3: Univ. Depression (EO) | 142 | 119 | 0.346 [0.251, 0.450] | 40.1% | 0.319 [0.231, 0.422] | 39.7% |
| **Study 1: Sleep Deprivation** | Study 4: UCSD Parkinson's (All) | 142 | 46 | **0.802 [0.663, 0.929]** | **68.8%** | 0.652 [0.498, 0.803] | 56.7% |
| **Study 2: Outpatient MDD (EC)** | Study 1: Sleep Deprivation | 58 | 142 | 0.490 [0.399, 0.588] | 49.3% | 0.439 [0.342, 0.539] | 50.0% |
| **Study 2: Outpatient MDD (EC)** | Study 3: Univ. Depression (EO) | 58 | 119 | 0.465 [0.353, 0.581] | 50.5% | **0.604 [0.485, 0.705]** | 50.7% |
| **Study 2: Outpatient MDD (EC)** | Study 4: UCSD Parkinson's (All) | 58 | 46 | 0.406 [0.235, 0.589] | 50.2% | 0.306 [0.154, 0.479] | 47.1% |
| **Study 3: Univ. Depression (EO)** | Study 1: Sleep Deprivation | 119 | 142 | 0.410 [0.313, 0.506] | 44.4% | 0.392 [0.304, 0.492] | 45.1% |
| **Study 3: Univ. Depression (EO)** | Study 2: Outpatient MDD (EC) | 119 | 58 | **0.832 [0.706, 0.935]** | **81.1%** | **0.752 [0.621, 0.881]** | 63.1% |
| **Study 3: Univ. Depression (EO)** | Study 4: UCSD Parkinson's (All) | 119 | 46 | 0.271 [0.121, 0.446] | 37.5% | 0.342 [0.177, 0.520] | 39.0% |
| **Study 4: UCSD Parkinson's (All)** | Study 1: Sleep Deprivation | 46 | 142 | **0.590 [0.491, 0.682]** | 54.2% | **0.584 [0.489, 0.679]** | 54.9% |
| **Study 4: UCSD Parkinson's (All)** | Study 2: Outpatient MDD (EC) | 46 | 58 | 0.385 [0.235, 0.534] | 52.5% | 0.213 [0.098, 0.348] | 33.7% |
| **Study 4: UCSD Parkinson's (All)** | Study 3: Univ. Depression (EO) | 46 | 119 | 0.475 [0.365, 0.581] | 44.8% | 0.385 [0.285, 0.487] | 45.5% |

*Note: Site-calibrated direct transfer across Studies 1–4 (`reports/multicohort_full_transfer_matrix.csv`) without fine-tuning. Brackets denote 1,000-bootstrap 95% confidence intervals. Slight third-decimal `NormativeGCN` differences (≤ 0.003, e.g., 0.832 vs. 0.833 and 0.277 vs. 0.274) relative to the expanded 7-cohort run in Table 3 and Table S17 (`reports/comprehensive_transdiagnostic_transfer_matrix.csv`) reflect independent PyTorch weight-initialization seeds, whereas deterministic `Logistic Regression` AUCs match identically.*

---

### Table S13. Recording-State Sensitivity: Within-Cohort Electrophysiological Robustness and Cross-Site Generalization Across Eye States

#### Panel A. Within-Cohort Electrophysiological and Machine-Learning Benchmark in Study 2 Outpatient MDD (Eyes-Closed vs. Eyes-Open)
| Diagnostic Metric / Classifier Model | Eyes-Closed Protocol (Primary, N = 58: 30 MDD vs. 28 HC) | Eyes-Open Protocol (Secondary, N = 60: 32 MDD vs. 28 HC) | Paired Intra-Individual Concordance (N = 55 Overlapping Participants) |
| :--- | :---: | :---: | :---: |
| **Frontal Periodic Theta (4–8 Hz Residual)** | d = -0.826 [-1.365, -0.287], p = 0.0027 | d = -0.872 [-1.405, -0.340], p = 0.0015 | r = +0.742, p = 7.82 × 10^-11 |
| **Frontal Aperiodic Exponent (χ)** | d = -0.370 [-0.890, +0.151], p = 0.1610 | d = -0.428 [-0.942, +0.086], p = 0.1009 | r = +0.814, p = 1.12 × 10^-13 |
| **Global Cost Efficiency (GCE, OMST)** | d = +0.494 [-0.029, +1.017], p = 0.0728 | d = +0.052 [-0.456, +0.559], p = 0.8407 | r = +0.418, p = 0.0015 |
| **NormativeGCN (OMST Topology)** | AUC = 0.902 [0.821, 0.974], Bal. Acc: 81.3% | AUC = 0.828 [0.709, 0.929], Bal. Acc: 78.1% | Within-Site Cross-State AUC = 0.937 [0.874, 0.982] |
| **Tuned Logistic Regression** | AUC = 0.889 [0.793, 0.967], Bal. Acc: 81.0% | AUC = 0.814 [0.689, 0.918], Bal. Acc: 74.6% | Within-Site Cross-State AUC = 0.931 [0.849, 0.994] |
| **Tuned Ridge Classifier** | AUC = 0.896 [0.806, 0.966], Bal. Acc: 77.7% | AUC = 0.886 [0.780, 0.969], Bal. Acc: 88.4% | Within-Site Cross-State AUC = 0.918 [0.835, 0.986] |
| **Random Forest (100 Trees)** | AUC = 0.933 [0.864, 0.982], Bal. Acc: 79.5% | AUC = 0.877 [0.777, 0.960], Bal. Acc: 81.7% | Within-Site Cross-State AUC = 0.925 [0.842, 0.990] |
| **Single θ (5-Ch Univariate Frontal Theta)** | AUC = 0.694 [0.545, 0.828], Bal. Acc: 65.5% | AUC = 0.701 [0.568, 0.835], Bal. Acc: 67.6% | Within-Site Cross-State AUC = 0.697 [0.558, 0.832] |

#### Panel B. Cross-Site Generalization Between Study 2 Outpatient MDD and Study 3 University Depression Screening Across Eye States
| Transfer Configuration | Source Dataset & State | Target Dataset & State | `NormativeGCN` AUC [95% CI] | `NormativeGCN` Bal. Acc. | Logistic Regression AUC [95% CI] | Logistic Bal. Acc. |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **State-Mismatched Forward** | Study 3 Univ. Dep. (EO, N = 119) | Study 2 Outpatient MDD (EC, N = 58) | **0.832 [0.706, 0.935]** | **81.1%** | 0.752 [0.621, 0.881] | 63.1% |
| **State-Matched Forward** | Study 3 Univ. Dep. (EO, N = 119) | Study 2 Outpatient MDD (EO, N = 60) | 0.758 [0.638, 0.875] | 68.1% | **0.795 [0.664, 0.903]** | **69.6%** |
| **State-Mismatched Reverse** | Study 2 Outpatient MDD (EC, N = 58) | Study 3 Univ. Dep. (EO, N = 119) | 0.465 [0.353, 0.581] | 50.5% | **0.604 [0.485, 0.705]** | 50.7% |
| **State-Matched Reverse** | Study 2 Outpatient MDD (EO, N = 60) | Study 3 Univ. Dep. (EO, N = 119) | 0.569 [0.455, 0.669] | 54.2% | **0.592 [0.483, 0.705]** | **55.2%** |
| **Within-Site Cross-State** | Study 2 Outpatient MDD (EO, N = 60) | Study 2 Outpatient MDD (EC, N = 58) | **0.937 [0.874, 0.982]** | 82.6% | 0.931 [0.849, 0.994] | **93.3%** |
| **Within-Site Cross-State** | Study 2 Outpatient MDD (EC, N = 58) | Study 2 Outpatient MDD (EO, N = 60) | 0.875 [0.777, 0.953] | 67.2% | **0.887 [0.792, 0.963]** | **82.6%** |

*Note: Panel A evaluates within-cohort electrophysiological effect sizes (Welch's t, Cohen's d with 95% CI) and 5-fold cross-validated classifier performance across the primary Eyes-Closed (N=58, identical to Main Text Table 1, Table 2, and Supplementary Table S11) and secondary Eyes-Open (N=60) recordings in Study 2 Outpatient MDD (`reports/mumtaz_eo_model_comparison.csv`), alongside intra-individual Pearson correlations across the N=55 shared participants recorded under both states. Note that row 2 reports the region-specific Frontal Aperiodic Exponent (χ) to mirror Frontal Periodic θ, whereas Table 1 reports the 19-channel Global Exponent (d = -0.744, p = 0.0062). Both eye states exhibit strong frontal periodic theta attenuation (d = -0.826 and d = -0.872) with high within-subject consistency (r = +0.742, p < 10^-9). Panel B evaluates cross-dataset generalization with Study 3 University Depression Screening (N=119, recorded eyes-open; `reports/mumtaz_eo_state_matched_transfer.csv`). Matching the eyes-open recording state (`Study 2 EO -> Study 3 EO`) improves reverse GCN transfer from 0.465 to 0.569 and linear logistic transfer to 0.592. Note that within-site cross-state rows in Panel B evaluate models trained on one eye state and tested on the other within Study 2, demonstrating high cross-state diagnostic consistency (AUC = 0.875–0.937).*

---

{ts14}

---

{ts15}

---

{ts16_17}

---

{ts18}

---

{ts19_23}

---

## Section S3. Supplementary Figures (Figures S1–S6)

![](../reports/fig_S1_specparam_example.png){{width=6.2in}}

**Figure S1. Illustrative Single-Channel Spectral Parameterization (`specparam`) and Isolated Periodic Residual Spectrum.** *(A) Raw log₁₀ PSD(f) (dark slate), fitted aperiodic background L(f) = b - χ log₁₀ f (red dashed; b = 0.650 B, χ = 1.152), and combined model fit (R² = 0.993, blue solid). (B) Flattened periodic residual spectrum after subtracting L(f), isolating narrowband theta and alpha peaks above the 0.0 B baseline.*

---

![](../reports/fig_S2_omst_topology.png){{width=5.4in}}

**Figure S2. Canonical 19-Channel 10–20 Orthogonal Minimum Spanning Tree (OMST) Connectome Schematic.** *Primary Minimum Spanning Tree (Tree 1, blue solid lines, 18 edges) and orthogonal second spanning tree (Tree 2, amber dashed lines, 18 disjoint edges; curved arcs improve visual legibility for midline and homologous chords). Across all N = 566 analyzed participants and 677 primary sessions, GCE optimization selected M = 1.52 ± 0.50 trees (E = 27.33 ± 9.00 edges, mean degree mean degree k̄ = 2.88 ± 0.95, edge density 15.98% ± 5.26%) with 0.0000% isolated nodes.*

---

![](../reports/fig_S3_gain_invariance.png){{width=6.0in}}

**Figure S3. Numerical Verification of Multiplicative Amplifier Gain Invariance.** *(A) Unadjusted broadband offset b shifts linearly as Δ b = 2 log₁₀(k) across multiplicative gain factors k ∈ [0.01, 100]. (B) Absolute numerical deviations of periodic band residuals (θ, δ, α) and aperiodic exponent χ remain below 4 × 10^-15 across four orders of magnitude.*

---

![](../reports/fig_S4_cohort_spectra.png){{width=6.5in}}

**Figure S4. Representative Cohort Power Spectra: Total Raw log₁₀ PSD vs. Isolated Periodic Residuals.** *Top row (A1–D1): Grand-average total log₁₀ PSD (1–40 Hz) for Study 1 Sleep Deprivation (N = 71 pairs), Study 2 Outpatient MDD (N = 30 MDD vs. 28 HC), Study 3 University Depression Screening (N = 44 BDI-II ≥ 14 vs. 75 controls BDI-II ≤ 7), and Study 4 UCSD Parkinson's Disease (N = 15 PD-ON vs. 16 HC). Bottom row (A2–D2): Corresponding periodic residual spectra after subtracting fitted aperiodic 1/f^χ floors.*

---

![](../reports/fig_S5_enrollment_flowchart.png){{width=6.5in}}

**Figure S5. Multi-Dataset Participant Enrollment, Quality Control, and Harmonization Flowchart.** *From N = 574 enrolled participants (685 primary sessions; 745 including Study 2 eyes-open) across seven public archives (6 OpenNeuro + 1 Figshare), 8 participants (1.4%) were excluded under pre-specified artifact (N = 6 in Study 2) and borderline screening (N = 2 with 10 ≤ BDI-II ≤ 13 in Study 3) criteria, yielding N = 566 participants analyzed in the primary recording state across 677 primary sessions (737 total sessions including Study 2 eyes-open).*

---

![](../reports/fig_S6_permutation_tests.png){{width=6.5in}}

**Figure S6. Label-Permutation Null Distributions for `NormativeGCN` Benchmarks (1,000 Permutations).** *Null ROC-AUC histograms against observed 50-epoch `GCN-OMST` performance for: (A) Study 2 Outpatient MDD 5-fold CV (AUC = 0.902); (B) Study 3 University Depression Screening 5-fold CV (AUC = 0.647); (C) Study 3 → Study 2 cross-dataset transfer (AUC = 0.833); (D) Study 1 Sleep Deprivation → Study 4 UCSD PD transfer (AUC = 0.802); (E) Study 1 Sleep Deprivation → Study 2 Outpatient MDD sub-chance transfer (AUC = 0.274); and (F) Study 3 University Depression → Study 4 UCSD PD sub-chance transfer (AUC = 0.271). All observed departures from chance (0.50) exceed the permutation null (two-sided  p < 0.002).*
"""
    return supp


def format_docx_styling(doc_path: Path):
    doc = docx.Document(doc_path)

    for s in doc.sections:
        s.top_margin = Inches(0.9)
        s.bottom_margin = Inches(0.9)
        s.left_margin = Inches(0.9)
        s.right_margin = Inches(0.9)

    style_normal = doc.styles["Normal"]
    style_normal.font.name = "Times New Roman"
    style_normal.font.size = Pt(10.5)
    style_normal.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    in_references = False
    for p in doc.paragraphs:
        txt = p.text.strip()
        if p.style.name.startswith("Heading 1"):
            in_references = False
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(6)
            for r in p.runs:
                r.font.name = "Times New Roman"
                r.font.size = Pt(14)
                r.font.bold = True
                r.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        elif p.style.name.startswith("Heading 2"):
            in_references = txt == "References"
            p.paragraph_format.space_before = Pt(11)
            p.paragraph_format.space_after = Pt(4)
            for r in p.runs:
                r.font.name = "Times New Roman"
                r.font.size = Pt(12)
                r.font.bold = True
                r.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
        elif p.style.name.startswith("Heading 3"):
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(3)
            for r in p.runs:
                r.font.name = "Times New Roman"
                r.font.size = Pt(11)
                r.font.bold = True
                r.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
        elif in_references and re.match(r"^\[\d+\]\s", txt):
            p.paragraph_format.left_indent = Inches(0.35)
            p.paragraph_format.first_line_indent = Inches(-0.35)
            p.paragraph_format.line_spacing = 1.12
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(3.5)
            for r in p.runs:
                r.font.name = "Times New Roman"
                r.font.size = Pt(9.5)
        elif txt.startswith(("Figure 1.", "Figure 2.", "Figure 3.", "Figure 4.", "Figure S")):
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(10)
            p.paragraph_format.line_spacing = 1.15
            for r in p.runs:
                r.font.name = "Times New Roman"
                r.font.size = Pt(9.5)
        else:
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_after = Pt(4)
            for r in p.runs:
                r.font.name = "Times New Roman"

    for t in doc.tables:
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        tblPr = t._tbl.tblPr
        borders_xml = parse_xml(
            r"""
            <w:tblBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
                <w:top w:val="single" w:sz="12" w:space="0" w:color="0F172A"/>
                <w:left w:val="none"/>
                <w:bottom w:val="single" w:sz="12" w:space="0" w:color="0F172A"/>
                <w:right w:val="none"/>
                <w:insideH w:val="none"/>
                <w:insideV w:val="none"/>
            </w:tblBorders>
            """
        )
        tblPr.append(borders_xml)

        if len(t.rows) > 0:
            for cell in t.rows[0].cells:
                tcPr = cell._tc.get_or_add_tcPr()
                tcPr.append(
                    parse_xml(
                        r"""
                        <w:tcBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
                            <w:bottom w:val="single" w:sz="6" w:space="0" w:color="0F172A"/>
                        </w:tcBorders>
                        """
                    )
                )
                tcPr.append(
                    parse_xml(
                        r'<w:shd xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:fill="F8FAFC"/>'
                    )
                )
                for cp in cell.paragraphs:
                    cp.paragraph_format.space_before = Pt(2)
                    cp.paragraph_format.space_after = Pt(2)
                    for r in cp.runs:
                        r.font.name = "Times New Roman"
                        r.font.size = Pt(8.5)
                        r.font.bold = True

            for row in t.rows[1:]:
                for cell in row.cells:
                    for cp in cell.paragraphs:
                        cp.paragraph_format.space_before = Pt(1.5)
                        cp.paragraph_format.space_after = Pt(1.5)
                        for r in cp.runs:
                            r.font.name = "Times New Roman"
                            r.font.size = Pt(8.0)

    doc.save(doc_path)


def patch_settings_with_namespace(docx_path: Path):
    """Declare xmlns:m on <w:settings> before injecting <m:mathPr> so strict XML parsers succeed."""
    math_ns = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'
    math_pr = (
        '<m:mathPr><m:mathFont m:val="Cambria Math"/><m:brkBin m:val="before"/>'
        '<m:brkBinSub m:val="--"/><m:smallFrac m:val="0"/><m:dispDef/>'
        '<m:lMargin m:val="0"/><m:rMargin m:val="0"/><m:defJc m:val="centerGroup"/>'
        '<m:wrapIndent m:val="1440"/><m:intLim m:val="subSup"/><m:naryLim m:val="undOvr"/></m:mathPr>'
    )
    with zipfile.ZipFile(docx_path, "r") as z_in:
        contents = {item.filename: z_in.read(item.filename) for item in z_in.infolist()}

    settings_xml = contents["word/settings.xml"].decode("utf-8")
    # Ensure xmlns:m is declared on <w:settings ...>
    if "xmlns:m=" not in settings_xml:
        settings_xml = re.sub(r"(<w:settings\b[^>]*)(>)", rf"\1 {math_ns}\2", settings_xml, count=1)
    if "<m:mathPr>" not in settings_xml:
        open_end = settings_xml.find(">", settings_xml.find("<w:settings")) + 1
        settings_xml = settings_xml[:open_end] + math_pr + settings_xml[open_end:]

    # Validate immediately via ElementTree
    ET.fromstring(settings_xml.encode("utf-8"))
    contents["word/settings.xml"] = settings_xml.encode("utf-8")

    with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as z_out:
        for fname, data in contents.items():
            z_out.writestr(fname, data)


def compile_docx(md_path: Path, docx_path: Path):
    cmd = [
        "pandoc",
        str(md_path),
        f"--resource-path={MANUSCRIPT_DIR}:{REPORTS_DIR}",
        "-f",
        "markdown+tex_math_dollars",
        "-t",
        "docx",
        "-o",
        str(docx_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Pandoc failed for {md_path.name}:\n{res.stderr}")
    format_docx_styling(docx_path)
    patch_settings_with_namespace(docx_path)


def run_strict_audits(main_md: str, supp_md: str):
    print("\n🔍 Running strict automated verification audits...")

    # 1. XML & python-docx integrity across every archive member
    for p in (V2_DOCX, SUPP_DOCX):
        with zipfile.ZipFile(p, "r") as z:
            xml_count = 0
            for name in z.namelist():
                if name.endswith((".xml", ".rels")):
                    ET.fromstring(z.read(name))
                    xml_count += 1
        d = docx.Document(p)
        print(f"   ✓ {p.name}: {xml_count} XML parts parsed cleanly; {len(d.paragraphs)} paragraphs, {len(d.tables)} tables")

    # 2. Verify 43 separate reference paragraphs with hanging indents in V2_DOCX
    d_main = docx.Document(V2_DOCX)
    ref_paras = [p for p in d_main.paragraphs if re.match(r"^\[\d+\]\s", p.text.strip())]
    assert len(ref_paras) == 43, f"Expected 43 separate reference paragraphs, got {len(ref_paras)}"
    for idx, rp in enumerate(ref_paras, start=1):
        assert rp.text.strip().startswith(f"[{idx}] "), f"Ref order mismatch at {idx}: {rp.text[:30]}"
        assert rp.paragraph_format.first_line_indent == Inches(-0.35), "Missing hanging indent on reference"
    print("   ✓ References: 43 separate paragraphs in exact [1]..[43] order with hanging indent")

    # 3. Verify sequential first-appearance order of citations in Main Manuscript body
    body_before_refs = main_md.split("## References")[0]
    seen_max = 0
    for m in re.finditer(r"\[(\d+(?:[–\-,\s]+\d+)*)\]", body_before_refs):
        token = m.group(1)
        for part in [x.strip() for x in token.split(",")]:
            if "–" in part:
                a, b = [int(v) for v in part.split("–")]
                nums = list(range(a, b + 1))
            else:
                nums = [int(part)]
            for n in nums:
                if n > seen_max:
                    assert n == seen_max + 1, f"Citation [{n}] appeared before [{seen_max + 1}]! Context: {m.group(0)}"
                    seen_max = n
    assert seen_max == 43, f"Expected max citation 43, got {seen_max}"
    print("   ✓ In-text citation sequence: strictly monotonic 1 -> 43 on first appearance")

    # 4. Single-author pronoun & banned hyperbolic intensifier check
    combined = main_md + "\n" + supp_md
    plural_hits = re.findall(r"\b(we|our|ours|us)\b", combined, flags=re.IGNORECASE)
    assert not plural_hits, f"Found plural pronouns: {plural_hits}"
    banned_words = [
        "crucially", "pivotal", "profound", "profoundly", "genuine",
        "indispensable", "catastrophic", "overwhelmingly", "stark", "statistically indistinguishable"
    ]
    for w in banned_words:
        hits = re.findall(rf"\b{re.escape(w)}\b", combined, flags=re.IGNORECASE)
        assert not hits, f"Found banned hyperbolic phrase '{w}': {hits}"
    print("   ✓ Voice & tone audit: 0 plural pronouns, 0 hyperbolic intensifiers")

    # 5. Cross-reference & stale-text regression checks
    assert "Figure S6" not in main_md.split("### 2.6")[0].split("### 2.5")[1], "§2.5 still cites Figure S6!"
    assert "Table S11b" in main_md, "Main manuscript must cite Table S11b for Wilcoxon tests!"
    assert "Table S7c" in main_md and "Table S7c" in supp_md, "Table S7c GCN hyperparameter sensitivity must be present!"
    assert "Table S19" in main_md and "Table S20" in main_md and "Table S21" in main_md and "Table S22" in main_md and "Table S23" in main_md, "Extended Tables S19-S23 must be cited in main text!"
    assert "Table S19" in supp_md and "Table S23" in supp_md, "Extended Tables S19-S23 must be present in Supplementary Material!"
    assert "### S1.5" not in supp_md, "Supplementary S1.5 post-hoc power section must be removed!"
    assert "AdamW" not in combined and "35 epochs" not in combined and "35 training epochs" not in combined, "Stale AdamW/35 epochs found!"
    assert "Multilayer Perceptron" not in supp_md, "Stale MLP found in Table S7b!"
    assert EXACT_TITLE in supp_md, "Exact title missing in Supplementary Material!"
    assert "Challenges Graph Neural Network Superiority" not in combined, "Old title still present!"
    assert "Author Background and Collaborative Data Governance" not in main_md, "Verbose author background section still present!"
    assert "22.4" not in combined, "Stale Study 1 age 22.4 still present!"
    assert "R^2 = 0.052" not in combined and "2.431" not in combined, "Stale Iowa PD MoCA SVR R^2=0.052 / MAE=2.431 still present!"
    assert "Eyes-closed; MoCA" not in combined and "Eyes-closed recordings from idiopathic PD" not in combined, "Stale Study 7 eye state still present!"
    assert "Study 4a UCSD PD-ON" not in combined, "Stale inverted Study 4a=PD-ON label still present!"
    assert "in three cohorts" not in combined, "Stale Single_Theta 3-cohort claim still present!"
    assert "0.699vs.0.679" not in combined and "0.699 vs. 0.679" not in combined, "Stale Table S20 GCN mean AUC 0.699 vs 0.679 still present!"
    assert "Gershgorin" not in combined, "Stale Gershgorin label still present instead of Eigenvalue-Adaptive Diagonal-Loaded SPD!"
    assert "+0.918" in combined and "0.0159" in combined, "Participant-averaged UCSD PD (15 vs 16, d=+0.918, p=0.0159) must be present!"
    assert "+0.267" in combined, "Study 1 independent-groups equivalent d_IG=+0.267 must be present!"
    assert "Split-Target-HC" in combined, "Split-Target-HC holdout calibration check must be present!"

    # 6. Strict docx text extraction & blank-value remediation audit
    print("   🔍 Auditing extracted docx text for missing values and table completeness...")
    for docx_path, doc_label in [(V2_DOCX, "Main Manuscript"), (SUPP_DOCX, "Supplementary Material")]:
        d_check = docx.Document(docx_path)
        all_para_text = "\n".join(p.text for p in d_check.paragraphs)
        
        # Check for empty parentheses that indicate missing numbers
        empty_parens = re.findall(r"\(\s*\)", all_para_text)
        assert not empty_parens, f"{doc_label} contains {len(empty_parens)} empty parentheses: {empty_parens}"
        
        # Check for empty ranges like ( to )
        empty_ranges = re.findall(r"\(\s*to\s*\)", all_para_text)
        assert not empty_ranges, f"{doc_label} contains {len(empty_ranges)} empty range expressions: {empty_ranges}"
        
        # Check for missing values in model names
        assert "Random Forest ," not in all_para_text, f"{doc_label} contains unpopulated model baseline list!"

    # Specific Main Manuscript extracted content checks
    main_doc = docx.Document(V2_DOCX)
    main_text = "\n".join(p.text for p in main_doc.paragraphs)
    expected_values = [
        "566", "677", "737", "d = -0.826", "d = -0.676", "d = +0.803",
        "0.752", "0.749", "0.743", "0.738", "0.731", "0.754", "0.743",
        "0.590", "0.811", "R² = 0.035", "λ_min ≤ -7.370"
    ]
    for val in expected_values:
        assert val in main_text, f"Main manuscript extracted docx text is missing expected value: {val}"
    print("   ✓ Extracted docx text audit: 0 empty () or ( to ), all 17 core statistics present in plain text")

    # Table completeness check across both documents
    for docx_path, doc_label in [(V2_DOCX, "Main Manuscript"), (SUPP_DOCX, "Supplementary Material")]:
        d_check = docx.Document(docx_path)
        for t_idx, tbl in enumerate(d_check.tables, start=1):
            for r_idx, row in enumerate(tbl.rows[1:], start=2):
                row_texts = [c.text.strip() for c in row.cells]
                assert any(row_texts), f"{doc_label} Table {t_idx} row {r_idx} is completely blank!"
    print("   ✓ Table audit: all tables verified with 0 blank rows")

    print("   ✓ Cross-references, model specs, title alignment, and section structure verified 100%")


def build_all():
    print("=" * 80)
    print("🚀 Step 32: Building Clean Revised Paper 2 Main Manuscript & Supplementary Material")
    print("=" * 80)

    main_md = build_main_manuscript_markdown()
    supp_md = build_supplementary_markdown()

    V2_MD.write_text(main_md, encoding="utf-8")
    SUPP_MD.write_text(supp_md, encoding="utf-8")
    SUPP_MD_COPY.write_text(supp_md, encoding="utf-8")
    print(f"✓ Wrote {V2_MD.name} ({len(main_md.split())} words) and {SUPP_MD.name} ({len(supp_md.split())} words)")

    compile_docx(V2_MD, V2_DOCX)
    compile_docx(SUPP_MD, SUPP_DOCX)
    shutil.copy(SUPP_DOCX, SUPP_DOCX_COPY)

    run_strict_audits(main_md, supp_md)

    if ARTIFACT_DIR.exists():
        for src in (V2_MD, V2_DOCX, SUPP_MD, SUPP_DOCX):
            shutil.copy(src, ARTIFACT_DIR / src.name)
        print(f"✓ Mirrored all 4 deliverables to Artifact Directory: {ARTIFACT_DIR}")

    print("\n🎉 Step 32 complete: All manuscripts compiled and verified with zero errors!")


if __name__ == "__main__":
    build_all()
