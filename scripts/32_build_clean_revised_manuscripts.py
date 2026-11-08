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
ARTIFACT_DIR = Path("/Users/tan/.gemini/antigravity/brain/d5fe5c10-15e9-43a6-af60-4b1baf6f9a8a")

sys.path.insert(0, str(SCRIPTS_DIR))
tbl = importlib.import_module("32a_manuscript_tables")

V2_MD = MANUSCRIPT_DIR / "PAPER_2_MULTI_COHORT_TRANSDIAGNOSTIC_EEG_BENCHMARK_v2.md"
V2_DOCX = MANUSCRIPT_DIR / "PAPER_2_MULTI_COHORT_TRANSDIAGNOSTIC_EEG_BENCHMARK_v2.docx"
SUPP_MD = MANUSCRIPT_DIR / "PAPER_2_SUPPLEMENTARY_MATERIAL.md"
SUPP_DOCX = MANUSCRIPT_DIR / "PAPER_2_SUPPLEMENTARY_MATERIAL.docx"
SUPP_MD_COPY = MANUSCRIPT_DIR / "Supplementary_Material.md"
SUPP_DOCX_COPY = MANUSCRIPT_DIR / "Supplementary_Material.docx"

EXACT_TITLE = (
    "A Gain-Invariant, Leak-Free EEG Framework Delineates Frontal Theta Polarity "
    "and Evaluates Graph Neural Network Utility in Standard Clinical Montages"
)


def build_main_manuscript_markdown() -> str:
    t1 = tbl.build_main_table_1()
    t2 = tbl.build_main_table_2()
    t3 = tbl.build_main_table_3()
    t4 = tbl.build_main_table_4()

    body = f"""# {EXACT_TITLE}

**Chushu Tan**$^{{1,*}}$

$^{{1}}$ School of Sociology, China University of Political Science and Law, Beijing 102249, China  
$^*$ Correspondence: Guangzhou, Guangdong 510006, China; Email: chushuta@gmail.com; ORCID: 0009-0007-3093-0160

---

## Abstract

**Background:** Quantitative electroencephalography (qEEG) provides accessible measurement of human cortical dynamics, yet cross-study comparability is limited by montage differences, uncalibrated amplifier gain scaling, aperiodic $1/f^\\chi$ signal conflation, and validation leakage. Although Graph Neural Networks (GNNs) are increasingly applied to EEG classification, whether graph message passing outperforms regularized classical models on standard 19-channel montages remains unclear.

**Methods:** I developed an open, gain-invariant, leak-free 19-channel EEG benchmarking framework and evaluated it across seven public EEG datasets ($N = 566$ unique participants, $677$ primary sessions; $737$ sessions including secondary eyes-open recordings) spanning experimental sleep deprivation, university depression screening, outpatient major depressive disorder (MDD), Parkinson's disease (PD), and clinical dementia. The pipeline integrates `specparam` periodic–aperiodic decomposition, Phase Lag Index Orthogonal Minimum Spanning Tree (OMST) graphs, and participant-grouped, fold-nested healthy-control normative calibration across 12 model configurations.

**Results:** Across seven heterogeneous datasets, frontal periodic theta was lower in the two depression cohorts ($d = -0.826$ and $-0.676$) and higher in selected Parkinson's disease ($d = +0.865$ to $+1.150$) and dementia ($d = +0.903$ to $+1.279$) comparisons. Graph models (`GCN-OMST` mean $\\text{{AUC}} = 0.738$, range $0.529$–$0.902$) did not consistently outperform classical baselines (`Random Forest` $0.752$, `Tuned Ridge` $0.750$) on 19-channel EEG. Parkinson's cross-site GCN transfer AUCs ranged from $0.590$ to $0.811$ across six directions, with target-site healthy controls used for normative calibration. Cross-sectional cognitive-score prediction was modest, and the dementia MMSE association weakened in the patient-only analysis ($r = +0.246, p = 0.0607$).

**Conclusions:** These findings support a multi-cohort 19-channel benchmarking framework and delineate opposing resting frontal theta directions across affective and neurodegenerative samples, while indicating that prospective clinical utility remains to be established.

**Keywords:** Quantitative EEG, Spectral Parameterization, Gain Invariance, Orthogonal Minimum Spanning Trees, Frontal Theta, Parkinson's Disease, Dementia, Depression, Graph Neural Networks, Multi-Cohort Benchmark.

---

## 1. Introduction

Quantitative electroencephalography (qEEG) measures macro-scale post-synaptic cortical synchronization with millisecond temporal resolution {{ref:Nunez2006,LopesDaSilva2013}}, linking rhythmic neural activity to sleep regulation, cognition, and neuropsychiatric syndromes {{ref:Buzsaki2006,Basar2006,Fries2005,BuzsakiDraguhn2004,Cohen2014}}. Despite growing interest in automated network and machine learning biomarkers {{ref:Stam2014}}, single-cohort EEG classifiers frequently fail to generalize across independent recording sites {{ref:Kapur2012,Billeci2020}}. Practical comparability is hindered by heterogeneous electrode montages (ranging from 19-channel clinical 10–20 layouts to high-density research caps), multiplicative amplifier gain differences across hardware platforms, and optimistic validation bias introduced when repeated sessions cross fold boundaries or when normative standardization precedes data splitting {{ref:Scheinost2019,Varoquaux2018,Wolfers2018,Rutherford2022}}.

At the physiological level, conventional narrowband filtering conflates resonant periodic oscillations with broadband $1/f^\\chi$ aperiodic activity {{ref:Donoghue2020,Voytek2015,ColeVoytek2017}}, an electrophysiological slope associated in computational models with microcircuit excitation/inhibition (E/I) balance and population firing statistics {{ref:Gao2017,Donoghue2022}}. This periodic–aperiodic conflation complicates the interpretation of resting frontal theta activity (4–8 Hz) {{ref:CavanaghFrank2014,Mitchell2008}}. Frontal theta increases during prolonged wakefulness as an index of homeostatic sleep pressure {{ref:DeGennaro2001}}, yet resting-state studies in major depressive disorder (MDD) have reported both elevated frontal theta {{ref:Pizzagalli2002,Tenke2011}} and reduced frontal theta {{ref:Stewart2014,AllenCoan2004}}. Meanwhile, low-frequency slowing is well established in Parkinson's disease (PD) and Alzheimer's disease (AD) {{ref:Stoffers2007,Bosboom2006,Stam2010}}, leaving open how parameterized periodic theta compares across affective, sleep-manipulated, and neurodegenerative datasets processed through an identical pipeline.

A parallel methodological question concerns graph representation learning on low-density scalp montages {{ref:BullmoreSporns2009,RubinovSporns2010}}. Functional connectivity measures that suppress zero-lag volume conduction, such as the Phase Lag Index (PLI) {{ref:Stam2007,Vinck2011}}, paired with parameter-free Orthogonal Minimum Spanning Trees (OMST) {{ref:Dimitriadis2015,Dimitriadis2017}}, avoid arbitrary edge-density thresholds and isolated nodes. Building on such graphs, Graph Convolutional Networks (GCNs) {{ref:KipfWelling2017}} are widely assumed to confer relational inductive bias for EEG decoding. However, GNN studies in clinical and translational EEG rarely benchmark topological message passing against isolated-node graph ablations, regularized linear models, and tree ensembles under identical leak-free cross-validation on 19-channel recordings.

Here, I evaluate a standardized, gain-invariant, leak-free 19-channel EEG benchmarking framework across seven independent public datasets ($N = 566$ unique participants, $677$ primary sessions) spanning experimental acute sleep deprivation {{ref:Zhou2024}}, outpatient MDD {{ref:Mumtaz2017}}, university depression screening {{ref:Cavanagh2021}}, three clinical PD cohorts recorded under dopaminergic manipulation or multi-scale clinical assessment {{ref:George2020,Darcy2021,Patel2023}}, and clinical dementia {{ref:Miltiadous2023}}. Using participant-stratified cross-validation and fold-nested healthy-control calibration, I address three research questions:
1. **Question 1 (Electrophysiological Directionality):** Once broadband $1/f^\\chi$ background activity is separated from narrowband peaks, how does resting frontal periodic theta differ across depression, acute sleep deprivation, Parkinson's disease, and dementia comparisons?
2. **Question 2 (Graph Inductive Bias):** Does a compact `NormativeGCN` operating on PLI-OMST topology outperform graph adjacency ablations and classical tabular classifiers on 19-channel resting EEG?
3. **Question 3 (Cross-Site Transfer and Cross-Sectional Score Prediction):** To what extent do site-calibrated 19-channel features transfer across three independent PD centers and across diagnostic domains, and how accurately do they predict cross-sectional cognitive and motor scale scores?

---

## 2. Methods

### 2.1 Public EEG Datasets, Study Designs, and Participant Accounting
I conducted a secondary computational analysis of seven de-identified, publicly available resting-state EEG datasets retrieved from OpenNeuro and Figshare (repository accessions and software specifications in Supplementary Table S9). Because the archive combines an experimental sleep-deprivation crossover study, a university symptom-screening sample, and five clinical neurological or psychiatric case-control cohorts (two incorporating within-patient dopaminergic ON/OFF sessions), each dataset was analyzed according to its native experimental or clinical design rather than treated as a uniform hospital registry. Across all seven public releases, $N = 574$ unique participants ($685$ primary sessions; $745$ including Study 2 eyes-open recordings) were enrolled. Applying pre-specified automated quality and diagnostic screening rules excluded $8$ participants ($1.4\\%$: $6$ in Study 2 with $>20\\%$ artifact-rejected epochs or channel bridging; $2$ in Study 3 with borderline screening scores $10 \\le \\text{{BDI-II}} \\le 13$; $0$ exclusions in the remaining five datasets), yielding an analyzed sample of $N = 566$ unique participants and $677$ primary sessions ($737$ total sessions including Study 2 eyes-open; participant flow in Supplementary Figure S5; per-cohort demographics, hardware systems, and native sampling rates in Supplementary Table S3):

- **Study 1 — Experimental Acute Sleep Deprivation (`OpenNeuro ds004902`, $N = 71$ participants, $142$ paired sessions)** {{ref:Zhou2024}}: Healthy young adults completed counterbalanced eyes-open resting EEG under normal sleep (NS) and 36 h monitored total sleep deprivation (SD).
- **Study 2 — Outpatient Major Depressive Disorder (`Figshare 4244171`, $N = 58$ primary eyes-closed; $N = 60$ secondary eyes-open)** {{ref:Mumtaz2017}}: DSM-IV outpatient MDD patients ($N = 30$ EC) and healthy controls ($N = 28$ EC) recorded on a native 19-channel 10–20 system.
- **Study 3 — University Depression-Screening Cohort (`OpenNeuro ds003478`, $N = 119$ participants)** {{ref:Cavanagh2021}}: Undergraduate students grouped into a depressive-symptom screening group ($\\text{{BDI-II}} \\ge 14$, $N = 44$) and low-symptom controls ($\\text{{BDI-II}} \\le 7$, $N = 75$) under eyes-open resting EEG. Based on accompanying Structured Clinical Interview for DSM (SCID) records, the 44 $\\text{{BDI-II}} \\ge 14$ participants comprised 11 with current MDD, 11 with past MDD, 8 who did not meet criteria for current or past MDD, and 14 who were not interviewed; this cohort is therefore designated throughout as a university depression-screening sample rather than clinical MDD.
- **Study 4 — UCSD Parkinson's Disease ON/OFF Cohort (`OpenNeuro ds002778`, $N = 31$ participants, $46$ sessions)** {{ref:George2020}}: Idiopathic PD patients ($N = 15$) recorded in both ON-medication and overnight OFF-withdrawal ($\\ge 12\\text{{ h}}$) sessions alongside healthy controls ($N = 16$).
- **Study 5 — AHEPA Clinical Dementia Cohort (`OpenNeuro ds004504`, $N = 88$ participants)** {{ref:Miltiadous2023}}: Eyes-closed 19-channel clinical recordings from patients with Alzheimer's disease (AD, $N = 36$), frontotemporal dementia (FTD, $N = 23$), and cognitively unimpaired elderly controls ($N = 29$), with Mini-Mental State Examination (MMSE) scores.
- **Study 6 — UNM Parkinson's Disease ON/OFF Washout Cohort (`OpenNeuro ds003490`, $N = 50$ participants, $75$ sessions)** {{ref:Darcy2021}}: Idiopathic PD patients ($N = 25$) evaluated in counterbalanced ON-medication and standardized $\\ge 15\\text{{ h}}$ overnight OFF-washout sessions alongside matched controls ($N = 25$).
- **Study 7 — Iowa Parkinson's Multi-Scale Cohort (`OpenNeuro ds004584`, $N = 149$ participants)** {{ref:Patel2023}}: Eyes-closed recordings from idiopathic PD patients ($N = 100$) and elderly controls ($N = 49$) with Montreal Cognitive Assessment (MoCA) and UPDRS Part-III motor ratings.

All primary studies obtained institutional review board approval and written informed consent. Secondary computational analysis of these de-identified public archives required no additional human-subjects intervention and was exempt under 45 CFR 46.104(d)(4) and institutional guidelines.

### 2.2 Signal Preprocessing and Universal 19-Channel Montage Harmonization
All recordings were mapped onto the canonical 19-channel international 10–20 montage:
$$\\mathcal{{C}}_{{19}} = \\{{\\text{{Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, P7, P3, Pz, P4, P8, O1, O2}}\\}} $$
Legacy labels (`T3, T4, T5, T6`) were mapped to modern nomenclature (`T7, T8, P7, P8`; channel mapping in Supplementary Table S5). Where `Pz` served as the native physical recording reference (Study 7, `ds004584`), an initial zero-valued `Pz` channel was appended prior to Common Average Referencing (CAR), algebraically recovering its voltage relative to the 19-channel spatial mean. Signals were zero-phase FIR bandpass filtered (1.0–40.0 Hz, Hamming window), resampled to 250 Hz, CAR re-referenced, and segmented into non-overlapping 4.0 s epochs (1,000 samples). Epochs exceeding 150 $\\mu\\text{{V}}$ peak-to-peak amplitude were rejected; to avoid selective attrition in geriatric tremor recordings, an adaptive fallback retained at least 8 lowest-amplitude epochs (200 $\\mu\\text{{V}}$ threshold in Study 7; preprocessing parameters and per-cohort clean epoch durations in Supplementary Tables S8 and S8b).

### 2.3 Spectral Parameterization and Multiplicative Gain Invariance
Channel power spectral densities (PSDs) were estimated via Welch's method (2.0 s Hamming sub-windows, 50% overlap, $\\Delta f = 0.50\\text{{ Hz}}$) and decomposed in log-log space using `specparam` (v1.1.0; fixed aperiodic mode) {{ref:Donoghue2020}} (single-subject example in Supplementary Figure S1):
$$\\log_{{10}} P(f) = L(f) + \\sum_{{k=1}}^K G_k(f), \\quad L(f) = b - \\chi \\log_{{10}}(f)$$
where $b$ is the broadband offset, $\\chi$ is the aperiodic exponent, and $G_k(f)$ denotes Gaussian oscillatory peaks. Under any positive multiplicative amplifier scaling $s'(t) = \\gamma s(t)$ ($\\gamma > 0$), the PSD scales as $P'(f) = \\gamma^2 P(f)$, shifting $\\log_{{10}} P'(f) = \\log_{{10}} P(f) + 2 \\log_{{10}} \\gamma$. Because $2 \\log_{{10}} \\gamma$ is frequency-independent, it is absorbed entirely into the offset ($b' = b + 2 \\log_{{10}} \\gamma$), leaving both the aperiodic exponent ($\\chi' \\equiv \\chi$) and the flattened periodic residual spectrum ($\\tilde{{P}}'(f) = \\log_{{10}} P'(f) - L'(f) \\equiv \\tilde{{P}}(f)$) algebraically invariant (analytical derivation and non-multiplicative hardware boundary conditions in Supplementary Sections S1.3 and S1.3.1; numerical verification across $\\gamma \\in [10^{{-2}}, 10^2]$ in Supplementary Table S2 and Supplementary Figure S3). Integrating $\\tilde{{P}}(f)$ across canonical bands yielded four features per channel—Periodic Theta (4–8 Hz), Periodic Delta (1–4 Hz), Periodic Alpha (8–13 Hz), and Aperiodic Exponent $\\chi$—forming a node feature matrix $\\mathbf{{X}}_i \\in \\mathbb{{R}}^{{19 \\times 4}}$ per session.

### 2.4 Functional Connectomics via Phase Lag Index and OMST
Pairwise phase synchronization was estimated in the combined theta–alpha band (4–13 Hz) using the Phase Lag Index (PLI) {{ref:Stam2007}}:
$$\\text{{PLI}}_{{jk}} = \\left| \\langle \\operatorname{{sign}}[\\sin(\\Delta \\phi_{{jk}}(t))] \\rangle \\right|$$
which discards zero-lag coupling and is invariant to positive amplitude scaling. Binary undirected graphs were extracted without arbitrary density thresholds via Orthogonal Minimum Spanning Trees (OMST) {{ref:Dimitriadis2017}}, iteratively uniting disjoint Kruskal spanning trees ($18$ edges per tree) to maximize Global Cost Efficiency:
$$\\text{{GCE}}(m) = \\frac{{E_{{\\text{{global}}}}(G_m)}}{{E_{{\\text{{global}}}}(G_{{\\text{{full}}}})}} - \\text{{Cost}}(G_m)$$
Selecting $m^* = \\arg\\max_m \\text{{GCE}}(m)$ guarantees a connected 19-node topology with zero isolated vertices for every session (Supplementary Section S1.2).

### 2.5 Leak-Free Cross-Validation and Fold-Nested Normative Calibration
To standardize features relative to healthy reference variation without data leakage {{ref:Scheinost2019,Wolfers2018}}, channel-wise $z$-scores were computed strictly from healthy control (or rested reference) sessions inside each training fold:
$$Z_{{i,c,k}} = \\frac{{X_{{i,c,k}} - \\hat{{\\mu}}_{{\\text{{HC}},c,k}}^{{\\text{{train}}}}}}{{\\hat{{\\sigma}}_{{\\text{{HC}},c,k}}^{{\\text{{train}}}} + \\epsilon}}$$
Validation and test folds were scaled solely using $(\\hat{{\\mu}}_{{\\text{{HC}}}}^{{\\text{{train}}}}, \\hat{{\\sigma}}_{{\\text{{HC}}}}^{{\\text{{train}}}})$. All within-cohort classification benchmarks used 5-fold cross-validation; in datasets with repeated sessions per participant (Studies 1, 4, and 6c), `StratifiedGroupKFold(n_splits=5)` grouped by participant ID ensured that all sessions from a given individual resided in the same fold. For cross-site and cross-disorder transfer, *site-calibrated transfer* standardized target-cohort sessions using only the target cohort's healthy control distribution, without exposing target patient labels, gradients, or tuning decisions.

### 2.6 Benchmarking Models, Hyperparameters, and Statistical Testing
Twelve classification configurations spanning three tiers were evaluated identically across cohorts (architectural details in Supplementary Section S1.4 and Table S7; classical estimator settings in Supplementary Table S7b; TRIPOD+AI checklist in Supplementary Table S10):
1. **Graph Neural Networks (4 variants):** `NormativeGCN` comprises two spectral graph convolution layers (`4 -> 32 -> 32`, ReLU, dropout $p = 0.25$) with symmetric normalized adjacency $\\mathbf{{\\tilde{{D}}}}^{{-1/2}}(\\mathbf{{A}} + \\mathbf{{I}}_{{19}})\\mathbf{{\\tilde{{D}}}}^{{-1/2}}$, concatenated global mean and max pooling ($64$-D), and a two-layer classification head (`Linear(64, 32) -> ReLU -> Dropout(0.25) -> Linear(32, 2)`), totaling $3,362$ trainable parameters. All GCN variants (`GCN-OMST`, isolated-node `GCN-Identity` [$\\mathbf{{A}} = \\mathbf{{I}}$], random 18-edge `GCN-Random`, and dense `GCN-Full` [$\\mathbf{{A}} = \\mathbf{{1}}$]) were trained for a fixed 50 epochs using PyTorch `Adam` ($\\eta = 0.005$, weight decay $10^{{-4}}$) with zero test-fold early stopping.
2. **Classical Tabular Baselines (6 models on flattened $76$-D node features):** L2 `Tuned Logistic Regression` ($C \\in \\{{0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0\\}}$), `Tuned Ridge Classifier` ($\\alpha \\in \\{{0.01, 0.1, 1.0, 10.0, 50.0, 100.0\\}}$), `Linear SVM` ($C \\in \\{{0.01, 0.1, 1.0, 10.0\\}}$), and `RBF SVM` ($C \\in \\{{0.1, 1.0, 10.0\\}}$, $\\gamma \\in \\{{\\text{{'scale'}}, \\text{{'auto'}}, 0.01, 0.1\\}}$)—each tuned via nested 3-fold `StratifiedKFold` inside the outer training split—plus fixed `Random Forest` (`n_estimators=100`, `max_depth=5`) and `GBDT` (`HistGradientBoostingClassifier`, `max_iter=100`, `max_depth=4`).
3. **Univariate Physiological Baselines (2 models):** L2 logistic regression on 5-channel canonical frontal periodic theta (`Fp1, Fp2, F3, F4, Fz`) alone (`Single θ`) and on 19-channel global mean aperiodic exponent $\\chi$ alone (`Single χ`). Leave-one-feature-out 3-feature `NormativeGCN` ablations and additional single-marker baselines were also evaluated (Supplementary Table S6).

Group biophysical contrasts were quantified via paired Cohen's $d_z$ (paired $t$-test) in Study 1 and Cohen's $d$ with Welch's unequal-variance $t$-tests in Studies 2–7, controlling the false discovery rate via Benjamini–Hochberg adjustment across the 10 primary frontal theta contrasts ($q_{{10}}$; Table 1) and conservatively across all 50 marker-by-cohort tests ($q_{{50}}$; Supplementary Table S4). Out-of-fold ROC-AUCs are reported with 1,000-bootstrap 95% confidence intervals, and head-to-head model rankings were compared via exact two-sided paired Wilcoxon signed-rank tests (Supplementary Table S11b).

### 2.7 Cross-Sectional Clinical Scale Regression
To evaluate cross-sectional prediction of inter-individual symptom scores, four regressors (`Ridge` with $\\alpha = 10$, RBF `SVR`, `Random Forest Regressor` [`max_depth=5`], and `GBDT Regressor` [`max_depth=4`]) were evaluated via 5-fold cross-validation on training-fold-standardized 76-D features for MMSE in Study 5 ($N = 88$), MoCA in Study 7 ($N = 149$), and UPDRS-III motor scores in Study 7 ($N = 100$ PD patients). To test whether cognitive associations reflected diagnostic group separation (e.g., ceiling MMSE $= 30.0$ in healthy controls) versus within-patient score variation, sensitivity regressions were repeated exclusively within dementia patients ($N = 59$) and within PD patients ($N = 100$).

---

## 3. Results

### 3.1 Harmonized 19-Channel Pipeline, Numerical Gain Verification, and OMST Topology
Across all seven datasets and $677$ primary sessions, the harmonization adapter recovered all 19 canonical 10–20 channels, including algebraic reconstruction of `Pz` under Common Average Referencing in Study 7. Numerical gain scaling across four orders of magnitude ($\\gamma \\in [10^{{-2}}, 10^2]$) yielded $R^2 = 1.000000$ and maximum absolute error $< 10^{{-14}}$ for periodic band residuals and aperiodic $\\chi$, while shifting only the unadjusted offset $b$ by $2 \\log_{{10}} \\gamma$ (Supplementary Table S2 and Supplementary Figure S3). Across all $677$ primary PLI-OMST connectomes ($12,863$ node instances), GCE optimization selected $M = 1.52 \\pm 0.50$ orthogonal trees ($48.15\\%$ $M = 1$, $51.85\\%$ $M = 2$), corresponding to $E = 27.33 \\pm 9.00$ edges, mean node degree $\\bar{{k}} = 2.88 \\pm 0.95$ (degree sum $54.66$), edge density $15.98\\% \\pm 5.26\\%$, mean $\\text{{GCE}} = 0.4939 \\pm 0.0438$, and $0.0000\\%$ isolated nodes (Supplementary Section S1.2, Supplementary Table S1, and Supplementary Figure S2).

---

### 3.2 Multi-Cohort Electrophysiological Comparisons and Sensitivity Analyses
Effect sizes across the ten primary group contrasts are presented in Figure 1 and Table 1, with the pooled 50-test statistical table in Supplementary Table S4 and representative cohort spectra in Supplementary Figure S4.

![Figure 1](../reports/Figure_1_Multicohort_Biophysical_Effect_Sizes.png){{width=6.5in}}
**Figure 1. Standardized Electrophysiological Effect Sizes Across Seven Public EEG Datasets.** Forest plots show independent-groups Cohen's $d$ (or paired $d_z$ in Study 1) with 95% confidence intervals across ten contrasts. (A) Frontal Periodic Theta (4–8 Hz) is lower in both depression cohorts and higher across Parkinson's disease and Alzheimer's/combined dementia contrasts. (B) Global Periodic Delta (1–4 Hz). (C) Posterior Periodic Alpha (8–13 Hz) is selectively lower in Alzheimer's disease and frontotemporal dementia. (D) Global Aperiodic Exponent ($\\chi$) is higher (steeper $1/f$ slope) in Alzheimer's and combined dementia and lower (flatter) in outpatient MDD. Square markers indicate uncorrected $p < 0.05$ (\\* $p < 0.05$, \\*\\* $p < 0.01$, \\*\\*\\* $p < 0.001$).

{t1}

Four primary electrophysiological findings emerged:
1. **Opposing Frontal Periodic Theta Directions:** Resting frontal periodic theta was significantly lower than controls in outpatient MDD (Study 2: $d = -0.826$, $p = 0.0027$, $q_{{10}} = 0.0052$, $q_{{50}} = 0.0121$; eyes-open vs. eyes-closed sensitivity in Supplementary Table S13) and in the university depression-screening group (Study 3: $d = -0.676$, $p = 4.96 \\times 10^{{-4}}$, $q_{{10}} = 0.0012$, $q_{{50}} = 0.0035$). Conversely, frontal periodic theta was significantly higher than controls in clinical PD contrasts—including Study 7 Iowa PD ($d = +0.899$, $p = 7.16 \\times 10^{{-8}}$, $q_{{10}} = 7.16 \\times 10^{{-7}}$), Study 6a UNM PD-OFF ($d = +0.865$, $p = 0.0037$, $q_{{10}} = 0.0052$), and Study 4a UCSD PD-ON ($d = +1.150$, $p = 0.0035$, $q_{{10}} = 0.0052$; UCSD PD-OFF $d = +0.559$, $p = 0.1303$, ns)—as well as in Study 5a AD ($d = +1.279$, $p = 1.41 \\times 10^{{-6}}$, $q_{{10}} = 7.05 \\times 10^{{-6}}$) and Study 5c combined dementia ($d = +0.903$, $p = 1.30 \\times 10^{{-5}}$, $q_{{10}} = 4.33 \\times 10^{{-5}}$; FTD alone $d = +0.546$, $p = 0.0720$, ns). These opposing signs are consistent with divergent resting oscillatory regimes across affective and neurodegenerative conditions once broadband $1/f$ tilt is removed.
2. **Posterior Periodic Alpha Attenuation and Aperiodic Steepening in Dementia:** Posterior periodic alpha was substantially lower in AD ($d = -1.059$, $p = 4.24 \\times 10^{{-5}}$, $q_{{50}} = 3.54 \\times 10^{{-4}}$), FTD ($d = -0.949$, $p = 0.0018$, $q_{{50}} = 0.0091$), and combined dementia ($d = -0.981$, $p = 9.31 \\times 10^{{-6}}$, $q_{{50}} = 1.55 \\times 10^{{-4}}$), whereas PD and depression cohorts did not show comparable posterior alpha loss. Concurrently, global $\\chi$ was higher (steeper spectral slope) in AD ($d = +1.096$, $p = 2.18 \\times 10^{{-5}}$, $q_{{50}} = 2.18 \\times 10^{{-4}}$) and lower (flatter slope) in outpatient MDD ($d = -0.744$, $p = 0.0062$, $q_{{50}} = 0.0237$).
3. **Modest Within-Subject Theta Shift Under Acute Sleep Deprivation:** In Study 1 ($N = 71$ pairs), 36 h wakefulness produced a small-to-moderate paired increase in frontal periodic theta ($d_z = +0.284$ [$+0.047, +0.521$], paired $p = 0.0194$; global theta $d_z = +0.568$, $p = 9.05 \\times 10^{{-6}}$). Frontal periodic theta remained statistically significant within the 10-contrast frontal theta family ($q_{{10}} = 0.0243$) but fell just above the threshold under conservative 50-test benchmark-wide FDR control ($q_{{50}} = 0.0540$; Supplementary Table S4), while global $\\chi$ ($d_z = +0.183$, $p = 0.1269$) and posterior alpha ($d_z = +0.096$, $p = 0.4198$) did not differ significantly.
4. **Absence of Group Differences in Scalar OMST Global Cost Efficiency:** Across all ten contrasts, scalar OMST GCE did not differ significantly between case and reference groups (all raw $p \\ge 0.0728$, all $q_{{50}} \\ge 0.1516$), indicating that diagnostic separation in these 19-channel resting recordings is concentrated in regional spectral parameters rather than global binary wiring efficiency.

**Sensitivity Analyses:** Three supplementary checks assessed potential artifacts and confounders. First, *frontopolar channel exclusion* recomputed frontal periodic theta using only non-frontopolar leads (`F3, Fz, F4`), omitting `Fp1` and `Fp2`: effect sizes remained closely aligned in Study 2 MDD ($d = -0.831$, $p = 0.0024$), Study 3 university depression screening ($d = -0.653$, $p = 8.25 \\times 10^{{-4}}$), Study 5c dementia ($d = +0.935$, $p = 6.81 \\times 10^{{-6}}$), Study 6a UNM PD-OFF ($d = +0.922$, $p = 0.0021$), and Study 7 Iowa PD ($d = +0.944$, $p = 1.75 \\times 10^{{-8}}$), while Study 1 sleep deprivation attenuated slightly ($d_z = +0.237$, $p = 0.0501$). This pattern argues against purely frontopolar blink contamination, although broader volume-conducted ocular or muscular contributions cannot be fully excluded without source modeling. Second, *within-subject dopaminergic ON vs. OFF comparisons* differed by clinical protocol (Figure 2B and Supplementary Table S14): in Study 6 UNM ($N = 25$ pairs following counterbalanced $\\ge 15\\text{{ h}}$ overnight washout), no statistically significant ON$-$OFF differences were detected across frontal periodic theta (mean diff $= -0.0056$ [$-0.0308, +0.0195$], $d_z = -0.092$, paired $p = 0.6490$), global delta ($-0.0017$ [$-0.0183, +0.0148$], $p = 0.8299$), posterior alpha ($-0.0276$ [$-0.0691, +0.0139$], $p = 0.1826$), global $\\chi$ ($+0.0111$ [$-0.0474, +0.0696$], $p = 0.6991$), or OMST GCE ($p = 0.9911$); whereas in Study 4 UCSD ($N = 15$ pairs), ON sessions showed higher frontal periodic theta ($+0.1009$ [$+0.0163, +0.1855$], $d_z = +0.660$, $p = 0.0228$) and lower global delta ($-0.0824$ [$-0.1430, -0.0218$], $d_z = -0.753$, $p = 0.0113$). Third, *chronological age ANCOVA* (`Marker ~ Group + Age`) across all five between-subject datasets with participant-level age records (7 independent-sample contrasts, 28 models; Supplementary Table S18) confirmed that every group difference significant unadjusted remained statistically significant after adjusting for age—including frontal periodic theta in Study 3 ($p = 4.63 \\times 10^{{-4}}$), Study 4b UCSD ON ($p = 0.0038$), Study 5 dementia ($p = 2.49 \\times 10^{{-4}}$), Study 6a UNM OFF ($p = 0.0040$), Study 6b UNM ON ($p = 0.0073$), and Study 7 Iowa PD ($p = 3.26 \\times 10^{{-7}}$), as well as Study 5 posterior alpha ($p = 5.49 \\times 10^{{-5}}$) and $\\chi$ ($p = 0.0075$)—while unadjusted non-significant contrasts (such as Study 4a UCSD OFF frontal theta, $p = 0.1367$, and Study 7 posterior alpha, $p = 0.3043$, and $\\chi$, $p = 0.1343$) remained non-significant, and no age term reached $p < 0.05$ across the 28 models (minimum age $p = 0.0609$).

---

### 3.3 Within-Cohort Classification Benchmarks and Graph Ablations
Table 2 summarizes 5-fold cross-validated ROC-AUC across the 9 primary diagnostic contrasts ($9 \\times 12 = 108$ evaluations); the complete $11 \\times 12 = 132$ within-cohort benchmark (adding Study 6b UNM PD-ON and Study 6c UNM All) is reported in Supplementary Table S11, exact paired Wilcoxon signed-rank comparisons in Supplementary Table S11b, single-marker and leave-one-feature-out ablations in Supplementary Table S6, and label-permutation null distributions in Supplementary Figure S6.

{t2}

Across the 9 primary contrasts, `GCN-OMST` achieved a mean ROC-AUC of $0.738$ (range: $0.529$ in single-session sleep deprivation and $0.647$ in university depression screening to $0.842$ in combined dementia and $0.902$ in outpatient MDD). No statistically significant performance advantage of `GCN-OMST` was detected over classical regularized classifiers or graph topology ablations (Supplementary Table S11b). In paired evaluations across the 9 primary contrasts, `GCN-OMST` recorded 4 wins and 5 losses against `Random Forest` (mean $\\text{{AUC}} = 0.752$, Wilcoxon $W = 17.0$, $p = 0.5703$, $q = 0.8970$), 4 wins and 5 losses against `Tuned Ridge` (mean $\\text{{AUC}} = 0.750$, $W = 16.0$, $p = 0.4961$, $q = 0.8970$), and 5 wins and 4 losses against `Tuned Logistic Regression` (mean $\\text{{AUC}} = 0.739$, $W = 21.0$, $p = 0.9102$, $q = 1.0000$). Similarly, no significant differences were detected between `GCN-OMST` and isolated-node `GCN-Identity` ($0.742$, $W = 18.0$, $p = 0.6523$), random-topology `GCN-Random` ($0.754$, $W = 12.0$, $p = 0.2500$), or dense `GCN-Full` ($0.739$, $W = 22.0$, $p = 1.0000$; Figure 3B). Only `RBF SVM` ($0.660$, raw $p = 0.0117$, $q = 0.0645$) and univariate `Single χ` ($0.567$, raw $p = 0.0078$, $q = 0.0645$) yielded lower uncorrected AUCs, neither surviving FDR correction across the 11 pairwise comparisons.

In three cohorts—Study 3 university depression screening (`Single θ` $\\text{{AUC}} = 0.683$ vs. `GCN-OMST` $0.647$), Study 4 UCSD PD ($0.738$ vs. $0.725$), and Study 7 Iowa PD ($0.739$ vs. $0.685$)—univariate 5-channel frontal periodic theta alone exceeded the 3,362-parameter `GCN-OMST` model. Leave-one-feature-out ablations of `NormativeGCN` (Supplementary Table S6) corroborated this pattern: omitting periodic theta (`Ablate θ`, retaining $[\\delta, \\alpha, \\chi]$) reduced `GCN-OMST` AUC from $0.902$ to $0.727$ in Study 2 MDD, $0.647$ to $0.581$ in Study 3, $0.725$ to $0.508$ in Study 4 UCSD PD, $0.808$ to $0.643$ in Study 5b FTD, $0.842$ to $0.725$ in Study 5c combined dementia, $0.699$ to $0.622$ in Study 6a UNM PD-OFF, and $0.685$ to $0.550$ in Study 7 Iowa PD (with minimal change in Study 5a AD, $0.807 \\to 0.798$, where posterior alpha and $\\chi$ also carry strong signal, and Study 1 sleep deprivation, $0.529 \\to 0.525$). Conversely, omitting periodic alpha (`Ablate α`, retaining $[\\theta, \\delta, \\chi]$) slightly increased AUC in depression ($0.902 \\to 0.927$ in Study 2; $0.647 \\to 0.704$ in Study 3) while lowering AUC in dementia ($0.808 \\to 0.685$ in FTD; $0.842 \\to 0.808$ in combined dementia).

---

### 3.4 Cross-Center Parkinson's Transfer and Cross-Disorder Generalization
Site-calibrated external transfer results—standardized using target-site healthy controls without model retraining—are summarized in Table 3, Figure 2A, and Figure 3A (complete tri-center PD transfer in Supplementary Table S16; 12-pair 4-cohort transfer matrix in Supplementary Table S12; recording-state sensitivity in Supplementary Table S13; 14-pair cross-disorder matrix in Supplementary Table S17).

![Figure 2](../reports/Figure_2_Tri_Center_Parkinsons_Generalization.png){{width=6.5in}}
**Figure 2. Tri-Center Parkinson's Cross-Site Transfer and Within-Patient Dopaminergic Washout Comparison.** (A) `GCN-OMST` ROC-AUC across Iowa ($N = 149$), UNM OFF ($N = 50$), and UCSD ($N = 46$); diagonal cells show 5-fold cross-validated AUC and off-diagonal cells show site-calibrated cross-center transfer without retraining. (B) Within-participant paired differences ($\\text{{ON}} - \\text{{OFF}}$ with 95% confidence intervals) across $N = 25$ UNM PD patients following $\\ge 15\\text{{ h}}$ overnight washout; no statistically significant differences were detected across extracted markers (all paired $p \\ge 0.16$).

{t3}

![Figure 3](../reports/Figure_3_Transdiagnostic_Transfer_and_Ablations.png){{width=6.5in}}
**Figure 3. Cross-Disorder Transfer Matrix and Graph Topology Ablations.** (A) Four-domain `GCN-OMST` ROC-AUC matrix (diagonal: 5-fold CV; off-diagonal: site-calibrated transfer). Positive transfer occurs among Parkinson's, dementia, and sleep-deprivation slowing regimes, whereas models trained on slowing cohorts and evaluated on outpatient MDD fall below chance ($0.274$–$0.336$). (B) Comparison of `GCN-OMST`, isolated-node `GCN-Identity` ($\\mathbf{{A}} = \\mathbf{{I}}$), and dense `GCN-Full` ($\\mathbf{{A}} = \\mathbf{{1}}$) across five cohorts.

Two principal external-transfer patterns were observed:
1. **Directional Variability in Tri-Center Parkinson's Transfer:** Across the six directed PD transfer pairs (Table 3 and Supplementary Table S16), `GCN-OMST` ROC-AUC ranged from $0.590$ to $0.811$. Training on the largest cohort (Study 7 Iowa, $N = 149$) produced moderate-to-good discrimination at UNM ($\\text{{AUC}} = 0.811$ [$0.670, 0.935$]) and UCSD (`GCN-OMST` $0.740$ [$0.583, 0.883$]; `Random Forest` $0.804$). Transfers originating from smaller discovery cohorts were more modest, ranging from $0.686$–$0.694$ (`UCSD -> Iowa`, `UNM -> Iowa`) down to near-chance levels between the two smaller samples (`UCSD -> UNM`: `GCN-OMST` $0.621$, `Tuned Logistic` $0.698$; `UNM -> UCSD`: `GCN-OMST` $0.590$, `Random Forest` $0.616$). Across the six PD pairs, classical models and `GCN-Identity` frequently matched or exceeded `GCN-OMST`, indicating that cross-center generalization derives primarily from gain-invariant spectral parameterization and target-site healthy-control calibration rather than OMST message passing.
2. **Domain Alignment and Directional Asymmetry Across Disorders:** Within the affective spectrum, training on university depression screening transferred to outpatient MDD (`GCN-OMST` $\\text{{AUC}} = 0.833$ [$0.708, 0.936$]), whereas the reverse eye-state-mismatched transfer (`Outpatient MDD EC -> Univ. Depression EO`) was near chance for `GCN-OMST` ($0.463$; `Random Forest` $0.677$) and improved modestly when eye state was matched (`Eyes-Open -> Eyes-Open` `GCN-OMST` $0.569$, `Logistic` $0.592$; Supplementary Table S13). Across slowing conditions, positive transfer occurred for `Iowa PD -> Dementia` ($0.792$; `GCN-Identity` $0.835$), `Sleep Deprivation -> Dementia` ($0.840$), and `Sleep Deprivation -> Iowa PD` ($0.732$), with more modest reverse AUCs ($0.563$–$0.647$; Supplementary Table S17). Cross-disorder transfers involving outpatient MDD exhibited clear directional asymmetry: classifiers trained on `Iowa PD`, `Dementia`, or `Sleep Deprivation` and evaluated on `Outpatient MDD` yielded sub-chance ROC-AUCs ($0.325$, $0.336$, and $0.274$, respectively), as did `Outpatient MDD -> Dementia` ($0.338$), consistent with opposing frontal periodic theta polarity. By contrast, `Outpatient MDD -> Iowa PD` (`GCN-OMST` $0.539$, though `Logistic`/`RF`/`GBDT` were $0.325$–$0.330$) and `Outpatient MDD -> Sleep Deprivation` ($0.490$) hovered near chance because multivariate MDD weights combine frontal theta with aperiodic $\\chi$ differences that are much weaker in PD and sleep loss.

---

### 3.5 Cross-Sectional Clinical Scale Prediction
Out-of-fold 5-fold cross-validation results for continuous clinical scales are reported in Table 4 and Figure 4 (full 20-row regressor breakdown in Supplementary Table S15).

![Figure 4](../reports/Figure_4_Clinical_Scale_Continuous_Prediction.png){{width=6.5in}}
**Figure 4. Cross-Sectional Clinical Scale Associations and Out-of-Fold Regression.** (A) Dementia cohort MMSE ($N = 88$) vs. posterior periodic alpha ($r = +0.374$, $p = 3.36 \\times 10^{{-4}}$; 5-fold `GBDT` out-of-fold $r = +0.472$). (B) Dementia MMSE vs. global periodic theta ($r = -0.373$, $p = 3.43 \\times 10^{{-4}}$). (C) Iowa PD cohort MoCA ($N = 149$) vs. global periodic theta ($r = -0.343$, $p = 1.87 \\times 10^{{-5}}$; 5-fold `SVR` $r = +0.339$). (D) Patient-only sensitivity analysis in Iowa PD ($N = 100$ PD patients excluding controls): out-of-fold `SVR` predictions vs. observed MoCA ($r = +0.334$, $p = 6.74 \\times 10^{{-4}}$, $R^2 = 0.052$).

{t4}

In Study 5 ($N = 88$), full-cohort out-of-fold predictions correlated moderately with MMSE (`GBDT`: $r = +0.472$, $p = 3.48 \\times 10^{{-6}}$, $R^2 = 0.209$, $\\text{{MAE}} = 4.071$). However, because all 29 healthy controls had ceiling scores ($\\text{{MMSE}} = 30.0$), restricting regression strictly to the $N = 59$ dementia patients attenuated performance substantially (`GBDT`: $r = +0.246$, $p = 0.0607$, $R^2 = 0.030$, $\\text{{MAE}} = 3.172$; linear and SVR models $p \\ge 0.104$), indicating that full-cohort MMSE predictability was largely driven by case–control separation. In Study 7, out-of-fold `SVR` predictions correlated modestly with MoCA both across the full sample ($N = 149$: $r = +0.339$, $p = 2.32 \\times 10^{{-5}}$, $R^2 = 0.060$, $\\text{{MAE}} = 2.397$) and within PD patients alone ($N = 100$: $r = +0.334$, $p = 6.74 \\times 10^{{-4}}$, $R^2 = 0.052$, $\\text{{MAE}} = 2.431$), showing that resting spectral features capture a statistically significant but small fraction ($\\sim 5\\%$) of inter-individual cognitive score variance among PD patients. By contrast, UPDRS-III motor severity among the $N = 100$ PD patients was not significantly predicted by any resting-state regressor (best `GBDT` $r = +0.165$, $p = 0.1008$, $R^2 = -0.105$).

---

## 4. Discussion

### 4.1 Summary of Findings Relative to Research Questions
Evaluating seven heterogeneous public EEG datasets ($N = 566$ participants, $677$ primary sessions) under a single gain-invariant, leak-free 19-channel pipeline yielded three main conclusions. First, separating periodic peaks from broadband $1/f^\\chi$ activity {{ref:Donoghue2020}} revealed opposing group-level directions in resting frontal periodic theta: lower values in outpatient MDD and university depression screening, contrasted with higher values across selected PD and dementia comparisons. Second, `NormativeGCN` (`GCN-OMST` ROC-AUC $0.529$–$0.902$, mean $0.738$) did not systematically outperform isolated-node graph ablations or regularized classical classifiers (`Random Forest` $0.752$, `Tuned Ridge` $0.750$) on 19-channel montages. Third, site-calibrated cross-center PD transfer varied by training cohort size and direction (`GCN-OMST` AUC $0.590$–$0.811$), while cross-sectional cognitive score prediction was modest and attenuated in dementia once healthy controls were excluded.

### 4.2 Interpretation of Frontal Periodic Theta and Aperiodic Slope Differences
Prior resting-state EEG studies of depression and neurological slowing often relied on total narrowband power, which blends narrowband oscillations with broadband $1/f$ offset and slope shifts {{ref:Donoghue2020,Donoghue2022}}. After subtracting the parameterized aperiodic floor, lower frontal periodic theta in both the outpatient MDD sample and the university depressive-symptom group is consistent with reduced resting frontomedial oscillatory engagement {{ref:CavanaghFrank2014,Stewart2014}}, whereas elevated periodic theta in PD and AD aligns with thalamocortical and corticocortical slowing {{ref:Stoffers2007,Bosboom2006,Stam2010}}. In AD, concurrent posterior periodic alpha loss and steeper aperiodic $\\chi$ diverge from normative adult aging—where $\\chi$ typically flattens with age {{ref:Voytek2015}}—and may reflect synaptic loss, cholinergic deafferentation, or altered cortical E/I tone {{ref:Gao2017,Stam2010}}.

Several alternative, non-mechanistic explanations warrant caution. Resting EEG phenotypes in public archives can be influenced by unmeasured differences in vigilance, subclinical drowsiness, chronic antidepressant or anticholinergic exposure, skull conductivity, and eye-closure state. Moreover, in Study 3 only half of the $\\text{{BDI-II}} \\ge 14$ university group had SCID-confirmed current or past MDD, meaning that frontal theta reduction in that dataset tracks self-reported depressive symptom burden rather than a uniform clinical diagnosis. Likewise, the difference between Study 6 UNM (where no statistically significant ON$-$OFF differences were detected after $\\ge 15\\text{{ h}}$ washout) and Study 4 UCSD (where ON sessions exhibited higher periodic theta and lower delta) may reflect protocol differences in washout duration, disease stage, or sample size ($N = 25$ vs. $N = 15$ pairs); absence of statistical significance in UNM should not be interpreted as formal pharmacological equivalence without pre-specified equivalence bounds.

### 4.3 Why Graph Message Passing Showed No Advantage on 19-Channel Resting EEG
Across 132 within-cohort evaluations, `GCN-OMST` showed no statistically significant gain over `GCN-Identity` ($\\mathbf{{A}} = \\mathbf{{I}}$), `GCN-Full`, `Tuned Ridge`, or `Random Forest` (Supplementary Table S11b), and univariate frontal theta alone matched or exceeded `GCN-OMST` in three cohorts. Three structural factors plausibly explain this ceiling. First, a standard 10–20 montage has only 19 coarse scalp nodes, providing limited spatial degrees of freedom for relational neighborhood aggregation {{ref:KipfWelling2017}}. Second, although PLI-OMST eliminates zero-lag coupling and guarantees zero isolated nodes without arbitrary thresholds {{ref:Stam2007,Dimitriadis2017}}, sensor-level phase estimates over short resting recordings remain noisy, and scalar OMST GCE showed no significant case–control differences in any dataset. Third, leave-one-feature-out ablations demonstrated that diagnostic variance in these resting cohorts is concentrated in regional node attributes—chiefly frontal periodic theta and, in dementia, posterior alpha and $\\chi$—which linear and tree-based learners access directly from the 76-D feature table. Importantly, this null architectural result applies specifically to static spectral features on low-density 19-channel resting EEG; GNNs may still provide value in high-density (64–256 channel) arrays, source-reconstructed cortical parcellations, or task-evoked dynamic connectivity.

### 4.4 Cross-Site Calibration Boundaries and Modest Cross-Sectional Score Associations
In tri-center PD transfer, calibrating target-site features solely against local healthy controls enabled models trained on the largest cohort (Iowa, $N = 149$) to achieve AUCs of $0.811$ on UNM and $0.740$–$0.804$ on UCSD without retraining, whereas transfers trained on the smaller UCSD ($N = 46$) and UNM ($N = 50$) samples degraded toward $0.590$–$0.698$. Thus, standardized 19-channel parameterization and healthy-control calibration mitigate multiplicative gain differences and partially bridge site shifts when discovery sample size is sufficient, but they do not eliminate cross-hardware domain shift when training cohorts are small. Similarly, cross-disorder transfer revealed asymmetric boundaries: models trained on PD, dementia, or sleep deprivation yielded sub-chance AUCs ($0.274$–$0.336$) on outpatient MDD due to opposing theta weights, whereas reverse transfers from outpatient MDD to Iowa PD ($0.539$) and sleep deprivation ($0.490$) landed near chance because multivariate MDD classifiers also weight aperiodic slope shifts. Finally, cross-sectional regression results underscore the gap between group discrimination and individual severity estimation: patient-only dementia MMSE prediction fell below conventional significance ($r = +0.246$, $p = 0.0607$), Iowa PD patient-only MoCA predictions explained only $R^2 = 0.052$ of variance, and UPDRS-III motor scores were not predicted ($p = 0.1008$).

### 4.5 Limitations and Future Directions
Key limitations constrain clinical extrapolation: (1) reducing 32- and 64-channel recordings to 19 channels sacrifices spatial resolution to prioritize compatibility with standard clinical archives; (2) scalp CAR and PLI reduce but do not eliminate volume conduction or residual electromyographic/ocular artifacts, and Study 2 lacked participant-level age records for ANCOVA; (3) target-site normative calibration requires a local reference sample of healthy controls matched in recording protocol; and (4) all clinical scale analyses were cross-sectional rather than longitudinal. Prospective, multi-wave cohorts with standardized medication documentation and source-space benchmarking will be required to assess longitudinal prognostic or treatment-monitoring utility.

---

## 5. Conclusions

Across seven heterogeneous public EEG datasets ($N = 566$ participants, $677$ primary sessions), a gain-invariant, leak-free 19-channel pipeline delineated opposing resting frontal periodic theta directions between depression cohorts and neurodegenerative comparisons, while showing that `NormativeGCN` models did not consistently outperform classical regularized baselines on low-density montages. Site-calibrated PD transfer varied substantially with training sample size and transfer direction, and cross-sectional cognitive score prediction remained modest. Together, these results establish an open methodological baseline for multi-cohort qEEG evaluation and underscore the need for rigorous classical baselines and patient-only sensitivity checks before claiming clinical readiness.

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
"""
    return tbl.compile_sequential_citations(body)


def build_supplementary_markdown() -> str:
    ts1 = tbl.build_supp_table_s1()
    ts4 = tbl.build_supp_table_s4()
    ts6 = tbl.build_supp_table_s6()
    ts7_7b = tbl.build_supp_table_s7_and_s7b()
    ts11_11b = tbl.build_supp_table_s11_and_s11b()
    ts14 = tbl.build_supp_table_s14()
    ts15 = tbl.build_supp_table_s15()
    ts16_17 = tbl.build_supp_table_s16_and_s17()
    ts18 = tbl.build_supp_table_s18()

    supp = f"""# Supplementary Material

**{EXACT_TITLE}**

**Chushu Tan**$^{{1,*}}$

$^{{1}}$ School of Sociology, China University of Political Science and Law, Beijing 102249, China  
$^*$ Corresponding author: Chushu Tan, Guangzhou, Guangdong 510006, China; Email: chushuta@gmail.com; ORCID: 0009-0007-3093-0160

---

## Section S1. Extended Supplementary Methods

### S1.1 Parameterized Spectral Decomposition via `specparam`
To separate narrowband neural oscillations from broadband $1/f^\\chi$ background activity, resting-state EEG channel power spectra were decomposed using `specparam` (v1.1.0; Donoghue et al., 2020). Continuous 250 Hz signals were segmented into non-overlapping 4.0 s epochs. Power spectral density (PSD) was estimated via Welch's modified periodogram using 2.0 s Hamming sub-windows with 50% overlap, providing a frequency resolution of $\\Delta f = 1 / 2.0\\text{{ s}} = 0.50\\text{{ Hz}}$ across 1.0–40.0 Hz. Spectra were modeled in fixed aperiodic mode as:

$$\\log_{{10}} \\mathrm{{PSD}}(f) = L(f) + \\sum_{{n=1}}^N a_n \\exp\\left(-\\frac{{(f - c_n)^2}}{{2 w_n^2}}\\right)$$

where the broadband aperiodic component $L(f)$ is linear in log-log coordinates:

$$L(f) = b - \\chi \\log_{{10}}(f)$$

Here $b$ denotes the broadband offset (in $\\log_{{10}} \\mu\\text{{V}}^2/\\text{{Hz}}$, or Bell units, $\\text{{B}}$), $\\chi$ denotes the dimensionless aperiodic exponent, and $(c_n, a_n, 2w_n)$ parameterize Gaussian peak center frequencies, log-power amplitudes above $L(f)$, and bandwidths. Band-specific periodic residual power over $[f_1, f_2]$ was computed from the flattened residual spectrum:

$$P_{{\\mathrm{{periodic}}, [f_1, f_2]}} = \\frac{{1}}{{f_2 - f_1}} \\int_{{f_1}}^{{f_2}} \\left[\\log_{{10}} \\mathrm{{PSD}}(f) - L(f)\\right] df$$

evaluated across Periodic Theta (4.0–8.0 Hz, `x[:, 0]`), Periodic Delta (1.0–4.0 Hz, `x[:, 1]`), and Periodic Alpha (8.0–13.0 Hz, `x[:, 2]`), alongside channel-level $\\chi$ (`x[:, 3]`). Across cohorts, `specparam` fits achieved $R^2 > 0.985$ and mean absolute error $< 0.045\\text{{ B}}$.

### S1.2 Data-Driven Connectomics via Orthogonal Minimum Spanning Trees (OMST)
Pairwise functional phase coupling was quantified using the Phase Lag Index (PLI; Stam et al., 2007) in the 4.0–13.0 Hz band to discard zero-lag volume-conducted synchronization. Edge distances were defined as:

$$c_{{ij}} = (1.0 - \\mathrm{{PLI}}_{{ij}}) + 10^{{-4}}$$

where the $10^{{-4}}$ offset prevents zero-distance degeneracies in sparse graph routines. Starting from the primary Kruskal minimum spanning tree ($N - 1 = 18$ edges connecting all $N = 19$ channels), edge-disjoint orthogonal spanning trees were extracted iteratively and cumulative subgraphs $G_m$ were scored by Global Cost Efficiency (Dimitriadis et al., 2017):

$$\\mathrm{{GCE}}(m) = \\frac{{E_{{\\mathrm{{global}}}}(G_m)}}{{E_{{\\mathrm{{global}}}}(G_{{\\mathrm{{full}}}})}} - \\mathrm{{Cost}}(G_m)$$

where $E_{{\\mathrm{{global}}}}$ is average inverse shortest-path length and $\\mathrm{{Cost}}(G_m)$ is normalized wiring weight.

Across all $N = 566$ analyzed participants and $677$ primary recording sessions, optimal tree counts averaged $M = 1.52 \\pm 0.50$ ($48.15\\%$ $M = 1$, $51.85\\%$ $M = 2$; Study 1 Sleep Deprivation: $1.59 \\pm 0.49$, density $16.75\\% \\pm 5.19\\%$; Study 2 Outpatient MDD: $1.33 \\pm 0.47$, density $13.97\\% \\pm 4.98\\%$; Study 3 Univ. Depression Screening: $1.45 \\pm 0.50$, density $15.30\\% \\pm 5.26\\%$; Study 4 UCSD PD: $1.61 \\pm 0.49$, density $16.93\\% \\pm 5.19\\%$; Study 5 Dementia: $1.53 \\pm 0.50$, density $16.15\\% \\pm 5.28\\%$; Study 6 UNM PD: $1.63 \\pm 0.49$, density $17.12\\% \\pm 5.13\\%$; Study 7 Iowa PD: $1.48 \\pm 0.50$, density $15.61\\% \\pm 5.28\\%$). Across all $677$ graphs, selected edge counts averaged $E = 27.33 \\pm 9.00$, corresponding to mean node degree $\\bar{{k}} = 2.88 \\pm 0.95$ (degree sum $54.66$), edge density $15.98\\% \\pm 5.26\\%$, mean $\\mathrm{{GCE}} = 0.4939 \\pm 0.0438$, and an isolated node rate of $0.0000\\%$ across all $12,863$ node instances ($677 \\times 19$).

### S1.3 Analytical Proof of Multiplicative Amplifier Gain Invariance
Let raw EEG voltage undergo positive multiplicative scaling $v'(t) = k \\cdot v(t)$ ($k > 0$). Its Welch PSD scales quadratically:

$$\\mathrm{{PSD}}'(f) = k^2 \\cdot \\mathrm{{PSD}}(f)$$

producing a uniform additive constant in base-10 logarithmic power:

$$\\log_{{10}} \\mathrm{{PSD}}'(f) = \\log_{{10}} \\mathrm{{PSD}}(f) + 2 \\log_{{10}} k$$

Because $\\partial (2 \\log_{{10}} k) / \\partial f = 0$, fitting `specparam` to $\\log_{{10}} \\mathrm{{PSD}}'(f)$ yields:
1. $\\chi' = \\chi$ (aperiodic slope invariance);
2. $c'_n = c_n$, $w'_n = w_n$, and $a'_n = a_n$ (peak parameter invariance);
3. $b' = b + 2 \\log_{{10}} k$ (the unadjusted offset absorbs the entire multiplicative scale factor);
4. $\\log_{{10}} \\mathrm{{PSD}}'(f) - L'(f) = \\log_{{10}} \\mathrm{{PSD}}(f) - L(f)$ (residual periodic spectrum invariance).

Similarly, because positive amplitude scaling preserves instantaneous analytic signal phase ($\\angle(k \\cdot v(t)) = \\angle v(t)$), pairwise PLI matrices and resulting OMST topologies are invariant to $k > 0$.

### S1.3.1 Boundary Conditions: Multiplicative Gain vs. Non-Multiplicative Hardware Shifts
Whereas multiplicative digitizer gain ($v'(t) = k \\cdot v(t)$) is cancelled analytically by `specparam` and PLI, real-world multi-center EEG archives also differ in non-multiplicative hardware characteristics: (1) analog high-pass and anti-aliasing filter rolloffs; (2) sensor impedance and thermal noise floors; (3) native physical reference geometry prior to CAR; and (4) environmental shielding and eye-state instructions. To address center-specific baseline offsets and demographic shifts without test-set leakage, multiplicative gain invariance is paired with **fold-nested normative standardization against training-fold (or target-site) healthy controls** (Main Text Section 2.5).

### S1.4 `NormativeGCN` Architecture and Optimization Specification
`NormativeGCN` is implemented in PyTorch (`src/harmonizer_and_graph.py`) and operates on 19-node binary adjacency matrices $\\mathbf{{A}} \\in \\{{0, 1\\}}^{{19 \\times 19}}$ with 4-dimensional normatively calibrated node features $\\mathbf{{z}}_i = [z_\\theta, z_\\delta, z_\\alpha, z_\\chi]^T \\in \\mathbb{{R}}^4$:
- **Layer 1:** `PurePyTorchGCNLayer(4 -> 32)` with symmetric normalized adjacency $\\mathbf{{\\tilde{{A}}}} = \\mathbf{{\\tilde{{D}}}}^{{-1/2}}(\\mathbf{{A}} + \\mathbf{{I}}_{{19}})\\mathbf{{\\tilde{{D}}}}^{{-1/2}}$, ReLU activation, and Dropout ($p = 0.25$) ($4 \\times 32 + 32 = 160$ parameters).
- **Layer 2:** `PurePyTorchGCNLayer(32 -> 32)` with ReLU activation and Dropout ($p = 0.25$) ($32 \\times 32 + 32 = 1,056$ parameters).
- **Readout:** Concatenated global mean pooling ($32$-D) and global max pooling ($32$-D) across the 19 nodes, yielding $\\mathbf{{h}}_G = [\\mathbf{{h}}_{{\\mathrm{{mean}}}} \\,\\|\\, \\mathbf{{h}}_{{\\mathrm{{max}}}}] \\in \\mathbb{{R}}^{{64}}$.
- **Classification Head:** `Linear(64 -> 32)` ($2,080$ parameters) $\\to$ ReLU $\\to$ Dropout ($p = 0.25$) $\\to$ `Linear(32 -> 2)` ($66$ parameters).
- **Parameter Count & Training:** Exactly $3,362$ trainable parameters ($160 + 1,056 + 2,080 + 66$), trained via `torch.optim.Adam` (learning rate $\\eta = 0.005$, weight decay $\\lambda = 1.0 \\times 10^{{-4}}$, cross-entropy loss) for a fixed $50$ epochs per fold with zero evaluation-fold early stopping (`scripts/17_run_comprehensive_7cohort_benchmark.py`).

---

## Section S2. Supplementary Tables

{ts1}

---

### Table S2. Numerical Verification of Feature Invariance Across Four Orders of Magnitude of Amplifier Gain
| Gain Factor ($k$) | Theoretical Offset $b_{{\\mathrm{{theory}}}}$ (B) | Extracted Offset $b_{{\\mathrm{{sim}}}}$ (B) | Offset Error $\\Delta b$ (B) | Periodic $\\theta$ Error (B) | Periodic $\\delta$ Error (B) | Periodic $\\alpha$ Error (B) | Aperiodic $\\chi$ Error |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.01 | $-3.3500$ | $-3.3500$ | $< 1.0 \\times 10^{{-15}}$ | $2.1 \\times 10^{{-15}}$ | $1.8 \\times 10^{{-15}}$ | $2.5 \\times 10^{{-15}}$ | $1.2 \\times 10^{{-15}}$ |
| 0.10 | $-1.3500$ | $-1.3500$ | $< 1.0 \\times 10^{{-15}}$ | $1.4 \\times 10^{{-15}}$ | $1.1 \\times 10^{{-15}}$ | $1.6 \\times 10^{{-15}}$ | $8.0 \\times 10^{{-16}}$ |
| 0.50 | $+0.0479$ | $+0.0479$ | $< 1.0 \\times 10^{{-15}}$ | $8.0 \\times 10^{{-16}}$ | $5.0 \\times 10^{{-16}}$ | $9.0 \\times 10^{{-16}}$ | $4.0 \\times 10^{{-16}}$ |
| 1.00 (Base) | $+0.6500$ | $+0.6500$ | $0.00$ | $0.00$ | $0.00$ | $0.00$ | $0.00$ |
| 2.00 | $+1.2521$ | $+1.2521$ | $< 1.0 \\times 10^{{-15}}$ | $9.5 \\times 10^{{-16}}$ | $7.0 \\times 10^{{-16}}$ | $1.1 \\times 10^{{-15}}$ | $5.0 \\times 10^{{-16}}$ |
| 10.00 | $+2.6500$ | $+2.6500$ | $< 1.0 \\times 10^{{-15}}$ | $1.8 \\times 10^{{-15}}$ | $1.5 \\times 10^{{-15}}$ | $2.0 \\times 10^{{-15}}$ | $1.0 \\times 10^{{-15}}$ |
| 100.00 | $+4.6500$ | $+4.6500$ | $< 1.0 \\times 10^{{-15}}$ | $2.7 \\times 10^{{-15}}$ | $2.3 \\times 10^{{-15}}$ | $3.1 \\times 10^{{-15}}$ | $1.6 \\times 10^{{-15}}$ |

*Note: Evaluated on 19-channel EEG epochs scaled by multiplicative factor $k \\in [0.01, 100]$ ($b_0 = 0.650\\text{{ B}}$, $\\chi_0 = 1.152$). Theoretical offset shift $\\Delta b = 2 \\log_{{10}}(k)$ is confirmed, while periodic residuals and $\\chi$ exhibit deviations bounded by IEEE-754 double-precision machine epsilon ($< 10^{{-14}}$).*

---

### Table S3. Multi-Cohort Study Designs, Demographics, Hardware Systems, and Clinical Scales ($N = 566$ Unique Participants, $677$ Primary Sessions)
| Study ID | Dataset & Setting | Analyzed $N$ (Participants / Primary Sessions) | Group Breakdown | Age in Years (Mean $\\pm$ SD) | Female (%) | Native Hardware & Channel Count | Sampling Rate (Native $\\to$ Harmonized) | Protocol / Clinical Assessments |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- | :---: | :--- |
| **Study 1** | Acute Sleep Deprivation (`ds004902`, Experimental Crossover) | 71 / 142 | SD (71) vs. NS (71) [Paired] | 22.4 ± 2.6 | 47.9% (34/71) | BrainVision actiCHamp (64 Ch) | 500 Hz $\\to$ 250 Hz | Monitored 36 h wakefulness vs. normal sleep (EO) |
| **Study 2** | Outpatient Clinical MDD (`Figshare 4244171`, Psychiatric Clinic) | 58 / 58 (+60 EO) | MDD (30) vs. HC (28) [EC] | MDD 40.3 ± 12.9; HC 38.2 ± 15.6 | 37.9% (22/58) | Nihon Kohden (19 Ch native 10–20) | 256 Hz $\\to$ 250 Hz | DSM-IV MDD; BDI (24.8 ± 8.6 vs. 3.9 ± 3.4) |
| **Study 3** | Univ. Depression Screening (`ds003478`, Student Screening) | 119 / 119 | BDI-II $\\ge 14$ (44) vs. $\\le 7$ (75) | 19.4 ± 2.0 ($N=118$ w/ age) | 60.5% (72/119) | BioSemi ActiveTwo (64 Ch) | 500 Hz $\\to$ 250 Hz | BDI-II screening; partial SCID (11 curr., 11 past MDD, 8 non-MDD, 14 uninterviewed) |
| **Study 4** | UCSD Parkinson's (`ds002778`, Neurology Clinic) | 31 / 46 | PD (15 ON + 15 OFF) vs. HC (16) | PD 63.2 ± 8.1; HC 63.5 ± 9.6 | 54.8% (17/31) | BioSemi ActiveTwo (32 Ch) | 512 Hz $\\to$ 250 Hz | Counterbalanced ON vs. $\\ge 12\\text{{ h}}$ OFF withdrawal |
| **Study 5** | AHEPA Clinical Dementia (`ds004504`, Hospital Neurology) | 88 / 88 | AD (36), FTD (23), HC (29) | AD 72.8 ± 7.9; FTD 68.2 ± 7.3; HC 70.1 ± 6.8 | 55.7% (49/88) | Nihon Kohden (19 Ch native 10–20) | 500 Hz $\\to$ 250 Hz | Eyes-closed resting; MMSE (AD 17.8 ± 4.5, FTD 22.2 ± 4.8, HC 30.0 ± 0.0) |
| **Study 6** | UNM Parkinson's (`ds003490`, Clinical Research Center) | 50 / 75 | PD (25 ON + 25 OFF) vs. HC (25) | PD 67.4 ± 8.7; HC 69.1 ± 8.2 | 42.0% (21/50) | BrainVision (64 Ch) | 500 Hz $\\to$ 250 Hz | Counterbalanced ON vs. $\\ge 15\\text{{ h}}$ overnight washout |
| **Study 7** | Iowa Parkinson's (`ds004584`, Hospital Movement Clinic) | 149 / 149 | PD (100) vs. HC (49) | PD 68.5 ± 8.1; HC 70.9 ± 7.6 | 42.3% (63/149) | BrainVision (64 Ch, native `Pz` ref) | 500 Hz $\\to$ 250 Hz | Eyes-closed; MoCA ($N=149$, 25.1 ± 3.6) & UPDRS-III ($N=100$, 12.5 ± 7.1) |

*Note: Master accounting totals $N = 566$ unique analyzed participants and $677$ primary sessions ($142 + 58 + 119 + 46 + 88 + 75 + 149 = 677$), or $737$ total sessions when including the $N = 60$ Study 2 eyes-open recordings (32 MDD, 28 HC). In Study 2, only group-level age summary statistics were published in Figshare.*

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

### Table S8. Signal Preprocessing and Artifact Control Specifications
| Pipeline Stage | Parameter / Setting | Software Implementation | Methodological Purpose |
| :--- | :--- | :--- | :--- |
| **Bandpass Filter** | 1.0–40.0 Hz | Zero-phase FIR (Hamming window, MNE `raw.filter`) | Attenuates slow drift ($<1\\text{{ Hz}}$) and 50/60 Hz mains interference |
| **Resampling** | 250 Hz | Polyphase anti-aliased resampling (`raw.resample`) | Uniform 4 ms sample interval across 256, 500, and 512 Hz native archives |
| **Spatial Reference** | Common Average (CAR) | Instantaneous 19-channel mean subtraction | Harmonizes reference geometry and reconstructs native `Pz` reference in Study 7 |
| **Epoching** | 4.0 s non-overlapping | 1,000 samples/epoch (`mne.make_fixed_length_epochs`) | Supports 2.0 s Welch sub-windows ($\\Delta f = 0.50\\text{{ Hz}}$) |
| **Artifact Threshold** | 150 $\\mu\\text{{V}}$ peak-to-peak | Channel-wise max peak-to-peak screening | Rejects gross ocular blink and electromyographic bursts |
| **Geriatric Fallback** | 200 $\\mu\\text{{V}}$ ($\\ge 8$ epochs) | Lowest-amplitude epoch retention in Study 7 | Prevents tremor-related attrition in elderly movement-disorder recordings |

### Table S8b. Retained Clean Epoch Counts and Usable Recording Duration Across Datasets
| Study ID | Dataset | Condition | Analyzed Sessions | Threshold | Clean Epochs (Mean $\\pm$ SD [Range]) | Clean Duration in Seconds (Mean $\\pm$ SD [Range]) | Retained Epoch % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Study 1** | Sleep Deprivation (`ds004902`) | Paired SD & NS | 142 | 150 $\\mu\\text{{V}}$ | 135.1 ± 21.2 [50–174] | 540.4 ± 84.8 s [200–696 s] | 90.1% |
| **Study 2a** | Outpatient MDD (`4244171`) | Eyes-Closed (Primary) | 58 | 150 $\\mu\\text{{V}}$ | 71.6 ± 7.5 [43–94] | 286.4 ± 30.0 s [172–376 s] | 95.5% |
| **Study 2b** | Outpatient MDD (`4244171`) | Eyes-Open (Secondary) | 60 | 150 $\\mu\\text{{V}}$ | 54.6 ± 18.3 [10–78] | 218.4 ± 73.2 s [40–312 s] | 87.4% |
| **Study 3** | Univ. Depression (`ds003478`) | Eyes-Open | 119 | 150 $\\mu\\text{{V}}$ | 75.1 ± 21.1 [11–120] | 300.4 ± 84.4 s [44–480 s] | 83.4% |
| **Study 4** | UCSD Parkinson's (`ds002778`) | ON & OFF | 46 | 150 $\\mu\\text{{V}}$ | 27.3 ± 14.9 [7–49] | 109.2 ± 59.6 s [28–196 s] | 78.0% |
| **Study 5** | AHEPA Dementia (`ds004504`) | Eyes-Closed | 88 | 150 $\\mu\\text{{V}}$ | 171.9 ± 44.2 [12–319] | 687.7 ± 176.9 s [48–1,276 s] | 91.4% |
| **Study 6** | UNM Parkinson's (`ds003490`) | ON & OFF | 75 | 150 $\\mu\\text{{V}}$ | 73.8 ± 43.7 [9–173] | 295.0 ± 174.6 s [36–692 s] | 82.0% |
| **Study 7** | Iowa Parkinson's (`ds004584`) | Eyes-Closed | 149 | 200 $\\mu\\text{{V}}$ | 28.6 ± 12.8 [5–85] | 114.4 ± 51.2 s [20–340 s] | 76.3% |

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
| **Abstract** | 2 | Structured summary of background, methods, results, conclusions. | Main Text Abstract ($<250$ words) |
| **Background & Aims** | 3a–3b | Clinical/methodological context and explicit research questions. | Main Text §1 (four-paragraph introduction and three research questions) |
| **Data Sources** | 4a–4b | Study designs, public archives, inclusion/exclusion rules. | Main Text §2.1, Table S3, and Figure S5 ($574$ enrolled $\\to 8$ excluded $\\to 566$ analyzed) |
| **Predictors** | 5a–5b | Feature definitions and outcome-blind extraction. | Main Text §2.2–§2.4, §S1.1–§S1.3 (automated `specparam` + PLI-OMST extraction blind to labels) |
| **Sample Size** | 6 | Full accounting of participants and recording sessions. | $N = 566$ unique participants, $677$ primary sessions ($737$ with Study 2 EO); Table S3 & Figure S5 |
| **Missing Data** | 7 | Handling of reference channels, artifacts, and missing covariates. | Main Text §2.2, Table S8, Table S18 note (Study 2 lacks individual ages; 1 missing age in Study 3) |
| **Model Specification** | 8a–8b | Full architectures, tuning grids, and leak-free calibration. | Main Text §2.5–§2.6, §S1.4, Table S7, and Table S7b |
| **Validation** | 9–10 | Internal group-stratified CV, external site-calibrated transfer, metrics. | Main Text §2.5–§2.7; ROC-AUC (1,000-bootstrap 95% CIs), Balanced Accuracy, Wilcoxon tests, $r$, $R^2$, MAE |
| **Results** | 11–15 | Participant flow, baseline comparisons, ablations, permutations. | Main Text Tables 1–4, Figures 1–4; Supplementary Tables S4, S6, S11, S11b, S12–S18; Figures S5–S6 |
| **Interpretation & Scope** | 16–18 | Alternative explanations, 19-channel scope limits, cross-sectional bounds. | Main Text §4.1–§4.5 |
| **Open Science & Ethics** | 19–22 | Persistent dataset IDs, GitHub code URL, IRB exemption, competing interests. | Main Text Data and Code Availability |

---

{ts11_11b}

---

### Table S12. Initial Four-Cohort $4 \\times 4$ Cross-Dataset Transfer Matrix (12 Directed Pairs)
| Source Dataset | Target Dataset | Source $N$ | Target $N$ | `NormativeGCN` AUC [95% CI] | `NormativeGCN` Bal. Acc. | Logistic Regression AUC [95% CI] | Logistic Bal. Acc. |
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

*Note: Site-calibrated direct transfer across Studies 1–4 (`reports/multicohort_full_transfer_matrix.csv`) without fine-tuning. Brackets denote 1,000-bootstrap 95% confidence intervals. Slight third-decimal `NormativeGCN` differences ($\\le 0.003$, e.g., $0.832$ vs. $0.833$ and $0.277$ vs. $0.274$) relative to the expanded 7-cohort run in Table 3 and Table S17 (`reports/comprehensive_transdiagnostic_transfer_matrix.csv`) reflect independent PyTorch weight-initialization seeds, whereas deterministic `Logistic Regression` AUCs match identically.*

---

### Table S13. Recording-State Sensitivity: Study 2 Eyes-Open vs. Eyes-Closed Transfer with Study 3
| Transfer Configuration | Source Dataset & State | Target Dataset & State | `NormativeGCN` AUC [95% CI] | `NormativeGCN` Bal. Acc. | Logistic Regression AUC [95% CI] | Logistic Bal. Acc. |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **State-Mismatched Forward** | Study 3 Univ. Dep. (EO, $N=119$) | Study 2 Outpatient MDD (EC, $N=58$) | **0.832 [0.706, 0.935]** | **81.1%** | 0.752 [0.621, 0.881] | 63.1% |
| **State-Matched Forward** | Study 3 Univ. Dep. (EO, $N=119$) | Study 2 Outpatient MDD (EO, $N=60$) | 0.758 [0.638, 0.875] | 68.1% | **0.795 [0.664, 0.903]** | **69.6%** |
| **State-Mismatched Reverse** | Study 2 Outpatient MDD (EC, $N=58$) | Study 3 Univ. Dep. (EO, $N=119$) | 0.465 [0.353, 0.581] | 50.5% | **0.604 [0.485, 0.705]** | 50.7% |
| **State-Matched Reverse** | Study 2 Outpatient MDD (EO, $N=60$) | Study 3 Univ. Dep. (EO, $N=119$) | 0.569 [0.455, 0.669] | 54.2% | **0.592 [0.483, 0.705]** | **55.2%** |
| **Within-Site Cross-State** | Study 2 Outpatient MDD (EO, $N=60$) | Study 2 Outpatient MDD (EC, $N=58$) | **0.937 [0.874, 0.982]** | 82.6% | 0.931 [0.849, 0.994] | **93.3%** |
| **Within-Site Cross-State** | Study 2 Outpatient MDD (EC, $N=58$) | Study 2 Outpatient MDD (EO, $N=60$) | 0.875 [0.777, 0.953] | 67.2% | **0.887 [0.792, 0.963]** | **82.6%** |

*Note: Matching eyes-open state (`Study 2 EO -> Study 3 EO`) increases reverse `NormativeGCN` transfer from $0.465$ to $0.569$ (`reports/mumtaz_eo_state_matched_transfer.csv`). Note that within-site cross-state rows share overlapping Study 2 participants across EC and EO conditions and therefore measure cross-condition consistency rather than out-of-subject generalization.*

---

{ts14}

---

{ts15}

---

{ts16_17}

---

{ts18}

---

## Section S3. Supplementary Figures (Figures S1–S6)

![Figure S1](../reports/fig_S1_specparam_example.png){{width=6.2in}}

**Figure S1. Illustrative Single-Channel Spectral Parameterization (`specparam`) and Isolated Periodic Residual Spectrum.** *(A) Raw $\\log_{{10}} \\mathrm{{PSD}}(f)$ (dark slate), fitted aperiodic background $L(f) = b - \\chi \\log_{{10}} f$ (red dashed; $b = 0.650\\text{{ B}}$, $\\chi = 1.152$), and combined model fit ($R^2 = 0.993$, blue solid). (B) Flattened periodic residual spectrum after subtracting $L(f)$, isolating narrowband theta and alpha peaks above the $0.0\\text{{ B}}$ baseline.*

---

![Figure S2](../reports/fig_S2_omst_topology.png){{width=5.4in}}

**Figure S2. Canonical 19-Channel 10–20 Orthogonal Minimum Spanning Tree (OMST) Connectome Schematic.** *Primary Minimum Spanning Tree (Tree 1, blue solid lines, 18 edges) and orthogonal second spanning tree (Tree 2, amber dashed lines, 18 disjoint edges; curved arcs improve visual legibility for midline and homologous chords). Across all $N = 566$ analyzed participants and $677$ primary sessions, GCE optimization selected $M = 1.52 \\pm 0.50$ trees ($E = 27.33 \\pm 9.00$ edges, mean degree $\\bar{{k}} = 2.88 \\pm 0.95$, edge density $15.98\\% \\pm 5.26\\%$) with $0.0000\\%$ isolated nodes.*

---

![Figure S3](../reports/fig_S3_gain_invariance.png){{width=6.0in}}

**Figure S3. Numerical Verification of Multiplicative Amplifier Gain Invariance.** *(A) Unadjusted broadband offset $b$ shifts linearly as $\\Delta b = 2 \\log_{{10}}(k)$ across multiplicative gain factors $k \\in [0.01, 100]$. (B) Absolute numerical deviations of periodic band residuals ($\\theta, \\delta, \\alpha$) and aperiodic exponent $\\chi$ remain below $4 \\times 10^{{-15}}$ across four orders of magnitude.*

---

![Figure S4](../reports/fig_S4_cohort_spectra.png){{width=6.5in}}

**Figure S4. Representative Cohort Power Spectra: Total Raw $\\log_{{10}} \\mathrm{{PSD}}$ vs. Isolated Periodic Residuals.** *Top row (A1–D1): Grand-average total $\\log_{{10}} \\mathrm{{PSD}}$ (1–40 Hz) for Study 1 Sleep Deprivation ($N = 71$ pairs), Study 2 Outpatient MDD ($N = 30$ MDD vs. $28$ HC), Study 3 University Depression Screening ($N = 44$ $\\text{{BDI-II}} \\ge 14$ vs. $75$ controls $\\text{{BDI-II}} \\le 7$), and Study 4 UCSD Parkinson's Disease ($N = 15$ PD-ON vs. $16$ HC). Bottom row (A2–D2): Corresponding periodic residual spectra after subtracting fitted aperiodic $1/f^\\chi$ floors.*

---

![Figure S5](../reports/fig_S5_enrollment_flowchart.png){{width=6.5in}}

**Figure S5. Multi-Dataset Participant Enrollment, Quality Control, and Harmonization Flowchart.** *From $N = 574$ unique enrolled participants ($685$ primary sessions; $745$ including Study 2 eyes-open) across seven public archives (6 OpenNeuro + 1 Figshare), $8$ participants ($1.4\\%$) were excluded under pre-specified artifact ($N = 6$ in Study 2) and borderline screening ($N = 2$ with $10 \\le \\text{{BDI-II}} \\le 13$ in Study 3) criteria, yielding $N = 566$ unique analyzed participants across $677$ primary sessions ($737$ total sessions including Study 2 eyes-open).*

---

![Figure S6](../reports/fig_S6_permutation_tests.png){{width=6.5in}}

**Figure S6. Label-Permutation Null Distributions for `NormativeGCN` Benchmarks (1,000 Permutations).** *Null ROC-AUC histograms against observed 50-epoch `GCN-OMST` performance for: (A) Study 2 Outpatient MDD 5-fold CV ($\\text{{AUC}} = 0.902$); (B) Study 3 University Depression Screening 5-fold CV ($\\text{{AUC}} = 0.647$); (C) Study 3 $\\to$ Study 2 cross-dataset transfer ($\\text{{AUC}} = 0.833$); (D) Study 1 Sleep Deprivation $\\to$ Study 4 UCSD PD transfer ($\\text{{AUC}} = 0.802$); (E) Study 1 Sleep Deprivation $\\to$ Study 2 Outpatient MDD sub-chance transfer ($\\text{{AUC}} = 0.274$); and (F) Study 3 University Depression $\\to$ Study 4 UCSD PD sub-chance transfer ($\\text{{AUC}} = 0.271$). All observed departures from chance ($0.50$) exceed the permutation null ($\\text{{two-sided }} p < 0.002$).*
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
    assert "### S1.5" not in supp_md, "Supplementary S1.5 post-hoc power section must be removed!"
    assert "AdamW" not in combined and "35 epochs" not in combined and "35 training epochs" not in combined, "Stale AdamW/35 epochs found!"
    assert "ElasticNet" not in supp_md and "Multilayer Perceptron" not in supp_md, "Stale ElasticNet/MLP found in Table S7b!"
    assert EXACT_TITLE in supp_md, "Exact title missing in Supplementary Material!"
    assert "Challenges Graph Neural Network Superiority" not in combined, "Old title still present!"
    assert "Author Background and Collaborative Data Governance" not in main_md, "Verbose author background section still present!"
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
