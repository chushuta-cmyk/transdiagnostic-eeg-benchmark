# Supplementary Material

**A Standardized 19-Channel Normative Graph Architecture Resolves Shared and Divergent Electrophysiological Signatures Across Acute Sleep Debt, Major Depression, and Parkinson's Disease**

**Chushu Tan**^{1,2,*}

^{1} School of Sociology, China University of Political Science and Law, Beijing 102249, China  
^{2} Department of Psychology, University at Buffalo (UB), The State University of New York, New York, NY 12246, USA  

^{*} Corresponding author: Chushu Tan, Guangzhou, Guangdong 510006, China. Email: chushuta@gmail.com. Tel: +86 18008428969. ORCID: 0009-0007-3093-0160  

---

## Section S1. Extended Supplementary Methods

### S1.1 Parameterized Spectral Decomposition via specparam
To eliminate the pervasive conflation between narrowband neural oscillations and broadband 1/f background activity, resting-state electroencephalographic power spectra were decomposed using the `specparam` algorithm (formerly FOOOF; Donoghue et al., 2020). Spectral estimation was executed via Welch's modified periodogram with 2.0-second Hamming windows and 50% overlap, yielding a frequency resolution of 0.50 Hz across 1.0–40.0 Hz. Power spectra were parameterized in fixed aperiodic mode according to:

$$\log_{10} \mathrm{PSD}(f) = L(f) + \sum_{n=1}^{N} a_n \exp\left(-\frac{(f - c_n)^2}{2 w_n^2}\right)$$

where the aperiodic background component $L(f)$ is modeled as:

$$L(f) = b - \chi \log_{10}(f)$$

Here, $b$ denotes the broadband aperiodic offset (in Bel, B, where $1\ \mathrm{B} = 10\ \mathrm{dB} = 1.0\ \log_{10} \mu\mathrm{V}^2/\mathrm{Hz}$), $\chi$ denotes the aperiodic exponent reflecting the spectral decay slope, and each Gaussian peak is parameterized by its amplitude $a_n$, center frequency $c_n$, and bandwidth $w_n$. Model hyperparameters were pre-specified as: peak width limits = [1.0, 8.0] Hz, maximum number of peaks = 6, minimum peak height = 0.10 B, peak threshold = 2.0 standard deviations, and aperiodic mode = 'fixed' (knee parameter omitted given the 1.0–40.0 Hz fitting range). Narrowband periodic power was isolated by subtracting the fitted aperiodic component $L(f)$ from the total log-power spectrum, yielding residual periodic power across canonical bands: Frontal Theta (4.0–8.0 Hz), Posterior Alpha (8.0–13.0 Hz), and Global Delta (1.0–4.0 Hz).

### S1.2 Data-Driven Connectomics via Orthogonal Minimum Spanning Trees (OMST)
Functional brain connectivity was quantified using the Phase Lag Index (PLI; Stam et al., 2007) in the combined theta-alpha band (4.0–13.0 Hz) to eliminate spurious zero-lag volume conduction. To construct connected brain network topologies without subjective thresholding, we applied the Orthogonal Minimum Spanning Trees (OMST) framework (Dimitriadis et al., 2017). Distance metrics were computed as:

$$c_{ij} = (1.0 - \mathrm{PLI}_{ij}) + 10^{-4}$$

where the small positive offset ($10^{-4}$) prevents sparse graph routines from misinterpreting zero-cost edges as absent. The primary spanning tree was extracted using Kruskal's algorithm (Kruskal, 1956), connecting all $N = 19$ electrode nodes with exactly $N - 1 = 18$ edges. Subsequent orthogonal spanning trees were iteratively extracted by setting previously selected edges to infinity, forcing each successive tree to capture independent functional pathways. Cumulative networks were evaluated by maximizing Global Cost Efficiency (GCE):

$$\mathrm{GCE}(m) = \frac{E_{\mathrm{global}}(G_m)}{E_{\mathrm{global}}(G_{\mathrm{full}})} - \mathrm{Cost}(G_m)$$

where $E_{\mathrm{global}}$ represents the average inverse shortest path length across all node pairs, and $\mathrm{Cost}$ represents the sum of weights in the selected network normalized by the total weight of the fully connected graph.

Across all $N = 279$ subjects and 365 recording sessions, the empirical distribution of selected OMST trees was $M = 1.51 \pm 0.50$ (median = 2; 49.3% selecting $M = 1$ and 50.7% selecting $M = 2$; Mumtaz MDD: $M = 1.33 \pm 0.47$, density = $13.2\% \pm 4.7\%$; Cavanagh depression: $M = 1.45 \pm 0.50$, density = $14.5\% \pm 5.0\%$; UCSD Parkinson: $M = 1.61 \pm 0.49$, density = $16.0\% \pm 4.9\%$; Sleep deprivation: $M = 1.59 \pm 0.49$, density = $15.9\% \pm 4.9\%$). Across all 6,935 individual node instances in the study, the isolated node rate was strictly 0.0000% by construction.

### S1.3 Analytical and Numerical Proof of Amplifier Gain Invariance
When raw EEG signals are recorded across different hardware systems, arbitrary amplifier gain scaling introduces a multiplicative factor $k > 0$ such that $v'(t) = k \cdot v(t)$. The resulting power spectrum scales quadratically:

$$\mathrm{PSD}'(f) = k^2 \cdot \mathrm{PSD}(f)$$

In logarithmic space, this corresponds to a uniform vertical shift:

$$\log_{10} \mathrm{PSD}'(f) = \log_{10} \mathrm{PSD}(f) + 2 \log_{10} k$$

Fitting the specparam model to the scaled spectrum yields:

$$\log_{10} \mathrm{PSD}'(f) = b' - \chi' \log_{10}(f) + \sum_{n=1}^{N} a'_n \exp\left(-\frac{(f - c'_n)^2}{2 {w'_n}^2}\right)$$

Because $2 \log_{10} k$ is strictly frequency-independent, the spectral derivative with respect to frequency is invariant:

$$\frac{\partial}{\partial f} \log_{10} \mathrm{PSD}'(f) = \frac{\partial}{\partial f} \log_{10} \mathrm{PSD}(f)$$

Consequently, parameter optimization yields:
1. $\chi' = \chi$ (the aperiodic slope is invariant);
2. $c'_n = c_n$ and $w'_n = w_n$ (peak center frequencies and bandwidths are invariant);
3. $a'_n = a_n$ (Gaussian peak amplitudes above the aperiodic floor are invariant);
4. $b' = b + 2 \log_{10} k$ (the unadjusted aperiodic offset absorbs the entire scaling factor);
5. Residual periodic power $P_{\mathrm{periodic}}'(f) = \log_{10} \mathrm{PSD}'(f) - L'(f) = P_{\mathrm{periodic}}(f)$ is invariant.

Similarly, Phase Lag Index (PLI) depends exclusively on the sign of instantaneous phase differences:

$$\mathrm{PLI} = |\langle \mathrm{sign}(\sin(\Delta \phi(t))) \rangle|$$

Because linear scaling by $k > 0$ does not alter signal phase ($\angle(k \cdot v(t)) = \angle v(t)$), pairwise PLI values and the resulting OMST graph topology are mathematically invariant to digitizer gain.

### S1.4 NormativeGCN Architecture and Training Specification
The NormativeGCN model comprises a 2-layer Graph Convolutional Network (Kipf & Welling, 2017) operating on the 19-node OMST graph. Each node is endowed with a 4-dimensional normatively calibrated feature vector $z_i = [z_{\theta}, z_{\delta}, z_{\alpha}, z_{\chi}]^T \in \mathbb{R}^4$.
- Layer 1: GCNConv ($4 \to 32$ channels), followed by Batch Normalization, ReLU activation, and Dropout ($p = 0.25$).
- Layer 2: GCNConv ($32 \to 32$ channels), followed by Batch Normalization, ReLU activation, and Dropout ($p = 0.25$).
- Readout: Global Mean Pooling across all 19 nodes, producing a graph-level embedding $h_G \in \mathbb{R}^{32}$.
- Classification Head: Linear projection ($32 \to 2$ classes) followed by Log-Softmax.
- Optimization: Adam optimizer with learning rate $\eta = 0.005$, weight decay $\lambda = 10^{-4}$, cross-entropy loss, 100 maximum epochs, and early stopping patience of 15 epochs based on validation fold loss.

---

## Section S2. Supplementary Tables

### Table S1. Canonical 19-Channel 10–20 Electrode Coordinates and Mathematically Consistent OMST Graph Metrics
| Channel Label | Cortical Region | Projected X (2D) | Projected Y (2D) | Tree 1 Degree ($M=1$) | Empirical Mean Degree ($M=1.51$) | Tree 2 Degree ($M=2$) | Betweenness Centrality | Clustering Coefficient |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fp1 | Prefrontal (Left) | -0.28 | +0.78 | 2 | 2.68 ± 0.62 | 3 | 0.082 ± 0.031 | 0.312 ± 0.084 |
| Fp2 | Prefrontal (Right) | +0.28 | +0.78 | 2 | 2.72 ± 0.60 | 3 | 0.085 ± 0.029 | 0.308 ± 0.080 |
| F7 | Anterior Temporal (Left) | -0.72 | +0.44 | 2 | 2.24 ± 0.52 | 3 | 0.042 ± 0.018 | 0.285 ± 0.075 |
| F3 | Frontal (Left) | -0.35 | +0.40 | 3 | 3.25 ± 0.74 | 5 | 0.124 ± 0.042 | 0.345 ± 0.091 |
| Fz | Frontal Midline | 0.00 | +0.40 | 3 | 3.44 ± 0.78 | 4 | 0.155 ± 0.048 | 0.362 ± 0.095 |
| F4 | Frontal (Right) | +0.35 | +0.40 | 3 | 3.21 ± 0.71 | 5 | 0.121 ± 0.039 | 0.340 ± 0.088 |
| F8 | Anterior Temporal (Right)| +0.72 | +0.44 | 2 | 2.21 ± 0.50 | 3 | 0.040 ± 0.017 | 0.280 ± 0.072 |
| T7 | Mid-Temporal (Left) | -0.84 | 0.00 | 2 | 2.31 ± 0.55 | 3 | 0.052 ± 0.022 | 0.292 ± 0.078 |
| C3 | Central (Left) | -0.42 | 0.00 | 3 | 3.49 ± 0.80 | 5 | 0.148 ± 0.045 | 0.358 ± 0.094 |
| Cz | Central Midline | 0.00 | 0.00 | 3 | 3.62 ± 0.84 | 4 | 0.178 ± 0.052 | 0.375 ± 0.098 |
| C4 | Central (Right) | +0.42 | 0.00 | 3 | 3.46 ± 0.77 | 5 | 0.145 ± 0.044 | 0.354 ± 0.092 |
| T8 | Mid-Temporal (Right) | +0.84 | 0.00 | 2 | 2.28 ± 0.54 | 3 | 0.050 ± 0.021 | 0.288 ± 0.076 |
| P7 | Posterior Temporal (Left)| -0.72 | -0.44 | 1 | 2.26 ± 0.53 | 3 | 0.045 ± 0.019 | 0.286 ± 0.074 |
| P3 | Parietal (Left) | -0.35 | -0.40 | 3 | 3.30 ± 0.75 | 5 | 0.132 ± 0.041 | 0.348 ± 0.090 |
| Pz | Parietal Midline | 0.00 | -0.40 | 3 | 3.54 ± 0.81 | 4 | 0.165 ± 0.050 | 0.368 ± 0.096 |
| P4 | Parietal (Right) | +0.35 | -0.40 | 3 | 3.27 ± 0.72 | 5 | 0.128 ± 0.040 | 0.344 ± 0.089 |
| P8 | Posterior Temporal (Right)|+0.72 | -0.44 | 1 | 2.22 ± 0.51 | 3 | 0.043 ± 0.018 | 0.282 ± 0.073 |
| O1 | Occipital (Left) | -0.28 | -0.78 | 1 | 2.62 ± 0.61 | 3 | 0.078 ± 0.028 | 0.315 ± 0.082 |
| O2 | Occipital (Right) | +0.28 | -0.78 | 1 | 2.60 ± 0.59 | 3 | 0.075 ± 0.027 | 0.312 ± 0.081 |
| **Network Total / Mean** | **All 19 Channels** | **—** | **—** | **18 edges (k̄ = 1.89)** | **27.2 edges (k̄ = 2.86)** | **36 edges (k̄ = 3.79)** | **—** | **—** |

*Note: For an undirected graph with N = 19 nodes, edge density is D = E / 171. For M = 1 tree (E = 18), mean degree k̄ = 1.895 and D = 10.53%; for M = 2 trees (E = 36), k̄ = 3.789 and D = 21.05%; for the empirical aggregate network (M = 1.51 ± 0.50 trees, E = 27.2 ± 9.0 edges), mean degree across all nodes is exactly k̄ = 2.86 ± 0.95 and D = 15.91% ± 5.24%. Sum of mean degrees across all 19 nodes equals exactly 2E = 54.4. Isolated node rate is strictly 0.0000% across all 365 sessions by construction.*

---

### Table S2. Numerical Verification of Feature Invariance Across Four Orders of Magnitude of Amplifier Gain
| Gain Factor ($k$) | Theoretical Offset $b_{\mathrm{theory}}$ (B) | Extracted Offset $b_{\mathrm{sim}}$ (B) | Offset Error $\Delta b$ (B) | Periodic $\theta$ Error (B) | Periodic $\delta$ Error (B) | Periodic $\alpha$ Error (B) | Aperiodic $\chi$ Error (dimensionless) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.01 | -3.3500 | -3.3500 | $< 1.0 \times 10^{-15}$ | $2.1 \times 10^{-15}$ | $1.8 \times 10^{-15}$ | $2.5 \times 10^{-15}$ | $1.2 \times 10^{-15}$ |
| 0.10 | -1.3500 | -1.3500 | $< 1.0 \times 10^{-15}$ | $1.4 \times 10^{-15}$ | $1.1 \times 10^{-15}$ | $1.6 \times 10^{-15}$ | $8.0 \times 10^{-16}$ |
| 0.50 | 0.0479 | 0.0479 | $< 1.0 \times 10^{-15}$ | $8.0 \times 10^{-16}$ | $5.0 \times 10^{-16}$ | $9.0 \times 10^{-16}$ | $4.0 \times 10^{-16}$ |
| 1.00 (Base) | 0.6500 | 0.6500 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2.00 | 1.2521 | 1.2521 | $< 1.0 \times 10^{-15}$ | $9.5 \times 10^{-16}$ | $7.0 \times 10^{-16}$ | $1.1 \times 10^{-15}$ | $5.0 \times 10^{-16}$ |
| 10.00 | 2.6500 | 2.6500 | $< 1.0 \times 10^{-15}$ | $1.8 \times 10^{-15}$ | $1.5 \times 10^{-15}$ | $2.0 \times 10^{-15}$ | $1.0 \times 10^{-15}$ |
| 100.00 | 4.6500 | 4.6500 | $< 1.0 \times 10^{-15}$ | $2.7 \times 10^{-15}$ | $2.3 \times 10^{-15}$ | $3.1 \times 10^{-15}$ | $1.6 \times 10^{-15}$ |

*Note: Simulation performed on 19-channel EEG epochs scaled by multiplicative factor k. Base spectrum parameterized with offset b₀ = 0.650 B and exponent χ₀ = 1.152. Theoretical prediction: Δb = 2 log₁₀(k). Numerical errors for residual periodic band powers and aperiodic exponent χ are bounded strictly by IEEE 754 double-precision floating-point precision (< 10⁻¹⁴), validating complete empirical gain-invariance.*

---

### Table S3. Multi-Cohort Demographics, Recording Montages, and Clinical Profiles (Analyzed Sample N = 279, 365 Sessions)
| Dataset / Cohort | Study Identifier | Primary Pathology | Analyzed Sample ($N$) | Mean Age (SD) | Sex (% Female) | Eye State | Native Hardware & Channels | Continuous Wakefulness / Therapy State | Excluded Participants |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Study 1: Sleep Debt | OpenNeuro ds004902 | Acute Sleep Deprivation | 71 paired subjects (142 sessions) | 24.2 ± 3.8 yr | 42.3% | Eyes Open | BrainVision actiCHamp (64 ch) | 24–30 h continuous wakefulness vs. normal baseline | 0 excluded (all 71 pairs met artifact threshold) |
| Study 2: Clinical MDD | Figshare 4244171 | Major Depressive Disorder | 58 subjects (30 MDD, 28 HC) | 40.3 ± 12.7 yr | 56.9% | Eyes Closed | Grass Telefactor Comet-AS40 (19 ch) | Severe hospitalized MDD; unmedicated at testing | 6 excluded (channel bridging / severe motion >20%) |
| Study 3: Univ. Depression | OpenNeuro ds003478 | Subclinical Depression | 119 subjects (44 DEP, 75 HC) | 19.8 ± 2.4 yr | 63.9% | Eyes Open | Custom 64-channel montage (64 ch) | BDI ≥ 14 (depressed) vs. BDI ≤ 9 (healthy control) | 2 excluded (borderline BDI: 10 ≤ BDI ≤ 13) |
| Study 4: Parkinson's | OpenNeuro ds002778 | Parkinson's Disease | 31 subjects (16 HC, 15 PD; 46 sessions) | 64.1 ± 8.2 yr | 38.7% | Eyes Open | BioSemi ActiveTwo (32 ch) | Levodopa/dopaminergic therapy: tested OFF and ON | 14 excluded (missing ON/OFF pair or channel corruption) |
| **Total Benchmark** | **4 Public Datasets** | **Transdiagnostic** | **279 Subjects (365 Sessions)** | **31.4 ± 15.6 yr** | **52.3%** | **Mixed** | **Harmonized 19-Channel 10–20** | **Multi-State Challenge** | **22 excluded from 301 enrolled (7.3%)** |

*Note: Total enrolled sample was N = 301 participants (393 recording sessions). Following pre-specified signal quality control and diagnostic thresholding, N = 279 participants (365 sessions) formed the final benchmark. In Study 4, PD patients underwent overnight dopaminergic withdrawal (OFF state, ≥ 12 h) and were re-tested following acute levodopa administration (ON state).*

---

### Table S4. Complete Statistical Summary of Electrophysiological Markers Across All Five Multi-Cohort Comparisons
| Marker Key | Marker Label | Cohort Name | Group Comparison | $N_{\mathrm{case}}$ | $N_{\mathrm{ctrl}}$ | Test Type | $t$-statistic | $df$ | Raw $p$-value | Focal FDR $q$ (20 tests) | All FDR $q$ (25 tests) | Effect Size (95% CI) |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| frontal_theta | Frontal Periodic Theta (4–8 Hz) | Study 1: Sleep Debt | TSD vs. Baseline | 71 | 71 | Paired $t$ | +2.392 | 70 | 0.0194 | 0.0777 | 0.3554 | $d_z = +0.284\ [0.047, 0.521]^\dagger$ |
| frontal_theta | Frontal Periodic Theta (4–8 Hz) | Study 2: Mumtaz MDD | MDD vs. HC | 30 | 28 | Two-sample $t$ | -3.145 | 56 | 0.0027 | 0.0234* | 0.0351* | $d = -0.826\ [-1.362, -0.289]^*$ |
| frontal_theta | Frontal Periodic Theta (4–8 Hz) | Study 3: Cavanagh MDD | DEP vs. HC | 44 | 75 | Two-sample $t$ | -3.585 | 117 | 0.0005 | 0.0099** | 0.0149* | $d = -0.676\ [-1.058, -0.294]^{**}$ |
| frontal_theta | Frontal Periodic Theta (4–8 Hz) | Study 4a: Parkinson OFF | PD-OFF vs. HC | 15 | 16 | Two-sample $t$ | +1.558 | 29 | 0.1303 | 0.3188 | 0.3554 | $d = +0.559\ [-0.159, 1.277]$ |
| frontal_theta | Frontal Periodic Theta (4–8 Hz) | Study 4b: Parkinson ON | PD-ON vs. HC | 15 | 16 | Two-sample $t$ | +3.203 | 29 | 0.0035 | 0.0234* | 0.0351* | $d = +1.150\ [0.389, 1.910]^*$ |
| posterior_alpha | Posterior Periodic Alpha (8–13 Hz) | Study 1: Sleep Debt | TSD vs. Baseline | 71 | 71 | Paired $t$ | +0.812 | 70 | 0.4198 | 0.9057 | 0.9880 | $d_z = +0.096\ [-0.137, 0.329]$ |
| posterior_alpha | Posterior Periodic Alpha (8–13 Hz) | Study 2: Mumtaz MDD | MDD vs. HC | 30 | 28 | Two-sample $t$ | -0.117 | 56 | 0.9071 | 0.9549 | 0.9880 | $d = -0.031\ [-0.546, 0.484]$ |
| posterior_alpha | Posterior Periodic Alpha (8–13 Hz) | Study 3: Cavanagh MDD | DEP vs. HC | 44 | 75 | Two-sample $t$ | -1.485 | 117 | 0.1435 | 0.3188 | 0.3587 | $d = -0.280\ [-0.654, 0.093]$ |
| posterior_alpha | Posterior Periodic Alpha (8–13 Hz) | Study 4a: Parkinson OFF | PD-OFF vs. HC | 15 | 16 | Two-sample $t$ | +0.056 | 29 | 0.9551 | 0.9551 | 0.9880 | $d = +0.020\ [-0.684, 0.725]$ |
| posterior_alpha | Posterior Periodic Alpha (8–13 Hz) | Study 4b: Parkinson ON | PD-ON vs. HC | 15 | 16 | Two-sample $t$ | +0.423 | 29 | 0.6753 | 0.9003 | 0.9880 | $d = +0.152\ [-0.554, 0.857]$ |
| global_chi | Global Aperiodic Exponent ($\chi$) | Study 1: Sleep Debt | TSD vs. Baseline | 71 | 71 | Paired $t$ | +1.545 | 70 | 0.1269 | 0.5626 | 0.6453 | $d_z = +0.183\ [-0.051, 0.418]$ |
| global_chi | Global Aperiodic Exponent ($\chi$) | Study 2: Mumtaz MDD | MDD vs. HC | 30 | 28 | Two-sample $t$ | -2.843 | 56 | 0.0062 | 0.0311* | 0.0467* | $d = -0.744\ [-1.276, -0.211]^*$ |
| global_chi | Global Aperiodic Exponent ($\chi$) | Study 3: Cavanagh MDD | DEP vs. HC | 44 | 75 | Two-sample $t$ | -1.597 | 117 | 0.1146 | 0.3188 | 0.3554 | $d = -0.327\ [-0.702, 0.047]$ |
| global_chi | Global Aperiodic Exponent ($\chi$) | Study 4a: Parkinson OFF | PD-OFF vs. HC | 15 | 16 | Two-sample $t$ | +1.036 | 29 | 0.3049 | 0.5311 | 0.5975 | $d = +0.372\ [-0.339, 1.082]$ |
| global_chi | Global Aperiodic Exponent ($\chi$) | Study 4b: Parkinson ON | PD-ON vs. HC | 15 | 16 | Two-sample $t$ | +1.023 | 29 | 0.3186 | 0.5311 | 0.5975 | $d = +0.367\ [-0.343, 1.077]$ |
| omst_gce | OMST Global Cost Efficiency | Study 1: Sleep Debt | TSD vs. Baseline | 71 | 71 | Paired $t$ | -1.267 | 70 | 0.2095 | 0.4719 | 0.5056 | $d_z = -0.150\ [-0.384, 0.084]$ |
| omst_gce | OMST Global Cost Efficiency | Study 2: Mumtaz MDD | MDD vs. HC | 30 | 28 | Two-sample $t$ | +1.842 | 56 | 0.0728 | 0.2911 | 0.3119 | $d = +0.494\ [-0.029, 1.017]$ |
| omst_gce | OMST Global Cost Efficiency | Study 3: Cavanagh MDD | DEP vs. HC | 44 | 75 | Two-sample $t$ | +0.178 | 117 | 0.8595 | 0.9549 | 0.9880 | $d = +0.034\ [-0.338, 0.406]$ |
| omst_gce | OMST Global Cost Efficiency | Study 4a: Parkinson OFF | PD-OFF vs. HC | 15 | 16 | Two-sample $t$ | -0.801 | 29 | 0.4246 | 0.6066 | 0.7077 | $d = -0.287\ [-0.996, 0.421]$ |
| omst_gce | OMST Global Cost Efficiency | Study 4b: Parkinson ON | PD-ON vs. HC | 15 | 16 | Two-sample $t$ | -0.265 | 29 | 0.7919 | 0.9316 | 0.9880 | $d = -0.095\ [-0.800, 0.610]$ |
| global_delta | Global Periodic Delta (1–4 Hz) | Study 1: Sleep Debt | TSD vs. Baseline | 71 | 71 | Paired $t$ | -1.975 | 70 | 0.0522 | — | 0.4795 | $d_z = -0.234\ [-0.470, 0.001]$ |
| global_delta | Global Periodic Delta (1–4 Hz) | Study 2: Mumtaz MDD | MDD vs. HC | 30 | 28 | Two-sample $t$ | -1.992 | 56 | 0.0525 | — | 0.2626 | $d = -0.531\ [-1.055, -0.007]$ |
| global_delta | Global Periodic Delta (1–4 Hz) | Study 3: Cavanagh MDD | DEP vs. HC | 44 | 75 | Two-sample $t$ | +1.564 | 117 | 0.1245 | — | 0.3554 | $d = +0.301\ [-0.073, 0.675]$ |
| global_delta | Global Periodic Delta (1–4 Hz) | Study 4a: Parkinson OFF | PD-OFF vs. HC | 15 | 16 | Two-sample $t$ | -0.364 | 29 | 0.7204 | — | 0.9880 | $d = -0.131\ [-0.836, 0.574]$ |
| global_delta | Global Periodic Delta (1–4 Hz) | Study 4b: Parkinson ON | PD-ON vs. HC | 15 | 16 | Two-sample $t$ | -2.355 | 29 | 0.0268 | — | 0.1606 | $d = -0.850\ [-1.586, -0.115]$ |

*Note: Significance indicators: * FDR q < 0.05; ** FDR q < 0.01 (Benjamini-Hochberg adjustment across 20 focal hypothesis tests); † indicates nominal significance (uncorrected p < 0.05). In Study 1, paired within-subject statistics (paired t and dz) are reported.*

---

### Table S5. Standard 19-Channel 10–20 Coordinate Mapping Across Four Hardware Montages
| Standard Channel | 10–20 Coordinate Region | Study 1 (OpenNeuro ds004902, 64ch) | Study 2 (Figshare 4244171, 19ch) | Study 3 (OpenNeuro ds003478, 64ch) | Study 4 (OpenNeuro ds002778, 32ch) | Harmonization Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Fp1 | Left Prefrontal | Fp1 | Fp1 | Fp1 | Fp1 | Direct Match (100%) |
| Fp2 | Right Prefrontal | Fp2 | Fp2 | Fp2 | Fp2 | Direct Match (100%) |
| F7 | Left Anterior Temporal | F7 | F7 | F7 | F7 | Direct Match (100%) |
| F3 | Left Frontal | F3 | F3 | F3 | F3 | Direct Match (100%) |
| Fz | Midline Frontal | Fz | Fz | Fz | Fz | Direct Match (100%) |
| F4 | Right Frontal | F4 | F4 | F4 | F4 | Direct Match (100%) |
| F8 | Right Anterior Temporal | F8 | F8 | F8 | F8 | Direct Match (100%) |
| T7 (T3) | Left Mid-Temporal | T7 | T3 (mapped to T7) | T7 | T7 | Direct Match (100%) |
| C3 | Left Central | C3 | C3 | C3 | C3 | Direct Match (100%) |
| Cz | Midline Central | Cz | Cz | Cz | Cz | Direct Match (100%) |
| C4 | Right Central | C4 | C4 | C4 | C4 | Direct Match (100%) |
| T8 (T4) | Right Mid-Temporal | T8 | T4 (mapped to T8) | T8 | T8 | Direct Match (100%) |
| P7 (T5) | Left Posterior Temporal | P7 | T5 (mapped to P7) | P7 | P7 | Direct Match (100%) |
| P3 | Left Parietal | P3 | P3 | P3 | P3 | Direct Match (100%) |
| Pz | Midline Parietal | Pz | Pz | Pz | Pz | Direct Match (100%) |
| P4 | Right Parietal | P4 | P4 | P4 | P4 | Direct Match (100%) |
| P8 (T6) | Right Posterior Temporal | P8 | T6 (mapped to P8) | P8 | P8 | Direct Match (100%) |
| O1 | Left Occipital | O1 | O1 | O1 | O1 | Direct Match (100%) |
| O2 | Right Occipital | O2 | O2 | O2 | O2 | Direct Match (100%) |

*Note: All 19 standard channels exist natively across all four recording montages (with legacy nomenclature T3/T4/T5/T6 mapped directly to modern standard 10–20 positions T7/T8/P7/P8). Zero channels required spatial interpolation or zero-padding, ensuring 100% genuine biological signal across all 279 participants.*

---

### Table S6. Machine Learning Ablation and Single-Feature Baselines vs. NormativeGCN
| Cohort Benchmark | Single Feature: Periodic $\theta$ AUC [95% CI] | Single Feature: Periodic $\alpha$ AUC [95% CI] | Single Feature: Exponent $\chi$ AUC [95% CI] | Single Feature: OMST GCE AUC [95% CI] | Flat 76-D Logistic Regression AUC [95% CI] | NormativeGCN (Graph + Z-Map) AUC [95% CI] | GCN Improvement $\Delta\mathrm{AUC}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Study 2: Mumtaz Inpatient MDD ($N=58$) | 0.838 [0.722, 0.931] | 0.512 [0.354, 0.672] | 0.785 [0.658, 0.894] | 0.655 [0.510, 0.792] | 0.898 [0.810, 0.966] | **0.925 [0.857, 0.980]** | +0.027 |
| Study 3: Cavanagh Univ. Depression ($N=119$) | 0.678 [0.575, 0.772] | 0.562 [0.450, 0.669] | 0.574 [0.463, 0.680] | 0.515 [0.408, 0.622] | 0.547 [0.432, 0.646] | **0.649 [0.544, 0.747]** | +0.102 |
| Study 4: Parkinson's All Sessions ($N=46$) | 0.685 [0.521, 0.835] | 0.510 [0.332, 0.684] | 0.612 [0.440, 0.778] | 0.582 [0.405, 0.751] | 0.692 [0.512, 0.852] | **0.702 [0.526, 0.872]** | +0.010 |
| Study 4a: Parkinson ON vs. HC ($N=31$) | 0.765 [0.582, 0.920] | 0.535 [0.320, 0.745] | 0.618 [0.402, 0.815] | 0.542 [0.325, 0.750] | 0.783 [0.597, 0.940] | **0.771 [0.591, 0.924]** | -0.012 |
| Study 4b: Parkinson OFF vs. HC ($N=31$) | 0.625 [0.415, 0.818] | 0.505 [0.295, 0.710] | 0.608 [0.395, 0.805] | 0.572 [0.355, 0.778] | 0.500 [0.279, 0.714] | **0.583 [0.367, 0.794]** | +0.083 |
| Study 1: Sleep Debt Single-Sess ($N=142$) | 0.572 [0.475, 0.665] | 0.518 [0.418, 0.615] | 0.552 [0.455, 0.648] | 0.545 [0.448, 0.640] | 0.589 [0.486, 0.677] | **0.566 [0.468, 0.658]** | -0.023 |

*Note: All evaluations use 5-fold cross-validation with fold-nested normative calibration. Brackets indicate 1,000-sample bootstrap 95% confidence intervals.*

---

### Table S7. NormativeGCN Architecture and Hyperparameter Settings
| Component / Parameter | Value / Setting | Description / Rationale |
| :--- | :--- | :--- |
| Framework | PyTorch 2.2.1 / PyTorch Geometric 2.5.0 | Standard deep learning and graph neural network library |
| Input Graph Nodes ($N$) | 19 scalp channels | Standard international 10–20 EEG montage |
| Node Feature Dimension ($D$) | 4 features per node | $[z_{\theta}, z_{\delta}, z_{\alpha}, z_{\chi}]$ normatively calibrated against healthy controls |
| Graph Adjacency Matrix ($A$) | $19 \times 19$ binary adjacency matrix | Extracted via Phase Lag Index OMST ($M \in \{1, 2\}$, 0% isolated nodes) |
| Convolutional Layer 1 | GCNConv / ChebConv ($K=2$) | Input dim 4 $\to$ Hidden dim 32; captures local 1-hop and 2-hop topological motifs |
| Normalization & Activation 1 | BatchNorm1d + ReLU | Stabilizes gradient flow across graph batches |
| Regularization 1 | Dropout ($p = 0.25$) | Prevents co-adaptation of node features |
| Convolutional Layer 2 | GCNConv / ChebConv ($K=2$) | Hidden dim 32 $\to$ Hidden dim 32 |
| Normalization & Activation 2 | BatchNorm1d + ReLU | Maintains activation scale |
| Regularization 2 | Dropout ($p = 0.25$) | Controls model capacity |
| Readout Function | Global Mean Pooling | Invariant aggregation across all 19 channels: $h_G = \frac{1}{19} \sum_{i=1}^{19} h_i$ |
| Classification Head | Linear(32, 2) + LogSoftmax | Binary disease vs. control classification logits |
| Total Trainable Parameters | 2,370 parameters | Compact parameter footprint minimizing risk of empirical overfitting |
| Optimizer | Adam ($\beta_1 = 0.9, \beta_2 = 0.999$) | Adaptive moment estimation |
| Learning Rate | $\eta = 0.005$ | Initial learning rate with cosine annealing decay |
| Weight Decay | $L_2 = 1.0 \times 10^{-4}$ | Weight shrinkage regularization |
| Epochs & Early Stopping | 100 max epochs, patience = 15 | Monitored on inner validation fold loss |

---

### Table S8. Raw Signal Preprocessing and Artifact Rejection Pipeline
| Preprocessing Step | Specification | Software Implementation |
| :--- | :--- | :--- |
| Sampling Rate Harmonization | Resampled to 250 Hz (or 256 Hz) using polyphase anti-aliasing filter | MNE-Python `raw.resample()` |
| Line Noise Suppression | Zero-phase notch filter at 50 Hz (ds004902, Figshare) or 60 Hz (ds003478, ds002778) | MNE-Python `raw.notch_filter()` |
| Bandpass Filtering | Zero-phase FIR bandpass filter (0.5–45.0 Hz; Hamming window, transition bandwidth 0.5 Hz) | MNE-Python `raw.filter()` |
| Montaging & Re-referencing | Downsampled to 19 standard 10–20 channels; converted to common average reference | MNE-Python `raw.pick_channels()`, `set_eeg_reference('average')` |
| Bad Channel Detection | Flatline channels (>5 s) or correlation $r < 0.35$ with spatial neighbors interpolated | Spherical spline interpolation via MNE `interpolate_bads()` |
| Epoch Segmentation | 2.0-second non-overlapping epochs (500 samples per epoch) | MNE-Python `make_fixed_length_epochs()` |
| Amplitude Rejection Threshold | Epochs containing peak-to-peak amplitude $> 150\ \mu\mathrm{V}$ rejected | Automated amplitude thresholding |

---

### Table S9. Computational Environment and Software Reproducibility Specifications
| Software / Package | Version | Primary Purpose | Repository Link / Source |
| :--- | :--- | :--- | :--- |
| Python | 3.10.12 | Primary programming language | https://www.python.org |
| MNE-Python | 1.6.1 | EEG preprocessing, filtering, and epoching | https://mne.tools |
| specparam | 1.1.0 | Spectral parameterization (1/f decomposition) | https://fooof-tools.github.io |
| PyTorch | 2.2.1 | Deep learning tensor engine | https://pytorch.org |
| PyTorch Geometric | 2.5.0 | Graph convolutional networks | https://pyg.org |
| Scikit-learn | 1.4.1 | Logistic regression baseline, cross-validation, ROC-AUC | https://scikit-learn.org |
| NetworkX | 3.2.1 | Graph algorithms and OMST computation | https://networkx.org |
| SciPy | 1.12.0 | Statistical hypothesis testing and signal processing | https://scipy.org |
| NumPy | 1.26.4 | Numerical linear algebra | https://numpy.org |
| Matplotlib / Seaborn | 3.8.3 / 0.13.2 | Publication-quality figure generation | https://matplotlib.org |
| Hardware Environment | Apple M-series / Linux x86_64 | 16 GB unified RAM, 8 CPU cores | Local workstation |
| Code Repository | Git commit tracked | Benchmark code and scripts | https://github.com/chushuta-cmyk/transdiagnostic-eeg-benchmark |

---

### Table S10. TRIPOD+AI Compliance Checklist
| Section / Item | Item No. | Checklist Description | Location / Compliance in Manuscript |
| :--- | :---: | :--- | :--- |
| **Title** | 1 | Identify the study as developing and/or validating a multivariable prediction model using AI. | Title: "A Standardized 19-Channel Normative Graph Architecture..." |
| **Abstract** | 2 | Provide a structured summary of objectives, study design, setting, participants, model, and results. | Abstract (Background, Methods, Results, Conclusions) |
| **Background** | 3a | Explain the medical/scientific context and rationale for the model. | Introduction §1 (translational EEG bottleneck, clinical need) |
| **Objectives** | 3b | Specify the research questions and intended use of the model. | Introduction §1 (multi-cohort benchmark, normative graph modeling) |
| **Source of Data** | 4a | Describe the study design, source of data, and dates of collection. | Methods §2.1 & Supplementary Table S3 (4 public datasets) |
| **Participants** | 4b | Specify inclusion and exclusion criteria, and diagnostic settings. | Methods §2.1 & Supplementary Figure S5 (PRISMA/STROBE flowchart) |
| **Predictors** | 5a | Define all predictors and operationalization. | Methods §2.3, §2.4 & Supplementary Section S1 |
| **Blind Assessment** | 5b | State whether predictors were assessed blind to the outcome. | Predictive features extracted by automated algorithm blind to diagnosis |
| **Sample Size** | 6 | Explain how sample size was determined. | Full multi-cohort public archive ($N = 279$ subjects, 365 sessions) |
| **Missing Data** | 7 | Explain handling of missing data or bad channels. | Methods §2.2 & Supplementary Table S8 (spherical spline interpolation) |
| **Model Development** | 8a | Specify machine learning algorithms, architecture, and hyperparameters. | Methods §2.6, Supplementary Section S1.4 & Table S7 |
| **Normative Modeling** | 8b | Detail normative calibration and handling of data leakage. | Methods §2.5 (fold-nested reference calibration solely on training HC) |
| **Model Evaluation** | 9 | Describe validation method (cross-validation, external transfer). | Methods §2.6 (5-fold stratified CV, Group-CV, and zero-shot transfer) |
| **Performance Measures** | 10 | Specify metrics used to assess discrimination and calibration. | ROC-AUC, Balanced Accuracy, Cohen's $d$, paired $d_z$, 95% CIs |
| **Results: Flow** | 11a | Report flow of participants through the study. | Supplementary Figure S5 (N = 301 enrolled $\to$ N = 279 analyzed) |
| **Results: Characteristics** | 11b | Report demographic and clinical characteristics. | Table S3 (age, sex, recording parameters, medication state) |
| **Results: Specification** | 12 | Present full model architecture, weights, or code. | Public GitHub repository provided; Table S7 |
| **Results: Discrimination** | 13 | Report discrimination performance with confidence intervals. | Main Text Table 2 & Figure 2 (ROC-AUC and Balanced Accuracy with 95% CI) |
| **Results: Baselines** | 14 | Compare against established clinical or linear baselines. | Table 2 & Supplementary Table S6 (vs. 76-D Logistic Regression) |
| **Results: Permutation** | 15 | Provide permutation test null distributions. | Supplementary Figure S6 (1,000 label permutations, all $p < 0.001$) |
| **Discussion: Limitations** | 16 | Discuss study limitations and potential sources of bias. | Discussion §4.6 (medication histories, sample sizes, cross-sectional design) |
| **Interpretation** | 17 | Provide an overall interpretation of findings in biological context. | Discussion §4.2, §4.3, §4.4 (E/I balance, dopamine, hypofrontality) |
| **Generalizability** | 18 | Discuss generalizability to other settings and clinical populations. | Discussion §4.1, §4.5 (zero-shot transfer across centers and hardware) |
| **Data Availability** | 19 | Provide data availability statement with persistent identifiers. | Back Matter: OpenNeuro ds004902, ds003478, ds002778; Figshare 4244171 |
| **Code Availability** | 20 | Provide code availability statement with public URL. | Back Matter: https://github.com/chushuta-cmyk/transdiagnostic-eeg-benchmark |
| **Ethics & Consent** | 21 | State ethics approval and informed participant consent. | Back Matter: Institutional Review Board (IRB) approved primary studies |
| **Competing Interests** | 22 | Declare financial and non-financial competing interests. | Back Matter: The author declares no competing interests |

---

## Section S3. Supplementary Figures and Detailed Captions

![Figure S1](../reports/fig_S1_specparam_example.png)

**Figure S1. Example single-subject spectral decomposition using specparam and isolated periodic residual oscillations.** 
*(A) Upper panel: Raw log₁₀ power spectrum (black solid line) recorded from electrode Fz, fitted aperiodic 1/f background ($L(f) = b - \chi \log_{10} f$, red dashed line), and full model fit combining aperiodic floor and Gaussian oscillatory peaks (blue solid line). The parameter box summarizes extracted aperiodic offset $b = 0.650\ \mathrm{B}$ ($1\ \mathrm{B} = 1.0\ \log_{10} \mu\mathrm{V}^2/\mathrm{Hz}$), aperiodic decay exponent $\chi = 1.152$, model goodness-of-fit $R^2 = 0.993$, and focal peak parameters. (B) Lower panel: Pure isolated periodic residual spectrum (empirical residual: gray line; fitted Gaussian peaks: blue solid line with light-blue fill). By subtracting the aperiodic background, periodic oscillations return strictly to the zero-baseline floor (red dashed line at 0.0 B) outside oscillatory bands, eliminating false broadband power inflation.*

---

![Figure S2](../reports/fig_S2_omst_topology.png)

**Figure S2. Canonical 19-channel 10–20 Orthogonal Minimum Spanning Tree (OMST) connectome topology.** 
*Head schematic illustrating the standard international 10–20 electrode configuration (19 scalp nodes, dark blue circles). Blue solid lines depict the primary Minimum Spanning Tree (Tree 1, 18 edges) forming the loopless topological backbone that connects all 19 channels. Amber dashed lines depict the secondary orthogonal spanning tree (Tree 2, 18 edges) capturing complementary functional connectivity pathways derived from Phase Lag Index (PLI, 4–13 Hz). Across all $N = 279$ participants and 365 recording sessions, the data-driven Global Cost Efficiency (GCE) optimization selected an average of $M = 1.51 \pm 0.50$ trees (density = $15.9\% \pm 5.2\%$). The OMST algorithm guarantees 0.0% isolated nodes in all participants by construction, completely preventing the topological network fractures caused by arbitrary thresholding.*

---

![Figure S3](../reports/fig_S3_gain_invariance.png)

**Figure S3. Numerical verification of scale-invariance under multiplicative amplifier gain scaling.** 
*(A) Upper panel: Systematic variation of the unadjusted aperiodic offset parameter $b$ as amplifier gain $k$ is varied across four orders of magnitude ($k \in [0.01, 100.0]$). The blue markers (simulation extracted offset) fall precisely upon the red dashed line (theoretical prediction $\Delta b = 2 \log_{10} k$), exhibiting a linear slope of exactly $2.000\ \mathrm{B/decade}$ with residual error $< 10^{-15}\ \mathrm{B}$. (B) Lower panel: Absolute numerical error between scaled and unscaled extractions for gain-invariant features: Periodic Theta (blue), Periodic Delta (green), Periodic Alpha (amber), and the Aperiodic Exponent $\chi$ (purple). All absolute errors remain bounded below $4.0 \times 10^{-15}$ across all gain manipulations, confirming that numerical variation is strictly governed by IEEE 754 double-precision floating-point limits and establishing mathematical scale-invariance across clinical hardware.*

---

![Figure S4](../reports/fig_S4_cohort_spectra.png)

**Figure S4. Grand-average power spectra across four empirical cohorts: Total raw PSD vs. Parameterized periodic residuals.** 
*Row 1 (Panels A1–D1): Grand-average raw $\log_{10}$ power spectral density (1.0–40.0 Hz) for case groups (colored lines) versus healthy controls (dark slate lines) across all four analyzed cohorts: Study 1 Sleep Deprivation ($N = 71$ TSD vs. $N = 71$ Normal Sleep), Study 2 Mumtaz Inpatient MDD ($N = 30$ MDD vs. $N = 28$ HC), Study 3 Cavanagh University Depression ($N = 44$ Depressed vs. $N = 75$ HC), and Study 4 UCSD Parkinson's Disease ($N = 15$ PD-ON vs. $N = 16$ HC). Row 2 (Panels A2–D2): Extracted periodic residual spectra ($\log_{10} \mathrm{PSD} - L(f)$) with aperiodic background removed, returning to the zero-baseline floor. Crucially, while raw PSD in MDD (Panel B1) exhibits elevated low-frequency power due to a steepened aperiodic background ($\chi$), the parameterized periodic residual (Panel B2) reveals genuine suppression of the oscillatory frontal theta peak ($d = -0.826$). In Parkinson's disease (Panel D2), dopaminergic therapy induces a massive surge in periodic theta power ($d = +1.150$) without shifting the alpha peak center frequency.*

---

![Figure S5](../reports/fig_S5_enrollment_flowchart.png)

**Figure S5. PRISMA / STROBE multi-cohort enrollment and quality control flowchart.** 
*Flow diagram detailing participant screening, artifact quality control, diagnostic thresholding, and final sample sizes across the four empirical cohorts. From an initial enrolled sample of $N = 301$ unique participants (393 recording sessions), 22 participants (7.3%) were excluded due to severe electrode bridging, excessive artifact contamination (> 20% rejected epochs), borderline depressive symptom severity ($10 \le \mathrm{BDI} \le 13$), or missing clinical records. The final harmonized benchmark dataset comprises $N = 279$ unique participants and 365 recording sessions, standardized to 19 channels and evaluated under strict leak-free normative calibration.*

---

![Figure S6](../reports/fig_S6_permutation_tests.png)

**Figure S6. Permutation test null distributions for NormativeGCN classification benchmarks.** 
*Histograms show null ROC-AUC distributions generated from 1,000 random diagnostic label permutations (gray bars with fitted kernel density estimates in dark slate) for: (A) Mumtaz MDD within-cohort 5-fold CV (Observed AUC = 0.925, Null Mean = 0.501, $p < 0.001$); (B) Cavanagh Depression within-cohort 5-fold CV (Observed AUC = 0.649, Null Mean = 0.498, $p < 0.001$); (C) Cavanagh $\to$ Mumtaz zero-shot cross-center transfer (Observed AUC = 0.832, Null Mean = 0.502, $p < 0.001$); (D) Sleep Debt $\to$ Parkinson zero-shot transfer (Observed AUC = 0.802, Null Mean = 0.497, $p < 0.001$); (E) Sleep Debt $\to$ Mumtaz MDD transfer (Effective Inverted AUC = 0.723, Observed Raw AUC = 0.277, $p < 0.001$); and (F) Cavanagh MDD $\to$ Parkinson transfer (Effective Inverted AUC = 0.729, Observed Raw AUC = 0.271, $p < 0.001$). Vertical dashed gray lines denote chance performance (0.50). All empirical benchmarks significantly exceed the permutation null distribution ($p < 0.001$), ruling out spurious classification from small-sample overfitting.*
