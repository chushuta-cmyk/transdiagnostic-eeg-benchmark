#!/usr/bin/env python3
"""
scripts/50_build_paper2_neuroinformatics_v6.py
----------------------------------------------
Builds the complete, Springer-Neuroinformatics-compliant v6 submission package:
  - 01_Cover_Letter_Neuroinformatics.(md, docx)
  - 02_Research_Highlights_Neuroinformatics.(md, docx)
  - 03_Title_Page_Neuroinformatics.(md, docx) [Official 'Original Article', deduplicated metadata]
  - 04_Main_Manuscript_Neuroinformatics.(md, docx) + manuscript/PAPER_2_NEUROINFORMATICS_v6.(md, docx)
      * Shortened, precise Title & Running Title
      * 150-250 word Abstract with zero undefined abbreviations
      * Fixed-width OOXML tables (Table 1, Table 2, split Table 3a/3b, split Table 4a/4b, clarified Table 5)
      * Springer Fig. 1..4 bold captions (no trailing punctuation, lowercase (a)-(d))
      * Clean single centered PAGE footer OOXML (no Footer style conflicts)
      * AI-assisted workflow disclosed directly inside Methods Section 2.7
      * Full v6 sensitivities: Study 7 Pz/CAR 18-ch & Spline, Study 4/6 participant-level 10x5 AUC,
        k=7 shared-HC-free & Holm TOST equivalence, Excl-Study-2 macro-AUCs, 4-model max-stat clinical
        regression permutations, fold-wise Spatial Cov Fréchet invariance, Study 2 181-EDF accounting,
        full 13,927-spectra fooof v1.1.1 check, and separated MIT code vs CC BY 4.0 data licensing.
  - 05_Online_Resource_1_Neuroinformatics.(md, docx, pdf) [Typeset PDF via Chrome + Pandoc]
  - 05b_Online_Resource_2_Benchmark_Tables.xlsx [Machine-readable workbook of key v6 CSVs]
  - 06_Declarations_CRediT_COI_Ethics.(md, docx)
  - 07_Suggested_Reviewers.(md, docx)
  - 08_Submission_Checklist_Neuroinformatics.md
"""

import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
RELEASE_DIR = ROOT / "release_artifacts"
MANUSCRIPT_DIR = ROOT / "manuscript"
SUBMISSION_DIR = ROOT / "submission_Neuroinformatics"
ARTIFACT_DIR = Path("/Users/tan/.gemini/antigravity/brain/874c84f2-3204-455c-a632-daaf2336a273")
CHROME_BIN = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

JOURNAL_NAME = "Neuroinformatics"
PUBLISHER_NAME = "Springer Nature"
EDITOR_NAME = "Prof. John Darrell Van Horn, Ph.D."
EDITOR_TITLE = "Editor-in-Chief, Neuroinformatics"
EDITOR_AFFIL = (
    "Department of Psychology and School of Data Science, "
    "University of Virginia, Charlottesville, VA, USA"
)

EXACT_TITLE = (
    "An Open Multi-Cohort EEG Benchmark for Spectral Parameterization, "
    "Sensor Graphs, and Reproducibility Audits"
)
SHORT_TITLE = "Open Multi-Cohort EEG Benchmarking Suite"
ARTICLE_TYPE = "Original Article"

AUTHOR_NAME = "Chushu Tan"
AFFIL_1 = "School of Sociology, China University of Political Science and Law, Beijing, China"
AFFIL_2 = "Department of Psychology, University at Buffalo, The State University of New York, Buffalo, NY 14260, USA"
CORRESPONDENCE = "Beijing, China; Email: chushuta@gmail.com; ORCID: 0009-0007-3093-0160"
GITHUB_URL = "https://github.com/chushuta-cmyk/transdiagnostic-eeg-benchmark"
RELEASE_TAG = "v6.0.0"
SUBMISSION_DATE = "November 19, 2026"

MAIN_MD = MANUSCRIPT_DIR / "PAPER_2_NEUROINFORMATICS_v6.md"
MAIN_DOCX = MANUSCRIPT_DIR / "PAPER_2_NEUROINFORMATICS_v6.docx"
MAIN_PDF = MANUSCRIPT_DIR / "PAPER_2_NEUROINFORMATICS_v6.pdf"
SUPP_MD = MANUSCRIPT_DIR / "PAPER_2_ONLINE_RESOURCE_1_v6.md"
SUPP_DOCX = MANUSCRIPT_DIR / "PAPER_2_ONLINE_RESOURCE_1_v6.docx"
SUPP_PDF = MANUSCRIPT_DIR / "PAPER_2_ONLINE_RESOURCE_1_v6.pdf"

REFS_ALPHA = [
    ("Anjum et al., 2024", "Anjum, M. F., Espinoza, A. I., Cole, R. C., Singh, A., May, P., Uc, E. Y., Dasgupta, S., & Narayanan, N. S. (2024). Resting-state EEG measures cognitive impairment in Parkinson's disease. *npj Parkinson's Disease*, 10, 6. https://doi.org/10.1038/s41531-023-00602-0"),
    ("Barachant et al., 2013", "Barachant, A., Bonnet, S., Congedo, M., & Jutten, C. (2013). Classification of covariance matrices using a Riemannian-based kernel for BCI applications. *Neurocomputing*, 112, 172–178. https://doi.org/10.1016/j.neucom.2012.12.039"),
    ("Bosboom et al., 2006", "Bosboom, J. L. W., Stoffers, D., Stam, C. J., van Dijk, B. W., Verbunt, J., Berendse, H. W., & Wolters, E. C. (2006). Resting state oscillatory brain dynamics in Parkinson's disease: An MEG study. *Clinical Neurophysiology*, 117(11), 2521–2531. https://doi.org/10.1016/j.clinph.2006.06.720"),
    ("Brookshire et al., 2024", "Brookshire, G., Kasper, J., Blauch, N. M., Wu, Y. C., Glatt, R., Merrill, D. A., Gerrol, S., Yoder, K. J., Quirk, C., & Lucero, C. (2024). Data leakage in deep learning studies of translational EEG. *Frontiers in Neuroscience*, 18, 1373515. https://doi.org/10.3389/fnins.2024.1373515"),
    ("Bullmore & Sporns, 2009", "Bullmore, E., & Sporns, O. (2009). Complex brain networks: Graph theoretical analysis of structural and functional systems. *Nature Reviews Neuroscience*, 10(3), 186–198. https://doi.org/10.1038/nrn2575"),
    ("Buzsáki, 2006", "Buzsáki, G. (2006). *Rhythms of the Brain*. Oxford University Press. https://doi.org/10.1093/acprof:oso/9780195301069.001.0001"),
    ("Buzsáki & Draguhn, 2004", "Buzsáki, G., & Draguhn, A. (2004). Neuronal oscillations in cortical networks. *Science*, 304(5679), 1926–1929. https://doi.org/10.1126/science.1099745"),
    ("Cavanagh et al., 2019", "Cavanagh, J. F., Bismark, A. W., Frank, M. J., & Allen, J. J. B. (2019). Multiple dissociations between comorbid depression and anxiety on reward and punishment processing: Evidence from computationally informed EEG. *Computational Psychiatry*, 3, 1–17. https://doi.org/10.1162/cpsy_a_00024 (Dataset: OpenNeuro ds003478, https://doi.org/10.18112/openneuro.ds003478.v1.1.0)"),
    ("Cavanagh & Frank, 2014", "Cavanagh, J. F., & Frank, M. J. (2014). Frontal theta as a mechanism for cognitive control. *Trends in Cognitive Sciences*, 18(8), 414–421. https://doi.org/10.1016/j.tics.2014.04.012"),
    ("Cavanagh et al., 2018", "Cavanagh, J. F., Kumar, P., Mueller, A. A., Richardson, S. P., & Mueen, A. (2018). Diminished EEG habituation to novel events effectively classifies Parkinson's patients. *Clinical Neurophysiology*, 129(2), 409–418. https://doi.org/10.1016/j.clinph.2017.11.023 (Dataset: OpenNeuro ds003490, https://doi.org/10.18112/openneuro.ds003490.v1.1.0)"),
    ("Cohen, 2014", "Cohen, M. X. (2014). *Analyzing Neural Time Series Data: Theory and Practice*. MIT Press."),
    ("Cole & Voytek, 2017", "Cole, S. R., & Voytek, B. (2017). Brain oscillations and the importance of waveform shape. *Trends in Cognitive Sciences*, 21(2), 137–149. https://doi.org/10.1016/j.tics.2016.12.008"),
    ("Colombo et al., 2019", "Colombo, M. A., Napolitani, M., Boly, M., Gosseries, O., Casarotto, S., Rosanova, M., Brichant, J.-F., Boveroux, P., Rex, S., Laureys, S., Massimini, M., Chieregato, A., & Sarasso, S. (2019). The spectral exponent of the resting EEG indexes the presence of consciousness during unresponsiveness induced by propofol, xenon, and ketamine. *NeuroImage*, 189, 631–644. https://doi.org/10.1016/j.neuroimage.2019.01.024"),
    ("Congedo et al., 2017", "Congedo, M., Barachant, A., & Bhatia, R. (2017). Riemannian geometry for EEG-based brain-computer interfaces: A primer and a review. *Brain-Computer Interfaces*, 4(3), 155–174. https://doi.org/10.1080/2326263X.2017.1297192"),
    ("Delorme & Makeig, 2004", "Delorme, A., & Makeig, S. (2004). EEGLAB: An open source toolbox for analysis of single-trial EEG dynamics including independent component analysis. *Journal of Neuroscience Methods*, 134(1), 9–21. https://doi.org/10.1016/j.jneumeth.2003.10.009"),
    ("DerSimonian & Laird, 1986", "DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials*, 7(3), 177–188. https://doi.org/10.1016/0197-2456(86)90046-2"),
    ("Dimitriadis et al., 2017a", "Dimitriadis, S. I., Antonakakis, M., Simos, P., Fletcher, J. M., & Papanicolaou, A. C. (2017a). Data-driven topological filtering based on orthogonal minimal spanning trees: Application to multigroup magnetoencephalography resting-state connectivity. *Brain Connectivity*, 7(10), 661–670. https://doi.org/10.1089/brain.2017.0512"),
    ("Dimitriadis et al., 2017b", "Dimitriadis, S. I., Salis, C., Tarnanas, I., & Linden, D. E. J. (2017b). Topological filtering of dynamic functional brain networks unfolds informative chronnectomics: A novel data-driven thresholding scheme based on orthogonal minimal spanning trees (OMSTs). *Frontiers in Neuroinformatics*, 11, 28. https://doi.org/10.3389/fninf.2017.00028"),
    ("Donoghue et al., 2020", "Donoghue, T., Haller, M., Peterson, E. J., Varma, P., Sebastian, P., Gao, R., Noto, T., Lara, A. H., Wallis, J. D., Knight, R. T., Shestyuk, A., & Voytek, B. (2020). Parameterizing neural power spectra into periodic and aperiodic components. *Nature Neuroscience*, 23(12), 1655–1665. https://doi.org/10.1038/s41593-020-00744-x"),
    ("Donoghue et al., 2022", "Donoghue, T., Schaworonkow, N., & Voytek, B. (2022). Methodological considerations for studying neural oscillations. *European Journal of Neuroscience*, 55(11–12), 3502–3527. https://doi.org/10.1111/ejn.15361"),
    ("Finelli et al., 2000", "Finelli, L. A., Baumann, H., Borbély, A. A., & Achermann, P. (2000). Dual electroencephalogram markers of human sleep homeostasis: Correlation between theta activity in waking and slow-wave activity in sleep. *Neuroscience*, 101(3), 523–529. https://doi.org/10.1016/s0306-4522(00)00409-7"),
    ("Fries, 2005", "Fries, P. (2005). A mechanism for cognitive dynamics: Neuronal communication through coherence. *Trends in Cognitive Sciences*, 9(10), 474–480. https://doi.org/10.1016/j.tics.2005.08.011"),
    ("Gao et al., 2017", "Gao, R., Peterson, E. J., & Voytek, B. (2017). Inferring synaptic excitation/inhibition balance from field potentials. *NeuroImage*, 158, 70–78. https://doi.org/10.1016/j.neuroimage.2017.06.078"),
    ("Gemein et al., 2020", "Gemein, L. A. W., Schirrmeister, R. T., Chrabąszcz, P., Wilson, D., Boedecker, J., Schulze-Bonhage, A., Hutter, F., & Ball, T. (2020). Machine-learning-based diagnostics of EEG pathology. *NeuroImage*, 220, 117021. https://doi.org/10.1016/j.neuroimage.2020.117021"),
    ("George et al., 2013", "George, J. S., Strunk, J., Mak-McCully, R., Houser, M., Poizner, H., & Aron, A. R. (2013). Dopaminergic therapy in Parkinson's disease decreases cortical beta band coherence in the resting state and increases beta coherence during motor control. *NeuroImage: Clinical*, 3, 261–270. https://doi.org/10.1016/j.nicl.2013.07.013"),
    ("Gerster et al., 2022", "Gerster, M., Waterstraat, G., Litvak, V., Lehnertz, K., Schnitzler, A., Florin, E., Curio, G., & Nikulin, V. (2022). Separating neural oscillations from aperiodic 1/f activity: Challenges and recommendations. *Neuroinformatics*, 20(4), 991–1012. https://doi.org/10.1007/s12021-022-09581-8"),
    ("Gramfort et al., 2013", "Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D., Brodbeck, C., Goj, R., Jas, M., Brooks, T., Parkkonen, L., & Hämäläinen, M. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267. https://doi.org/10.3389/fnins.2013.00267"),
    ("Hartung & Knapp, 2001", "Hartung, J., & Knapp, G. (2001). On tests of the overall treatment effect in meta-analysis with normally distributed responses. *Statistics in Medicine*, 20(12), 1771–1782. https://doi.org/10.1002/sim.791"),
    ("Holm, 1979", "Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70."),
    ("Jaworska et al., 2012", "Jaworska, N., Blier, P., Fusee, W., & Knott, V. (2012). Alpha power, alpha asymmetry and anterior cingulate cortex activity in depressed males and females. *Journal of Psychiatric Research*, 46(11), 1483–1491. https://doi.org/10.1016/j.jpsychires.2012.08.003"),
    ("Kapur et al., 2012", "Kapur, S., Phillips, A. G., & Insel, T. R. (2012). Why has it taken so long for biological psychiatry to develop clinical tests and what to do about it? *Molecular Psychiatry*, 17(12), 1174–1179. https://doi.org/10.1038/mp.2012.105"),
    ("Kipf & Welling, 2017", "Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *Proceedings of the 5th International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1609.02907"),
    ("Knott et al., 2001", "Knott, V., Mahoney, C., Kennedy, S., & Evans, K. (2001). EEG power, frequency, asymmetry and coherence in male depression. *Psychiatry Research: Neuroimaging*, 106(2), 123–140. https://doi.org/10.1016/s0925-4927(00)00080-9"),
    ("Lakens, 2017", "Lakens, D. (2017). Equivalence tests: A practical primer for t tests, correlations, and meta-analyses. *Social Psychological and Personality Science*, 8(4), 355–362. https://doi.org/10.1177/1948550617697177"),
    ("Lawhern et al., 2018", "Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering*, 15(5), 056013. https://doi.org/10.1088/1741-2552/aace8c"),
    ("Little et al., 2017", "Little, M. A., Varoquaux, G., Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. (2017). Using and understanding cross-validation strategies: Perspectives on Saeb et al. *GigaScience*, 6(5), gix020. https://doi.org/10.1093/gigascience/gix020"),
    ("Lopes da Silva, 2013", "Lopes da Silva, F. H. (2013). EEG and MEG: Relevance to neuroscience. *Neuron*, 80(5), 1112–1128. https://doi.org/10.1016/j.neuron.2013.10.017"),
    ("Mahato & Paul, 2020", "Mahato, S., & Paul, S. (2020). Classification of depression patients and normal subjects based on electroencephalogram (EEG) signal using alpha power and theta asymmetry. *Journal of Medical Systems*, 44(1), 28. https://doi.org/10.1007/s10916-019-1486-z"),
    ("Miltiadous et al., 2023", "Miltiadous, A., Tzimourta, K. D., Afrantou, T., Ioannidis, P., Grigoriadis, N., Tsalikakis, D. G., Angelidis, P., Tsipouras, M. G., Glavas, E., Giannakeas, N., & Tzallas, A. T. (2023). A dataset of scalp EEG recordings of Alzheimer's disease, frontotemporal dementia and healthy subjects from routine EEG. *Data*, 8(6), 95. https://doi.org/10.3390/data8060095 (Dataset: OpenNeuro ds004504, https://doi.org/10.18112/openneuro.ds004504.v1.0.6)"),
    ("Mitchell et al., 2008", "Mitchell, D. J., McNaughton, N., Flanagan, D., & Kirk, I. J. (2008). Frontal-midline theta from the perspective of hippocampal 'theta'. *Progress in Neurobiology*, 86(3), 156–185. https://doi.org/10.1016/j.pneurobio.2008.09.005"),
    ("Mumtaz et al., 2017", "Mumtaz, W., Xia, L., Ali, S. S. A., Yasin, M. A. M., Hussain, M., & Malik, A. S. (2017). Electroencephalogram (EEG)-based computer-aided technique to diagnose major depressive disorder (MDD). *Biomedical Signal Processing and Control*, 31, 108–115. https://doi.org/10.1016/j.bspc.2016.07.006 (Dataset: Figshare 4244171, https://doi.org/10.6084/m9.figshare.4244171.v5)"),
    ("Nadeau & Bengio, 2003", "Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52(3), 239–281. https://doi.org/10.1023/A:1024068626366"),
    ("Newson & Thiagarajan, 2019", "Newson, J. J., & Thiagarajan, T. C. (2019). EEG frequency bands in psychiatric disorders: A review of resting state studies. *Frontiers in Human Neuroscience*, 12, 521. https://doi.org/10.3389/fnhum.2018.00521"),
    ("Nunez & Srinivasan, 2006", "Nunez, P. L., & Srinivasan, R. (2006). *Electric Fields of the Brain: The Neurophysics of EEG* (2nd ed.). Oxford University Press."),
    ("Olbrich & Arns, 2013", "Olbrich, S., & Arns, M. (2013). EEG biomarkers in major depressive disorder: Discriminative power and prediction of treatment response. *International Review of Psychiatry*, 25(5), 604–618. https://doi.org/10.3109/09540261.2013.816269"),
    ("Rockhill et al., 2020", "Rockhill, A. P., Jackson, N., George, J., Aron, A., & Swann, N. C. (2020). UC San Diego Resting State EEG Data from Patients with Parkinson's Disease. *OpenNeuro*, ds002778. https://doi.org/10.18112/openneuro.ds002778.v1.0.5"),
    ("Roy et al., 2019", "Roy, Y., Banville, H., Albuquerque, I., Gramfort, A., Falk, T. H., & Faubert, J. (2019). Deep learning-based electroencephalography analysis: A systematic review. *Journal of Neural Engineering*, 16(5), 051001. https://doi.org/10.1088/1741-2552/ab260c"),
    ("Rubinov & Sporns, 2010", "Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity: Uses and interpretations. *NeuroImage*, 52(3), 1059–1069. https://doi.org/10.1016/j.neuroimage.2009.10.003"),
    ("Rutherford et al., 2022", "Rutherford, S., Kia, S. M., Wolfers, T., Fraza, C., Zabihi, M., Dinga, R., Berthet, P., Worker, A., Verdi, S., Ruhe, H. G., Beckmann, C. F., & Marquand, A. F. (2022). The normative modeling framework for computational psychiatry. *Nature Protocols*, 17(7), 1711–1734. https://doi.org/10.1038/s41596-022-00696-5"),
    ("Saeb et al., 2017", "Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. (2017). The need to approximate the use-case in clinical machine learning. *GigaScience*, 6(5), gix019. https://doi.org/10.1093/gigascience/gix019"),
    ("Saeedi et al., 2021", "Saeedi, A., Saeedi, M., Maghsoudi, A., & Shalbaf, A. (2021). Major depressive disorder diagnosis based on effective connectivity in EEG signals: A convolutional neural network and long short-term memory approach. *Cognitive Neurodynamics*, 15(2), 239–252. https://doi.org/10.1007/s11571-020-09619-0"),
    ("Scheinost et al., 2019", "Scheinost, D., Noble, S., Horien, C., Greene, A. S., Lake, E. M. R., Salehi, M., Gao, S., Shen, X., O'Connor, D., Barron, D. S., Yip, S. W., Rosenberg, M. D., & Constable, R. T. (2019). Ten simple rules for predictive modeling of individual differences in neuroimaging. *NeuroImage*, 193, 35–45. https://doi.org/10.1016/j.neuroimage.2019.02.057"),
    ("Schirrmeister et al., 2017", "Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F., Burgard, W., & Ball, T. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391–5420. https://doi.org/10.1002/hbm.23730"),
    ("Schuirmann, 1987", "Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), 657–680. https://doi.org/10.1007/BF01068419"),
    ("Singh et al., 2023", "Singh, A., Cole, R., Espinoza, A., Cavanagh, J., & Narayanan, N. (2023). Rest eyes open (Version v1.0.0) [Data set]. *OpenNeuro*, ds004584. https://doi.org/10.18112/openneuro.ds004584.v1.0.0"),
    ("Stam, 2010", "Stam, C. J. (2010). Use of magnetoencephalography (MEG) to study functional brain networks in neurodegenerative disorders. *Journal of the Neurological Sciences*, 289(1–2), 128–134. https://doi.org/10.1016/j.jns.2009.08.028"),
    ("Stam, 2014", "Stam, C. J. (2014). Modern network science of neurological disorders. *Nature Reviews Neuroscience*, 15(10), 683–695. https://doi.org/10.1038/nrn3801"),
    ("Stam et al., 2007", "Stam, C. J., Nolte, G., & Daffertshofer, A. (2007). Phase lag index: Assessment of functional connectivity from multi-channel EEG and MEG with diminished bias from common sources. *Human Brain Mapping*, 28(11), 1178–1193. https://doi.org/10.1002/hbm.20346"),
    ("Stoffers et al., 2007", "Stoffers, D., Bosboom, J. L. W., Deijen, J. B., Wolters, E. C., Berendse, H. W., & Stam, C. J. (2007). Slowing of oscillatory brain activity is a stable characteristic of Parkinson's disease without dementia. *Brain*, 130(7), 1847–1860. https://doi.org/10.1093/brain/awm034"),
    ("Varoquaux, 2018", "Varoquaux, G. (2018). Cross-validation failure: Small sample sizes lead to large error bars. *NeuroImage*, 180, 68–77. https://doi.org/10.1016/j.neuroimage.2017.06.061"),
    ("Vinck et al., 2011", "Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F., & Pennartz, C. M. A. (2011). An improved index of phase-synchronization for electrophysiological data in the presence of volume-conduction, noise and sample-size bias. *NeuroImage*, 55(4), 1548–1565. https://doi.org/10.1016/j.neuroimage.2011.01.055"),
    ("von Hippel, 2015", "von Hippel, P. T. (2015). The heterogeneity statistic I2 can be biased in small meta-analyses. *BMC Medical Research Methodology*, 15, 35. https://doi.org/10.1186/s12874-015-0024-z"),
    ("Voytek et al., 2015", "Voytek, B., Kramer, M. A., Case, J., Lepage, K. Q., Tempesta, Z. R., Knight, R. T., & Gazzaley, A. (2015). Age-related changes in 1/f neural electrophysiological noise. *Journal of Neuroscience*, 35(38), 13257–13265. https://doi.org/10.1523/JNEUROSCI.2332-14.2015"),
    ("Wolfers et al., 2018", "Wolfers, T., Doan, N. T., Kaufmann, T., Alnæs, D., Moberget, T., Agartz, I., Buitelaar, J. K., Ueland, T., Melle, I., Franke, B., Andreassen, O. A., Beckmann, C. F., Westlye, L. T., & Marquand, A. F. (2018). Mapping the heterogeneous phenotype of schizophrenia and bipolar disorder using normative models. *JAMA Psychiatry*, 75(11), 1146–1155. https://doi.org/10.1001/jamapsychiatry.2018.2467"),
    ("Xiang et al., 2024", "Xiang, C., Fan, X., Bai, D., Lv, K., & Lei, X. (2024). A resting-state EEG dataset for sleep deprivation. *Scientific Data*, 11, 427. https://doi.org/10.1038/s41597-024-03268-2 (Dataset: OpenNeuro ds004902, https://doi.org/10.18112/openneuro.ds004902.v1.1.0)"),
]


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "—"
    if p < 1e-4:
        exp = int(np.floor(np.log10(max(p, 1e-300))))
        mant = p / (10.0 ** exp)
        return f"{mant:.2f} × 10^{exp}"
    return f"{p:.4f}"


def fmt_sig(p: float) -> str:
    if pd.isna(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def strip_md_bold(s: str) -> str:
    return str(s).replace("**", "").strip()


# ==============================================================================
# OOXML Document Styling Helpers (Fixed-Width Tables + Clean Single Page Footer)
# ==============================================================================
def add_continuous_line_numbering(doc):
    for section in doc.sections:
        sectPr = section._sectPr
        lnNumType = OxmlElement("w:lnNumType")
        lnNumType.set(qn("w:countBy"), "1")
        lnNumType.set(qn("w:restart"), "continuous")
        sectPr.append(lnNumType)


def _compute_col_widths_dxa(col_count: int, total_dxa: int = 9020) -> list[int]:
    """Allocate generous width to Column 1 (Study/Contrast) so row labels never stack on standard A4 (6.27 in / 9020 dxa body)."""
    if col_count <= 1:
        return [total_dxa]
    if col_count == 2:
        return [int(total_dxa * 0.36), total_dxa - int(total_dxa * 0.36)]
    if col_count <= 6:
        c0 = 2600
    elif col_count <= 8:
        c0 = 2300
    else:
        c0 = 2100
    rem = total_dxa - c0
    cw = rem // (col_count - 1)
    widths = [c0] + [cw] * (col_count - 1)
    widths[0] += total_dxa - sum(widths)
    return widths


def apply_three_line_table_styling(tbl):
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tblPr = tbl._tbl.tblPr
    for tag in ("w:tblBorders", "w:tblLayout", "w:tblW"):
        for existing in tblPr.findall(qn(tag)):
            tblPr.remove(existing)
    tblPr.append(parse_xml('<w:tblW xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:w="9020" w:type="dxa"/>'))
    tblPr.append(parse_xml('<w:tblLayout xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:type="fixed"/>'))
    tblPr.append(parse_xml(r"""
        <w:tblBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
            <w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>
            <w:left w:val="none"/><w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>
            <w:right w:val="none"/><w:insideH w:val="none"/><w:insideV w:val="none"/>
        </w:tblBorders>
    """))

    col_count = len(tbl.rows[0].cells) if tbl.rows else len(tbl.columns)
    widths = _compute_col_widths_dxa(col_count, 9020)

    tblGrid = tbl._tbl.tblGrid
    if tblGrid is not None:
        for gc in list(tblGrid):
            tblGrid.remove(gc)
        for w_dxa in widths:
            tblGrid.append(parse_xml(f'<w:gridCol xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:w="{w_dxa}"/>'))

    is_wide = col_count >= 8
    hdr_pt, data_pt = (Pt(8.2), Pt(7.8)) if is_wide else (Pt(9.0), Pt(8.6))
    line_sp, sp_pad = (1.05, Pt(1.5)) if is_wide else (1.12, Pt(2.0))

    for r_idx, row in enumerate(tbl.rows):
        trPr = row._tr.get_or_add_trPr()
        if trPr.find(qn("w:cantSplit")) is None:
            trPr.append(parse_xml(r'<w:cantSplit xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>'))
        if r_idx == 0 and trPr.find(qn("w:tblHeader")) is None:
            trPr.append(parse_xml(r'<w:tblHeader xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>'))
        for c_idx, cell in enumerate(row.cells):
            tcPr = cell._tc.get_or_add_tcPr()
            for existing in tcPr.findall(qn("w:tcW")):
                tcPr.remove(existing)
            w_val = widths[c_idx] if c_idx < len(widths) else widths[-1]
            tcPr.append(parse_xml(f'<w:tcW xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:w="{w_val}" w:type="dxa"/>'))

    if len(tbl.rows) > 0:
        for cell in tbl.rows[0].cells:
            tcPr = cell._tc.get_or_add_tcPr()
            for existing in tcPr.findall(qn("w:tcBorders")):
                tcPr.remove(existing)
            tcPr.append(parse_xml(r"""
                <w:tcBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
                    <w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/>
                    <w:top w:val="none"/><w:left w:val="none"/><w:right w:val="none"/>
                </w:tcBorders>
            """))
            for cp in cell.paragraphs:
                pPr = cp._p.get_or_add_pPr()
                if pPr.find(qn("w:noLnn")) is None:
                    pPr.append(parse_xml(r'<w:noLnn xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>'))
                cp.paragraph_format.line_spacing = line_sp
                cp.paragraph_format.space_before = sp_pad
                cp.paragraph_format.space_after = sp_pad
                for r in cp.runs:
                    r.font.name = "Times New Roman"
                    r.font.size = hdr_pt
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)

    for row in tbl.rows[1:]:
        for cell in row.cells:
            tcPr = cell._tc.get_or_add_tcPr()
            for existing in tcPr.findall(qn("w:tcBorders")):
                tcPr.remove(existing)
            tcPr.append(parse_xml(r"""
                <w:tcBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
                    <w:top w:val="none"/><w:left w:val="none"/>
                    <w:bottom w:val="none"/><w:right w:val="none"/>
                </w:tcBorders>
            """))
            for cp in cell.paragraphs:
                pPr = cp._p.get_or_add_pPr()
                if pPr.find(qn("w:noLnn")) is None:
                    pPr.append(parse_xml(r'<w:noLnn xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>'))
                cp.paragraph_format.line_spacing = line_sp
                cp.paragraph_format.space_before = sp_pad
                cp.paragraph_format.space_after = sp_pad
                for r in cp.runs:
                    r.font.name = "Times New Roman"
                    r.font.size = data_pt
                    r.font.color.rgb = RGBColor(0x11, 0x18, 0x27)


def add_footer_page_numbering(doc):
    """Add a single clean centered OOXML PAGE field without non-existent Footer pStyle or empty runs."""
    for idx, s in enumerate(doc.sections):
        if idx > 0:
            s.footer.is_linked_to_previous = True
            continue
        s.footer.is_linked_to_previous = False
        footer = s.footer
        for ep in list(footer.paragraphs)[1:]:
            ep._element.getparent().remove(ep._element)
        p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        for child in list(p._p):
            p._p.remove(child)
        pPr = p._p.get_or_add_pPr()
        pPr.append(parse_xml('<w:jc xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:val="center"/>'))
        rPr = '<w:rPr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="20"/><w:color w:val="475569"/></w:rPr>'
        for frag in [
            f'<w:r xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">{rPr}<w:fldChar w:fldCharType="begin"/></w:r>',
            f'<w:r xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">{rPr}<w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>',
            f'<w:r xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">{rPr}<w:fldChar w:fldCharType="separate"/></w:r>',
            f'<w:r xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">{rPr}<w:fldChar w:fldCharType="end"/></w:r>',
        ]:
            p._p.append(parse_xml(frag))


def style_submission_doc(doc_path: Path, is_main: bool = False):
    doc = docx.Document(doc_path)
    doc.core_properties.author = ""
    doc.core_properties.last_modified_by = ""
    doc.core_properties.comments = ""
    for s in doc.sections:
        s.page_width = Inches(8.27); s.page_height = Inches(11.69)
        s.top_margin = Inches(1.0); s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0); s.right_margin = Inches(1.0)
    if is_main:
        add_continuous_line_numbering(doc)
    add_footer_page_numbering(doc)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor(0x11, 0x18, 0x27)

    in_references = False
    for p in doc.paragraphs:
        txt = p.text.strip()
        if is_main:
            p.paragraph_format.line_spacing = 1.35
        if p.style.name.startswith("Heading 1"):
            in_references = False
            p.paragraph_format.space_before = Pt(14); p.paragraph_format.space_after = Pt(6)
            for r in p.runs:
                r.font.name = "Times New Roman"; r.font.size = Pt(14); r.font.bold = True; r.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        elif p.style.name.startswith("Heading 2"):
            in_references = (txt == "References")
            p.paragraph_format.space_before = Pt(11); p.paragraph_format.space_after = Pt(4)
            for r in p.runs:
                r.font.name = "Times New Roman"; r.font.size = Pt(12); r.font.bold = True; r.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
        elif p.style.name.startswith("Heading 3"):
            p.paragraph_format.space_before = Pt(8); p.paragraph_format.space_after = Pt(3)
            for r in p.runs:
                r.font.name = "Times New Roman"; r.font.size = Pt(11); r.font.bold = True; r.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
        elif in_references and re.search(r"\((?:19|20)\d{2}[a-z]?\)\.", txt):
            p.paragraph_format.left_indent = Inches(0.35)
            p.paragraph_format.first_line_indent = Inches(-0.35)
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_before = Pt(0); p.paragraph_format.space_after = Pt(3)
            for r in p.runs:
                r.font.name = "Times New Roman"; r.font.size = Pt(10)

    for tbl in doc.tables:
        apply_three_line_table_styling(tbl)
    doc.save(doc_path)


def patch_settings(docx_path: Path):
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
    if "xmlns:m=" not in settings_xml:
        settings_xml = re.sub(r"(<w:settings [^>]*)(>)", r"\1 " + math_ns + r"\2", settings_xml, count=1)
    if "<m:mathPr>" not in settings_xml:
        open_end = settings_xml.find(">", settings_xml.find("<w:settings")) + 1
        settings_xml = settings_xml[:open_end] + math_pr + settings_xml[open_end:]
    contents["word/settings.xml"] = settings_xml.encode("utf-8")
    with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as z_out:
        for fname, data in contents.items():
            z_out.writestr(fname, data)


def add_markdown_sub_sup(s: str) -> str:
    """Convert statistical and parameter notation to standard Markdown ~subscript~ and ^superscript^ for real OpenXML vertAlign and HTML tags."""
    # 10^-X -> 10^-X^
    s = re.sub(r"`10\^-(\d+)`", r"10^-\1^", s)
    s = re.sub(r"\b10\^-(\d+)\b", r"10^-\1^", s)
    # d_RE -> *d*~RE~
    s = re.sub(r"`d_RE\s*=\s*([^`]+)`", r"*d*~RE~ = \1", s)
    s = re.sub(r"`d_RE`", r"*d*~RE~", s)
    s = re.sub(r"\bd_RE\b", r"*d*~RE~", s)
    # p_HKSJ -> *p*~HKSJ~
    s = re.sub(r"`p_HKSJ\s*=\s*([^`]+)`", r"*p*~HKSJ~ = \1", s)
    s = re.sub(r"`p_HKSJ`", r"*p*~HKSJ~", s)
    s = re.sub(r"\bp_HKSJ\b", r"*p*~HKSJ~", s)
    # p_max -> *p*~max~
    s = re.sub(r"`p_max\s*=\s*([^`]+)`", r"*p*~max~ = \1", s)
    s = re.sub(r"`p_max`", r"*p*~max~", s)
    s = re.sub(r"\bp_max\b", r"*p*~max~", s)
    # p_perm -> *p*~perm~
    s = re.sub(r"`p_perm\s*=\s*([^`]+)`", r"*p*~perm~ = \1", s)
    s = re.sub(r"`p_perm`", r"*p*~perm~", s)
    s = re.sub(r"\bp_perm\b", r"*p*~perm~", s)
    # p_adj -> *p*~adj~
    s = re.sub(r"`p_adj\s*=\s*([^`]+)`", r"*p*~adj~ = \1", s)
    s = re.sub(r"`p_adj`", r"*p*~adj~", s)
    s = re.sub(r"\bp_adj\b", r"*p*~adj~", s)
    # q_50 -> *q*~50~
    s = re.sub(r"`q_50\s*([<>=]+)\s*([^`]+)`", r"*q*~50~ \1 \2", s)
    s = re.sub(r"`q_50`", r"*q*~50~", s)
    s = re.sub(r"\bq_50\b", r"*q*~50~", s)
    # q_10 -> *q*~10~
    s = re.sub(r"`q_10\s*([<>=]+)\s*([^`]+)`", r"*q*~10~ \1 \2", s)
    s = re.sub(r"`q_10`", r"*q*~10~", s)
    s = re.sub(r"\bq_10\b", r"*q*~10~", s)
    # r_chi / r_χ -> *r*~χ~
    s = re.sub(r"`r_chi\s*=\s*([^`]+)`", r"*r*~χ~ = \1", s)
    s = re.sub(r"`r_chi`", r"*r*~χ~", s)
    s = re.sub(r"\br_chi\b", r"*r*~χ~", s)
    s = re.sub(r"`r_χ\s*=\s*([^`]+)`", r"*r*~χ~ = \1", s)
    s = re.sub(r"\br_χ\b", r"*r*~χ~", s)
    # r_theta / r_θ -> *r*~θ~
    s = re.sub(r"`r_theta\s*=\s*([^`]+)`", r"*r*~θ~ = \1", s)
    s = re.sub(r"`r_theta`", r"*r*~θ~", s)
    s = re.sub(r"\br_theta\b", r"*r*~θ~", s)
    s = re.sub(r"`r_θ\s*=\s*([^`]+)`", r"*r*~θ~ = \1", s)
    s = re.sub(r"\br_θ\b", r"*r*~θ~", s)
    # 1/f^chi / 1/f^χ -> 1/f^χ^
    s = re.sub(r"`1/f\^(?:chi|χ)\^?`", r"1/f^χ^", s)
    s = re.sub(r"1/f\^(?:chi|χ)\^?", r"1/f^χ^", s)
    # I_19 -> *I*~19~
    s = re.sub(r"`I_19`", r"*I*~19~", s)
    s = re.sub(r"\bI_19\b", r"*I*~19~", s)
    # SE_NB / SE_RE
    s = re.sub(r"`SE_NB`", r"*SE*~NB~", s)
    s = re.sub(r"\bSE_NB\b", r"*SE*~NB~", s)
    s = re.sub(r"`SE_RE`", r"*SE*~RE~", s)
    s = re.sub(r"\bSE_RE\b", r"*SE*~RE~", s)
    return s


def md_to_docx(md_path: Path, docx_path: Path, is_main: bool = False):
    scratch_dir = Path("/Users/tan/.gemini/antigravity/scratch")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    proc_md = scratch_dir / f"{md_path.stem}_docx_proc.md"
    raw_txt = md_path.read_text(encoding="utf-8")
    proc_md.write_text(add_markdown_sub_sup(raw_txt), encoding="utf-8")
    res_path = f"{MANUSCRIPT_DIR}:{REPORTS_DIR}"
    cmd = ["pandoc", str(proc_md), f"--resource-path={res_path}", "-f", "markdown+tex_math_dollars", "-t", "docx", "-o", str(docx_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Pandoc failed for {md_path.name}:\n{res.stderr}")
    style_submission_doc(docx_path, is_main=is_main)
    patch_settings(docx_path)


def md_to_pdf_via_chrome(md_path: Path, pdf_path: Path, is_main: bool = False):
    """Render Main Manuscript or Online Resource 1 Markdown to a cleanly typeset Springer A4 PDF via Pandoc HTML + Chrome."""
    scratch_dir = Path("/Users/tan/.gemini/antigravity/scratch")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    md_pdf_tmp = scratch_dir / f"{md_path.stem}_pdf.md"
    md_txt = md_path.read_text(encoding="utf-8")
    tex_disp = "$$\\log_{10} \\mathrm{PSD}(f) = (b - \\chi \\log_{10} f) + \\sum_{n=1}^N a_n \\exp\\left(-\\frac{(f - c_n)^2}{2 w_n^2}\\right)$$"
    html_disp = (
        '<div style="text-align:center;font-family:\'Times New Roman\',Times,serif;margin:8px 0;font-size:10pt;">'
        'log<sub>10</sub> PSD(<i>f</i>) = (<i>b</i> − χ log<sub>10</sub> <i>f</i>) + '
        'Σ<sub><i>n</i>=1..<i>N</i></sub> <i>a</i><sub><i>n</i></sub> exp(−(<i>f</i> − <i>c</i><sub><i>n</i></sub>)<sup>2</sup> / (2<i>w</i><sub><i>n</i></sub><sup>2</sup>))'
        '</div>'
    )
    tex_eq1 = "$$\\log_{10} P(f) = (b - \\chi \\log_{10} f) + \\sum_{k=1}^K G_k(f), \\quad f \\in [2.0, 30.0]\\text{ Hz} \\qquad (1)$$"
    html_eq1 = (
        '<div style="text-align:center;font-family:\'Times New Roman\',Times,serif;margin:10px 0;font-size:10.5pt;">'
        'log<sub>10</sub> <i>P</i>(<i>f</i>) = (<i>b</i> − χ log<sub>10</sub> <i>f</i>) + '
        'Σ<sub><i>k</i>=1..<i>K</i></sub> <i>G</i><sub><i>k</i></sub>(<i>f</i>), &nbsp;&nbsp; <i>f</i> ∈ [2.0, 30.0] Hz &nbsp;&nbsp;&nbsp;&nbsp; (1)'
        '</div>'
    )
    md_txt = md_txt.replace(tex_disp, html_disp).replace(tex_eq1, html_eq1)
    # Strip explicit {width=...} Pandoc attribute when rendering Chrome PDF (CSS max-width handles scaling cleanly)
    md_txt = re.sub(r"(\!\[[^\]]*\]\([^)]+\))\{width=[^}]+\}", r"\1", md_txt)
    md_txt = add_markdown_sub_sup(md_txt)
    md_pdf_tmp.write_text(md_txt, encoding="utf-8")
    html_tmp = scratch_dir / f"{md_path.stem}.html"
    res_path = f"{MANUSCRIPT_DIR}:{REPORTS_DIR}"
    cmd = ["pandoc", str(md_pdf_tmp), f"--resource-path={res_path}", "--standalone", "--embed-resources", "-f", "markdown", "-t", "html5", "-o", str(html_tmp)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Pandoc HTML failed for {md_path.name}:\n{res.stderr}")

    body_pt = "10.5pt" if is_main else "9.5pt"
    body_lh = "1.42" if is_main else "1.38"
    page_m = "0.75in 0.70in" if is_main else "0.55in 0.50in"
    css = f"""
    <style>
      @page {{ size: A4; margin: {page_m}; }}
      body {{ font-family: "Times New Roman", Times, Georgia, serif; font-size: {body_pt}; line-height: {body_lh}; color: #111827; max-width: 100%; }}
      h1 {{ font-size: 14pt; color: #0f172a; border-bottom: 1.5px solid #cbd5e1; padding-bottom: 4px; }}
      h2 {{ font-size: 11.5pt; color: #1e293b; margin-top: 16px; page-break-after: avoid; }}
      h3 {{ font-size: 10pt; color: #1e293b; margin-top: 12px; page-break-after: avoid; }}
      h4 {{ font-size: 9.5pt; color: #334155; margin-top: 8px; page-break-after: avoid; }}
      img {{ max-width: 100%; height: auto; display: block; margin: 6px auto; page-break-inside: avoid; }}
      table {{ width: 100%; border-collapse: collapse; margin: 6px 0 12px 0; font-size: 7.4pt; page-break-inside: auto; }}
      thead {{ display: table-header-group; }}
      tr {{ page-break-inside: avoid; }}
      th, td {{ border: none; padding: 3px 4px; text-align: center; vertical-align: middle; }}
      th:first-child, td:first-child {{ text-align: left; }}
      thead tr {{ border-top: 1.5px solid #000; border-bottom: 0.8px solid #000; font-weight: bold; }}
      tbody tr:last-child {{ border-bottom: 1.5px solid #000; }}
      code {{ font-family: "Courier New", monospace; font-size: 8pt; background: #f1f5f9; padding: 1px 2px; border-radius: 2px; }}
      hr {{ border: 0; border-top: 1px solid #cbd5e1; margin: 12px 0; }}
    </style>
    """
    html_txt = html_tmp.read_text(encoding="utf-8").replace("</head>", css + "\n</head>", 1)
    html_tmp.write_text(html_txt, encoding="utf-8")

    pdf_cmd = [
        str(CHROME_BIN), "--headless", "--disable-gpu", "--no-sandbox",
        "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", html_tmp.as_uri(),
    ]
    pres = subprocess.run(pdf_cmd, capture_output=True, text=True)
    if pres.returncode != 0 or not pdf_path.exists():
        raise RuntimeError(f"Chrome PDF export failed:\n{pres.stderr}")


def load_all_benchmark_data() -> dict:
    with open(REPORTS_DIR / "master_results.json", encoding="utf-8") as f:
        master = json.load(f)
    with open(REPORTS_DIR / "real_permutation_distributions.json", encoding="utf-8") as f:
        perms = json.load(f)
    with open(REPORTS_DIR / "extra_biophysics_summary.json", encoding="utf-8") as f:
        extra_bio = json.load(f)

    data = {
        "master": master, "perms": perms, "extra_bio": extra_bio,
        "fit": pd.read_csv(REPORTS_DIR / "specparam_fit_diagnostics_by_cohort.csv"),
        "gain_chk": pd.read_csv(REPORTS_DIR / "specparam_gain_invariance_check.csv"),
        "bio_wide": pd.read_csv(REPORTS_DIR / "ablation_specparam_vs_ols_biophysics_wide.csv"),
        "bio_ab": pd.read_csv(REPORTS_DIR / "ablation_specparam_vs_ols_biophysics.csv"),
        "boot_d": pd.read_csv(REPORTS_DIR / "ablation_specparam_vs_ols_bootstrap_deltas.csv"),
        "sim_mc": pd.read_csv(REPORTS_DIR / "synthetic_spectral_leakage_simulation.csv"),
        "meta": pd.read_csv(REPORTS_DIR / "meta_analysis_frontal_theta_polarity.csv"),
        "bio_exp": pd.read_csv(REPORTS_DIR / "multicohort_biophysical_effect_sizes_expanded.csv"),
        "f3f4": pd.read_csv(REPORTS_DIR / "biophysics_f3_fz_f4_sensitivity.csv"),
        "cls0": pd.read_csv(REPORTS_DIR / "multicohort_7cohort_extended_model_comparison.csv"),
        "mum_eo": pd.read_csv(REPORTS_DIR / "mumtaz_eo_model_comparison.csv"),
        "wilcoxon": pd.read_csv(REPORTS_DIR / "multicohort_wilcoxon_model_comparisons.csv"),
        "lofo": pd.read_csv(REPORTS_DIR / "multicohort_7cohort_feature_ablation.csv"),
        "hparam": pd.read_csv(REPORTS_DIR / "benchmark_gcn_hyperparameter_sensitivity.csv"),
        "rep10": pd.read_csv(REPORTS_DIR / "benchmark_repeated_10x5_summary.csv"),
        "pli_audit": pd.read_csv(REPORTS_DIR / "pli_spd_eigenvalue_audit.csv"),
        "pli_sum": pd.read_csv(REPORTS_DIR / "pli_spd_eigenvalue_summary.csv"),
        "riem": pd.read_csv(REPORTS_DIR / "benchmark_riemannian_tangent_space.csv"),
        "pd_tx": pd.read_csv(REPORTS_DIR / "cross_site_pd_transfer_benchmarks.csv"),
        "td_tx": pd.read_csv(REPORTS_DIR / "comprehensive_transdiagnostic_transfer_matrix.csv"),
        "gain_cal": pd.read_csv(REPORTS_DIR / "benchmark_gain_and_calibration_ablation.csv"),
        "leakage": pd.read_csv(REPORTS_DIR / "benchmark_leakage_ablation.csv"),
        "heur": pd.read_csv(REPORTS_DIR / "benchmark_heuristic_slowing.csv"),
        "reg": pd.read_csv(REPORTS_DIR / "multicohort_clinical_regression_benchmarks_v6.csv"),
        "ancova": pd.read_csv(REPORTS_DIR / "multicohort_age_ancova_complete.csv"),
        "anc_dur": pd.read_csv(REPORTS_DIR / "multicohort_age_and_duration_ancova.csv"),
        "med": pd.read_csv(REPORTS_DIR / "unm_ucsd_paired_medication_biophysics.csv"),
        "unm_sens": pd.read_csv(REPORTS_DIR / "unm_ec_vs_eo_sensitivity.csv"),
        "iapf": pd.read_csv(REPORTS_DIR / "iapf_biophysics_summary.csv"),
        "mum_conc": pd.read_csv(REPORTS_DIR / "mumtaz_ec_eo_concordance.csv"),
        "mum_tx": pd.read_csv(REPORTS_DIR / "mumtaz_eo_state_matched_transfer.csv"),
        "ep_audit": pd.read_csv(REPORTS_DIR / "cohort_epoch_duration_audit.csv"),
        "s1_omst": pd.read_csv(REPORTS_DIR / "table_s1_omst_channel_metrics.csv"),
        "mum_dups": pd.read_csv(REPORTS_DIR / "mumtaz_duplicate_edf_hashes.csv"),
        "mum_forensic": pd.read_csv(REPORTS_DIR / "study2_duplicate_and_batch_sensitivity.csv"),
        "s2_unfilt": pd.read_csv(REPORTS_DIR / "study2_unfiltered_psd_and_header_audit.csv"),
        "grid2d": pd.read_csv(REPORTS_DIR / "specparam_2d_window_grid_sensitivity.csv"),
        "fooof_eq": pd.read_csv(REPORTS_DIR / "fooof_v1_vs_specparam_v2_equivalence.csv"),
        # v6 & v7 sensitivity / remediation datasets
        "s7_pz": pd.read_csv(REPORTS_DIR / "study7_pz_car_sensitivity.csv"),
        "part_sess": pd.read_csv(REPORTS_DIR / "participant_vs_session_level_auc_10x5.csv"),
        "tost_v6": pd.read_csv(REPORTS_DIR / "tost_dependency_and_holm_sensitivity.csv"),
        "macro_v6": pd.read_csv(REPORTS_DIR / "macro_auc_study2_sensitivity.csv"),
        "scov_fm": pd.read_csv(REPORTS_DIR / "spatial_cov_foldwise_frechet_check.csv"),
        "s2_181": pd.read_csv(REPORTS_DIR / "study2_complete_181_file_accounting.csv"),
        "fooof_full": pd.read_csv(REPORTS_DIR / "fooof_v111_full13927_verification.csv"),
        "v7_tost": pd.read_csv(REPORTS_DIR / "v7_per_contrast_and_mixed_effects_tost.csv"),
        "v7_airm": pd.read_csv(REPORTS_DIR / "v7_airm_foldwise_riemannian_benchmark.csv"),
        "v7_deep": pd.read_csv(REPORTS_DIR / "v7_deep_baseline_and_tuned_gcn_10x5.csv"),
        "v7_lc": pd.read_csv(REPORTS_DIR / "v7_sample_size_learning_curves.csv"),
    }
    data["reg"]["cohort"] = data["reg"]["cohort"].apply(strip_md_bold)
    return data


# ==============================================================================
# Dynamic Main Tables Builders (Split Wide Tables + Disambiguated Regressions)
# ==============================================================================
def build_table_1(d: dict) -> str:
    fit_230 = d["fit"][d["fit"]["fit_window_hz"] == "2.0-30.0"].set_index("cohort_key")
    bio = d["bio_exp"]
    f3 = d["f3f4"].set_index("cohort")
    rows_cfg = [
        ("Study 1: Sleep Deprivation (N=71 pairs)", "Study 1: Acute Sleep Deprivation", "sleep_ds004902", "Paired t"),
        ("Study 2: Outpatient MDD EC (29 vs 27)*", "Study 2: Outpatient MDD", "mumtaz_ec", "Welch t"),
        ("Study 3: Univ. Depression (44 vs 75)", "Study 3: Univ. Depression Screening", "cavanagh_ds003478", "Welch t"),
        ("Study 4: UCSD PD SubjMean (15 vs 16)", "Study 4: UCSD Parkinson All", "ucsd_ds002778", "Welch t"),
        ("Study 5a: Dementia AD (36 vs 29)", "Study 5a: Alzheimer's Disease (AD)", "dementia_ds004504", "Welch t"),
        ("Study 5b: Dementia FTD (23 vs 29)", "Study 5b: Frontotemporal Dementia (FTD)", "dementia_ds004504", "Welch t"),
        ("Study 5c: Dementia All (59 vs 29)", "Study 5c: Clinical Dementia (All)", "dementia_ds004504", "Welch t"),
        ("Study 6a: UNM PD-OFF EO (25 vs 25)", "Study 6a: UNM Parkinson OFF", "unm_ds003490", "Welch t"),
        ("Study 6b: UNM PD-ON EO (25 vs 25)", "Study 6b: UNM Parkinson ON", "unm_ds003490", "Welch t"),
        ("Study 7: Iowa PD (100 vs 49)", "Study 7: Iowa Parkinson Cohort", "iowa_ds004584", "Welch t"),
    ]
    lines = [
        "### Table 1. Canonical Interior [2.0, 30.0] Hz Parameterized EEG Biophysics and Fit Quality Across Ten Contrasts",
        "| Study & Contrast | Design | Frontal θ (4–8 Hz) d [95% CI] | p / q_10 | Posterior α d (p) | Global χ d (p) | Fit R² / MAE (B) | F3–Fz–F4 θ d (p) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for label, bname, fkey, design in rows_cfg:
        ft = bio[(bio["cohort"] == bname) & (bio["marker_key"] == "frontal_theta")].iloc[0]
        pa = bio[(bio["cohort"] == bname) & (bio["marker_key"] == "posterior_alpha")].iloc[0]
        gc = bio[(bio["cohort"] == bname) & (bio["marker_key"] == "global_chi")].iloc[0]
        f3r, finfo = f3.loc[bname], fit_230.loc[fkey]
        lines.append(
            f"| {label} | {design} | {ft['cohens_d']:+.3f} [{ft['ci_lower']:+.3f}, {ft['ci_upper']:+.3f}] | "
            f"{fmt_p(ft['p_value'])} / {fmt_p(ft['fdr_q_10'])} | {pa['cohens_d']:+.3f} ({fmt_p(pa['p_value'])}) | "
            f"{gc['cohens_d']:+.3f} ({fmt_p(gc['p_value'])}) | {finfo['r2_median']:.3f} / {finfo['mae_median']:.3f} | "
            f"{f3r['cohens_d']:+.3f} ({fmt_p(f3r['p_value'])}) |"
        )
    meta_p = d["meta"][(d["meta"]["method"] == "specparam_2to30_resid") & (d["meta"]["analysis_set"] == "Primary_K6_SubjMean_PD_OFF")].set_index("subgroup")
    loo_dem = d["meta"][(d["meta"]["method"] == "specparam_2to30_resid") & (d["meta"]["analysis_set"] == "LOO_Omit_Dementia") & (d["meta"]["subgroup"] == "PD_Dementia_Cohorts")].iloc[0]
    ndg = meta_p.loc["PD_Dementia_Cohorts"]
    ucsd_all = d["extra_bio"]["UCSD_All_theta"]
    lines.append("")
    lines.append(
        f"*Note: Units of log power error (`MAE`) are Bels (`B`, log10[μV²/Hz]). Values derive from `specparam` `[2.0, 30.0] Hz` (validated against `fooof v1.1.1` in Section 2.3 and Online Resource 1, Supplementary Table S27B). *Study 2 removes two SHA-256 duplicate pairs (`29 MDD vs. 27 HC` unique; batch sensitivity in Supplementary Table S26). In Study 4, participant averaging avoids treating repeated PD sessions as independent (pooled sessions `d = {ucsd_all['cohens_d']:+.3f}`, `p = {fmt_p(ucsd_all['p_value'])}`). Across k = 4 neurodegenerative PD/dementia cohorts (Studies 4 SubjMean, 5c, 6a, 7), frontal periodic theta elevation pools at `d_RE = {ndg['pooled_d']:+.3f}` (HKSJ 95% CI `[{ndg['hksj_ci_lower']:+.3f}, {ndg['hksj_ci_upper']:+.3f}]`, `p_HKSJ = {fmt_p(ndg['p_hksj'])}`, `I² = {ndg['I2_percent']:.1f}%`; two of four individual studies—Study 4 and Study 6a—have 95% CIs spanning zero), and remains comparable across the state-homogeneous eyes-open PD-only subset (Studies 4, 6a, 7, k = 3: `d_RE = {loo_dem['pooled_d']:+.3f}`, HKSJ 95% CI `[{loo_dem['hksj_ci_lower']:+.3f}, {loo_dem['hksj_ci_upper']:+.3f}]`, `p_HKSJ = {fmt_p(loo_dem['p_hksj'])}`, `I² = {loo_dem['I2_percent']:.1f}%`; `I²` carries wide uncertainty at small `k`, `von Hippel, 2015`). Full `K = 6` and `k = 2` subgroup pooling appear in Supplementary Table S22.*"
    )
    return "\n".join(lines)


def build_table_2(d: dict) -> str:
    w, boot = d["bio_wide"], d["boot_d"]
    ft_df = w[w["marker"] == "frontal_theta_5ch"].set_index("cohort")
    chi_df = w[w["marker"] == "global_chi"].set_index("cohort")
    b_win = boot[(boot["marker"] == "frontal_theta_5ch") & (boot["comparison"] == "SP_1_40_minus_SP_2_30")].set_index("cohort")
    b_ols = boot[(boot["marker"] == "frontal_theta_5ch") & (boot["comparison"] == "OLS_1_40_minus_SP_2_30")].set_index("cohort")
    lines = [
        "### Table 2. Spectral Estimation and Fit-Window Confounding Audit Across Legacy Masked OLS, Filter-Edge [1, 40] Hz, and Canonical [2, 30] Hz `specparam`",
        "| Cohort Contrast | Masked OLS [1–40] θ d (p) | `specparam` [1–40] θ d (p) | `specparam` [2–30] Peaks θ d (p) | `specparam` [2–30] Resid θ d (p) | Δd [1–40 vs 2–30] (Boot p) | Global χ [1–40 vs 2–30] d |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    cohort_order = [
        ("Study 1: Acute Sleep Deprivation", "Study 1: Sleep Deprivation (N=71 pairs)"),
        ("Study 2: Outpatient MDD", "Study 2: Outpatient MDD EC (29 vs 27)"),
        ("Study 3: Univ. Depression Screening", "Study 3: Univ. Depression (44 vs 75)"),
        ("Study 4: UCSD Parkinson All", "Study 4: UCSD PD SubjMean (15 vs 16)"),
        ("Study 5a: Alzheimer's Disease (AD)", "Study 5a: Dementia AD (36 vs 29)"),
        ("Study 5b: Frontotemporal Dementia (FTD)", "Study 5b: Dementia FTD (23 vs 29)"),
        ("Study 5c: Clinical Dementia (All)", "Study 5c: Dementia All (59 vs 29)"),
        ("Study 6a: UNM Parkinson OFF", "Study 6a: UNM PD-OFF EO (25 vs 25)"),
        ("Study 6b: UNM Parkinson ON", "Study 6b: UNM PD-ON EO (25 vs 25)"),
        ("Study 7: Iowa Parkinson Cohort", "Study 7: Iowa PD (100 vs 49)"),
    ]
    for cname, label in cohort_order:
        fr, cr, bw = ft_df.loc[cname], chi_df.loc[cname], b_win.loc[cname]
        lines.append(
            f"| {label} | {fr['cohens_d_masked_ols_1to40']:+.3f} ({fmt_p(fr['p_value_masked_ols_1to40'])}) | "
            f"{fr['cohens_d_specparam_resid_1to40']:+.3f} ({fmt_p(fr['p_value_specparam_resid_1to40'])}) | "
            f"{fr['cohens_d_specparam_peaks_2to30']:+.3f} ({fmt_p(fr['p_value_specparam_peaks_2to30'])}) | "
            f"{fr['cohens_d_specparam_resid_2to30']:+.3f} ({fmt_p(fr['p_value_specparam_resid_2to30'])}) | "
            f"{bw['delta_d']:+.3f} ({fmt_p(bw['boot_p'])}) | {cr['cohens_d_specparam_resid_1to40']:+.3f} / {cr['cohens_d_specparam_resid_2to30']:+.3f} |"
        )
    b_s2 = b_win.loc["Study 2: Outpatient MDD"]
    b_s3_ols = b_ols.loc["Study 3: Univ. Depression Screening"]
    lines.append("")
    lines.append(
        f"*Note: Δd reports 1,000-bootstrap paired `d_[1–40] − d_[2–30]` shifts (full deltas and `OLS [2–30]` in Online Resource 1, Supplementary Table S11). Fitting into Study 2's 40 Hz upstream roll-off attenuated nominal `[1–40]` theta (`d = {ft_df.loc['Study 2: Outpatient MDD', 'cohens_d_specparam_resid_1to40']:+.3f}` vs. `{ft_df.loc['Study 2: Outpatient MDD', 'cohens_d_specparam_resid_2to30']:+.3f}`; Δd = `{b_s2['delta_d']:+.3f}`, `p = {fmt_p(b_s2['boot_p'])}`) and inverted global χ; unmasked `[7, 14]` OLS in Study 3 inflated theta negativity (`{ft_df.loc['Study 3: Univ. Depression Screening', 'cohens_d_masked_ols_1to40']:+.3f}` vs. `{ft_df.loc['Study 3: Univ. Depression Screening', 'cohens_d_specparam_resid_2to30']:+.3f}`; Δd = `{b_s3_ols['delta_d']:+.3f}`, `p = {fmt_p(b_s3_ols['boot_p'])}`).*"
    )
    return "\n".join(lines)


def build_table_3(d: dict) -> str:
    """Split Table 3 cleanly into Table 3a (Graph Topologies) and Table 3b (Classical & Univariate) + sensitivity rows."""
    rep = d["rep10"]
    piv = rep.pivot(index="cohort", columns="model", values="mean_auc")
    mac = d["macro_v6"].set_index("scope")
    tost = d["tost_v6"]
    ps = d["part_sess"].set_index(["cohort", "model"])

    p9_cohorts = [
        ("Study 2: Mumtaz MDD (Eyes Closed)", "Study 2: MDD EC (N=56)"),
        ("Study 3: Cavanagh University Depression", "Study 3: Univ. Dep. (N=119)"),
        ("Study 4: UCSD Parkinson All Sessions", "Study 4: UCSD PD Sess. (N=46)"),
        ("Study 5a: Dementia AD vs Control", "Study 5a: Dementia AD (N=65)"),
        ("Study 5b: Dementia FTD vs Control", "Study 5b: Dementia FTD (N=52)"),
        ("Study 5c: Dementia All (AD+FTD) vs Control", "Study 5c: Dementia All (N=88)"),
        ("Study 6a: UNM PD-OFF vs Control", "Study 6a: UNM PD-OFF EO (N=50)"),
        ("Study 7: Iowa Parkinson (ds004584)", "Study 7: Iowa PD (N=149)"),
        ("Study 1: Sleep Deprivation (ds004902)", "Study 1: Sleep Depr. (N=142)"),
    ]
    gnn_cols = [("GCN_OMST", "GCN-OMST"), ("GCN_Weighted_PLI", "GCN-WPLI"), ("GCN_Identity", "GCN-Id"), ("GCN_Random_EdgeMatched", "GCN-Rand*"), ("GCN_Full", "GCN-Full")]
    cls_cols = [("Tuned_LogisticRegression", "Tuned LR"), ("Tuned_Ridge", "Ridge"), ("Linear_SVM", "Lin-SVM"), ("RBF_SVM", "RBF-SVM"), ("RandomForest", "RF"), ("GBDT", "GBDT"), ("Single_Theta", "Single θ"), ("Single_Chi", "Single χ")]

    lines = [
        "### Table 3. True 10×5 Repeated Participant-Grouped Mean ROC-AUC Across Contrasts, Graph Topologies, and Classical Baselines",
        "#### Panel 3a. Static Fixed-Hyperparameter 19-Node `NormativeGCN` Topologies (Including Participant-Level Sensitivity)",
        "| Primary Contrast | " + " | ".join(lbl for _, lbl in gnn_cols) + " | Best Classical (Table 3b) |",
        "| :--- | " + " | ".join(":---:" for _ in range(len(gnn_cols) + 1)) + " |",
    ]
    for cname, slabel in p9_cohorts:
        g_vals = [float(piv.loc[cname, m]) for m, _ in gnn_cols]
        c_max = max(float(piv.loc[cname, m]) for m, _ in cls_cols)
        lines.append(f"| {slabel} | " + " | ".join(f"**{v:.3f}**" if abs(v - max(g_vals)) < 1e-9 else f"{v:.3f}" for v in g_vals) + f" | {c_max:.3f} |")

    s4_p_vals = [float(ps.loc[("Study 4: UCSD Parkinson All Sessions", m), "participant_mean_auc_10x5"]) for m, _ in gnn_cols]
    s4_p_cls = max(float(ps.loc[("Study 4: UCSD Parkinson All Sessions", m), "participant_mean_auc_10x5"]) for m, _ in cls_cols)
    lines.append(f"| *Study 4 Participant-Level (n = 31)* | " + " | ".join(f"{v:.3f}" for v in s4_p_vals) + f" | {s4_p_cls:.3f} |")

    for s_key, s_lbl in [
        ("k=8 Non-Nested (Session-Level Study 4)", "Mean k = 8 Non-Nested (Session Study 4)"),
        ("k=8 Non-Nested (Participant-Level Study 4)", "Mean k = 8 Non-Nested (Part-Level Study 4)"),
        ("k=7 Excl. Study 2 Batch-Confounded", "Mean k = 7 Excl. Study 2 (Batch-Confounded)"),
        ("k=7 Drop 5b Shared-HC (AD Only)", "Mean k = 7 Drop Shared-HC 5b (AD Only)"),
        ("k=9 Primary (Incl. 5c)", "Mean k = 9 Primary (Incl. Nested 5c)"),
    ]:
        r = mac.loc[s_key]
        gv = [float(r[m]) for m, _ in gnn_cols]
        cm = max(float(r[m]) for m, _ in cls_cols)
        lines.append(f"| {s_lbl} | " + " | ".join(f"**{v:.3f}**" if abs(v - max(gv)) < 1e-9 else f"{v:.3f}" for v in gv) + f" | {cm:.3f} |")

    lines.extend([
        "",
        "#### Panel 3b. Tuned Classical 76-D Feature Models and Univariate Spectral Baselines",
        "| Primary Contrast | " + " | ".join(lbl for _, lbl in cls_cols) + " |",
        "| :--- | " + " | ".join(":---:" for _ in cls_cols) + " |",
    ])
    for cname, slabel in p9_cohorts:
        vals = [float(piv.loc[cname, m]) for m, _ in cls_cols]
        mx = max(vals)
        lines.append(f"| {slabel} | " + " | ".join(f"**{v:.3f}**" if abs(v - mx) < 1e-9 else f"{v:.3f}" for v in vals) + " |")
    s4_c_vals = [float(ps.loc[("Study 4: UCSD Parkinson All Sessions", m), "participant_mean_auc_10x5"]) for m, _ in cls_cols]
    lines.append(f"| *Study 4 Participant-Level (n = 31)* | " + " | ".join(f"{v:.3f}" for v in s4_c_vals) + " |")
    for s_key, s_lbl in [
        ("k=8 Non-Nested (Session-Level Study 4)", "Mean k = 8 Non-Nested (Session Study 4)"),
        ("k=8 Non-Nested (Participant-Level Study 4)", "Mean k = 8 Non-Nested (Part-Level Study 4)"),
        ("k=7 Excl. Study 2 Batch-Confounded", "Mean k = 7 Excl. Study 2 (Batch-Confounded)"),
        ("k=7 Drop 5b Shared-HC (AD Only)", "Mean k = 7 Drop Shared-HC 5b (AD Only)"),
        ("k=9 Primary (Incl. 5c)", "Mean k = 9 Primary (Incl. Nested 5c)"),
    ]:
        r = mac.loc[s_key]
        cv = [float(r[m]) for m, _ in cls_cols]
        lines.append(f"| {s_lbl} | " + " | ".join(f"**{v:.3f}**" if abs(v - max(cv)) < 1e-9 else f"{v:.3f}" for v in cv) + " |")

    vtost = d["v7_tost"]
    vt8 = vtost[(vtost["record_type"] == "Random_Effects_Pool") & (vtost["scope"] == "k=8 Non-Nested")].set_index("comparator_model")
    vt7b = vtost[(vtost["record_type"] == "Random_Effects_Pool") & (vtost["scope"] == "k=7 Drop 5b (AD Only)")].set_index("comparator_model")
    vdeep = d["v7_deep"].groupby("model")["mean_auc"].mean()
    vairm = d["v7_airm"][(d["v7_airm"]["representation"] == "Foldwise_AIRM_Frechet_Z") & (d["v7_airm"]["classifier"] == "RF")]["auc_mean"].mean()
    lines.append("")
    lines.append(
        f"*Note: Cells report 10-repeat mean out-of-fold ROC-AUC across 50 outer participant-grouped folds (`± SD`, Split-0 95% CIs, and Studies 2b/6b/6c in Online Resource 1, Supplementary Tables S10 and S20). All five Panel 3a `NormativeGCN` variants use fixed hyperparameters (`lr = 0.005`, `epochs = 50`, `hidden = 32`); `GCN-Rand*` denotes the session edge-count-matched Erdős–Rényi random graph baseline. Additional `k = 8` extensions (inner-tuned `GCN-OMST-Tuned = {vdeep['GCN_OMST_Tuned']:.3f}`, compact 1,770-parameter `ShallowSpecCNN = {vdeep['ShallowSpecCNN']:.3f}`, and fold-wise AIRM Fréchet `Spatial Cov RF = {vairm:.3f}` alongside `Identity Log-Euclidean RF = 0.724`; `Linear SVM = 0.721`) appear in Supplementary Tables S7c and S21. Random-effects TOST pooling (`DerSimonian & Laird, 1986`) over per-contrast `Nadeau & Bengio (2003)` variance (`k = 8`) confirmed `±0.02` equivalence (`Holm p <= 0.0370`) between `GCN-OMST` (`0.705`) and `GCN-Rand*` (Δ = `{vt8.loc['GCN_Random_EdgeMatched', 'mean_diff_auc']:+.4f}`, 90% CI `[{vt8.loc['GCN_Random_EdgeMatched', 'ci90_lo']:+.4f}, {vt8.loc['GCN_Random_EdgeMatched', 'ci90_hi']:+.4f}]`, `Holm p = {vt8.loc['GCN_Random_EdgeMatched', 'tost02_holm_gnn4']:.4f}`), `GCN-Full` (Δ = `{vt8.loc['GCN_Full', 'mean_diff_auc']:+.4f}`, `Holm p = {vt8.loc['GCN_Full', 'tost02_holm_gnn4']:.4f}`), and `GCN-WPLI` (Δ = `{vt8.loc['GCN_Weighted_PLI', 'mean_diff_auc']:+.4f}`, `Holm p = {vt8.loc['GCN_Weighted_PLI', 'tost02_holm_gnn4']:.4f}`), with `GCN-Rand*` and `GCN-Full` persisting when dropping shared-control Study 5b (`k = 7`: `Holm p = {vt7b.loc['GCN_Random_EdgeMatched', 'tost02_holm_gnn4']:.4f}`). By contrast, `GCN-Id` (Δ = `{vt8.loc['GCN_Identity', 'mean_diff_auc']:+.4f}`, 90% CI `[{vt8.loc['GCN_Identity', 'ci90_lo']:+.4f}, {vt8.loc['GCN_Identity', 'ci90_hi']:+.4f}]`, `I² = {vt8.loc['GCN_Identity', 'i2_percent']:.1f}%`) was inconclusive at `±0.02` (`Holm p = {vt8.loc['GCN_Identity', 'tost02_holm_gnn4']:.4f}`) and met `±0.04` equivalence (`Holm p = {vt8.loc['GCN_Identity', 'tost04_holm_gnn4']:.4f}`; per-contrast SEs in Supplementary Table S10b).*"
    )
    return "\n".join(lines)


def build_table_4(d: dict) -> str:
    """Clean single-panel Table 4 for disjoint Split-Target-HC transfer (Panel 4b reported in Supplementary Table S19)."""
    pd_tx, td_tx = d["pd_tx"], d["td_tx"]
    def mean_sd_str(row):
        return f"{row['auc']:.3f} (±{row['split_hc_sd']:.3f})"

    lines = [
        "### Table 4. Leak-Free Disjoint 50/50 Split-Target-HC External Transfer Without Retraining (Mean ROC-AUC Across 20 Splits)",
        "| Source Cohort → Target Cohort | Domain Regime | GCN-OMST (±SD) | GCN-Identity | Tuned LR | Random Forest | GBDT |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    pd_piv = pd_tx.pivot(index=["source_cohort", "target_cohort"], columns="model", values="auc")
    pd_raw = pd_tx[pd_tx["model"] == "GCN_OMST"].set_index(["source_cohort", "target_cohort"])
    for src, tgt, lbl in [
        ("Iowa_PD", "UNM_PD_OFF", "Iowa PD (149) → UNM PD-OFF EO (50)"),
        ("Iowa_PD", "UCSD_PD", "Iowa PD (149) → UCSD PD (46)"),
        ("UCSD_PD", "Iowa_PD", "UCSD PD (46) → Iowa PD (149)"),
        ("UCSD_PD", "UNM_PD_OFF", "UCSD PD (46) → UNM PD-OFF EO (50)"),
        ("UNM_PD_OFF", "Iowa_PD", "UNM PD-OFF EO (50) → Iowa PD (149)"),
        ("UNM_PD_OFF", "UCSD_PD", "UNM PD-OFF EO (50) → UCSD PD (46)"),
    ]:
        r_g = pd_raw.loc[(src, tgt)]
        vals = [float(pd_piv.loc[(src, tgt), m]) for m in ("GCN_OMST", "GCN_Identity", "Tuned_LR", "RandomForest", "GBDT")]
        mx = max(vals)
        lines.append(f"| {lbl} | Cross-Site PD | " + " | ".join(f"**{mean_sd_str(r_g) if i == 0 else f'{v:.3f}'}**" if abs(v - mx) < 1e-9 else (mean_sd_str(r_g) if i == 0 else f"{v:.3f}") for i, v in enumerate(vals)) + " |")

    td_piv = td_tx.pivot(index=["source_cohort", "target_cohort"], columns="model", values="auc")
    td_raw = td_tx[td_tx["model"] == "GCN_OMST"].set_index(["source_cohort", "target_cohort"])
    for src, tgt, lbl, rtype in [
        ("Cavanagh_MDD", "Mumtaz_MDD", "Univ. Dep. (119) → Outpatient MDD EC (56)", "Affective"),
        ("Mumtaz_MDD", "Cavanagh_MDD", "Outpatient MDD EC (56) → Univ. Dep. (119)", "Affective"),
        ("Iowa_PD", "Dementia_All", "Iowa PD (149) → Dementia All (88)", "Elevated Theta (Cross-Disease)"),
        ("Sleep_Debt", "Dementia_All", "Sleep Deprivation (142) → Dementia All (88)", "Elevated Theta (Acute→Dem.)"),
        ("Sleep_Debt", "Iowa_PD", "Sleep Deprivation (142) → Iowa PD (149)", "Elevated Theta (Acute→PD)"),
        ("Dementia_All", "Mumtaz_MDD", "Dementia All (88) → Outpatient MDD EC (56)", "Cross-Disorder (Discordant)"),
        ("Sleep_Debt", "Mumtaz_MDD", "Sleep Deprivation (142) → Outpatient MDD EC (56)", "Cross-Disorder (Discordant)"),
        ("Mumtaz_MDD", "Dementia_All", "Outpatient MDD EC (56) → Dementia All (88)", "Cross-Disorder (Reverse)"),
    ]:
        r_g = td_raw.loc[(src, tgt)]
        vals = [float(td_piv.loc[(src, tgt), m]) for m in ("GCN_OMST", "GCN_Identity", "Tuned_LR", "RandomForest", "GBDT")]
        mx = max(vals)
        lines.append(f"| {lbl} | {rtype} | " + " | ".join(f"**{mean_sd_str(r_g) if i == 0 else f'{v:.3f}'}**" if abs(v - mx) < 1e-9 else (mean_sd_str(r_g) if i == 0 else f"{v:.3f}") for i, v in enumerate(vals)) + " |")

    lines.append("")
    lines.append("*Note: Target-HC z-scoring statistics are estimated on 50% of target healthy controls strictly disjoint from evaluated target controls and patients across 20 random splits (`Oracle full-HC` references, calibration trade-offs, and `5×` amplifier gain stress-tests appear in Online Resource 1, Supplementary Tables S15, S16, and S19).*")
    return "\n".join(lines)


def build_table_5(d: dict) -> str:
    reg = d["reg"]
    lines = [
        "### Table 5. Exploratory Out-of-Fold Clinical Severity Regression Benchmarks (Split-0 Paired 4-Model Max-Statistic Permutations and 10×5 Summary)",
        "| Cohort & Scope | Scale | N | Selected Model | Split-0 r | Split-0 2-S Perm p | Max-Stat 4M p (1-S / 2-S) | 10×5 Mean r (±SD) | Mean OOF R² |",
        "| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for ckey, cname, sname, nval in [
        ("Dementia_MMSE_All_N88", "Study 5: Dementia All (ds004504)", "MMSE (4–30)", 88),
        ("Dementia_MMSE_Patients_Only_N59", "Study 5: Dementia Patients Only", "MMSE (4–29)", 59),
        ("Iowa_PD_MoCA_All_N149", "Study 7: Iowa PD All (ds004584)", "MoCA (9–30)", 149),
        ("Iowa_PD_MoCA_Patients_Only_N100", "Study 7: Iowa PD Patients Only", "MoCA (9–30)", 100),
        ("Iowa_PD_UPDRS_Patients_N100", "Study 7: Iowa PD Patients Only", "UPDRS-III (1–28)", 100),
    ]:
        sub = reg[reg["cohort_key"] == ckey]
        best = sub.loc[sub["pearson_r_mean"].idxmax()]
        bname = best["model"].replace("_Regressor", "").replace("_", " ")
        lines.append(
            f"| {cname} | {sname} | {nval} | {bname} | {best['split0_r']:+.3f} | "
            f"{fmt_p(best['perm_p_split0_2s'])} | {fmt_p(best['maxstat4_perm_p_1s'])} / {fmt_p(best['maxstat4_perm_p_2s'])} | "
            f"{best['pearson_r_mean']:+.3f} (±{best['pearson_r_sd']:.3f}) | {best['r2_mean']:+.3f} |"
        )
    lines.append("")
    lines.append(
        "*Note: `Selected Model` identifies the highest `10×5 Mean r` regressor among `GBDT`, `Random Forest`, `SVR RBF`, and `Ridge` on 76-D `[2.0, 30.0] Hz` `specparam` features; `Split-0 r` reflects the first CV repeat paired with its exact two-sided permutation `p` (`1,000` permutations), whereas `10×5 Mean r` averages across all 10 repeats (`Supplementary Table S14`; e.g., Dementia `All` `Random Forest` reached `Split-0 r = +0.433` vs. `10×5 Mean r = +0.408`, whereas `SVR RBF` reached `Split-0 r = +0.393` vs. `10×5 Mean r = +0.431`). `Max-Stat 4M p (1-S / 2-S)` evaluates exact paired 1,000-permutation `max` / `max |r|` null distributions across all four models per endpoint to remove model-selection bias. Within patient-only Dementia MMSE (`N = 59`), `10×5 Mean OOF R²` spans `-0.110 to -0.047` across all four regressors (`Ridge`: `Split-0 r = +0.134`, `10×5 Mean r = +0.233`, `OOF R² = -0.105`; `Random Forest`: `OOF R² = -0.110`; `GBDT`: `-0.094`; `SVR RBF`: `-0.047`), and all 12 patient-only models explain `<= 3.5%` out-of-fold variance (`10/12` negative).*"
    )
    return "\n".join(lines)


# ==============================================================================
# Main Manuscript Builder (Neuroinformatics v6)
# ==============================================================================
def build_main_manuscript(d: dict) -> str:
    t1, t2, t3, t4, t5 = build_table_1(d), build_table_2(d), build_table_3(d), build_table_4(d), build_table_5(d)
    acc, omst = d["master"]["accounting"], d["master"]["omst_topology"]
    perms = d["perms"]
    bio_w = d["bio_wide"]
    ft_w = bio_w[bio_w["marker"] == "frontal_theta_5ch"].set_index("cohort")
    chi_w = bio_w[bio_w["marker"] == "global_chi"].set_index("cohort")
    bio_exp, boot_d, sim_mc = d["bio_exp"], d["boot_d"], d["sim_mc"]
    fit230 = d["fit"][d["fit"]["fit_window_hz"] == "2.0-30.0"].set_index("cohort_key")
    fit140 = d["fit"][d["fit"]["fit_window_hz"] == "1.0-40.0"].set_index("cohort_key")
    gchk = d["gain_chk"]
    pli_s = d["pli_sum"].iloc[0]
    pli_a = d["pli_audit"].set_index("cohort")
    riem = d["riem"]
    nn8_r_cohs = [c for c in riem["cohort"].unique() if "5c" not in c and "6b" not in c]
    no_s2_r_cohs = [c for c in nn8_r_cohs if not c.startswith("Study 2:")]
    riem_nn8 = riem[riem["cohort"].isin(nn8_r_cohs)].groupby(["representation", "classifier"])["auc"].mean()
    riem_no_s2 = riem[riem["cohort"].isin(no_s2_r_cohs)].groupby(["representation", "classifier"])["auc"].mean()
    scov_fm = d["scov_fm"]

    rep = d["rep10"]
    p9_cnames = [c for c in rep["cohort"].unique() if not any(k in c for k in ("2b", "6b", "6c"))]
    nn8_cnames = [c for c in p9_cnames if "5c" not in c]
    rep_p9 = rep[rep["cohort"].isin(p9_cnames)].pivot(index="cohort", columns="model", values="mean_auc")
    rep_nn8 = rep[rep["cohort"].isin(nn8_cnames)].pivot(index="cohort", columns="model", values="mean_auc")
    mac = d["macro_v6"].set_index("scope")
    tost_v6 = d["tost_v6"]
    vtost = d["v7_tost"]
    vt_pool = vtost[vtost["record_type"] == "Random_Effects_Pool"]
    vt8 = vt_pool[vt_pool["scope"] == "k=8 Non-Nested"].set_index("comparator_model")
    vt7b = vt_pool[vt_pool["scope"] == "k=7 Drop 5b (AD Only)"].set_index("comparator_model")
    vt7a = vt_pool[vt_pool["scope"] == "k=7 Drop 5a (FTD Only)"].set_index("comparator_model")
    vt7_nos2 = vt_pool[vt_pool["scope"] == "k=7 Excl. Study 2"].set_index("comparator_model")
    vt8_id_c = vtost[(vtost["record_type"] == "Per_Contrast") & (vtost["comparator_model"] == "GCN_Identity") & (vtost["cohort"].isin(nn8_cnames))].set_index("cohort")
    vdeep = d["v7_deep"].groupby("model")["mean_auc"].mean()
    vairm_m = d["v7_airm"].groupby(["representation", "classifier"])["auc_mean"].mean()
    vlc = d["v7_lc"].groupby(["cohort", "train_fraction", "model"])[["mean_n_train", "mean_auc"]].mean()
    ps_df = d["part_sess"].set_index(["cohort", "model"])

    meta_230 = d["meta"][(d["meta"]["method"] == "specparam_2to30_resid") & (d["meta"]["analysis_set"] == "Primary_K6_SubjMean_PD_OFF")].set_index("subgroup")
    loo_dem_m = d["meta"][(d["meta"]["method"] == "specparam_2to30_resid") & (d["meta"]["analysis_set"] == "LOO_Omit_Dementia") & (d["meta"]["subgroup"] == "PD_Dementia_Cohorts")].iloc[0]
    ndg_m = meta_230.loc["PD_Dementia_Cohorts"]

    leak = d["leakage"].set_index(["cohort", "regime", "model"])
    gc_sub = d["gain_cal"]
    cal_raw_lr_sp = gc_sub[(gc_sub["experiment"] == "Calibration_Strategy") & (gc_sub["calibration"] == "Uncalibrated_Raw") & (gc_sub["feature_type"] == "specparam_periodic_residual")]["lr_auc"].mean()
    cal_raw_lr_raw = gc_sub[(gc_sub["experiment"] == "Calibration_Strategy") & (gc_sub["calibration"] == "Uncalibrated_Raw") & (gc_sub["feature_type"] == "raw_log_band_power")]["lr_auc"].mean()
    cal_raw_lr_rel = gc_sub[(gc_sub["experiment"] == "Calibration_Strategy") & (gc_sub["calibration"] == "Uncalibrated_Raw") & (gc_sub["feature_type"] == "relative_band_power")]["lr_auc"].mean()
    cal_raw_bacc_sp = gc_sub[(gc_sub["experiment"] == "Calibration_Strategy") & (gc_sub["calibration"] == "Uncalibrated_Raw") & (gc_sub["feature_type"] == "specparam_periodic_residual")]["lr_bacc"].mean()
    cal_thc_lr_sp = gc_sub[(gc_sub["experiment"] == "Calibration_Strategy") & (gc_sub["calibration"] == "Target_HC_Normative") & (gc_sub["feature_type"] == "specparam_periodic_residual")]
    amp_raw_5x = gc_sub[(gc_sub["experiment"] == "Amplifier_Gain_Ablation") & (gc_sub["feature_type"] == "raw_log_band_power") & (gc_sub["gain_k"].astype(str) == "5.0")]
    amp_raw_mix = gc_sub[(gc_sub["experiment"] == "Amplifier_Gain_Ablation") & (gc_sub["feature_type"] == "raw_log_band_power") & (gc_sub["gain_k"].astype(str) == "Heterogeneous_0.5x_5.0x")]

    reg = d["reg"]
    reg_pt_mmse = reg[reg["cohort_key"] == "Dementia_MMSE_Patients_Only_N59"]
    dem_all_rf = reg[(reg["cohort_key"] == "Dementia_MMSE_All_N88") & (reg["model"] == "Random_Forest_Regressor")].iloc[0]
    dem_all_svr = reg[(reg["cohort_key"] == "Dementia_MMSE_All_N88") & (reg["model"] == "SVR_RBF")].iloc[0]
    dem_pt_rf = reg[(reg["cohort_key"] == "Dementia_MMSE_Patients_Only_N59") & (reg["model"] == "Random_Forest_Regressor")].iloc[0]
    dem_pt_ridge = reg[(reg["cohort_key"] == "Dementia_MMSE_Patients_Only_N59") & (reg["model"] == "Ridge_Regression")].iloc[0]
    iowa_all_svr = reg[(reg["cohort_key"] == "Iowa_PD_MoCA_All_N149") & (reg["model"] == "SVR_RBF")].iloc[0]
    iowa_pt_svr = reg[(reg["cohort_key"] == "Iowa_PD_MoCA_Patients_Only_N100") & (reg["model"] == "SVR_RBF")].iloc[0]
    iowa_pt_ridge = reg[(reg["cohort_key"] == "Iowa_PD_MoCA_Patients_Only_N100") & (reg["model"] == "Ridge_Regression")].iloc[0]
    iowa_updrs_gbdt = reg[(reg["cohort_key"] == "Iowa_PD_UPDRS_Patients_N100") & (reg["model"] == "GBDT_Regressor")].iloc[0]
    iowa_updrs_rf = reg[(reg["cohort_key"] == "Iowa_PD_UPDRS_Patients_N100") & (reg["model"] == "Random_Forest_Regressor")].iloc[0]

    med_df = d["med"].set_index(["cohort", "marker"])
    unm_med_ft = med_df.loc[("Study 6: UNM PD Pure EO (>=15h Washout) (N=25)", "Frontal Periodic Theta (4–8 Hz)")]
    ucsd_med_ft = med_df.loc[("Study 4: UCSD PD (Acute Washout) (N=15)", "Frontal Periodic Theta (4–8 Hz)")]
    unm_sens_df = d["unm_sens"].set_index("condition")
    iapf_df = d["iapf"].set_index("cohort")
    iowa_iapf = iapf_df.loc["Study 7: Iowa Parkinson Cohort"]
    mum_conc = d["mum_conc"].set_index("marker")
    hparam_means = d["hparam"].groupby("config_label")["auc"].mean()

    pd_tx = d["pd_tx"].pivot(index=["source_cohort", "target_cohort"], columns="model", values="auc")
    td_tx = d["td_tx"].pivot(index=["source_cohort", "target_cohort"], columns="model", values="auc")
    s2_for = d["mum_forensic"].set_index("subset")
    s2_b = d["s2_unfilt"][~d["s2_unfilt"]["is_duplicate_sha256"]].groupby("batch")[["logp_6hz", "drop_2_to_1hz", "drop_30_to_40hz", "line_50hz_excess"]].mean()
    ff_all = d["fooof_full"][d["fooof_full"]["scope"] == "All_733Sessions_13927Spectra"].iloc[0]
    g2d = d["grid2d"]
    g_s2_45 = g2d[(g2d["psd_source"] == "Unfiltered_Raw_EDF") & (g2d["f_min_hz"] == 1.0) & (g2d["f_max_hz"] == 45.0)].iloc[0]
    g_s2_230 = g2d[(g2d["psd_source"] == "Unfiltered_Raw_EDF") & (g2d["f_min_hz"] == 2.0) & (g2d["f_max_hz"] == 30.0)].iloc[0]
    s7_pz = d["s7_pz"].set_index("regime")

    eb_sl = d["extra_bio"]["Sleep_global_theta"]
    eb_m_eo_dl = d["extra_bio"]["Mumtaz_EO_delta"]
    eb_m_eo_ch = d["extra_bio"]["Mumtaz_EO_chi"]
    ad_pa = bio_exp[(bio_exp["cohort"] == "Study 5a: Alzheimer's Disease (AD)") & (bio_exp["marker_key"] == "posterior_alpha")].iloc[0]
    ad_gc = bio_exp[(bio_exp["cohort"] == "Study 5a: Alzheimer's Disease (AD)") & (bio_exp["marker_key"] == "global_chi")].iloc[0]
    b_s2_w = boot_d[(boot_d["cohort"] == "Study 2: Outpatient MDD") & (boot_d["marker"] == "frontal_theta_5ch") & (boot_d["comparison"] == "SP_1_40_minus_SP_2_30")].iloc[0]
    b_s3_o = boot_d[(boot_d["cohort"] == "Study 3: Univ. Depression Screening") & (boot_d["marker"] == "frontal_theta_5ch") & (boot_d["comparison"] == "OLS_1_40_minus_SP_2_30")].iloc[0]
    sim_s2_ols = sim_mc[(sim_mc["scenario"] == "S2_TrueNullTheta_BroadAlphaAndSlowCurve") & (sim_mc["estimator"] == "Masked_OLS_1to40")].iloc[0]
    sim_s2_230 = sim_mc[(sim_mc["scenario"] == "S2_TrueNullTheta_BroadAlphaAndSlowCurve") & (sim_mc["estimator"] == "specparam_2to30")].iloc[0]
    sim_s3_140 = sim_mc[(sim_mc["scenario"] == "S3_FilterEdgeAttenuation_Study2Replica") & (sim_mc["estimator"] == "specparam_1to40")].iloc[0]
    sim_s3_230 = sim_mc[(sim_mc["scenario"] == "S3_FilterEdgeAttenuation_Study2Replica") & (sim_mc["estimator"] == "specparam_2to30")].iloc[0]

    # Abstract strictly <= 250 words with ZERO undefined acronyms!
    abstract_text = (
        f"Reproducible electroencephalography (EEG) neuroinformatics requires open benchmarks testing whether spectral biomarkers, graph convolutional networks (GCNs), and deep models survive methodological auditing. "
        f"I release an open Python suite across seven public 19-channel datasets ({acc['total_unique_participants']} unique participants; {acc['total_analyzed_sessions']} primary sessions; {acc['total_spectra']:,} spectra) as a three-stage verification workflow spanning sleep deprivation, affective screening, depression, Parkinson's disease (PD), and dementia. "
        f"First, interior `[2.0, 30.0] Hz` `specparam` fits across a 25-configuration grid resolved upstream roll-off collapse in Study 2 while avoiding masked ordinary least-squares theta bias in Study 3 (`d = {ft_w.loc['Study 3: Univ. Depression Screening', 'cohens_d_masked_ols_1to40']:+.3f}` vs. `{ft_w.loc['Study 3: Univ. Depression Screening', 'cohens_d_specparam_resid_2to30']:+.3f}`). "
        f"Raw European Data Format (EDF) forensics exposed duplicates and diagnosis–batch confounding: removing January-2013 controls eliminated Study 2 theta separation (`d = -0.082`, `p = 0.752`), whereas PD and dementia frontal periodic theta elevation replicated (`pooled d = {ndg_m['pooled_d']:+.3f}`, `p = {ndg_m['p_hksj']:.4f}`; eyes-open PD `k = 3`: `d = {loo_dem_m['pooled_d']:+.3f}`). "
        f"Second, across eight non-nested contrasts under 10×5 participant-grouped validation, orthogonal-minimum-spanning-tree `GCN-OMST` (receiver operating characteristic `AUC = {rep_nn8['GCN_OMST'].mean():.3f}`) met `±0.02` two one-sided equivalence (`TOST`; Nadeau–Bengio random-effects Holm `p <= {vt8.loc['GCN_Weighted_PLI', 'tost02_holm_gnn4']:.4f}`) against random (`{rep_nn8['GCN_Random_EdgeMatched'].mean():.3f}`), dense (`{rep_nn8['GCN_Full'].mean():.3f}`), and weighted phase-lag (`{rep_nn8['GCN_Weighted_PLI'].mean():.3f}`) graphs—while isolated-node `GCN-Identity` (`{rep_nn8['GCN_Identity'].mean():.3f}`) failed `±0.02` equivalence (`p = {vt8.loc['GCN_Identity', 'tost02_holm_gnn4']:.3f}`)—matched tuned `GCN-OMST` (`{vdeep['GCN_OMST_Tuned']:.3f}`) and `ShallowSpecCNN` (`{vdeep['ShallowSpecCNN']:.3f}`), and did not outperform linear support vector machines (`{rep_nn8['Linear_SVM'].mean():.3f}`) or fold-wise Affine-Invariant Riemannian Metric (`AIRM`) covariance Random Forests (`{vairm_m.loc[('Foldwise_AIRM_Frechet_Z', 'RF')]:.3f}`). "
        f"Third, `{pli_s['pct_indefinite_raw']:.2f}%` of symmetrized phase-lag matrices were indefinite, session mixing biased `AUC` (`-0.094` vs. `+0.080`), and patient-only regressions explained `<= 3.5%` out-of-fold variance."
    )
    abs_wc = len(abstract_text.split())
    assert 150 <= abs_wc <= 250, f"Abstract word count {abs_wc} outside [150, 250]!"
    print(f"   ✓ Abstract word count: {abs_wc} words (<= 250 limit)")

    refs_md = "\n\n".join(entry for _, entry in REFS_ALPHA)

    md = f"""# {EXACT_TITLE}

**Running Title:** {SHORT_TITLE}

**Article Type:** {ARTICLE_TYPE}

## Abstract

{abstract_text}

**Keywords:** Neuroinformatics Benchmark; Quantitative EEG; Spectral Parameterization; Graph Neural Networks; Dataset Forensics; Reproducibility Auditing

---

## 1. Introduction

Open computational neuroinformatics depends on standardized software benchmarks testing whether proposed quantitative electroencephalography (qEEG) biomarkers, deep architectures, and graph models generalize across independent centers and clinical recording pipelines (Nunez & Srinivasan, 2006; Kapur et al., 2012; Lopes da Silva, 2013; Cohen, 2014; Roy et al., 2019; Gemein et al., 2020). Across sleep regulation, affective illness, and neurodegeneration (Buzsáki & Draguhn, 2004; Buzsáki, 2006; Fries, 2005; Stam, 2014), automated deep networks—such as convolutional recurrent models trained on effective connectivity (Saeedi et al., 2021) or band-power asymmetry features (Mahato & Paul, 2020)—often report high single-site accuracy yet fail under independent center transfer (Scheinost et al., 2019; Varoquaux, 2018; Brookshire et al., 2024). Rather than isolated issues, reproducibility failures in clinical EEG pipelines propagate sequentially through three stages: (i) spectral decomposition and archive metadata errors, (ii) unvalidated graph, manifold, or deep model claims on correlated cross-validation folds in small cohorts, and (iii) uncalibrated external transfer and clinical severity inflation.

At Stage 1 (signal decomposition and archive integrity), separating periodic oscillations from the broadband `1/f^chi` aperiodic background is foundational (Voytek et al., 2015; Cole & Voytek, 2017; Gao et al., 2017; Colombo et al., 2019; Donoghue et al., 2020). Because the aperiodic exponent `chi` shifts with age, arousal, and cortical excitation–inhibition balance, fixed-band ratios conflate broadband shifts with genuine oscillatory peaks, clouding resting frontal theta (`4–8 Hz`) comparisons across sleep pressure (Finelli et al., 2000), Parkinson's disease (PD), Alzheimer's disease (AD) (Bosboom et al., 2006; Stoffers et al., 2007; Stam, 2010), and affective disorders (Mitchell et al., 2008; Cavanagh & Frank, 2014; Knott et al., 2001; Jaworska et al., 2012; Olbrich & Arns, 2013; Newson & Thiagarajan, 2019). Recent spectral guidelines demonstrated how known digital filter corners and `[7, 14] Hz`-masked ordinary least squares (`OLS`) bias synthetic and clean fits (Donoghue et al., 2022; Gerster et al., 2022). Moving beyond prior single-site or synthetic evaluations, this resource uncovers an insidious real-world failure mode across public clinical European Data Format (`EDF`) archives: upstream pre-export spectral roll-off that leaves EDF/BIDS filter metadata labels (`HP:0.5 Hz LP:80 Hz`) identical across batches while colliding systematically with acquisition dates, inverting `1/f` slopes and creating spurious diagnostic separation unless inspected directly via raw spectra and file hashes.

At Stage 2 (connectivity, deep learning, and validation hygiene), clinical 19-channel 10–20 EEG cohorts typically span modest sample sizes (`N = 31–149`) (Bullmore & Sporns, 2009; Rubinov & Sporns, 2010). Phase-lag connectomes (`PLI`) mitigate zero-lag volume conduction (Stam et al., 2007; Vinck et al., 2011), Orthogonal Minimum Spanning Trees (`OMST`) prune graphs parameter-free via Global Cost Efficiency (`GCE`) (Dimitriadis et al., 2017a; Dimitriadis et al., 2017b), Riemannian geometry targets symmetric positive-definite (`SPD`) spatial covariances (Barachant et al., 2013; Congedo et al., 2017), and Graph Convolutional Networks (`GCNs`) (Kipf & Welling, 2017) as well as compact convolutional neural networks (`CNNs`) (Lawhern et al., 2018; Schirrmeister et al., 2017) offer structured inductive biases. Yet within small clinical 19-channel cohorts, three statistical questions remain unresolved: whether symmetrized `PLI` matrices violate positive semi-definiteness (`PSD`) when treated as manifold inputs; whether `GCN-OMST` or compact spectral-spatial `CNNs` outperform random graphs, isolated nodes, tuned linear/tree baselines, or fold-wise Affine-Invariant Riemannian Metric (`AIRM`) spatial covariances when equivalence tests properly account for cross-validation correlation (Nadeau & Bengio, 2003); and how session-mixing leakage distorts estimates bidirectionally (Saeb et al., 2017; Little et al., 2017; Brookshire et al., 2024).

At Stage 3 (transfer and clinical staging), single-cohort accuracy rarely establishes whether fold-nested normative calibration (Wolfers et al., 2018; Rutherford et al., 2022) sustains cross-center or cross-disorder generalization, or whether within-patient severity correlations survive removal of healthy controls and correction for multi-model selection. To test all three stages sequentially within one executable Python resource, I aggregated seven public 19-channel archives (`{acc['total_unique_participants']}` unique participants; `{acc['total_analyzed_sessions']}` primary sessions; `{acc['total_spectra'] // 19}` total recordings yielding `{acc['total_spectra']:,}` channel spectra; Online Resource 1, Supplementary Figure S5) using `MNE-Python` (Gramfort et al., 2013) and `specparam v2.0.0rc7` (Donoghue et al., 2020) to address three coupled questions:

1. Stage 1 (Spectral Boundaries, Reference Sensitivity, and EDF Batch Forensics): Across a 25-configuration 2D fit grid (`105` cohort evaluations across `[1.0–3.0] × [25.0–45.0] Hz`), `18`- vs. `19`-channel referencing in Study 7, legacy `[7, 14] Hz`-masked OLS, and raw EDF forensics, how severely do upstream pre-export roll-off and acquisition-date confounding distort apparent case–control spectral effects relative to replicated neurodegenerative shifts?
2. Stage 2 (Architectural Equivalence Across Graph, Compact CNN, and AIRM Baselines): Under true 10×5 participant-grouped validation in small clinical cohorts (`N = 31–149`) with per-contrast `Nadeau & Bengio (2003)` variance and random-effects `TOST` (`DerSimonian & Laird, 1986; Schuirmann, 1987; Lakens, 2017`), does `GCN-OMST` (fixed or inner-tuned) or a compact 1,770-parameter `ShallowSpecCNN` surpass random/dense/PLI graphs, isolated nodes, regularized tabular baselines, or fold-wise `AIRM` Fréchet-whitened spatial covariances, and across what training-sample fractions?
3. Stage 3 (Disjoint Transfer, Gain Calibration, and Clinical Severity Floor): Under leak-free disjoint `Split-Target-HC` transfer without retraining, how do cross-site and cross-disorder generalization, gain invariance vs. target-control calibration noise, and patient-only severity `OOF R²` behave once healthy controls and model-selection bias are removed?

---

## 2. Materials and Methods

### 2.1 Public Datasets, Sub-Contrast Taxonomy, and Archive Accounting
This benchmarking resource aggregates seven de-identified, publicly downloadable resting-state EEG archives from OpenNeuro and Figshare processed under EEGLAB/BIDS and MNE conventions (Delorme & Makeig, 2004; Gramfort et al., 2013; study hardware and demographics appear in Online Resource 1, Supplementary Table S3, and pinned environment versions in Supplementary Table S9). Primary analyses respected each archive's experimental structure via explicit sub-contrasts (`2b`, `4a–b`, `5a–c`, `6a–c`). Across all seven releases, `574` nominal participants were enrolled (`685` primary sessions; `745` total recordings including Study 2 eyes-open). Pre-analysis QC excluded `8` participants (`6` Study 2 EC sessions with >20% artifact rejection or channel bridging; `2` Study 3 participants with borderline BDI-II scores `10–13`), yielding `566` nominal participants (`677` primary sessions; `737` total). Raw-file SHA-256 forensics in Study 2 removed `2` exact byte-identical duplicate pairs (`H S27` = `H S30` and `MDD S33` = `MDD S34`), yielding a canonical archive of `N = 564` unique primary participants, `675` primary sessions, and `733` total recordings (`13,927` unique spectra; complete `181`-EDF accounting in Online Resource 1, Supplementary Table S26 Panel D and Supplementary Figure S5):

1. Study 1 (Sleep Deprivation, OpenNeuro `ds004902`, `N = 71` participants, `142` paired sessions) (Xiang et al., 2024): Young adults (`20.0 ± 1.4` years, `47.9%` female) recorded eyes-open after normal sleep (`NS`) and `36 h` total sleep deprivation (`SD`).
2. Study 2 (Outpatient MDD, Figshare `4244171`, de-duplicated `N = 56` primary EC `Study 2a` [`29 MDD vs. 27 HC`]; `N = 58` secondary EO `Study 2b` [`31 MDD vs. 27 HC`]; parsed from `181` total EDF files including `61` excluded task files and `2` Figshare ID-prefixed copies of `H S15 EO`) (Mumtaz et al., 2017): DSM-IV outpatients and healthy controls recorded on a `19`-channel clinical linked-ear (`-LE`) system (`256 Hz`); individual ages were omitted from the public release. Beyond duplicate hashes, acquisition-date forensics identified structural diagnosis–batch confounding (`all 29` MDD recorded in Apr–Jul 2013 vs. `27` HC split across `2011 [n=9]`, `Jan 12–14, 2013 [n=14]`, and `Apr–Jul 2013 [n=4]`; Online Resource 1, Supplementary Table S26).
3. Study 3 (University Depression Screening, OpenNeuro `ds003478`, `N = 119` participants) (Cavanagh et al., 2019): Undergraduate BDI-II symptom screening cohort (rather than clinically ascertained MDD) contrasting high-symptom students (`BDI-II >= 14`, `N = 44`: SCID-verified `11` current MDD, `11` past MDD, `8` non-MDD, `14` uninterviewed) against low-symptom controls (`BDI-II <= 7`, `N = 75`) during eyes-open rest.
4. Study 4 (UCSD PD ON/OFF, OpenNeuro `ds002778`, `N = 31` participants, `46` sessions) (George et al., 2013; Rockhill et al., 2020): Fifteen PD patients recorded OFF (`>= 12 h` withdrawal, `Study 4a`) and ON dopaminergic therapy (`Study 4b`) plus `16` HC (eyes-open). Biophysical contrasts average PD ON/OFF within participant (`Study 4 SubjMean`, `15 vs. 16`); 10×5 CV reports both session-level (`46` sessions) and participant-averaged (`31` participants) ROC-AUC under strict participant grouping.
5. Study 5 (AHEPA Clinical Dementia, OpenNeuro `ds004504`, `N = 88` participants) (Miltiadous et al., 2023): Eyes-closed 19-channel EEG across AD (`Study 5a`, `36 vs. 29`), FTD (`Study 5b`, `23 vs. 29`), and combined dementia (`Study 5c`, `59 vs. 29`) with MMSE scores. Because `5a`/`5b` share the same `29` healthy controls and `5c` pools both patient groups, `TOST` benchmarks report `k = 8` (excluding nested `5c`) alongside `k = 7` sensitivity sets dropping `5b`, `5a`, or Study 2, and `k = 9` primary sets.
6. Study 6 (UNM PD Washout, OpenNeuro `ds003490`, `N = 50` participants, `75` sessions) (Cavanagh et al., 2018): Twenty-five PD patients recorded OFF (`>= 15 h` withdrawal, `Study 6a`) and ON medication (`Study 6b`; combined `Study 6c`, `75` sessions across `50` participants) plus `25` CTL, rebuilt strictly from the initial contiguous `~60 s` pure eyes-open rest block preceding the oddball task (`10.46 ± 3.27` vs. `9.60 ± 3.06` clean epochs, `p = 0.267`), plus a `74`-session pure eyes-closed sensitivity check (Online Resource 1, Supplementary Table S25).
7. Study 7 (Iowa PD Multi-Scale, OpenNeuro `ds004584`, `N = 149` participants) (Singh et al., 2023; Anjum et al., 2024): Two-minute eyes-open rest from `100` PD patients ON medication and `49` HC with MoCA and UPDRS-III clinical scores.

### 2.2 Harmonization, Study 7 `Pz` Online Reference Sensitivity, and Unfiltered EDF Audits
Multichannel EEG arrays were mapped to the canonical `19`-channel 10–20 montage (`Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, P7, P3, Pz, P4, P8, O1, O2`; Online Resource 1, Supplementary Table S5), Common Average Referenced (`CAR`), zero-phase FIR bandpass filtered (`1.0–40.0 Hz`, Hamming window), resampled to `250 Hz`, and segmented into `4.0 s` epochs (`150 μV` peak-to-peak rejection; `200 μV` fallback in Study 7; Online Resource 1, Supplementary Table S8). Throughout the benchmark, log-10 power spectral densities (`PSD`) and spectral fitting errors are expressed in **Bels (`B`, log~10~[μV^2^/Hz])**. In Study 7 (`ds004584`), BIDS metadata (`sub-001_task-Rest_eeg.json`) records `63` BrainVision channels referenced online to `"EEGReference": "Pz"`. Reconstructing zero-initialized `Pz` prior to 19-channel CAR recovers `V_Pz - CAR_19 = -CAR_19`; to confirm independence from reference reconstruction across all `149` Iowa participants, strictly `18`-channel physically recorded `CAR18_NoPz` (`72-D` features, `18×18` OMST) and spherical-spline `CAR19_SplinePz` pipelines were additionally extracted directly from raw BIDS files (Online Resource 1, Supplementary Table S28). Clean epoch counts differed modestly across groups in Studies 1 (`p = 0.0103`), 2a (`p = 0.0040`), and 5 (`p = 0.0006`) and were balanced elsewhere; joint age and clean-epoch ANCOVAs verified robustness (Supplementary Table S25). In Study 2, raw unfiltered EDF headers (`sfreq`, `highpass`, `lowpass`, `prefilter`, `meas_date`, `SHA-256`) and unfiltered `0.5–80 Hz` Welch PSDs (`0.50 Hz` step) were audited across all `58` EC files.

### 2.3 Interior [2.0, 30.0] Hz Parameterization, 2D Fit-Window Grid, and Mechanistic Demonstrations
Welch PSDs (`2.0 s` Hamming windows, `50%` overlap, `0.50 Hz` step) were parameterized across all `13,927` unique spectra via `SpectralGroupModel` in `specparam v2.0.0rc7` (Donoghue et al., 2020) (`aperiodic_mode='fixed'`, `peak_width_limits=[1.0, 12.0]` Hz, `max_n_peaks=6`, `min_peak_height=0.05 B`, `peak_threshold=2.0` SD; Online Resource 1, Supplementary Section S1.1 and Supplementary Figure S1). To ensure the `2.0.0rc7` release candidate matched the stable branch without numerical discrepancy, all `13,927` unique channel spectra were cross-checked against `fooof v1.1.1` (overall `r_chi = {ff_all['pearson_r_chi']:.4f}`, `r_theta = {ff_all['pearson_r_theta']:.4f}`; lowest cohort Iowa `r_chi = 0.9902`; Supplementary Table S27 Panel B). A 25-configuration 2D grid (`105` cohort evaluations across `[1.0–3.0] × [25.0–45.0] Hz`) crossed `f_min in [1.0, 1.5, 2.0, 2.5, 3.0] Hz` against `f_max in [25.0, 30.0, 35.0, 40.0] Hz` across Studies 2 (`N = 56`), 3 (`N = 119`), 5c (`N = 88`), and 7 (`N = 149`), plus `f_max` up to `45.0 Hz` on unfiltered Study 2 EDF spectra (Supplementary Table S27 Panel A). Primary parameterization fitted `[2.0, 30.0] Hz`:

$$\\log_{{10}} P(f) = (b - \\chi \\log_{{10}} f) + \\sum_{{k=1}}^K G_k(f), \\quad f \\in [2.0, 30.0]\\text{{ Hz}} \\qquad (1)$$

Four nodal features were extracted: band-averaged flattened residuals (`log10 PSD - (b - chi * log10 f)`) over theta (`4.0–8.0 Hz`), low-frequency periodic residual (`2.0–4.0 Hz`), and alpha (`8.0–13.0 Hz`), plus aperiodic exponent `chi`, aggregated over five frontal sites (`Fp1, Fp2, F3, F4, Fz`), dorsal `F3–Fz–F4`, five posterior sites (`P3, P4, Pz, O1, O2`), or all `19` sites, alongside posterior individual alpha peak frequency (`IAPF`, `7.0–13.5 Hz`). Every `[2.0, 30.0] Hz` contrast was compared against `[1.0, 40.0] Hz` `specparam`, `[7, 14] Hz`-masked OLS (`1–40` and `2–30 Hz`), and discrete summed Gaussian peaks (`specparam_peaks_2to30`) via 1,000 stratified bootstraps (`Delta d`). Here, positive `4–8 Hz` residual shifts denote frontal periodic theta elevation, whereas downward `IAPF` shifts index peak-frequency slowing. Three calibrated Monte Carlo simulations (`150` trials × `N = 29 vs. 27`; `SD(chi) = 0.16`, `SD(theta) = 0.12`, `SD(alpha) = 0.20`, `SD(b) = 0.30`, spectral noise `0.035 B`) served as mechanistic, by-construction demonstrations verifying bias directionality under synthetic ground truth alongside exact multiplicative voltage gain invariance (`gamma in [0.01, 100.0]`, `285` spectra; Online Resource 1, Supplementary Section S1.3, Supplementary Table S2, Supplementary Figure S3).

### 2.4 Connectomics (`PLI`, `OMST`), Manifold Suitability, and Fold-Wise AIRM Spatial Covariance Baselines
Pairwise `4.0–13.0 Hz` `PLI` matrices (Stam et al., 2007; Vinck et al., 2011) were mapped to edge costs `c_ij = (1.0 - PLI_ij) + 1e-4` and pruned via `OMST` maximizing `GCE` (Dimitriadis et al., 2017a; Dimitriadis et al., 2017b; Online Resource 1, Supplementary Section S1.2, Supplementary Figure S2). Graph adjacency inputs (`A`) to message-passing neural networks do not require positive semi-definiteness (`PSD`). To test instead whether symmetrized `C = I_19 + PLI` matrices can serve directly as `SPD` inputs for Riemannian tangent-space embeddings, eigenvalues were audited across all `675` primary sessions alongside Euclidean `171-D` PLI vectors and `SPD`-conditioned PLI tangent projections (`diagonal loading >= 0.01`; `eigen-clipped >= 10^-4`). In parallel, `19 × 19` sample spatial covariance matrices `Sigma = (1/T) X X^T + 10^-5 (tr(Sigma)/19) I_19` (`1.0–40.0 Hz`, `250 Hz` clean epochs) were mapped into `190-D` tangent space (Barachant et al., 2013; Congedo et al., 2017). Because tangent mappings relative to `I_19` (Log-Euclidean metric) are invariant under linear centering shifts (zero divergence under training-fold Log-Euclidean mean centering), I additionally benchmarked **true fold-wise Affine-Invariant Riemannian Metric (`AIRM`) Fréchet-mean congruence whitening** (`P_train^(-1/2) Sigma_i P_train^(-1/2)`, fitting the iterative Karcher/Fréchet mean strictly inside each outer training fold across `10×5` CV with and without fold-wise z-scoring, versus an intentionally leaky global-AIRM reference; Online Resource 1, Supplementary Table S21 Panel C).

### 2.5 10×5 Validation, Compact Deep Baseline, Learning Curves, and Random-Effects `TOST`
An explicit stratified participant-cluster splitter shuffled unique participants across `10 repeats × 5 folds` (`50` outer folds; `0.0%` participant overlap), with healthy-control normative z-scoring (`z_ij = (x_ij - mu_HC_train) / sd_HC_train`) (Rutherford et al., 2022) executed strictly inside each outer training split. Five static, fixed-hyperparameter 3,362-parameter `NormativeGCN` topologies (`A + I`, `4->32->32` layers, mean+max readout, Adam `lr = 0.005`, `weight_decay = 1e-4`, dropout `0.25`, `50` fixed epochs) were compared across `12` contrasts: `GCN-OMST`, `GCN-Weighted-PLI` (`A_bin ⊙ PLI`), `GCN-Identity` (`A = I`), `GCN-Random*` (the **session edge-count-matched Erdős–Rényi random graph baseline**), and `GCN-Full` (Online Resource 1, Supplementary Tables S7, S7b, S10, S20). Alongside six inner 3-fold `GridSearchCV`-tuned classical 76-D models (`Tuned Logistic Regression`, `Tuned Ridge`, `Linear SVM`, `RBF SVM`, 100-tree `Random Forest` depth 5, `GBDT` depth 4) and univariate `Single Theta` / `Single Chi`, I evaluated three additional controls across all `k = 8` non-nested contrasts (`Supplementary Table S7c`): (a) `GCN-OMST-Tuned` (inner 3-fold selection across `lr in {0.002, 0.005, 0.01}`, `hidden in {16, 32}`, `dropout in {0.1, 0.25}` strictly inside training folds); (b) **`ShallowSpecCNN`**, a compact `1,770`-parameter factorized spectral-spatial convolutional baseline (`19 × 57` `[2.0, 30.0] Hz` standardized log-PSD tensors, depthwise spatial + temporal-spectral filters after EEGNet/ShallowConvNet; Lawhern et al., 2018; Schirrmeister et al., 2017); and (c) `5×5` sample-size learning curves (`25%, 50%, 75%, 100%` training-fold fractions on Studies 1, 3, 7). Both session-level and participant-averaged OOF ROC-AUCs were computed for repeated-visit PD cohorts (`Studies 4` and `6c`; Supplementary Table S10c).

Crucially, because repeated `10×5` cross-validation folds share training and test samples, computing standard errors strictly across repeat means underestimates variance. For every per-contrast paired AUC difference (`GCN-OMST` minus comparator), I therefore applied the Nadeau & Bengio (2003) corrected repeated CV variance estimator combining within-repeat fold variance (scaled by `1/K + n_te / n_tr`) and between-repeat variance (`t` critical values with `df = 9` for `90%` / `95%` intervals), and pooled per-contrast `TOST` equivalence effects (`±0.02` and `±0.04` AUC bounds; Schuirmann, 1987; Lakens, 2017) across `k = 8` and `k = 7` sensitivity sets using DerSimonian & Laird (1986) random-effects meta-analysis (Hartung & Knapp, 2001 adjustment where applicable) with Holm (1979) multiplicity correction across the four graph-topology comparisons (Supplementary Table S10b Panels B–C). Aligned 1,000-permutation retraining tests (Supplementary Figure S6) and 10×5 session-mixing leakage ablations (`Strict Leak-Free`, `Leaky Pre-Standardization`, `Leaky Subject Mixing`) audited validation integrity in Studies 1, 4, and 6c.

### 2.6 Disjoint Transfer, Gain Stress-Testing, Max-Stat Regressions, and Meta-Analysis
Six directed PD cross-site pairs and eight cross-disorder pairs used `20` random 50/50 `Split-Target-HC` splits estimating target HC statistics strictly on one disjoint control half (Oracle references, calibration metrics, and uniform `k = 0.5–5.0` / mixed `0.5x/5.0x` gain stress-testing in Online Resource 1, Supplementary Tables S15, S16, and S19). For MMSE (Study 5) and MoCA/UPDRS-III (Study 7), 10×5 OOF regressions (`GBDT`, `Random Forest`, `RBF SVR`, `Ridge`) were paired with exact 1,000-permutation K-fold retraining sharing identical label permutations (`seeds 20000..20999`) across all four regressors, yielding both single-model Split-0 two-sided `p`-values and paired 4-model max-statistic `p`-values (`max_m r_null >= r_obs`; `max_m |r_null| >= |r_obs|`; Supplementary Table S14). Group comparisons used Welch/paired t-tests (`q_10`, `q_50`), and neurodegenerative periodic theta pooling used DerSimonian–Laird / `HKSJ` random effects (`k = 4` PD/dementia and `k = 3` eyes-open PD-only), noting that `I²` heterogeneity carries wide uncertainty at small `k` (von Hippel, 2015; full `K = 6`/`k = 2` sensitivity pooling in Supplementary Table S22).

### 2.7 Computational Environment, Open Artifacts, and AI-Assisted Software Workflow
Analyses executed in pinned `Python 3.13.7`, `PyTorch 2.8.0`, `MNE-Python 1.12.1`, `specparam 2.0.0rc7`, `scikit-learn 1.7.2`, `scipy 1.16.2`, and `statsmodels 0.14.6` (`release_artifacts/requirements_lock.txt`; Supplementary Table S9). Derived CSVs (`derived_session_features_733.csv`, `cv_fold_indices_split0.csv`, `oof_predictions_split0.csv`) carry unique `record_uid` primary keys across repeated UNM PD sessions. During codebase and manuscript preparation, the author used AI-assisted coding and editorial language tools (Antigravity / Gemini LLM coding assistants) to draft modular Python benchmarking scripts and summary builders, and directly audited all preprocessing mappings, fold-nested standardization boundaries, raw disk hashes, and statistical outputs against deterministic pipeline CSVs, assuming full responsibility for all code and scientific claims.

---

## 3. Results

### 3.1 Stage 1: Passband-Interior [2.0, 30.0] Hz Parameterization, Study 2 EDF Forensics, and Cohort Biophysics
Interior `[2.0, 30.0] Hz` `specparam v2.0.0rc7` achieved zero failed fits across all `13,927` unique spectra (median `R² = {fit230['r2_median'].min():.3f}–{fit230['r2_median'].max():.3f}`, `MAE = {fit230['mae_median'].min():.3f}–{fit230['mae_median'].max():.3f} B`; Table 1, Online Resource 1, Supplementary Table S23 and Supplementary Figure S4), matching stable `fooof v1.1.1` across all spectra (`r_chi = {ff_all['pearson_r_chi']:.4f}`, `r_theta = {ff_all['pearson_r_theta']:.4f}`; Supplementary Table S27 Panel B). Extending fit bounds into `[1.0, 40.0] Hz` selectively collapsed fits in Study 2 (`median R² = {fit140.loc['mumtaz_ec', 'r2_median']:.3f}`, `{fit140.loc['mumtaz_ec', 'pct_r2_lt_080']:.1f}% < 0.80` vs. `{fit230.loc['mumtaz_ec', 'r2_median']:.3f}` and `{fit230.loc['mumtaz_ec', 'pct_r2_lt_080']:.1f}%` at `[2.0, 30.0] Hz`), severely attenuating nominal theta (`d = {ft_w.loc['Study 2: Outpatient MDD', 'cohens_d_specparam_resid_1to40']:+.3f}` vs. `{ft_w.loc['Study 2: Outpatient MDD', 'cohens_d_specparam_resid_2to30']:+.3f}`; bootstrap Δd = `{b_s2_w['delta_d']:+.3f} [{b_s2_w['ci_lower']:+.3f}, {b_s2_w['ci_upper']:+.3f}], p = {fmt_p(b_s2_w['boot_p'])}`) and inverting `χ` (`{chi_w.loc['Study 2: Outpatient MDD', 'cohens_d_specparam_resid_1to40']:+.3f}` vs. `{chi_w.loc['Study 2: Outpatient MDD', 'cohens_d_specparam_resid_2to30']:+.3f}`; Table 2, Fig. 2a). Across the 25-configuration 2D grid (`105` evaluations; `0` failures; Supplementary Table S27 Panel A), Studies 3 (`-0.388 to -0.571`), 5c (`+0.923 to +1.105`), and 7 (`+0.807 to +0.897`) remained boundary-invariant, whereas in Study 2 raising `f_max` (`30 -> 45 Hz`) even on raw unfiltered EDF spectra inverted `χ d` monotonically (`-0.945 -> +1.803`) and flipped `[1.0, 45.0] Hz` theta from `{g_s2_230['theta_d']:+.3f}` to `{g_s2_45['theta_d']:+.3f}`. Conversely, in clean BioSemi Study 3 spectra (`R² = {fit230.loc['cavanagh_ds003478', 'r2_median']:.3f}`), `[7, 14] Hz`-masked OLS inflated negative theta estimates (`{ft_w.loc['Study 3: Univ. Depression Screening', 'cohens_d_masked_ols_1to40']:+.3f}` vs. `{ft_w.loc['Study 3: Univ. Depression Screening', 'cohens_d_specparam_resid_2to30']:+.3f}`; Δd = `{b_s3_o['delta_d']:+.3f} [{b_s3_o['ci_lower']:+.3f}, {b_s3_o['ci_upper']:+.3f}], p = {fmt_p(b_s3_o['boot_p'])}`). Mechanistic, by-construction Monte Carlo demonstrations (`150` trials × `N = 29 vs. 27`; Fig. 2b, Supplementary Table S11) confirmed both failure directions (`{sim_s2_ols['theta_d_bias']:+.3f} d` OLS bias in S2 vs. `{sim_s2_230['theta_d_bias']:+.3f} d`; `{sim_s3_140['est_theta_d_mean']:+.3f}` theta flip under edge roll-off in S3 vs. true `{sim_s3_230['est_theta_d_mean']:+.3f}`).

Raw archive forensics across all `181` EDF files in Figshare `4244171` (Mumtaz et al., 2017; Fig. 2c, Supplementary Table S26 Panels A–D) revealed two physical mechanisms uniting batch-date confounding with spectral roll-off. After SHA-256 deduplication (`H S27` = `H S30`, `MDD S33` = `MDD S34`), leaving `N = 56` unique EC (`29 MDD vs. 27 HC`, `d_theta = {s2_for.loc['Study 2a EC: Deduplicated Unique (29 MDD vs 27 HC, N=56)', 'theta_d']:+.3f}`), raw `0.5–80 Hz` EDF spectra—despite identical `HP:0.5 Hz LP:80 Hz` headers everywhere—showed that `Jan-2013` HCs (`n = 14`) combined anomalous `6 Hz` excess power (`mean {s2_b.loc['HC_2013_01', 'logp_6hz']:.3f} B` vs. `{s2_b.loc['HC_2011', 'logp_6hz']:.3f} B` in `2011` HC and `{s2_b.loc['HC_2013_04_07', 'logp_6hz']:.3f} B` / `{s2_b.loc['MDD_2013_04_07', 'logp_6hz']:.3f} B` in Apr–Jul 2013) with `30–50 Hz` high-frequency noise (`30->40 Hz` rise `{s2_b.loc['HC_2013_01', 'drop_30_to_40hz']:+.3f} B` and unnotched `50 Hz` spike `+{s2_b.loc['HC_2013_01', 'line_50hz_excess']:.2f} B`), creating an artificial `d_theta = +3.347` (`p < 1e-4`) between `Jan-2013` and `2011` healthy controls alone. Simultaneously, both `Apr–Jul 2013` MDD (`n = 29`, `1->2 Hz: {s2_b.loc['MDD_2013_04_07', 'drop_2_to_1hz']:+.3f} B`, `30->40 Hz: {s2_b.loc['MDD_2013_04_07', 'drop_30_to_40hz']:+.3f} B`) and the `4` date-matched `Apr–Jul 2013` HCs (`{s2_b.loc['HC_2013_04_07', 'drop_2_to_1hz']:+.3f} B`, `{s2_b.loc['HC_2013_04_07', 'drop_30_to_40hz']:+.3f} B`) shared pre-export spectral roll-off absent from `2011` (`{s2_b.loc['HC_2011', 'drop_30_to_40hz']:+.3f} B`). Removing `Jan-2013` HCs abolished Study 2 periodic theta separation (`d = {s2_for.loc['Study 2a EC: Deduplicated Excl. Jan-2013 HC Batch (29 MDD vs 13 HC, N=42)', 'theta_d']:+.3f}`, 95% CI `[-0.737, +0.572]`, `p = {s2_for.loc['Study 2a EC: Deduplicated Excl. Jan-2013 HC Batch (29 MDD vs 13 HC, N=42)', 'theta_p']:.4f}`; vs. `2011` HCs `d = -0.161`; vs. `4` date-matched `Apr–Jul 2013` HCs `d = +0.104`), leaving batch-linked `χ` differences (`d = -1.48 to -1.61`) to sustain `RF AUC = {s2_for.loc['Study 2a EC: Deduplicated vs Apr-Jul 2013 Date-Matched HC (29 MDD vs 4 HC, N=33)', 'rf_10x5_mean']:.3f}–{s2_for.loc['Study 2a EC: Deduplicated vs 2011 HC Batch Only (29 MDD vs 9 HC, N=38)', 'rf_10x5_mean']:.3f}`; EO `N = 58` mirrored elevated delta `d = {eb_m_eo_dl['cohens_d']:+.3f}` and null `χ` `p = {eb_m_eo_ch['p_value']:.4f}`). Because zero MDD cases were recorded in `2011` or `Jan-2013`, batch and diagnosis cannot be statistically separated within Study 2; this archive audit highlights the need to pair the valuable public archive of Mumtaz et al. (2017) with explicit acquisition-date stratification in cross-cohort ML benchmarks.

{t1}

{t2}

Across remaining cohorts free of acquisition-batch corruption (Table 1, Fig. 1a), `[2.0, 30.0] Hz` frontal periodic theta elevation appeared in acute sleep deprivation (`d_z = {ft_w.loc['Study 1: Acute Sleep Deprivation', 'cohens_d_specparam_resid_2to30']:+.3f}, p = {fmt_p(ft_w.loc['Study 1: Acute Sleep Deprivation', 'p_value_specparam_resid_2to30'])}`; global theta `d_z = {eb_sl['cohens_d']:+.3f}`), dementia (`AD d = {ft_w.loc["Study 5a: Alzheimer's Disease (AD)", 'cohens_d_specparam_resid_2to30']:+.3f}`, plus posterior alpha loss `d = {ad_pa['cohens_d']:+.3f}` and steeper `χ` `d = {ad_gc['cohens_d']:+.3f}`; `FTD d = {ft_w.loc['Study 5b: Frontotemporal Dementia (FTD)', 'cohens_d_specparam_resid_2to30']:+.3f}`; `Combined d = {ft_w.loc['Study 5c: Clinical Dementia (All)', 'cohens_d_specparam_resid_2to30']:+.3f}`), and Iowa PD (`d = {ft_w.loc['Study 7: Iowa Parkinson Cohort', 'cohens_d_specparam_resid_2to30']:+.3f}, p = {fmt_p(ft_w.loc['Study 7: Iowa Parkinson Cohort', 'p_value_specparam_resid_2to30'])}`, alongside true posterior `IAPF` slowing `{iowa_iapf['diff_hz']:+.3f} Hz`, `p = {fmt_p(iowa_iapf['p_value'])}`; Supplementary Table S24), whereas smaller PD studies had 95% CIs spanning zero (`UCSD SubjMean d = +0.528 [-0.189, +1.244]`; `UNM PD-OFF EO d = +0.427 [-0.134, +0.988]`; `UNM PD-ON EO d = +0.633 [+0.065, +1.201]`), while student BDI screening Study 3 showed lower periodic theta (`d = -0.464, p = 0.0161`). Effects persisted in `F3–Fz–F4` clusters and joint age/epoch ANCOVAs (Supplementary Tables S17, S25). Recomputing Study 7 across all `149` participants without reconstructing `Pz` (`CAR18_NoPz`: `theta d = {s7_pz.loc['CAR18_NoPz', 'frontal_theta_d']:+.3f}`, `IAPF d = {s7_pz.loc['CAR18_NoPz', 'iapf_d']:+.3f}`) or via spline `Pz` (`CAR19_SplinePz`: `theta d = {s7_pz.loc['CAR19_SplinePz', 'frontal_theta_d']:+.3f}`) matched reference reconstruction (`CAR19_RefReconPz`: `theta d = {s7_pz.loc['CAR19_RefReconPz', 'frontal_theta_d']:+.3f}`; Supplementary Table S28). In Study 6, pure-EC rest (`N = 74`) amplified PD theta separation (`OFF d = {unm_sens_df.loc['UNM Pure EC (N=25 PD-OFF vs 24 CTL)', 'theta_d']:+.3f}` vs. EO `{unm_sens_df.loc['UNM Pure EO (N=25 PD-OFF vs 25 CTL)', 'theta_d']:+.3f}`), and paired ON–OFF medication contrasts were null (`UNM p = {unm_med_ft['paired_p']:.4f}`; `UCSD p = {ucsd_med_ft['paired_p']:.4f}`; Supplementary Table S13). Meta-analytic pooling confirmed replicated neurodegenerative frontal periodic theta elevation across both `k = 4` PD/dementia (`d_RE = {ndg_m['pooled_d']:+.3f}`, HKSJ 95% CI `[{ndg_m['hksj_ci_lower']:+.3f}, {ndg_m['hksj_ci_upper']:+.3f}]`, `p_HKSJ = {fmt_p(ndg_m['p_hksj'])}`, `I² = {ndg_m['I2_percent']:.1f}%`) and state-homogeneous eyes-open PD-only cohorts (`k = 3`: `d_RE = {loo_dem_m['pooled_d']:+.3f}`, HKSJ 95% CI `[{loo_dem_m['hksj_ci_lower']:+.3f}, {loo_dem_m['hksj_ci_upper']:+.3f}]`, `p_HKSJ = {fmt_p(loo_dem_m['p_hksj'])}`, `I² = {loo_dem_m['I2_percent']:.1f}%`; Supplementary Table S22).

![Fig. 1](Figure_1.png){{width=6.5in}}
**Fig. 1** Canonical interior `[2.0, 30.0] Hz` parameterized biophysical effect-size landscape (`specparam`) across ten contrasts (`564` unique primary participants) showing Cohen's `d` (`95%` confidence intervals) for **(a)** five-channel frontal periodic theta (`4–8 Hz`), **(b)** 19-channel low-frequency periodic residual (`2–4 Hz`), **(c)** five-channel posterior periodic alpha (`8–13 Hz`), and **(d)** 19-channel global aperiodic exponent `χ` (`* q_50 < 0.05`, `** q_50 < 0.01`, `*** q_50 < 0.001`, `† q_10 < 0.05`). *Study 2 denotes the de-duplicated unique sample (`29 MDD vs. 27 HC`). Neurodegenerative PD and dementia cohorts replicate positive frontal periodic theta elevation under meta-analytic pooling, whereas Study 2's negative shift reflects acquisition-batch confounding

![Fig. 2](Figure_2.png){{width=6.5in}}
**Fig. 2** Spectral estimator confounding, mechanistic by-construction demonstrations, and Study 2 raw EDF batch forensics showing **(a)** frontal periodic theta effect sizes (`Cohen's d`) across `[7, 14] Hz`-masked ordinary least squares (`OLS`), filter-skirt `[1, 40] Hz` `specparam`, and canonical `[2, 30] Hz` `specparam`, **(b)** calibrated synthetic demonstration bias (`150` trials/scenario) illustrating spurious OLS theta negativity (`-0.32 d` in S2) and `[1, 40] Hz` roll-off polarity inversion (`+2.98 d` in S3), and **(c)** unfiltered raw EDF batch spectra (`0.5–80 Hz` computed, `0.5–65 Hz` displayed; `mean ± SEM`) exposing Apr–Jul 2013 pre-export attenuation in MDD (`n = 29`) and date-matched controls (`n = 4`) versus unfiltered `2011` (`n = 9`) and `50 Hz`-contaminated `Jan-2013` controls (`n = 14`)

### 3.2 Stage 2: GNN Equivalence, `ShallowSpecCNN`, Learning Curves, and `AIRM` Audits
Across `675` primary OMST graphs (`{omst['total_node_instances']:,}` nodes; Online Resource 1, Supplementary Tables S1 and S4), spanning trees averaged `{omst['mean_trees']:.2f} ± {omst['sd_trees']:.2f}` (`{omst['mean_edges']:.2f} ± {omst['sd_edges']:.2f}` edges, `GCE = {omst['mean_gce']:.4f} ± {omst['sd_gce']:.4f}`, zero case–control GCE shifts `q_50 >= 0.087`). Under true 10×5 participant-grouped CV across `k = 8` non-nested contrasts (Table 3, Fig. 3a, Supplementary Tables S7, S7b, S10, S20), `GCN-OMST` averaged `{rep_nn8['GCN_OMST'].mean():.3f}` (`{rep_p9['GCN_OMST'].mean():.3f}` across `k = 9`; `{mac.loc['k=7 Excl. Study 2 Batch-Confounded', 'GCN_OMST']:.3f}` excluding Study 2). Using per-contrast `Nadeau & Bengio (2003)` variance and random-effects `TOST` pooling (Supplementary Table S10b Panels B–C), `GCN-OMST` met `±0.02` equivalence (`Holm p <= {vt8.loc['GCN_Weighted_PLI', 'tost02_holm_gnn4']:.4f}`) against `GCN-Random*` (`{rep_nn8['GCN_Random_EdgeMatched'].mean():.3f}`, Δ = `{vt8.loc['GCN_Random_EdgeMatched', 'mean_diff_auc']:+.4f}`, 90% CI `[{vt8.loc['GCN_Random_EdgeMatched', 'ci90_lo']:+.4f}, {vt8.loc['GCN_Random_EdgeMatched', 'ci90_hi']:+.4f}]`, `Holm p = {vt8.loc['GCN_Random_EdgeMatched', 'tost02_holm_gnn4']:.4f}`, `8/8` contrasts within `±0.02`), `GCN-Full` (`{rep_nn8['GCN_Full'].mean():.3f}`, Δ = `{vt8.loc['GCN_Full', 'mean_diff_auc']:+.4f} [{vt8.loc['GCN_Full', 'ci90_lo']:+.4f}, {vt8.loc['GCN_Full', 'ci90_hi']:+.4f}]`, `Holm p = {vt8.loc['GCN_Full', 'tost02_holm_gnn4']:.4f}`, `8/8` within `±0.02`), and `GCN-Weighted-PLI` (`{rep_nn8['GCN_Weighted_PLI'].mean():.3f}`, Δ = `{vt8.loc['GCN_Weighted_PLI', 'mean_diff_auc']:+.4f} [{vt8.loc['GCN_Weighted_PLI', 'ci90_lo']:+.4f}, {vt8.loc['GCN_Weighted_PLI', 'ci90_hi']:+.4f}]`, `Holm p = {vt8.loc['GCN_Weighted_PLI', 'tost02_holm_gnn4']:.4f}`, `6/8` within `±0.02`), persisting against random and dense graphs across `k = 7` drop-5b, drop-5a, and excluding-Study-2 sensitivity sets (`Holm p <= {vt7_nos2.loc['GCN_Random_EdgeMatched', 'tost02_holm_gnn4']:.4f}`). Crucially, however, `GCN-Identity` (`{rep_nn8['GCN_Identity'].mean():.3f}`) exhibited high between-cohort heterogeneity (`I² = {vt8.loc['GCN_Identity', 'i2_percent']:.1f}%`; Δ spanning `{vt8_id_c.loc['Study 5b: Dementia FTD vs Control', 'mean_diff_auc']:+.4f}` in Study 5b to `{vt8_id_c.loc['Study 3: Cavanagh University Depression', 'mean_diff_auc']:+.4f}` in Study 3, only `3/8` contrasts within `±0.02`) and therefore **failed `±0.02` equivalence** under proper Nadeau–Bengio random-effects testing (Δ = `{vt8.loc['GCN_Identity', 'mean_diff_auc']:+.4f}`, 90% CI `[{vt8.loc['GCN_Identity', 'ci90_lo']:+.4f}, {vt8.loc['GCN_Identity', 'ci90_hi']:+.4f}]`, `Holm p = {vt8.loc['GCN_Identity', 'tost02_holm_gnn4']:.4f}`), passing only at `±0.04` (`Holm p = {vt8.loc['GCN_Identity', 'tost04_holm_gnn4']:.4f}` on `k = 8`).

Furthermore, neither inner-tuned `GCN-OMST-Tuned` (`k = 8` mean `{vdeep['GCN_OMST_Tuned']:.3f}`, gaining only `+0.003` over fixed `GCN-OMST`; Supplementary Table S7c Panel B) nor compact end-to-end spectral-spatial `ShallowSpecCNN` (`{vdeep['ShallowSpecCNN']:.3f}`, 1,770 parameters) matched tuned classical 76-D models (`Linear SVM {rep_nn8['Linear_SVM'].mean():.3f}`, `Tuned LR {rep_nn8['Tuned_LogisticRegression'].mean():.3f}`, `Random Forest {rep_nn8['RandomForest'].mean():.3f}`, `Tuned Ridge {rep_nn8['Tuned_Ridge'].mean():.3f}`). Sample-size learning curves (`25% -> 100%` training data; Supplementary Table S7c Panel C) confirmed that across `N = 119–149` cohorts, `GCN-OMST` tracked `GCN-Random*` almost identically at every sample size (`Study 1: {vlc.loc[('Study 1: Sleep Deprivation (ds004902)', 0.25, 'GCN_OMST'), 'mean_auc']:.3f} -> {vlc.loc[('Study 1: Sleep Deprivation (ds004902)', 1.0, 'GCN_OMST'), 'mean_auc']:.3f}` vs. random `{vlc.loc[('Study 1: Sleep Deprivation (ds004902)', 0.25, 'GCN_Random_EdgeMatched'), 'mean_auc']:.3f} -> {vlc.loc[('Study 1: Sleep Deprivation (ds004902)', 1.0, 'GCN_Random_EdgeMatched'), 'mean_auc']:.3f}`; `Study 7: {vlc.loc[('Study 7: Iowa Parkinson (ds004584)', 0.25, 'GCN_OMST'), 'mean_auc']:.3f} -> {vlc.loc[('Study 7: Iowa Parkinson (ds004584)', 1.0, 'GCN_OMST'), 'mean_auc']:.3f}` vs. `{vlc.loc[('Study 7: Iowa Parkinson (ds004584)', 0.25, 'GCN_Random_EdgeMatched'), 'mean_auc']:.3f} -> {vlc.loc[('Study 7: Iowa Parkinson (ds004584)', 1.0, 'GCN_Random_EdgeMatched'), 'mean_auc']:.3f}`), while `Linear SVM` overtook `GCN-OMST` at full Study 1 data (`{vlc.loc[('Study 1: Sleep Deprivation (ds004902)', 1.0, 'Linear_SVM'), 'mean_auc']:.3f}`) and `ShallowSpecCNN` trailed `Random Forest` throughout (`Study 7` at `100%`: `{vlc.loc[('Study 7: Iowa Parkinson (ds004584)', 1.0, 'ShallowSpecCNN'), 'mean_auc']:.3f}` vs. `{vlc.loc[('Study 7: Iowa Parkinson (ds004584)', 1.0, 'RandomForest'), 'mean_auc']:.3f}`). Session-level and participant-averaged OOF AUCs agreed closely across repeated-visit PD cohorts (`Study 4 GCN-OMST`: `{ps_df.loc[('Study 4: UCSD Parkinson All Sessions', 'GCN_OMST'), 'participant_mean_auc_10x5']:.3f}` vs. `{ps_df.loc[('Study 4: UCSD Parkinson All Sessions', 'GCN_OMST'), 'session_mean_auc_10x5']:.3f}`; `Study 6c Linear SVM`: `{ps_df.loc[('Study 6c: UNM PD All Sessions', 'Linear_SVM'), 'participant_mean_auc_10x5']:.3f}` vs. `{ps_df.loc[('Study 6c: UNM PD All Sessions', 'Linear_SVM'), 'session_mean_auc_10x5']:.3f}`; Supplementary Table S10c), Study 7 remained invariant under `18`-channel CAR (`GCN-OMST {s7_pz.loc['CAR18_NoPz', 'gcn_omst_auc']:.3f}` vs. `{s7_pz.loc['CAR19_RefReconPz', 'gcn_omst_auc']:.3f}`), and aligned 1,000-permutation tests (Supplementary Figure S6) confirmed above-chance decoding in Study 5c (`p <= {perms['A']['p_perm_gcn']:.4f}`) and Study 7 (`p <= {perms['B']['p_perm_gcn']:.4f}`) while `GCN-OMST` fell inside the null in Study 6a (`{perms['C']['obs_gcn_auc']:.3f}, p = {perms['C']['p_perm_gcn']:.4f}`).

{t3}

Geometric and leakage audits explained these limits (Fig. 3b–c, Supplementary Tables S18 and S21). First, `{int(pli_s['n_indefinite_raw'])}/{int(pli_s['n_sessions'])}` (`{pli_s['pct_indefinite_raw']:.2f}%`) symmetrized `I_19 + PLI` matrices were indefinite (`min λ = {pli_s['min_eigenvalue']:.4f}`), so neither `SPD`-conditioned PLI tangent models (`{riem_nn8.loc[('Riemannian_Higham_SPD_190D', 'LR')]:.3f}–{riem_nn8.loc[('Riemannian_Gershgorin_SPD_190D', 'RF')]:.3f}`) nor Euclidean `171-D` PLI (`{riem_nn8.loc[('Euclidean_UpperTri_171D', 'LR')]:.3f}`) escaped chance. Second, true `SPD` `19 × 19` sample spatial covariance features achieved `0.724` (`RF`), `0.718` (`Ridge`), and `0.710` (`LR`) under Log-Euclidean identity tangent mapping (`0.710 RF` excluding Study 2), and remained competitive under **strict fold-wise `AIRM` Fréchet-mean congruence whitening** (`Foldwise_AIRM_Frechet_Z`: `RF {vairm_m.loc[('Foldwise_AIRM_Frechet_Z', 'RF')]:.3f}`, `Ridge {vairm_m.loc[('Foldwise_AIRM_Frechet_Z', 'Ridge')]:.3f}`, `LR {vairm_m.loc[('Foldwise_AIRM_Frechet_Z', 'LR')]:.3f}`; exposing `+0.0043` `RF` optimistic inflation under leaky global AIRM whitening [`{vairm_m.loc[('Leaky_Global_AIRM_Raw', 'RF')]:.3f}`]; Supplementary Table S21 Panel C). Third, 10×5 session mixing shifted AUC bidirectionally by label concordance (`Study 1 RF {leak.loc[('Study 1: Sleep Deprivation', 'Leaky_Subject_Mixing', 'RandomForest'), 'mean_delta_vs_strict']:+.3f} [95% CI {leak.loc[('Study 1: Sleep Deprivation', 'Leaky_Subject_Mixing', 'RandomForest'), 'delta_ci_lo']:+.3f}, {leak.loc[('Study 1: Sleep Deprivation', 'Leaky_Subject_Mixing', 'RandomForest'), 'delta_ci_hi']:+.3f}]` vs. `Study 4 PD RF {leak.loc[('Study 4: UCSD Parkinson', 'Leaky_Subject_Mixing', 'RandomForest'), 'mean_delta_vs_strict']:+.3f} [{leak.loc[('Study 4: UCSD Parkinson', 'Leaky_Subject_Mixing', 'RandomForest'), 'delta_ci_lo']:+.3f}, {leak.loc[('Study 4: UCSD Parkinson', 'Leaky_Subject_Mixing', 'RandomForest'), 'delta_ci_hi']:+.3f}]`).

![Fig. 3](Figure_3.png){{width=6.5in}}
**Fig. 3** Architectural equivalence, bidirectional leakage, and connectivity definiteness across 19-channel EEG benchmarks showing **(a)** 10×5 repeated participant-grouped ROC-AUC (`mean ± SD`) across `k = 8` non-nested contrasts including inner-tuned `GCN-OMST-Tuned`, compact `ShallowSpecCNN`, and fold-wise `AIRM` Fréchet `Spatial Cov RF` (`GCN-Random*` denotes the session edge-count-matched Erdős–Rényi random graph baseline), **(b)** 10×5 mean `ΔAUC` (`± 95%` confidence intervals) under session mixing relative to strict participant grouping, and **(c)** cohort percentages of symmetrized `I_19 + PLI` matrices that are indefinite for direct symmetric positive-definite Riemannian manifold embedding (`{int(pli_s['n_indefinite_raw'])}/{int(pli_s['n_sessions'])} = {pli_s['pct_indefinite_raw']:.2f}%`)

### 3.3 Stage 3: Leak-Free Transfer, Gain Calibration Trade-Offs, and Clinical Severity Floor
Analytically and numerically, `[2.0, 30.0] Hz` `specparam` preserved `chi` and periodic residuals under multiplicative gain scaling (`gamma in [0.01, 100.0]`, `285` spectra, `max |Δchi| = {gchk['max_abs_diff_chi'].max():.2e}`; Online Resource 1, Supplementary Table S2 and Supplementary Figure S3). In uncalibrated 6-pair PD transfer (`Supplementary Table S19`), unparameterized `log10` power suffered mean probability drift `{amp_raw_5x['lr_mean_prob_drift'].mean():.3f}` (`max = {amp_raw_5x['lr_max_prob_drift'].max():.3f}`) under `5x` gain (`LR AUC {cal_raw_lr_raw:.3f} -> {amp_raw_mix['lr_auc'].mean():.3f}`), whereas `specparam` had zero gain drift (`LR = {cal_raw_lr_sp:.3f}` vs. relative power `{cal_raw_lr_rel:.3f}`), while target-HC z-scoring traded balanced accuracy (`{cal_raw_bacc_sp:.3f} -> {cal_thc_lr_sp['lr_bacc'].mean():.3f}`) for higher Brier error (`{cal_thc_lr_sp['lr_brier'].mean():.3f}`). Under disjoint 20-split `Split-Target-HC` transfer (Table 4, Fig. 4a, Supplementary Tables S15–S16), `Iowa PD -> UNM PD-OFF` (`Tuned LR {pd_tx.loc[('Iowa_PD', 'UNM_PD_OFF'), 'Tuned_LR']:.3f}`) and `Iowa PD -> UCSD PD` (`LR {pd_tx.loc[('Iowa_PD', 'UCSD_PD'), 'Tuned_LR']:.3f}`) transferred moderately well, elevated-theta cohorts cross-generalized (`Sleep <-> Iowa PD <-> Dementia`, `RF {td_tx.loc[[('Sleep_Debt', 'Iowa_PD'), ('Sleep_Debt', 'Dementia_All'), ('Iowa_PD', 'Dementia_All')], 'RandomForest'].min():.3f}–{td_tx.loc[[('Sleep_Debt', 'Iowa_PD'), ('Sleep_Debt', 'Dementia_All'), ('Iowa_PD', 'Dementia_All')], 'RandomForest'].max():.3f}`), and transfer into batch-confounded Study 2 collapsed (`GCN-OMST {td_tx.loc[[('Sleep_Debt', 'Mumtaz_MDD'), ('Mumtaz_MDD', 'Dementia_All'), ('Dementia_All', 'Mumtaz_MDD')], 'GCN_OMST'].min():.3f}–{td_tx.loc[[('Sleep_Debt', 'Mumtaz_MDD'), ('Mumtaz_MDD', 'Dementia_All'), ('Dementia_All', 'Mumtaz_MDD')], 'GCN_OMST'].max():.3f}`, `RF {td_tx.loc[[('Sleep_Debt', 'Mumtaz_MDD'), ('Mumtaz_MDD', 'Dementia_All'), ('Dementia_All', 'Mumtaz_MDD')], 'RandomForest'].min():.3f}–{td_tx.loc[[('Sleep_Debt', 'Mumtaz_MDD'), ('Mumtaz_MDD', 'Dementia_All'), ('Dementia_All', 'Mumtaz_MDD')], 'RandomForest'].max():.3f}`; Study 2 EO sensitivity in Supplementary Table S12, EC–EO theta `r = {mum_conc.loc['Frontal Periodic Theta (4–8 Hz Residual)', 'pearson_r']:+.3f}`).

{t4}

![Fig. 4](Figure_4.png){{width=6.5in}}
**Fig. 4** Leak-free cross-disorder generalization and within-patient clinical severity limits displaying **(a)** `Split-Target-HC` cross-disorder `Random Forest` ROC-AUC (`*` denotes diagonal 10×5 within-cohort repeated cross-validation), **(b)** dementia `Split-0` MMSE out-of-fold predictions (`Random Forest`: `r = {dem_all_rf['split0_r']:.3f}` all [`N = 88`] vs. `r = {dem_pt_rf['split0_r']:.3f}` [`OOF R² = {dem_pt_rf['r2_mean']:.3f}`] patients only [`N = 59`]; compare Table 5 Selected Models `SVR RBF Split-0 r = {dem_all_svr['split0_r']:.3f}` [`10×5 r = {dem_all_svr['pearson_r_mean']:.3f}`] and `Ridge Split-0 r = {dem_pt_ridge['split0_r']:.3f}` [`10×5 r = {dem_pt_ridge['pearson_r_mean']:.3f}`, `OOF R² = {dem_pt_ridge['r2_mean']:.3f}`]), and **(c)** 10×5 out-of-fold severity correlations (`mean ± SD`) annotated with Table 5 Selected Model 4-model max-statistic two-sided permutation `p`-values (`p_max`)

Paired 4-model max-statistic 1,000-permutation regressions revealed severe control-ceiling inflation and a within-patient variance floor (Table 5, Fig. 4b–c, Online Resource 1, Supplementary Table S14). Including healthy ceilings (`MMSE = 30.0`, `N = 88`) yielded strong MMSE prediction (`SVR RBF 10×5 r = {dem_all_svr['pearson_r_mean']:+.3f}`, `Split-0 r = {dem_all_svr['split0_r']:+.3f}`, `max-stat 2-S perm p = {dem_all_svr['maxstat4_perm_p_2s']:.4f}`, `OOF R² = {dem_all_svr['r2_mean']:+.3f}`), which collapsed upon restricting to `N = 59` AD/FTD patients: across all four patient-only MMSE regressors, `10×5 Mean OOF R²` spanned **`{reg_pt_mmse['r2_mean'].min():.3f}` to `{reg_pt_mmse['r2_mean'].max():.3f}`** (`Ridge 10×5 r = {dem_pt_ridge['pearson_r_mean']:+.3f}`, `Split-0 r = {dem_pt_ridge['split0_r']:+.3f}`, `OOF R² = {dem_pt_ridge['r2_mean']:+.3f}`, `max-stat 2-S p = {dem_pt_ridge['maxstat4_perm_p_2s']:.4f}`; `RF Split-0 r = {dem_pt_rf['split0_r']:+.3f}`, `OOF R² = {dem_pt_rf['r2_mean']:+.3f}`; all `perm p >= 0.111`). In Iowa PD (`N = 149: MoCA SVR 10×5 r = {iowa_all_svr['pearson_r_mean']:+.3f}`, `max-stat 2-S p = {iowa_all_svr['maxstat4_perm_p_2s']:.4f}`), restricting to `N = 100` PD patients preserved only weak rank correlation (`SVR 10×5 r = {iowa_pt_svr['pearson_r_mean']:+.3f}`, `2-S max-stat p = {iowa_pt_svr['maxstat4_perm_p_2s']:.4f}`, `OOF R² = {iowa_pt_svr['r2_mean']:+.3f}`; `Ridge OOF R² = {iowa_pt_ridge['r2_mean']:+.3f}`, invariant under `CAR18_NoPz` `{s7_pz.loc['CAR18_NoPz', 'moca_svr_r']:+.3f}`), while `UPDRS-III` showed weak borderline `GBDT` tree signal (`10×5 r = {iowa_updrs_gbdt['pearson_r_mean']:+.3f}`, `max-stat 1-S/2-S p = {iowa_updrs_gbdt['maxstat4_perm_p_1s']:.4f} / {iowa_updrs_gbdt['maxstat4_perm_p_2s']:.4f}`, `OOF R² = {iowa_updrs_gbdt['r2_mean']:+.3f}`, `RF 2-S p = {iowa_updrs_rf['perm_p_split0_2s']:.4f}`), with every patient-only model capped at `OOF R² <= {reg[reg['cohort_key'].str.contains('Patients')]['r2_mean'].max():.3f}` (`10/12` negative).

{t5}

---

## 4. Discussion

### 4.1 Neuroinformatics Resource Contributions and Three-Stage Methodological Lessons
By organizing `{acc['total_unique_participants']}` unique participants (`{acc['total_analyzed_sessions']}` primary sessions; `{acc['total_spectra'] // 19}` total recordings) into one sequential three-stage verification workflow, this resource isolates where published qEEG and graph-learning claims break down. Across Stage 1, interior `[2.0, 30.0] Hz` `specparam` fits protect effect sizes against `[7, 14] Hz`-masked OLS bias and passband roll-off, while raw EDF forensics uncover hidden duplicates and batch-date roll-off confounding Study 2. Across Stage 2 (`N = 31–149` clinical cohorts), proper Nadeau–Bengio random-effects `TOST` confirms `±0.02` equivalence between `GCN-OMST` and random/dense/WPLI graphs (while exposing `GCN-Identity` cross-cohort heterogeneity), and neither inner-tuned `GCN-OMST-Tuned` (`0.708`) nor `ShallowSpecCNN` (`0.709`) surpasses linear baselines (`0.721`) or fold-wise `AIRM` Fréchet spatial covariances (`0.718`). Across Stage 3, `11.70%` `PLI` indefiniteness, bidirectional session-mixing bias, and the `<= 3.5%` patient-only `OOF R²` severity floor delineate clear translational boundaries.

### 4.2 Stage 1 Discussion: Novel Pre-Export Roll-Off Forensics and Affective vs. Neurodegenerative Distinctions
Extending Donoghue et al. (2022) and Gerster et al. (2022), who diagnosed known software filter corners and synthetic shoulder distortions, this benchmark shows that multi-center EDF releases can harbor **header-invisible upstream roll-off** (`HP:0.5 Hz LP:80 Hz` stamped identically on distorted `Apr–Jul 2013` and flat `2011` recordings) alongside `Jan-2013` `6 Hz` and `50 Hz` noise bursts (`d_theta = +3.347` between healthy control batches alone). Rather than detracting from the open legacy of Mumtaz et al. (2017), documenting these exact file hashes and recording-date strata empowers future benchmarks to stratify Study 2 responsibly instead of mistaking acquisition-batch `χ` and spectral roll-off (`RF AUC ~ 0.80–0.88`) for pathophysiological depression biomarkers (Mahato & Paul, 2020; Saeedi et al., 2021). Equally important, Study 3 (`ds003478`) represents a BDI-II undergraduate symptom screen rather than clinically confirmed MDD, reinforcing why mood-disorder EEG synthesis must separate clinical diagnoses from analogue symptom cohorts (Knott et al., 2001; Olbrich & Arns, 2013; Newson & Thiagarajan, 2019). By contrast, neurodegenerative periodic theta elevation replicated meta-analytically across both `k = 4` PD/dementia (`d_RE = {ndg_m['pooled_d']:+.3f}, p_HKSJ = {fmt_p(ndg_m['p_hksj'])}`) and state-homogeneous eyes-open PD-only studies (`k = 3`: `d_RE = {loo_dem_m['pooled_d']:+.3f}, p_HKSJ = {fmt_p(loo_dem_m['p_hksj'])}`), even where smaller single studies (`UCSD`, `UNM`) spanned zero, accompanied by genuine posterior `IAPF` slowing in Iowa PD (Stoffers et al., 2007; Cavanagh et al., 2018; Anjum et al., 2024) and invariance to `Pz` reconstruction (`CAR18_NoPz`).

### 4.3 Stage 2 Discussion: Scope of GNN Equivalence, Compact CNNs, and `AIRM` vs. `PLI` Geometry
Strictly scoped to small clinical 19-channel resting cohorts (`N = 31–149`), these findings do not contradict deep-learning scaling on massive clinical archives (`N > 1,000`; Roy et al., 2019; Gemein et al., 2020). Within `N <= 149`, however, accounting properly for `10×5` fold correlation (`Nadeau & Bengio, 2003`) reveals a nuanced picture: structured sparse `OMST` edges (`GCN-OMST`) offer zero classification benefit over `GCN-Random*` (`±0.02` Holm `p = 0.0036`) or dense `GCN-Full` (`p = 0.0028`), whereas eliminating message passing altogether (`GCN-Identity`) increases variance (`I² = {vt8.loc['GCN_Identity', 'i2_percent']:.1f}%`, inconclusive at `±0.02`, passing at `±0.04`). Thus, neighborhood averaging can stabilize predictions on specific contrasts without encoding disease-specific connectivity topology—a conclusion reinforced by identical learning-curve trajectories for `GCN-OMST` and `GCN-Random*` from `25%` to `100%` data and the inability of tuned GCNs (`0.708`) or `ShallowSpecCNN` (`0.709`) to beat simple regularized linear SVMs (`0.721`).

Meanwhile, contrasting `PLI` with `19 × 19` sample spatial covariance clarifies manifold geometry: `11.70%` indefiniteness in `I_19 + PLI` precludes naive `SPD` projection without conditioning, whereas broadband spatial amplitude covariance excels under both translation-invariant Log-Euclidean mapping (`0.724 RF`) and strict fold-wise `AIRM` Fréchet-mean congruence whitening (`0.718 RF`), even after removing `+0.0043` leaky global whitening inflation. Alongside bidirectional session-mixing distortion (`-0.094` vs. `+0.080`), these audits establish essential validation baselines for graph and geometric EEG learners (Saeb et al., 2017; Brookshire et al., 2024).

### 4.4 Stage 3 Discussion: Transfer Limits, Clinical Severity Floor, and Study Limitations
External `Split-Target-HC` transfer achieved moderate accuracy (`{pd_tx.loc[('Iowa_PD', 'UCSD_PD'), 'Tuned_LR']:.3f}–{pd_tx.loc[('Iowa_PD', 'UNM_PD_OFF'), 'Tuned_LR']:.3f}`) from the largest PD site (`N = 149`) while highlighting small-control target calibration variance (Rutherford et al., 2022). Critically, max-statistic permutations prove that apparent MMSE regression performance (`r = +0.431`) rests on healthy controls at ceiling (`MMSE = 30.0`): inside dementia patients (`N = 59`), every model failed (`perm p >= 0.111`, `OOF R² = -0.110 to -0.047`), and patient-only Iowa PD `MoCA` / `UPDRS-III` models explained `<= 3.5%` variance (`10/12` negative). Key limitations include small-to-moderate cohort sample sizes (`N = 31–149`) limiting deep-network capacity, channel harmonization to 19 electrodes, missing individual ages in Study 2, short `~41 s` clean segments in Study 6, wide `I²` uncertainty at small `k` (von Hippel, 2015), and primary execution on release candidate `specparam v2.0.0rc7` (cross-verified across all `13,927` spectra against `fooof v1.1.1`).

---

## 5. Conclusions

Across `{acc['total_unique_participants']}` unique participants (`{acc['total_analyzed_sessions']}` primary sessions; `{acc['total_spectra'] // 19}` total recordings) from seven public archives, this three-stage neuroinformatics benchmark establishes three actionable conclusions. First, passband-interior `[2.0, 30.0] Hz` `specparam` fits protect against masked-OLS and roll-off distortions, while raw EDF forensics uncover hidden duplicates and batch-date roll-off confounding Study 2 alongside replicated neurodegenerative theta elevation (`k = 4 d_RE = +0.787`; eyes-open PD `k = 3 d_RE = +0.709`). Second, in small clinical 19-channel cohorts (`N = 31–149`), neither static nor tuned `GCN-OMST` nor compact `ShallowSpecCNN` surpasses random graphs within `±0.02` AUC (`Nadeau–Bengio Holm p = 0.0036`), tuned linear baselines (`0.721`), or fold-wise `AIRM` spatial covariance `RF` (`0.718`). Third, `11.70%` `PLI` indefiniteness, bidirectional session-mixing leakage, and a `<= 3.5%` patient-only `OOF R²` severity ceiling define concrete reproducibility guardrails for computational EEG.

---

## Information Sharing Statement

All seven datasets remain permanently accessible under their original repository DOIs and licenses: **Study 1** (`ds004902`, CC0: https://doi.org/10.18112/openneuro.ds004902.v1.1.0; Xiang et al., 2024); **Study 2** (`4244171`, CC BY 4.0: https://doi.org/10.6084/m9.figshare.4244171.v5; Mumtaz et al., 2017); **Study 3** (`ds003478`, CC0: https://doi.org/10.18112/openneuro.ds003478.v1.1.0; Cavanagh et al., 2019); **Study 4** (`ds002778`, CC0: https://doi.org/10.18112/openneuro.ds002778.v1.0.5; Rockhill et al., 2020); **Study 5** (`ds004504`, CC0: https://doi.org/10.18112/openneuro.ds004504.v1.0.6; Miltiadous et al., 2023); **Study 6** (`ds003490`, CC0: https://doi.org/10.18112/openneuro.ds003490.v1.0.0; Cavanagh et al., 2018); and **Study 7** (`ds004584`, CC0: https://doi.org/10.18112/openneuro.ds004584.v1.0.0; Singh et al., 2023). All Python benchmarking scripts (`src/`, `scripts/47_run_v5_forensics_and_refinements.py`, `scripts/49_run_v6_sensitivities.py`, `scripts/50_build_paper2_neuroinformatics_v6.py`, `scripts/51_generate_v6_figures.py`, `scripts/52_run_v7_reviewer_remediations.py`), `CITATION.cff`, `requirements_lock.txt`, derived features (`release_artifacts/derived_session_features_733.csv`), participant-grouped fold splits (`cv_fold_indices_split0.csv`), and out-of-fold predictions (`oof_predictions_split0.csv`) are archived publicly at `{GITHUB_URL}` (release tag `{RELEASE_TAG}`; MIT License for code, CC BY 4.0 for derived tables per `release_artifacts/DATA_LICENSES.md`), with full supplementary tables and figures embedded directly in `Online Resource 1`.

---

## Statements and Declarations

### Funding
This research received no specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

### Competing Interests
The author declares no financial or non-financial competing interests.

### Ethics Approval and Consent to Participate
This computational resource study performed secondary analysis exclusively on de-identified, publicly accessible open-access EEG archives and involved no direct human-subjects recruitment or intervention. Ethical approval and written informed consent for original data collection were obtained by the respective primary investigators at Southwest University (`H20039`, `ds004902`), Universiti Sains Malaysia (`USM/JEPeM/1401015`, Figshare `4244171`), University of New Mexico IRB (`ds003478` and `ds003490`), UC San Diego HRPP (`ds002778`), AHEPA University Hospital (`142/12-04-2023`, `ds004504`), and University of Iowa IRB (`201707828`, `ds004584`).

### Consent for Publication
Not applicable (secondary computational analysis of de-identified public archives; no identifiable individual participant data or images are published).

### Author Contributions (CRediT)
Chushu Tan: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing – original draft, Writing – review & editing, Visualization, Project administration.

---

## References

{refs_md}
"""
    return md


# ==============================================================================
# Online Resource 1 (Supplementary Tables S1-S28 + Figs S1-S6)
# ==============================================================================
def build_supplementary_material(d: dict) -> str:
    omst, acc = d["master"]["omst_topology"], d["master"]["accounting"]
    s1_df, gain_chk, bio_exp, lofo, hparam = d["s1_omst"], d["gain_chk"], d["bio_exp"], d["lofo"], d["hparam"]
    cls0, sim_mc, boot_d, mum_eo = d["cls0"], d["sim_mc"], d["boot_d"], d["mum_eo"]
    mum_conc, mum_tx, med, reg, pd_tx, td_tx = d["mum_conc"], d["mum_tx"], d["med"], d["reg"], d["pd_tx"], d["td_tx"]
    anc, anc_dur, leak, gc, rep10 = d["ancova"], d["anc_dur"], d["leakage"], d["gain_cal"], d["rep10"]
    pli_a, pli_s, riem, meta_df, fit_df = d["pli_audit"], d["pli_sum"].iloc[0], d["riem"], d["meta"], d["fit"]
    iapf_df, heur_df, ep_df, unm_sens = d["iapf"], d["heur"], d["ep_audit"], d["unm_sens"]
    mum_dups, mum_for, s2_unf, grid2d, feq = d["mum_dups"], d["mum_forensic"], d["s2_unfilt"], d["grid2d"], d["fooof_eq"]
    s7_pz, part_sess, tost_v6, macro_v6 = d["s7_pz"], d["part_sess"], d["tost_v6"], d["macro_v6"]
    scov_fm, s2_181, fooof_full = d["scov_fm"], d["s2_181"], d["fooof_full"]
    v7_tost, v7_airm, v7_deep, v7_lc = d["v7_tost"], d["v7_airm"], d["v7_deep"], d["v7_lc"]

    parts = [f"""# Online Resource 1: Supplementary Material

**Article Title:** {EXACT_TITLE}  
**Journal:** *{JOURNAL_NAME}* ({PUBLISHER_NAME})  
**Author:** {AUTHOR_NAME}^1,2*^  
^1^ {AFFIL_1}  
^2^ {AFFIL_2}  
^*^ Correspondence: {CORRESPONDENCE}

---

## Contents

### Extended Supplementary Methods
- **Section S1.1:** Interior [2.0, 30.0] Hz Parameterized Spectral Decomposition (`specparam v2.0.0rc7`) and 2D Grid Sensitivity
- **Section S1.2:** Orthogonal Minimum Spanning Tree (OMST) Graph Extraction and Cost Efficiency
- **Section S1.3:** Analytical Scaling Arithmetic and Non-Multiplicative Site Boundaries
- **Section S1.4:** `NormativeGCN` Architecture and Training Protocol

### Supplementary Figures (`Fig. S1–S6`)
- **Fig. S1:** Single-Subject Interior [2.0, 30.0] Hz `specparam` vs. Masked-OLS Illustration
- **Fig. S2:** Canonical 19-Channel 10–20 OMST Connectome Topology (`N = 675` primary graphs)
- **Fig. S3:** Empirical Gain-Invariance Verification Across Four Orders of Magnitude (`n = 285` spectra)
- **Fig. S4:** Grand-Average Welch Power Spectra (1–40 Hz) and [2.0, 30.0] Hz Residuals Across 7 Cohorts
- **Fig. S5:** Seven-Cohort Participant Flow, QC Exclusions, and Study 2 SHA-256 Deduplication
- **Fig. S6:** Aligned 1,000-Permutation Retraining Distributions (`GridSearchCV Tuned LR` and `GCN-OMST`)

### Supplementary Tables (`Table S1–S28`)
- **Table S1:** Canonical 19-Channel Coordinates and Empirical OMST Metrics Across `675` Primary Graphs
- **Table S2:** Empirical Gain-Invariance Audit Across `285` Spectra (`[2.0, 30.0] Hz` Solver)
- **Table S3:** Cohort Designs, Demographics, Hardware Systems, and Clinical Scales
- **Table S4:** Complete 50-Test Multi-Cohort Biophysical Summary (`[2.0, 30.0] Hz` `specparam`)
- **Table S5:** Canonical 19-Channel 10–20 Label Harmonization Mapping
- **Table S6:** Single-Biomarker Baselines and Leave-One-Feature-Out `NormativeGCN` Ablations (Split-0)
- **Table S7, S7b, S7c:** `NormativeGCN`, `GCN-OMST-Tuned`, Compact `ShallowSpecCNN` (`1,770` Params), Classical Grids, and Sample-Size Learning Curves (`25%–100%`)
- **Table S8:** Preprocessing Specifications Across All Seven Datasets
- **Table S9:** Verified Computational Environment (`requirements_lock.txt`) and Open Repository (`v6.0.0`)
- **Table S10, S10b, S10c:** Extended Model Comparison, Per-Contrast `Nadeau & Bengio (2003)` + Random-Effects `TOST` (`DL`/`HKSJ`), and Participant- vs. Session-Level AUCs (Studies 4 & 6c)
- **Table S11:** Empirical-Variance-Calibrated Monte Carlo Simulation and Bootstrap Paired Δd Contrasts
- **Table S12:** Study 2 Secondary Eyes-Open Benchmark (`N = 58` unique EO) and State-Matched Transfer
- **Table S13:** Within-Subject Paired Dopaminergic Medication Comparisons (`UNM N = 25` & UCSD `N = 15`)
- **Table S14:** Complete 20-Row 10×5 & Split-0 Clinical Scale Regression Benchmark With Paired Max-Stat Permutations
- **Table S15 & S16:** Leak-Free 50/50 `Split-Target-HC` Cross-Site PD and Cross-Domain Transfer Matrices
- **Table S17:** Complete Age-Adjusted ANCOVA Across Four EEG Markers
- **Table S18:** 10×5 Repeated Validation Leakage Quantification Across Repeated-Session Cohorts
- **Table S19:** Cross-Site PD Calibration Ablation and Gain Stress-Testing
- **Table S20:** True 10×5 Repeated Participant-Grouped Cross-Validation Summary Across All `12` Contrasts × `13` Models
- **Table S21:** PLI Matrix Definiteness Audit (`675` Sessions), Spatial Covariance Riemannian Baselines, and Fold-Wise `AIRM` Fréchet-Mean Congruence Whitening Check
- **Table S22:** Random-Effects Meta-Analysis (`DL`, `HKSJ`, `K = 6` & `K = 5` Omit Study 2) and Leave-One-Out Suite
- **Table S23:** Per-Cohort `specparam v2.0.0rc7` Diagnostics (`[1.0, 40.0]` vs. `[2.0, 30.0] Hz`)
- **Table S24:** Posterior Individual Alpha Peak Frequency (`IAPF`) and 10×5 Heuristic Slowing Benchmarks
- **Table S25:** Clean Epoch Balance, Joint Age + Duration ANCOVA, and UNM Pure EO vs. Pure EC Sensitivity
- **Table S26:** Figshare `4244171` (Study 2) Duplicate Hashing, Batch Sensitivity, Unfiltered EDF Profiles, and Full `181`-File Accounting
- **Table S27:** Complete 25-Configuration 2D Grid (105 Cohort Evaluations across `[1.0–3.0] × [25.0–45.0] Hz`) and Full `13,927`-Spectrum Stable `fooof v1.1.1` Verification
- **Table S28:** Study 7 (`Iowa PD ds004584`, `N = 149`) `Pz` Online-Reference CAR vs. `18`-Channel Real CAR vs. Spherical-Spline `Pz` Sensitivity

---

## Section S1. Extended Supplementary Methods

### S1.1 Interior [2.0, 30.0] Hz Parameterized Spectral Decomposition (`specparam v2.0.0rc7`)
Welch PSDs (`4.0 s` epochs, `2.0 s` Hamming sub-windows, `50%` overlap, `0.50 Hz` step) were parameterized via `SpectralGroupModel` in `specparam v2.0.0rc7` (`aperiodic_mode='fixed'`, `peak_width_limits=[1.0, 12.0]` Hz, `max_n_peaks=6`, `min_peak_height=0.05` B, `peak_threshold=2.0` SD) over primary `[2.0, 30.0] Hz`, edge `[1.0, 40.0] Hz`, and a 25-configuration 2D sensitivity grid (`105` cohort evaluations across `[1.0–3.0] × [25.0–45.0] Hz`):

$$\\log_{{10}} \\mathrm{{PSD}}(f) = (b - \\chi \\log_{{10}} f) + \\sum_{{n=1}}^N a_n \\exp\\left(-\\frac{{(f - c_n)^2}}{{2 w_n^2}}\\right)$$

### S1.2 Data-Driven Connectomics via Orthogonal Minimum Spanning Trees (OMST)
Across `675` canonical primary graphs (`{omst['total_node_instances']:,}` nodes), iterative Kruskal OMST trees maximizing `GCE` averaged `{omst['mean_trees']:.2f} ± {omst['sd_trees']:.2f}` trees (`{omst['pct_tree_1']:.2f}%` `M = 1`, `{omst['pct_tree_2']:.2f}%` `M >= 2`), `{omst['mean_edges']:.2f} ± {omst['sd_edges']:.2f}` edges, `GCE = {omst['mean_gce']:.4f} ± {omst['sd_gce']:.4f}`, and `0.0%` isolated nodes.

### S1.3 Elementary Gain Scaling and Non-Multiplicative Boundaries
Multiplying time-series voltage by scalar `k > 0` adds `2 log10(k)` uniformly across `log10` Welch power, absorbed analytically into offset `b' = b + 2 log10(k)`. Upstream roll-offs, line noise, references, and demographic batch shifts are non-multiplicative and require archive auditing and fold-nested standardization.

### S1.4 `NormativeGCN` Architecture
Pure-PyTorch `NormativeGCN` (`src/harmonizer_and_graph.py`) processes `[N_ch, 4]` z-scored features and `N_ch × N_ch` adjacency (`A + I`, evaluated both as binary `GCN_OMST` and PLI-weighted `GCN_Weighted_PLI = A_bin ⊙ PLI`), applying `4->32->32` symmetric-normalized graph convolutions (Dropout `0.25`), concatenated mean+max pooling (`64-D`), and `64->32->2` MLP classification totaling `3,362` parameters (Adam `lr=0.005`, `wd=1e-4`, `50` fixed epochs).

---

## Supplementary Figures

![Fig. S1](fig_S1_specparam_example.png){{width=6.3in}}
**Fig. S1** Single-subject real spectrum (`Study 5` AD participant `sub-018`, `Ch Fz`) illustrating **(a)** `[2.0, 30.0] Hz` `specparam` versus `[7, 14] Hz`-masked OLS fit (`χ_OLS = 1.287` vs. `specparam χ = 1.074`) and **(b)** flattened periodic residuals showing OLS attenuation in the `4–8 Hz` theta band (`R² = 0.954`, `MAE = 0.083 B`)

![Fig. S2](fig_S2_omst_topology.png){{width=6.0in}}
**Fig. S2** Canonical 19-channel 10–20 OMST connectome topology across `675` primary graphs (`{omst['mean_edges']:.2f}` edges, `0.0%` isolated nodes)

![Fig. S3](fig_S3_gain_invariance.png){{width=6.2in}}
**Fig. S3** Empirical scaling stability across `285` EEG spectra (`k in [0.01, 100.0]`) showing **(a)** exact `2 log10(k)` aperiodic offset tracking and **(b)** `< {gain_chk['max_abs_diff_chi'].max():.2e}` parameter drift

![Fig. S4](fig_S4_cohort_spectra.png){{width=6.5in}}
**Fig. S4** Cohort-averaged Welch power spectra (`1–40 Hz`) and interior `[2.0, 30.0] Hz` periodic residuals across **(a)–(g)** all seven primary cohorts (Study 2 de-duplicated: `29 MDD vs. 27 HC EC`)

![Fig. S5](fig_S5_enrollment_flowchart.png){{width=6.4in}}
**Fig. S5** Seven-cohort participant accounting, pre-analysis QC exclusions, Study 2 `181`-EDF auditing (`H S15 EO` ID-prefixed skip and `61` excluded TASK files), Study 6 pure-EO isolation, and Study 7 `18`-channel sensitivity flow

![Fig. S6](fig_S6_permutation_tests.png){{width=6.5in}}
**Fig. S6** Aligned 1,000-permutation retraining null distributions across **(a)–(d)** four within-cohort benchmarks and **(e)–(f)** two `Split-Target-HC` transfer pairs using identical `GridSearchCV Tuned LR` and participant-cluster splits

---

## Supplementary Tables

### Table S1. Canonical 19-Channel Coordinates and Empirical OMST Metrics Across All `675` Primary Graphs
| Channel | Region | X | Y | T1 Deg | T2 Deg | Comb Deg | Empirical Mean Degree | Betweenness Centrality | Clustering Coeff |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"""]

    for _, r in s1_df.iterrows():
        parts.append(f"| {r['channel']} | {r['region']} | {r['proj_x']:+.2f} | {r['proj_y']:+.2f} | {int(r['tree1_deg'])} | {int(r['tree2_deg'])} | {int(r['combined_deg'])} | {r['emp_deg_mean']:.2f} ± {r['emp_deg_sd']:.2f} | {r['betw_mean']:.3f} ± {r['betw_sd']:.3f} | {r['clus_mean']:.3f} ± {r['clus_sd']:.3f} |")
    parts.append(f"| Total / Mean | 19 Channels | — | — | 18 edges | 18 edges | 36 edges | {omst['mean_edges']:.2f} ± {omst['sd_edges']:.2f} edges | — | — |")

    parts.append("""
### Table S2. Empirical Scaling Verification of Interior [2.0, 30.0] Hz `specparam` Across 285 Spectra
| Gain Factor (k) | Spectra (n) | Expected Offset 2 log10(k) (B) | Measured Offset Shift Mean (B) | Max Abs(Δχ) Across 285 Spectra | Max Abs(Δθ_resid) (B) |
| :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in gain_chk.iterrows():
        parts.append(f"| {r['gain_k']:.2f} | {int(r['n_spectra'])} | {r['expected_offset_shift']:+.5f} | {r['measured_offset_shift_mean']:+.5f} | {r['max_abs_diff_chi']:.2e} | {r['max_abs_diff_theta_resid']:.2e} |")

    parts.append("""
### Table S3. Multi-Cohort Study Designs, Demographics, Hardware Systems, and Clinical Scales
| Study ID | Dataset & Setting | Analyzed N (Pts / Sess) | Group Breakdown | Age Years (Mean ± SD) | Female (%) | Native Hardware | Sampling Rate | Protocol / Assessments |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- | :---: | :--- |
| Study 1 | Sleep Deprivation (`ds004902`) | 71 / 142 | SD (71) vs. NS (71) [Paired] | 20.0 ± 1.4 | 47.9% | BrainVision actiCHamp (64 Ch) | 500 → 250 Hz | 36 h monitored wakefulness crossover (EO) |
| Study 2 | Outpatient MDD (`Figshare 4244171`) | 56 / 56 (+58 EO uniq.) | 29 MDD vs. 27 HC EC (31 vs. 27 EO) | MDD 40.3 ± 12.9; HC 38.2 ± 15.6 (cohort summary) | 37.9% | 19-ch clinical linked-ear (`-LE`) system | 256 → 250 Hz | DSM-IV MDD; 2 SHA-256 dup pairs removed + 3 batch regimes |
| Study 3 | Univ. Depression (`ds003478`) | 119 / 119 | BDI >= 14 (44) vs. <= 7 (75) | 19.4 ± 2.0 (N=118) | 60.5% | BioSemi ActiveTwo (64 Ch) | 500 → 250 Hz | EO screening; SCID (11 curr, 11 past MDD, 8 non-MDD, 14 unint.) |
| Study 4 | UCSD PD (`ds002778`) | 31 / 46 | PD (15 ON + 15 OFF) vs. HC (16) | PD 63.2 ± 8.1; HC 63.5 ± 9.6 | 54.8% | BioSemi ActiveTwo (32 Ch) | 512 → 250 Hz | EO fixation; counterbalanced ON vs. >= 12 h OFF |
| Study 5 | AHEPA Dementia (`ds004504`) | 88 / 88 | AD (36), FTD (23), HC (29) | AD 72.8 ± 7.9; FTD 68.2 ± 7.3; HC 70.1 ± 6.8 | 55.7% | Nihon Kohden (19 Ch) | 500 → 250 Hz | EC; MMSE (AD 17.8 ± 4.5, FTD 22.2 ± 4.8, HC 30.0) |
| Study 6 | UNM PD (`ds003490`) | 50 / 75 | PD (25 ON + 25 OFF EO) vs. HC (25 EO) | PD 67.4 ± 8.7; HC 69.1 ± 8.2 | 42.0% | BrainVision (64 Ch) | 500 → 250 Hz | Pure 60s EO rest (+74 pure EC sens); >= 15 h washout |
| Study 7 | Iowa PD (`ds004584`) | 149 / 149 | PD (100 ON) vs. HC (49) | PD 68.5 ± 8.1; HC 70.9 ± 7.6 | 42.3% | BrainVision (64 Ch, Pz ref) | 500 → 250 Hz | EO (PD ON med); MoCA (25.1 ± 3.6) & UPDRS-III (12.5 ± 7.1) |

### Table S4. Complete 50-Test Multi-Cohort Biophysical Summary (`[2.0, 30.0] Hz` `specparam`)
| Cohort | Marker | N (Case/Ref) | t-Stat | Exact df | Raw p | Pooled FDR q_50 | Per-Marker q_10 | Cohen's d [95% CI] | Sig |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in bio_exp.iterrows():
        parts.append(f"| {r['cohort']} | {r['marker']} | {int(r['n_case'])}/{int(r['n_ctrl'])} | {r['t_statistic']:+.3f} | {r['df_satterthwaite']:.1f} | {fmt_p(r['p_value'])} | {fmt_p(r['fdr_q_50'])} | {fmt_p(r['fdr_q_10'])} | {r['cohens_d']:+.3f} [{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}] | {fmt_sig(r['fdr_q_50'])} |")

    parts.append("""
### Table S5. Canonical 19-Channel 10–20 Label Harmonization Mapping Across Seven Datasets
| Canonical 10–20 | Region | Study 1 (64 Ch) | Study 2 (19 Ch) | Study 3 (64 Ch) | Study 4 (32 Ch) | Study 5 (19 Ch) | Study 6 (64 Ch) | Study 7 (64 Ch) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fp1 / Fp2 | Frontopolar | Direct | `EEG Fp*-LE` | Direct | Direct | `EEG Fp*` | Direct | Direct |
| F7/F3/Fz/F4/F8 | Frontal | Direct | `EEG F*-LE` | Direct | Direct | `EEG F*` | Direct | Direct |
| T7 / T8 | Mid-Temporal | Direct | `T3 / T4` mapped | Direct | Direct | `T3 / T4` mapped | Direct | Direct |
| C3 / Cz / C4 | Central | Direct | `EEG C*-LE` | Direct | Direct | `EEG C*` | Direct | Direct |
| P7 / P8 | Post-Temporal | Direct | `T5 / T6` mapped | Direct | Direct | `T5 / T6` mapped | Direct | Direct |
| P3 / P4 / O1 / O2 | Parietal / Occipital | Direct | `-LE` suffix | Direct | Direct | `EEG` prefix | Direct | Direct |
| Pz | Parietal Midline | Direct | `EEG Pz-LE` | Direct | Direct | `EEG Pz` | Direct | Online-ref CAR reconstr. |

### Table S6. Single-Biomarker Baselines and Leave-One-Feature-Out `NormativeGCN` Ablations (Split-0)
| Cohort Contrast | Single θ | Single δ | Single α | Single χ | Single GCE | GCN Full | Ablate θ | Ablate δ | Ablate α | Ablate χ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in lofo.iterrows():
        parts.append(f"| {r['cohort']} | {r['Single_Theta_auc']:.3f} | {r['Single_Delta_auc']:.3f} | {r['Single_Alpha_auc']:.3f} | {r['Single_Chi_auc']:.3f} | {r['Single_GCE_auc']:.3f} | {r['GCN_Full4_auc']:.3f} | {r['GCN_Ablate_Theta_auc']:.3f} | {r['GCN_Ablate_Delta_auc']:.3f} | {r['GCN_Ablate_Alpha_auc']:.3f} | {r['GCN_Ablate_Chi_auc']:.3f} |")

    parts.append("""
### Table S7. `NormativeGCN` Architecture and Optimization Specifications
| Component | Exact Specification | Parameter Breakdown |
| :--- | :--- | :--- |
| Engine & Module | PyTorch 2.8.0 (`NormativeGCN`) | `src/harmonizer_and_graph.py` |
| Input Tensor | `N_ch` nodes × 4 z-scored features | Fold-nested control standardization |
| GCN Layers | 2× `PurePyTorchGCNLayer` (`4->32->32`, ReLU, Dropout 0.25) | 160 + 1,056 = 1,216 params (`GCN-OMST` & `GCN-WPLI`) |
| Global Readout | Concatenated Mean + Max Pool across nodes | 64-D representation |
| MLP Head | `Linear(64->32) -> ReLU -> Dropout(0.25) -> Linear(32->2)` | 2,080 + 66 = 2,146 params |
| Optimization Scope | 3,362 trainable parameters; Fixed 50 Adam epochs (`lr=0.005`, `wd=1e-4`) | Evaluated on final epoch without early stopping or inner tuning |

### Table S7b. Classical and Univariate Baseline Configurations
| Model Identifier | Input Space | Inner Tuning / Settings | Estimator |
| :--- | :--- | :--- | :--- |
| `Tuned_LogisticRegression` | Flattened 76-D | Nested 3-fold CV L2 `C in {0.001..10.0}` | `LogisticRegression` |
| `Tuned_Ridge` | Flattened 76-D | Nested 3-fold CV `alpha in {0.01..100.0}` | `RidgeClassifier` |
| `Linear_SVM` / `RBF_SVM` | Flattened 76-D | Nested 3-fold CV `C`, `gamma`; `sigmoid(decision_function)` scoring | `SVC(probability=False)` |
| `RandomForest` | Flattened 76-D | 100 trees, `max_depth=5` | `RandomForestClassifier` |
| `GBDT` | Flattened 76-D | 100 iterations, `max_depth=4` | `HistGradientBoostingClassifier` |
| `Single_Theta` / `Single_Chi` | 1-D θ / χ | Default L2 (`C=1.0`) | `LogisticRegression` |

### Table S7c. Hyperparameter Sensitivity, Inner-Tuned `GCN-OMST-Tuned`, Compact `ShallowSpecCNN`, and Sample-Size Learning Curves
#### Panel A. Fixed-vs-Varied `NormativeGCN` Hyperparameter Sweep Across Five Cohorts (Split-0)
| Hyperparameter Configuration | Study 2 MDD | Study 3 Dep. | Study 4 UCSD | Study 6a UNM | Study 7 Iowa | 5-Cohort Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |""")
    hp_piv = hparam.pivot(index="config_label", columns="cohort", values="auc")
    hp_cols = list(hp_piv.columns)
    for cfg in hparam["config_label"].drop_duplicates():
        row_vals = [float(hp_piv.loc[cfg, c]) for c in hp_cols]
        parts.append(f"| {cfg} | " + " | ".join(f"{v:.3f}" for v in row_vals) + f" | {np.mean(row_vals):.3f} |")

    parts.append("")
    parts.append("#### Panel B. True 10×5 `k = 8` Extensions: Inner-Tuned `GCN-OMST-Tuned`, Compact `ShallowSpecCNN` (`1,770` Params), and Fold-Wise `AIRM` RF")
    parts.append("| Non-Nested Contrast | `GCN-OMST` | `GCN-OMST-Tuned` | `GCN-Rand*` | `ShallowSpecCNN` | `Linear SVM` | `RandomForest` | `Foldwise AIRM RF` |")
    parts.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    vdp = v7_deep.pivot(index="cohort", columns="model", values="mean_auc")
    vdp_mods = ["GCN_OMST", "GCN_OMST_Tuned", "GCN_Random_EdgeMatched", "ShallowSpecCNN", "Linear_SVM", "RandomForest", "Foldwise_AIRM_RF"]
    for cname in v7_deep["cohort"].drop_duplicates():
        vals = [float(vdp.loc[cname, m]) for m in vdp_mods]
        parts.append(f"| {cname} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    parts.append(f"| **Mean (`k = 8` Non-Nested)** | " + " | ".join(f"**{float(vdp[m].mean()):.3f}**" for m in vdp_mods) + " |")

    parts.append("")
    parts.append("#### Panel C. Sample-Size Learning Curves (`5×5` Participant-Grouped CV Across `25%–100%` Training-Fold Fractions)")
    parts.append("| Cohort (Total N) | Train Fraction (Mean `n_tr`) | `GCN-OMST` (±SD) | `GCN-Rand*` (±SD) | `ShallowSpecCNN` (±SD) | `Linear SVM` (±SD) | `RandomForest` (±SD) |")
    parts.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    lcp = v7_lc.pivot(index=["cohort", "n_total_samples", "train_fraction", "mean_n_train"], columns="model", values="mean_auc")
    lcs = v7_lc.pivot(index=["cohort", "n_total_samples", "train_fraction", "mean_n_train"], columns="model", values="sd_auc")
    for (cname, n_tot, frac, n_tr), row in lcp.iterrows():
        srow = lcs.loc[(cname, n_tot, frac, n_tr)]
        cells = [f"{float(row[m]):.3f} (±{float(srow[m]):.3f})" for m in ["GCN_OMST", "GCN_Random_EdgeMatched", "ShallowSpecCNN", "Linear_SVM", "RandomForest"]]
        parts.append(f"| {cname} (`N={int(n_tot)}`) | {int(round(frac * 100))}% (`{n_tr:.0f}`) | " + " | ".join(cells) + " |")

    parts.append("""
### Table S8. Preprocessing Specifications and Thresholds Across All Seven Datasets
| Dataset | Condition | Analyzed Sessions | Threshold | Sampling / Filter | Mean Clean Epochs ± SD [Range] | Retained Duration (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in ep_df.iterrows():
        thr = "200 μV" if "Iowa" in r["cohort"] else "150 μV"
        parts.append(f"| {r['cohort']} | EO/EC Rest | {int(r['n_sessions'])} | {thr} | 250 Hz, 1–40 Hz FIR | {r['all_mean']:.1f} ± {r['all_sd']:.1f} [{int(r['all_min'])}–{int(r['all_max'])}] | {4.0 * r['all_mean']:.1f} ± {4.0 * r['all_sd']:.1f} s |")

    parts.append(f"""
### Table S9. Verified Computational Environment, Versioning, and Layered Open Licensing (`v6.0.0`)
| Package / Artifact | Version / Path | License / Role |
| :--- | :---: | :--- |
| Python / PyTorch | 3.13.7 / 2.8.0 | Core execution, `NormativeGCN`, & `ShallowSpecCNN` (`MIT` scripts) |
| MNE-Python / `specparam` | 1.12.1 / 2.0.0rc7 | Preprocessing & spectral fits (verified across all 13,927 spectra vs. `fooof 1.1.1`) |
| `scikit-learn` / `scipy` / `statsmodels` | 1.7.2 / 1.16.2 / 0.14.6 | Nested CV, Nadeau–Bengio variance, Wilcoxon, `TOST`, ANCOVA, Holm/BH corrections |
| Source Code (`src/`, `scripts/`) | GitHub `{GITHUB_URL}` (`{RELEASE_TAG}`, incl. `scripts/52_run_v7_reviewer_remediations.py`) | **MIT License** (`requirements_lock.txt` pinned) |
| Derived Benchmark Archive (`release_artifacts/`) | `derived_session_features_733.csv`, `cv_fold_indices_split0.csv`, `oof_predictions_split0.csv` | **CC BY 4.0** (unique `record_uid` + `session` keys; embedded `Online Resource 1` tables) |

### Table S10. Split-0 and 10×5 Within-Cohort Benchmark Across All 13 Models
| Cohort Contrast | Model | N | 10×5 Mean AUC (±SD) | Split-0 ROC-AUC [95% CI] | Balanced Acc | Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |""")
    n_map_s10 = {**dict(zip(lofo["cohort"], lofo["n_samples"])), "Study 2b: Mumtaz MDD (Eyes Open)": 58}
    for _, r in cls0.iterrows():
        parts.append(f"| {r['cohort']} | `{r['model']}` | {int(n_map_s10[r['cohort']])} | {r['mean_auc_10x5']:.3f} (±{r['sd_auc_10x5']:.3f}) | {r['auc']:.3f} [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}] | {100.0 * r['balanced_accuracy']:.1f}% | {100.0 * r['accuracy']:.1f}% |")

    parts.append("""
### Table S10b. Shared-Control Dependency, Per-Contrast `Nadeau & Bengio` + Random-Effects `TOST`, and Excl-Study-2 Macro-AUCs
#### Panel A. Uncorrected Across-Contrast Paired `TOST` Reference (v6 Benchmark Comparison)
| Evaluation Scope | Comparator Model | Diff [90% Equivalence CI] | TOST ±0.02 p | Holm p (GNN 4) | TOST ±0.03 p | Holm ±0.03 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in tost_v6[tost_v6["is_gnn_family"]].iterrows():
        parts.append(
            f"| {r['scope']} (k={int(r['k_benchmarks'])}) | `{r['comparator_model']}` | "
            f"{r['mean_diff_auc']:+.4f} [{r['ci90_diff_lo']:+.4f}, {r['ci90_diff_hi']:+.4f}] | "
            f"{r['tost_p_002']:.4f} | {r['tost02_holm_gnn4']:.4f} | {r['tost_p_003']:.4f} | {r['tost03_holm_gnn4']:.4f} |"
        )
    parts.append("")
    parts.append("#### Panel B. Cross-Benchmark Macro-AUC Sensitivity Across Benchmark Scopes (`k=8`, `Participant-Level`, `Excl. Study 2`, `Drop 5b/5a`)")
    parts.append("| Benchmark Scope | k | `GCN_OMST` | `GCN_WPLI` | `GCN_Rand` | `GCN_Full` | `Tuned_LR` | `Linear_SVM` | `RandomForest` | `SpatialCov_RF` |")
    parts.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, r in macro_v6.iterrows():
        parts.append(f"| {r['scope']} | {int(r['k_contrasts'])} | {r['GCN_OMST']:.3f} | {r['GCN_Weighted_PLI']:.3f} | {r['GCN_Random_EdgeMatched']:.3f} | {r['GCN_Full']:.3f} | {r['Tuned_LogisticRegression']:.3f} | {r['Linear_SVM']:.3f} | {r['RandomForest']:.3f} | {r['SpatialCov_RF']:.3f} |")

    parts.append("")
    parts.append("#### Panel C. Primary `Nadeau & Bengio (2003)` Variance + `DerSimonian & Laird (1986)` Random-Effects `TOST` Pooling (`GCN_OMST` vs. Topology Comparators)")
    parts.append("| Scope | Comparator Model | RE ΔAUC [90% CI] | SE_RE | I² | Holm ±0.02 p | Holm ±0.04 p | Abs(Δ) <= 0.02 (`<= 0.04`) |")
    parts.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    vtp = v7_tost[(v7_tost["record_type"] == "Random_Effects_Pool") & (v7_tost["is_gnn_family"] == True)]
    for _, r in vtp.iterrows():
        parts.append(
            f"| {r['scope']} (k={int(r['k_contrasts'])}) | `{r['comparator_model']}` | "
            f"{r['mean_diff_auc']:+.4f} [{r['ci90_lo']:+.4f}, {r['ci90_hi']:+.4f}] | "
            f"{r['se_re_dl']:.4f} | {r['i2_percent']:.1f}% | "
            f"{r['tost02_holm_gnn4']:.4f} | {r['tost04_holm_gnn4']:.4f} | "
            f"{int(r['n_contrasts_abs_le_002'])}/{int(r['k_contrasts'])} ({int(r['n_contrasts_abs_le_004'])}/{int(r['k_contrasts'])}) |"
        )
    parts.append("")
    parts.append("#### Panel D. Per-Contrast `Nadeau & Bengio (2003)` Breakdown for `GCN_OMST` vs. Isolated-Node `GCN_Identity` (`I² = 79.6%`)")
    parts.append("| Contrast | Δ (`OMST − Identity`) | SE_NB | 90% CI | TOST ±0.02 p | TOST ±0.04 p |")
    parts.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    vtc_id = v7_tost[(v7_tost["record_type"] == "Per_Contrast") & (v7_tost["comparator_model"] == "GCN_Identity")]
    for _, r in vtc_id.iterrows():
        parts.append(
            f"| {r['cohort']} | {r['mean_diff_auc']:+.4f} | {r['se_nb']:.4f} | "
            f"[{r['ci90_lo']:+.4f}, {r['ci90_hi']:+.4f}] | {r['tost_p_002']:.4f} | {r['tost_p_004']:.4f} |"
        )

    parts.append("""
### Table S10c. True 10×5 Session-Level vs. Participant-Averaged ROC-AUC in Repeated-Session PD Cohorts (Studies 4 & 6c)
| Cohort | Model | Sess / Pts | Session AUC (±SD) | Participant AUC (±SD) | Δ (Part − Sess) | Split-0 Participant AUC [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in part_sess.iterrows():
        parts.append(f"| {r['cohort']} | `{r['model']}` | {int(r['n_sessions'])} / {int(r['n_participants'])} | {r['session_mean_auc_10x5']:.3f} (±{r['session_sd_auc_10x5']:.3f}) | {r['participant_mean_auc_10x5']:.3f} (±{r['participant_sd_auc_10x5']:.3f}) | {r['delta_part_minus_sess']:+.4f} | {r['participant_split0_auc']:.3f} [{r['participant_split0_ci_lo']:.3f}, {r['participant_split0_ci_hi']:.3f}] |")

    parts.append("""
### Table S11. Calibrated Monte Carlo Leakage Simulation (`n = 150 Trials`, `N = 29 vs. 27`) and Bootstrap Paired Δd Contrasts
#### Panel A. Known-Ground-Truth Monte Carlo Simulation Bias Across Spectral Estimators
| Scenario | Estimator | True θ d | Est θ d (±SD) | θ d Bias | Est χ d | χ d Bias | Raw χ Bias | Median R² |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in sim_mc.iterrows():
        r2s = f"{r['median_r2']:.3f}" if pd.notnull(r["median_r2"]) else "—"
        parts.append(f"| {r['scenario']} | `{r['estimator']}` | {r['true_theta_d']:+.3f} | {r['est_theta_d_mean']:+.3f} (±{r['est_theta_d_sd']:.2f}) | {r['theta_d_bias']:+.3f} | {r['est_chi_d_mean']:+.3f} | {r['chi_d_bias']:+.3f} | {r['chi_raw_bias']:+.3f} | {r2s} |")
    parts.append("")
    parts.append("#### Panel B. Empirical Stratified 1,000-Bootstrap Δd Contrasts (Frontal Periodic Theta)")
    parts.append("| Cohort Contrast | Comparison (Alt vs. SP 2–30) | Alt d | SP 2–30 d | Paired Δd [95% Boot CI] | Boot p |")
    parts.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
    for _, r in boot_d[boot_d["marker"] == "frontal_theta_5ch"].iterrows():
        parts.append(f"| {r['cohort']} | `{r['comparison']}` | {r['d_alt']:+.3f} | {r['d_sp_2_30']:+.3f} | {r['delta_d']:+.3f} [{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}] | {fmt_p(r['boot_p'])} |")

    parts.append("""
### Table S12. Study 2 Secondary Recording-State Sensitivity (Unique Eyes-Open `N = 58`) and State-Matched Transfer
#### Panel A. Eyes-Open Secondary Benchmark (Study 2b EO: 10×5 Means and Split-0 95% CIs)
| Model | 10×5 Mean AUC (±SD) | Split-0 ROC-AUC [95% CI] | Balanced Acc | Accuracy |
| :--- | :---: | :---: | :---: | :---: |""")
    for _, r in mum_eo.iterrows():
        parts.append(f"| `{r['model']}` | {r['mean_auc_10x5']:.3f} (±{r['sd_auc_10x5']:.3f}) | {r['auc']:.3f} [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}] | {100.0 * r['balanced_accuracy']:.1f}% | {100.0 * r['accuracy']:.1f}% |")
    parts.append("")
    parts.append("#### Panel B. Intra-Individual EC–EO Concordance (`N = 55`) & `Split-Target-HC` State Transfer (Mean ± SD)")
    parts.append("| Analysis / Pair | Metric | `GCN-OMST` (±SD) | `Tuned LR` (±SD) |")
    parts.append("| :--- | :---: | :---: | :---: |")
    for _, r in mum_conc.iterrows():
        parts.append(f"| Intra-Individual EC–EO (`N = {int(r['n_shared'])}`) | {r['marker']} | r = {r['pearson_r']:+.3f} (p={fmt_p(r['p_value'])}) | — |")
    for _, r in mum_tx.iterrows():
        parts.append(f"| {r['source_cohort']} -> {r['target_cohort']} | `Split-Target-HC` Transfer | {r['gcn_auc']:.3f} (±{r['gcn_sd']:.3f}) | {r['lr_auc']:.3f} (±{r['lr_sd']:.3f}) |")

    parts.append("""
### Table S13. Within-Subject Paired Dopaminergic Medication Comparisons (UNM Pure EO `N = 25` & UCSD `N = 15`)
| Cohort & Protocol | Marker | Mean OFF | Mean ON | Diff (ON - OFF) | Paired dz [95% CI] | Paired p | Wilcoxon p |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in med.iterrows():
        parts.append(f"| {r['cohort']} | {r['marker']} | {r['mean_off']:+.4f} | {r['mean_on']:+.4f} | {r['mean_diff']:+.4f} | {r['paired_dz']:+.3f} [{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}] | {fmt_p(r['paired_p'])} | {fmt_p(r['wilcoxon_p'])} |")

    parts.append("""
### Table S14. Complete 20-Row Clinical Scale Regression Benchmark (Single-Model & Paired 4-Model / 12-Model Max-Stat Permutations)
| Cohort & Scope | Scale | N | Regressor | 10×5 r (±SD) | Mean R² | Split-0 r | Split-0 2-S Perm p | Max-Stat 4M p (1-S / 2-S) | Family 12M p (2-S) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in reg.iterrows():
        f12 = fmt_p(r["maxstat12_pt_perm_p_2s"]) if pd.notnull(r["maxstat12_pt_perm_p_2s"]) else "—"
        parts.append(
            f"| {r['cohort']} | {r['scale']} | {int(r['n_samples'])} | `{r['model']}` | {r['pearson_r_mean']:+.3f} (±{r['pearson_r_sd']:.3f}) | "
            f"{r['r2_mean']:+.3f} | {r['split0_r']:+.3f} | {fmt_p(r['perm_p_split0_2s'])} | "
            f"{fmt_p(r['maxstat4_perm_p_1s'])} / {fmt_p(r['maxstat4_perm_p_2s'])} | {f12} |"
        )

    parts.append("""
### Table S15. Tri-Center PD `Split-Target-HC` External Transfer Matrix (20 Splits, w/ Oracle Full-HC Comparison)
| Source | Target | Model | Split-HC Mean AUC (±SD) | [p05, p95] Range | Bal Acc | Oracle Full-HC AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |""")
    for _, r in pd_tx.iterrows():
        parts.append(f"| {r['source_cohort']} | {r['target_cohort']} | `{r['model']}` | {r['auc']:.3f} (±{r['split_hc_sd']:.3f}) | [{r['split_hc_p05']:.3f}, {r['split_hc_p95']:.3f}] | {100.0 * r['balanced_accuracy']:.1f}% | {r['oracle_full_hc_auc']:.3f} |")

    parts.append("""
### Table S16. Cross-Disorder and Affective `Split-Target-HC` Transfer Matrix
| Transfer Type | Source -> Target | Model | Split-HC Mean AUC (±SD) | [p05, p95] Range | Bal Acc | Oracle Full-HC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |""")
    for _, r in td_tx[td_tx["transfer_type"] != "Within_Disorder_PD_Cross_Site"].iterrows():
        parts.append(f"| {r['transfer_type']} | {r['source_cohort']} -> {r['target_cohort']} | `{r['model']}` | {r['auc']:.3f} (±{r['split_hc_sd']:.3f}) | [{r['split_hc_p05']:.3f}, {r['split_hc_p95']:.3f}] | {100.0 * r['balanced_accuracy']:.1f}% | {r['oracle_full_hc_auc']:.3f} |")

    parts.append("""
### Table S17. Age-Adjusted ANCOVA Across All Cohorts with Individual Age Metadata
| Cohort Contrast | Marker | Group Coef (SE) | Group t | Group p | Age Coef | Age p | Model R² |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in anc.iterrows():
        parts.append(f"| {r['cohort']} | {r['marker']} | {r['group_coef']:+.4f} ({r['group_se']:.4f}) | {r['group_t']:+.3f} | {fmt_p(r['group_p'])} | {r['age_coef']:+.4f} | {fmt_p(r['age_p'])} | {r['r2']:.3f} |")

    parts.append("""
### Table S18. 10×5 Repeated Validation Leakage Quantification Across Repeated-Session Cohorts
| Cohort (Design) | Evaluation Regime | Model | Fold Overlap | 10×5 AUC (±SD) | Mean ΔAUC [95% CI] |
| :--- | :--- | :--- | :---: | :---: | :---: |""")
    for _, r in leak.iterrows():
        parts.append(f"| {r['cohort']} ({r['design_type']}) | {r['regime']} | `{r['model']}` | {r['subject_overlap_pct']:.1f}% | {r['mean_auc']:.3f} (±{r['sd_auc']:.3f}) | {r['mean_delta_vs_strict']:+.3f} [{r['delta_ci_lo']:+.3f}, {r['delta_ci_hi']:+.3f}] |")

    parts.append("""
### Table S19. Six-Pair Cross-Site PD Calibration & Gain Stress-Test Summary
| Experiment | Calibration | Feature Type | Gain k | LR AUC (BalAcc / Brier) | RF AUC | Mean / Max Abs(ΔFeat) | Mean / Max Abs(Δp) (LR) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |""")
    gsum = gc.groupby(["experiment", "calibration", "feature_type", "gain_k"], as_index=False).agg(lr_auc=("lr_auc", "mean"), lr_bacc=("lr_bacc", "mean"), lr_brier=("lr_brier", "mean"), rf_auc=("rf_auc", "mean"), mean_feat=("mean_feat_drift", "mean"), max_feat=("max_feat_drift", "max"), mean_p=("lr_mean_prob_drift", "mean"), max_p=("lr_max_prob_drift", "max"))
    for _, r in gsum.iterrows():
        parts.append(f"| {r['experiment']} | {r['calibration']} | `{r['feature_type']}` | {r['gain_k']} | {r['lr_auc']:.3f} ({r['lr_bacc']:.3f} / {r['lr_brier']:.3f}) | {r['rf_auc']:.3f} | {r['mean_feat']:.3f} / {r['max_feat']:.3f} | {r['mean_p']:.3f} / {r['max_p']:.3f} |")

    parts.append("""
### Table S20. True 10×5 Repeated Participant-Grouped Cross-Validation Summary (All `12` Contrasts × `13` Models)
| Cohort Contrast | Model | Seed-42 AUC | 10×5 Mean AUC (±SD) | Min–Max Range | Mean Δ vs. GCN-OMST |
| :--- | :--- | :---: | :---: | :---: | :---: |""")
    for _, r in rep10.iterrows():
        parts.append(f"| {r['cohort']} | `{r['model']}` | {r['seed42_auc']:.3f} | {r['mean_auc']:.3f} (±{r['std_auc']:.3f}) | [{r['min_auc']:.3f}, {r['max_auc']:.3f}] | {r['mean_delta_vs_omst']:+.3f} |")

    parts.append("""
### Table S21. PLI Manifold Suitability Audit (`675` Sessions), Spatial Covariance Riemannian Baselines, and Fold-Wise `AIRM` Fréchet-Mean Congruence Whitening Check
#### Panel A. Eigenvalue Spectrum of `19 × 19` Symmetrized PLI Matrices (`I_19 + PLI`)
| Cohort | Sessions (N) | Min λ_min | Median λ_min | Indefinite (λ_min <= 0) | % Indefinite | After +10^-4 I |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in pd.concat([pli_a, d["pli_sum"]], ignore_index=True).iterrows():
        parts.append(f"| {r['cohort']} | {int(r['n_sessions'])} | {r['min_eigenvalue']:+.4f} | {r['median_min_eigenvalue']:+.4f} | {int(r['n_indefinite_raw'])}/{int(r['n_sessions'])} | {r['pct_indefinite_raw']:.2f}% | {int(r['n_indefinite_after_1e4_shift'])} still <= 0 |")
    parts.append("")
    parts.append("#### Panel B. 10×5 Euclidean PLI (171-D), SPD-Conditioned PLI Tangent, and `19 × 19` Sample Spatial Covariance Tangent (`Riemannian_Spatial_Cov_190D`)")
    parts.append("| Cohort | Classifier | Euclidean PLI 171-D | Diag-Loaded PLI SPD | Eigen-Clipped PLI SPD | `Riemannian_Spatial_Cov_190D` (±SD) |")
    parts.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    rp_m = riem.pivot(index=["cohort", "classifier"], columns="representation", values="auc")
    rp_s = riem.pivot(index=["cohort", "classifier"], columns="representation", values="sd_auc")
    for (cname, clf), r in rp_m.iterrows():
        rs = rp_s.loc[(cname, clf)]
        parts.append(f"| {cname} | `{clf}` | {r['Euclidean_UpperTri_171D']:.3f} (±{rs['Euclidean_UpperTri_171D']:.3f}) | {r['Riemannian_Gershgorin_SPD_190D']:.3f} (±{rs['Riemannian_Gershgorin_SPD_190D']:.3f}) | {r['Riemannian_Higham_SPD_190D']:.3f} (±{rs['Riemannian_Higham_SPD_190D']:.3f}) | {r['Riemannian_Spatial_Cov_190D']:.3f} (±{rs['Riemannian_Spatial_Cov_190D']:.3f}) |")
    parts.append("")
    parts.append("#### Panel C. Fold-Wise `AIRM` Fréchet-Mean Congruence Whitening vs. Identity Log-Euclidean Mapping (`k = 8` Non-Nested Mean AUC)")
    parts.append("| Covariance Representation | LR Mean (`Δ vs Id_Z`) | Ridge Mean (`Δ vs Id_Z`) | RF Mean (`Δ vs Id_Z`) | Methodological Note |")
    parts.append("| :--- | :---: | :---: | :---: | :--- |")
    ag = v7_airm.groupby(["representation", "classifier"])[["auc_mean", "delta_vs_id_z"]].mean()
    rep_meta = [
        ("Identity_LogEuclidean_Z", "Fixed `I_19` tangent reference (`Log-Euclidean`, translation-invariant to linear centering)"),
        ("Foldwise_AIRM_Frechet_Z", "Strict fold-wise `AIRM` congruence whitening `P_tr^(-1/2) C P_tr^(-1/2)` + z-score"),
        ("Foldwise_AIRM_Frechet_Raw", "Strict fold-wise `AIRM` congruence whitening without feature z-scoring"),
        ("Leaky_Global_AIRM_Raw", "Intentionally leaky global `AIRM` whitening (exposes `+0.0043` RF optimism bias)"),
    ]
    for rkey, rnote in rep_meta:
        lr_m, lr_d = ag.loc[(rkey, "LR"), "auc_mean"], ag.loc[(rkey, "LR"), "delta_vs_id_z"]
        rd_m, rd_d = ag.loc[(rkey, "Ridge"), "auc_mean"], ag.loc[(rkey, "Ridge"), "delta_vs_id_z"]
        rf_m, rf_d = ag.loc[(rkey, "RF"), "auc_mean"], ag.loc[(rkey, "RF"), "delta_vs_id_z"]
        parts.append(f"| `{rkey}` | {lr_m:.4f} ({lr_d:+.4f}) | {rd_m:.4f} ({rd_d:+.4f}) | {rf_m:.4f} ({rf_d:+.4f}) | {rnote} |")

    parts.append("""
### Table S22. Random-Effects Meta-Analysis (`DL`, `HKSJ`, `K = 6` & `K = 5` Omitting Study 2) and Leave-One-Out Suite
| Method | Scope | Subgroup | k | Pooled d_RE | DL 95% CI | HKSJ 95% CI | DL p | HKSJ p | I² | Q_between (p) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in meta_df.iterrows():
        parts.append(f"| `{r['method']}` | {r['analysis_set']} | {r['subgroup']} | {int(r['k_studies'])} | {r['pooled_d']:+.3f} | [{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}] | [{r['hksj_ci_lower']:+.3f}, {r['hksj_ci_upper']:+.3f}] | {fmt_p(r['p_value'])} | {fmt_p(r['p_hksj'])} | {r['I2_percent']:.1f}% | {r['Q_between']:.2f} ({fmt_p(r['p_between'])}) |")

    parts.append("""
### Table S23. Per-Cohort `specparam v2.0.0rc7` Diagnostics Across Fit Windows (`[1.0, 40.0]` vs. `[2.0, 30.0] Hz`)
| Fit Window | Cohort Key | Sessions | Spectra | Failed | % R² < 0.80 | R² p05 / Median / Mean | MAE Med / Mean | Mean Peaks | Mean χ_OLS vs. χ_sp |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in fit_df.iterrows():
        parts.append(f"| {r['fit_window_hz']} Hz | `{r['cohort_key']}` | {int(r['n_sessions'])} | {int(r['n_spectra'])} | {int(r['n_failed'])} | {r['pct_r2_lt_080']:.2f}% | {r['r2_p05']:.3f} / {r['r2_median']:.3f} / {r['r2_mean']:.3f} | {r['mae_median']:.4f} / {r['mae_mean']:.4f} | {r['mean_n_peaks']:.2f} | {r['mean_chi_ols']:.3f} vs. {r['mean_chi_specparam']:.3f} |")

    parts.append("""
### Table S24. Posterior Individual Alpha Peak Frequency (`IAPF`) and 10×5 Heuristic Slowing Benchmarks
#### Panel A. Group Differences in Posterior `IAPF` (`7.0–13.5 Hz`)
| Cohort Contrast | Valid N (Case / Ref) | Mean IAPF Case (Hz) | Mean IAPF Ctrl (Hz) | Diff (Hz) | Cohen's d | p-Value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in iapf_df.iterrows():
        parts.append(f"| {r['cohort']} | {int(r['n_valid_case'])}/{int(r['n_valid_ctrl'])} | {r['mean_iapf_case_hz']:.3f} (±{r['sd_iapf_case_hz']:.2f}) | {r['mean_iapf_ctrl_hz']:.3f} (±{r['sd_iapf_ctrl_hz']:.2f}) | {r['diff_hz']:+.3f} | {r['cohens_d']:+.3f} | {fmt_p(r['p_value'])} |")
    parts.append("")
    parts.append("#### Panel B. 10×5 Heuristic Frontal (Theta − Alpha) Slowing vs. Canonical `Single_Theta`")
    parts.append("| Cohort Contrast | N | Heuristic Raw AUC | Oriented AUC | Balanced Acc | `Single_Theta` Oriented AUC |")
    parts.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for _, r in heur_df.iterrows():
        parts.append(f"| {r['cohort']} | {int(r['n_samples'])} | {r['auc_raw']:.3f} | {r['auc_oriented']:.3f} | {100.0 * r['balanced_accuracy']:.1f}% | {r['single_theta_auc_oriented']:.3f} |")

    parts.append("""
### Table S25. Epoch-Duration Balance, Joint Age + Duration ANCOVA, and UNM Pure Rest Sensitivity
#### Panel A. Clean 4 s Epoch Balance and Combined Chronological Age + Clean-Epoch ANCOVA (Frontal Periodic Theta)
| Cohort | Valid N | Age-Only t (p) | Age Coef p | Age + Epoch Group t (p) | Epoch Duration p | Case vs. Ctrl Epoch p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |""")
    ep_map = {
        "Study 3: Univ. Depression (N=118)": "Study 3: Univ. Depression (ds003478)",
        "Study 4a: UCSD PD-OFF (N=31)": "Study 4: UCSD PD ON/OFF (ds002778)",
        "Study 4b: UCSD PD-ON (N=31)": "Study 4: UCSD PD ON/OFF (ds002778)",
        "Study 5c: Dementia All (N=88)": "Study 5: AHEPA Dementia (ds004504)",
        "Study 6a: UNM PD-OFF (N=50)": "Study 6: UNM PD Pure EO (ds003490)",
        "Study 6b: UNM PD-ON (N=50)": "Study 6: UNM PD Pure EO (ds003490)",
        "Study 7: Iowa PD (N=149)": "Study 7: Iowa PD (ds004584)",
        "Study 2a: Outpatient MDD EC (N=56)": "Study 2a: Outpatient MDD EC (4244171)",
    }
    ep_idx = ep_df.set_index("cohort")
    for _, r in anc_dur.iterrows():
        ep_p = ep_idx.loc[ep_map[r["cohort"]], "case_vs_ctrl_p"]
        a_t = f"{r['age_only_grp_t']:+.3f} ({fmt_p(r['age_only_grp_p'])})" if pd.notnull(r["age_only_grp_t"]) else "— (No indiv. age)"
        a_p = fmt_p(r["age_p"]) if pd.notnull(r["age_p"]) else "—"
        parts.append(f"| {r['cohort']} | {int(r['n'])} | {a_t} | {a_p} | {r['age_plus_epoch_grp_t']:+.3f} ({fmt_p(r['age_plus_epoch_grp_p'])}) | {fmt_p(r['epoch_p'])} | {fmt_p(ep_p)} |")
    parts.append("")
    parts.append("#### Panel B. Study 6 (UNM `ds003490`) Contiguous Pure Eyes-Open (`N = 75`) vs. Pure Eyes-Closed (`N = 74`) Rest")
    parts.append("| Recording Block | Frontal θ d (p) | Posterior α d (p) | Global χ d (p) |")
    parts.append("| :--- | :---: | :---: | :---: |")
    for _, r in unm_sens.iterrows():
        parts.append(f"| {r['condition']} | {r['theta_d']:+.3f} ({fmt_p(r['theta_p'])}) | {r['alpha_d']:+.3f} ({fmt_p(r['alpha_p'])}) | {r['chi_d']:+.3f} ({fmt_p(r['chi_p'])}) |")

    parts.append("""
### Table S26. Figshare `4244171` (Study 2) Duplicate Hashing, Batch Sensitivity, Unfiltered EDF Profiles, and Full `181`-File Accounting
#### Panel A. Exact Byte-Identical SHA-256 Duplicate File Groups Detected in Public Release
| SHA-256 Prefix | Identical Files (n) | Filenames in Figshare Release |
| :---: | :---: | :--- |""")
    for _, r in mum_dups.iterrows():
        parts.append(f"| `{r['sha256_prefix']}` | {int(r['n_files'])} | `{r['filenames']}` |")
    parts.append("")
    parts.append("#### Panel B. Deduplicated and Batch-Stratified Biophysics (w/ 95% CIs) & 10×5 Classification Sensitivity (`N = 29 Unique MDD, 27 Unique HC EC`)")
    parts.append("| Subset / Batch Sensitivity | Total N (Case/Ctrl) | Frontal θ d [95% CI] (p) | Global χ d [95% CI] (p) | 10×5 Tuned LR (±SD) | 10×5 RF (±SD) |")
    parts.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for _, r in mum_for.iterrows():
        lrs = f"{r['lr_10x5_mean']:.3f} (±{r['lr_10x5_sd']:.3f})" if pd.notnull(r["lr_10x5_mean"]) else "—"
        rfs = f"{r['rf_10x5_mean']:.3f} (±{r['rf_10x5_sd']:.3f})" if pd.notnull(r["rf_10x5_mean"]) else "—"
        parts.append(f"| {r['subset']} | {int(r['n_total'])} ({int(r['n_case'])}/{int(r['n_ctrl'])}) | {r['theta_d']:+.3f} [{r['theta_ci_lo']:+.3f}, {r['theta_ci_hi']:+.3f}] ({fmt_p(r['theta_p'])}) | {r['chi_d']:+.3f} [{r['chi_ci_lo']:+.3f}, {r['chi_ci_hi']:+.3f}] ({fmt_p(r['chi_p'])}) | {lrs} | {rfs} |")
    parts.append("")
    parts.append("#### Panel C. Raw Unfiltered EDF `0.5–80 Hz` Welch Log10 PSD Profiles by Acquisition Batch (Identical Nominal `HP:0.5 Hz LP:80 Hz` Headers)")
    parts.append("| Acquisition Batch (Unique EC) | n | log10(1 Hz) | log10(2 Hz) | log10(6 Hz) | log10(30 Hz) | log10(40 Hz) | log10(50 Hz) | 1−2 Hz Drop | 40−30 Hz Roll-Off | 50 Hz Spike |")
    parts.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    s2u_g = s2_unf[~s2_unf["is_duplicate_sha256"]].groupby("batch", sort=False).agg(n=("subject_id", "count"), logp_1hz=("logp_1hz", "mean"), logp_2hz=("logp_2hz", "mean"), logp_6hz=("logp_6hz", "mean"), logp_30hz=("logp_30hz", "mean"), logp_40hz=("logp_40hz", "mean"), logp_50hz=("logp_50hz", "mean"), d21=("drop_2_to_1hz", "mean"), d34=("drop_30_to_40hz", "mean"), l50=("line_50hz_excess", "mean"))
    for bname, r in s2u_g.iterrows():
        parts.append(f"| `{bname}` | {int(r['n'])} | {r['logp_1hz']:.3f} | {r['logp_2hz']:.3f} | {r['logp_6hz']:.3f} | {r['logp_30hz']:.3f} | {r['logp_40hz']:.3f} | {r['logp_50hz']:.3f} | {r['d21']:+.3f} | {r['d34']:+.3f} | {r['l50']:+.3f} |")
    parts.append("")
    parts.append("#### Panel D. Complete `181`-File Figshare `4244171` Accounting (`EC`, `EO`, `TASK`)")
    parts.append("| Recording Block | Disposition Across On-Disk EDF Archive | File Count (n) |")
    parts.append("| :---: | :--- | :---: |")
    for (st_blk, disp), cnt in s2_181.groupby(["state_block", "disposition"]).size().items():
        parts.append(f"| `{st_blk}` | {disp} | {int(cnt)} |")
    parts.append(f"| **Total** | **All public EDF files in Figshare `4244171` release** | **{len(s2_181)}** |")

    parts.append("""
### Table S27. Complete 25-Configuration 2D Grid (105 Cohort Evaluations) and Full-Archive `13,927`-Spectrum Stable `fooof v1.1.1` Verification
#### Panel A. Representative 2D Fit-Window Grid Across Cohorts and Unfiltered Study 2 EDFs (`0/105` Failed Sets)
| Cohort & PSD Source | f_min (Hz) | f_max (Hz) | Spectra | Failed | Median R² | Median MAE | Frontal θ d (p) | Global χ d (p) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |""")
    sub_g = grid2d[grid2d["f_min_hz"].isin([1.0, 2.0, 3.0]) & grid2d["f_max_hz"].isin([25.0, 30.0, 35.0, 40.0, 45.0])]
    for _, r in sub_g.iterrows():
        parts.append(f"| {r['cohort']} | {r['f_min_hz']:.1f} | {r['f_max_hz']:.1f} | {int(r['n_spectra'])} | {int(r['n_failed'])} | {r['r2_median']:.3f} | {r['mae_median']:.4f} | {r['theta_d']:+.3f} ({fmt_p(r['theta_p'])}) | {r['chi_d']:+.3f} ({fmt_p(r['chi_p'])}) |")
    parts.append("")
    parts.append("#### Panel B. Exhaustive Full-Archive Equivalence Between Stable `fooof v1.1.1` and `specparam v2.0.0rc7` (`13,927` Unique Spectra; `[2.0, 30.0] Hz`)")
    parts.append("| Scope | Sessions | Spectra | `fooof` Null Fits | Pearson r (χ) | Mean Abs(Δχ) | Pearson r (θ Resid) | Mean Abs(Δθ) (B) |")
    parts.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, r in fooof_full.iterrows():
        parts.append(f"| `{r['scope']}` | {int(r['n_sessions'])} | {int(r['n_spectra'])} | {int(r['fooof_n_null_total'])} | {r['pearson_r_chi']:.4f} | {r['mean_abs_diff_chi']:.4f} | {r['pearson_r_theta']:.4f} | {r['mean_abs_diff_theta']:.5f} |")

    parts.append("""
### Table S28. Study 7 (`Iowa PD ds004584`, `N = 149`) Online-Reference `Pz` CAR vs. `18`-Channel Physical CAR vs. Spherical-Spline `Pz` Sensitivity
| Reference / CAR Regime | Channels | Frontal θ d (p) | Posterior IAPF d (p) | `GCN-OMST` 10×5 AUC | `Tuned LR` AUC | `Linear SVM` AUC | `Random Forest` AUC | Patient MoCA SVR r | Patient UPDRS GBDT r |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in s7_pz.iterrows():
        parts.append(
            f"| `{r['regime']}` | {int(r['n_channels'])} | {r['frontal_theta_d']:+.3f} ({fmt_p(r['frontal_theta_p'])}) | "
            f"{r['iapf_d']:+.3f} ({fmt_p(r['iapf_p'])}) | {r['gcn_omst_auc']:.3f} (±{r['gcn_omst_sd']:.3f}) | "
            f"{r['tuned_lr_auc']:.3f} | {r['linear_svm_auc']:.3f} | {r['rf_auc']:.3f} | "
            f"{r['moca_svr_r']:+.3f} (R²={r['moca_svr_r2']:+.3f}) | {r['updrs_gbdt_r']:+.3f} (R²={r['updrs_gbdt_r2']:+.3f}) |"
        )
    return "\n".join(parts)


# ==============================================================================
# Supporting Documents & Machine-Readable Workbook
# ==============================================================================
def build_title_page() -> str:
    """Strict Springer Title Page without duplicated Abstract, Keywords, or Declarations."""
    return f"""# Title Page

**Full Title:** {EXACT_TITLE}

**Short Running Title:** {SHORT_TITLE}

**Article Type:** {ARTICLE_TYPE}

**Target Journal:** *{JOURNAL_NAME}* ({PUBLISHER_NAME})

**Author:** {AUTHOR_NAME}^1,2*^

**Affiliations:**  
^1^ {AFFIL_1}  
^2^ {AFFIL_2}

**Corresponding Author:**  
{AUTHOR_NAME}  
{CORRESPONDENCE}

---

*Note for Editorial Office:* Abstract, Keywords, Information Sharing Statement, and Statements and Declarations appear in the Main Manuscript (`04_Main_Manuscript_Neuroinformatics.docx`) without duplication.
"""


def build_cover_letter(d: dict) -> str:
    acc = d["master"]["accounting"]
    rep = d["rep10"]
    nn8 = [c for c in rep["cohort"].unique() if not any(k in c for k in ("2b", "5c", "6b", "6c"))]
    piv8 = rep[rep["cohort"].isin(nn8)].pivot(index="cohort", columns="model", values="mean_auc")
    vt8 = d["v7_tost"][(d["v7_tost"]["record_type"] == "Random_Effects_Pool") & (d["v7_tost"]["scope"] == "k=8 Non-Nested")].set_index("comparator_model")
    vdeep = d["v7_deep"].groupby("model")["mean_auc"].mean()
    vairm = d["v7_airm"][(d["v7_airm"]["representation"] == "Foldwise_AIRM_Frechet_Z") & (d["v7_airm"]["classifier"] == "RF")]["auc_mean"].mean()
    pli_s = d["pli_sum"].iloc[0]
    ndg = d["meta"][(d["meta"]["method"] == "specparam_2to30_resid") & (d["meta"]["analysis_set"] == "Primary_K6_SubjMean_PD_OFF") & (d["meta"]["subgroup"] == "PD_Dementia_Cohorts")].iloc[0]
    loo_dem = d["meta"][(d["meta"]["method"] == "specparam_2to30_resid") & (d["meta"]["analysis_set"] == "LOO_Omit_Dementia") & (d["meta"]["subgroup"] == "PD_Dementia_Cohorts")].iloc[0]
    dem_pt = d["reg"][(d["reg"]["cohort_key"] == "Dementia_MMSE_Patients_Only_N59") & (d["reg"]["model"] == "Ridge_Regression")].iloc[0]
    iowa_pt = d["reg"][(d["reg"]["cohort_key"] == "Iowa_PD_MoCA_Patients_Only_N100") & (d["reg"]["model"] == "SVR_RBF")].iloc[0]
    fit2 = d["fit"][d["fit"]["fit_window_hz"] == "2.0-30.0"].set_index("cohort_key")
    fit1 = d["fit"][d["fit"]["fit_window_hz"] == "1.0-40.0"].set_index("cohort_key")
    ff = d["fooof_full"][d["fooof_full"]["scope"] == "All_733Sessions_13927Spectra"].iloc[0]

    return f"""# Cover Letter

**Date:** {SUBMISSION_DATE}

**To:**  
{EDITOR_NAME}  
{EDITOR_TITLE}  
{EDITOR_AFFIL}

**Subject:** {ARTICLE_TYPE} Submission to *{JOURNAL_NAME}*

Dear Prof. Van Horn,

Please consider the {ARTICLE_TYPE} entitled **"{EXACT_TITLE}"** for publication in *{JOURNAL_NAME}*.

This manuscript introduces an open Python benchmarking and dataset-auditing suite spanning seven public 19-channel resting-state EEG archives (`{acc['total_unique_participants']}` unique primary participants, `{acc['total_analyzed_sessions']}` primary sessions, `{acc['total_with_mumtaz_eo']}` total recordings, `{acc['total_spectra']:,}` channel spectra) organized as a unified three-stage verification workflow:

1. **Stage 1 — Spectral Fit-Window Stability (`25` Grid Configurations), Reference Sensitivity, and EDF Batch Forensics:** Interior `[2.0, 30.0] Hz` `specparam` (cross-checked against `fooof v1.1.1`, `r = {ff['pearson_r_chi']:.4f}`) prevented upstream roll-off collapse in Study 2 (`R² {fit1.loc['mumtaz_ec', 'r2_median']:.3f} -> {fit2.loc['mumtaz_ec', 'r2_median']:.3f}`) and eliminated masked OLS theta bias in Study 3. Raw `181`-EDF hashing and unfiltered `0.5–80 Hz` spectra revealed byte-identical duplicates, `Jan-2013` `6 Hz` and `50 Hz` control noise (`d_theta = +3.347` vs. `2011` HC), and header-invisible `Apr–Jul 2013` roll-off confounding diagnosis with acquisition date (`d = -1.188` vs. `-0.082, p = 0.752` excluding `Jan-2013` HCs), whereas neurodegenerative PD and dementia frontal periodic theta elevation replicated (`k = 4 d_RE = {ndg['pooled_d']:+.3f}, p_HKSJ = {ndg['p_hksj']:.4f}`; state-homogeneous eyes-open PD `k = 3 d_RE = {loo_dem['pooled_d']:+.3f}`), invariant under `18`- vs. `19`-channel `Pz`-referenced CAR in Study 7.
2. **Stage 2 — `Nadeau & Bengio` Random-Effects `TOST`, `ShallowSpecCNN`, Tuned `GCN-OMST`, and Fold-Wise `AIRM` Audits:** Across `k = 8` small clinical cohorts (`N = 31–149`), fixed `GCN-OMST` (`{piv8['GCN_OMST'].mean():.3f}`) met `±0.02` equivalence (`Holm p <= {vt8.loc['GCN_Weighted_PLI', 'tost02_holm_gnn4']:.4f}`) against `GCN-Weighted-PLI` (`{piv8['GCN_Weighted_PLI'].mean():.3f}`), session edge-count-matched Erdős–Rényi `GCN-Random*` (`{piv8['GCN_Random_EdgeMatched'].mean():.3f}`, `Holm p = {vt8.loc['GCN_Random_EdgeMatched', 'tost02_holm_gnn4']:.4f}`), and dense `GCN-Full` (`{piv8['GCN_Full'].mean():.3f}`, `Holm p = {vt8.loc['GCN_Full', 'tost02_holm_gnn4']:.4f}`), while isolated-node `GCN-Identity` (`{piv8['GCN_Identity'].mean():.3f}`) was inconclusive at `±0.02` (`Holm p = {vt8.loc['GCN_Identity', 'tost02_holm_gnn4']:.4f}`) and met `±0.04` (`Holm p = {vt8.loc['GCN_Identity', 'tost04_holm_gnn4']:.4f}`). Inner-tuned `GCN-OMST-Tuned` (`{vdeep['GCN_OMST_Tuned']:.3f}`) and compact `ShallowSpecCNN` (`{vdeep['ShallowSpecCNN']:.3f}`) never outperformed `Linear SVM` (`{piv8['Linear_SVM'].mean():.3f}`) or fold-wise `AIRM` Fréchet-whitened `Spatial Cov RF` (`{vairm:.3f}`), while `{int(pli_s['n_indefinite_raw'])}/{int(pli_s['n_sessions'])} ({pli_s['pct_indefinite_raw']:.2f}%)` symmetrized `I_19 + PLI` matrices were indefinite for direct manifold embedding.
3. **Stage 3 — External Transfer Boundaries and Max-Stat Clinical Permutations:** Under disjoint `Split-Target-HC` transfer and 4-model max-statistic 1,000-permutation tests, removing ceiling controls eliminated dementia MMSE prediction (`N = 59: Ridge r = {dem_pt['pearson_r_mean']:+.3f}, max-stat 2-S perm p = {dem_pt['maxstat4_perm_p_2s']:.4f}, OOF R² = -0.110 to -0.047`), and all 12 patient-only severity models stayed below `<= 3.5%` OOF `R²` (`10/12` negative; Iowa PD MoCA `SVR r = {iowa_pt['pearson_r_mean']:+.3f}`).

All figures (`Fig. 1–4` in the Main Manuscript and `Fig. S1–S6` in `Online Resource 1`) are embedded directly within their respective `.docx` and `.pdf` submission files alongside self-contained `Table S1–S28` results.

**Dataset Overlap Disclosure:** For full transparency, subsets of two of the seven public archives examined here are analyzed in distinct single-domain companion manuscripts by the author (one submitted to *Journal of Alzheimer's Disease* examining clinical decision-curve utility and electrode-reference sensitivity on Figshare `4244171` vs. Alzheimer's/depression baselines, and one studying acute sleep-deprivation neurophysiology on OpenNeuro `ds004902`). The present submission is wholly distinct in scope, methodology, and open benchmark contribution and neither overlaps verbatim nor duplicates any figures, tables, or central claims elsewhere.

All code (`MIT` license) and derived `CC BY 4.0` artifacts are released publicly at `{GITHUB_URL}` (`{RELEASE_TAG}`), alongside permanent OpenNeuro and Figshare DOIs cited in the Information Sharing Statement.

Sincerely yours,

**{AUTHOR_NAME}**  
{AFFIL_1}  
{AFFIL_2}  
{CORRESPONDENCE}
"""


def build_highlights(d: dict) -> str:
    rep = d["rep10"]
    nn8 = [c for c in rep["cohort"].unique() if not any(k in c for k in ("2b", "5c", "6b", "6c"))]
    piv = rep[rep["cohort"].isin(nn8)].pivot(index="cohort", columns="model", values="mean_auc")
    vt8 = d["v7_tost"][(d["v7_tost"]["record_type"] == "Random_Effects_Pool") & (d["v7_tost"]["scope"] == "k=8 Non-Nested")].set_index("comparator_model")
    vdeep = d["v7_deep"].groupby("model")["mean_auc"].mean()
    vairm = d["v7_airm"][(d["v7_airm"]["representation"] == "Foldwise_AIRM_Frechet_Z") & (d["v7_airm"]["classifier"] == "RF")]["auc_mean"].mean()
    pli_s = d["pli_sum"].iloc[0]
    ff = d["fooof_full"][d["fooof_full"]["scope"] == "All_733Sessions_13927Spectra"].iloc[0]
    return f"""# Research Highlights

*Journal Target:* *{JOURNAL_NAME}* ({PUBLISHER_NAME})  
*Article Type:* {ARTICLE_TYPE}  
*Manuscript Title:* {EXACT_TITLE}

1. Open 7-cohort three-stage Python EEG benchmark (`564` unique participants, `13,927` spectra verified vs. `fooof v1.1.1` `r = {ff['pearson_r_chi']:.4f}`).
2. Unfiltered `181`-EDF forensics show `Jan-2013` `6 Hz`/`50 Hz` artifacts and `Apr–Jul 2013` roll-off confound Study 2 diagnosis (`d = -0.082` excl. `Jan-2013` HCs).
3. Under Nadeau–Bengio random-effects `TOST` (`k = 8`), `GCN-OMST` (`{piv['GCN_OMST'].mean():.3f}`) matches random (`{piv['GCN_Random_EdgeMatched'].mean():.3f}`), dense, and PLI graphs within `±0.02` AUC (`Holm p <= {vt8.loc['GCN_Weighted_PLI', 'tost02_holm_gnn4']:.4f}`).
4. Tuned `GCN-OMST` (`{vdeep['GCN_OMST_Tuned']:.3f}`) and `ShallowSpecCNN` (`{vdeep['ShallowSpecCNN']:.3f}`) do not surpass `Linear SVM` (`{piv['Linear_SVM'].mean():.3f}`) or fold-wise `AIRM` spatial covariance `RF` (`{vairm:.3f}`).
5. `{pli_s['pct_indefinite_raw']:.2f}%` of `I_19 + PLI` matrices are indefinite; patient-only clinical regressions hit `<= 3.5%` OOF `R²` (MMSE `-0.110 to -0.047`).
"""


def build_declarations() -> str:
    return f"""# Statements, Ethical Approval, Data/Code Availability, and CRediT Declarations

**Manuscript Title:** *{EXACT_TITLE}*  
**Article Type:** {ARTICLE_TYPE} | **Target Journal:** *{JOURNAL_NAME}* ({PUBLISHER_NAME})  
**Author:** {AUTHOR_NAME}^1,2*^ (`{CORRESPONDENCE}`)

1. **Ethics Approval & Consent:** Secondary analysis exclusively on de-identified public archives (Southwest Univ. `H20039`, `USM/JEPeM/1401015`, UNM IRB `ds003478`/`ds003490`, UCSD HRPP `ds002778`, AHEPA `142/12-04-2023`, Univ. Iowa IRB `201707828`). Consent for publication: Not applicable.
2. **Competing Interests & Funding:** The author declares no competing interests and received no specific grant funding.
3. **Data & Code Licensing Separation:** Executable Python source code (`{GITHUB_URL}`, release tag `{RELEASE_TAG}`) is licensed under **MIT**; derived tabular benchmark CSVs (`release_artifacts/`) are licensed under **CC BY 4.0** (`release_artifacts/DATA_LICENSES.md`) with full tables/figures embedded directly in `Online Resource 1`; upstream EEG archives remain permanently cited under their repository DOIs (`ds004902.v1.1.0`, `figshare.4244171.v5`, `ds003478.v1.1.0`, `ds002778.v1.0.5`, `ds004504.v1.0.6`, `ds003490.v1.0.0`, `ds004584.v1.0.0`).
4. **CRediT:** **Chushu Tan:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data Curation, Writing – Original Draft, Writing – Review & Editing, Visualization, Project administration.
5. **AI-Assisted Workflow Disclosure (Full details also documented in Methods Section 2.7):** AI-assisted engineering and editorial tools (Antigravity / Gemini LLM coding assistants) were used to draft pipeline scripts, structure summary tables, and refine English phrasing. The author manually verified all code, preprocessing assumptions, and numbers directly against raw files and pipeline CSVs.
"""


def build_suggested_reviewers() -> str:
    return f"""# Suggested Reviewers

**Manuscript Title:** *{EXACT_TITLE}*  
**Target Journal:** *{JOURNAL_NAME}*

| Reviewer | Affiliation | Expertise Relevant to Manuscript | Email |
| :--- | :--- | :--- | :--- |
| Prof. Bradley Voytek, Ph.D. | Department of Cognitive Science, University of California San Diego, USA | Creator of `specparam`/FOOOF periodic–aperiodic parameterization | `bvoytek@ucsd.edu` |
| Prof. Stavros I. Dimitriadis, Ph.D. | School of Psychology / Neuroinformatics, Cardiff University, UK | Developer of OMST topological graph filtering in connectomics | `DimitriadisS@cardiff.ac.uk` |
| Prof. Alexandre Gramfort, Ph.D. | Meta AI / Inria Saclay, France | Riemannian tangent-space EEG decoding and MNE-Python tooling | `alexandre.gramfort@inria.fr` |
| Prof. Arnaud Delorme, Ph.D. | Swartz Center for Computational Neuroscience, UC San Diego / CNRS | Large-scale multi-cohort EEG BIDS preprocessing and benchmarking | `adelorme@ucsd.edu` |

*Conflict Check:* The author has no past or present collaborations, co-authorships, funding ties, or institutional conflicts with any suggested reviewer.
"""


def build_submission_checklist(main_md: str) -> str:
    abs_wc = len(main_md.split("## Abstract\n\n")[1].split("\n\n**Keywords:**")[0].split())
    return f"""# Springer *Neuroinformatics* Submission Checklist & Guide (v7 Reviewer-Ready)

**Journal:** *{JOURNAL_NAME}* (Springer Nature, Editorial Manager)  
**Target Editor:** {EDITOR_NAME} | **Article Type:** {ARTICLE_TYPE} | **Date:** {SUBMISSION_DATE}

| File in `submission_Neuroinformatics/` | Editorial Manager Slot | Notes |
| :--- | :--- | :--- |
| `01_Cover_Letter_Neuroinformatics.docx` | Cover Letter | Three-stage Original Article summary & dataset overlap disclosure |
| `02_Research_Highlights_Neuroinformatics.docx` | Author Highlights | 5 concise verified takeaways |
| `03_Title_Page_Neuroinformatics.docx` | Title Page | Title, Running Title, `Original Article`, Author, Affiliations, ORCID/Email only |
| `04_Main_Manuscript_Neuroinformatics.docx` (+ `.pdf`) | Manuscript | `Abstract` ({abs_wc} words, zero undefined abbreviations), **4 embedded main figures (`Fig. 1–4`)**, Tables 1–5, AI in Methods Sec 2.7, 65 APA refs |
| `05_Online_Resource_1_Neuroinformatics.pdf` (+ `.docx`) | Online Resource 1 (ESM) | Supplementary Methods S1.1–S1.4, **6 embedded figures (`Fig. S1–S6`)**, `Table S1–S28` (self-contained) |
| `06_Declarations_CRediT_COI_Ethics.docx` | Declarations Attachment | Ethics, COI, Licensing (`MIT` Code / `CC BY 4.0` Derivatives), permanent 7 DOIs + GitHub `{RELEASE_TAG}`, AI disclosure |
| `07_Suggested_Reviewers.docx` | Suggested Reviewers | Voytek, Dimitriadis, Gramfort, Delorme |

*Note:* All figures are embedded directly inside `04_Main_Manuscript_Neuroinformatics.(docx/pdf)` and `05_Online_Resource_1_Neuroinformatics.(docx/pdf)`, with separate figure directories and auxiliary spreadsheet attachments consolidated prior to submission.
"""


def run_strict_audits(main_md: str, supp_md: str, extra_texts: list[str], doc_paths: list[Path]):
    print("\n🔍 Running strict Neuroinformatics v7 verification audits...")
    for p in doc_paths:
        with zipfile.ZipFile(p, "r") as z:
            xc = 0
            for name in z.namelist():
                if name.endswith((".xml", ".rels")):
                    ET.fromstring(z.read(name))
                    xc += 1
        d_obj = docx.Document(p)
        assert d_obj.core_properties.author == "" and d_obj.core_properties.last_modified_by == ""
        print(f"   ✓ {p.name}: {xc} XML parts valid, blank metadata ({len(d_obj.paragraphs)} paras, {len(d_obj.tables)} tables)")

    for docx_target, expected_imgs in [
        (MAIN_DOCX, 4),
        (SUBMISSION_DIR / "04_Main_Manuscript_Neuroinformatics.docx", 4),
        (SUPP_DOCX, 6),
        (SUBMISSION_DIR / "05_Online_Resource_1_Neuroinformatics.docx", 6),
    ]:
        with zipfile.ZipFile(docx_target, "r") as z:
            media_files = [n for n in z.namelist() if n.startswith("word/media/")]
            assert len(media_files) == expected_imgs, f"{docx_target.name} has {len(media_files)} embedded images != {expected_imgs}!"
    print("   ✓ Embedded figures verified: 4 in Main Manuscript DOCX and 6 in Online Resource 1 DOCX!")
    assert not (SUBMISSION_DIR / "Figures").exists(), "Separate submission_Neuroinformatics/Figures directory must not exist!"
    assert not (SUBMISSION_DIR / "05b_Online_Resource_2_Benchmark_Tables.xlsx").exists(), "Stale Online Resource 2 .xlsx still present!"

    with zipfile.ZipFile(MAIN_DOCX, "r") as z:
        assert "lnNumType" in z.read("word/document.xml").decode("utf-8")
        doc_xml = z.read("word/document.xml").decode("utf-8")
        assert 'w:val="subscript"' in doc_xml, "No real subscripts found in document.xml!"
        assert 'w:val="superscript"' in doc_xml, "No real superscripts found in document.xml!"
        ftr_xml = z.read("word/footer1.xml").decode("utf-8")
        assert ftr_xml.count("fldCharType=\"begin\"") == 1 and "Footer" not in ftr_xml, f"Unexpected footer XML: {ftr_xml}"
        root = ET.fromstring(z.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        for idx, tbl in enumerate(root.findall(".//w:tbl", ns)):
            gcols = [int(c.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}w")) for c in tbl.findall("./w:tblGrid/w:gridCol", ns)]
            assert gcols[0] >= 2000, f"Table {idx+1} col 0 too narrow ({gcols[0]} dxa)!"
            assert sum(gcols) == 9020, f"Table {idx+1} total width {sum(gcols)} dxa != 9020 dxa!"
        print(f"   ✓ OOXML footer (clean single PAGE field), table widths (9020 dxa), and real sub/sup ({doc_xml.count('w:val=\"subscript\"')}/{doc_xml.count('w:val=\"superscript\"')}) verified!")

    # Abstract abbreviation & word-count audit
    abs_text = main_md.split("## Abstract\n\n")[1].split("\n\n**Keywords:**")[0]
    abs_wc = len(abs_text.split())
    assert 150 <= abs_wc <= 250
    allowed_abs_tokens = {"EEG", "EDF", "GCN", "GCNs", "OMST", "OLS", "ROC", "AUC", "TOST", "OOF", "PLI", "SPD", "RF", "SVM", "II", "BDI", "PD", "AIRM"}
    upper_toks = set(re.findall(r"\b[A-Z]{2,}\b", abs_text))
    undefined = []
    for tok in sorted(upper_toks):
        if tok in {"EEG", "EDF", "TOST", "ROC", "AUC", "GCN", "OMST", "GCNs", "PLI", "SPD", "PD", "AIRM"}:
            assert (f"({tok})" in abs_text or f"`{tok}" in abs_text or (tok == "OMST" and "GCN-OMST" in abs_text) or (tok == "PLI" and "GCN-Weighted-PLI" in abs_text)), f"Unintroduced {tok}"
        elif tok not in allowed_abs_tokens:
            undefined.append(tok)
    assert not undefined, f"Undefined uppercase acronyms in Abstract: {undefined}"
    print(f"   ✓ Abstract audit passed ({abs_wc} words; zero undefined acronyms)")

    # Springer Fig. N caption audit
    for i in range(1, 5):
        cap_m = re.search(rf"^\*\*Fig\. {i}\*\*\s+([^\n]+)", main_md, flags=re.M)
        assert cap_m, f"Missing bold '**Fig. {i}** ' caption!"
        cap_line = cap_m.group(1).strip()
        assert not cap_line.endswith("."), f"Fig. {i} caption ends with a period!"
        assert "(a)" in cap_line and "(b)" in cap_line, f"Fig. {i} missing lowercase (a)/(b) panel markers!"

    # Author-Year references bijection
    body_text = main_md.split("## References")[0]
    refs_section = main_md.split("## References")[1]
    assert len([l for l in refs_section.strip().splitlines() if l.strip()]) == len(REFS_ALPHA)
    for key, _ in REFS_ALPHA:
        sn, yr = [x.strip() for x in key.rsplit(",", 1)]
        p1, p2 = re.escape(key), rf"{re.escape(sn)}\s*\({re.escape(yr)}\)"
        p3 = rf"{re.escape(sn.replace('&', 'and'))}\s*\({re.escape(yr)}\)"
        assert re.search(p1, body_text) or re.search(p2, body_text) or re.search(p3, body_text), f"Uncited {key}"
    print(f"   ✓ Author-Year References: {len(REFS_ALPHA)}/{len(REFS_ALPHA)} verified!")

    # Markdown table pipe-count integrity audit
    for m_lbl, m_txt in [("Main Manuscript", main_md), ("Online Resource 1", supp_md)]:
        m_lines = m_txt.splitlines()
        idx_l = 0
        while idx_l < len(m_lines):
            if m_lines[idx_l].startswith("|") and idx_l + 1 < len(m_lines) and ":---" in m_lines[idx_l + 1]:
                exp_c = len(m_lines[idx_l + 1].strip("|").split("|"))
                hdr_c = len(m_lines[idx_l].strip("|").split("|"))
                assert hdr_c == exp_c, f"{m_lbl} table header col mismatch line {idx_l+1}: {hdr_c} != {exp_c}"
                j_l = idx_l + 2
                while j_l < len(m_lines) and m_lines[j_l].startswith("|"):
                    row_c = len(m_lines[j_l].strip("|").split("|"))
                    assert row_c == exp_c, f"{m_lbl} table row mismatch line {j_l+1}: {row_c} != {exp_c}"
                    j_l += 1
                idx_l = j_l
            else:
                idx_l += 1
    print("   ✓ Markdown table pipe-column alignment verified across all main & supplementary tables!")

    # Guard against stale scientific claims, placeholders, & phrasing
    combined = "\n".join([main_md, supp_md] + extra_texts)
    assert not re.findall(r"\b(we|our|ours|ourselves|us)\b", main_md + "\n" + supp_md, flags=re.IGNORECASE)
    for bad in [
        "theta slowing", "hardware prefiltering", "pre-filtering", "prefiltering",
        "Original Research / Resource", "12246", "102249", "18008428969", "45 CFR 46",
        "[Zenodo", "Online Resource 2", "degree-preserving random-rewired",
    ]:
        assert bad.lower() not in combined.lower(), f"Found guarded/placeholder term '{bad}'!"
    print("   ✓ Terminology, zero-placeholder, embedded-figure, & single-author audits passed cleanly!")


def main():
    print("=" * 80)
    print("Building Paper 2 (Springer Neuroinformatics v7) Consolidated Package")
    print("=" * 80)
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    if (SUBMISSION_DIR / "Figures").exists():
        shutil.rmtree(SUBMISSION_DIR / "Figures")
    for stale_fn in [
        "05_Supplementary_Material_Neuroinformatics.docx",
        "05_Supplementary_Material_Neuroinformatics.md",
        "05b_Online_Resource_2_Benchmark_Tables.xlsx",
    ]:
        sp_old = SUBMISSION_DIR / stale_fn
        if sp_old.exists():
            sp_old.unlink()

    d = load_all_benchmark_data()
    main_md = build_main_manuscript(d)
    supp_md = build_supplementary_material(d)

    MAIN_MD.write_text(main_md, encoding="utf-8")
    SUPP_MD.write_text(supp_md, encoding="utf-8")
    md_to_docx(MAIN_MD, MAIN_DOCX, is_main=True)
    md_to_docx(SUPP_MD, SUPP_DOCX, is_main=False)
    md_to_pdf_via_chrome(MAIN_MD, MAIN_PDF, is_main=True)
    md_to_pdf_via_chrome(SUPP_MD, SUPP_PDF, is_main=False)
    print(f"✓ Generated {MAIN_DOCX.name}, {MAIN_PDF.name}, {SUPP_DOCX.name}, & {SUPP_PDF.name}")

    files_to_build = [
        ("01_Cover_Letter_Neuroinformatics", build_cover_letter(d), False),
        ("02_Research_Highlights_Neuroinformatics", build_highlights(d), False),
        ("03_Title_Page_Neuroinformatics", build_title_page(), False),
        ("04_Main_Manuscript_Neuroinformatics", main_md, True),
        ("05_Online_Resource_1_Neuroinformatics", supp_md, False),
        ("06_Declarations_CRediT_COI_Ethics", build_declarations(), False),
        ("07_Suggested_Reviewers", build_suggested_reviewers(), False),
    ]
    generated_docx = [MAIN_DOCX, SUPP_DOCX]
    extra_texts = []
    for stem, content, is_main_flag in files_to_build:
        mdp = SUBMISSION_DIR / f"{stem}.md"
        docxp = SUBMISSION_DIR / f"{stem}.docx"
        mdp.write_text(content, encoding="utf-8")
        md_to_docx(mdp, docxp, is_main=is_main_flag)
        generated_docx.append(docxp)
        if not is_main_flag:
            extra_texts.append(content)
        print(f"✓ Generated {stem} (.md, .docx)")

    shutil.copy2(MAIN_PDF, SUBMISSION_DIR / "04_Main_Manuscript_Neuroinformatics.pdf")
    shutil.copy2(SUPP_PDF, SUBMISSION_DIR / "05_Online_Resource_1_Neuroinformatics.pdf")

    for lock_f in list(SUBMISSION_DIR.glob("~$*")):
        try:
            lock_f.unlink()
        except Exception:
            pass

    chk_txt = build_submission_checklist(main_md)
    chk_p = SUBMISSION_DIR / "08_Submission_Checklist_Neuroinformatics.md"
    chk_p.write_text(chk_txt, encoding="utf-8")
    extra_texts.append(chk_txt)
    run_strict_audits(main_md, supp_md, extra_texts, generated_docx)

    if ARTIFACT_DIR.exists():
        mirror_map = [
            (MAIN_MD, ARTIFACT_DIR / "PAPER_2_NEUROINFORMATICS_v6.md"),
            (MAIN_DOCX, ARTIFACT_DIR / "PAPER_2_NEUROINFORMATICS_v6.docx"),
            (MAIN_PDF, ARTIFACT_DIR / "PAPER_2_NEUROINFORMATICS_v6.pdf"),
            (SUPP_MD, ARTIFACT_DIR / "Online_Resource_1_v6.md"),
            (SUPP_DOCX, ARTIFACT_DIR / "Online_Resource_1_v6.docx"),
            (SUPP_PDF, ARTIFACT_DIR / "Online_Resource_1_v6.pdf"),
            (SUBMISSION_DIR / "01_Cover_Letter_Neuroinformatics.docx", ARTIFACT_DIR / "Cover_Letter_Neuroinformatics_v6.docx"),
            (SUBMISSION_DIR / "02_Research_Highlights_Neuroinformatics.docx", ARTIFACT_DIR / "Highlights_Neuroinformatics_v6.docx"),
            (SUBMISSION_DIR / "03_Title_Page_Neuroinformatics.docx", ARTIFACT_DIR / "Title_Page_Neuroinformatics_v6.docx"),
            (REPORTS_DIR / "Figure_1.png", ARTIFACT_DIR / "Paper2_v6_Figure_1.png"),
            (REPORTS_DIR / "Figure_2.png", ARTIFACT_DIR / "Paper2_v6_Figure_2.png"),
            (REPORTS_DIR / "Figure_3.png", ARTIFACT_DIR / "Paper2_v6_Figure_3.png"),
            (REPORTS_DIR / "Figure_4.png", ARTIFACT_DIR / "Paper2_v6_Figure_4.png"),
        ]
        for s_p, d_p in mirror_map:
            shutil.copy2(s_p, d_p)
        print(f"✓ Mirrored {len(mirror_map)} core v7 deliverables to Artifact Directory.")

    print("=" * 80)
    print("ALL PAPER 2 NEUROINFORMATICS v7 DELIVERABLES BUILT AND AUDITED CLEANLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
