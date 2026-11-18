#!/usr/bin/env python3
"""
scripts/48_build_paper2_neuroinformatics_v5.py
----------------------------------------------
Builds the complete, fully remediated Springer Neuroinformatics (v5) manuscript,
Online Resource 1 (Supplementary Tables S1-S27 & Figures S1-S6), and Editorial Manager
submission package directly from canonical [2.0, 30.0] Hz specparam v2.0.0rc7,
10x5 repeated grouped-CV (13 models w/ GCN_Weighted_PLI & fixed SVMs), Study 2
SHA-256 deduplicated & unfiltered EDF batch audits, 2D fit-window grid, fooof v1.1.1
equivalence, 19x19 spatial covariance Riemannian baselines, and 1,000-permutation
clinical severity regression outputs.

Enforces Springer Neuroinformatics Author-Year citations, alphabetical References with
https://doi.org/... links, 100% citation bijection, scrubbed DOCX author/modifier
metadata, continuous margin line numbering, three-line booktabs tables, and full audits.
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

JOURNAL_NAME = "Neuroinformatics"
PUBLISHER_NAME = "Springer Nature"
EDITOR_NAME = "Prof. John Darrell Van Horn, Ph.D."
EDITOR_TITLE = "Editor-in-Chief, Neuroinformatics"
EDITOR_AFFIL = (
    "Department of Psychology and School of Data Science, "
    "University of Virginia, Charlottesville, VA, USA"
)

EXACT_TITLE = (
    "An Open Multi-Cohort Benchmarking and Auditing Suite for Parameterized Clinical EEG "
    "and Graph Neural Networks Exposes Spectral Window Confounding and Connectivity Limits"
)
SHORT_TITLE = "Open Clinical EEG & GNN Benchmarking Suite"

AUTHOR_NAME = "Chushu Tan"
AFFIL_1 = "School of Sociology, China University of Political Science and Law, Beijing, China"
AFFIL_2 = "Department of Psychology, University at Buffalo, The State University of New York, Buffalo, NY 14260, USA"
CORRESPONDENCE = "Beijing, China; Email: chushuta@gmail.com; ORCID: 0009-0007-3093-0160"
GITHUB_URL = "https://github.com/chushuta-cmyk/transdiagnostic-eeg-benchmark"
SUBMISSION_DATE = "November 18, 2026"

MAIN_MD = MANUSCRIPT_DIR / "PAPER_2_NEUROINFORMATICS_v5.md"
MAIN_DOCX = MANUSCRIPT_DIR / "PAPER_2_NEUROINFORMATICS_v5.docx"
SUPP_MD = MANUSCRIPT_DIR / "PAPER_2_SUPPLEMENTARY_MATERIAL_v5.md"
SUPP_DOCX = MANUSCRIPT_DIR / "PAPER_2_SUPPLEMENTARY_MATERIAL_v5.docx"


# ==============================================================================
# Author-Year References Registry (Strictly Alphabetical; Verified DOIs)
# ==============================================================================
REFS_ALPHA = [
    ("Anjum et al., 2024", "Anjum, M. F., Espinoza, A. I., Cole, R. C., Singh, A., May, P., Uc, E. Y., Dasgupta, S., & Narayanan, N. S. (2024). Resting-state EEG measures cognitive impairment in Parkinson's disease. *npj Parkinson's Disease*, 10, 6. https://doi.org/10.1038/s41531-023-00602-0"),
    ("Barachant et al., 2013", "Barachant, A., Bonnet, S., Congedo, M., & Jutten, C. (2013). Classification of covariance matrices using a Riemannian-based kernel for BCI applications. *Neurocomputing*, 112, 172–178. https://doi.org/10.1016/j.neucom.2012.12.039"),
    ("Bosboom et al., 2006", "Bosboom, J. L. W., Stoffers, D., Stam, C. J., van Dijk, B. W., Verbunt, J., Berendse, H. W., & Wolters, E. C. (2006). Resting state oscillatory brain dynamics in Parkinson's disease: An MEG study. *Clinical Neurophysiology*, 117(11), 2521–2531. https://doi.org/10.1016/j.clinph.2006.06.720"),
    ("Brookshire et al., 2024", "Brookshire, G., Kasper, J., Blauch, N. M., Wu, Y. C., Glatt, R., Merrill, D. A., Gerrol, S., Yoder, K. J., Quirk, C., & Lucero, C. (2024). Data leakage in deep learning studies of translational EEG. *Frontiers in Neuroscience*, 18, 1373515. https://doi.org/10.3389/fnins.2024.1373515"),
    ("Bullmore & Sporns, 2009", "Bullmore, E., & Sporns, O. (2009). Complex brain networks: Graph theoretical analysis of structural and functional systems. *Nature Reviews Neuroscience*, 10(3), 186–198. https://doi.org/10.1038/nrn2575"),
    ("Buzsáki, 2006", "Buzsáki, G. (2006). *Rhythms of the Brain*. Oxford University Press. https://doi.org/10.1093/acprof:oso/9780195301069.001.0001"),
    ("Buzsáki & Draguhn, 2004", "Buzsáki, G., & Draguhn, A. (2004). Neuronal oscillations in cortical networks. *Science*, 304(5679), 1926–1929. https://doi.org/10.1126/science.1099745"),
    ("Cavanagh & Frank, 2014", "Cavanagh, J. F., & Frank, M. J. (2014). Frontal theta as a mechanism for cognitive control. *Trends in Cognitive Sciences*, 18(8), 414–421. https://doi.org/10.1016/j.tics.2014.04.012"),
    ("Cavanagh et al., 2018", "Cavanagh, J. F., Kumar, P., Mueller, A. A., Richardson, S. P., & Mueen, A. (2018). Diminished EEG habituation to novel events effectively classifies Parkinson's patients. *Clinical Neurophysiology*, 129(2), 409–418. https://doi.org/10.1016/j.clinph.2017.11.023 (Dataset: OpenNeuro ds003490, https://doi.org/10.18112/openneuro.ds003490.v1.1.0)"),
    ("Cavanagh et al., 2019", "Cavanagh, J. F., Bismark, A. W., Frank, M. J., & Allen, J. J. B. (2019). Multiple dissociations between comorbid depression and anxiety on reward and punishment processing: Evidence from computationally informed EEG. *Computational Psychiatry*, 3, 1–17. https://doi.org/10.1162/cpsy_a_00024 (Dataset: OpenNeuro ds003478, https://doi.org/10.18112/openneuro.ds003478.v1.1.0)"),
    ("Cohen, 2014", "Cohen, M. X. (2014). *Analyzing Neural Time Series Data: Theory and Practice*. MIT Press."),
    ("Cole & Voytek, 2017", "Cole, S. R., & Voytek, B. (2017). Brain oscillations and the importance of waveform shape. *Trends in Cognitive Sciences*, 21(2), 137–149. https://doi.org/10.1016/j.tics.2016.12.008"),
    ("Colombo et al., 2019", "Colombo, M. A., Napolitani, M., Boly, M., Gosseries, O., Casarotto, S., Rosanova, M., Brichant, J.-F., Boveroux, P., Rex, S., Laureys, S., Massimini, M., Chieregato, A., & Sarasso, S. (2019). The spectral exponent of the resting EEG indexes the presence of consciousness during unresponsiveness induced by propofol, xenon, and ketamine. *NeuroImage*, 189, 631–644. https://doi.org/10.1016/j.neuroimage.2019.01.024"),
    ("Congedo et al., 2017", "Congedo, M., Barachant, A., & Bhatia, R. (2017). Riemannian geometry for EEG-based brain-computer interfaces: A primer and a review. *Brain-Computer Interfaces*, 4(3), 155–174. https://doi.org/10.1080/2326263X.2017.1297192"),
    ("Delorme & Makeig, 2004", "Delorme, A., & Makeig, S. (2004). EEGLAB: An open source toolbox for analysis of single-trial EEG dynamics including independent component analysis. *Journal of Neuroscience Methods*, 134(1), 9–21. https://doi.org/10.1016/j.jneumeth.2003.10.009"),
    ("Dimitriadis et al., 2017a", "Dimitriadis, S. I., Salis, C., Tarnanas, I., & Linden, D. E. J. (2017a). Topological filtering of dynamic functional brain networks unfolds informative chronnectomics: A novel data-driven thresholding scheme based on orthogonal minimal spanning trees (OMSTs). *Frontiers in Neuroinformatics*, 11, 28. https://doi.org/10.3389/fninf.2017.00028"),
    ("Dimitriadis et al., 2017b", "Dimitriadis, S. I., Antonakakis, M., Simos, P., Fletcher, J. M., & Papanicolaou, A. C. (2017b). Data-driven topological filtering based on orthogonal minimal spanning trees: Application to multigroup magnetoencephalography resting-state connectivity. *Brain Connectivity*, 7(10), 661–670. https://doi.org/10.1089/brain.2017.0512"),
    ("Donoghue et al., 2020", "Donoghue, T., Haller, M., Peterson, E. J., Varma, P., Sebastian, P., Gao, R., Noto, T., Lara, A. H., Wallis, J. D., Knight, R. T., Shestyuk, A., & Voytek, B. (2020). Parameterizing neural power spectra into periodic and aperiodic components. *Nature Neuroscience*, 23(12), 1655–1665. https://doi.org/10.1038/s41593-020-00744-x"),
    ("Donoghue et al., 2022", "Donoghue, T., Schaworonkow, N., & Voytek, B. (2022). Methodological considerations for studying neural oscillations. *European Journal of Neuroscience*, 55(11–12), 3502–3527. https://doi.org/10.1111/ejn.15361"),
    ("Finelli et al., 2000", "Finelli, L. A., Baumann, H., Borbély, A. A., & Achermann, P. (2000). Dual electroencephalogram markers of human sleep homeostasis: Correlation between theta activity in waking and slow-wave activity in sleep. *Neuroscience*, 101(3), 523–529. https://doi.org/10.1016/s0306-4522(00)00409-7"),
    ("Fries, 2005", "Fries, P. (2005). A mechanism for cognitive dynamics: Neuronal communication through coherence. *Trends in Cognitive Sciences*, 9(10), 474–480. https://doi.org/10.1016/j.tics.2005.08.011"),
    ("Gao et al., 2017", "Gao, R., Peterson, E. J., & Voytek, B. (2017). Inferring synaptic excitation/inhibition balance from field potentials. *NeuroImage*, 158, 70–78. https://doi.org/10.1016/j.neuroimage.2017.06.078"),
    ("Gemein et al., 2020", "Gemein, L. A. W., Schirrmeister, R. T., Chrabąszcz, P., Wilson, D., Boedecker, J., Schulze-Bonhage, A., Hutter, F., & Ball, T. (2020). Machine-learning-based diagnostics of EEG pathology. *NeuroImage*, 220, 117021. https://doi.org/10.1016/j.neuroimage.2020.117021"),
    ("George et al., 2013", "George, J. S., Strunk, J., Mak-McCully, R., Houser, M., Poizner, H., & Aron, A. R. (2013). Dopaminergic therapy in Parkinson's disease decreases cortical beta band coherence in the resting state and increases beta coherence during motor control. *NeuroImage: Clinical*, 3, 261–270. https://doi.org/10.1016/j.nicl.2013.07.013"),
    ("Gerster et al., 2022", "Gerster, M., Waterstraat, G., Litvak, V., Lehnertz, K., Schnitzler, A., Florin, E., Curio, G., & Nikulin, V. (2022). Separating neural oscillations from aperiodic 1/f activity: Challenges and recommendations. *Neuroinformatics*, 20(4), 991–1012. https://doi.org/10.1007/s12021-022-09581-8"),
    ("Gramfort et al., 2013", "Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D., Brodbeck, C., Goj, R., Jas, M., Brooks, T., Parkkonen, L., & Hämäläinen, M. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267. https://doi.org/10.3389/fnins.2013.00267"),
    ("Jaworska et al., 2012", "Jaworska, N., Blier, P., Fusee, W., & Knott, V. (2012). Alpha power, alpha asymmetry and anterior cingulate cortex activity in depressed males and females. *Journal of Psychiatric Research*, 46(11), 1483–1491. https://doi.org/10.1016/j.jpsychires.2012.08.003"),
    ("Kapur et al., 2012", "Kapur, S., Phillips, A. G., & Insel, T. R. (2012). Why has it taken so long for biological psychiatry to develop clinical tests and what to do about it? *Molecular Psychiatry*, 17(12), 1174–1179. https://doi.org/10.1038/mp.2012.105"),
    ("Kipf & Welling, 2017", "Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *Proceedings of the 5th International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1609.02907"),
    ("Knott et al., 2001", "Knott, V., Mahoney, C., Kennedy, S., & Evans, K. (2001). EEG power, frequency, asymmetry and coherence in male depression. *Psychiatry Research: Neuroimaging*, 106(2), 123–140. https://doi.org/10.1016/s0925-4927(00)00080-9"),
    ("Little et al., 2017", "Little, M. A., Varoquaux, G., Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. (2017). Using and understanding cross-validation strategies: Perspectives on Saeb et al. *GigaScience*, 6(5), gix020. https://doi.org/10.1093/gigascience/gix020"),
    ("Lopes da Silva, 2013", "Lopes da Silva, F. H. (2013). EEG and MEG: Relevance to neuroscience. *Neuron*, 80(5), 1112–1128. https://doi.org/10.1016/j.neuron.2013.10.017"),
    ("Mahato & Paul, 2020", "Mahato, S., & Paul, S. (2020). Classification of depression patients and normal subjects based on electroencephalogram (EEG) signal using alpha power and theta asymmetry. *Journal of Medical Systems*, 44(1), 28. https://doi.org/10.1007/s10916-019-1486-z"),
    ("Miltiadous et al., 2023", "Miltiadous, A., Tzimourta, K. D., Afrantou, T., Ioannidis, P., Grigoriadis, N., Tsalikakis, D. G., Angelidis, P., Tsipouras, M. G., Glavas, E., Giannakeas, N., & Tzallas, A. T. (2023). A dataset of scalp EEG recordings of Alzheimer's disease, frontotemporal dementia and healthy subjects from routine EEG. *Data*, 8(6), 95. https://doi.org/10.3390/data8060095 (Dataset: OpenNeuro ds004504, https://doi.org/10.18112/openneuro.ds004504.v1.0.8)"),
    ("Mitchell et al., 2008", "Mitchell, D. J., McNaughton, N., Flanagan, D., & Kirk, I. J. (2008). Frontal-midline theta from the perspective of hippocampal 'theta'. *Progress in Neurobiology*, 86(3), 156–185. https://doi.org/10.1016/j.pneurobio.2008.09.005"),
    ("Mumtaz et al., 2017", "Mumtaz, W., Xia, L., Ali, S. S. A., Yasin, M. A. M., Hussain, M., & Malik, A. S. (2017). Electroencephalogram (EEG)-based computer-aided technique to diagnose major depressive disorder (MDD). *Biomedical Signal Processing and Control*, 31, 108–115. https://doi.org/10.1016/j.bspc.2016.07.006 (Dataset: Figshare 4244171, https://doi.org/10.6084/m9.figshare.4244171)"),
    ("Newson & Thiagarajan, 2019", "Newson, J. J., & Thiagarajan, T. C. (2019). EEG frequency bands in psychiatric disorders: A review of resting state studies. *Frontiers in Human Neuroscience*, 12, 521. https://doi.org/10.3389/fnhum.2018.00521"),
    ("Nunez & Srinivasan, 2006", "Nunez, P. L., & Srinivasan, R. (2006). *Electric Fields of the Brain: The Neurophysics of EEG* (2nd ed.). Oxford University Press."),
    ("Olbrich & Arns, 2013", "Olbrich, S., & Arns, M. (2013). EEG biomarkers in major depressive disorder: Discriminative power and prediction of treatment response. *International Review of Psychiatry*, 25(5), 604–618. https://doi.org/10.3109/09540261.2013.816269"),
    ("Rockhill et al., 2020", "Rockhill, A. P., Jackson, N., George, J., Aron, A., & Swann, N. C. (2020). UC San Diego Resting State EEG Data from Patients with Parkinson's Disease. *OpenNeuro*, ds002778. https://doi.org/10.18112/openneuro.ds002778"),
    ("Roy et al., 2019", "Roy, Y., Banville, H., Albuquerque, I., Gramfort, A., Falk, T. H., & Faubert, J. (2019). Deep learning-based electroencephalography analysis: A systematic review. *Journal of Neural Engineering*, 16(5), 051001. https://doi.org/10.1088/1741-2552/ab260c"),
    ("Rubinov & Sporns, 2010", "Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity: Uses and interpretations. *NeuroImage*, 52(3), 1059–1069. https://doi.org/10.1016/j.neuroimage.2009.10.003"),
    ("Rutherford et al., 2022", "Rutherford, S., Kia, S. M., Wolfers, T., Fraza, C., Zabihi, M., Dinga, R., Berthet, P., Worker, A., Verdi, S., Ruhe, H. G., Beckmann, C. F., & Marquand, A. F. (2022). The normative modeling framework for computational psychiatry. *Nature Protocols*, 17(7), 1711–1734. https://doi.org/10.1038/s41596-022-00696-5"),
    ("Saeb et al., 2017", "Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. (2017). The need to approximate the use-case in clinical machine learning. *GigaScience*, 6(5), gix019. https://doi.org/10.1093/gigascience/gix019"),
    ("Saeedi et al., 2021", "Saeedi, A., Saeedi, M., Maghsoudi, A., & Shalbaf, A. (2021). Major depressive disorder diagnosis based on effective connectivity in EEG signals: A convolutional neural network and long short-term memory approach. *Cognitive Neurodynamics*, 15(2), 239–252. https://doi.org/10.1007/s11571-020-09619-0"),
    ("Scheinost et al., 2019", "Scheinost, D., Noble, S., Horien, C., Greene, A. S., Lake, E. M. R., Salehi, M., Gao, S., Shen, X., O'Connor, D., Barron, D. S., Yip, S. W., Rosenberg, M. D., & Constable, R. T. (2019). Ten simple rules for predictive modeling of individual differences in neuroimaging. *NeuroImage*, 193, 35–45. https://doi.org/10.1016/j.neuroimage.2019.02.057"),
    ("Singh et al., 2023", "Singh, A., Cole, R., Espinoza, A., Cavanagh, J., & Narayanan, N. (2023). Rest eyes open (Version v1.0.0) [Data set]. *OpenNeuro*, ds004584. https://doi.org/10.18112/openneuro.ds004584.v1.0.0"),
    ("Stam et al., 2007", "Stam, C. J., Nolte, G., & Daffertshofer, A. (2007). Phase lag index: Assessment of functional connectivity from multi-channel EEG and MEG with diminished bias from common sources. *Human Brain Mapping*, 28(11), 1178–1193. https://doi.org/10.1002/hbm.20346"),
    ("Stam, 2010", "Stam, C. J. (2010). Use of magnetoencephalography (MEG) to study functional brain networks in neurodegenerative disorders. *Journal of the Neurological Sciences*, 289(1–2), 128–134. https://doi.org/10.1016/j.jns.2009.08.028"),
    ("Stam, 2014", "Stam, C. J. (2014). Modern network science of neurological disorders. *Nature Reviews Neuroscience*, 15(10), 683–695. https://doi.org/10.1038/nrn3801"),
    ("Stoffers et al., 2007", "Stoffers, D., Bosboom, J. L. W., Deijen, J. B., Wolters, E. C., Berendse, H. W., & Stam, C. J. (2007). Slowing of oscillatory brain activity is a stable characteristic of Parkinson's disease without dementia. *Brain*, 130(7), 1847–1860. https://doi.org/10.1093/brain/awm034"),
    ("Varoquaux, 2018", "Varoquaux, G. (2018). Cross-validation failure: Small sample sizes lead to large error bars. *NeuroImage*, 180, 68–77. https://doi.org/10.1016/j.neuroimage.2017.06.061"),
    ("Vinck et al., 2011", "Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F., & Pennartz, C. M. A. (2011). An improved index of phase-synchronization for electrophysiological data in the presence of volume-conduction, noise and sample-size bias. *NeuroImage*, 55(4), 1548–1565. https://doi.org/10.1016/j.neuroimage.2011.01.055"),
    ("Voytek et al., 2015", "Voytek, B., Kramer, M. A., Case, J., Lepage, K. Q., Tempesta, Z. R., Knight, R. T., & Gazzaley, A. (2015). Age-related changes in 1/f neural electrophysiological noise. *Journal of Neuroscience*, 35(38), 13257–13265. https://doi.org/10.1523/JNEUROSCI.2332-14.2015"),
    ("Wolfers et al., 2018", "Wolfers, T., Doan, N. T., Kaufmann, T., Alnæs, D., Moberget, T., Agartz, I., Buitelaar, J. K., Ueland, T., Melle, I., Franke, B., Andreassen, O. A., Beckmann, C. F., Westlye, L. T., & Marquand, A. F. (2018). Mapping the heterogeneous phenotype of schizophrenia and bipolar disorder using normative models. *JAMA Psychiatry*, 75(11), 1146–1155. https://doi.org/10.1001/jamapsychiatry.2018.2467"),
    ("Xiang et al., 2024", "Xiang, C., Fan, X., Bai, D., Lv, K., & Lei, X. (2024). A resting-state EEG dataset for sleep deprivation. *Scientific Data*, 11, 427. https://doi.org/10.1038/s41597-024-03268-2 (Dataset: OpenNeuro ds004902)"),
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
# OOXML Document Styling Helpers
# ==============================================================================
def add_continuous_line_numbering(doc):
    for section in doc.sections:
        sectPr = section._sectPr
        lnNumType = OxmlElement("w:lnNumType")
        lnNumType.set(qn("w:countBy"), "1")
        lnNumType.set(qn("w:restart"), "continuous")
        sectPr.append(lnNumType)


def apply_three_line_table_styling(tbl):
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tblPr = tbl._tbl.tblPr
    for existing in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(existing)
    tblPr.append(parse_xml(r"""
        <w:tblBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
            <w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>
            <w:left w:val="none"/><w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>
            <w:right w:val="none"/><w:insideH w:val="none"/><w:insideV w:val="none"/>
        </w:tblBorders>
    """))
    col_count = len(tbl.columns) if tbl.columns else (len(tbl.rows[0].cells) if tbl.rows else 0)
    is_wide = col_count > 7
    hdr_pt, data_pt = (Pt(8.2), Pt(7.8)) if is_wide else (Pt(9.5), Pt(9.0))
    line_sp, sp_pad = (1.05, Pt(1.4)) if is_wide else (1.15, Pt(2.0))

    for r_idx, row in enumerate(tbl.rows):
        trPr = row._tr.get_or_add_trPr()
        if trPr.find(qn("w:cantSplit")) is None:
            trPr.append(parse_xml(r'<w:cantSplit xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>'))
        if r_idx == 0 and trPr.find(qn("w:tblHeader")) is None:
            trPr.append(parse_xml(r'<w:tblHeader xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>'))

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
    for s in doc.sections:
        footer = s.footer
        p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.text = ""
        fld_xml = parse_xml(r'''
            <w:fldSimple xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:instr="PAGE">
                <w:r>
                    <w:rPr>
                        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
                        <w:sz w:val="20"/>
                        <w:color w:val="64748B"/>
                    </w:rPr>
                    <w:t>1</w:t>
                </w:r>
            </w:fldSimple>
        ''')
        p._p.append(fld_xml)


def style_submission_doc(doc_path: Path, is_main: bool = False):
    doc = docx.Document(doc_path)
    doc.core_properties.author = ""
    doc.core_properties.last_modified_by = ""
    doc.core_properties.comments = ""
    for s in doc.sections:
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


def patch_settings_and_clean_core(docx_path: Path):
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


def md_to_docx(md_path: Path, docx_path: Path, is_main: bool = False):
    res_path = f"{MANUSCRIPT_DIR}:{REPORTS_DIR}:{SUBMISSION_DIR / 'Figures' / 'Main_Figures'}:{SUBMISSION_DIR / 'Figures' / 'Supplementary_Figures'}"
    cmd = ["pandoc", str(md_path), f"--resource-path={res_path}", "-f", "markdown+tex_math_dollars", "-t", "docx", "-o", str(docx_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Pandoc failed for {md_path.name}:\n{res.stderr}")
    style_submission_doc(docx_path, is_main=is_main)
    patch_settings_and_clean_core(docx_path)


# ==============================================================================
# Live Data Loader
# ==============================================================================
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
        "reg": pd.read_csv(REPORTS_DIR / "multicohort_clinical_regression_benchmarks.csv"),
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
    }
    data["reg"]["cohort"] = data["reg"]["cohort"].apply(strip_md_bold)
    return data


# ==============================================================================
# Dynamic Main Tables Builders
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
        "### Table 1. Canonical Interior [2.0, 30.0] Hz Parameterized EEG Biophysics and Fit Quality Across All Ten Contrasts",
        "| Study & Contrast | Design | Frontal θ (4–8 Hz) d [95% CI] | p / q_10 | Posterior α d (p) | Global χ d (p) | Fit R² / MAE (B) | F3–Fz–F4 θ d (p) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for label, bname, fkey, design in rows_cfg:
        ft = bio[(bio["cohort"] == bname) & (bio["marker_key"] == "frontal_theta")].iloc[0]
        pa = bio[(bio["cohort"] == bname) & (bio["marker_key"] == "posterior_alpha")].iloc[0]
        gc = bio[(bio["cohort"] == bname) & (bio["marker_key"] == "global_chi")].iloc[0]
        f3r = f3.loc[bname]
        finfo = fit_230.loc[fkey]
        lines.append(
            f"| {label} | {design} | "
            f"{ft['cohens_d']:+.3f} [{ft['ci_lower']:+.3f}, {ft['ci_upper']:+.3f}] | "
            f"{fmt_p(ft['p_value'])} / {fmt_p(ft['fdr_q_10'])} | "
            f"{pa['cohens_d']:+.3f} ({fmt_p(pa['p_value'])}) | "
            f"{gc['cohens_d']:+.3f} ({fmt_p(gc['p_value'])}) | "
            f"{finfo['r2_median']:.3f} / {finfo['mae_median']:.3f} | "
            f"{f3r['cohens_d']:+.3f} ({fmt_p(f3r['p_value'])}) |"
        )
    meta_p = d["meta"][(d["meta"]["method"] == "specparam_2to30_resid") & (d["meta"]["analysis_set"] == "Primary_K6_SubjMean_PD_OFF")].set_index("subgroup")
    meta_k5 = d["meta"][(d["meta"]["method"] == "specparam_2to30_resid") & (d["meta"]["analysis_set"] == "Sens_K5_Omit_Study2_BatchConfounded")].set_index("subgroup")
    dep = meta_p.loc["Depression_Cohorts_k2"]
    ndg = meta_p.loc["PD_Dementia_Cohorts"]
    all6 = meta_p.loc["Overall_Pooled"]
    all5 = meta_k5.loc["Overall_Pooled"]
    ucsd_all = d["extra_bio"]["UCSD_All_theta"]
    lines.append("")
    lines.append(
        f"*Note: All values derive from `specparam v2.0.0rc7` fitted inside the passband-interior window [2.0, 30.0] Hz "
        f"(five frontal sites `Fp1, Fp2, F3, F4, Fz` for Frontal θ; five posterior sites `P3, P4, Pz, O1, O2` for Posterior α; all 19 sites for Global χ). "
        f"*Study 2 removes two SHA-256 duplicate pairs (`29 MDD vs. 27 HC` unique); batch-stratified sensitivity appears in Online Resource 1 (Supplementary Table S26). "
        f"In Study 4, participant averaging avoids treating paired PD visits as independent; pooling all 30 PD sessions gives d = {ucsd_all['cohens_d']:+.3f} (p = {fmt_p(ucsd_all['p_value'])}). "
        f"Study 6 analyzes contiguous 60 s pure EO rest (pure EC sensitivity in Online Resource 1, Supplementary Table S25). "
        f"The primary replicated random-effects finding across k = 4 neurodegenerative PD/dementia cohorts (Studies 4 SubjMean, 5c, 6a, 7) yields "
        f"d_RE = {ndg['pooled_d']:+.3f} (HKSJ 95% CI [{ndg['hksj_ci_lower']:+.3f}, {ndg['hksj_ci_upper']:+.3f}], p_HKSJ = {fmt_p(ndg['p_hksj'])}, I² = {ndg['I2_percent']:.1f}%). "
        f"Across K = 6 chronic cohorts (N = 493 unique), depression k = 2 pools at d_RE = {dep['pooled_d']:+.3f} (DL p = {fmt_p(dep['p_value'])}; df = 1 HKSJ p = {fmt_p(dep['p_hksj'])}; Q_between = {dep['Q_between']:.2f}, p = {fmt_p(dep['p_between'])}, pooled I² = {all6['I2_percent']:.1f}%); "
        f"omitting batch-confounded Study 2 (K = 5, N = 437) gives Q_between = {all5['Q_between']:.2f} (p = {fmt_p(all5['p_between'])}; pooled I² = {all5['I2_percent']:.1f}%; Online Resource 1, Supplementary Table S22).*"
    )
    return "\n".join(lines)


def build_table_2(d: dict) -> str:
    w = d["bio_wide"]
    boot = d["boot_d"]
    ft_df = w[w["marker"] == "frontal_theta_5ch"].set_index("cohort")
    chi_df = w[w["marker"] == "global_chi"].set_index("cohort")
    b_ols = boot[(boot["marker"] == "frontal_theta_5ch") & (boot["comparison"] == "OLS_1_40_minus_SP_2_30")].set_index("cohort")
    b_win = boot[(boot["marker"] == "frontal_theta_5ch") & (boot["comparison"] == "SP_1_40_minus_SP_2_30")].set_index("cohort")
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
            f"| {label} | "
            f"{fr['cohens_d_masked_ols_1to40']:+.3f} ({fmt_p(fr['p_value_masked_ols_1to40'])}) | "
            f"{fr['cohens_d_specparam_resid_1to40']:+.3f} ({fmt_p(fr['p_value_specparam_resid_1to40'])}) | "
            f"{fr['cohens_d_specparam_peaks_2to30']:+.3f} ({fmt_p(fr['p_value_specparam_peaks_2to30'])}) | "
            f"{fr['cohens_d_specparam_resid_2to30']:+.3f} ({fmt_p(fr['p_value_specparam_resid_2to30'])}) | "
            f"{bw['delta_d']:+.3f} ({fmt_p(bw['boot_p'])}) | "
            f"{cr['cohens_d_specparam_resid_1to40']:+.3f} / {cr['cohens_d_specparam_resid_2to30']:+.3f} |"
        )
    b_s2 = b_win.loc["Study 2: Outpatient MDD"]
    b_s3_ols = b_ols.loc["Study 3: Univ. Depression Screening"]
    lines.append("")
    lines.append(
        f"*Note: Δd reports the stratified 1,000-bootstrap paired shift d_[1–40] − d_[2–30] with two-sided bootstrap p-value (full bootstrap deltas and OLS[2–30] sensitivity appear in Online Resource 1, Supplementary Table S11). "
        f"`specparam [2–30] Peaks` sums fitted Gaussian components (zeroing sub-threshold peaks below `min_peak_height = 0.05 B`), whereas `Resid` (`log10 PSD − aperiodic fit`) captures continuous low-amplitude periodic shifts; `Peaks` matches strong AD slowing (`{ft_df.loc['Study 5a: Alzheimer\'s Disease (AD)', 'cohens_d_specparam_peaks_2to30']:+.3f}` vs. `{ft_df.loc['Study 5a: Alzheimer\'s Disease (AD)', 'cohens_d_specparam_resid_2to30']:+.3f}`) while showing slightly smaller effect sizes when discrete peaks fall near detection threshold. "
        f"In Study 2 MDD EC, extending fits into 40 Hz pre-export skirts masked [1–40] theta ({ft_df.loc['Study 2: Outpatient MDD', 'cohens_d_specparam_resid_1to40']:+.3f} vs. {ft_df.loc['Study 2: Outpatient MDD', 'cohens_d_specparam_resid_2to30']:+.3f}; bootstrap Δd = {b_s2['delta_d']:+.3f} [{b_s2['ci_lower']:+.3f}, {b_s2['ci_upper']:+.3f}], p = {fmt_p(b_s2['boot_p'])}) and inverted global χ ({chi_df.loc['Study 2: Outpatient MDD', 'cohens_d_specparam_resid_1to40']:+.3f} vs. {chi_df.loc['Study 2: Outpatient MDD', 'cohens_d_specparam_resid_2to30']:+.3f}). "
        f"In Study 3 university depression, leaving 4–7 Hz unmasked in OLS inflated theta negativity ({ft_df.loc['Study 3: Univ. Depression Screening', 'cohens_d_masked_ols_1to40']:+.3f} vs. {ft_df.loc['Study 3: Univ. Depression Screening', 'cohens_d_specparam_resid_2to30']:+.3f}; Δd = {b_s3_ols['delta_d']:+.3f} [{b_s3_ols['ci_lower']:+.3f}, {b_s3_ols['ci_upper']:+.3f}], p = {fmt_p(b_s3_ols['boot_p'])}).*"
    )
    return "\n".join(lines)


def build_table_3(d: dict) -> str:
    rep = d["rep10"]
    wil8 = d["wilcoxon"][d["wilcoxon"]["scope"] == "NonNested_8_Contrasts_10x5Mean"].set_index("comparator_model")
    wil9 = d["wilcoxon"][d["wilcoxon"]["scope"] == "Primary_9_Contrasts_10x5Mean"].set_index("comparator_model")
    p9_cohorts = [
        ("Study 2: Mumtaz MDD (Eyes Closed)", "Study 2: MDD EC (N=56)"),
        ("Study 3: Cavanagh University Depression", "Study 3: Univ. Dep. (N=119)"),
        ("Study 4: UCSD Parkinson All Sessions", "Study 4: UCSD PD (N=46)"),
        ("Study 5a: Dementia AD vs Control", "Study 5a: Dementia AD (N=65)"),
        ("Study 5b: Dementia FTD vs Control", "Study 5b: Dementia FTD (N=52)"),
        ("Study 5c: Dementia All (AD+FTD) vs Control", "Study 5c: Dementia All (N=88)"),
        ("Study 6a: UNM PD-OFF vs Control", "Study 6a: UNM PD-OFF EO (N=50)"),
        ("Study 7: Iowa Parkinson (ds004584)", "Study 7: Iowa PD (N=149)"),
        ("Study 1: Sleep Deprivation (ds004902)", "Study 1: Sleep Depr. (N=142)"),
    ]
    models = [
        ("GCN_OMST", "GCN-OMST"),
        ("GCN_Weighted_PLI", "GCN-WPLI"),
        ("GCN_Identity", "GCN-Id"),
        ("GCN_Random_EdgeMatched", "GCN-Rand*"),
        ("GCN_Full", "GCN-Full"),
        ("Tuned_LogisticRegression", "Tuned LR"),
        ("Tuned_Ridge", "Ridge"),
        ("Linear_SVM", "Lin-SVM"),
        ("RBF_SVM", "RBF-SVM"),
        ("RandomForest", "RF"),
        ("GBDT", "GBDT"),
        ("Single_Theta", "Single θ"),
        ("Single_Chi", "Single χ"),
    ]
    piv = rep.pivot(index="cohort", columns="model", values="mean_auc")
    hdr = "| Primary Contrast | " + " | ".join(lbl for _, lbl in models) + " |"
    sep = "| :--- | " + " | ".join(":---:" for _ in models) + " |"
    lines = [
        "### Table 3. True 10×5 Repeated Participant-Grouped Cross-Validated Mean ROC-AUC Across Contrasts and Thirteen Models",
        hdr, sep,
    ]
    for cname, slabel in p9_cohorts:
        vals = [float(piv.loc[cname, mkey]) for mkey, _ in models]
        mx = max(vals)
        lines.append(f"| {slabel} | " + " | ".join(f"**{v:.3f}**" if abs(v - mx) < 1e-9 else f"{v:.3f}" for v in vals) + " |")

    p9_cnames = [c for c, _ in p9_cohorts]
    nn8_cnames = [c for c in p9_cnames if "5c" not in c]
    means8 = [float(piv.loc[nn8_cnames, mkey].mean()) for mkey, _ in models]
    means9 = [float(piv.loc[p9_cnames, mkey].mean()) for mkey, _ in models]
    mx8, mx9 = max(means8), max(means9)
    lines.append("| Mean Across k = 8 Non-Nested Contrasts | " + " | ".join(f"**{v:.3f}**" if abs(v - mx8) < 1e-9 else f"{v:.3f}" for v in means8) + " |")
    lines.append("| Mean Across k = 9 Primary Contrasts | " + " | ".join(f"**{v:.3f}**" if abs(v - mx9) < 1e-9 else f"{v:.3f}" for v in means9) + " |")
    lines.append("")
    lines.append(
        f"*Note: Each cell reports the 10-repeat mean out-of-fold ROC-AUC (`50` outer grouped folds across 10 shuffled participant-grouped seeds; 0.0% participant overlap; fold-nested normative z-scoring; ± SD and Split-0 95% CIs appear in Online Resource 1, Supplementary Tables S10 and S20). "
        f"`GCN-WPLI` replaces binary OMST adjacency with continuous PLI-weighted OMST edges (`A_bin ⊙ PLI`). `*GCN-Rand` uses session-specific degree-agnostic random undirected graphs matching exact OMST edge counts. "
        f"Across k = 8 strictly non-nested contrasts, `GCN-OMST` ({means8[0]:.3f}) is statistically equivalent within ±0.02 AUC to edge-matched `GCN-Rand` ({means8[3]:.3f}, ΔAUC = {wil8.loc['GCN_Random_EdgeMatched', 'mean_diff_auc']:+.3f} [95% CI {wil8.loc['GCN_Random_EdgeMatched', 'ci95_diff_lo']:+.3f}, {wil8.loc['GCN_Random_EdgeMatched', 'ci95_diff_hi']:+.3f}], Wilcoxon p = {wil8.loc['GCN_Random_EdgeMatched', 'raw_p']:.4f}, TOST p = {wil8.loc['GCN_Random_EdgeMatched', 'tost_p_002']:.4f}), `GCN-Full` ({means8[4]:.3f}, TOST p = {wil8.loc['GCN_Full', 'tost_p_002']:.4f}), and `GCN-WPLI` ({means8[1]:.3f}, TOST p = {wil8.loc['GCN_Weighted_PLI', 'tost_p_002']:.4f}), and numerically trailed `Linear SVM` ({means8[7]:.3f}), `Tuned LR` ({means8[5]:.3f}), `Random Forest` ({means8[9]:.3f}), and `Tuned Ridge` ({means8[6]:.3f}); k = 9 primary results match (`TOST p = {wil9.loc['GCN_Random_EdgeMatched', 'tost_p_002']:.4f}` vs. `GCN-Rand`; Online Resource 1, Supplementary Table S10b).*"
    )
    return "\n".join(lines)


def build_table_4(d: dict) -> str:
    pd_tx, td_tx, gc = d["pd_tx"], d["td_tx"], d["gain_cal"]

    def mean_sd_str(row):
        return f"{row['auc']:.3f} (±{row['split_hc_sd']:.3f})"

    lines = [
        "### Table 4. Leak-Free `Split-Target-HC` External Transfer, Calibration Trade-Offs, and Gain Stress-Testing Under [2.0, 30.0] Hz `specparam`",
        "#### Panel A. Disjoint 50/50 `Split-Target-HC` External Transfer Without Model Retraining (Mean ROC-AUC Across 20 Splits)",
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
        ("Iowa_PD", "Dementia_All", "Iowa PD (149) → Dementia All (88)", "Slowing (Cross-Disease)"),
        ("Sleep_Debt", "Dementia_All", "Sleep Deprivation (142) → Dementia All (88)", "Slowing (Acute → Degenerative)"),
        ("Sleep_Debt", "Iowa_PD", "Sleep Deprivation (142) → Iowa PD (149)", "Slowing (Acute → Degenerative)"),
        ("Dementia_All", "Mumtaz_MDD", "Dementia All (88) → Outpatient MDD EC (56)", "Cross-Disorder (Discordant)"),
        ("Sleep_Debt", "Mumtaz_MDD", "Sleep Deprivation (142) → Outpatient MDD EC (56)", "Cross-Disorder (Discordant)"),
        ("Mumtaz_MDD", "Dementia_All", "Outpatient MDD EC (56) → Dementia All (88)", "Cross-Disorder (Reverse)"),
    ]:
        r_g = td_raw.loc[(src, tgt)]
        vals = [float(td_piv.loc[(src, tgt), m]) for m in ("GCN_OMST", "GCN_Identity", "Tuned_LR", "RandomForest", "GBDT")]
        mx = max(vals)
        lines.append(f"| {lbl} | {rtype} | " + " | ".join(f"**{mean_sd_str(r_g) if i == 0 else f'{v:.3f}'}**" if abs(v - mx) < 1e-9 else (mean_sd_str(r_g) if i == 0 else f"{v:.3f}") for i, v in enumerate(vals)) + " |")

    lines.append("")
    lines.append("#### Panel B. Mean 6-Pair Cross-Site PD Transfer Across Calibration Regimes and Gain Shifts (Mean and Worst-Case Max Drift)")
    lines.append("| Experiment Regime | Spectral Feature | Calibration / Gain k | Tuned LR AUC (BalAcc / Brier) | Random Forest AUC | Mean / Max |ΔFeat| | Mean / Max |Δp| (LR) |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")

    cal_sub = gc[gc["experiment"] == "Calibration_Strategy"]
    for cal_name, cal_lbl in [("Uncalibrated_Raw", "Uncalibrated"), ("Global_Cohort_ZScore", "Global Cohort Z"), ("Target_HC_Normative", "Full Target-HC Z")]:
        for fkey, flbl in [
            ("specparam_periodic_residual", "`specparam` [2–30] + χ"),
            ("raw_log_band_power", "Unparam. log10 Power + χ"),
            ("relative_band_power", "Relative Power + χ"),
            ("masked_ols_residual", "[7,14] Masked-OLS [2–30] + χ"),
        ]:
            sub = cal_sub[(cal_sub["calibration"] == cal_name) & (cal_sub["feature_type"] == fkey)]
            lines.append(f"| Baseline Calibration (k = 1.0) | {flbl} | {cal_lbl} | {sub['lr_auc'].mean():.3f} ({sub['lr_bacc'].mean():.3f} / {sub['lr_brier'].mean():.3f}) | {sub['rf_auc'].mean():.3f} | 0.000 / 0.000 | 0.000 / 0.000 |")

    amp_sub = gc[gc["experiment"] == "Amplifier_Gain_Ablation"]
    for fkey, flbl in [("specparam_periodic_residual", "`specparam` [2–30] + χ"), ("raw_log_band_power", "Unparam. log10 Power + χ")]:
        for gk, klbl in [("5.0", "Uniform k = 5.0"), ("Heterogeneous_0.5x_5.0x", "Mixed 0.5x / 5.0x")]:
            sub = amp_sub[(amp_sub["feature_type"] == fkey) & (amp_sub["gain_k"].astype(str) == gk)]
            lines.append(f"| Gain Perturbation (Uncalibrated) | {flbl} | {klbl} | {sub['lr_auc'].mean():.3f} ({sub['lr_bacc'].mean():.3f} / {sub['lr_brier'].mean():.3f}) | {sub['rf_auc'].mean():.3f} | {sub['mean_feat_drift'].mean():.3f} / {sub['max_feat_drift'].max():.3f} | {sub['lr_mean_prob_drift'].mean():.3f} / {sub['lr_max_prob_drift'].max():.3f} |")

    lines.append("")
    lines.append("*Note: Panel A reports leak-free 50/50 `Split-Target-HC` external evaluation in which half of target healthy controls calibrate standardization strictly disjointly from held-out target controls and patients (20 splits; Oracle full-HC benchmarks appear in Online Resource 1, Supplementary Tables S15, S16, and S19).*")
    return "\n".join(lines)


def build_table_5(d: dict) -> str:
    reg = d["reg"]
    lines = [
        "### Table 5. Exploratory 10×5 Repeated Out-of-Fold Clinical Severity Regression Benchmarks (With 1,000-Permutation Tests)",
        "| Cohort & Scope | Clinical Scale | N | Best Regressor | 10×5 Mean r (±SD) | Perm p / Median p | Mean OOF R² | Mean MAE | Ridge 10×5 r (Perm p) |",
        "| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for ckey, cname, sname, nval in [
        ("Dementia_MMSE_All_N88", "Study 5: Dementia All (ds004504)", "MMSE (4.0–30.0)", 88),
        ("Dementia_MMSE_Patients_Only_N59", "Study 5: Dementia Patients Only", "MMSE (4.0–29.0)", 59),
        ("Iowa_PD_MoCA_All_N149", "Study 7: Iowa PD All (ds004584)", "MoCA (9.0–30.0)", 149),
        ("Iowa_PD_MoCA_Patients_Only_N100", "Study 7: Iowa PD Patients Only", "MoCA (9.0–30.0)", 100),
        ("Iowa_PD_UPDRS_Patients_N100", "Study 7: Iowa PD Patients Only", "UPDRS-III (1.0–28.0)", 100),
    ]:
        sub = reg[reg["cohort_key"] == ckey]
        best = sub.loc[sub["pearson_r_mean"].idxmax()]
        lin = sub[sub["model"] == "Ridge_Regression"].iloc[0]
        bname = best["model"].replace("_", " ")
        lines.append(
            f"| {cname} | {sname} | {nval} | {bname} | {best['pearson_r_mean']:+.3f} (±{best['pearson_r_sd']:.3f}) | "
            f"{fmt_p(best['perm_p'])} / {fmt_p(best['median_p'])} | {best['r2_mean']:.3f} | {best['mae_mean']:.3f} | "
            f"{lin['pearson_r_mean']:+.3f} ({fmt_p(lin['perm_p'])}) |"
        )
    lines.append("")
    lines.append("*Note: Out-of-fold 10×5 K-fold regressions on flattened 76-D canonical [2.0, 30.0] Hz `specparam` features; `Perm p` denotes one-sided Split-0 1,000-label-permutation retraining p-value (`r_null >= r_obs`; full 20-row breakdown in Online Resource 1, Supplementary Table S14). Removing ceiling controls (`MMSE = 30.0`) abolishes dementia MMSE significance (`N = 59`: all `perm p >= 0.111`, `OOF R² <= -0.047`), and while larger Iowa PD patient samples (`N = 100`) retain weak rank correlation (`MoCA SVR r = +0.293, perm p = 0.0020`; `UPDRS-III GBDT r = +0.242, perm p = 0.0070`), out-of-fold variance explained remains below 3.6% across all 12 patient-only models (`OOF R² <= +0.035`; 10/12 negative `R²`).*")
    return "\n".join(lines)


# ==============================================================================
# Main Manuscript Builder (Neuroinformatics v5)
# ==============================================================================
def build_main_manuscript(d: dict) -> str:
    t1, t2, t3, t4, t5 = build_table_1(d), build_table_2(d), build_table_3(d), build_table_4(d), build_table_5(d)
    acc = d["master"]["accounting"]
    omst = d["master"]["omst_topology"]
    perms = d["perms"]
    bio_w = d["bio_wide"]
    ft_w = bio_w[bio_w["marker"] == "frontal_theta_5ch"].set_index("cohort")
    chi_w = bio_w[bio_w["marker"] == "global_chi"].set_index("cohort")
    bio_exp = d["bio_exp"]
    boot_d = d["boot_d"]
    sim_mc = d["sim_mc"]
    fit230 = d["fit"][d["fit"]["fit_window_hz"] == "2.0-30.0"].set_index("cohort_key")
    fit140 = d["fit"][d["fit"]["fit_window_hz"] == "1.0-40.0"].set_index("cohort_key")
    gchk = d["gain_chk"]
    pli_s = d["pli_sum"].iloc[0]
    pli_a = d["pli_audit"].set_index("cohort")
    riem = d["riem"]
    nn8_r_cohs = [c for c in riem["cohort"].unique() if "5c" not in c and "6b" not in c]
    riem_nn8 = riem[riem["cohort"].isin(nn8_r_cohs)].groupby(["representation", "classifier"])["auc"].mean()

    rep = d["rep10"]
    p9_cnames = [c for c in rep["cohort"].unique() if not any(k in c for k in ("2b", "6b", "6c"))]
    nn8_cnames = [c for c in p9_cnames if "5c" not in c]
    rep_p9 = rep[rep["cohort"].isin(p9_cnames)].pivot(index="cohort", columns="model", values="mean_auc")
    rep_nn8 = rep[rep["cohort"].isin(nn8_cnames)].pivot(index="cohort", columns="model", values="mean_auc")
    rep_idx = rep.set_index(["cohort", "model"])

    wil8 = d["wilcoxon"][d["wilcoxon"]["scope"] == "NonNested_8_Contrasts_10x5Mean"].set_index("comparator_model")
    wil9 = d["wilcoxon"][d["wilcoxon"]["scope"] == "Primary_9_Contrasts_10x5Mean"].set_index("comparator_model")

    meta_230 = d["meta"][(d["meta"]["method"] == "specparam_2to30_resid") & (d["meta"]["analysis_set"] == "Primary_K6_SubjMean_PD_OFF")].set_index("subgroup")
    meta_k5 = d["meta"][(d["meta"]["method"] == "specparam_2to30_resid") & (d["meta"]["analysis_set"] == "Sens_K5_Omit_Study2_BatchConfounded")].set_index("subgroup")
    dep_m, ndg_m = meta_230.loc["Depression_Cohorts_k2"], meta_230.loc["PD_Dementia_Cohorts"]

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
    dem_all_svr = reg[(reg["cohort_key"] == "Dementia_MMSE_All_N88") & (reg["model"] == "SVR_RBF")].iloc[0]
    dem_pt_rf = reg[(reg["cohort_key"] == "Dementia_MMSE_Patients_Only_N59") & (reg["model"] == "Random_Forest_Regressor")].iloc[0]
    dem_pt_ridge = reg[(reg["cohort_key"] == "Dementia_MMSE_Patients_Only_N59") & (reg["model"] == "Ridge_Regression")].iloc[0]
    iowa_all_svr = reg[(reg["cohort_key"] == "Iowa_PD_MoCA_All_N149") & (reg["model"] == "SVR_RBF")].iloc[0]
    iowa_pt_svr = reg[(reg["cohort_key"] == "Iowa_PD_MoCA_Patients_Only_N100") & (reg["model"] == "SVR_RBF")].iloc[0]
    iowa_pt_ridge = reg[(reg["cohort_key"] == "Iowa_PD_MoCA_Patients_Only_N100") & (reg["model"] == "Ridge_Regression")].iloc[0]
    iowa_updrs_gbdt = reg[(reg["cohort_key"] == "Iowa_PD_UPDRS_Patients_N100") & (reg["model"] == "GBDT_Regressor")].iloc[0]

    med_df = d["med"].set_index(["cohort", "marker"])
    unm_med_ft = med_df.loc[("Study 6: UNM PD Pure EO (>=15h Washout) (N=25)", "Frontal Periodic Theta (4–8 Hz)")]
    ucsd_med_ft = med_df.loc[("Study 4: UCSD PD (Acute Washout) (N=15)", "Frontal Periodic Theta (4–8 Hz)")]
    unm_sens_df = d["unm_sens"].set_index("condition")
    iapf_df = d["iapf"].set_index("cohort")
    iowa_iapf = iapf_df.loc["Study 7: Iowa Parkinson Cohort"]
    mum_conc = d["mum_conc"].set_index("marker")
    mum_tx = d["mum_tx"].set_index(["source_cohort", "target_cohort"])
    hparam_means = d["hparam"].groupby("config_label")["auc"].mean()

    pd_tx = d["pd_tx"].pivot(index=["source_cohort", "target_cohort"], columns="model", values="auc")
    td_tx = d["td_tx"].pivot(index=["source_cohort", "target_cohort"], columns="model", values="auc")
    s2_for = d["mum_forensic"].set_index("subset")
    s2_b = d["s2_unfilt"][~d["s2_unfilt"]["is_duplicate_sha256"]].groupby("batch")[["drop_2_to_1hz", "drop_30_to_40hz", "line_50hz_excess"]].mean()
    feq_230 = d["fooof_eq"][(d["fooof_eq"]["cohort"] == "All_7Cohorts_Pooled") & (d["fooof_eq"]["fit_window_hz"] == "2.0-30.0")].iloc[0]
    feq_140 = d["fooof_eq"][(d["fooof_eq"]["cohort"] == "All_7Cohorts_Pooled") & (d["fooof_eq"]["fit_window_hz"] == "1.0-40.0")].iloc[0]
    g2d = d["grid2d"]
    g_s2_45 = g2d[(g2d["psd_source"] == "Unfiltered_Raw_EDF") & (g2d["f_min_hz"] == 1.0) & (g2d["f_max_hz"] == 45.0)].iloc[0]
    g_s2_230 = g2d[(g2d["psd_source"] == "Unfiltered_Raw_EDF") & (g2d["f_min_hz"] == 2.0) & (g2d["f_max_hz"] == 30.0)].iloc[0]

    eb_sl = d["extra_bio"]["Sleep_global_theta"]
    eb_m_eo_dl = d["extra_bio"]["Mumtaz_EO_delta"]
    eb_m_eo_ch = d["extra_bio"]["Mumtaz_EO_chi"]
    m2_ec_dl = bio_exp[(bio_exp["cohort"] == "Study 2: Outpatient MDD") & (bio_exp["marker_key"] == "global_delta")].iloc[0]
    ad_pa = bio_exp[(bio_exp["cohort"] == "Study 5a: Alzheimer's Disease (AD)") & (bio_exp["marker_key"] == "posterior_alpha")].iloc[0]
    ad_gc = bio_exp[(bio_exp["cohort"] == "Study 5a: Alzheimer's Disease (AD)") & (bio_exp["marker_key"] == "global_chi")].iloc[0]

    b_s2_w = boot_d[(boot_d["cohort"] == "Study 2: Outpatient MDD") & (boot_d["marker"] == "frontal_theta_5ch") & (boot_d["comparison"] == "SP_1_40_minus_SP_2_30")].iloc[0]
    b_s3_o = boot_d[(boot_d["cohort"] == "Study 3: Univ. Depression Screening") & (boot_d["marker"] == "frontal_theta_5ch") & (boot_d["comparison"] == "OLS_1_40_minus_SP_2_30")].iloc[0]
    sim_s2_ols = sim_mc[(sim_mc["scenario"] == "S2_TrueNullTheta_BroadAlphaAndSlowCurve") & (sim_mc["estimator"] == "Masked_OLS_1to40")].iloc[0]
    sim_s2_230 = sim_mc[(sim_mc["scenario"] == "S2_TrueNullTheta_BroadAlphaAndSlowCurve") & (sim_mc["estimator"] == "specparam_2to30")].iloc[0]
    sim_s3_140 = sim_mc[(sim_mc["scenario"] == "S3_FilterEdgeAttenuation_Study2Replica") & (sim_mc["estimator"] == "specparam_1to40")].iloc[0]
    sim_s3_230 = sim_mc[(sim_mc["scenario"] == "S3_FilterEdgeAttenuation_Study2Replica") & (sim_mc["estimator"] == "specparam_2to30")].iloc[0]

    abstract_text = (
        f"Reproducible clinical electroencephalography (EEG) neuroinformatics requires an open reference suite to verify whether spectral biomarkers, Phase Lag Index (PLI) connectomes, and graph neural networks (GNNs) survive methodological auditing. "
        f"I present a harmonized Python benchmarking suite spanning seven public 19-channel datasets ({acc['total_unique_participants']} unique primary participants; {acc['total_analyzed_sessions']} primary sessions; {acc['total_with_mumtaz_eo']} total sessions; {acc['total_spectra']:,} channel spectra) across sleep deprivation, depression, Parkinson's disease (PD), and dementia. "
        f"Grid audits (105 windows; zero failures; `fooof` equivalence r >= 0.992) showed interior `[2.0, 30.0] Hz` `specparam` fits resolved upstream roll-off distortion in Study 2 outpatient MDD (median R² {fit140.loc['mumtaz_ec', 'r2_median']:.3f} vs. {fit230.loc['mumtaz_ec', 'r2_median']:.3f}) while avoiding `[7, 14] Hz`-masked OLS leakage in Study 3 (d = {ft_w.loc['Study 3: Univ. Depression Screening', 'cohens_d_masked_ols_1to40']:+.3f} vs. {ft_w.loc['Study 3: Univ. Depression Screening', 'cohens_d_specparam_resid_2to30']:+.3f}). "
        f"Raw EDF forensics in Study 2 exposed duplicate subjects (`29 MDD vs. 27 HC` unique), `50 Hz` Jan-2013 control artifacts (`d = +3.347` vs. 2011 controls), and Apr–Jul 2013 pre-export skirts confounding diagnosis with batch date; excluding Jan-2013 controls eliminated MDD frontal theta differences (`d = -0.082, p = 0.752`), whereas PD/dementia theta slowing replicated reliably (`k = 4 d_RE = {ndg_m['pooled_d']:+.3f}, p_HKSJ = {ndg_m['p_hksj']:.4f}`). "
        f"Across eight non-nested contrasts under 10×5 participant-grouped validation, `GCN-OMST` (`{rep_nn8['GCN_OMST'].mean():.3f}`) was equivalent within ±0.02 AUC to edge-matched `GCN-Random` (`{rep_nn8['GCN_Random_EdgeMatched'].mean():.3f}`, TOST p = {wil8.loc['GCN_Random_EdgeMatched', 'tost_p_002']:.4f}), `GCN-Weighted-PLI` (`{rep_nn8['GCN_Weighted_PLI'].mean():.3f}`), and regularized classical or `Riemannian Spatial Covariance` baselines (`{rep_nn8['Linear_SVM'].mean():.3f}`–`{riem_nn8.loc[('Riemannian_Spatial_Cov_190D', 'RF')]:.3f}`), while `{pli_s['pct_indefinite_raw']:.2f}%` of PLI matrices violated positive semi-definiteness, session mixing skewed AUC bidirectionally, and patient-only severity regressions explained `<= 3.5%` OOF variance (`OOF R² <= {reg[reg['cohort_key'].str.contains('Patients')]['r2_mean'].max():.3f}`, dementia `perm p >= {reg[reg['cohort_key'] == 'Dementia_MMSE_Patients_Only_N59']['perm_p'].min():.3f}`)."
    )
    abs_wc = len(abstract_text.split())
    assert 150 <= abs_wc <= 250, f"Abstract word count {abs_wc} outside [150, 250]!"
    print(f"   ✓ Abstract word count: {abs_wc} words (<= 250 limit)")

    refs_md = "\n\n".join(entry for _, entry in REFS_ALPHA)

    md = f"""# {EXACT_TITLE}

**Running Title:** {SHORT_TITLE}

**Author:** {AUTHOR_NAME}^1,2*^

^1^ {AFFIL_1}  
^2^ {AFFIL_2}  
^*^ Correspondence: {CORRESPONDENCE}

---

## Abstract

{abstract_text}

**Keywords:** Neuroinformatics Benchmark; Quantitative EEG; Spectral Parameterization; Graph Neural Networks; Dataset Forensics; Reproducibility Auditing

---

## 1. Introduction

Open computational neuroinformatics depends on standardized software benchmarks that test whether proposed quantitative electroencephalography (qEEG) biomarkers and graph architectures generalize across independent clinical hardware and centers (Nunez & Srinivasan, 2006; Kapur et al., 2012; Lopes da Silva, 2013; Cohen, 2014; Roy et al., 2019; Gemein et al., 2020). Across sleep regulation, affective illness, and neurodegeneration (Buzsáki & Draguhn, 2004; Buzsáki, 2006; Fries, 2005; Stam, 2014), automated deep network pipelines—such as convolutional recurrent models trained on effective connectivity (Saeedi et al., 2021) or band-power asymmetry features (Mahato & Paul, 2020)—frequently report high single-site diagnostic accuracy, yet fail to sustain performance under independent center transfer (Scheinost et al., 2019; Varoquaux, 2018; Brookshire et al., 2024). Key threats to multi-cohort reproducibility include trial- or session-level cross-validation leakage across repeated participant recordings (Saeb et al., 2017; Little et al., 2017; Brookshire et al., 2024), undocumented upstream hardware prefiltering and batch-date clustering, and unstandardized normative calibration across amplifiers (Wolfers et al., 2018; Rutherford et al., 2022).

A foundational signal-processing prerequisite in multi-center qEEG benchmarking is cleanly separating periodic oscillatory peaks from the broadband `1/f^chi` aperiodic background (Voytek et al., 2015; Cole & Voytek, 2017; Gao et al., 2017; Colombo et al., 2019; Donoghue et al., 2020). Because the aperiodic exponent `chi` shifts with age, arousal, and cortical excitation–inhibition ratio, conventional fixed-band power ratios conflate broadband slope shifts with true rhythmic oscillations (Donoghue et al., 2020; Donoghue et al., 2022; Gerster et al., 2022). That confounding has clouded transdiagnostic resting frontal theta (4–8 Hz) interpretation (Mitchell et al., 2008; Cavanagh & Frank, 2014): theta increases reliably during homeostatic sleep pressure (Finelli et al., 2000) and in Parkinson's disease (PD) and Alzheimer's disease (AD) slowing (Bosboom et al., 2006; Stoffers et al., 2007; Stam, 2010), whereas major depressive disorder (MDD) literature reports heterogeneous or conflicting shifts (Knott et al., 2001; Jaworska et al., 2012; Olbrich & Arns, 2013; Newson & Thiagarajan, 2019). Furthermore, spectral decomposition introduces estimator- and window-specific failure modes (Donoghue et al., 2022; Gerster et al., 2022): naive `[7, 14] Hz`-masked log-linear ordinary least-squares (`OLS`) regressions leave 4–7 Hz theta power inside the linear fit interval, while extending `specparam` fits to filter passband edges (`1.0` or `40.0 Hz`) exposes linear `1/f` background estimates to analog or digital filter skirts.

Parallel methodological ambiguity surrounds network and manifold learning on routine 19-channel clinical 10–20 EEG arrays (Bullmore & Sporns, 2009; Rubinov & Sporns, 2010). Phase-based metrics like the Phase Lag Index (`PLI`) reduce zero-lag volume conduction (Stam et al., 2007; Vinck et al., 2011), and Orthogonal Minimum Spanning Trees (`OMST`) extract parameter-free connected backbones via Global Cost Efficiency (`GCE`) (Dimitriadis et al., 2017a; Dimitriadis et al., 2017b). While Riemannian tangent-space geometry operates natively on symmetric positive-definite (`SPD`) sample spatial covariance matrices (Barachant et al., 2013; Congedo et al., 2017) and Graph Convolutional Networks (`GCNs`) (Kipf & Welling, 2017) promise relational inductive bias over sensor connectomes, two empirical prerequisites remain under-tested across multi-cohort 19-channel clinical EEG: whether `PLI` matrices violate positive semi-definiteness (`PSD`) on real short-window segments, and whether binary or PLI-weighted OMST topology actually outperforms degree-agnostic random graphs, isolated nodes, `SPD` spatial covariances, or simple tuned linear and tree baselines when evaluated strictly with participant-level grouping.

To provide a transparent, reusable software resource for auditing these exact neuroinformatics failure modes, I constructed and released an open 19-channel Python benchmarking suite spanning seven public archives (`564` unique primary participants after de-duplication, `675` primary sessions, `733` total sessions including secondary eyes-open recordings, and `13,927` unique channel spectra; Figure S5 in Online Resource 1). Leveraging `MNE-Python` (Gramfort et al., 2013), `specparam v2.0.0rc7` (Donoghue et al., 2020) cross-verified against legacy `fooof v1.1.1`, 2D fit-window grid searches, raw file SHA-256 and unfiltered-PSD forensics, 10×5 participant-grouped cross-validation with fold-nested normative z-scoring (Rutherford et al., 2022), two one-sided equivalence tests (`TOST`), aligned 1,000-permutation retraining tests, and disjoint `Split-Target-HC` external evaluation, I addressed three empirical questions:

1. Question 1 (Spectral Fit-Window Stability and Archive Forensics): Across 105 fit-window combinations (`[1.0–3.0] × [25.0–45.0] Hz`) and legacy `[7, 14] Hz`-masked OLS, how severely do filter-skirt roll-off, estimator leakage, epoch-duration imbalances, pure rest vs. task contamination, duplicate EDF files, and batch-date prefiltering distort apparent case–control spectral findings?
2. Question 2 (GNN Equivalence Testing, Connectome Definiteness, and Partition Leakage): Under strict 10×5 participant-grouped cross-validation, `TOST` equivalence bounds (`±0.02` AUC), and aligned permutation tests, does static `NormativeGCN` on binary or continuous PLI-weighted OMST graphs beat edge-matched random topology, isolated nodes, regularized linear/tree models, or empirical `19 × 19` sample spatial covariance Riemannian tangent-space baselines, and how do PLI non-PSD violations and session-mixing leakage bias evaluation?
3. Question 3 (External Transfer Boundaries, Gain Calibration, and Clinical Severity Floor): Under leak-free disjoint `Split-Target-HC` transfer without retraining, how do cross-center PD and cross-disorder predictions perform, how does `specparam` gain invariance interact with small-sample target calibration variance, and does case–control separability survive once healthy ceiling controls are removed in within-patient severity regression?

---

## 2. Materials and Methods

### 2.1 Public Datasets, Sub-Contrast Taxonomy, and Archive Accounting
This benchmarking resource aggregates seven de-identified, publicly downloadable resting-state EEG archives from OpenNeuro and Figshare processed under EEGLAB/BIDS and MNE conventions (Delorme & Makeig, 2004; Gramfort et al., 2013; study hardware and demographics appear in Online Resource 1, Supplementary Table S3, and software versions in Supplementary Table S9). Primary analyses respected each archive's experimental structure via explicit sub-contrasts (`2b`, `4a–b`, `5a–c`, `6a–c`). Across all seven releases, `574` nominal participants were enrolled (`685` primary sessions; `745` total including Study 2 eyes-open). Pre-analysis QC excluded `8` participants (`6` Study 2 EC sessions with >20% artifact rejection or channel bridging; `2` Study 3 participants with borderline BDI-II scores `10–13`), yielding `566` nominal participants (`677` primary sessions; `737` total). Raw-file SHA-256 forensics in Study 2 further removed `2` exact byte-identical duplicate pairs (`H S27` = `H S30` and `MDD S33` = `MDD S34`) from all primary analyses, yielding a canonical de-duplicated archive of `N = 564` unique primary participants, `675` primary sessions, and `733` total sessions (`13,927` spectra; Online Resource 1, Supplementary Figure S5; nominal `566/677/737` retained strictly in sensitivity audits):

1. Study 1 (Sleep Deprivation, OpenNeuro `ds004902`, `N = 71` participants, `142` paired sessions) (Xiang et al., 2024): Young adults (`20.0 ± 1.4` years, `47.9%` female) recorded eyes-open after normal sleep (`NS`) and after `36 h` total sleep deprivation (`SD`).
2. Study 2 (Outpatient MDD, Figshare `4244171`, de-duplicated `N = 56` primary EC `Study 2a` [`29 MDD vs. 27 HC`]; `N = 58` secondary EO `Study 2b` [`31 MDD vs. 27 HC`]; nominal pre-dedup `58 EC / 60 EO`) (Mumtaz et al., 2017): DSM-IV outpatients and healthy controls recorded on a `19`-channel clinical linked-ear (`-LE`) system (`256 Hz`); individual ages were omitted from the public release. Beyond exact duplicate files (`mumtaz_hc_030 == mumtaz_hc_027`; `mumtaz_mdd_034 == mumtaz_mdd_033`), acquisition-date forensics identified structural diagnosis–batch confounding (`all 29` MDD recorded in Apr–Jul 2013 vs. `27` HC split across `2011 [n=9]`, `Jan 12–14, 2013 [n=14]`, and `Apr–Jul 2013 [n=4]`; Online Resource 1, Supplementary Table S26).
3. Study 3 (University Depression Screening, OpenNeuro `ds003478`, `N = 119` participants) (Cavanagh et al., 2019): Undergraduates stratified by BDI-II `>= 14` (`N = 44`) versus `<= 7` (`N = 75`) during eyes-open rest (`SCID`: `11` current MDD, `11` past MDD, `8` non-MDD, `14` uninterviewed high-BDI).
4. Study 4 (UCSD PD ON/OFF, OpenNeuro `ds002778`, `N = 31` participants, `46` sessions) (George et al., 2013; Rockhill et al., 2020): Fifteen PD patients recorded OFF (`>= 12 h` withdrawal, `Study 4a`) and ON dopaminergic therapy (`Study 4b`) plus `16` HC (eyes-open). Biophysical contrasts average PD ON/OFF within participant (`Study 4 SubjMean`, `15 vs. 16`); 10×5 CV evaluates all `46` sessions strictly under participant-level grouping.
5. Study 5 (AHEPA Clinical Dementia, OpenNeuro `ds004504`, `N = 88` participants) (Miltiadous et al., 2023): Eyes-closed 19-channel EEG across AD (`Study 5a`, `36 vs. 29`), FTD (`Study 5b`, `23 vs. 29`), and combined dementia (`Study 5c`, `59 vs. 29`) with MMSE scores. Because `5c` shares controls and patients with `5a` and `5b`, cross-model Wilcoxon and `TOST` benchmarks report both `k = 8` strictly non-nested contrasts (excluding `5c`) and `k = 9` primary contrasts.
6. Study 6 (UNM PD Washout, OpenNeuro `ds003490`, `N = 50` participants, `75` sessions) (Cavanagh et al., 2018): Twenty-five PD patients recorded OFF (`>= 15 h` withdrawal, `Study 6a`) and ON medication (`Study 6b`; combined `Study 6c`, `75` sessions) plus `25` CTL. Because raw `.set` files concatenate resting blocks with a 3-stimulus auditory oddball task, Study 6 was rebuilt cleanly from the initial contiguous `~60 s` pure eyes-open rest block (`10.46 ± 3.27` vs. `9.60 ± 3.06` clean epochs, `p = 0.267`), alongside a `74`-session pure eyes-closed sensitivity analysis (excluding `sub-005`, lacking `>= 2` EC epochs; Online Resource 1, Supplementary Table S25).
7. Study 7 (Iowa PD Multi-Scale, OpenNeuro `ds004584`, `N = 149` participants) (Singh et al., 2023; Anjum et al., 2024): Two-minute eyes-open rest from `100` PD patients ON medication and `49` HC with MoCA and UPDRS-III scores.

### 2.2 Harmonization, Unfiltered Raw EDF Audits, and Epoch Balance
Multichannel EEG arrays were mapped to the canonical `19`-channel 10–20 montage (`Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, P7, P3, Pz, P4, P8, O1, O2`; Online Resource 1, Supplementary Table S5), inserting a zero-initialized `Pz` reference in Study 7 before Common Average Referencing (`CAR`), zero-phase FIR bandpass filtering (`1.0–40.0 Hz`, Hamming window), resampling to `250 Hz`, and `4.0 s` epoching (`150 μV` peak-to-peak rejection; `200 μV` fallback in Study 7; Online Resource 1, Supplementary Table S8). Clean epoch counts differed modestly across groups in Studies 1 (`p = 0.0103`), 2a (`p = 0.0040`), and 5 (`p = 0.0006`) and were balanced in Studies 2b, 3, 4, 6 EO (`p = 0.267`), and 7; joint age and clean-epoch ANCOVAs verified robustness (Online Resource 1, Supplementary Table S25). To isolate upstream acquisition prefiltering from pipeline FIR filtering in Study 2, raw unfiltered EDF headers (`sfreq`, `highpass`, `lowpass`, `prefiltering`, `meas_date`, `SHA-256`) and unfiltered `0.5–80 Hz` Welch spectra (`0.50 Hz` resolution, clean epoch mask) were also extracted across all `58` EC files.

### 2.3 Interior [2.0, 30.0] Hz Parameterization, 2D Fit-Window Grid, and Simulation Calibration
Welch PSDs (`2.0 s` Hamming windows, `50%` overlap, `0.50 Hz` step) were parameterized across all `13,927` unique spectra via `SpectralGroupModel` in `specparam v2.0.0rc7` (Donoghue et al., 2020) (`aperiodic_mode='fixed'`, `peak_width_limits=[1.0, 12.0]` Hz, `max_n_peaks=6`, `min_peak_height=0.05` B, `peak_threshold=2.0` SD; Online Resource 1, Supplementary Section S1.1 and Supplementary Figure S1). To evaluate passband-edge sensitivity without committing solely to two boundary choices, a 105-configuration 2D grid crossed `f_min in [1.0, 1.5, 2.0, 2.5, 3.0] Hz` against `f_max in [25.0, 30.0, 35.0, 40.0] Hz` across Studies 2 (`N = 56`), 3 (`N = 119`), 5c (`N = 88`), and 7 (`N = 149`), plus `f_max in [25.0..45.0] Hz` on Study 2 unfiltered raw EDF spectra (Online Resource 1, Supplementary Table S27). Empirical cross-version equivalence was also verified across `665` shared spectra (`5 sessions × 7 cohorts × 19 channels`) against legacy `fooof v1.1.1`. Primary parameterization fitted the clean interior `[2.0, 30.0] Hz` interval:

$$\\log_{{10}} P(f) = (b - \\chi \\log_{{10}} f) + \\sum_{{k=1}}^K G_k(f), \\quad f \\in [2.0, 30.0]\\text{{ Hz}} \\qquad (1)$$

Four nodal features were computed: band-averaged flattened residuals (`log10 PSD - (b - chi * log10 f)`) over theta (`4.0–8.0 Hz`), low-frequency periodic residual (`2.0–4.0 Hz`), and alpha (`8.0–13.0 Hz`), plus aperiodic exponent `chi`, summarized across five frontal sites (`Fp1, Fp2, F3, F4, Fz`), three dorsal frontal sites (`F3, Fz, F4`), five posterior sites (`P3, P4, Pz, O1, O2`), or all `19` sites, alongside posterior `IAPF` (`7.0–13.5 Hz`). Every `[2.0, 30.0] Hz` contrast was compared against `[1.0, 40.0] Hz` `specparam`, `[7, 14] Hz`-masked OLS (`1–40` and `2–30 Hz`), and discrete summed Gaussian peaks (`G_k`, `specparam_peaks_2to30`), using 1,000 stratified participant-level bootstraps (`Delta d`).

Three calibrated Monte Carlo simulations (`150` trials × `N = 29 vs. 27`, calibrated to empirical human resting-EEG dispersion: `SD(chi) = 0.16`, `SD(theta) = 0.12`, `SD(alpha) = 0.20`, `SD(b) = 0.30`, spectral noise `0.035 B`) verified bias directionality under: (S1) true null periodic theta with an isolated exponent drop (`Delta chi = -0.15`), (S2) true null theta with alpha reduction (`-0.22 B`) and slope flattening (`-0.12`), and (S3) negative periodic theta (`-0.14 B`) coupled with `< 2 Hz` and `> 30 Hz` edge roll-off mimicking Study 2. Multiplicative voltage gain invariance (`gamma in [0.01, 100.0]`, `285` spectra; Online Resource 1, Supplementary Section S1.3, Supplementary Table S2, Supplementary Figure S3) was verified solely as a baseline arithmetic check.

### 2.4 Connectomics (`PLI`, `OMST`), Positive Semi-Definiteness, and Spatial Covariance Riemannian Baselines
Pairwise `4.0–13.0 Hz` `PLI` matrices (Stam et al., 2007; Vinck et al., 2011) were mapped to edge costs `c_ij = (1.0 - PLI_ij) + 1e-4` and pruned via `OMST` maximizing `GCE` (Dimitriadis et al., 2017a; Dimitriadis et al., 2017b; Online Resource 1, Supplementary Section S1.2, Supplementary Figure S2). Symmetrized matrices `C = I_19 + PLI` were audited for negative eigenvalues across all `675` primary sessions. In addition to comparing Euclidean 171-D upper-triangular PLI vectors and two SPD-conditioned PLI Log-Euclidean tangent representations (`diagonal loading >= 0.01`; `eigen-clipped >= 10^-4`), I extracted empirical `19 × 19` sample spatial covariance matrices `Sigma = (1/T) X X^T + 10^-5 (tr(Sigma)/19) I_19` across all `675` sessions (`1.0–40.0 Hz`, `250 Hz` clean epochs) and evaluated exact `190-D` Log-Euclidean spatial covariance Riemannian tangent features (`Riemannian_Spatial_Cov_190D`) (Barachant et al., 2013; Congedo et al., 2017) under identical `10 × 5` grouped CV.

### 2.5 True 10×5 Participant-Grouped CV, 13-Model Comparison, `TOST`, and Permutations
Because deterministic ties occurred in `sklearn` `StratifiedGroupKFold` on Study 1's balanced crossover, an explicit stratified participant-cluster splitter shuffled unique participants across `10 repeats × 5 folds` (`50` outer folds; `0.0%` participant overlap). Healthy-control normative z-scoring (`z_ij = (x_ij - mu_HC_train) / sd_HC_train`) (Rutherford et al., 2022) and inner 3-fold `GridSearchCV` hyperparameter tuning executed strictly inside each outer training split. Thirteen configurations were benchmarked across `12` contrasts (Online Resource 1, Supplementary Tables S7, S7b, S10, S20): the 3,362-parameter `NormativeGCN` with binary OMST adjacency (`GCN-OMST`; `A + I`, `4->32->32` layers, mean+max readout, `50` Adam epochs; Online Resource 1, Supplementary Section S1.4), continuous PLI-weighted OMST edges (`GCN-Weighted-PLI`, `A_bin ⊙ PLI`), `GCN-Identity` (`A = I`), `GCN-Random` (session edge-matched unweighted random graphs drawn per repeat), `GCN-Full` (dense off-diagonal ones), six 76-D classical models (`Tuned Logistic Regression`, `Tuned Ridge`, `Linear SVM` and `RBF SVM` scored via `1 / (1 + exp(-decision_function))` to eliminate small-sample Platt sign flips, 100-tree `Random Forest` depth 5, `GBDT` depth 4), and univariate `Single Theta` / `Single Chi`. Cross-model comparisons across `k = 8` non-nested contrasts (and `k = 9` / `k = 11` sets) reported paired `Delta AUC` 95% t-intervals, Wilcoxon signed-rank tests with BH-FDR, and paired two one-sided equivalence tests (`TOST` bounds `±0.02` and `±0.03` AUC; Online Resource 1, Supplementary Table S10b).

Aligned 1,000-permutation retraining tests (exact `GridSearchCV` `Tuned LR` and participant-cluster splits; Online Resource 1, Supplementary Figure S6) audited four within-cohort and two `Split-Target-HC` benchmarks, while 10×5 session-mixing leakage ablations (`Strict Leak-Free`, `Leaky Pre-Standardization`, `Leaky Subject Mixing`) quantified validation inflation and deflation in Studies 1, 4, and 6c.

### 2.6 Disjoint Transfer, Gain Stress-Testing, and 1,000-Permutation Clinical Regressions
Six directed PD cross-site pairs and eight cross-disorder pairs used `20` random 50/50 `Split-Target-HC` splits estimating target HC statistics strictly on one disjoint control half (Oracle full-HC references in Online Resource 1, Supplementary Tables S15, S16, S19). Uniform (`k = 0.5, 2.0, 5.0`) and mixed (`50% 0.5x / 50% 5.0x`) voltage scaling tested calibration sensitivity. For MMSE (Study 5) and MoCA/UPDRS-III (Study 7), 10×5 OOF regressions (`GBDT`, `Random Forest`, `RBF SVR`, `Ridge`) were paired with exact 1,000-label-permutation K-fold retraining (`perm_p = (count(r_perm >= r_Split0) + 1) / 1001`, reported alongside `perm_p_two_sided`). Statistical inference used two-sided Welch or paired t-tests (`q_10` across 10 theta contrasts, `q_50` across 50 marker comparisons) and DerSimonian–Laird / variance-floored `HKSJ` random-effects pooling (`k = 4` PD/dementia core; `K = 6` `N = 493`; `K = 5` `N = 437` omitting Study 2; Online Resource 1, Supplementary Table S22).

---

## 3. Results

### 3.1 Passband-Interior [2.0, 30.0] Hz Parameterization, 2D Grid Sensitivity, and Study 2 Forensics (RQ1)
Interior `[2.0, 30.0] Hz` `specparam v2.0.0rc7` achieved zero failed fits across all `13,927` unique spectra (median `R² = {fit230['r2_median'].min():.3f}–{fit230['r2_median'].max():.3f}`, `MAE = {fit230['mae_median'].min():.3f}–{fit230['mae_median'].max():.3f} B`; Table 1, Online Resource 1, Supplementary Table S23 and Supplementary Figure S4) and matched legacy `fooof v1.1.1` across `665` tested spectra (`[2.0, 30.0] Hz`: `r_chi = {feq_230['pearson_r_chi']:.4f}`, `mean |Δχ| = {feq_230['mean_abs_diff_chi']:.4f}`, `r_theta = {feq_230['pearson_r_theta_resid']:.4f}`, `mean |Δθ| = {feq_230['mean_abs_diff_theta_resid']:.5f} B`; `[1.0, 40.0] Hz`: `r_chi = {feq_140['pearson_r_chi']:.4f}`, `r_theta = {feq_140['pearson_r_theta_resid']:.4f}`). Extending fit bounds into `[1.0, 40.0] Hz` selectively degraded Study 2 MDD EC fits (`median R² = {fit140.loc['mumtaz_ec', 'r2_median']:.3f}`, `{fit140.loc['mumtaz_ec', 'pct_r2_lt_080']:.1f}% < 0.80` vs. `{fit230.loc['mumtaz_ec', 'r2_median']:.3f}` and `{fit230.loc['mumtaz_ec', 'pct_r2_lt_080']:.1f}%` at `[2.0, 30.0] Hz`), severely masking nominal theta (`d = {ft_w.loc['Study 2: Outpatient MDD', 'cohens_d_specparam_resid_1to40']:+.3f}` vs. `{ft_w.loc['Study 2: Outpatient MDD', 'cohens_d_specparam_resid_2to30']:+.3f}`; bootstrap Δd = `{b_s2_w['delta_d']:+.3f} [{b_s2_w['ci_lower']:+.3f}, {b_s2_w['ci_upper']:+.3f}], p = {fmt_p(b_s2_w['boot_p'])}`) and inverting global `χ` (`{chi_w.loc['Study 2: Outpatient MDD', 'cohens_d_specparam_resid_1to40']:+.3f}` vs. `{chi_w.loc['Study 2: Outpatient MDD', 'cohens_d_specparam_resid_2to30']:+.3f}`; Table 2, Figure 2A). Across the full 105-configuration 2D grid (`0` failed fits everywhere; Online Resource 1, Supplementary Table S27), Studies 3 (`theta d = -0.388` to `-0.571`), 5c (`+0.923` to `+1.105`), and 7 (`+0.807` to `+0.897`) remained invariant across boundaries, whereas in Study 2 increasing `f_max` from `30 -> 35 -> 40 -> 45 Hz` even on unfiltered raw EDF PSDs inverted `χ d` monotonically (`-0.945 -> -0.273 -> +1.061 -> +1.803`) and flipped `[1.0, 45.0] Hz` theta from `{g_s2_230['theta_d']:+.3f}` to `{g_s2_45['theta_d']:+.3f}` (`chi d = {g_s2_45['chi_d']:+.3f}`). Conversely, in clean BioSemi Study 3 spectra (`R² = {fit230.loc['cavanagh_ds003478', 'r2_median']:.3f}`), legacy `[7, 14] Hz`-masked OLS inflated depression theta negativity (`{ft_w.loc['Study 3: Univ. Depression Screening', 'cohens_d_masked_ols_1to40']:+.3f}` vs. `{ft_w.loc['Study 3: Univ. Depression Screening', 'cohens_d_specparam_resid_2to30']:+.3f}`; bootstrap Δd = `{b_s3_o['delta_d']:+.3f} [{b_s3_o['ci_lower']:+.3f}, {b_s3_o['ci_upper']:+.3f}], p = {fmt_p(b_s3_o['boot_p'])}`).

![](Figure_1.png){{width=6.5in}}
**Figure 1. Canonical Interior [2.0, 30.0] Hz Parameterized Biophysical Landscape (`specparam v2.0.0rc7`).** Standardized effect sizes (Cohen's d, 95% CIs) across ten contrasts on de-duplicated data (`564` unique primary participants) for (A) 5-channel frontal periodic theta (4–8 Hz), (B) 19-channel low-frequency periodic residual (2–4 Hz), (C) 5-channel posterior periodic alpha (8–13 Hz), and (D) 19-channel global aperiodic exponent chi. Neurodegenerative PD and dementia show positive frontal theta slowing, whereas Study 2's nominal negative shift is batch-confounded.

Calibrated Monte Carlo simulations (`150` trials × `N = 29 vs. 27`; Figure 2B and Online Resource 1, Supplementary Table S11) confirmed both mechanisms without exaggerated effect dispersion: under true null theta with broadband alpha attenuation and exponent flattening (Scenario S2), `[7, 14] Hz`-masked OLS injected spurious negative theta bias (`{sim_s2_ols['theta_d_bias']:+.3f} d`), over 2.4-fold larger than interior `[2.0, 30.0] Hz` `specparam` (`{sim_s2_230['theta_d_bias']:+.3f} d`; exponent bias `{sim_s2_230['chi_d_bias']:+.3f}` vs. `{sim_s2_ols['chi_d_bias']:+.3f}`); under Study 2-like edge roll-off (Scenario S3), `[1.0, 40.0] Hz` `specparam` degraded `R²` (`{sim_s3_140['median_r2']:.3f}`) and flipped theta from true `{sim_s3_140['true_theta_d']:+.3f}` to `{sim_s3_140['est_theta_d_mean']:+.3f}` (`bias = {sim_s3_140['theta_d_bias']:+.3f} d`) while wiping out `chi` (`est {sim_s3_140['est_chi_d_mean']:+.3f}`), whereas `[2.0, 30.0] Hz` recovered true theta (`{sim_s3_230['est_theta_d_mean']:+.3f}`, bias `{sim_s3_230['theta_d_bias']:+.3f}`) and `chi` (`{sim_s3_230['est_chi_d_mean']:+.3f}`).

Raw archive forensics on Figshare `4244171` (Study 2; Figure 2C and Online Resource 1, Supplementary Table S26) established that diagnostic status and upstream acquisition/prefiltering batches are structurally confounded. SHA-256 hashing uncovered exact byte-identical duplicates across nominal participants (`H S27` = `H S30`, `MDD S33` = `MDD S34`, plus duplicate `H S15 EO` and `10` identical HC task files), yielding `N = 56` unique EC (`29 MDD vs. 27 HC`, `d_theta = {s2_for.loc['Study 2a EC: Deduplicated Unique (29 MDD vs 27 HC, N=56)', 'theta_d']:+.3f}`). Specifically, while every EDF header declared identical `HP:0.5 Hz LP:80 Hz` (`256 Hz`) filter metadata, all `29` unique MDD recordings came from Apr–Jul 2013 alongside only `4` HCs, whereas `23` HCs came from Jul–Aug 2011 (`n = 9`) and Jan 12–14, 2013 (`n = 14`). Raw unfiltered `0.5–80 Hz` spectra proved that BOTH Apr–Jul 2013 batches—MDD (`1 Hz - 2 Hz = {s2_b.loc['MDD_2013_04_07', 'drop_2_to_1hz']:+.3f} B`; `40 Hz - 30 Hz = {s2_b.loc['MDD_2013_04_07', 'drop_30_to_40hz']:+.3f} B`) AND the `4` Apr–Jul 2013 HCs (`{s2_b.loc['HC_2013_04_07', 'drop_2_to_1hz']:+.3f} B`; `{s2_b.loc['HC_2013_04_07', 'drop_30_to_40hz']:+.3f} B`)—received identical upstream `< 2 Hz` and `> 30 Hz` pre-export attenuation that was absent in `2011` (`{s2_b.loc['HC_2011', 'drop_2_to_1hz']:+.3f} B`; `{s2_b.loc['HC_2011', 'drop_30_to_40hz']:+.3f} B`) and `Jan-2013` (`{s2_b.loc['HC_2013_01', 'drop_2_to_1hz']:+.3f} B`, unnotched `50 Hz` spike `+{s2_b.loc['HC_2013_01', 'line_50hz_excess']:.2f} B`, `±π/2` phase artifacts). Consequently, `Jan-2013` vs. `2011` HCs alone diverged by `d_theta = +3.347` (`p < 1e-4`), and removing `Jan-2013` HCs eliminated Study 2 MDD vs. HC periodic theta (`d = {s2_for.loc['Study 2a EC: Deduplicated Excl. Jan-2013 HC Batch (29 MDD vs 13 HC, N=42)', 'theta_d']:+.3f}`, 95% CI `[{s2_for.loc['Study 2a EC: Deduplicated Excl. Jan-2013 HC Batch (29 MDD vs 13 HC, N=42)', 'theta_ci_lo']:+.3f}, {s2_for.loc['Study 2a EC: Deduplicated Excl. Jan-2013 HC Batch (29 MDD vs 13 HC, N=42)', 'theta_ci_hi']:+.3f}]`, `p = {s2_for.loc['Study 2a EC: Deduplicated Excl. Jan-2013 HC Batch (29 MDD vs 13 HC, N=42)', 'theta_p']:.4f}`; vs. 2011 HCs `d = {s2_for.loc['Study 2a EC: Deduplicated vs 2011 HC Batch Only (29 MDD vs 9 HC, N=38)', 'theta_d']:+.3f}`; vs. `4` date-matched Apr–Jul 2013 HCs `d = {s2_for.loc['Study 2a EC: Deduplicated vs Apr-Jul 2013 Date-Matched HC (29 MDD vs 4 HC, N=33)', 'theta_d']:+.3f}`; strict-PSD PLI `22 vs. 12` subset retaining `8` 2011 HCs `d = {s2_for.loc['Study 2a EC: Deduplicated + Strictly PSD PLI Both Groups (22 MDD vs 12 HC, N=34)', 'theta_d']:+.3f}, p = {s2_for.loc['Study 2a EC: Deduplicated + Strictly PSD PLI Both Groups (22 MDD vs 12 HC, N=34)', 'theta_p']:.4f}`), while batch-linked aperiodic `χ` shifts (`d = -1.48` to `-1.61`) still supported `RF AUC = {s2_for.loc['Study 2a EC: Deduplicated vs Apr-Jul 2013 Date-Matched HC (29 MDD vs 4 HC, N=33)', 'rf_10x5_mean']:.3f}–{s2_for.loc['Study 2a EC: Deduplicated vs 2011 HC Batch Only (29 MDD vs 9 HC, N=38)', 'rf_10x5_mean']:.3f}` even after dropping Jan-2013 HCs. Secondary EO recordings (`N = 58`) mirrored high delta (`d = {eb_m_eo_dl['cohens_d']:+.3f}`) and null `χ` (`p = {eb_m_eo_ch['p_value']:.4f}`).

{t1}

{t2}

Across remaining cohorts free of acquisition-batch phase corruption (Table 1 and Figure 1A), canonical `[2.0, 30.0] Hz` frontal periodic theta shifts replicated cleanly: positive elevation in acute sleep deprivation (`d_z = {ft_w.loc['Study 1: Acute Sleep Deprivation', 'cohens_d_specparam_resid_2to30']:+.3f}, p = {fmt_p(ft_w.loc['Study 1: Acute Sleep Deprivation', 'p_value_specparam_resid_2to30'])}`; global theta `d_z = {eb_sl['cohens_d']:+.3f}`), PD (`Iowa PD d = {ft_w.loc['Study 7: Iowa Parkinson Cohort', 'cohens_d_specparam_resid_2to30']:+.3f}, p = {fmt_p(ft_w.loc['Study 7: Iowa Parkinson Cohort', 'p_value_specparam_resid_2to30'])}`; `UNM PD-ON EO d = {ft_w.loc['Study 6b: UNM Parkinson ON', 'cohens_d_specparam_resid_2to30']:+.3f}`; `UNM PD-OFF EO d = {ft_w.loc['Study 6a: UNM Parkinson OFF', 'cohens_d_specparam_resid_2to30']:+.3f}`; `UCSD SubjMean d = {ft_w.loc['Study 4: UCSD Parkinson All', 'cohens_d_specparam_resid_2to30']:+.3f}`), and dementia (`AD d = {ft_w.loc["Study 5a: Alzheimer's Disease (AD)", 'cohens_d_specparam_resid_2to30']:+.3f}`; `FTD d = {ft_w.loc['Study 5b: Frontotemporal Dementia (FTD)', 'cohens_d_specparam_resid_2to30']:+.3f}`; `Combined d = {ft_w.loc['Study 5c: Clinical Dementia (All)', 'cohens_d_specparam_resid_2to30']:+.3f}`), whereas Study 3 student depression retained lower theta (`d = {ft_w.loc['Study 3: Univ. Depression Screening', 'cohens_d_specparam_resid_2to30']:+.3f}, p = {fmt_p(ft_w.loc['Study 3: Univ. Depression Screening', 'p_value_specparam_resid_2to30'])}`). AD additionally exhibited marked posterior alpha attenuation (`d = {ad_pa['cohens_d']:+.3f}`) and steeper slope (`d = {ad_gc['cohens_d']:+.3f}`). Effects persisted in dorsal `F3–Fz–F4` clusters (Table 1) and after joint chronological-age and clean-epoch ANCOVAs (`p <= 0.034` in Studies 3, 5c, 6b, and 7; clean epochs `p >= 0.072`; Online Resource 1, Supplementary Tables S17 and S25). In Study 6, pure-EC rest (`N = 74`) amplified PD theta separation (`OFF d = {unm_sens_df.loc['UNM Pure EC (N=25 PD-OFF vs 24 CTL)', 'theta_d']:+.3f}, p = {fmt_p(unm_sens_df.loc['UNM Pure EC (N=25 PD-OFF vs 24 CTL)', 'theta_p'])}`) over pure-EO (`{unm_sens_df.loc['UNM Pure EO (N=25 PD-OFF vs 25 CTL)', 'theta_d']:+.3f}`), whereas paired ON–OFF medication comparisons showed null frontal theta differences (`UNM p = {unm_med_ft['paired_p']:.4f}`; `UCSD p = {ucsd_med_ft['paired_p']:.4f}`; Online Resource 1, Supplementary Table S13) and posterior `IAPF` slowed in Iowa PD (`{iowa_iapf['diff_hz']:+.3f} Hz, d = {iowa_iapf['cohens_d']:+.3f}, p = {fmt_p(iowa_iapf['p_value'])}`; Online Resource 1, Supplementary Table S24). Meta-analysis established replicated neurodegenerative slowing across `k = 4` PD/dementia cohorts (`d_RE = {ndg_m['pooled_d']:+.3f}`, `HKSJ 95% CI [{ndg_m['hksj_ci_lower']:+.3f}, {ndg_m['hksj_ci_upper']:+.3f}], p_HKSJ = {ndg_m['p_hksj']:.4f}, I² = {ndg_m['I2_percent']:.1f}%`; `K = 6` chronic `N = 493` vs. `K = 5` `N = 437` omitting Study 2 in Table 1 and Online Resource 1, Supplementary Table S22).

![](Figure_2.png){{width=6.5in}}
**Figure 2. Spectral Estimator Confounding, Calibrated Simulations, and Study 2 Unfiltered EDF Forensics.** (A) Frontal periodic theta effects across `[7, 14] Hz`-masked OLS, `[1, 40] Hz` `specparam`, and canonical `[2, 30] Hz` `specparam`. (B) Empirical-variance-calibrated Monte Carlo bias (`150` trials/scenario) proving false OLS theta negativity (`-0.32 d` bias in S2) and `[1, 40] Hz` filter-skirt polarity inversion (`+2.98 d` bias in S3). (C) Unfiltered `0.5–65 Hz` raw EDF batch spectra (`mean ± SEM`) proving identical Apr–Jul 2013 pre-export roll-off across MDD (`n = 29`) and date-matched HCs (`n = 4`) versus unfiltered `2011` (`n = 9`) and `50 Hz`-contaminated `Jan-2013` HCs (`n = 14`).

### 3.2 GNN Topological Equivalence Under 10×5 Grouped CV and Connectome Audits (RQ2)
Across `675` primary OMST graphs (`{omst['total_node_instances']:,}` nodes; Online Resource 1, Supplementary Tables S1 and S4), mean spanning trees averaged `{omst['mean_trees']:.2f} ± {omst['sd_trees']:.2f}` (`{omst['mean_edges']:.2f} ± {omst['sd_edges']:.2f}` edges, `GCE = {omst['mean_gce']:.4f} ± {omst['sd_gce']:.4f}`, `0.0%` isolated nodes, zero significant case–control GCE shifts `q_50 >= 0.087`). Under true 10×5 participant-grouped validation across `k = 8` strictly non-nested contrasts (Table 3, Figure 3A, Online Resource 1, Supplementary Tables S7, S7b, S10, S10b, S20), `GCN-OMST` averaged `{rep_nn8['GCN_OMST'].mean():.3f}` (`{rep_p9['GCN_OMST'].mean():.3f}` across `k = 9`). Pairwise tests established formal statistical equivalence within `±0.02` AUC relative to continuous PLI-weighted `GCN-Weighted-PLI` (`{rep_nn8['GCN_Weighted_PLI'].mean():.3f}, TOST p = {wil8.loc['GCN_Weighted_PLI', 'tost_p_002']:.4f}`), edge-matched `GCN-Random` (`{rep_nn8['GCN_Random_EdgeMatched'].mean():.3f}, Delta AUC = {wil8.loc['GCN_Random_EdgeMatched', 'mean_diff_auc']:+.3f} [{wil8.loc['GCN_Random_EdgeMatched', 'ci95_diff_lo']:+.3f}, {wil8.loc['GCN_Random_EdgeMatched', 'ci95_diff_hi']:+.3f}], Wilcoxon p = {wil8.loc['GCN_Random_EdgeMatched', 'raw_p']:.4f}, TOST p = {wil8.loc['GCN_Random_EdgeMatched', 'tost_p_002']:.4f}`), and dense `GCN-Full` (`{rep_nn8['GCN_Full'].mean():.3f}, TOST p = {wil8.loc['GCN_Full', 'tost_p_002']:.4f}`), while `GCN-Identity` reached `{rep_nn8['GCN_Identity'].mean():.3f}`. Once Platt probability scaling was replaced by `decision_function` scoring, `Linear SVM` (`{rep_nn8['Linear_SVM'].mean():.3f}` for `k = 8`, `{rep_p9['Linear_SVM'].mean():.3f}` for `k = 9`) tied `Tuned LR` (`{rep_nn8['Tuned_LogisticRegression'].mean():.3f}`), `Random Forest` (`{rep_nn8['RandomForest'].mean():.3f}`), and `Tuned Ridge` (`{rep_nn8['Tuned_Ridge'].mean():.3f}`), while `RBF SVM` rose to `{rep_nn8['RBF_SVM'].mean():.3f}`; `GCN-OMST` outperformed only `GBDT` (`{rep_nn8['GBDT'].mean():.3f}`) and univariate `Single Chi` (`{rep_nn8['Single_Chi'].mean():.3f}`). Leave-one-feature-out ablations (Online Resource 1, Supplementary Table S6) and 5-cohort hyperparameter sweeps (`{hparam_means.min():.3f}–{hparam_means.max():.3f}`, Supplementary Table S7c) confirmed local spectral dependence.

{t3}

Aligned 1,000-permutation tests (Online Resource 1, Supplementary Figure S6) confirmed above-chance decoding for both `GCN-OMST` and `Tuned LR` in Study 5c dementia (`p = {perms['A']['p_perm_gcn']:.4f}` GCN; `{perms['A']['p_perm_lr']:.4f}` LR) and Study 7 Iowa PD (`GCN {perms['B']['obs_gcn_auc']:.3f}, p = {perms['B']['p_perm_gcn']:.4f}`; `LR {perms['B']['obs_lr_auc']:.3f}, p = {perms['B']['p_perm_lr']:.4f}`), whereas in Study 6a pure-EO PD-OFF `Tuned LR` exceeded chance (`{perms['C']['obs_lr_auc']:.3f}, p = {perms['C']['p_perm_lr']:.4f}`) while `GCN-OMST` fell inside the permutation null (`{perms['C']['obs_gcn_auc']:.3f}, p = {perms['C']['p_perm_gcn']:.4f}`). Under `Split-Target-HC` transfer, `Iowa PD -> UNM PD-OFF` fell within target control-split null variance (`p >= {perms['E']['p_perm_lr']:.4f}`), whereas `Sleep Debt -> Dementia` reached `p = {perms['F']['p_perm_gcn']:.4f}` (`GCN`) / `p = {perms['F']['p_perm_lr']:.4f}` (`LR`).

Three structural audits clarified why graph convolutions failed to add value (Figure 3B–C, Online Resource 1, Supplementary Tables S18 and S21). First, `{int(pli_s['n_indefinite_raw'])}/{int(pli_s['n_sessions'])}` (`{pli_s['pct_indefinite_raw']:.2f}%`) `19 × 19` symmetrized `I_19 + PLI` matrices were indefinite (`min λ = {pli_s['min_eigenvalue']:.4f}`, led by Study 2 MDD `{pli_a.loc['Study 2: Outpatient MDD (EC)', 'pct_indefinite_raw']:.1f}%` and Study 5 dementia `{pli_a.loc['Study 5: Clinical Dementia (ds004504)', 'pct_indefinite_raw']:.1f}%`). Conditioning PLI for Log-Euclidean tangent mapping yielded near-chance `8`-contrast means (`{riem_nn8.loc[('Riemannian_Higham_SPD_190D', 'LR')]:.3f}–{riem_nn8.loc[('Riemannian_Gershgorin_SPD_190D', 'RF')]:.3f}`), identical to Euclidean 171-D PLI (`{riem_nn8.loc[('Euclidean_UpperTri_171D', 'LR')]:.3f}`). In sharp contrast, direct SPD `19 × 19` sample spatial covariance Riemannian tangent representations (`Riemannian_Spatial_Cov_190D`) reached `{riem_nn8.loc[('Riemannian_Spatial_Cov_190D', 'RF')]:.3f}` (`RF`), `{riem_nn8.loc[('Riemannian_Spatial_Cov_190D', 'Ridge')]:.3f}` (`Ridge`), and `{riem_nn8.loc[('Riemannian_Spatial_Cov_190D', 'LR')]:.3f}` (`LR`), matching 76-D spectral baselines cleanly (`Supplementary Table S21`) and demonstrating that PLI discards broadband spatial covariance geometry. Second, 10×5 session-mixing leakage shifted AUC bidirectionally by label concordance (`Study 1 crossover RF {leak.loc[('Study 1: Sleep Deprivation', 'Leaky_Subject_Mixing', 'RandomForest'), 'mean_delta_vs_strict']:+.3f} [95% CI {leak.loc[('Study 1: Sleep Deprivation', 'Leaky_Subject_Mixing', 'RandomForest'), 'delta_ci_lo']:+.3f}, {leak.loc[('Study 1: Sleep Deprivation', 'Leaky_Subject_Mixing', 'RandomForest'), 'delta_ci_hi']:+.3f}]` vs. `Study 4 PD RF {leak.loc[('Study 4: UCSD Parkinson', 'Leaky_Subject_Mixing', 'RandomForest'), 'mean_delta_vs_strict']:+.3f} [{leak.loc[('Study 4: UCSD Parkinson', 'Leaky_Subject_Mixing', 'RandomForest'), 'delta_ci_lo']:+.3f}, {leak.loc[('Study 4: UCSD Parkinson', 'Leaky_Subject_Mixing', 'RandomForest'), 'delta_ci_hi']:+.3f}]`).

![](Figure_3.png){{width=6.5in}}
**Figure 3. 10×5 GNN Topological Equivalence, Bidirectional Session-Mixing Leakage, and PLI Indefiniteness.** (A) 10×5 repeated grouped-CV ROC-AUC (`mean ± 1 SD`) across `k = 8` strictly non-nested contrasts including `GCN-Weighted-PLI` and fixed `Linear SVM`. (B) 10×5 mean ΔAUC (`± 95% CI`) under session mixing relative to strict participant grouping. (C) Cohort breakdown of indefinite `I_19 + PLI` matrices (`{int(pli_s['n_indefinite_raw'])}/{int(pli_s['n_sessions'])} = {pli_s['pct_indefinite_raw']:.2f}%`).

### 3.3 Leak-Free Transfer, Elementary Calibration Trade-Offs, and Clinical Severity Floor (RQ3)
As expected from log-offset arithmetic, `[2.0, 30.0] Hz` `specparam` preserved `chi` and periodic residuals under multiplicative scaling (`gamma in [0.01, 100.0]`, `285` spectra, `max |Δchi| = {gchk['max_abs_diff_chi'].max():.2e}`; Online Resource 1, Supplementary Table S2 and Supplementary Figure S3). In uncalibrated 6-pair PD transfer (Table 4B, Online Resource 1, Supplementary Table S19), unparameterized `log10` power suffered mean drift `{amp_raw_5x['lr_mean_prob_drift'].mean():.3f}` (`max = {amp_raw_5x['lr_max_prob_drift'].max():.3f}`) under `5x` gain and AUC degradation (`{cal_raw_lr_raw:.3f} -> {amp_raw_mix['lr_auc'].mean():.3f}`), whereas `specparam` showed zero gain drift (`LR = {cal_raw_lr_sp:.3f}` vs. relative power `{cal_raw_lr_rel:.3f}`). However, normative target-HC standardization traded modest balanced-accuracy gains (`{cal_raw_bacc_sp:.3f} -> {cal_thc_lr_sp['lr_bacc'].mean():.3f}`) for higher Brier error (`{cal_thc_lr_sp['lr_brier'].mean():.3f}`) due to small-control variance. Under disjoint 20-split `Split-Target-HC` transfer (Table 4A, Figure 4A, Online Resource 1, Supplementary Tables S15–S16), `Iowa PD -> UNM PD-OFF` (`Tuned LR {pd_tx.loc[('Iowa_PD', 'UNM_PD_OFF'), 'Tuned_LR']:.3f}`) and `Iowa PD -> UCSD PD` (`LR {pd_tx.loc[('Iowa_PD', 'UCSD_PD'), 'Tuned_LR']:.3f}`) transferred moderately well, slowing cohorts cross-generalized (`Sleep <-> Iowa PD <-> Dementia`, `{td_tx.loc[[('Sleep_Debt', 'Iowa_PD'), ('Sleep_Debt', 'Dementia_All'), ('Iowa_PD', 'Dementia_All')], 'RandomForest'].min():.3f}–{td_tx.loc[[('Sleep_Debt', 'Iowa_PD'), ('Sleep_Debt', 'Dementia_All'), ('Iowa_PD', 'Dementia_All')], 'GCN_OMST'].max():.3f}`; `RF` off-diagonals `0.586–0.749` in Figure 4A), and crossing with batch-confounded Study 2 degraded linear and GNN transfer (`GCN-OMST {td_tx.loc[[('Sleep_Debt', 'Mumtaz_MDD'), ('Mumtaz_MDD', 'Dementia_All'), ('Dementia_All', 'Mumtaz_MDD')], 'GCN_OMST'].min():.3f}–{td_tx.loc[[('Sleep_Debt', 'Mumtaz_MDD'), ('Mumtaz_MDD', 'Dementia_All'), ('Dementia_All', 'Mumtaz_MDD')], 'GCN_OMST'].max():.3f}`, `LR {td_tx.loc[[('Sleep_Debt', 'Mumtaz_MDD'), ('Mumtaz_MDD', 'Dementia_All'), ('Dementia_All', 'Mumtaz_MDD')], 'Tuned_LR'].min():.3f}–{td_tx.loc[[('Sleep_Debt', 'Mumtaz_MDD'), ('Mumtaz_MDD', 'Dementia_All'), ('Dementia_All', 'Mumtaz_MDD')], 'Tuned_LR'].max():.3f}`, `RF {td_tx.loc[[('Sleep_Debt', 'Mumtaz_MDD'), ('Mumtaz_MDD', 'Dementia_All'), ('Dementia_All', 'Mumtaz_MDD')], 'RandomForest'].min():.3f}–{td_tx.loc[[('Sleep_Debt', 'Mumtaz_MDD'), ('Mumtaz_MDD', 'Dementia_All'), ('Dementia_All', 'Mumtaz_MDD')], 'RandomForest'].max():.3f}`; Study 2 EO sensitivity in Online Resource 1, Supplementary Table S12, EC–EO theta `r = {mum_conc.loc['Frontal Periodic Theta (4–8 Hz Residual)', 'pearson_r']:+.3f}`).

{t4}

![](Figure_4.png){{width=6.5in}}
**Figure 4. Leak-Free Cross-Disorder Generalization and Within-Patient Clinical Severity Limits.** (A) `Split-Target-HC` cross-disorder `Random Forest` ROC-AUC (`10×5` diagonals). (B) Dementia Split-0 MMSE predictions showing control-ceiling inflation (`N = 88`) versus flat patient-only predictions (`N = 59`). (C) 10×5 OOF severity correlations (`mean ± SD`) annotated with Split-0 `1,000`-permutation p-values.

Finally, 10×5 clinical regressions paired with 1,000-permutation tests exposed control-ceiling inflation and a severe within-patient variance floor (Table 5, Figure 4B–C, Online Resource 1, Supplementary Table S14). Including ceiling healthy controls (`MMSE = 30.0`, `N = 88`) yielded significant MMSE prediction (`SVR RBF r = {dem_all_svr['pearson_r_mean']:+.3f}, perm p = {fmt_p(dem_all_svr['perm_p'])}, R² = {dem_all_svr['r2_mean']:.3f}`), which collapsed to permutation null upon restricting to `59` AD/FTD patients across all four models (`Ridge r = {dem_pt_ridge['pearson_r_mean']:+.3f}, perm p = {dem_pt_ridge['perm_p']:.4f}, R² = {dem_pt_ridge['r2_mean']:.3f}`; `RF r = {dem_pt_rf['pearson_r_mean']:+.3f}, perm p = {dem_pt_rf['perm_p']:.4f}, R² = {dem_pt_rf['r2_mean']:.3f}`; all `perm p >= 0.111`). In Study 7 Iowa PD (`N = 149: SVR RBF MoCA r = {iowa_all_svr['pearson_r_mean']:+.3f}, perm p = {fmt_p(iowa_all_svr['perm_p'])}, R² = {iowa_all_svr['r2_mean']:.3f}`), restricting to `N = 100` PD patients preserved weak above-chance correlation (`MoCA SVR r = {iowa_pt_svr['pearson_r_mean']:+.3f}, perm p = {iowa_pt_svr['perm_p']:.4f}, R² = {iowa_pt_svr['r2_mean']:.3f}`; `Ridge r = {iowa_pt_ridge['pearson_r_mean']:+.3f}, perm p = {iowa_pt_ridge['perm_p']:.4f}, R² = {iowa_pt_ridge['r2_mean']:.3f}`; `UPDRS-III GBDT r = {iowa_updrs_gbdt['pearson_r_mean']:+.3f}, perm p = {iowa_updrs_gbdt['perm_p']:.4f}, R² = {iowa_updrs_gbdt['r2_mean']:.3f}` vs. permutation-null `Ridge perm p = 0.1169` and `SVR perm p = 0.1598`), yet out-of-fold variance explained never exceeded `R² <= {reg[reg['cohort_key'].str.contains('Patients')]['r2_mean'].max():.3f}` (`10/12` patient-only models negative `OOF R²`).

{t5}

---

## 4. Discussion

### 4.1 Neuroinformatics Resource Contributions and Methodological Lessons
By consolidating `564` unique primary participants (`13,927` unique channel spectra across seven cohorts) into an executable Python verification suite, this study provides concrete diagnostic checks against common computational qEEG errors. Three structural findings stand out: (1) interior `[2.0, 30.0] Hz` `specparam` parameterization (matching `fooof v1.1.1` within `r >= 0.992`) protects effect estimates against both `[7, 14] Hz`-masked OLS theta leakage and analog/digital passband-skirt roll-off, whereas Study 2 MDD separability traces substantially to duplicate files and upstream acquisition-batch prefiltering; (2) under true 10×5 grouped validation, binary and PLI-weighted 19-channel `GCN-OMST` models are formally equivalent within `±0.02` AUC to random graphs and fail to improve over regularized linear, Random Forest, or SPD `19 × 19` sample spatial covariance Riemannian baselines; and (3) `11.70%` non-PSD sensor PLI matrices, bidirectional session-mixing bias, and a strict within-patient severity variance floor (`OOF R² <= 0.035`; dementia MMSE `perm p >= 0.111`) define practical boundaries for clinical EEG translation.

### 4.2 Spectral Fit Windows, Estimator Trade-Offs, and Archive Batch Confounding
Systematic 2D grid evaluation across `105` bounds confirms Donoghue et al. (2022) and Gerster et al. (2022): when PSDs are free of high-frequency attenuation (Studies 3, 5c, and 7), periodic and aperiodic effect sizes stabilize across `[1.0–3.0] × [25.0–40.0] Hz`. When passbands roll off steeply near cutoff—whether from pipeline filtering or pre-export hardware filters—fitting `[1.0, 40.0]` or `[1.0, 45.0] Hz` bends the linear log–log fit downward, inflating or inverting `chi` and masking periodic residuals. Simultaneously comparing continuous `specparam_resid_2to30` against discrete `specparam_peaks_2to30` highlights a principled sensitivity–specificity trade-off: discrete Gaussian peaks suppress baseline fluctuations in strong neurodegenerative slowing (`Study 5a`), whereas continuous residuals preserve small-amplitude low-frequency shifts that thresholded peak finders zero out.

Study 2 (`Figshare 4244171`) illustrates why automated ML benchmarks must inspect raw archive metadata alongside spectral curves (Mahato & Paul, 2020; Saeedi et al., 2021). Despite identical EDF header strings (`HP:0.5 Hz LP:80 Hz`), Apr–Jul 2013 recordings (`all 29` unique MDD + `4` HCs) underwent pre-export high-pass (`~1.5 Hz`) and low-pass (`~35 Hz`) attenuation relative to `2011` and `50 Hz`-contaminated `Jan-2013` controls. Removing `Jan-2013` controls abolished MDD periodic theta shift (`d = -0.082`, 95% CI `[-0.737, +0.572]`; strict-PSD subset `d = -0.675, p = 0.086` retaining `2011` HCs), leaving broadband aperiodic/pre-filter shape (`chi`) as the dominant driver of single-site MDD classification (`~0.80–0.88`). Future studies evaluating affective theta must verify balanced acquisition dates and hardware filter logs before interpreting ML accuracy clinically (Knott et al., 2001; Olbrich & Arns, 2013; Newson & Thiagarajan, 2019). Meanwhile, PD and dementia frontal theta slowing replicated robustly across independent North American and European centers (`k = 4 d_RE = {ndg_m['pooled_d']:+.3f}, p_HKSJ = {ndg_m['p_hksj']:.4f}`; `Anjum et al., 2024`; `Stoffers et al., 2007`; `Cavanagh et al., 2018`).

### 4.3 Topological Equivalence, `Spatial Cov` vs. `PLI` Manifolds, and Validation Hygiene
Two methodological refinements clarify why static 19-node sensor GCNs hit a performance ceiling. First, replacing small-sample Platt scaling (which caused `< 0.50` SVM inversions) with `decision_function` scoring placed `Linear SVM` (`{rep_nn8['Linear_SVM'].mean():.3f}`), `Tuned LR` (`{rep_nn8['Tuned_LogisticRegression'].mean():.3f}`), `Random Forest` (`{rep_nn8['RandomForest'].mean():.3f}`), and `Tuned Ridge` (`{rep_nn8['Tuned_Ridge'].mean():.3f}`) uniformly atop the `k = 8` benchmark, while `TOST (±0.02)` proved equivalence across `GCN-OMST` (`{rep_nn8['GCN_OMST'].mean():.3f}`), `GCN-Weighted-PLI` (`{rep_nn8['GCN_Weighted_PLI'].mean():.3f}`), `GCN-Random` (`{rep_nn8['GCN_Random_EdgeMatched'].mean():.3f}`), and `GCN-Full` (`{rep_nn8['GCN_Full'].mean():.3f}`). Second, contrasting `PLI` against true `19 × 19` sample spatial covariance matrices exposes why PLI-based graph or Riemannian pipelines underperformed: short-window PLI violates positive semi-definiteness (`{int(pli_s['n_indefinite_raw'])}/{int(pli_s['n_sessions'])}` sessions) and discards zero-lag spatial amplitude structure, whereas direct Log-Euclidean spatial covariance features matched top 76-D spectral models (`{riem_nn8.loc[('Riemannian_Spatial_Cov_190D', 'RF')]:.3f}` RF). This negative finding pertains specifically to static 4-feature 19-electrode sensor graphs; dense arrays, source-space cortical meshes, or temporal dynamics remain open tests (Roy et al., 2019; Gemein et al., 2020). Furthermore, 10×5 session-mixing tests confirm quantitatively that record-wise CV can either deflate (`{leak.loc[('Study 1: Sleep Deprivation', 'Leaky_Subject_Mixing', 'RandomForest'), 'mean_delta_vs_strict']:+.3f}` Sleep crossover) or inflate (`{leak.loc[('Study 4: UCSD Parkinson', 'Leaky_Subject_Mixing', 'RandomForest'), 'mean_delta_vs_strict']:+.3f}` UCSD PD) accuracy depending on within-subject label concordance (Saeb et al., 2017; Brookshire et al., 2024).

### 4.4 Transfer Boundaries, Clinical Staging Limits, and Limitations
External `Split-Target-HC` evaluation confirmed moderate cross-center PD transfer (`{pd_tx.loc[('Iowa_PD', 'UCSD_PD'), 'Tuned_LR']:.3f}–{pd_tx.loc[('Iowa_PD', 'UNM_PD_OFF'), 'Tuned_LR']:.3f}`) from `N = 149` Iowa training sets and cross-disorder transfer governed by shared low-frequency slowing (`0.629–0.792`), whereas small-control target z-scoring traded modest balanced accuracy gains for elevated Brier error (Rutherford et al., 2022). Specifically, formal permutation tests on clinical regressions prove that apparent full-cohort `MMSE` and `MoCA` predictability (`r = {iowa_all_svr['pearson_r_mean']:.3f}–{dem_all_svr['pearson_r_mean']:.3f}`) is heavily amplified by healthy control ceilings (`MMSE = 30.0`): within AD/FTD patients alone (`N = 59`), every model failed permutation significance (`perm p >= {reg[reg['cohort_key'] == 'Dementia_MMSE_Patients_Only_N59']['perm_p'].min():.3f}, R² <= -0.047`), and while `N = 100` Iowa PD patients retained weak rank correlation (`r = 0.197–0.293, perm p = 0.002–0.020`), out-of-fold explained variance never exceeded `R² <= {reg[reg['cohort_key'].str.contains('Patients')]['r2_mean'].max():.3f}` (`10/12` patient-only models `R² < 0`). Key limitations include downsampling 32- and 64-channel arrays to 19 clinical electrodes, absence of individual chronological ages in Study 2, short `~41 s` clean EO segments in Study 6, and static epoch-averaged representations.

---

## 5. Conclusions

Across `{acc['total_unique_participants']}` unique primary participants (`{acc['total_analyzed_sessions']}` primary sessions; `{acc['total_spectra']:,}` channel spectra) from seven public archives, this open neuroinformatics benchmark shows that passband-interior `[2.0, 30.0] Hz` `specparam` parameterization resolves dual filter-skirt and masked-OLS leakage pitfalls, whereas raw EDF forensics reveal duplicate recordings and acquisition-batch prefiltering confounding Study 2 MDD differences. Under 10×5 participant-grouped cross-validation and `TOST` equivalence tests, static 19-channel `GCN-OMST` and `GCN-Weighted-PLI` match edge-matched random topology and fail to beat simple regularized linear, tree, or SPD sample spatial covariance models, while `{pli_s['pct_indefinite_raw']:.2f}%` indefinite PLI matrices, bidirectional session-mixing bias, and low within-patient severity variance explained (`OOF R² <= 0.035`; dementia `perm p >= 0.111`) underscore the value of transparent open verification suites prior to clinical deployment.

---

## Information Sharing Statement

All seven EEG datasets are publicly accessible via OpenNeuro and Figshare, and all Python preprocessing, Study 2 EDF forensics, 2D fit-window grid, `fooof` equivalence checks, 10×5 grouped CV, `Spatial Cov` Riemannian baselines, figure generation (`scripts/42_generate_all_specparam_figures.py`, `scripts/47_run_v5_forensics_and_refinements.py`, `scripts/48_build_paper2_neuroinformatics_v5.py`, `src/harmonizer_and_graph.py`), and Zenodo-ready tabular artifacts (`release_artifacts/derived_session_features_733.csv`, `cv_fold_indices_split0.csv`, `oof_predictions_split0.csv`, `requirements_lock.txt`) are openly released under the MIT license at `{GITHUB_URL}`. Dataset accessions: Study 1 OpenNeuro `ds004902` (Xiang et al., 2024); Study 2 Figshare `4244171` (Mumtaz et al., 2017); Study 3 OpenNeuro `ds003478` (Cavanagh et al., 2019); Study 4 OpenNeuro `ds002778` (Rockhill et al., 2020); Study 5 OpenNeuro `ds004504` (Miltiadous et al., 2023); Study 6 OpenNeuro `ds003490` (Cavanagh et al., 2018); Study 7 OpenNeuro `ds004584` (Singh et al., 2023).

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

### Data Availability
All underlying raw EEG recordings are hosted permanently at OpenNeuro (`ds004902`, `ds003478`, `ds002778`, `ds004504`, `ds003490`, `ds004584`) and Figshare (`4244171`). Derived session-level spectral/topological features (`derived_session_features_733.csv`), participant-grouped fold indices, and out-of-fold probability tables are deposited openly in `{GITHUB_URL}/tree/main/release_artifacts`.

### Code Availability
Complete executable source code (`src/harmonizer_and_graph.py`, `scripts/47_run_v5_forensics_and_refinements.py`, `scripts/42_generate_all_specparam_figures.py`, `scripts/48_build_paper2_neuroinformatics_v5.py`) and pinned dependencies (`requirements_lock.txt`) are available at `{GITHUB_URL}` under the MIT license.

### Author Contributions (CRediT)
Chushu Tan: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing – original draft, Writing – review & editing, Visualization, Project administration.

### AI-Assisted Coding and Editorial Verification Disclosure
During the preparation of this work, the author used AI-assisted software engineering and language-editing tools to draft analysis scripts, structure verification tables, and refine English phrasing. The author independently verified all numerical metrics, statistical tests, and scientific interpretations against raw computational pipeline outputs and takes full responsibility for the accuracy and integrity of the article.

---

## References

{refs_md}
"""
    return md


# ==============================================================================
# Online Resource 1 (Supplementary Tables S1–S27 + Figs S1–S6)
# ==============================================================================
def build_supplementary_material(d: dict) -> str:
    omst, acc = d["master"]["omst_topology"], d["master"]["accounting"]
    s1_df, gain_chk, bio_exp, lofo, hparam = d["s1_omst"], d["gain_chk"], d["bio_exp"], d["lofo"], d["hparam"]
    cls0, wil, sim_mc, boot_d, mum_eo = d["cls0"], d["wilcoxon"], d["sim_mc"], d["boot_d"], d["mum_eo"]
    mum_conc, mum_tx, med, reg, pd_tx, td_tx = d["mum_conc"], d["mum_tx"], d["med"], d["reg"], d["pd_tx"], d["td_tx"]
    anc, anc_dur, leak, gc, rep10 = d["ancova"], d["anc_dur"], d["leakage"], d["gain_cal"], d["rep10"]
    pli_a, pli_s, riem, meta_df, fit_df = d["pli_audit"], d["pli_sum"].iloc[0], d["riem"], d["meta"], d["fit"]
    iapf_df, heur_df, ep_df, unm_sens = d["iapf"], d["heur"], d["ep_audit"], d["unm_sens"]
    mum_dups, mum_for, s2_unf, grid2d, feq = d["mum_dups"], d["mum_forensic"], d["s2_unfilt"], d["grid2d"], d["fooof_eq"]

    parts = [f"""# Online Resource 1: Supplementary Material

**{EXACT_TITLE}**

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

### Supplementary Figures
- **Figure S1:** Single-Subject Interior [2.0, 30.0] Hz `specparam` vs. Masked-OLS Illustration
- **Figure S2:** Canonical 19-Channel 10–20 OMST Connectome Topology (`N = 675` primary graphs)
- **Figure S3:** Empirical Gain-Invariance Verification Across Four Orders of Magnitude (`n = 285` spectra)
- **Figure S4:** Grand-Average Welch Power Spectra (1–40 Hz) and [2.0, 30.0] Hz Residuals Across 7 Cohorts
- **Figure S5:** Seven-Cohort Participant Flow, QC Exclusions, and Study 2 SHA-256 Deduplication
- **Figure S6:** Aligned 1,000-Permutation Retraining Distributions (`GridSearchCV Tuned LR` and `GCN-OMST`)

### Supplementary Tables
- **Table S1:** Canonical 19-Channel Coordinates and Empirical OMST Metrics Across `675` Primary Graphs
- **Table S2:** Empirical Gain-Invariance Audit Across `285` Spectra (`[2.0, 30.0] Hz` Solver)
- **Table S3:** Cohort Designs, Demographics, Hardware Systems, and Clinical Scales
- **Table S4:** Complete 50-Test Multi-Cohort Biophysical Summary (`[2.0, 30.0] Hz` `specparam`)
- **Table S5:** Canonical 19-Channel 10–20 Label Harmonization Mapping
- **Table S6:** Single-Biomarker Baselines and Leave-One-Feature-Out `NormativeGCN` Ablations (Split-0)
- **Table S7, S7b, S7c:** `NormativeGCN` Architecture, Classical Grids, and Hyperparameter Sensitivity
- **Table S8:** Preprocessing Specifications Across All Seven Datasets
- **Table S9:** Verified Computational Environment (`requirements_lock.txt`) and Open Repository
- **Table S10 & S10b:** Complete Extended Model Comparison (Split-0 95% CIs + 10×5 Means), Paired ΔAUC CIs, and `TOST` Equivalence
- **Table S11:** Empirical-Variance-Calibrated Monte Carlo Simulation and Bootstrap Paired Δd Contrasts
- **Table S12:** Study 2 Secondary Eyes-Open Benchmark (`N = 58` unique EO) and State-Matched Transfer
- **Table S13:** Within-Subject Paired Dopaminergic Medication Comparisons (`UNM N = 25` & `UCSD N = 15`)
- **Table S14:** Complete 20-Row 10×5 Repeated Clinical Regression Benchmark With `1,000`-Permutation `p`-Values
- **Table S15 & S16:** Leak-Free 50/50 `Split-Target-HC` Cross-Site PD and Cross-Domain Transfer Matrices
- **Table S17:** Complete Age-Adjusted ANCOVA Across Four EEG Markers
- **Table S18:** 10×5 Repeated Validation Leakage Quantification Across Repeated-Session Cohorts
- **Table S19:** Cross-Site PD Calibration Ablation and Gain Stress-Testing
- **Table S20:** True 10×5 Repeated Participant-Grouped Cross-Validation Summary Across All `12` Contrasts × `13` Models
- **Table S21:** PLI Matrix Definiteness Audit (`675` Primary Sessions) and 10×5 Euclidean / PLI / `19 × 19` Sample Spatial Covariance Riemannian Baselines
- **Table S22:** Random-Effects Meta-Analysis (`DL`, `HKSJ`, `K = 6` & `K = 5` Omit Study 2) and Leave-One-Out Suite
- **Table S23:** Per-Cohort `specparam v2.0.0rc7` Diagnostics (`[1.0, 40.0]` vs. `[2.0, 30.0] Hz`)
- **Table S24:** Posterior Individual Alpha Peak Frequency (`IAPF`) and 10×5 Heuristic Slowing Benchmarks
- **Table S25:** Clean Epoch Balance, Joint Age + Duration ANCOVA, and UNM Pure EO vs. Pure EC Sensitivity
- **Table S26:** Figshare `4244171` (Study 2) Duplicate SHA-256 Hashing, Acquisition-Batch Sensitivity (w/ 95% CIs), and Raw Unfiltered EDF `0.5–80 Hz` Batch Profiles
- **Table S27:** Complete 105-Configuration 2D `specparam` Fit-Window Grid (`[1.0–3.0] × [25.0–45.0] Hz`) and Legacy `fooof v1.1.1` Equivalence Audit

---

## Section S1. Extended Supplementary Methods

### S1.1 Interior [2.0, 30.0] Hz Parameterized Spectral Decomposition (`specparam v2.0.0rc7`)
Welch PSDs (`4.0 s` epochs, `2.0 s` Hamming sub-windows, `50%` overlap, `0.50 Hz` step) were parameterized via `SpectralGroupModel` in `specparam v2.0.0rc7` (`aperiodic_mode='fixed'`, `peak_width_limits=[1.0, 12.0]` Hz, `max_n_peaks=6`, `min_peak_height=0.05` B, `peak_threshold=2.0` SD) over primary `[2.0, 30.0] Hz`, edge `[1.0, 40.0] Hz`, and the full 105-setting 2D sensitivity grid (`[1.0–3.0] × [25.0–45.0] Hz`):

$$\\log_{{10}} \\mathrm{{PSD}}(f) = (b - \\chi \\log_{{10}} f) + \\sum_{{n=1}}^N a_n \\exp\\left(-\\frac{{(f - c_n)^2}}{{2 w_n^2}}\\right)$$

### S1.2 Data-Driven Connectomics via Orthogonal Minimum Spanning Trees (OMST)
Across `675` canonical primary graphs (`{omst['total_node_instances']:,}` nodes), iterative Kruskal OMST trees maximizing `GCE` averaged `{omst['mean_trees']:.2f} ± {omst['sd_trees']:.2f}` trees (`{omst['pct_tree_1']:.2f}%` `M = 1`, `{omst['pct_tree_2']:.2f}%` `M >= 2`), `{omst['mean_edges']:.2f} ± {omst['sd_edges']:.2f}` edges, `GCE = {omst['mean_gce']:.4f} ± {omst['sd_gce']:.4f}`, and `0.0%` isolated nodes.

### S1.3 Elementary Gain Scaling and Non-Multiplicative Boundaries
Multiplying time-series voltage by scalar `k > 0` adds `2 log10(k)` uniformly across `log10` Welch power, absorbed analytically into offset `b' = b + 2 log10(k)`. Hardware roll-offs, line noise, references, and demographic batch shifts are non-multiplicative and require archive auditing and fold-nested standardization.

### S1.4 `NormativeGCN` Architecture
Pure-PyTorch `NormativeGCN` (`src/harmonizer_and_graph.py`) processes `[19, 4]` z-scored features and `19 × 19` adjacency (`A + I`, evaluated both as binary `GCN_OMST` and PLI-weighted `GCN_Weighted_PLI = A_bin ⊙ PLI`), applying `4->32->32` symmetric-normalized graph convolutions (Dropout `0.25`), concatenated mean+max pooling (`64-D`), and `64->32->2` MLP classification totaling `3,362` parameters (Adam `lr=0.005`, `wd=1e-4`, `50` fixed epochs).

---

## Supplementary Figures

![Figure S1](fig_S1_specparam_example.png){{width=6.4in}}
**Figure S1. Single-Subject Interior [2.0, 30.0] Hz `specparam` vs. Masked-OLS Illustration.** Real empirical spectrum from Study 5 AD participant `sub-018` (`Ch Fz`), showing `[7, 14] Hz`-masked OLS pulling slope upward (`χ_OLS = 1.287` vs. `specparam χ = 1.074`) and under-estimating 4–8 Hz periodic theta (`R² = 0.954`, `MAE = 0.083 B`).

![Figure S2](fig_S2_omst_topology.png){{width=6.2in}}
**Figure S2. Canonical 19-Channel 10–20 OMST Connectome Topology.** Group-averaged spanning tree structure across `675` primary graphs (`{omst['mean_edges']:.2f}` edges, zero isolated nodes).

![Figure S3](fig_S3_gain_invariance.png){{width=6.4in}}
**Figure S3. Empirical Scaling Stability of Interior [2.0, 30.0] Hz `specparam`.** Verified across `n = 285` spectra (`k in [0.01, 100.0]`) showing exact `2 log10(k)` offset tracking and `<= {gain_chk['max_abs_diff_chi'].max():.2e}` max deviation.

![Figure S4](fig_S4_cohort_spectra.png){{width=6.5in}}
**Figure S4. Cohort-Averaged Welch Power Spectra (1–40 Hz) and Interior [2.0, 30.0] Hz Residuals Across 7 Cohorts.** Study 2 reflects de-duplicated unique participants (`29 MDD vs. 27 HC EC`).

![Figure S5](fig_S5_enrollment_flowchart.png){{width=6.4in}}
**Figure S5. Seven-Cohort Participant Flow, Pre-Analysis QC, Study 6 Pure-EO Isolation, and Study 2 SHA-256 Deduplication.**

![Figure S6](fig_S6_permutation_tests.png){{width=6.5in}}
**Figure S6. Aligned 1,000-Permutation Retraining Null Distributions.** Identical `GridSearchCV Tuned LR` and participant-cluster splits across four within-cohort and two `Split-Target-HC` benchmarks.

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
| Gain Factor (k) | Spectra (n) | Expected Offset 2 log10(k) (B) | Measured Offset Shift Mean (B) | Max |Δchi| Across 285 Spectra | Max |Δθ_resid| (B) |
| :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in gain_chk.iterrows():
        parts.append(f"| {r['gain_k']:.2f} | {int(r['n_spectra'])} | {r['expected_offset_shift']:+.5f} | {r['measured_offset_shift_mean']:+.5f} | {r['max_abs_diff_chi']:.2e} | {r['max_abs_diff_theta_resid']:.2e} |")

    parts.append("""
### Table S3. Multi-Cohort Study Designs, Demographics, Hardware Systems, and Clinical Scales
| Study ID | Dataset & Setting | Analyzed N (Pts / Sess) | Group Breakdown | Age Years (Mean ± SD) | Female (%) | Native Hardware | Sampling Rate | Protocol / Assessments |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- | :---: | :--- |
| Study 1 | Sleep Deprivation (`ds004902`) | 71 / 142 | SD (71) vs. NS (71) [Paired] | 20.0 ± 1.4 | 47.9% | BrainVision actiCHamp (64 Ch) | 500 → 250 Hz | 36 h monitored wakefulness crossover (EO) |
| Study 2 | Outpatient MDD (`Figshare 4244171`) | 56 / 56 (+58 EO uniq.) | 29 MDD vs. 27 HC EC (31 vs. 27 EO; nom. 58/60) | MDD 40.3 ± 12.9; HC 38.2 ± 15.6 (cohort summary) | 37.9% | 19-ch clinical linked-ear (`-LE`) system | 256 → 250 Hz | DSM-IV MDD; 2 SHA-256 dup pairs removed + 3 batch regimes |
| Study 3 | Univ. Depression (`ds003478`) | 119 / 119 | BDI >= 14 (44) vs. <= 7 (75) | 19.4 ± 2.0 (N=118) | 60.5% | BioSemi ActiveTwo (64 Ch) | 500 → 250 Hz | EO; SCID (11 curr, 11 past MDD, 8 non-MDD, 14 unint.) |
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
| Pz | Parietal Midline | Direct | `EEG Pz-LE` | Direct | Direct | `EEG Pz` | Direct | CAR-recovered |

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
| Input Tensor | 19 nodes × 4 z-scored features | Fold-nested control standardization |
| GCN Layers | 2× `PurePyTorchGCNLayer` (`4->32->32`, ReLU, Dropout 0.25) | 160 + 1,056 = 1,216 params (`GCN-OMST` & `GCN-WPLI`) |
| Global Readout | Concatenated Mean + Max Pool across 19 nodes | 64-D representation |
| MLP Head | `Linear(64->32) -> ReLU -> Dropout(0.25) -> Linear(32->2)` | 2,080 + 66 = 2,146 params |
| Total Parameters | 3,362 trainable parameters | Fixed 50 Adam epochs (`lr=0.005`, `wd=1e-4`) |

### Table S7b. Classical and Univariate Baseline Configurations
| Model Identifier | Input Space | Inner Tuning / Settings | Estimator |
| :--- | :--- | :--- | :--- |
| `Tuned_LogisticRegression` | Flattened 76-D | Nested 3-fold CV L2 `C in {0.001..10.0}` | `LogisticRegression` |
| `Tuned_Ridge` | Flattened 76-D | Nested 3-fold CV `alpha in {0.01..100.0}` | `RidgeClassifier` |
| `Linear_SVM` / `RBF_SVM` | Flattened 76-D | Nested 3-fold CV `C`, `gamma`; `sigmoid(decision_function)` scoring | `SVC(probability=False)` |
| `RandomForest` | Flattened 76-D | 100 trees, `max_depth=5` | `RandomForestClassifier` |
| `GBDT` | Flattened 76-D | 100 iterations, `max_depth=4` | `HistGradientBoostingClassifier` |
| `Single_Theta` / `Single_Chi` | 1-D θ / χ | Default L2 (`C=1.0`) | `LogisticRegression` |

### Table S7c. `NormativeGCN` Hyperparameter Sensitivity Across Five Cohorts (Split-0)
| Hyperparameter Configuration | Study 2 MDD | Study 3 Dep. | Study 4 UCSD | Study 6a UNM | Study 7 Iowa | 5-Cohort Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |""")
    hp_piv = hparam.pivot(index="config_label", columns="cohort", values="auc")
    hp_cols = list(hp_piv.columns)
    for cfg in hparam["config_label"].drop_duplicates():
        row_vals = [float(hp_piv.loc[cfg, c]) for c in hp_cols]
        parts.append(f"| {cfg} | " + " | ".join(f"{v:.3f}" for v in row_vals) + f" | {np.mean(row_vals):.3f} |")

    parts.append("""
### Table S8. Preprocessing Specifications and Thresholds Across All Seven Datasets
| Dataset | Condition | Analyzed Sessions | Threshold | Sampling / Filter | Mean Clean Epochs ± SD [Range] | Retained Duration (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in ep_df.iterrows():
        thr = "200 μV" if "Iowa" in r["cohort"] else "150 μV"
        parts.append(f"| {r['cohort']} | EO/EC Rest | {int(r['n_sessions'])} | {thr} | 250 Hz, 1–40 Hz FIR | {r['all_mean']:.1f} ± {r['all_sd']:.1f} [{int(r['all_min'])}–{int(r['all_max'])}] | {4.0 * r['all_mean']:.1f} ± {4.0 * r['all_sd']:.1f} s |")

    parts.append("""
### Table S9. Verified Computational Environment and Reproducibility Artifacts
| Package / Artifact | Version / Path | Role |
| :--- | :---: | :--- |
| Python / PyTorch | 3.13.7 / 2.8.0 | Core execution & `NormativeGCN` |
| MNE-Python / `specparam` | 1.12.1 / 2.0.0rc7 | Preprocessing & spectral parameterization (cross-checked vs. `fooof 1.1.1`) |
| `scikit-learn` / `scipy` / `statsmodels` | 1.7.2 / 1.16.2 / 0.14.6 | Nested CV classifiers, Wilcoxon, `TOST`, ANCOVA, BH-FDR |
| Zenodo Release Derivatives | `release_artifacts/` | `derived_session_features_733.csv`, `cv_fold_indices_split0.csv`, `oof_predictions_split0.csv`, `requirements_lock.txt` |

### Table S10. Split-0 and 10×5 Within-Cohort Benchmark Across All 13 Models
| Cohort Contrast | Model | N | 10×5 Mean AUC (±SD) | Split-0 ROC-AUC [95% CI] | Balanced Acc | Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |""")
    n_map_s10 = {**dict(zip(lofo["cohort"], lofo["n_samples"])), "Study 2b: Mumtaz MDD (Eyes Open)": 58}
    for _, r in cls0.iterrows():
        parts.append(f"| {r['cohort']} | `{r['model']}` | {int(n_map_s10[r['cohort']])} | {r['mean_auc_10x5']:.3f} (±{r['sd_auc_10x5']:.3f}) | {r['auc']:.3f} [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}] | {100.0 * r['balanced_accuracy']:.1f}% | {100.0 * r['accuracy']:.1f}% |")

    parts.append("""
### Table S10b. Pairwise Wilcoxon, Paired ΔAUC 95% CIs, and `TOST` Equivalence on 10×5 Means (`GCN_OMST` vs. Comparators)
| Evaluation Scope | Comparator Model | GCN Mean | Comp Mean | Mean Diff [95% CI] | TOST ±0.02 p | TOST ±0.03 p | W / Raw p / FDR q |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in wil.iterrows():
        parts.append(
            f"| {r['scope']} (k={int(r['n_benchmarks'])}) | `{r['comparator_model']}` | {r['gcn_mean_auc']:.4f} | {r['comp_mean_auc']:.4f} | "
            f"{r['mean_diff_auc']:+.4f} [{r['ci95_diff_lo']:+.4f}, {r['ci95_diff_hi']:+.4f}] | "
            f"{r['tost_p_002']:.4f} | {r['tost_p_003']:.4f} | "
            f"{r['wilcoxon_W']:.1f} / {r['raw_p']:.4f} / {r['fdr_q']:.4f} |"
        )

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
### Table S14. Complete 20-Row 10×5 Repeated Clinical Scale Regression Benchmark (With `1,000`-Permutation Tests)
| Cohort & Scope | Scale | N | Regressor | 10×5 r (±SD) | Perm p (1-sid) | Median Param p | Mean R² | Mean MAE | Split-0 r |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in reg.iterrows():
        parts.append(f"| {r['cohort']} | {r['scale']} | {int(r['n_samples'])} | `{r['model']}` | {r['pearson_r_mean']:+.3f} (±{r['pearson_r_sd']:.3f}) | {fmt_p(r['perm_p'])} | {fmt_p(r['median_p'])} | {r['r2_mean']:.3f} | {r['mae_mean']:.3f} | {r['split0_r']:+.3f} |")

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
| Experiment | Calibration | Feature Type | Gain k | LR AUC (BalAcc / Brier) | RF AUC | Mean / Max |ΔFeat| | Mean / Max |Δp| (LR) |
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
### Table S21. PLI Matrix Definiteness Audit (`675` Primary Sessions) and 10×5 Riemannian Baselines (Including Sample Spatial Covariance)
#### Panel A. Eigenvalue Spectrum of `19 × 19` Symmetrized PLI Matrices (`I_19 + PLI`)
| Cohort | Sessions (N) | Min λ_min | Median λ_min | Indefinite (λ_min <= 0) | % Indefinite | After +10^-4 I |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in pd.concat([pli_a, d["pli_sum"]], ignore_index=True).iterrows():
        parts.append(f"| {r['cohort']} | {int(r['n_sessions'])} | {r['min_eigenvalue']:+.4f} | {r['median_min_eigenvalue']:+.4f} | {int(r['n_indefinite_raw'])}/{int(r['n_sessions'])} | {r['pct_indefinite_raw']:.2f}% | {int(r['n_indefinite_after_1e4_shift'])} still <= 0 |")
    parts.append("")
    parts.append("#### Panel B. 10×5 Euclidean PLI (171-D), SPD-Conditioned PLI Tangent (190-D), and `19 × 19` Sample Spatial Covariance Tangent (`Riemannian_Spatial_Cov_190D`)")
    parts.append("| Cohort | Classifier | Euclidean PLI 171-D | Diag-Loaded PLI SPD | Eigen-Clipped PLI SPD | `Riemannian_Spatial_Cov_190D` (±SD) |")
    parts.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    rp_m = riem.pivot(index=["cohort", "classifier"], columns="representation", values="auc")
    rp_s = riem.pivot(index=["cohort", "classifier"], columns="representation", values="sd_auc")
    for (cname, clf), r in rp_m.iterrows():
        rs = rp_s.loc[(cname, clf)]
        parts.append(f"| {cname} | `{clf}` | {r['Euclidean_UpperTri_171D']:.3f} (±{rs['Euclidean_UpperTri_171D']:.3f}) | {r['Riemannian_Gershgorin_SPD_190D']:.3f} (±{rs['Riemannian_Gershgorin_SPD_190D']:.3f}) | {r['Riemannian_Higham_SPD_190D']:.3f} (±{rs['Riemannian_Higham_SPD_190D']:.3f}) | {r['Riemannian_Spatial_Cov_190D']:.3f} (±{rs['Riemannian_Spatial_Cov_190D']:.3f}) |")

    parts.append("""
### Table S22. Random-Effects Meta-Analysis (`DL`, `HKSJ`, `K = 6` & `K = 5` Omitting Study 2) and Leave-One-Out Suite
| Method | Scope | Subgroup | k | Pooled d_RE | DL 95% CI | HKSJ 95% CI | DL p | HKSJ p | I² | Q_between (p) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in meta_df.iterrows():
        parts.append(f"| `{r['method']}` | {r['analysis_set']} | {r['subgroup']} | {int(r['k_studies'])} | {r['pooled_d']:+.3f} | [{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}] | [{r['hksj_ci_lower']:+.3f}, {r['hksj_ci_upper']:+.3f}] | {fmt_p(r['p_value'])} | {fmt_p(r['p_hksj'])} | {r['I2_percent']:.1f}% | {r['Q_between']:.2f} ({fmt_p(r['p_between'])}) |")

    parts.append("""
### Table S23. Per-Cohort `specparam v2.0.0rc7` Diagnostics Across Fit Windows (`[1.0, 40.0]` vs. `[2.0, 30.0] Hz`)
| Fit Window | Cohort Key | Sessions | Spectra | Failed | % R² < 0.80 | R² p05 / Median / Mean | MAE Med / Mean | Mean Peaks | Mean χ_OLS vs. χ_sp |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |""")
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
### Table S26. Figshare `4244171` (Study 2) Duplicate Hashing, Batch Sensitivity (w/ 95% CIs), and Raw Unfiltered EDF `0.5–80 Hz` Profiles
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
        parts.append(
            f"| {r['subset']} | {int(r['n_total'])} ({int(r['n_case'])}/{int(r['n_ctrl'])}) | "
            f"{r['theta_d']:+.3f} [{r['theta_ci_lo']:+.3f}, {r['theta_ci_hi']:+.3f}] ({fmt_p(r['theta_p'])}) | "
            f"{r['chi_d']:+.3f} [{r['chi_ci_lo']:+.3f}, {r['chi_ci_hi']:+.3f}] ({fmt_p(r['chi_p'])}) | "
            f"{lrs} | {rfs} |"
        )
    parts.append("")
    parts.append("#### Panel C. Raw Unfiltered EDF `0.5–80 Hz` Welch Log10 PSD Profiles by Acquisition Batch (Identical Nominal `HP:0.5 Hz LP:80 Hz` Headers)")
    parts.append("| Acquisition Batch (Unique EC) | n | Header Filter | log10(1 Hz) | log10(2 Hz) | log10(30 Hz) | log10(40 Hz) | log10(50 Hz) | 1−2 Hz Drop | 40−30 Hz Roll-Off | 50 Hz Spike |")
    parts.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    s2u_g = s2_unf[~s2_unf["is_duplicate_sha256"]].groupby("batch", sort=False).agg(
        n=("subject_id", "count"), logp_1hz=("logp_1hz", "mean"), logp_2hz=("logp_2hz", "mean"),
        logp_30hz=("logp_30hz", "mean"), logp_40hz=("logp_40hz", "mean"), logp_50hz=("logp_50hz", "mean"),
        d21=("drop_2_to_1hz", "mean"), d34=("drop_30_to_40hz", "mean"), l50=("line_50hz_excess", "mean"),
    )
    for bname, r in s2u_g.iterrows():
        parts.append(f"| `{bname}` | {int(r['n'])} | `HP:0.5Hz LP:80Hz` | {r['logp_1hz']:.3f} | {r['logp_2hz']:.3f} | {r['logp_30hz']:.3f} | {r['logp_40hz']:.3f} | {r['logp_50hz']:.3f} | {r['d21']:+.3f} | {r['d34']:+.3f} | {r['l50']:+.3f} |")

    parts.append("""
### Table S27. Complete 105-Configuration 2D `specparam` Fit-Window Grid (`[1.0–3.0] × [25.0–45.0] Hz`) and `fooof v1.1.1` Equivalence
#### Panel A. Representative 2D Fit-Window Grid Across Cohorts and Unfiltered Study 2 EDFs (`0/105` Failed Sets; Selected Windows Across `[1.0..3.0] × [25.0..45.0] Hz`; Full CSV in Repo)
| Cohort & PSD Source | f_min (Hz) | f_max (Hz) | Spectra | Failed | Median R² | Median MAE | Frontal θ d (p) | Global χ d (p) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |""")
    sub_g = grid2d[grid2d["f_min_hz"].isin([1.0, 2.0, 3.0]) & grid2d["f_max_hz"].isin([25.0, 30.0, 35.0, 40.0, 45.0])]
    for _, r in sub_g.iterrows():
        parts.append(f"| {r['cohort']} | {r['f_min_hz']:.1f} | {r['f_max_hz']:.1f} | {int(r['n_spectra'])} | {int(r['n_failed'])} | {r['r2_median']:.3f} | {r['mae_median']:.4f} | {r['theta_d']:+.3f} ({fmt_p(r['theta_p'])}) | {r['chi_d']:+.3f} ({fmt_p(r['chi_p'])}) |")
    parts.append("")
    parts.append("#### Panel B. Empirical Algorithmic Equivalence Between Legacy `fooof v1.1.1` and `specparam v2.0.0rc7` (`665` Shared Real Spectra)")
    parts.append("| Fit Window | Scope | Spectra (n) | r (χ) | Mean |Δχ| | r (θ Residual) | Mean |Δθ| (B) | Max |ΔR²| |")
    parts.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, r in feq.iterrows():
        parts.append(f"| {r['fit_window_hz']} Hz | `{r['cohort']}` | {int(r['n_spectra'])} | {r['pearson_r_chi']:.4f} | {r['mean_abs_diff_chi']:.4f} | {r['pearson_r_theta_resid']:.4f} | {r['mean_abs_diff_theta_resid']:.5f} | {r['max_abs_diff_r2']:.4f} |")

    return "\n".join(parts)


# ==============================================================================
# Supporting Documents (Cover Letter w/ Overlap Disclosure, Highlights, Title, Decl, Checklist)
# ==============================================================================
def build_cover_letter(d: dict) -> str:
    acc = d["master"]["accounting"]
    rep = d["rep10"]
    nn8 = [c for c in rep["cohort"].unique() if not any(k in c for k in ("2b", "5c", "6b", "6c"))]
    piv8 = rep[rep["cohort"].isin(nn8)].pivot(index="cohort", columns="model", values="mean_auc")
    wil8 = d["wilcoxon"][d["wilcoxon"]["scope"] == "NonNested_8_Contrasts_10x5Mean"].set_index("comparator_model")
    pli_s = d["pli_sum"].iloc[0]
    ndg = d["meta"][(d["meta"]["method"] == "specparam_2to30_resid") & (d["meta"]["analysis_set"] == "Primary_K6_SubjMean_PD_OFF") & (d["meta"]["subgroup"] == "PD_Dementia_Cohorts")].iloc[0]
    dem_pt = d["reg"][(d["reg"]["cohort_key"] == "Dementia_MMSE_Patients_Only_N59") & (d["reg"]["model"] == "Ridge_Regression")].iloc[0]
    fit2 = d["fit"][d["fit"]["fit_window_hz"] == "2.0-30.0"].set_index("cohort_key")
    fit1 = d["fit"][d["fit"]["fit_window_hz"] == "1.0-40.0"].set_index("cohort_key")

    return f"""# Cover Letter

**Date:** {SUBMISSION_DATE}

**To:**  
{EDITOR_NAME}  
{EDITOR_TITLE}  
{EDITOR_AFFIL}

**Subject:** Original Research Article / Neuroinformatics Resource Submission to *{JOURNAL_NAME}*

Dear Prof. Van Horn,

Please consider the original research article entitled **"{EXACT_TITLE}"** for publication in *{JOURNAL_NAME}*.

This manuscript introduces an open Python benchmarking and dataset-auditing suite spanning seven public 19-channel resting-state EEG archives (`{acc['total_unique_participants']}` unique primary participants, `{acc['total_analyzed_sessions']}` primary sessions, `{acc['total_with_mumtaz_eo']}` total sessions, `{acc['total_spectra']:,}` unique channel spectra) addressing three pervasive methodological failure modes:

1. **Spectral Fit-Window Stability (`105` 2D Grid Settings) and Raw Archive Forensics:** Interior `[2.0, 30.0] Hz` `specparam v2.0.0rc7` (cross-verified against legacy `fooof v1.1.1`, `r >= 0.992`) prevented high-frequency roll-off collapse in Study 2 outpatient MDD (`R² {fit1.loc['mumtaz_ec', 'r2_median']:.3f} -> {fit2.loc['mumtaz_ec', 'r2_median']:.3f}`) and eliminated `[7, 14] Hz`-masked OLS bias in Study 3 student depression and calibrated Monte Carlo simulations. Raw-EDF SHA-256 and unfiltered `0.5–80 Hz` spectral forensics further exposed duplicate files (`H S27` = `H S30`, `MDD S33` = `MDD S34`) and identical Apr–Jul 2013 pre-export attenuation confounding diagnostic group with acquisition batch (`d = -1.188` vs. `-0.082, p = 0.752` excluding Jan-2013 controls), while PD and dementia slowing replicated across independent cohorts (`d_RE = {ndg['pooled_d']:+.3f}, p_HKSJ = {ndg['p_hksj']:.4f}`).
2. **True 10×5 Grouped-CV Equivalence Testing (`TOST`), Spatial Covariance Riemannian Baselines, and Permutation Checks:** Across `k = 8` strictly non-nested contrasts and `13` configurations (adding continuous `GCN_Weighted_PLI` and fixing `Linear_SVM` / `RBF_SVM` via `decision_function`), `GCN-OMST` (`{piv8['GCN_OMST'].mean():.3f}`) was statistically equivalent within `±0.02` AUC (`TOST p <= {wil8.loc['GCN_Random_EdgeMatched', 'tost_p_002']:.4f}`) to `GCN-Weighted-PLI` (`{piv8['GCN_Weighted_PLI'].mean():.3f}`) and edge-matched `GCN-Random` (`{piv8['GCN_Random_EdgeMatched'].mean():.3f}`), trailing regularized classical baselines (`Linear SVM {piv8['Linear_SVM'].mean():.3f}`, `Tuned LR {piv8['Tuned_LogisticRegression'].mean():.3f}`, `RF {piv8['RandomForest'].mean():.3f}`) and true `19 × 19` sample spatial covariance Riemannian tangent features (`0.724` RF). Meanwhile, `{int(pli_s['n_indefinite_raw'])}/{int(pli_s['n_sessions'])} ({pli_s['pct_indefinite_raw']:.2f}%)` symmetrized PLI matrices were indefinite.
3. **External Transfer Boundaries and Patient-Only Severity Permutations:** Under disjoint `Split-Target-HC` transfer and `10×5` clinical regressions paired with `1,000`-permutation retraining tests, removing ceiling controls abolished dementia patient MMSE significance (`N = 59: Ridge r = {dem_pt['pearson_r_mean']:+.3f}, perm p = {dem_pt['perm_p']:.4f}, R² = {dem_pt['r2_mean']:.3f}; all perm p >= 0.111`), and although `N = 100` Iowa PD patients retained weak correlation (`MoCA SVR r = +0.293, perm p = 0.0020; UPDRS-III GBDT r = +0.242, perm p = 0.0070`), out-of-fold variance explained never exceeded `R² <= 0.035` (`10/12` patient-only models `R² < 0`).

**Dataset Overlap Disclosure:** For full transparency, subsets of two of the seven public archives examined here are analyzed in distinct single-domain companion manuscripts by the author (one manuscript submitted to *Journal of Alzheimer's Disease* examining clinical decision-curve utility and electrode-reference sensitivity on Figshare `4244171` vs. Alzheimer's/depression baselines, and one manuscript studying acute sleep-deprivation neurophysiology on OpenNeuro `ds004902`). The present submission is wholly distinct in scope, methodology, and open software contribution—assembling all seven cohorts jointly into an open Python benchmarking, 2D window-auditing, topological equivalence (`TOST`), PLI-indefiniteness, and spatial-covariance comparison resource—and neither overlaps verbatim nor duplicates any figures, tables, or central claims elsewhere.

All code, `release_artifacts/derived_session_features_733.csv`, fold indices, and `requirements_lock.txt` are openly released at `{GITHUB_URL}`. This work has not been published previously and is not under consideration elsewhere.

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
    pli_s = d["pli_sum"].iloc[0]
    return f"""# Research Highlights

*Journal Target:* *{JOURNAL_NAME}* ({PUBLISHER_NAME})  
*Manuscript Title:* {EXACT_TITLE}

1. Open 7-cohort Python EEG benchmark (`564` unique subjects, `13,927` spectra, `105` 2D windows).
2. Unfiltered EDF audit shows Study 2 MDD confounds pre-export skirts with batch dates.
3. `GCN-OMST` (`{piv['GCN_OMST'].mean():.3f}`) & `GCN-WPLI` (`{piv['GCN_Weighted_PLI'].mean():.3f}`) equal random topology (`TOST p <= 0.012`).
4. `{pli_s['pct_indefinite_raw']:.2f}%` of `PLI` matrices are indefinite, while `19×19` spatial covariance tangent reaches `{piv['RandomForest'].mean():.3f}`/`0.724`.
5. Session mixing warps AUC (`-0.094` to `+0.080`); patient-only `OOF R²` never exceeds `0.035` (dementia `perm p >= 0.111`).
"""


def build_title_page(main_md: str) -> str:
    abs_block = main_md.split("## Abstract\n\n")[1].split("\n\n**Keywords:**")[0]
    kw_line = main_md.split("**Keywords:** ")[1].split("\n")[0]
    info_block = main_md.split("## Information Sharing Statement\n\n")[1].split("\n\n---")[0]
    decl_block = main_md.split("## Statements and Declarations\n\n")[1].split("\n\n---")[0]
    return f"""# Title Page

**Full Title:** {EXACT_TITLE}

**Short Running Title:** {SHORT_TITLE}

**Article Type:** Original Research / Resource Article

**Target Journal:** *{JOURNAL_NAME}* ({PUBLISHER_NAME})

**Author:** {AUTHOR_NAME}^1,2*^

**Affiliations:**  
^1^ {AFFIL_1}  
^2^ {AFFIL_2}  

**Corresponding Author:**  
{AUTHOR_NAME}  
{CORRESPONDENCE}

---

## Abstract

{abs_block}

**Keywords:** {kw_line}

---

## Information Sharing Statement

{info_block}

---

## Statements and Declarations

{decl_block}
"""


def build_declarations() -> str:
    return f"""# Statements, Ethical Approval, Data/Code Availability, and CRediT Declarations

**Manuscript Title:** *{EXACT_TITLE}*  
**Target Journal:** *{JOURNAL_NAME}* ({PUBLISHER_NAME})  
**Author:** {AUTHOR_NAME}^1,2*^  
^1^ {AFFIL_1}  
^2^ {AFFIL_2}  
^*^ Correspondence: {CORRESPONDENCE}

---

## 1. Ethics Approval and Consent to Participate
This computational resource study performed secondary analysis exclusively on de-identified, publicly accessible open-access EEG archives and involved no direct human-subjects recruitment or intervention. Ethical approval and written informed consent for original data collection were obtained by the respective primary investigators: Southwest University (`H20039`, `ds004902`), Universiti Sains Malaysia (`USM/JEPeM/1401015`, Figshare `4244171`), University of New Mexico IRB (`ds003478` and `ds003490`), UC San Diego HRPP (`ds002778`), AHEPA University Hospital (`142/12-04-2023`, `ds004504`), and University of Iowa IRB (`201707828`, `ds004584`).

## 2. Consent for Publication
Not applicable (secondary computational analysis of de-identified public archives; no identifiable participant data or images appear).

## 3. Competing Interests
The author declares no competing financial or non-financial interests.

## 4. Funding
This research received no specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## 5. Data and Code Availability
All raw EEG datasets are hosted permanently at OpenNeuro and Figshare. Complete Python source code (`scripts/47_run_v5_forensics_and_refinements.py`, `scripts/42_generate_all_specparam_figures.py`, `scripts/48_build_paper2_neuroinformatics_v5.py`, `src/harmonizer_and_graph.py`), derived features (`release_artifacts/derived_session_features_733.csv`), CV splits, and pinned package dependencies (`requirements_lock.txt`) are released openly at `{GITHUB_URL}` under the MIT license.

## 6. Author Contributions (CRediT)
**Chushu Tan:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Resources, Data Curation, Writing – Original Draft, Writing – Review & Editing, Visualization, Project administration.

## 7. AI-Assisted Coding and Editorial Verification Disclosure
During manuscript and code preparation, the author used AI-assisted software engineering and language-editing tools to draft analysis scripts, format statistical tables, and polish English phrasing. The author verified every numerical result, figure, statistical test, and citation directly against raw pipeline outputs and assumes full responsibility for all article content.

**Signed:** Chushu Tan  
**Date:** {SUBMISSION_DATE}
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
    return f"""# Springer *Neuroinformatics* Submission Checklist & Guide (v5 Remediated)

**Journal:** *{JOURNAL_NAME}* (Springer Nature, Editorial Manager)  
**Target Editor:** {EDITOR_NAME}  
**Date:** {SUBMISSION_DATE}

| File in `submission_Neuroinformatics/` | Editorial Manager Slot | Notes |
| :--- | :--- | :--- |
| `01_Cover_Letter_Neuroinformatics.docx` | Cover Letter | Includes open resource summary & dataset overlap disclosure |
| `02_Research_Highlights_Neuroinformatics.docx` | Author Highlights | 5 concise findings matching v5 CSVs |
| `03_Title_Page_Neuroinformatics.docx` | Title Page | Single-paragraph Abstract ({abs_wc} <= 250 words), `Beijing` correspondence, full Declarations |
| `04_Main_Manuscript_Neuroinformatics.docx` | Manuscript | Continuous line numbers, 3-line tables, Author–Year citations, Online Resource 1 refs |
| `05_Supplementary_Material_Neuroinformatics.docx` | Online Resource 1 | Sections S1.1–S1.4, Figures S1–S6, Tables S1–S27 |
| `06_Declarations_CRediT_COI_Ethics.docx` | Declarations Attachment | Ethics (no 45 CFR 46), Consent, Funding, COI, Data/Code Availability, AI Verification Disclosure |
| `07_Suggested_Reviewers.docx` | Suggested Reviewers | Voytek, Dimitriadis, Gramfort, Delorme |
| `Figures/Main_Figures/Figure_1..4.(png/pdf)` | Figure Files | 300 DPI |
| `Figures/Supplementary_Figures/fig_S1..S6.(png/pdf)` | Online Resource Figures | 300 DPI |

### Pre-Submission Audits Verified
- **Abstract Word Count:** {abs_wc} words (`<= 250` Springer limit).
- **Author–Year References:** 56 entries sorted alphabetically with `https://doi.org/...` links, verified for 100% in-text bijection.
- **DOCX Author / Modifier Properties:** Scrubbed clean across all `.docx` outputs.
"""


# ==============================================================================
# Strict Automated v5 Verification Audits
# ==============================================================================
def run_strict_audits(main_md: str, supp_md: str, doc_paths: list[Path]):
    print("\n🔍 Running strict Neuroinformatics v5 audits...")
    for p in doc_paths:
        with zipfile.ZipFile(p, "r") as z:
            xc = 0
            for name in z.namelist():
                if name.endswith((".xml", ".rels")):
                    ET.fromstring(z.read(name))
                    xc += 1
        d_obj = docx.Document(p)
        assert d_obj.core_properties.author == "", f"{p.name} has non-empty author!"
        assert d_obj.core_properties.last_modified_by == "", f"{p.name} has non-empty modifier!"
        print(f"   ✓ {p.name}: {xc} XML parts valid, blank metadata ({len(d_obj.paragraphs)} paras, {len(d_obj.tables)} tables)")

    with zipfile.ZipFile(MAIN_DOCX, "r") as z:
        assert "xmlns:m=" in z.read("word/settings.xml").decode("utf-8")
        assert "lnNumType" in z.read("word/document.xml").decode("utf-8")

    # Author-Year reference & bijection check
    body_text = main_md.split("## References")[0]
    refs_section = main_md.split("## References")[1]
    assert not re.search(r"^\[\d+\]\s", refs_section, flags=re.M), "Found leftover numbered [N] entry in References!"
    # Check alphabetical ordering by surname key
    ref_lines = [l.strip() for l in refs_section.strip().splitlines() if l.strip()]
    assert len(ref_lines) == len(REFS_ALPHA), f"Expected {len(REFS_ALPHA)} refs, got {len(ref_lines)}"

    # Verify 100% Author-Year bijection:
    # 1) Every key in REFS_ALPHA must appear in Main Text (matching exact key e.g. 'Anjum et al., 2024' or narrative 'Anjum et al. (2024)')
    uncited = []
    for key, _ in REFS_ALPHA:
        surname_part, yr = [x.strip() for x in key.rsplit(",", 1)]
        pat1 = re.escape(key)
        pat2 = rf"{re.escape(surname_part)}\s*\({re.escape(yr)}\)"
        if "&" in surname_part:
            narr = surname_part.replace("&", "and")
            pat3 = rf"{re.escape(narr)}\s*\({re.escape(yr)}\)"
        else:
            pat3 = pat2
        if not (re.search(pat1, body_text) or re.search(pat2, body_text) or re.search(pat3, body_text)):
            uncited.append(key)
    assert not uncited, f"Uncited references in REFS_ALPHA: {uncited}"

    # 2) Every parenthetical or narrative Author (Year) citation in body_text must exist in REFS_ALPHA
    ref_key_set = {k for k, _ in REFS_ALPHA} | {k.replace(" & ", " and ") for k, _ in REFS_ALPHA}
    found_cites = set()
    # Parenthetical multi-citations: (A, 2020; B et al., 2021a)
    for grp_m in re.finditer(r"\(([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+(?:\s+(?:da\s+Silva|et\s+al\.|&\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+|and\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+))?,\s*(?:19|20)\d{2}[a-z]?(?:;\s*[A-Z][^)]+)*)\)", body_text):
        for piece in grp_m.group(1).split(";"):
            piece = piece.strip()
            if re.search(r",\s*(?:19|20)\d{2}[a-z]?$", piece):
                found_cites.add(piece)
    # Narrative citations: Author (YYYY) or Author et al. (YYYY)
    for narr_m in re.finditer(r"([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+(?:\s+(?:da\s+Silva|et\s+al\.|(?:and|&)\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+))?)\s*\(((?:19|20)\d{2}[a-z]?)\)", body_text):
        found_cites.add(f"{narr_m.group(1).strip()}, {narr_m.group(2).strip()}")
    missing_from_refs = sorted(c for c in found_cites if c not in ref_key_set)
    assert not missing_from_refs, f"In-text Author-Year citations missing from REFS_ALPHA: {missing_from_refs}"

    # 3) Verify no leftover numeric [1]..[60] citations in prose
    cleaned_body = re.sub(r"`[^`]+`", "", body_text)
    num_cits = re.findall(r"(?<![A-Za-z0-9_])\[(\d{1,2})\](?!\s*Hz)", cleaned_body)
    assert not num_cits, f"Found leftover numeric citations in main text: {num_cits}"
    print(f"   ✓ Author-Year References: {len(REFS_ALPHA)}/{len(REFS_ALPHA)} alphabetical entries <-> {len(found_cites)} verified in-text citations (100% bijection)!")

    # Voice, banned hype words, <=2 rather than, 0 inline bold/italics
    combined = main_md + "\n" + supp_md
    plural_hits = re.findall(r"\b(we|our|ours|ourselves|us)\b", combined, flags=re.IGNORECASE)
    assert not plural_hits, f"Found plural pronouns: {plural_hits}"
    for w in ["crucially", "notably", "in stark contrast", "pivotal", "profound", "profoundly", "genuine", "indispensable", "catastrophic", "overwhelmingly", "stark", "statistically indistinguishable"]:
        hits = re.findall(rf"\b{re.escape(w)}\b", combined, flags=re.IGNORECASE)
        assert not hits, f"Found banned word/phrase '{w}': {hits}"
    rather_cnt = len(re.findall(r"\brather than\b", main_md, flags=re.IGNORECASE))
    assert rather_cnt <= 2, f"'rather than' count {rather_cnt} exceeds <= 2!"

    for line in body_text.splitlines():
        st = line.strip()
        if not st or st.startswith(("#", "|", "![", "*", "-", "---", "$$")) or re.match(r"^\d+\.\s", st) or st.startswith("**") or re.match(r"^\^\d|^\^\*", st):
            continue
        assert "**" not in st, f"Inline bold ** in prose: {st[:80]}"
        cleaned = re.sub(r"`[^`]*`", "", re.sub(r"\$\$?[^$]*\$\$?", "", st))
        if re.search(r"(^|[\s(])\*[^*\s][^*]*\*", cleaned):
            raise AssertionError(f"Inline italics *...* in prose: {st[:80]}")
    print(f"   ✓ Style audit: 0 plural pronouns, 0 banned words, rather than={rather_cnt} (<=2), 0 inline bold/italics")

    # Stale string & Online Resource completeness guard
    for p in doc_paths:
        tx = "\n".join(par.text for par in docx.Document(p).paragraphs)
        for bad in ["12246", "102249", "18008428969", "0.902", "27.33", "45 CFR 46", "Grass Telefactor", "derived_session_features_737"]:
            assert bad not in tx, f"{p.name} contains stale string '{bad}'!"
    assert "Nihon Kohden" not in body_text.split("## References")[0], "Check Study 2 hardware phrasing in Main Text!"
    supp_table_ids = re.findall(r"^### Table (S\d+[a-z]?)\.", supp_md, flags=re.M)
    unused_tables = [tid for tid in supp_table_ids if not re.search(rf"\b{re.escape(tid)}\b", main_md)]
    assert not unused_tables, f"Unused Supplementary tables in Main Text: {unused_tables}"
    supp_fig_ids = [f"Figure S{i}" for i in range(1, 7)]
    unused_figs = [fid for fid in supp_fig_ids if fid not in main_md]
    assert not unused_figs, f"Unused Supplementary figures in Main Text: {unused_figs}"
    print(f"   ✓ Online Resource 1 bijection: {len(supp_table_ids)}/{len(supp_table_ids)} tables (S1–S27) and 6/6 figures cited in Main Text!")


# ==============================================================================
# Main Pipeline
# ==============================================================================
def main():
    print("=" * 80)
    print("Building Paper 2 (Springer Neuroinformatics v5) From Verified Forensic Outputs")
    print("=" * 80)

    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    (SUBMISSION_DIR / "Figures" / "Main_Figures").mkdir(parents=True, exist_ok=True)
    (SUBMISSION_DIR / "Figures" / "Supplementary_Figures").mkdir(parents=True, exist_ok=True)

    for idx in range(1, 5):
        for ext in ("png", "pdf"):
            shutil.copy2(REPORTS_DIR / f"Figure_{idx}.{ext}", SUBMISSION_DIR / "Figures" / "Main_Figures" / f"Figure_{idx}.{ext}")
    for sf in ["fig_S1_specparam_example", "fig_S2_omst_topology", "fig_S3_gain_invariance", "fig_S4_cohort_spectra", "fig_S5_enrollment_flowchart", "fig_S6_permutation_tests"]:
        for ext in ("png", "pdf"):
            shutil.copy2(REPORTS_DIR / f"{sf}.{ext}", SUBMISSION_DIR / "Figures" / "Supplementary_Figures" / f"{sf}.{ext}")
    print("✓ Copied Figures 1–4 and S1–S6 (.png & .pdf) into submission_Neuroinformatics/Figures/")

    d = load_all_benchmark_data()
    main_md = build_main_manuscript(d)
    supp_md = build_supplementary_material(d)

    MAIN_MD.write_text(main_md, encoding="utf-8")
    SUPP_MD.write_text(supp_md, encoding="utf-8")
    md_to_docx(MAIN_MD, MAIN_DOCX, is_main=True)
    md_to_docx(SUPP_MD, SUPP_DOCX, is_main=False)
    print(f"✓ Generated {MAIN_MD.name} & {MAIN_DOCX.name}")
    print(f"✓ Generated {SUPP_MD.name} & {SUPP_DOCX.name}")

    files_to_build = [
        ("01_Cover_Letter_Neuroinformatics", build_cover_letter(d), False),
        ("02_Research_Highlights_Neuroinformatics", build_highlights(d), False),
        ("03_Title_Page_Neuroinformatics", build_title_page(main_md), False),
        ("04_Main_Manuscript_Neuroinformatics", main_md, True),
        ("05_Supplementary_Material_Neuroinformatics", supp_md, False),
        ("06_Declarations_CRediT_COI_Ethics", build_declarations(), False),
        ("07_Suggested_Reviewers", build_suggested_reviewers(), False),
    ]
    generated_docx = [MAIN_DOCX, SUPP_DOCX]
    for stem, content, is_main_flag in files_to_build:
        mdp = SUBMISSION_DIR / f"{stem}.md"
        docxp = SUBMISSION_DIR / f"{stem}.docx"
        mdp.write_text(content, encoding="utf-8")
        md_to_docx(mdp, docxp, is_main=is_main_flag)
        generated_docx.append(docxp)
        print(f"✓ Generated {stem} (.md, .docx)")

    chk_p = SUBMISSION_DIR / "08_Submission_Checklist_Neuroinformatics.md"
    chk_p.write_text(build_submission_checklist(main_md), encoding="utf-8")
    print(f"✓ Generated {chk_p.name}")

    run_strict_audits(main_md, supp_md, generated_docx)

    if ARTIFACT_DIR.exists():
        mirror_map = [
            (MAIN_MD, ARTIFACT_DIR / "PAPER_2_NEUROINFORMATICS_v5.md"),
            (MAIN_DOCX, ARTIFACT_DIR / "PAPER_2_NEUROINFORMATICS_v5.docx"),
            (SUPP_MD, ARTIFACT_DIR / "PAPER_2_SUPPLEMENTARY_MATERIAL_v5.md"),
            (SUPP_DOCX, ARTIFACT_DIR / "PAPER_2_SUPPLEMENTARY_MATERIAL_v5.docx"),
            (SUBMISSION_DIR / "01_Cover_Letter_Neuroinformatics.docx", ARTIFACT_DIR / "Cover_Letter_Neuroinformatics_v5.docx"),
            (SUBMISSION_DIR / "02_Research_Highlights_Neuroinformatics.docx", ARTIFACT_DIR / "Highlights_Neuroinformatics_v5.docx"),
            (SUBMISSION_DIR / "03_Title_Page_Neuroinformatics.docx", ARTIFACT_DIR / "Title_Page_Neuroinformatics_v5.docx"),
            (REPORTS_DIR / "Figure_1.png", ARTIFACT_DIR / "Paper2_v5_Figure_1.png"),
            (REPORTS_DIR / "Figure_2.png", ARTIFACT_DIR / "Paper2_v5_Figure_2.png"),
            (REPORTS_DIR / "Figure_3.png", ARTIFACT_DIR / "Paper2_v5_Figure_3.png"),
            (REPORTS_DIR / "Figure_4.png", ARTIFACT_DIR / "Paper2_v5_Figure_4.png"),
            (REPORTS_DIR / "fig_S5_enrollment_flowchart.png", ARTIFACT_DIR / "Paper2_v5_Figure_S5_Flowchart.png"),
            (REPORTS_DIR / "fig_S6_permutation_tests.png", ARTIFACT_DIR / "Paper2_v5_Figure_S6_Permutations.png"),
        ]
        for s_p, d_p in mirror_map:
            shutil.copy2(s_p, d_p)
        print(f"✓ Mirrored {len(mirror_map)} core v5 deliverables to Artifact Directory.")

    print("=" * 80)
    print("ALL PAPER 2 NEUROINFORMATICS v5 DELIVERABLES BUILT AND AUDITED CLEANLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
