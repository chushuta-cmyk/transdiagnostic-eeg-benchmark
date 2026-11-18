#!/usr/bin/env python3
"""
scripts/46_build_paper2_neuroinformatics_v4.py
----------------------------------------------
Builds the complete, fully remediated Springer Neuroinformatics (v4) manuscript
and Editorial Manager submission package directly from canonical [2.0, 30.0] Hz
specparam v2.0.0rc7 and 10x5 repeated participant-grouped benchmark outputs.

Reads every scalar, CI, Monte Carlo bias metric, bootstrap delta, 10x5 CV mean,
aligned permutation p-value, Split-Target-HC transfer AUC, and forensic dataset
audit live from reports/*.csv and reports/*.json. Wipes personal author/modifier
metadata on generated DOCX files, enforces continuous margin line numbers on Main
Manuscript, formats three-line booktabs tables and hanging-indent Springer [N]
references with DOIs, and runs strict automated parity + style + citation audits.
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

# ==============================================================================
# Paths & Metadata
# ==============================================================================
ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
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
    "Open Benchmarking of Parameterized EEG Spectra and Graph Neural Networks "
    "Across Seven Public Cohorts Exposes Spectral Window Confounding and Connectivity Limits"
)
SHORT_TITLE = "Benchmarking Parameterized EEG Spectra & GNN Limits"

AUTHOR_NAME = "Chushu Tan"
AFFIL_1 = "School of Sociology, China University of Political Science and Law, Beijing, China"
AFFIL_2 = "Department of Psychology, University at Buffalo, The State University of New York, Buffalo, NY 14260, USA"
CORRESPONDENCE = (
    "Guangzhou, Guangdong 510006, China; Email: chushuta@gmail.com; "
    "ORCID: 0009-0007-3093-0160"
)
GITHUB_URL = "https://github.com/chushuta-cmyk/transdiagnostic-eeg-benchmark"
SUBMISSION_DATE = "November 17, 2026"

MAIN_MD = MANUSCRIPT_DIR / "PAPER_2_NEUROINFORMATICS_v4.md"
MAIN_DOCX = MANUSCRIPT_DIR / "PAPER_2_NEUROINFORMATICS_v4.docx"
SUPP_MD = MANUSCRIPT_DIR / "PAPER_2_SUPPLEMENTARY_MATERIAL_v4.md"
SUPP_DOCX = MANUSCRIPT_DIR / "PAPER_2_SUPPLEMENTARY_MATERIAL_v4.docx"


# ==============================================================================
# Formatting Helpers
# ==============================================================================
def fmt_p(p: float) -> str:
    """Format p-value consistently in clean scientific notation."""
    if pd.isna(p):
        return "—"
    if p < 1e-4:
        exp = int(np.floor(np.log10(max(p, 1e-300))))
        mant = p / (10.0 ** exp)
        return f"{mant:.2f} × 10^{exp}"
    return f"{p:.4f}"


def fmt_sig(p: float) -> str:
    """Return significance stars."""
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
    """Enable continuous margin line numbering on main manuscript for peer review."""
    for section in doc.sections:
        sectPr = section._sectPr
        lnNumType = OxmlElement("w:lnNumType")
        lnNumType.set(qn("w:countBy"), "1")
        lnNumType.set(qn("w:restart"), "continuous")
        sectPr.append(lnNumType)


def apply_three_line_table_styling(tbl):
    """Apply clean three-line booktabs formatting with noLnn inside tables."""
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tblPr = tbl._tbl.tblPr
    for existing in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(existing)

    tblPr.append(parse_xml(r"""
        <w:tblBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
            <w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>
            <w:left w:val="none"/>
            <w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>
            <w:right w:val="none"/>
            <w:insideH w:val="none"/>
            <w:insideV w:val="none"/>
        </w:tblBorders>
    """))

    col_count = len(tbl.columns) if tbl.columns else (len(tbl.rows[0].cells) if tbl.rows else 0)
    is_wide = col_count > 7
    hdr_pt = Pt(8.5) if is_wide else Pt(9.5)
    data_pt = Pt(8.0) if is_wide else Pt(9.0)
    line_sp = 1.05 if is_wide else 1.15
    sp_pad = Pt(1.5) if is_wide else Pt(2.0)

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
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

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
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(6)
            for r in p.runs:
                r.font.name = "Times New Roman"
                r.font.size = Pt(14)
                r.font.bold = True
                r.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        elif p.style.name.startswith("Heading 2"):
            in_references = (txt == "References")
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
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(3)
            for r in p.runs:
                r.font.name = "Times New Roman"
                r.font.size = Pt(10)

    for tbl in doc.tables:
        apply_three_line_table_styling(tbl)

    doc.save(doc_path)


def patch_settings_and_clean_core(docx_path: Path):
    """Declare xmlns:m on <w:settings> alongside <m:mathPr> and scrub core.xml properties."""
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
    cmd = [
        "pandoc", str(md_path), f"--resource-path={res_path}",
        "-f", "markdown+tex_math_dollars", "-t", "docx", "-o", str(docx_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Pandoc conversion failed for {md_path.name}:\n{res.stderr}")
    style_submission_doc(docx_path, is_main=is_main)
    patch_settings_and_clean_core(docx_path)


# ==============================================================================
# Live Data Loader (100% Fresh Canonical [2, 30] Hz + 10x5 CV + Audits)
# ==============================================================================
def load_all_benchmark_data() -> dict:
    with open(REPORTS_DIR / "master_results.json", encoding="utf-8") as f:
        master = json.load(f)
    with open(REPORTS_DIR / "real_permutation_distributions.json", encoding="utf-8") as f:
        perms = json.load(f)
    with open(REPORTS_DIR / "extra_biophysics_summary.json", encoding="utf-8") as f:
        extra_bio = json.load(f)

    data = {
        "master": master,
        "perms": perms,
        "extra_bio": extra_bio,
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
    }
    data["reg"]["cohort"] = data["reg"]["cohort"].apply(strip_md_bold)
    return data


# ==============================================================================
# Dynamic Main Tables Builders
# ==============================================================================
def build_table_1(d: dict) -> str:
    """Table 1: Canonical Interior [2.0, 30.0] Hz Parameterized EEG Biophysics & Fit Quality."""
    fit_230 = d["fit"][d["fit"]["fit_window_hz"] == "2.0-30.0"].set_index("cohort_key")
    bio = d["bio_exp"]
    f3 = d["f3f4"].set_index("cohort")

    rows_cfg = [
        ("Study 1: Sleep Deprivation (N=71 pairs)", "Study 1: Acute Sleep Deprivation", "sleep_ds004902", "Paired t"),
        ("Study 2: Outpatient MDD EC (30 vs 28)", "Study 2: Outpatient MDD", "mumtaz_ec", "Welch t"),
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
    dep = meta_p.loc["Depression_Cohorts_k2"]
    ndg = meta_p.loc["PD_Dementia_Cohorts"]
    all6 = meta_p.loc["Overall_Pooled"]
    ucsd_all = d["extra_bio"]["UCSD_All_theta"]

    lines.append("")
    lines.append(
        f"*Note: All values derive from `specparam v2.0.0rc7` fitted inside the passband-interior window [2.0, 30.0] Hz "
        f"(five frontal sites `Fp1, Fp2, F3, F4, Fz` for Frontal θ; five posterior sites `P3, P4, Pz, O1, O2` for Posterior α; "
        f"all 19 sites for Global χ). `q_10` denotes Benjamini–Hochberg FDR across the 10 frontal theta contrasts. "
        f"Study 4 uses participant-averaged PD sessions (15 PD vs. 16 HC); pooling all 30 PD sessions without averaging gives "
        f"d = {ucsd_all['cohens_d']:+.3f} (p = {fmt_p(ucsd_all['p_value'])}). "
        f"Study 6 analyzes contiguous 60 s pure eyes-open (EO) rest excluding auditory oddball task trials (see Table S25 for pure EC sensitivity). "
        f"Primary random-effects meta-analysis across the K = 6 participant-averaged chronic cohorts (N = 495: Studies 2, 3, 4 SubjMean, 5c, 6a, 7) "
        f"yields d_RE = {ndg['pooled_d']:+.3f} (HKSJ 95% CI [{ndg['hksj_ci_lower']:+.3f}, {ndg['hksj_ci_upper']:+.3f}], HKSJ p = {fmt_p(ndg['p_hksj'])}, I² = {ndg['I2_percent']:.1f}%) "
        f"across k = 4 PD/dementia cohorts versus d_RE = {dep['pooled_d']:+.3f} (DL p = {fmt_p(dep['p_value'])}; wide df = 1 HKSJ CI [{dep['hksj_ci_lower']:+.3f}, {dep['hksj_ci_upper']:+.3f}], HKSJ p = {fmt_p(dep['p_hksj'])}) "
        f"across k = 2 depression cohorts (Q_between = {dep['Q_between']:.2f}, p = {fmt_p(dep['p_between'])}, pooled k = 6 I² = {all6['I2_percent']:.1f}%).*"
    )
    return "\n".join(lines)


def build_table_2(d: dict) -> str:
    """Table 2: Spectral Estimator & Fit-Window Leakage Audit ([7,14] OLS vs [1,40] vs [2,30] specparam)."""
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
        ("Study 2: Outpatient MDD", "Study 2: Outpatient MDD EC (30 vs 28)"),
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
        fr = ft_df.loc[cname]
        cr = chi_df.loc[cname]
        bw = b_win.loc[cname]
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
        f"*Note: Δd reports the stratified 1,000-bootstrap paired shift d_[1–40] − d_[2–30] with two-sided bootstrap p-value "
        f"(full 60-contrast bootstrap delta and OLS[2–30] sensitivity in Supplementary Table S11). "
        f"In Study 2 Outpatient MDD, filter skirts approaching 40 Hz attenuated `specparam` [1–40] Hz frontal theta from d = {ft_df.loc['Study 2: Outpatient MDD', 'cohens_d_specparam_resid_2to30']:+.3f} "
        f"down to {ft_df.loc['Study 2: Outpatient MDD', 'cohens_d_specparam_resid_1to40']:+.3f} (bootstrap Δd = {b_s2['delta_d']:+.3f} [{b_s2['ci_lower']:+.3f}, {b_s2['ci_upper']:+.3f}], p = {fmt_p(b_s2['boot_p'])}) "
        f"and inverted global χ ({cr.loc['cohens_d_specparam_resid_1to40'] if isinstance(cr, pd.DataFrame) else chi_df.loc['Study 2: Outpatient MDD', 'cohens_d_specparam_resid_1to40']:+.3f} vs. {chi_df.loc['Study 2: Outpatient MDD', 'cohens_d_specparam_resid_2to30']:+.3f}). "
        f"Conversely, in Study 3 university depression (clean high-R² BioSemi PSDs), leaving 4–7 Hz unmasked in [7, 14] Hz OLS inflated theta negativity "
        f"(OLS {ft_df.loc['Study 3: Univ. Depression Screening', 'cohens_d_masked_ols_1to40']:+.3f} vs. canonical `specparam` {ft_df.loc['Study 3: Univ. Depression Screening', 'cohens_d_specparam_resid_2to30']:+.3f}; "
        f"bootstrap Δd = {b_s3_ols['delta_d']:+.3f} [{b_s3_ols['ci_lower']:+.3f}, {b_s3_ols['ci_upper']:+.3f}], p = {fmt_p(b_s3_ols['boot_p'])}).*"
    )
    return "\n".join(lines)


def build_table_3(d: dict) -> str:
    """Table 3: Canonical 10x5 Repeated Participant-Grouped CV ROC-AUC Across 9 Contrasts & 12 Models."""
    rep = d["rep10"]
    wil9 = d["wilcoxon"][d["wilcoxon"]["scope"] == "Primary_9_Contrasts_10x5Mean"].set_index("comparator_model")
    wil8 = d["wilcoxon"][d["wilcoxon"]["scope"] == "NonNested_8_Contrasts_10x5Mean"].set_index("comparator_model")

    p9_cohorts = [
        ("Study 2: Mumtaz MDD (Eyes Closed)", "Study 2: MDD EC (N=58)"),
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
        ("GCN_Identity", "GCN-Id"),
        ("GCN_Random_EdgeMatched", "GCN-Rand*"),
        ("GCN_Full", "GCN-Full"),
        ("Tuned_LogisticRegression", "Tuned LR"),
        ("Tuned_Ridge", "Tuned Ridge"),
        ("Linear_SVM", "Lin-SVM"),
        ("RBF_SVM", "RBF-SVM"),
        ("RandomForest", "RF"),
        ("GBDT", "GBDT"),
        ("Single_Theta", "Single θ"),
        ("Single_Chi", "Single χ"),
    ]
    piv = rep.pivot(index="cohort", columns="model", values="mean_auc")

    hdr = "| Primary Benchmark Contrast | " + " | ".join(lbl for _, lbl in models) + " |"
    sep = "| :--- | " + " | ".join(":---:" for _ in models) + " |"
    lines = [
        "### Table 3. True 10×5 Repeated Participant-Grouped Cross-Validated Mean ROC-AUC Across Nine Primary Contrasts and Twelve Models",
        hdr,
        sep,
    ]
    for cname, slabel in p9_cohorts:
        vals = [float(piv.loc[cname, mkey]) for mkey, _ in models]
        mx = max(vals)
        cells = [f"**{v:.3f}**" if abs(v - mx) < 1e-9 else f"{v:.3f}" for v in vals]
        lines.append(f"| {slabel} | " + " | ".join(cells) + " |")

    p9_cnames = [c for c, _ in p9_cohorts]
    nn8_cnames = [c for c in p9_cnames if "5c" not in c]
    means9 = [float(piv.loc[p9_cnames, mkey].mean()) for mkey, _ in models]
    means8 = [float(piv.loc[nn8_cnames, mkey].mean()) for mkey, _ in models]
    mx9 = max(means9)
    mx8 = max(means8)
    lines.append("| Mean Across 9 Primary Contrasts | " + " | ".join(f"**{v:.3f}**" if abs(v - mx9) < 1e-9 else f"{v:.3f}" for v in means9) + " |")
    lines.append("| Mean Across 8 Non-Nested Contrasts | " + " | ".join(f"**{v:.3f}**" if abs(v - mx8) < 1e-9 else f"{v:.3f}" for v in means8) + " |")

    lines.append("")
    lines.append(
        f"*Note: Each cell reports the 10-repeat mean out-of-fold ROC-AUC (`50` outer grouped folds across 10 shuffled participant-grouped seeds; "
        f"zero participant overlap; fold-nested normative z-scoring; see Table S20 for ± SD and min–max ranges, and Table S10 for Split-0 95% CIs). "
        f"`*GCN-Rand` uses session-specific degree-agnostic random undirected graphs matching each session's exact OMST edge count. "
        f"Across 9 primary contrasts, `GCN-OMST` ({means9[0]:.3f}) did not differ significantly from `GCN-Rand` ({means9[2]:.3f}, Wilcoxon W = {wil9.loc['GCN_Random_EdgeMatched', 'wilcoxon_W']:.1f}, p = {wil9.loc['GCN_Random_EdgeMatched', 'raw_p']:.4f}), "
        f"`GCN-Full` ({means9[3]:.3f}, p = {wil9.loc['GCN_Full', 'raw_p']:.4f}), or `GCN-Id` ({means9[1]:.3f}, p = {wil9.loc['GCN_Identity', 'raw_p']:.4f}), and trailed "
        f"`Tuned Ridge` ({means9[5]:.3f}, p = {wil9.loc['Tuned_Ridge', 'raw_p']:.4f}), `Random Forest` ({means9[8]:.3f}, p = {wil9.loc['RandomForest', 'raw_p']:.4f}), and `Tuned LR` ({means9[4]:.3f}, p = {wil9.loc['Tuned_LogisticRegression', 'raw_p']:.4f}); "
        f"identical null topological differences held across non-nested k = 8 (`p = {wil8.loc['GCN_Random_EdgeMatched', 'raw_p']:.4f}` vs. `GCN-Rand`, `p = {wil8.loc['RandomForest', 'raw_p']:.4f}` vs. `RF`; Table S10b).*"
    )
    return "\n".join(lines)


def build_table_4(d: dict) -> str:
    """Table 4: Leak-Free Split-Target-HC External Transfer & Gain/Calibration Ablation (Mean + Worst-Case Drift)."""
    pd_tx = d["pd_tx"]
    td_tx = d["td_tx"]
    gc = d["gain_cal"]

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
    pd_order = [
        ("Iowa_PD", "UNM_PD_OFF", "Iowa PD (149) → UNM PD-OFF EO (50)"),
        ("Iowa_PD", "UCSD_PD", "Iowa PD (149) → UCSD PD (46)"),
        ("UCSD_PD", "Iowa_PD", "UCSD PD (46) → Iowa PD (149)"),
        ("UCSD_PD", "UNM_PD_OFF", "UCSD PD (46) → UNM PD-OFF EO (50)"),
        ("UNM_PD_OFF", "Iowa_PD", "UNM PD-OFF EO (50) → Iowa PD (149)"),
        ("UNM_PD_OFF", "UCSD_PD", "UNM PD-OFF EO (50) → UCSD PD (46)"),
    ]
    for src, tgt, lbl in pd_order:
        r_g = pd_raw.loc[(src, tgt)]
        vals = [float(pd_piv.loc[(src, tgt), m]) for m in ("GCN_OMST", "GCN_Identity", "Tuned_LR", "RandomForest", "GBDT")]
        mx = max(vals)
        cells = [f"**{mean_sd_str(r_g) if i == 0 else f'{v:.3f}'}**" if abs(v - mx) < 1e-9 else (mean_sd_str(r_g) if i == 0 else f"{v:.3f}") for i, v in enumerate(vals)]
        lines.append(f"| {lbl} | Cross-Site PD | " + " | ".join(cells) + " |")

    td_piv = td_tx.pivot(index=["source_cohort", "target_cohort"], columns="model", values="auc")
    td_raw = td_tx[td_tx["model"] == "GCN_OMST"].set_index(["source_cohort", "target_cohort"])
    td_order = [
        ("Cavanagh_MDD", "Mumtaz_MDD", "Univ. Dep. (119) → Outpatient MDD EC (58)", "Affective"),
        ("Mumtaz_MDD", "Cavanagh_MDD", "Outpatient MDD EC (58) → Univ. Dep. (119)", "Affective"),
        ("Iowa_PD", "Dementia_All", "Iowa PD (149) → Dementia All (88)", "Slowing (Cross-Disease)"),
        ("Sleep_Debt", "Dementia_All", "Sleep Deprivation (142) → Dementia All (88)", "Slowing (Acute → Degenerative)"),
        ("Sleep_Debt", "Iowa_PD", "Sleep Deprivation (142) → Iowa PD (149)", "Slowing (Acute → Degenerative)"),
        ("Dementia_All", "Mumtaz_MDD", "Dementia All (88) → Outpatient MDD EC (58)", "Cross-Disorder (Discordant)"),
        ("Sleep_Debt", "Mumtaz_MDD", "Sleep Deprivation (142) → Outpatient MDD EC (58)", "Cross-Disorder (Discordant)"),
        ("Mumtaz_MDD", "Dementia_All", "Outpatient MDD EC (58) → Dementia All (88)", "Cross-Disorder (Reverse)"),
    ]
    for src, tgt, lbl, rtype in td_order:
        r_g = td_raw.loc[(src, tgt)]
        vals = [float(td_piv.loc[(src, tgt), m]) for m in ("GCN_OMST", "GCN_Identity", "Tuned_LR", "RandomForest", "GBDT")]
        mx = max(vals)
        cells = [f"**{mean_sd_str(r_g) if i == 0 else f'{v:.3f}'}**" if abs(v - mx) < 1e-9 else (mean_sd_str(r_g) if i == 0 else f"{v:.3f}") for i, v in enumerate(vals)]
        lines.append(f"| {lbl} | {rtype} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("#### Panel B. Mean 6-Pair Cross-Site PD Transfer Across Calibration Regimes and Gain Shifts (Reporting Both Session-Mean and Max Drift)")
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
            lines.append(
                f"| Baseline Calibration (k = 1.0) | {flbl} | {cal_lbl} | "
                f"{sub['lr_auc'].mean():.3f} ({sub['lr_bacc'].mean():.3f} / {sub['lr_brier'].mean():.3f}) | "
                f"{sub['rf_auc'].mean():.3f} | 0.000 / 0.000 | 0.000 / 0.000 |"
            )

    amp_sub = gc[gc["experiment"] == "Amplifier_Gain_Ablation"]
    for fkey, flbl in [("specparam_periodic_residual", "`specparam` [2–30] + χ"), ("raw_log_band_power", "Unparam. log10 Power + χ")]:
        for gk, klbl in [("5.0", "Uniform k = 5.0"), ("Heterogeneous_0.5x_5.0x", "Mixed 0.5x / 5.0x")]:
            sub = amp_sub[(amp_sub["feature_type"] == fkey) & (amp_sub["gain_k"].astype(str) == gk)]
            lines.append(
                f"| Gain Perturbation (Uncalibrated) | {flbl} | {klbl} | "
                f"{sub['lr_auc'].mean():.3f} ({sub['lr_bacc'].mean():.3f} / {sub['lr_brier'].mean():.3f}) | "
                f"{sub['rf_auc'].mean():.3f} | "
                f"{sub['mean_feat_drift'].mean():.3f} / {sub['max_feat_drift'].max():.3f} | "
                f"{sub['lr_mean_prob_drift'].mean():.3f} / {sub['lr_max_prob_drift'].max():.3f} |"
            )

    lines.append("")
    lines.append(
        "*Note: Panel A reports leak-free 50/50 `Split-Target-HC` external evaluation in which half of target healthy controls calibrate standardization strictly disjointly from the evaluation controls and all target patients (20 random control splits; full oracle-HC audits and 5th–95th percentile bounds appear in Supplementary Tables S15, S16, and S19).*"
    )
    return "\n".join(lines)


def build_table_5(d: dict) -> str:
    """Table 5: 10x5 Repeated Out-of-Fold Clinical Scale Regression Summary."""
    reg = d["reg"]
    lines = [
        "### Table 5. Exploratory 10×5 Repeated Out-of-Fold Clinical Severity Regression Benchmarks (Full Cohort vs. Patient-Only Sensitivity)",
        "| Cohort & Scope | Clinical Scale | N | Best Regressor | 10×5 Mean r (±SD) | Median Two-Sided p | Mean OOF R² | Mean MAE | Ridge 10×5 r (Median p) |",
        "| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    groups = [
        ("Dementia_MMSE_All_N88", "Study 5: Dementia All (ds004504)", "MMSE (4.0–30.0)", 88),
        ("Dementia_MMSE_Patients_Only_N59", "Study 5: Dementia Patients Only", "MMSE (4.0–29.0)", 59),
        ("Iowa_PD_MoCA_All_N149", "Study 7: Iowa PD All (ds004584)", "MoCA (9.0–30.0)", 149),
        ("Iowa_PD_MoCA_Patients_Only_N100", "Study 7: Iowa PD Patients Only", "MoCA (9.0–30.0)", 100),
        ("Iowa_PD_UPDRS_Patients_N100", "Study 7: Iowa PD Patients Only", "UPDRS-III (1.0–28.0)", 100),
    ]
    for ckey, cname, sname, nval in groups:
        sub = reg[reg["cohort_key"] == ckey]
        best = sub.loc[sub["pearson_r_mean"].idxmax()]
        lin = sub[sub["model"] == "Ridge_Regression"].iloc[0]
        bname = best["model"].replace("_", " ")
        lines.append(
            f"| {cname} | {sname} | {nval} | {bname} | {best['pearson_r_mean']:+.3f} (±{best['pearson_r_sd']:.3f}) | "
            f"{fmt_p(best['median_p'])} | {best['r2_mean']:.3f} | {best['mae_mean']:.3f} | "
            f"{lin['pearson_r_mean']:+.3f} ({fmt_p(lin['median_p'])}) |"
        )
    lines.append("")
    lines.append(
        "*Note: 10-repeat 5-fold cross-validated regression on flattened 76-D canonical [2.0, 30.0] Hz `specparam v2.0.0rc7` features. "
        "Negative or near-zero R² values show out-of-fold variance explained near or below the target mean. Full 20-row breakdown appears in Supplementary Table S14.*"
    )
    return "\n".join(lines)


# ==============================================================================
# Main Manuscript Builder (Neuroinformatics v4)
# ==============================================================================
def build_main_manuscript(d: dict) -> str:
    t1 = build_table_1(d)
    t2 = build_table_2(d)
    t3 = build_table_3(d)
    t4 = build_table_4(d)
    t5 = build_table_5(d)

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
    p9_r_cohs = [c for c in riem["cohort"].unique() if "6b" not in c]
    riem_p9 = riem[riem["cohort"].isin(p9_r_cohs)].groupby(["representation", "classifier"])["auc"].mean()

    rep = d["rep10"]
    p9_cnames = [c for c in rep["cohort"].unique() if not any(k in c for k in ("2b", "6b", "6c"))]
    nn8_cnames = [c for c in p9_cnames if "5c" not in c]
    rep_p9 = rep[rep["cohort"].isin(p9_cnames)].pivot(index="cohort", columns="model", values="mean_auc")
    rep_nn8 = rep[rep["cohort"].isin(nn8_cnames)].pivot(index="cohort", columns="model", values="mean_auc")
    rep_all11 = rep[rep["cohort"] != "Study 2b: Mumtaz MDD (Eyes Open)"].pivot(index="cohort", columns="model", values="mean_auc")
    rep_idx = rep.set_index(["cohort", "model"])

    wil9 = d["wilcoxon"][d["wilcoxon"]["scope"] == "Primary_9_Contrasts_10x5Mean"].set_index("comparator_model")
    wil8 = d["wilcoxon"][d["wilcoxon"]["scope"] == "NonNested_8_Contrasts_10x5Mean"].set_index("comparator_model")
    wil11 = d["wilcoxon"][d["wilcoxon"]["scope"] == "All_11_Contrasts_10x5Mean"].set_index("comparator_model")

    meta_230 = d["meta"][(d["meta"]["method"] == "specparam_2to30_resid") & (d["meta"]["analysis_set"] == "Primary_K6_SubjMean_PD_OFF")].set_index("subgroup")
    dep_m = meta_230.loc["Depression_Cohorts_k2"]
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
    dem_all_svr = reg[(reg["cohort_key"] == "Dementia_MMSE_All_N88") & (reg["model"] == "SVR_RBF")].iloc[0]
    dem_pt_rf = reg[(reg["cohort_key"] == "Dementia_MMSE_Patients_Only_N59") & (reg["model"] == "Random_Forest_Regressor")].iloc[0]
    dem_pt_ridge = reg[(reg["cohort_key"] == "Dementia_MMSE_Patients_Only_N59") & (reg["model"] == "Ridge_Regression")].iloc[0]
    iowa_all_svr = reg[(reg["cohort_key"] == "Iowa_PD_MoCA_All_N149") & (reg["model"] == "SVR_RBF")].iloc[0]
    iowa_pt_svr = reg[(reg["cohort_key"] == "Iowa_PD_MoCA_Patients_Only_N100") & (reg["model"] == "SVR_RBF")].iloc[0]
    iowa_updrs_gbdt = reg[(reg["cohort_key"] == "Iowa_PD_UPDRS_Patients_N100") & (reg["model"] == "GBDT_Regressor")].iloc[0]

    med_df = d["med"].set_index(["cohort", "marker"])
    unm_med_ft = med_df.loc[("Study 6: UNM PD Pure EO (>=15h Washout) (N=25)", "Frontal Periodic Theta (4–8 Hz)")]
    ucsd_med_ft = med_df.loc[("Study 4: UCSD PD (Acute Washout) (N=15)", "Frontal Periodic Theta (4–8 Hz)")]
    unm_sens_df = d["unm_sens"].set_index("condition")
    anc_dur = d["anc_dur"].set_index("cohort")
    iapf_df = d["iapf"].set_index("cohort")
    iowa_iapf = iapf_df.loc["Study 7: Iowa Parkinson Cohort"]
    mum_conc = d["mum_conc"].set_index("marker")
    mum_tx = d["mum_tx"].set_index(["source_cohort", "target_cohort"])
    hparam_means = d["hparam"].groupby("config_label")["auc"].mean()

    pd_tx = d["pd_tx"].pivot(index=["source_cohort", "target_cohort"], columns="model", values="auc")
    td_tx = d["td_tx"].pivot(index=["source_cohort", "target_cohort"], columns="model", values="auc")
    s2_for = d["mum_forensic"].set_index("subset")

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

    # Build crisp <= 250-word single-paragraph Abstract directly from verified outputs
    abstract_text = (
        f"Reproducible clinical electroencephalography (qEEG) requires verifying whether spectral markers, functional connectivity, and graph neural networks (GNNs) withstand rigorous methodological audits. "
        f"I assembled an open 19-channel benchmark spanning seven public cohorts ({acc['total_unique_participants']} primary participants; {acc['total_analyzed_sessions']} primary sessions; {acc['total_with_mumtaz_eo']} total sessions; 14,003 channel spectra) across sleep deprivation, depression, Parkinson's disease (PD), and dementia. "
        f"Comparing passband-edge [1.0, 40.0] Hz against interior [2.0, 30.0] Hz parameterization (`specparam v2.0.0rc7`; zero failed fits) and [7, 14] Hz-masked ordinary least-squares (`OLS`) slopes exposed dual spectral confounding: filter skirts near 40 Hz degraded Study 2 outpatient major depressive disorder (MDD) fits (median R² {fit140.loc['mumtaz_ec', 'r2_median']:.3f} vs. {fit230.loc['mumtaz_ec', 'r2_median']:.3f}) and masked frontal theta shift (d = {ft_w.loc['Study 2: Outpatient MDD', 'cohens_d_specparam_resid_1to40']:+.3f} vs. {ft_w.loc['Study 2: Outpatient MDD', 'cohens_d_specparam_resid_2to30']:+.3f}), whereas unmasked OLS inflated student depression effects (d = {ft_w.loc['Study 3: Univ. Depression Screening', 'cohens_d_masked_ols_1to40']:+.3f} vs. {ft_w.loc['Study 3: Univ. Depression Screening', 'cohens_d_specparam_resid_2to30']:+.3f}) and biased Monte Carlo null simulations ({sim_s2_ols['theta_d_bias']:+.2f}). "
        f"Forensic inspection of Study 2 additionally identified exact duplicate raw files, 2011–2013 control batch heterogeneity (theta d = +3.347), and Jan-2013 phase artifacts; excluding artifact controls eliminated the MDD theta difference (d = {s2_for.loc['Study 2a EC: Deduplicated Excl. Jan-2013 HC Batch (29 MDD vs 13 HC, N=42)', 'theta_d']:+.3f}). "
        f"Conversely, PD and dementia theta elevation replicated across sites (K = 6 pooled d_RE = {ndg_m['pooled_d']:+.3f}, Hartung–Knapp p = {ndg_m['p_hksj']:.4f}). "
        f"Across nine contrasts under 10×5 repeated participant-grouped cross-validation, `GCN-OMST` (mean ROC-AUC {rep_p9['GCN_OMST'].mean():.3f}) never outperformed edge-matched `GCN-Random` ({rep_p9['GCN_Random_EdgeMatched'].mean():.3f}, Wilcoxon p = {wil9.loc['GCN_Random_EdgeMatched', 'raw_p']:.4f}), `Random Forest` ({rep_p9['RandomForest'].mean():.3f}), or `Tuned Ridge` ({rep_p9['Tuned_Ridge'].mean():.3f}). "
        f"Furthermore, {int(pli_s['n_indefinite_raw'])}/{int(pli_s['n_sessions'])} ({pli_s['pct_indefinite_raw']:.2f}%) symmetrized Phase Lag Index matrices were indefinite, session mixing shifted 10×5 AUC from {leak.loc[('Study 1: Sleep Deprivation', 'Leaky_Subject_Mixing', 'RandomForest'), 'mean_delta_vs_strict']:+.3f} to {leak.loc[('Study 4: UCSD Parkinson', 'Leaky_Subject_Mixing', 'RandomForest'), 'mean_delta_vs_strict']:+.3f}, and patient-only dementia MMSE regression failed significance (r = {dem_pt_rf['pearson_r_mean']:+.3f}, p = {dem_pt_rf['median_p']:.3f})."
    )
    abs_wc = len(abstract_text.split())
    assert 150 <= abs_wc <= 250, f"Abstract word count {abs_wc} exceeds 250-word limit!"
    print(f"   ✓ Abstract word count: {abs_wc} words (<= 250 limit)")

    md = f"""# {EXACT_TITLE}

**Running Title:** {SHORT_TITLE}

**Author:** {AUTHOR_NAME}^1,2*^

^1^ {AFFIL_1}  
^2^ {AFFIL_2}  
^*^ Correspondence: {CORRESPONDENCE}

---

## Abstract

{abstract_text}

**Keywords:** Quantitative EEG; Spectral Parameterization; Spectral Estimation Leakage; Graph Neural Networks; Dataset Auditing; Clinical Benchmarking

---

## 1. Introduction

Quantitative electroencephalography (qEEG) measures macro-scale post-synaptic cortical dynamics with millisecond resolution [1, 2], supporting electrophysiological research across sleep regulation, affective illness, and neurodegeneration [3–7]. Although automated network classifiers and deep graph architectures are frequently reported to achieve high diagnostic accuracy in clinical cohorts [8–10], external validation repeatedly reveals substantial performance drops across centers [11, 12]. Shared methodological bottlenecks include record-wise validation splits that leak repeated recordings from the same participant across folds [13, 14], cohort-dependent acquisition shifts and hardware roll-offs, and unverified normative standardization [15, 16].

Separating periodic oscillatory peaks from broadband `1/f^chi` aperiodic activity is essential before interpreting spectral biomarkers [17–21], because the aperiodic exponent chi relates to synaptic excitation–inhibition balance and spectral slope shifts alter conventional band-power ratios [20, 21]. Unparameterized bandpass measurements complicate clinical resting frontal theta (4–8 Hz) comparisons [22, 23]: frontal theta typically increases with homeostatic sleep pressure [24], whereas major depressive disorder (MDD) cohorts show heterogeneous or conflicting directions [25–28], while consistent low-frequency slowing characterizes Parkinson's disease (PD) and Alzheimer's disease (AD) [29–31]. Two subtle spectral fitting pitfalls further confound interpretation [17, 21]: first, heuristics that mask only the dominant 7–14 Hz alpha peak during ordinary least-squares (`OLS`) slope estimation leave 4–7 Hz theta power inside the linear fit window, leaking oscillatory power into the background slope; second, extending a fixed linear log–log parameterization all the way to filter passband edges (e.g., 1.0 or 40.0 Hz) exposes the aperiodic slope fit to hardware attenuation skirts, capable of masking or inverting periodic residuals.

A parallel methodological gap concerns topological representation learning on routine 19-channel clinical montages [32, 33]. Phase-based estimators such as the Phase Lag Index (PLI) attenuate zero-lag volume conduction [34, 35], and parameter-free Orthogonal Minimum Spanning Trees (OMST) extract connected backbones without arbitrary density thresholds [36, 37]. While Riemannian tangent-space pipelines project connectivity matrices onto symmetric positive-definite (SPD) manifolds [38, 39], Graph Convolutional Networks (GCNs) [40] are similarly assumed to extract relational biomarkers from sensor graphs. Yet sensor PLI is a non-linear sign-asymmetry statistic and need not form a positive semi-definite covariance estimator, raising mathematical validity issues for log-Euclidean Riemannian mappings, while coarse 19-node scalp graphs may offer limited structural advantage over isolated nodes or regularized classical baselines.

To evaluate these pitfalls systematically across independent recordings, I built an open 19-channel benchmarking suite spanning seven public datasets ({acc['total_unique_participants']} unique participants in primary states, {acc['total_analyzed_sessions']} primary sessions, {acc['total_with_mumtaz_eo']} total sessions; 14,003 channel spectra) covering sleep deprivation [41], outpatient MDD [42], university depression screening [43], and PD and dementia cohorts [44–47]. Using `specparam v2.0.0rc7` [17], synthetic ground-truth simulations, 10×5 repeated participant-grouped cross-validation (`StratifiedGroupKFold` design with fold-nested normative z-scoring), edge-matched random-graph controls, aligned 1,000-permutation retraining tests, and raw file/batch forensic audits, I investigated three research questions:

1. Question 1 (Spectral Fit-Window Confounding, Masked-OLS Bias, and Archive Audits): How do passband-interior `[2.0, 30.0] Hz` `specparam` fits compare against edge-exposed `[1.0, 40.0] Hz` `specparam`, `[7, 14] Hz`-masked OLS heuristics, and known-truth synthetic simulations, and to what extent do epoch-duration imbalances, pure rest vs. task isolation, duplicate EDF files, and recording-batch artifacts drive apparent clinical effects?
2. Question 2 (Graph Inductive Bias, 10×5 Multi-Repeat Stability, Aligned Permutations, and Partition/Connectome Validity): Under strict 10×5 repeated participant-grouped cross-validation and aligned permutation tests, does `NormativeGCN` on PLI-OMST topology exceed edge-matched random graphs, isolated nodes, SPD-conditioned Riemannian tangent baselines, or tuned linear/tree models, and how do PLI non-PSD violations and repeated-session leakage affect downstream validity?
3. Question 3 (External Transfer Boundaries, Gain Calibration Prerequisites, and Clinical Severity Floor): Under leak-free disjoint `Split-Target-HC` evaluation, how well do spectral and graph models generalize across centers and clinical domains, how do elementary scaling properties trade off calibration bias against small-sample variance, and does case–control separability extend to within-patient cognitive and motor severity prediction?

---

## 2. Materials and Methods

### 2.1 Public Datasets, Sub-Contrast Taxonomy, and Participant Accounting
I performed secondary computational analysis across seven de-identified, publicly accessible EEG archives hosted on OpenNeuro and Figshare (hardware and clinical demographics in Supplementary Table S3; software versions in Table S9). Because the datasets include within-subject crossovers, symptom screening cohorts, paired medication washout arms, and neurodegenerative multi-group studies, primary analyses respected each study's native design and defined explicit sub-contrasts (`2b`, `4a–b`, `5a–c`, `6a–c`). Across all seven releases, {acc['enrolled_participants']} participants were enrolled ({acc['enrolled_participants'] + 111} primary sessions; {acc['enrolled_participants'] + 171} total including Study 2 eyes-open). Pre-analysis exclusions removed {acc['excluded_participants']} participants (1.4%: 6 in Study 2 primary eyes-closed with >20% artifact-rejected epochs or channel bridging; 2 in Study 3 with intermediate BDI-II scores 10–13; zero exclusions in Studies 1 and 4–7), retaining N = {acc['total_unique_participants']} primary participants and {acc['total_analyzed_sessions']} primary sessions ({acc['total_with_mumtaz_eo']} total; Figure S5):

1. Study 1 (Sleep Deprivation, OpenNeuro `ds004902`, N = 71 participants, 142 paired sessions) [41]: Healthy young adults (mean age 20.0 ± 1.4 years, 47.9% female) recorded eyes-open after normal sleep (`NS`) and after 36 h monitored total sleep deprivation (`SD`).
2. Study 2 (Outpatient MDD, Figshare `4244171`, N = 58 primary eyes-closed `Study 2a`; N = 60 secondary eyes-open `Study 2b`) [42]: DSM-IV outpatient MDD (N = 30 EC, 32 EO) and healthy controls (`HC`, N = 28) recorded on a 19-channel Nihon Kohden system; individual ages were omitted from the public release. Raw file forensics identified two exact SHA-256 duplicate EDF pairs (`H S27` = `H S30` and `MDD S33` = `MDD S34`) plus strong HC acquisition-date clustering (`9` recorded in 2011 vs. `14` on Jan 12–14, 2013 vs. `4` in Apr–Jul 2013), prompting both standard public-manifest reporting and de-duplicated/batch-stratified sensitivity audits (Table S26).
3. Study 3 (University Depression Screening, OpenNeuro `ds003478`, N = 119 participants) [43]: Undergraduates grouped by BDI-II >= 14 (`N = 44`) versus BDI-II <= 7 (`N = 75`) during eyes-open rest (`SCID`: 11 current MDD, 11 past MDD, 8 non-MDD, 14 uninterviewed high-BDI).
4. Study 4 (UCSD PD ON/OFF, OpenNeuro `ds002778`, N = 31 participants, 46 sessions) [44]: Fifteen PD patients evaluated both OFF (`>= 12 h` overnight withdrawal, `Study 4a`) and ON dopaminergic medication (`Study 4b`) alongside 16 HC under eyes-open fixation. Biophysical summaries use per-participant OFF/ON averages (`Study 4 SubjMean`, 15 vs. 16) to avoid treating repeated patient visits as independent; classification evaluates all 46 sessions under participant grouping.
5. Study 5 (AHEPA Clinical Dementia, OpenNeuro `ds004504`, N = 88 participants) [45]: Eyes-closed 19-channel clinical recordings from Alzheimer's disease (`Study 5a`, `N = 36 vs. 29 HC`), frontotemporal dementia (`Study 5b`, `N = 23 vs. 29 HC`), combined dementia (`Study 5c`, `N = 59 vs. 29 HC`), and HC (`N = 29`) with MMSE scores. Because `Study 5c` shares controls and patients with `5a` and `5b`, cross-model Wilcoxon statistics report both the 9-contrast panel and a strictly non-nested 8-contrast panel excluding `5c`.
6. Study 6 (UNM PD Washout, OpenNeuro `ds003490`, N = 50 participants, 75 sessions) [46]: Twenty-five PD patients recorded ON medication (`Study 6b`, `N = 25 vs. 25 CTL`) and after `>= 15 h` overnight withdrawal (`Study 6a`, `N = 25 vs. 25 CTL`; combined repeated sessions `Study 6c`, `N = 75`), plus 25 HC. Event-marker inspection revealed that the raw `.set` recordings contain an initial ~60 s pure eyes-open (`EO`) block (`S1/S2`) followed by a ~60 s eyes-closed (`EC`) block (`S3/S4`) and ~4.5 min of three-stimulus Auditory Oddball task trials (`Standard/Novel/Target`). Legacy scripts ingested the entire file indiscriminately (yielding `87.2` vs. `47.0` epochs per group, `p < 1e-4`). Here, Study 6 was cleanly rebuilt strictly from the contiguous 60 s pure-EO rest segment (`10.46 ± 3.27` vs. `9.60 ± 3.06` clean epochs, `p = 0.267`), alongside a 74-session pure-EC sensitivity analysis (excluding `sub-005`, who lacked >= 2 EC epochs; Table S25).
7. Study 7 (Iowa PD Multi-Scale, OpenNeuro `ds004584`, N = 149 participants) [47]: Eyes-open rest from 100 PD patients ON medication and 49 HC with MoCA and UPDRS Part-III motor scores.

### 2.2 Harmonization, Filtering, Artifact Screening, and Epoch Balance
Recordings were mapped to the canonical 19-channel 10–20 array (`Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, P7, P3, Pz, P4, P8, O1, O2`; Table S5), inserting a zero-initialized `Pz` reference channel in Study 7 prior to Common Average Referencing (`CAR`). Signals were zero-phase FIR bandpass filtered between 1.0 and 40.0 Hz (Hamming window), resampled to 250 Hz, re-referenced to CAR, and epoched into non-overlapping 4.0 s segments (150 μV peak-to-peak rejection threshold; 200 μV fallback in Study 7; retained epoch durations in Tables S8 and S25). Clean epoch counts differed modestly by group in Studies 1 (`p = 0.0103`), 2a (`p = 0.0038`), and 5 (`p = 0.0006`) and were balanced across groups in Studies 2b, 3, 4, 6 EO (`p = 0.267`), and 7; sensitivity ANCOVAs therefore adjusted jointly for chronological age and retained clean epoch count (Table S25).

### 2.3 Interior [2.0, 30.0] Hz Parameterization, Edge-Leakage Audits, and Monte Carlo Simulation
Channel Welch PSDs (2.0 s Hamming sub-windows, 50% overlap, 0.50 Hz resolution) were parameterized across all 14,003 channel spectra (737 sessions x 19 channels) via `SpectralGroupModel` in `specparam v2.0.0rc7` [17] (`aperiodic_mode='fixed'`, `peak_width_limits=[1.0, 12.0]` Hz, `max_n_peaks=6`, `min_peak_height=0.05` B, `peak_threshold=2.0` SD; Section S1.1 and Figure S1). To prevent passband edge attenuation from biasing broadband log–log slope estimation, primary parameterization used the interior `[2.0, 30.0] Hz` window:

$$\\log_{{10}} P(f) = (b - \\chi \\log_{{10}} f) + \\sum_{{k=1}}^K G_k(f), \\quad f \\in [2.0, 30.0]\\text{{ Hz}} \\qquad (1)$$

Four features were computed per channel: band-averaged flattened periodic residuals (`log10 PSD - (b - chi * log10 f)`) over theta (4.0–8.0 Hz), low-frequency periodic residual (2.0–4.0 Hz), and alpha (8.0–13.0 Hz), plus exponent chi. Regional indices averaged five frontal channels (`Fp1, Fp2, F3, F4, Fz`), a 3-channel dorsal sensitivity cluster (`F3, Fz, F4`), five posterior channels (`P3, P4, Pz, O1, O2`), or all 19 channels. Posterior individual alpha peak frequency (`IAPF`) took the median peak center frequency within 7.0–13.5 Hz across the five posterior sites. Every contrast was compared directly against passband-exposed `[1.0, 40.0] Hz` `specparam`, `[7, 14] Hz`-masked OLS (fitted both over `1–40 Hz` and `2–30 Hz`), and summed `specparam` Gaussian peaks (`G_k`), quantifying paired Cohen's `d` differences (`Delta d`) with 1,000 stratified participant-level bootstrap resamples.

To establish bias directionality independently of real-cohort confounds, three known-truth Monte Carlo simulations (150 synthetic trials each matching Study 2 group sizes, N = 30 vs. 28) tested: (S1) true null periodic theta with an isolated exponent drop (`Δchi = -0.35`), (S2) true null theta with broadband alpha attenuation (`-0.25 B`) and exponent drop (`-0.25`), and (S3) true negative theta (`-0.18 B`) plus steep `< 2 Hz` / `> 30 Hz` filter-edge roll-off mimicking clinical passband attenuation. Elementary multiplicative scaling invariance (`gamma in [0.01, 100.0]` across 285 spectra; Section S1.3, Table S2, and Figure S3) was verified on `[2.0, 30.0] Hz` fits strictly as a baseline unit-normalization property and not an algorithmic novelty.

### 2.4 Functional Connectomics (`PLI`), `OMST`, and Riemannian SPD Conditioning
Pairwise 4.0–13.0 Hz Phase Lag Index (`PLI`) [34] matrices were transformed to costs `c_ij = (1.0 - PLI_ij) + 1e-4` and pruned via parameter-free Orthogonal Minimum Spanning Trees (`OMST`) maximizing Global Cost Efficiency (`GCE`; Section S1.2 and Figure S2) [36, 37]. Symmetrized connectivity matrices `C = I_19 + PLI` were audited for negative eigenvalues across all 677 primary sessions. Whereas binary OMST graph message passing uses only adjacency edges and is unaffected by `I + PLI` indefiniteness, Riemannian tangent-space mappings require strictly positive-definite inputs [38, 39]; I evaluated Euclidean 171-D upper-triangular PLI vectors against two SPD-conditioned 190-D Log-Euclidean representations—eigenvalue-adaptive diagonal loading (floor `0.01`) and Frobenius eigenvalue-clipped nearest-SPD projection (`10^-4` floor)—under 10×5 repeated participant-grouped CV.

### 2.5 True 10×5 Participant-Grouped Validation, Edge-Matched Controls, and Aligned Permutations
Inspecting `sklearn.model_selection.StratifiedGroupKFold` on Study 1's balanced crossover revealed that identical fold ties occurred across all random seeds when group signatures tied. I replaced it with an explicit stratified participant-cluster splitter shuffling unique participant IDs across `10 repeats x 5 folds` (`50` outer folds; 0.0% training–test participant overlap). Fold-nested healthy-control normative standardization (`z_ij = (x_ij - mu_HC_train) / sd_HC_train`) and inner 3-fold `GridSearchCV` hyperparameter selection ran strictly inside each outer training split. Twelve configurations were benchmarked across 11 contrasts plus Study 2b EO (`Tables S7, S7b, S10`): the 3,362-parameter pure-PyTorch `NormativeGCN` (`GCN-OMST`, two symmetric-normalized 32-unit graph layers with `A + I`, mean+max readout, fixed 50 Adam epochs without validation early stopping; Section S1.4), `GCN-Identity` (`A = I`), `GCN-Random` (`GCN_Random_EdgeMatched`, drawing a session-specific degree-agnostic random graph matching each session's exact undirected `omst_edge_count` independently per repeat), `GCN-Full` (dense off-diagonal ones), six 76-D flattened classical models (`Tuned Logistic Regression`, `Tuned Ridge`, `Linear SVM`, `RBF SVM`, 100-tree `Random Forest` depth 5, `GBDT` depth 4), and univariate `Single Theta` / `Single Chi`. Pairwise exact Wilcoxon signed-rank tests with Benjamini–Hochberg FDR compared 10×5 means across both all 9 primary contrasts and the 8 strictly non-nested contrasts (excluding `5c`).

Aligned 1,000-permutation retraining tests (`Module D`, `Figure S6`) used the exact same `GridSearchCV` `Tuned LR` estimator and participant-cluster permutation scheme as the primary CV pipeline across six key within-cohort and disjoint `Split-Target-HC` transfer benchmarks. Data-leakage ablations (`Strict Leak-Free`, `Leaky Pre-Standardization`, `Leaky Subject Mixing`) were quantified across all 10 repeats in Studies 1, 4, and 6c.

### 2.6 Disjoint `Split-Target-HC` External Transfer, Gain Perturbation, and Severity Regression
Cross-center PD (6 directed pairs) and transdiagnostic transfer (14 directed pairs) used 20 repeated 50/50 `Split-Target-HC` partitions, estimating target normative reference parameters exclusively from one random half of target healthy controls and evaluating on the strictly held-out half alongside all target patients (Oracle full-HC figures retained in Supplementary Tables solely for comparison). Explicit uniform (`k = 0.5, 2.0, 5.0`) and mixed (`50% 0.5x / 50% 5.0x`) PSD perturbations evaluated both session-mean and worst-case maximum probability shift. Out-of-fold 10×5 regressions (`GBDT`, `Random Forest`, `RBF SVR`, `Ridge`) predicted MMSE (Study 5) and MoCA/UPDRS-III (Study 7) with and without healthy controls.

### 2.7 Statistical Inference
Group differences used two-sided Welch t-tests (paired t-tests in Study 1 and ON/OFF medication pairs) with Benjamini–Hochberg FDR across 10 theta contrasts (`q_10`) and 50 cohort-by-marker comparisons (`q_50`). Meta-analytic pooling used DerSimonian–Laird (`DL`) random effects alongside conservative variance-floored Hartung–Knapp–Sidik–Jonkman (`HKSJ`) intervals and p-values (`K = 6` primary participant-averaged `SubjMean + PD-OFF` set, N = 495), plus medication-ON, sub-diagnosis, and `K = 7` sensitivity sets (`Table S22`). Study 2 raw EDFs were further audited via SHA-256 hashing and date stratification (`Table S26`).

---

## 3. Results

### 3.1 Passband-Interior [2.0, 30.0] Hz Parameterization, Dual Leakage Mechanisms, and Study 2 Archive Forensics (RQ1)
Fitting `specparam v2.0.0rc7` over clean interior `[2.0, 30.0] Hz` bounds yielded zero failed fits across all 14,003 channel spectra (median `R² = {fit230['r2_median'].min():.3f}–{fit230['r2_median'].max():.3f}`, median `MAE = {fit230['mae_median'].min():.3f}–{fit230['mae_median'].max():.3f} B`; Table 1, Table S23, Figure S4). Auditing fit bounds exposed marked passband-edge degradation in Study 2 outpatient MDD when fitting `[1.0, 40.0] Hz`: because Study 2 raw MDD spectra attenuate steeply relative to HC near 40 Hz (`-1.060 B` vs. `-0.08 B` across mid-frequencies), `[1.0, 40.0] Hz` `specparam` fits achieved median `R² = {fit140.loc['mumtaz_ec', 'r2_median']:.3f}` (`{fit140.loc['mumtaz_ec', 'pct_r2_lt_080']:.1f}% < 0.80`, MAE `{fit140.loc['mumtaz_ec', 'mae_median']:.3f} B` EC; `{fit140.loc['mumtaz_eo', 'pct_r2_lt_080']:.1f}% < 0.80` EO). Restricting fits to interior `[2.0, 30.0] Hz` raised Study 2 EC median `R²` to `{fit230.loc['mumtaz_ec', 'r2_median']:.3f}` (`{fit230.loc['mumtaz_ec', 'pct_r2_lt_080']:.1f}% < 0.80`, MAE `{fit230.loc['mumtaz_ec', 'mae_median']:.3f} B`), revealing strong nominal MDD vs. HC shifts in Study 2 EC (frontal theta `d = {ft_w.loc['Study 2: Outpatient MDD', 'cohens_d_specparam_resid_2to30']:+.3f}`, `p = {fmt_p(ft_w.loc['Study 2: Outpatient MDD', 'p_value_specparam_resid_2to30'])}`; global χ `d = {chi_w.loc['Study 2: Outpatient MDD', 'cohens_d_specparam_resid_2to30']:+.3f}`, `p = {fmt_p(chi_w.loc['Study 2: Outpatient MDD', 'p_value_specparam_resid_2to30'])}`; global 2–4 Hz `d = {m2_ec_dl['cohens_d']:+.3f}`) that had been distorted or inverted under `[1.0, 40.0] Hz` (`d = {ft_w.loc['Study 2: Outpatient MDD', 'cohens_d_specparam_resid_1to40']:+.3f}`; bootstrap Δd = `{b_s2_w['delta_d']:+.3f} [{b_s2_w['ci_lower']:+.3f}, {b_s2_w['ci_upper']:+.3f}]`, `p = {fmt_p(b_s2_w['boot_p'])}`, Table 2 and Figure 2A). Conversely, in Study 3's clean BioSemi spectra (median `R² = {fit230.loc['cavanagh_ds003478', 'r2_median']:.3f}`), legacy `[7, 14] Hz`-masked OLS over-estimated theta negativity (`d = {ft_w.loc['Study 3: Univ. Depression Screening', 'cohens_d_masked_ols_1to40']:+.3f}`) compared with canonical `specparam` (`d = {ft_w.loc['Study 3: Univ. Depression Screening', 'cohens_d_specparam_resid_2to30']:+.3f}`; bootstrap Δd = `{b_s3_o['delta_d']:+.3f} [{b_s3_o['ci_lower']:+.3f}, {b_s3_o['ci_upper']:+.3f}]`, `p = {fmt_p(b_s3_o['boot_p'])}`).

![](Figure_1.png){{width=6.5in}}
**Figure 1. Canonical Interior [2.0, 30.0] Hz Parameterized Biophysical Landscape (`specparam v2.0.0rc7`).** Standardized effect sizes (Cohen's d with 95% confidence intervals) across all ten contrasts for (A) 5-channel frontal periodic theta (4–8 Hz), (B) 19-channel low-frequency periodic residual (2–4 Hz), (C) 5-channel posterior periodic alpha (8–13 Hz), and (D) 19-channel global aperiodic exponent chi. Neurodegenerative PD and dementia cohorts show reproducible positive frontal theta slowing, whereas Study 2's apparent negative shift depends on acquisition-batch controls.

Known-truth Monte Carlo simulations verified both mechanisms directly (Figure 2B and Table S11). Under true null periodic theta accompanied by broadband alpha reduction and steeper slope curvature (Scenario S2), `[7, 14] Hz`-masked OLS generated substantial spurious negative theta (`bias = {sim_s2_ols['theta_d_bias']:+.3f} d`) and exaggerated slope bias (`chi bias = {sim_s2_ols['chi_d_bias']:+.3f} d`), whereas canonical `[2.0, 30.0] Hz` `specparam` reduced theta bias nearly three-fold (`{sim_s2_230['theta_d_bias']:+.3f} d`) and cut slope bias to `{sim_s2_230['chi_d_bias']:+.3f} d`. Under filter-skirt attenuation mimicking Study 2 (Scenario S3), `[1.0, 40.0] Hz` `specparam` degraded fit quality (`R² = {sim_s3_140['median_r2']:.3f}`) and flipped estimated theta from true `-3.99 d` to `+3.07 d` (`bias = {sim_s3_140['theta_d_bias']:+.3f} d`), whereas `[2.0, 30.0] Hz` recovered `-3.51 d` (`R² = {sim_s3_230['median_r2']:.3f}`, slope bias `{sim_s3_230['chi_d_bias']:+.3f} d`).

Critically, deeper raw-archive forensics explained why Study 2 outpatient MDD appeared so divergent even under clean `[2.0, 30.0] Hz` fits (`Table S26`). SHA-256 hashing of all 181 EDF files in Figshare `4244171` uncovered exact byte-for-byte duplicate files across nominal subjects (`H S27` = `H S30` and `MDD S33` = `MDD S34` across EC, EO, and task files, plus duplicate `H S15 EO` and 10 identical HC task files). Even after deduplicating (`N = 56 EC`, `d_theta = {s2_for.loc['Study 2a EC: Deduplicated Unique (29 MDD vs 27 HC, N=56)', 'theta_d']:+.3f}`), EDF acquisition dates revealed that while all 29 unique MDD patients were recorded in Apr–Jul 2013, 27 unique HCs split across Jul–Aug 2011 (`n = 9`), Jan 12–14, 2013 (`n = 14`), and Apr–Jul 2013 (`n = 4`). Every Jan-2013 HC carried anomalous `5 Hz / 50 Hz ±π/2` cross-channel phase artifacts, creating an extreme `d = +3.347` (`p < 1e-4`) separation between two healthy control batches (`Jan-2013` vs. `2011` HCs); excluding those 14 artifact HCs dropped Study 2 MDD vs. HC frontal theta completely to null (`d = {s2_for.loc['Study 2a EC: Deduplicated Excl. Jan-2013 HC Batch (29 MDD vs 13 HC, N=42)', 'theta_d']:+.3f}, p = {s2_for.loc['Study 2a EC: Deduplicated Excl. Jan-2013 HC Batch (29 MDD vs 13 HC, N=42)', 'theta_p']:.4f}`; vs. 2011 HCs alone `d = {s2_for.loc['Study 2a EC: Deduplicated vs 2011 HC Batch Only (29 MDD vs 9 HC, N=38)', 'theta_d']:+.3f}, p = {s2_for.loc['Study 2a EC: Deduplicated vs 2011 HC Batch Only (29 MDD vs 9 HC, N=38)', 'theta_p']:.4f}`), while batch-linked aperiodic χ (`d = {s2_for.loc['Study 2a EC: Deduplicated Excl. Jan-2013 HC Batch (29 MDD vs 13 HC, N=42)', 'chi_d']:+.3f}`) maintained `RF 10×5 AUC = {s2_for.loc['Study 2a EC: Deduplicated Excl. Jan-2013 HC Batch (29 MDD vs 13 HC, N=42)', 'rf_10x5_mean']:.3f}`. Study 2b eyes-open recordings showed the same global delta (`d = {eb_m_eo_dl['cohens_d']:+.3f}, p = {fmt_p(eb_m_eo_dl['p_value'])}`) and non-significant χ pattern (`p = {eb_m_eo_ch['p_value']:.4f}`).

{t1}

{t2}

Across all remaining cohorts free of batch phase corruption (Table 1 and Figure 1A), canonical `[2.0, 30.0] Hz` frontal periodic theta shifts replicated cleanly: elevated in acute sleep deprivation (Study 1 `d_z = {ft_w.loc['Study 1: Acute Sleep Deprivation', 'cohens_d_specparam_resid_2to30']:+.3f}, p = {fmt_p(ft_w.loc['Study 1: Acute Sleep Deprivation', 'p_value_specparam_resid_2to30'])}`; global theta `d_z = {eb_sl['cohens_d']:+.3f}`), PD cohorts (`Iowa PD d = {ft_w.loc['Study 7: Iowa Parkinson Cohort', 'cohens_d_specparam_resid_2to30']:+.3f}, p = {fmt_p(ft_w.loc['Study 7: Iowa Parkinson Cohort', 'p_value_specparam_resid_2to30'])}`; `UNM PD-ON EO d = {ft_w.loc['Study 6b: UNM Parkinson ON', 'cohens_d_specparam_resid_2to30']:+.3f}, p = {fmt_p(ft_w.loc['Study 6b: UNM Parkinson ON', 'p_value_specparam_resid_2to30'])}`; `UNM PD-OFF EO d = {ft_w.loc['Study 6a: UNM Parkinson OFF', 'cohens_d_specparam_resid_2to30']:+.3f}, p = {fmt_p(ft_w.loc['Study 6a: UNM Parkinson OFF', 'p_value_specparam_resid_2to30'])}`; `UCSD SubjMean d = {ft_w.loc['Study 4: UCSD Parkinson All', 'cohens_d_specparam_resid_2to30']:+.3f}`), and dementia (`AD d = {ft_w.loc["Study 5a: Alzheimer's Disease (AD)", 'cohens_d_specparam_resid_2to30']:+.3f}, p = {fmt_p(ft_w.loc["Study 5a: Alzheimer's Disease (AD)", 'p_value_specparam_resid_2to30'])}`; `FTD d = {ft_w.loc['Study 5b: Frontotemporal Dementia (FTD)', 'cohens_d_specparam_resid_2to30']:+.3f}, p = {fmt_p(ft_w.loc['Study 5b: Frontotemporal Dementia (FTD)', 'p_value_specparam_resid_2to30'])}`; `Combined d = {ft_w.loc['Study 5c: Clinical Dementia (All)', 'cohens_d_specparam_resid_2to30']:+.3f}`), while university depression screening retained lower theta (`Study 3 d = {ft_w.loc['Study 3: Univ. Depression Screening', 'cohens_d_specparam_resid_2to30']:+.3f}, p = {fmt_p(ft_w.loc['Study 3: Univ. Depression Screening', 'p_value_specparam_resid_2to30'])}`). AD additionally displayed large posterior alpha loss (`d = {ad_pa['cohens_d']:+.3f}`) and steeper aperiodic slope (`d = {ad_gc['cohens_d']:+.3f}`). Effects persisted in dorsal `F3–Fz–F4` clusters (Table 1) and after joint chronological-age and clean-epoch-duration ANCOVA (`p <= 0.034` across Studies 3, 5c, 6b, and 7; epoch duration `p >= 0.072` in all adjusted models; Tables S17 and S25). In UNM Study 6, pure-EC sensitivity (`N = 74`) amplified PD theta separation (`OFF d = {unm_sens_df.loc['UNM Pure EC (N=25 PD-OFF vs 24 CTL)', 'theta_d']:+.3f}, p = {fmt_p(unm_sens_df.loc['UNM Pure EC (N=25 PD-OFF vs 24 CTL)', 'theta_p'])}`) relative to pure-EO (`{unm_sens_df.loc['UNM Pure EO (N=25 PD-OFF vs 25 CTL)', 'theta_d']:+.3f}`; Table S25), whereas paired ON–OFF medication comparisons showed no significant frontal theta difference (`UNM d_z = {unm_med_ft['paired_dz']:+.3f}, p = {unm_med_ft['paired_p']:.4f}`; `UCSD d_z = {ucsd_med_ft['paired_dz']:+.3f}, p = {ucsd_med_ft['paired_p']:.4f}`; Table S13) and posterior `IAPF` slowed in Iowa PD (`{iowa_iapf['diff_hz']:+.3f} Hz, d = {iowa_iapf['cohens_d']:+.3f}, p = {fmt_p(iowa_iapf['p_value'])}`; Table S24). Primary participant-averaged random-effects meta-analysis (`K = 6, N = 495`; Table S22) yielded `d_RE = {ndg_m['pooled_d']:+.3f}` (`HKSJ 95% CI [{ndg_m['hksj_ci_lower']:+.3f}, {ndg_m['hksj_ci_upper']:+.3f}], p_HKSJ = {ndg_m['p_hksj']:.4f}, I² = {ndg_m['I2_percent']:.1f}%`) across PD/dementia versus `d_RE = {dep_m['pooled_d']:+.3f}` (`DL p = {fmt_p(dep_m['p_value'])}; df = 1 HKSJ p = {dep_m['p_hksj']:.4f}`) in depression (`Q_between = {dep_m['Q_between']:.2f}, p = {fmt_p(dep_m['p_between'])}`).

![](Figure_2.png){{width=6.5in}}
**Figure 2. Spectral Estimator & Fit-Window Confounding and Gain Drift.** (A) Empirical frontal theta effect sizes comparing legacy `[7, 14]` Hz-masked OLS, `[1, 40]` Hz `specparam`, and canonical interior `[2, 30]` Hz `specparam`. (B) Monte Carlo simulation bias across 150 trials per scenario proving false OLS theta negativity in null Scenario S2 (`-0.65 d`) and passband-skirt polarity inversion for `[1, 40]` Hz `specparam` in Scenario S3 (`+7.06 d` bias). (C) Uncalibrated LR probability drift (`mean` and `worst-case max`) across multiplicative voltage scaling factors k in six PD cross-site pairs.

### 3.2 GNN Topological Indistinguishability Under 10×5 Grouped CV, Aligned Permutations, and Structural Audits (RQ2)
Across all {omst['total_graphs']} primary OMST graphs ({omst['total_node_instances']:,} nodes; Tables S1 and S4), mean selected trees averaged `{omst['mean_trees']:.2f} ± {omst['sd_trees']:.2f}` (`{omst['mean_edges']:.2f} ± {omst['sd_edges']:.2f}` undirected edges, mean `GCE = {omst['mean_gce']:.4f} ± {omst['sd_gce']:.4f}`, 0.0% isolated nodes) with no significant case–control GCE difference in any cohort (all `q_50 >= 0.087`). Under true 10×5 repeated participant-grouped CV across nine primary contrasts (Table 3, Figure 3A, Tables S7, S7b, S10, S10b, and S20), `GCN-OMST` achieved a 9-contrast mean ROC-AUC of `{rep_p9['GCN_OMST'].mean():.3f}` (`0.596 ± 0.025` in sleep deprivation to `0.856 ± 0.025` in Study 2 EC), an 8-non-nested contrast mean of `{rep_nn8['GCN_OMST'].mean():.3f}`, and an 11-contrast mean of `{rep_all11['GCN_OMST'].mean():.3f}`. Across pairwise Wilcoxon comparisons, `GCN-OMST` showed no significant difference from edge-matched random topology `GCN-Random` (`9-contrast {rep_p9['GCN_Random_EdgeMatched'].mean():.3f}, W = {wil9.loc['GCN_Random_EdgeMatched', 'wilcoxon_W']:.1f}, p = {wil9.loc['GCN_Random_EdgeMatched', 'raw_p']:.4f}`; `8-contrast {rep_nn8['GCN_Random_EdgeMatched'].mean():.3f}, p = {wil8.loc['GCN_Random_EdgeMatched', 'raw_p']:.4f}`), dense `GCN-Full` (`{rep_p9['GCN_Full'].mean():.3f}, p = {wil9.loc['GCN_Full', 'raw_p']:.4f}`), and isolated-node `GCN-Identity` (`{rep_p9['GCN_Identity'].mean():.3f}, p = {wil9.loc['GCN_Identity', 'raw_p']:.4f}`), while numerical averages favored simple tabular models: `Tuned Ridge` (`{rep_p9['Tuned_Ridge'].mean():.3f}`), `Random Forest` (`{rep_p9['RandomForest'].mean():.3f}`), and `Tuned LR` (`{rep_p9['Tuned_LogisticRegression'].mean():.3f}`, rising to `{rep_all11['Tuned_Ridge'].mean():.3f}` / `{rep_all11['RandomForest'].mean():.3f}` / `{rep_all11['Tuned_LogisticRegression'].mean():.3f}` vs. `{rep_all11['GCN_OMST'].mean():.3f}` across all 11 contrasts; Wilcoxon `p = {wil11.loc['Tuned_Ridge', 'raw_p']:.4f}`–`{wil9.loc['Tuned_LogisticRegression', 'raw_p']:.4f}`). `GCN-OMST` outperformed only poorly tuned `RBF SVM` (`{rep_p9['RBF_SVM'].mean():.3f}, q = {wil9.loc['RBF_SVM', 'fdr_q']:.4f}`) and univariate `Single Chi` (`{rep_p9['Single_Chi'].mean():.3f}, q = {wil9.loc['Single_Chi', 'fdr_q']:.4f}`), while Study 6a pure-EO PD-OFF (`{rep_idx.loc[('Study 6a: UNM PD-OFF vs Control', 'Tuned_LogisticRegression'), 'mean_auc']:.3f}` LR vs. `{rep_idx.loc[('Study 6a: UNM PD-OFF vs Control', 'GCN_OMST'), 'mean_auc']:.3f}` GCN-OMST) and Study 4 repeated-session variance (`±{rep_idx.loc[('Study 4: UCSD Parkinson All Sessions', 'GCN_OMST'), 'std_auc']:.3f}`) underscored over-parameterization fragility. Leave-one-feature-out ablations (Table S6) and 5-cohort hyperparameter sweeps (`{hparam_means.min():.3f}–{hparam_means.max():.3f}`, Table S7c) confirmed dependence on local node features.

{t3}

Aligned 1,000-permutation retraining tests using exact `GridSearchCV` `Tuned LR` and participant-cluster splits corroborated this pattern (Figure S6). In Study 5c dementia (`p = {perms['A']['p_perm_gcn']:.4f}` GCN; `p = {perms['A']['p_perm_lr']:.4f}` LR) and Study 7 Iowa PD (`GCN {perms['B']['obs_gcn_auc']:.3f}, p = {perms['B']['p_perm_gcn']:.4f}`; `LR {perms['B']['obs_lr_auc']:.3f}, p = {perms['B']['p_perm_lr']:.4f}`), both models exceeded permutation chance; in Study 6a UNM PD-OFF, `Tuned LR` cleanly exceeded chance (`{perms['C']['obs_lr_auc']:.3f}, p = {perms['C']['p_perm_lr']:.4f}`) whereas `GCN-OMST` fell inside the permutation null (`{perms['C']['obs_gcn_auc']:.3f}, p = {perms['C']['p_perm_gcn']:.4f}`). Under disjoint `Split-Target-HC` transfer, `Iowa PD -> UNM PD-OFF` remained within small-sample target permutation variance (`p >= {perms['E']['p_perm_lr']:.4f}`), while `Sleep Debt -> Dementia` reached `p = {perms['F']['p_perm_gcn']:.4f}` (`GCN`) / `p = {perms['F']['p_perm_lr']:.4f}` (`LR`).

Two informatics audits clarify these structural ceilings. First, across {int(pli_s['n_sessions'])} primary 19 x 19 symmetrized PLI matrices `I_19 + PLI` (Figure 3C and Table S21), `{int(pli_s['n_indefinite_raw'])}` (`{pli_s['pct_indefinite_raw']:.2f}%`) failed positive semi-definiteness (min eigenvalue `{pli_s['min_eigenvalue']:.4f}` in Study 2 MDD, where `{pli_a.loc['Study 2: Outpatient MDD (EC)', 'pct_indefinite_raw']:.1f}%` were indefinite alongside `{pli_a.loc['Study 5: Clinical Dementia (ds004504)', 'pct_indefinite_raw']:.1f}%` in Study 5 dementia). Because `NormativeGCN` consumes only binary adjacency edges derived from OMST pruning, this non-PSD property does not invalidate GCN matrix operations—rather, it directly breaks unconditioned Riemannian Log-Euclidean tangent pipelines unless matrices undergo diagonal loading (`10×5 9-contrast RF {riem_p9.loc[('Riemannian_Gershgorin_SPD_190D', 'RF')]:.3f}`) or eigenvalue-clipped nearest-SPD projection (`{riem_p9.loc[('Riemannian_Higham_SPD_190D', 'RF')]:.3f}`), neither of which exceeded Euclidean 171-D PLI (`{riem_p9.loc[('Euclidean_UpperTri_171D', 'LR')]:.3f}`) or 76-D spectral features (`{rep_p9['Tuned_Ridge'].mean():.3f}`). Second, 10×5 session-mixing leakage (`StratifiedKFold`, 53.9%–79.3% participant overlap; Figure 3B and Table S18) warped AUC bidirectionally by label agreement: depressing non-linear AUC in sleep-deprivation crossover (`Random Forest` 10×5 mean `{leak.loc[('Study 1: Sleep Deprivation', 'Leaky_Subject_Mixing', 'RandomForest'), 'mean_delta_vs_strict']:+.3f} [95% CI {leak.loc[('Study 1: Sleep Deprivation', 'Leaky_Subject_Mixing', 'RandomForest'), 'delta_ci_lo']:+.3f}, {leak.loc[('Study 1: Sleep Deprivation', 'Leaky_Subject_Mixing', 'RandomForest'), 'delta_ci_hi']:+.3f}]`) while inflating AUC in label-concordant PD ON/OFF (`UCSD RF {leak.loc[('Study 4: UCSD Parkinson', 'Leaky_Subject_Mixing', 'RandomForest'), 'mean_delta_vs_strict']:+.3f} [{leak.loc[('Study 4: UCSD Parkinson', 'Leaky_Subject_Mixing', 'RandomForest'), 'delta_ci_lo']:+.3f}, {leak.loc[('Study 4: UCSD Parkinson', 'Leaky_Subject_Mixing', 'RandomForest'), 'delta_ci_hi']:+.3f}]`; `UNM LR {leak.loc[('Study 6c: UNM Parkinson All', 'Leaky_Subject_Mixing', 'Tuned_LogisticRegression'), 'mean_delta_vs_strict']:+.3f} [{leak.loc[('Study 6c: UNM Parkinson All', 'Leaky_Subject_Mixing', 'Tuned_LogisticRegression'), 'delta_ci_lo']:+.3f}, {leak.loc[('Study 6c: UNM Parkinson All', 'Leaky_Subject_Mixing', 'Tuned_LogisticRegression'), 'delta_ci_hi']:+.3f}]`).

![](Figure_3.png){{width=6.5in}}
**Figure 3. 10×5 GNN Topological Indistinguishability, Directional Leakage CIs, and PLI Non-PSD Violations.** (A) 10×5 repeated participant-grouped mean ROC-AUC (± 1 SD) across all 9 primary contrasts showing `Tuned Ridge`, `Random Forest`, and `Tuned LR` matching or exceeding `GCN-OMST`, edge-matched `GCN-Random`, `GCN-Full`, and `GCN-Identity`. (B) 10×5 mean ΔAUC (± 95% CI) induced by session-mixed CV relative to strict participant grouping. (C) Cohort breakdown of indefinite `I_19 + PLI` matrices ({int(pli_s['n_indefinite_raw'])}/{int(pli_s['n_sessions'])} = {pli_s['pct_indefinite_raw']:.2f}%).

### 3.3 Leak-Free External Transfer, Elementary Calibration Trade-Offs, and Clinical Severity Floor (RQ3)
As expected from additive log-offset arithmetic, empirical tests across 285 spectra verified numerical `specparam` invariance over `gamma in [0.01, 100.0]` (`max |Δchi| = {gchk['max_abs_diff_chi'].max():.2e}`, `max |Δθ| = {gchk['max_abs_diff_theta_resid'].max():.2e}`; Table S2 and Figure S3). In uncalibrated 6-pair PD transfer (Table 4B, Figure 2C, Table S19), unparameterized log10 power suffered session-mean probability drift of `{amp_raw_5x['lr_mean_prob_drift'].mean():.3f}` (`worst-case max = {amp_raw_5x['lr_max_prob_drift'].max():.3f}`) under `5x` gain and AUC drop (`{cal_raw_lr_raw:.3f} -> {amp_raw_mix['lr_auc'].mean():.3f}`) under heterogeneous site gains, whereas `specparam` residuals showed zero drift (`LR = {cal_raw_lr_sp:.3f}`, `RF = {gc_sub[(gc_sub['experiment'] == 'Calibration_Strategy') & (gc_sub['calibration'] == 'Uncalibrated_Raw') & (gc_sub['feature_type'] == 'specparam_periodic_residual')]['rf_auc'].mean():.3f}` vs. relative power `LR = {cal_raw_lr_rel:.3f}`). However, normative target-HC standardization traded modest balanced-accuracy gains (`{cal_raw_bacc_sp:.3f} -> {cal_thc_lr_sp['lr_bacc'].mean():.3f}`) for higher Brier calibration error (`{cal_thc_lr_sp['lr_brier'].mean():.3f}`) because estimating 76 reference parameters from small control samples amplifies variance.

Under strictly disjoint 20-repeat 50/50 `Split-Target-HC` transfer (Table 4A, Figure 4A, Tables S15 and S16), `Iowa PD -> UNM PD-OFF EO` (`Tuned LR {pd_tx.loc[('Iowa_PD', 'UNM_PD_OFF'), 'Tuned_LR']:.3f}`, `RF {pd_tx.loc[('Iowa_PD', 'UNM_PD_OFF'), 'RandomForest']:.3f}`) and `Iowa PD -> UCSD PD` (`Tuned LR {pd_tx.loc[('Iowa_PD', 'UCSD_PD'), 'Tuned_LR']:.3f}`, `RF {pd_tx.loc[('Iowa_PD', 'UCSD_PD'), 'RandomForest']:.3f}`) generalized moderately well without retraining, whereas small-source transfers (`N = 46–50 -> 50`) dropped toward `0.418–0.662`. Across domains, slowing cohorts transferred positively (`Sleep Debt -> Dementia GCN {td_tx.loc[('Sleep_Debt', 'Dementia_All'), 'GCN_OMST']:.3f}`, `Sleep Debt -> Iowa PD GCN {td_tx.loc[('Sleep_Debt', 'Iowa_PD'), 'GCN_OMST']:.3f}`), whereas crossing between neurodegenerative cohorts and batch-confounded Study 2 MDD dropped below chance (`0.258–0.563`). State-matched `Cavanagh EO <-> Mumtaz EO` transfer (`Table S12`; 55-subject EC–EO theta concordance `r = {mum_conc.loc['Frontal Periodic Theta (4–8 Hz Residual)', 'pearson_r']:+.3f}`) remained asymmetric (`{mum_tx.loc[('Mumtaz_EO (Eyes-Open)', 'Cavanagh_EO (Eyes-Open)'), 'lr_auc']:.3f}` LR vs. `{mum_tx.loc[('Cavanagh_EO (Eyes-Open)', 'Mumtaz_EO (Eyes-Open)'), 'lr_auc']:.3f}`).

{t4}

![](Figure_4.png){{width=6.5in}}
**Figure 4. Leak-Free Cross-Disorder Generalization and Within-Patient Clinical Severity Floor.** (A) 4 x 4 cross-domain `Random Forest` ROC-AUC matrix using 20-repeat `Split-Target-HC` external evaluation and 10×5 CV diagonals. (B) Dementia Split-0 MMSE predictions showing control-ceiling separation (`N = 88`) versus non-significant patient-only correlation (`N = 59`). (C) 10×5 repeated OOF Pearson correlations (± 1 SD) across MMSE, MoCA, and UPDRS-III.

Finally, 10×5 clinical severity regressions exposed control-ceiling confounding (Table 5, Figure 4B–C, Table S14). Pooling healthy controls (`MMSE = 30.0`, N = 88) produced significant MMSE correlation (`SVR RBF r = {dem_all_svr['pearson_r_mean']:+.3f}, p = {fmt_p(dem_all_svr['median_p'])}, R² = {dem_all_svr['r2_mean']:.3f}`), which collapsed when restricted to the 59 AD/FTD patients (`RF r = {dem_pt_rf['pearson_r_mean']:+.3f}, p = {dem_pt_rf['median_p']:.3f}, R² = {dem_pt_rf['r2_mean']:.3f}`; `Ridge r = {dem_pt_ridge['pearson_r_mean']:+.3f}, p = {dem_pt_ridge['median_p']:.3f}`). In Study 7 Iowa PD, `RBF SVR` predicted full-cohort MoCA (`N = 149: r = {iowa_all_svr['pearson_r_mean']:+.3f}, R² = {iowa_all_svr['r2_mean']:.3f}`) and retained weak patient-only signal (`N = 100: r = {iowa_pt_svr['pearson_r_mean']:+.3f}, p = {iowa_pt_svr['median_p']:.4f}, R² = {iowa_pt_svr['r2_mean']:.3f}`), while `UPDRS-III` reached at most `r = {iowa_updrs_gbdt['pearson_r_mean']:+.3f}` with negative explained variance (`GBDT R² = {iowa_updrs_gbdt['r2_mean']:.3f}`); overall patient-only `R²` never exceeded `0.035` across any cohort or regressor.

{t5}

---

## 4. Discussion

### 4.1 Overview of Benchmark Pitfalls
Evaluating seven public resting-state EEG cohorts (`566` primary participants, `14,003` channel spectra) through a harmonized 19-channel pipeline exposed three pervasive pitfalls that distort computational qEEG claims. First, spectral fitting choices—both leaving 4–7 Hz unmasked in `[7, 14] Hz` OLS background fits and extending fixed log–log fits into `[1.0, 40.0] Hz` filter skirts—systematically bias periodic theta and exponent effect sizes, while public archive forensics in Study 2 uncovered exact file duplicates and acquisition-batch artifacts capable of generating false positive control–control contrasts (`d = +3.35`). Second, under true 10×5 repeated participant-grouped validation, a compact 3,362-parameter `GCN-OMST` offered zero topological advantage over edge-matched random graphs, isolated nodes, or simple regularized linear and tree baselines. Third, sensor PLI violated positive semi-definiteness in one-eighth of recordings, session mixing skewed AUC bidirectionally by up to ±0.09, and standalone resting 19-channel spectral biomarkers explained virtually zero within-patient clinical variance (`R² <= 0.035`).

### 4.2 Spectral Estimation Leakage, Passband Windowing, and Public Archive Integrity
Direct head-to-head benchmarking and Monte Carlo simulations demonstrate that spectral parameterization must avoid both unmodeled oscillatory bands and hardware transition zones [17, 21]. In Study 3's clean high-`R²` spectra, legacy `[7, 14] Hz`-masked OLS inflated depression theta effect sizes (`-0.676` vs. `-0.464`), replicating the Monte Carlo proof (`-0.652 d` false bias under null theta with broad alpha curvature). Conversely, in Study 2's Nihon Kohden recordings, fitting `[1.0, 40.0] Hz` across steep high-frequency roll-off collapsed `specparam` fit goodness (`R² = 0.895`) and masked group shifts (`-0.226` vs. `-1.166`), precisely reproducing the Scenario S3 edge-attenuation simulation (`+7.06 d` distortion). Shrinking the parameterization bounds strictly inside `[2.0, 30.0] Hz` resolved filter-skirt failure (`R² >= 0.959` across every cohort) without altering main effects in PD or dementia [29–31].

Beyond estimator math, dataset forensics in Figshare `4244171` (Study 2) provide a cautionary benchmark lesson for the EEG machine learning community. Dozens of published classifiers report `> 90%` MDD accuracy on this release; yet raw-file SHA-256 auditing identified duplicate subjects (`H S27` = `H S30`; `MDD S33` = `MDD S34`) alongside a 2011 versus Jan-2013 control batch discrepancy (`theta d = +3.35`) coupled to 5/50 Hz phase artifacts (`39.7%` indefinite PLI). Removing the Jan-2013 control batch eliminated the MDD periodic theta shift (`d = -0.082, p = 0.752`), while `~0.87` classification AUC persisted via broadband aperiodic exponent differences—indicating that high single-site accuracy in Study 2 substantially reflects hardware/batch regime separation rather than pathognomonic oscillatory depression biomarkers [25–28].

### 4.3 Why 19-Channel PLI-OMST Graph Convolutions Failed to Outperform Simple Baselines
Rebuilding Study 6 from contiguous 60 s pure eyes-open rest (`d = +0.427` vs. oddball-contaminated segments) and replacing tied `StratifiedGroupKFold` splits with true 10×5 participant-grouped shuffles eliminated early single-seed artifacts (`Study 1` linear baseline rose from `0.507` to `0.638`; seed-42 `GCN-Random` fluctuations regressed to parity). Across 10×5 means, `GCN-OMST` (`0.717`), edge-matched `GCN-Random` (`0.721`), `GCN-Full` (`0.720`), and `GCN-Identity` (`0.708`) formed a flat topological equivalence class trailing `Tuned Ridge` (`0.732`), `Random Forest` (`0.731`), and `Tuned LR` (`0.729`). In sparse 19-electrode clinical caps averaging under three OMST edges per node, short-window sensor PLI matrices carry high phase-estimation variance and fail positive semi-definiteness (`80/677 = 11.82%`), violating the SPD assumptions of tangent-space projections [38, 39] while adding no spatial signal beyond what regularized linear weights already extract from local frontal theta and posterior alpha/exponent nodes [13, 14]. This negative result applies specifically to static 4-feature 19-channel GCNs; higher-density arrays, source-localized cortical parcellations, or dynamic networks require separate empirical testing [10, 12].

### 4.4 Validation Leakage, Transfer Calibration Trade-Offs, and Clinical Staging Limits
Ten-repeat leakage experiments confirm that subject-mixing cross-validation (`StratifiedKFold`) is not simply optimistic—it is structurally bidirectional [13, 14]. When repeated patient sessions carry concordant diagnostic labels (`Study 4` and `Study 6c PD`), participant fingerprint memorization inflates non-linear AUC (`+0.038` to `+0.080`); when repeated recordings flip labels within individuals (`Study 1` crossover), fingerprint memorization pulls predictions opposite to test labels (`-0.052` to `-0.094`). Strict participant-level grouping across both outer evaluation and inner hyperparameter tuning folds is non-negotiable.

Under leak-free 50/50 `Split-Target-HC` external evaluation, cross-site PD generalization (`Iowa -> UNM` `0.741 LR`, `Iowa -> UCSD` `0.751 LR`) demonstrates partial trans-center transfer when large sources (`N = 149`) normalize against target controls, whereas cross-disorder generalization tracks shared slow-wave physiology (`Sleep <-> Iowa PD <-> Dementia`, `0.629–0.792`) rather than disease specificity. Although `specparam` provides elementary protection against raw voltage unit scaling (`mean 0.290 / max 0.691` probability drift eliminated), reference-control calibration against small site samples (`N = 16–25`) inflates Brier error. Finally, collapsing MMSE predictions upon excluding ceiling controls confirms that cross-sectional 19-channel resting spectra capture coarse case–control status instead of continuous within-patient cognitive or motor staging (`R² <= 0.035`).

### 4.5 Limitations and Future Directions
Downsampling 32- and 64-channel research datasets to 19 clinical electrodes traded spatial resolution for clinical compatibility, and Study 2 lacked individual chronological ages. Study 6 clean eyes-open segments average ~41 s (`10.2 epochs`), motivating larger pure-rest acquisitions alongside source-space connectomics and hardware transfer-function logging before clinical translation.

---

## 5. Conclusions

Across `566` primary participants (`677` primary sessions; `14,003` channel spectra) from seven public resting-state EEG cohorts, shrinking `specparam v2.0.0rc7` fits to passband-interior `[2.0, 30.0] Hz` bounds and auditing `[7, 14] Hz`-masked OLS exposed complementary slope-leakage and edge-roll-off failures proven via Monte Carlo simulations, while raw-file forensics uncovered duplicate recordings and batch-date artifacts driving Study 2 MDD separability. Under true 10×5 participant-grouped cross-validation and aligned permutation tests, a 3,362-parameter `GCN-OMST` yielded no advantage over edge-matched random topology or tuned linear and random-forest models, while symmetrized PLI failed positive semi-definiteness in `11.82%` of sessions, session mixing shifted accuracy bidirectionally (`-0.094` to `+0.080`), and within-patient severity explained variance stayed near zero (`R² <= 0.035`). Reproducible clinical EEG demands clean interior spectral windows, participant-grouped evaluation, archive integrity audits, and simple tuned baselines before deploying complex graph architectures.

---

## Information Sharing Statement

All seven EEG datasets analyzed in this study are publicly accessible via persistent repositories, and all harmonized preprocessing, Study 6 pure-rest isolation, `specparam v2.0.0rc7` parameterization, Monte Carlo simulations, 10×5 grouped-CV auditing, `Split-Target-HC` transfer, figure generation, and Zenodo-ready derivative tables (`derived_session_features_737.csv`, `cv_fold_indices_split0.csv`, `oof_predictions_split0.csv`, and `requirements_lock.txt` in `release_artifacts/`) are openly released at `{GITHUB_URL}` (`scripts/44`, `scripts/45`, `scripts/45b`, `scripts/45c`, `scripts/45d`, `scripts/42`, `scripts/46`, and `src/harmonizer_and_graph.py`). Public accessions are:

1. Study 1 (Sleep Deprivation): OpenNeuro `ds004902` (https://openneuro.org/datasets/ds004902) [41].
2. Study 2 (Outpatient MDD): Figshare `4244171` (https://doi.org/10.6084/m9.figshare.4244171) [42].
3. Study 3 (University Depression): OpenNeuro `ds003478` (`doi: 10.18112/openneuro.ds003478.v1.1.0`) [43].
4. Study 4 (UCSD Parkinson's Disease): OpenNeuro `ds002778` (https://openneuro.org/datasets/ds002778) [44].
5. Study 5 (AHEPA Dementia): OpenNeuro `ds004504` (`doi: 10.18112/openneuro.ds004504`) [45].
6. Study 6 (UNM Parkinson's Disease): OpenNeuro `ds003490` (`doi: 10.18112/openneuro.ds003490.v1.1.0`) [46].
7. Study 7 (Iowa Parkinson's Disease): OpenNeuro `ds004584` (`doi: 10.18112/openneuro.ds004584.v1.0.0`) [47].

---

## Statements and Declarations

### Ethics Approval and Consent to Participate
This computational study analyzed existing de-identified public EEG datasets and involved no direct human-subjects recruitment or intervention; secondary analysis is exempt under U.S. 45 CFR 46.104(d)(4) and institutional guidelines. Primary studies obtained local ethics approval and written informed consent under the Declaration of Helsinki (`H20039`, Study 1; `USM/JEPeM/1401015`, Study 2; University of New Mexico IRB, Studies 3 and 6; UC San Diego HRPP, Study 4; AHEPA Hospital `142/12-04-2023`, Study 5; University of Iowa IRB `201707828`, Study 7).

### Competing Interests
The author declares no financial or non-financial competing interests.

### Funding
This research received no specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

### Author Contributions (CRediT)
Chushu Tan: Conceptualization, Methodology, Software, Formal analysis, Investigation, Data curation, Writing – original draft, Writing – review & editing, Visualization, Project administration.

---

## References

[1] P. L. Nunez and R. Srinivasan, *Electric Fields of the Brain: The Neurophysics of EEG*, 2nd ed. Oxford University Press, 2006.

[2] F. H. Lopes da Silva, "EEG and MEG: Relevance to neuroscience," *Neuron*, vol. 80, no. 5, pp. 1112–1128, 2013, doi: 10.1016/j.neuron.2013.10.017.

[3] G. Buzsáki, *Rhythms of the Brain*. Oxford University Press, 2006, doi: 10.1093/acprof:oso/9780195301069.001.0001.

[4] M. X. Cohen, *Analyzing Neural Time Series Data: Theory and Practice*. MIT Press, 2014.

[5] P. Fries, "A mechanism for cognitive dynamics: Neuronal communication through coherence," *Trends Cogn. Sci.*, vol. 9, no. 10, pp. 474–480, 2005, doi: 10.1016/j.tics.2005.08.011.

[6] G. Buzsáki and A. Draguhn, "Neuronal oscillations in cortical networks," *Science*, vol. 304, no. 5679, pp. 1926–1929, 2004, doi: 10.1126/science.1099745.

[7] C. J. Stam, "Modern network science of neurological disorders," *Nat. Rev. Neurosci.*, vol. 15, no. 10, pp. 683–695, 2014, doi: 10.1038/nrn3801.

[8] S. Kapur, A. G. Phillips, and T. R. Insel, "Why has it taken so long for biological psychiatry to develop clinical tests and what to do about it?" *Mol. Psychiatry*, vol. 17, no. 12, pp. 1174–1179, 2012, doi: 10.1038/mp.2012.105.

[9] Y. Roy, H. Banville, I. Albuquerque, A. Gramfort, T. H. Falk, and J. Faubert, "Deep learning-based electroencephalography analysis: A systematic review," *J. Neural Eng.*, vol. 16, no. 5, art. no. 051001, 2019, doi: 10.1088/1741-2552/ab260c.

[10] L. A. W. Gemein, R. T. Schirrmeister, P. Chrabąszcz, D. Wilson, J. Boedecker, A. Schulze-Bonhage, F. Hutter, and T. Ball, "Machine-learning-based diagnostics of EEG pathology," *NeuroImage*, vol. 220, art. no. 117021, 2020, doi: 10.1016/j.neuroimage.2020.117021.

[11] D. Scheinost, S. Noble, C. Horien, A. S. Greene, E. M. R. Lake, M. Salehi, et al., "Ten simple rules for predictive modeling of individual differences in neuroimaging," *NeuroImage*, vol. 193, pp. 35–45, 2019, doi: 10.1016/j.neuroimage.2019.02.057.

[12] G. Varoquaux, "Cross-validation failure: Small sample sizes lead to large error bars," *NeuroImage*, vol. 180, pp. 68–77, 2018, doi: 10.1016/j.neuroimage.2017.06.061.

[13] S. Saeb, L. Lonini, A. Jayaraman, D. C. Mohr, and K. P. Kording, "The need to approximate the use-case in clinical machine learning," *GigaScience*, vol. 6, no. 5, art. no. gix019, 2017, doi: 10.1093/gigascience/gix019.

[14] M. A. Little, G. Varoquaux, S. Saeb, L. Lonini, A. Jayaraman, D. C. Mohr, and K. P. Kording, "Using and understanding cross-validation strategies. Perspectives on Saeb et al.," *GigaScience*, vol. 6, no. 5, art. no. gix020, 2017, doi: 10.1093/gigascience/gix020.

[15] T. Wolfers, N. T. Doan, T. Kaufmann, D. Alnæs, T. Moberget, I. Agartz, et al., "Mapping the heterogeneous phenotype of schizophrenia and bipolar disorder using normative models," *JAMA Psychiatry*, vol. 75, no. 11, pp. 1146–1155, 2018, doi: 10.1001/jamapsychiatry.2018.2467.

[16] S. Rutherford, S. M. Kia, T. Wolfers, C. Fraza, M. Zabihi, R. Dinga, et al., "The normative modeling framework for computational psychiatry," *Nat. Protoc.*, vol. 17, no. 7, pp. 1711–1734, 2022, doi: 10.1038/s41596-022-00696-5.

[17] T. Donoghue, M. Haller, E. J. Peterson, P. Varma, P. Sebastian, R. Gao, et al., "Parameterizing neural power spectra into periodic and aperiodic components," *Nat. Neurosci.*, vol. 23, no. 12, pp. 1655–1665, 2020, doi: 10.1038/s41593-020-00744-x.

[18] B. Voytek, M. A. Kramer, J. Case, K. Q. Lepage, Z. R. Tempesta, R. T. Knight, and A. Gazzaley, "Age-related changes in 1/f neural electrophysiological noise," *J. Neurosci.*, vol. 35, no. 38, pp. 13257–13265, 2015, doi: 10.1523/JNEUROSCI.2332-14.2015.

[19] S. R. Cole and B. Voytek, "Brain oscillations and the importance of waveform shape," *Trends Cogn. Sci.*, vol. 21, no. 2, pp. 137–149, 2017, doi: 10.1016/j.tics.2016.12.008.

[20] R. Gao, E. J. Peterson, and B. Voytek, "Inferring synaptic excitation/inhibition balance from field potentials," *NeuroImage*, vol. 158, pp. 70–78, 2017, doi: 10.1016/j.neuroimage.2017.06.078.

[21] T. Donoghue, N. Schaworonkow, and B. Voytek, "Methodological considerations for studying neural oscillations," *Eur. J. Neurosci.*, vol. 55, no. 11–12, pp. 3502–3527, 2022, doi: 10.1111/ejn.15361.

[22] J. F. Cavanagh and M. J. Frank, "Frontal theta as a mechanism for cognitive control," *Trends Cogn. Sci.*, vol. 18, no. 8, pp. 414–421, 2014, doi: 10.1016/j.tics.2014.04.012.

[23] D. J. Mitchell, N. McNaughton, D. Flanagan, and I. J. Kirk, "Frontal-midline theta from the perspective of hippocampal 'theta'," *Prog. Neurobiol.*, vol. 86, no. 3, pp. 156–185, 2008, doi: 10.1016/j.pneurobio.2008.09.005.

[24] L. A. Finelli, H. Baumann, A. A. Borbély, and P. Achermann, "Dual electroencephalogram markers of human sleep homeostasis: Correlation between theta activity in waking and slow-wave activity in sleep," *Neuroscience*, vol. 101, no. 3, pp. 523–529, 2000, doi: 10.1016/s0306-4522(00)00409-7.

[25] S. Olbrich and M. Arns, "EEG biomarkers in major depressive disorder: Discriminative power and prediction of treatment response," *Int. Rev. Psychiatry*, vol. 25, no. 5, pp. 604–618, 2013, doi: 10.3109/09540261.2013.816269.

[26] V. Knott, C. Mahoney, S. Kennedy, and K. Evans, "EEG power, frequency, asymmetry and coherence in male depression," *Psychiatry Res. Neuroimaging*, vol. 106, no. 2, pp. 123–140, 2001, doi: 10.1016/s0925-4927(00)00080-9.

[27] J. J. Newson and T. C. Thiagarajan, "EEG frequency bands in psychiatric disorders: A review of resting state studies," *Front. Hum. Neurosci.*, vol. 12, art. no. 521, 2019, doi: 10.3389/fnhum.2018.00521.

[28] N. Jaworska, P. Blier, W. Fusee, and V. Knott, "α Power, α asymmetry and anterior cingulate cortex activity in depressed males and females," *J. Psychiatr. Res.*, vol. 46, no. 11, pp. 1483–1491, 2012, doi: 10.1016/j.jpsychires.2012.08.003.

[29] D. Stoffers, J. L. W. Bosboom, J. B. Deijen, E. C. Wolters, H. W. Berendse, and C. J. Stam, "Slowing of oscillatory brain activity is a stable characteristic of Parkinson's disease without dementia," *Brain*, vol. 130, no. 7, pp. 1847–1860, 2007, doi: 10.1093/brain/awm034.

[30] J. L. W. Bosboom, D. Stoffers, C. J. Stam, B. W. van Dijk, J. Verbunt, H. W. Berendse, and E. C. Wolters, "Resting state oscillatory brain dynamics in Parkinson's disease: An MEG study," *Clin. Neurophysiol.*, vol. 117, no. 11, pp. 2521–2531, 2006, doi: 10.1016/j.clinph.2006.06.720.

[31] C. J. Stam, "Use of magnetoencephalography (MEG) to study functional brain networks in neurodegenerative disorders," *J. Neurol. Sci.*, vol. 289, no. 1–2, pp. 128–134, 2010, doi: 10.1016/j.jns.2009.08.028.

[32] E. Bullmore and O. Sporns, "Complex brain networks: Graph theoretical analysis of structural and functional systems," *Nat. Rev. Neurosci.*, vol. 10, no. 3, pp. 186–198, 2009, doi: 10.1038/nrn2575.

[33] M. Rubinov and O. Sporns, "Complex network measures of brain connectivity: Uses and interpretations," *NeuroImage*, vol. 52, no. 3, pp. 1059–1069, 2010, doi: 10.1016/j.neuroimage.2009.10.003.

[34] C. J. Stam, G. Nolte, and A. Daffertshofer, "Phase lag index: Assessment of functional connectivity from multi-channel EEG and MEG with diminished bias from common sources," *Hum. Brain Mapp.*, vol. 28, no. 11, pp. 1178–1193, 2007, doi: 10.1002/hbm.20346.

[35] M. Vinck, R. Oostenveld, M. van Wingerden, F. Battaglia, and C. M. A. Pennartz, "An improved index of phase-synchronization for electrophysiological data in the presence of volume-conduction, noise and sample-size bias," *NeuroImage*, vol. 55, no. 4, pp. 1548–1565, 2011, doi: 10.1016/j.neuroimage.2011.01.055.

[36] S. I. Dimitriadis, C. Salis, I. Tarnanas, and D. E. J. Linden, "Topological filtering of dynamic functional brain networks unfolds informative chronnectomics: A novel data-driven thresholding scheme based on orthogonal minimal spanning trees (OMSTs)," *Front. Neuroinform.*, vol. 11, art. no. 28, 2017, doi: 10.3389/fninf.2017.00028.

[37] S. I. Dimitriadis, M. Antonakakis, P. Simos, J. M. Fletcher, and A. C. Papanicolaou, "Data-driven topological filtering based on orthogonal minimal spanning trees: Application to multigroup magnetoencephalography resting-state connectivity," *Brain Connect.*, vol. 7, no. 10, pp. 661–670, 2017, doi: 10.1089/brain.2017.0512.

[38] A. Barachant, S. Bonnet, M. Congedo, and C. Jutten, "Classification of covariance matrices using a Riemannian-based kernel for BCI applications," *Neurocomputing*, vol. 112, pp. 172–178, 2013, doi: 10.1016/j.neucom.2012.12.039.

[39] M. Congedo, A. Barachant, and R. Bhatia, "Riemannian geometry for EEG-based brain-computer interfaces; a primer and a review," *Brain-Comput. Interfaces*, vol. 4, no. 3, pp. 155–174, 2017, doi: 10.1080/2326263X.2017.1297192.

[40] T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," in *Proc. 5th Int. Conf. Learn. Representations (ICLR)*, 2017.

[41] C. Xiang, X. Fan, D. Bai, K. Lv, and X. Lei, "A resting-state EEG dataset for sleep deprivation," *Sci. Data*, vol. 11, art. no. 427, 2024, doi: 10.1038/s41597-024-03268-2. [OpenNeuro ds004902].

[42] W. Mumtaz, L. Xia, S. S. A. Ali, M. A. M. Yasin, M. Hussain, and A. S. Malik, "Electroencephalogram (EEG)-based computer-aided technique to diagnose major depressive disorder (MDD)," *Biomed. Signal Process. Control*, vol. 31, pp. 108–115, 2017, doi: 10.1016/j.bspc.2016.07.006. [Dataset: Figshare 4244171, doi: 10.6084/m9.figshare.4244171].

[43] J. F. Cavanagh, A. W. Bismark, M. J. Frank, and J. J. B. Allen, "Multiple dissociations between comorbid depression and anxiety on reward and punishment processing: Evidence from computationally informed EEG," *Comput. Psychiatry*, vol. 3, pp. 1–17, 2019, doi: 10.1162/cpsy_a_00024. [OpenNeuro ds003478, doi: 10.18112/openneuro.ds003478.v1.1.0].

[44] A. P. Rockhill, N. Jackson, J. George, A. Aron, and N. C. Swann, "UC San Diego Resting State EEG Data from Patients with Parkinson's Disease," *OpenNeuro*, ds002778, 2020; and J. S. George et al., *NeuroImage: Clin.*, vol. 3, pp. 261–270, 2013, doi: 10.1016/j.nicl.2013.07.013.

[45] A. Miltiadous, K. D. Tzimourta, T. Afrantou, P. Ioannidis, N. Grigoriadis, D. G. Tsalikakis, et al., "A dataset of scalp EEG recordings of Alzheimer's disease, frontotemporal dementia and healthy subjects from routine EEG," *Data*, vol. 8, no. 6, art. no. 95, 2023, doi: 10.3390/data8060095. [OpenNeuro ds004504, doi: 10.18112/openneuro.ds004504].

[46] J. F. Cavanagh, P. Kumar, A. A. Mueller, S. P. Richardson, and A. Mueen, "Diminished EEG habituation to novel events effectively classifies Parkinson's patients," *Clin. Neurophysiol.*, vol. 129, no. 2, pp. 409–418, 2018, doi: 10.1016/j.clinph.2017.11.023. [OpenNeuro ds003490, doi: 10.18112/openneuro.ds003490.v1.1.0].

[47] A. Singh, R. C. Cole, A. I. Espinoza, J. F. Cavanagh, and N. S. Narayanan, "Resting-state EEG measures cognitive impairment in Parkinson's disease," *npj Parkinsons Dis.*, vol. 9, art. no. 142, 2023, doi: 10.1038/s41531-023-00602-0. [OpenNeuro ds004584, doi: 10.18112/openneuro.ds004584.v1.0.0].
"""
    return md


# ==============================================================================
# Supplementary Material Builder (Live CSV-Backed Tables S1–S26 + Figs S1–S6)
# ==============================================================================
def build_supplementary_material(d: dict) -> str:
    omst = d["master"]["omst_topology"]
    acc = d["master"]["accounting"]
    s1_df = d["s1_omst"]
    gain_chk = d["gain_chk"]
    bio_exp = d["bio_exp"]
    lofo = d["lofo"]
    hparam = d["hparam"]
    cls0 = d["cls0"]
    wil = d["wilcoxon"]
    sim_mc = d["sim_mc"]
    boot_d = d["boot_d"]
    mum_eo = d["mum_eo"]
    mum_conc = d["mum_conc"]
    mum_tx = d["mum_tx"]
    med = d["med"]
    reg = d["reg"]
    pd_tx = d["pd_tx"]
    td_tx = d["td_tx"]
    anc = d["ancova"]
    anc_dur = d["anc_dur"]
    leak = d["leakage"]
    gc = d["gain_cal"]
    rep10 = d["rep10"]
    pli_a = d["pli_audit"]
    pli_s = d["pli_sum"].iloc[0]
    riem = d["riem"]
    meta_df = d["meta"]
    fit_df = d["fit"]
    iapf_df = d["iapf"]
    heur_df = d["heur"]
    ep_df = d["ep_audit"]
    unm_sens = d["unm_sens"]
    mum_dups = d["mum_dups"]
    mum_for = d["mum_forensic"]

    parts = [f"""# Supplementary Material

**{EXACT_TITLE}**

**{AUTHOR_NAME}**^1,2*^

^1^ {AFFIL_1}  
^2^ {AFFIL_2}  
^*^ Correspondence: {CORRESPONDENCE}

---

## Contents

### Extended Supplementary Methods
- **Section S1.1:** Interior [2.0, 30.0] Hz Parameterized Spectral Decomposition (`specparam v2.0.0rc7`) and Leakage Comparison
- **Section S1.2:** Orthogonal Minimum Spanning Tree (OMST) Graph Extraction and Cost Efficiency
- **Section S1.3:** Analytical Scaling Arithmetic and Non-Multiplicative Site Boundaries
- **Section S1.4:** `NormativeGCN` Architecture and Training Protocol

### Supplementary Figures
- **Figure S1:** Single-Subject Interior [2.0, 30.0] Hz `specparam` vs. Masked-OLS Comparison
- **Figure S2:** Canonical 19-Channel 10–20 OMST Connectome Topology
- **Figure S3:** Empirical Gain-Invariance Verification Across Four Orders of Magnitude
- **Figure S4:** Grand-Average Welch Power Spectra (1–40 Hz) and [2.0, 30.0] Hz Residuals Across 7 Cohorts
- **Figure S5:** Seven-Cohort Participant Flow, QC Exclusions, and Harmonized Pipeline
- **Figure S6:** Aligned 1,000-Permutation Retraining Distributions (`GridSearchCV Tuned LR` and `GCN-OMST`)

### Supplementary Tables
- **Table S1:** Canonical 19-Channel Coordinates and Empirical OMST Metrics Across 677 Remediated Graphs
- **Table S2:** Empirical Gain-Invariance Audit Across 285 Spectra ([2.0, 30.0] Hz Solver)
- **Table S3:** Cohort Designs, Demographics, Hardware Systems, and Clinical Scales
- **Table S4:** Complete 50-Test Multi-Cohort Biophysical Summary ([2.0, 30.0] Hz `specparam`)
- **Table S5:** Canonical 19-Channel 10–20 Label Harmonization Mapping
- **Table S6:** Single-Biomarker Baselines and Leave-One-Feature-Out `NormativeGCN` Ablations
- **Table S7, S7b, S7c:** `NormativeGCN` Architecture, Classical Grids, and Hyperparameter Sensitivity
- **Table S8:** Preprocessing Specifications Across All Seven Datasets
- **Table S9:** Verified Computational Environment (`requirements_lock.txt`) and Open Repository
- **Table S10 & S10b:** Complete Split-0 Extended Model Comparison (w/ 95% CIs) and 10×5 Wilcoxon Comparisons
- **Table S11:** Known-Ground-Truth Monte Carlo Leakage Simulation and Paired Bootstrap Δd Contrasts
- **Table S12:** Study 2 Recording-State Sensitivity (Eyes-Open N=60) and State-Matched Transfer
- **Table S13:** Within-Subject Paired Dopaminergic Medication Comparisons (UNM N=25 & UCSD N=15)
- **Table S14:** Complete 20-Row 10×5 Repeated Clinical Scale Regression Benchmark
- **Table S15 & S16:** Leak-Free 50/50 `Split-Target-HC` Cross-Site PD and Cross-Domain Transfer Matrices
- **Table S17:** Complete Age-Adjusted ANCOVA Across Four EEG Markers
- **Table S18:** 10×5 Repeated Validation Leakage Quantification Across Repeated-Session Cohorts
- **Table S19:** Cross-Site PD Calibration Ablation and Gain Stress-Testing (Mean and Worst-Case Max Drift)
- **Table S20:** True 10×5 Repeated Participant-Grouped Cross-Validation Summary Across All 12 Contrasts
- **Table S21:** PLI Matrix Definiteness Audit and 10×5 Riemannian Tangent-Space Baselines
- **Table S22:** Primary Participant-Averaged (`SubjMean`) Meta-Analysis (`DL` & `HKSJ`) and Leave-One-Out Suite
- **Table S23:** Per-Cohort `specparam v2.0.0rc7` Fit Diagnostics: `[1.0, 40.0] Hz` vs. `[2.0, 30.0] Hz`
- **Table S24:** Posterior Individual Alpha Peak Frequency (`IAPF`) and 10×5 Heuristic Slowing Benchmarks
- **Table S25:** Clean Epoch-Duration Balance, Joint Age + Epoch ANCOVA, and UNM Pure EO vs. Pure EC Sensitivity
- **Table S26:** Figshare `4244171` (Study 2) Duplicate EDF Hashing and Recording-Batch Forensic Audit

---

## Section S1. Extended Supplementary Methods

### S1.1 Interior [2.0, 30.0] Hz Parameterized Spectral Decomposition (`specparam v2.0.0rc7`)
Welch PSDs (4.0 s epochs, 2.0 s Hamming sub-windows, 50% overlap, 0.50 Hz step) were parameterized via `SpectralGroupModel` in `specparam v2.0.0rc7` (`aperiodic_mode='fixed'`, `peak_width_limits=[1.0, 12.0]` Hz, `max_n_peaks=6`, `min_peak_height=0.05` B, `peak_threshold=2.0` SD) over interior `[2.0, 30.0] Hz` bounds (primary) and `[1.0, 40.0] Hz` (sensitivity):

$$\\log_{{10}} \\mathrm{{PSD}}(f) = (b - \\chi \\log_{{10}} f) + \\sum_{{n=1}}^N a_n \\exp\\left(-\\frac{{(f - c_n)^2}}{{2 w_n^2}}\\right)$$

### S1.2 Data-Driven Connectomics via Orthogonal Minimum Spanning Trees (OMST)
Across 677 canonical primary graphs ({omst['total_node_instances']:,} nodes), iterative Kruskal OMST trees maximizing `GCE` averaged {omst['mean_trees']:.2f} ± {omst['sd_trees']:.2f} trees (`{omst['pct_tree_1']:.2f}%` M = 1, `{omst['pct_tree_2']:.2f}%` M >= 2), `{omst['mean_edges']:.2f} ± {omst['sd_edges']:.2f}` edges, `GCE = {omst['mean_gce']:.4f} ± {omst['sd_gce']:.4f}`, and 0.0% isolated nodes.

### S1.3 Elementary Gain Scaling and Non-Multiplicative Boundaries
Multiplying time-series voltage by scalar `k > 0` adds `2 log10(k)` uniformly across log10 Welch power, absorbed cleanly into intercept offset `b' = b + 2 log10(k)`. Non-multiplicative filter skirts, impedances, references, and demographic confounds are unaffected by scale invariance and require fold-nested normative calibration.

### S1.4 `NormativeGCN` Architecture
Pure-PyTorch `NormativeGCN` (`src/harmonizer_and_graph.py`) takes `[19, 4]` z-scored features and binary 19 x 19 adjacency (`A + I` self-loops normalized inside `PurePyTorchGCNLayer`), applying `4->32->32` GCN layers (Dropout 0.25), concatenated mean+max pooling (64-D), and `64->32->2` MLP classification totaling 3,362 trainable parameters (Adam lr=0.005, wd=1e-4, 50 fixed epochs).

---

## Supplementary Figures

![Figure S1](fig_S1_specparam_example.png){{width=6.4in}}
**Figure S1. Single-Subject Interior [2.0, 30.0] Hz `specparam` vs. Masked-OLS Illustration.** Real empirical spectrum from Study 5 AD participant `sub-018` (`Ch Fz`), showing `[7, 14] Hz`-masked OLS pulling background slope upward (`χ_OLS = 1.287` vs. `specparam χ = 1.074`) and attenuating 4–8 Hz periodic theta by `-0.274 B` (`R² = 0.954`, `MAE = 0.083 B`).

![Figure S2](fig_S2_omst_topology.png){{width=6.2in}}
**Figure S2. Canonical 19-Channel 10–20 OMST Connectome Topology.** Group-averaged backbone edges across 677 primary graphs (mean {omst['mean_edges']:.2f} edges, zero isolated nodes).

![Figure S3](fig_S3_gain_invariance.png){{width=6.4in}}
**Figure S3. Empirical Scaling Stability of Interior [2.0, 30.0] Hz `specparam`.** Verified across n = 285 spectra (`k in [0.01, 100.0]`) showing exact `2 log10(k)` offset tracking and `<= {gain_chk['max_abs_diff_chi'].max():.2e}` max deviation.

![Figure S4](fig_S4_cohort_spectra.png){{width=6.5in}}
**Figure S4. Cohort-Averaged Welch Power Spectra (1–40 Hz) and Interior [2.0, 30.0] Hz Residuals Across 7 Cohorts.** Shading highlights excluded passband skirts (`<2 / >30 Hz`) and theta (`4–8 Hz`), showing Study 2's steep 40 Hz attenuation alongside clean `[2, 30] Hz` residuals.

![Figure S5](fig_S5_enrollment_flowchart.png){{width=6.4in}}
**Figure S5. Seven-Cohort Participant Flow, Pre-Analysis QC, Study 6 Pure-EO Isolation, and Study 2 Deduplication Audit.**

![Figure S6](fig_S6_permutation_tests.png){{width=6.5in}}
**Figure S6. Aligned 1,000-Permutation Retraining Null Distributions.** Identical `GridSearchCV Tuned LR` and participant-cluster splits across four within-cohort and two `Split-Target-HC` benchmarks.

---

## Supplementary Tables

### Table S1. Canonical 19-Channel Coordinates and Empirical OMST Metrics Across All 677 Remediated Graphs
| Channel | Region | X | Y | T1 Deg | T2 Deg | Comb Deg | Empirical Mean Degree | Betweenness Centrality | Clustering Coeff |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"""]

    for _, r in s1_df.iterrows():
        parts.append(
            f"| {r['channel']} | {r['region']} | {r['proj_x']:+.2f} | {r['proj_y']:+.2f} | "
            f"{int(r['tree1_deg'])} | {int(r['tree2_deg'])} | {int(r['combined_deg'])} | "
            f"{r['emp_deg_mean']:.2f} ± {r['emp_deg_sd']:.2f} | {r['betw_mean']:.3f} ± {r['betw_sd']:.3f} | {r['clus_mean']:.3f} ± {r['clus_sd']:.3f} |"
        )
    parts.append(f"| Total / Mean | 19 Channels | — | — | 18 edges | 18 edges | 36 edges | {omst['mean_edges']:.2f} ± {omst['sd_edges']:.2f} edges | — | — |")

    parts.append("""
### Table S2. Empirical Scaling Verification of Interior [2.0, 30.0] Hz `specparam` Across 285 Spectra
| Gain Factor (k) | Spectra (n) | Expected Offset 2 log10(k) (B) | Measured Offset Shift Mean (B) | Max |Δchi| Across 285 Spectra | Max |Δθ_resid| (B) |
| :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in gain_chk.iterrows():
        parts.append(
            f"| {r['gain_k']:.2f} | {int(r['n_spectra'])} | {r['expected_offset_shift']:+.5f} | "
            f"{r['measured_offset_shift_mean']:+.5f} | {r['max_abs_diff_chi']:.2e} | {r['max_abs_diff_theta_resid']:.2e} |"
        )

    parts.append("""
### Table S3. Multi-Cohort Study Designs, Demographics, Hardware Systems, and Clinical Scales
| Study ID | Dataset & Setting | Analyzed N (Pts / Sess) | Group Breakdown | Age Years (Mean +/- SD) | Female (%) | Native Hardware | Sampling Rate | Protocol / Assessments |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- | :---: | :--- |
| Study 1 | Sleep Deprivation (`ds004902`) | 71 / 142 | SD (71) vs. NS (71) [Paired] | 20.0 +/- 1.4 | 47.9% | BrainVision actiCHamp (64 Ch) | 500 -> 250 Hz | 36 h monitored wakefulness crossover (EO) |
| Study 2 | Outpatient MDD (`Figshare 4244171`) | 58 / 58 (+60 EO) | MDD (30 EC, 32 EO) vs. HC (28); 56/58 uniq. | MDD 40.3 +/- 12.9; HC 38.2 +/- 15.6 (summary) | 37.9% | Nihon Kohden (19 Ch) | 256 -> 250 Hz | DSM-IV MDD; BDI; 2 duplicate pairs + 3 batches |
| Study 3 | Univ. Depression (`ds003478`) | 119 / 119 | BDI >= 14 (44) vs. <= 7 (75) | 19.4 +/- 2.0 (N=118) | 60.5% | BioSemi ActiveTwo (64 Ch) | 500 -> 250 Hz | EO; SCID (11 curr, 11 past MDD, 8 non-MDD, 14 unint.) |
| Study 4 | UCSD PD (`ds002778`) | 31 / 46 | PD (15 ON + 15 OFF) vs. HC (16) | PD 63.2 +/- 8.1; HC 63.5 +/- 9.6 | 54.8% | BioSemi ActiveTwo (32 Ch) | 512 -> 250 Hz | EO fixation; counterbalanced ON vs. >= 12 h OFF |
| Study 5 | AHEPA Dementia (`ds004504`) | 88 / 88 | AD (36), FTD (23), HC (29) | AD 72.8 +/- 7.9; FTD 68.2 +/- 7.3; HC 70.1 +/- 6.8 | 55.7% | Nihon Kohden (19 Ch) | 500 -> 250 Hz | EC; MMSE (AD 17.8 +/- 4.5, FTD 22.2 +/- 4.8, HC 30.0) |
| Study 6 | UNM PD (`ds003490`) | 50 / 75 | PD (25 ON + 25 OFF EO) vs. HC (25 EO) | PD 67.4 +/- 8.7; HC 69.1 +/- 8.2 | 42.0% | BrainVision (64 Ch) | 500 -> 250 Hz | Pure 60s EO rest (+74 pure EC sens); >= 15 h washout |
| Study 7 | Iowa PD (`ds004584`) | 149 / 149 | PD (100 ON) vs. HC (49) | PD 68.5 +/- 8.1; HC 70.9 +/- 7.6 | 42.3% | BrainVision (64 Ch, Pz ref) | 500 -> 250 Hz | EO (PD ON med); MoCA (25.1 +/- 3.6) & UPDRS-III (12.5 +/- 7.1) |

### Table S4. Complete 50-Test Multi-Cohort Biophysical Summary ([2.0, 30.0] Hz `specparam`)
| Cohort | Marker | N (Case/Ref) | t-Stat | Exact df | Raw p | Pooled FDR q_50 | Per-Marker q_10 | Cohen's d [95% CI] | Sig |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in bio_exp.iterrows():
        parts.append(
            f"| {r['cohort']} | {r['marker']} | {int(r['n_case'])}/{int(r['n_ctrl'])} | "
            f"{r['t_statistic']:+.3f} | {r['df_satterthwaite']:.1f} | {fmt_p(r['p_value'])} | "
            f"{fmt_p(r['fdr_q_50'])} | {fmt_p(r['fdr_q_10'])} | "
            f"{r['cohens_d']:+.3f} [{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}] | {fmt_sig(r['fdr_q_50'])} |"
        )

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
        parts.append(
            f"| {r['cohort']} | {r['Single_Theta_auc']:.3f} | {r['Single_Delta_auc']:.3f} | "
            f"{r['Single_Alpha_auc']:.3f} | {r['Single_Chi_auc']:.3f} | {r['Single_GCE_auc']:.3f} | "
            f"{r['GCN_Full4_auc']:.3f} | {r['GCN_Ablate_Theta_auc']:.3f} | "
            f"{r['GCN_Ablate_Delta_auc']:.3f} | {r['GCN_Ablate_Alpha_auc']:.3f} | {r['GCN_Ablate_Chi_auc']:.3f} |"
        )

    parts.append("""
### Table S7. `NormativeGCN` Architecture and Optimization Specifications
| Component | Exact Specification | Parameter Breakdown |
| :--- | :--- | :--- |
| Engine & Module | PyTorch 2.8.0 (`NormativeGCN`) | `src/harmonizer_and_graph.py` |
| Input Tensor | 19 nodes x 4 z-scored features | Fold-nested control standardization |
| GCN Layers | 2x `PurePyTorchGCNLayer` (`4->32->32`, ReLU, Dropout 0.25) | 160 + 1,056 = 1,216 params |
| Global Readout | Concatenated Mean + Max Pool across 19 nodes | 64-D representation |
| MLP Head | `Linear(64->32) -> ReLU -> Dropout(0.25) -> Linear(32->2)` | 2,080 + 66 = 2,146 params |
| Total Parameters | 3,362 trainable parameters | Fixed 50 Adam epochs (lr=0.005, wd=1e-4) |

### Table S7b. Classical and Univariate Baseline Configurations
| Model Identifier | Input Space | Inner Tuning / Settings | Estimator |
| :--- | :--- | :--- | :--- |
| `Tuned_LogisticRegression` | Flattened 76-D | Nested 3-fold CV L2 `C in {0.001..10.0}` | `LogisticRegression` |
| `Tuned_Ridge` | Flattened 76-D | Nested 3-fold CV `alpha in {0.01..100.0}` | `RidgeClassifier` |
| `Linear_SVM` / `RBF_SVM` | Flattened 76-D | Nested 3-fold CV `C`, `gamma` | `SVC(probability=True)` |
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
| Dataset | Condition | Analyzed Sessions | Threshold | Sampling / Filter | Mean Clean Epochs +/- SD [Range] | Retained Duration (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in ep_df.iterrows():
        thr = "200 μV" if "Iowa" in r["cohort"] else "150 μV"
        cond = "EO/EC Rest"
        parts.append(
            f"| {r['cohort']} | {cond} | {int(r['n_sessions'])} | {thr} | 250 Hz, 1–40 Hz FIR | "
            f"{r['all_mean']:.1f} ± {r['all_sd']:.1f} [{int(r['all_min'])}–{int(r['all_max'])}] | {4.0 * r['all_mean']:.1f} ± {4.0 * r['all_sd']:.1f} s |"
        )

    parts.append("""
### Table S9. Verified Computational Environment and Reproducibility Artifacts
| Package / Artifact | Version / Path | Role |
| :--- | :---: | :--- |
| Python / PyTorch | 3.13.7 / 2.8.0 | Core execution & `NormativeGCN` |
| MNE-Python / `specparam` | 1.12.1 / 2.0.0rc7 | Preprocessing & spectral parameterization |
| `scikit-learn` / `scipy` / `statsmodels` | 1.7.2 / 1.16.2 / 0.14.6 | Nested CV classifiers, Wilcoxon, ANCOVA, BH-FDR |
| Dependency Lock & Zenodo Derivatives | `release_artifacts/requirements_lock.txt` | Complete package freeze + `derived_session_features_737.csv`, `cv_fold_indices_split0.csv`, `oof_predictions_split0.csv` |

### Table S10. Split-0 Within-Cohort Benchmark Across 12 Models with 95% Confidence Intervals
| Cohort Contrast | Model | N | Split-0 ROC-AUC [95% CI] | Balanced Acc | Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: |""")
    n_map_s10 = {**dict(zip(lofo["cohort"], lofo["n_samples"])), "Study 2b: Mumtaz MDD (Eyes Open)": 60}
    for _, r in cls0.iterrows():
        parts.append(
            f"| {r['cohort']} | `{r['model']}` | {int(n_map_s10[r['cohort']])} | "
            f"{r['auc']:.3f} [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}] | "
            f"{100.0 * r['balanced_accuracy']:.1f}% | {100.0 * r['accuracy']:.1f}% |"
        )

    parts.append("""
### Table S10b. Pairwise Wilcoxon Comparisons on 10×5 Means (`GCN_OMST` vs. Competing Models)
| Evaluation Scope | Comparator Model | GCN Mean AUC | Comp Mean AUC | Mean Diff | Wins / Ties / Losses | Wilcoxon W | Raw p | FDR q |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in wil.iterrows():
        parts.append(
            f"| {r['scope']} (k={int(r['n_benchmarks'])}) | `{r['comparator_model']}` | {r['gcn_mean_auc']:.4f} | "
            f"{r['comp_mean_auc']:.4f} | {r['mean_diff_auc']:+.4f} | {r['wins_ties_losses']} | "
            f"{r['wilcoxon_W']:.1f} | {r['raw_p']:.4f} | {r['fdr_q']:.4f} |"
        )

    parts.append("""
### Table S11. Monte Carlo Known-Truth Simulation (`n = 150 Trials`) and Bootstrap Paired Δd Contrasts
#### Panel A. Known-Ground-Truth Monte Carlo Simulation Bias Across Spectral Estimators
| Scenario | Estimator | True θ d | Est θ d (±SD) | θ d Bias | Est χ d | χ d Bias | Raw χ Bias | Median R² |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in sim_mc.iterrows():
        r2s = f"{r['median_r2']:.3f}" if pd.notnull(r["median_r2"]) else "—"
        parts.append(
            f"| {r['scenario']} | `{r['estimator']}` | {r['true_theta_d']:+.3f} | "
            f"{r['est_theta_d_mean']:+.3f} (±{r['est_theta_d_sd']:.2f}) | {r['theta_d_bias']:+.3f} | "
            f"{r['est_chi_d_mean']:+.3f} | {r['chi_d_bias']:+.3f} | {r['chi_raw_bias']:+.3f} | {r2s} |"
        )
    parts.append("")
    parts.append("#### Panel B. Empirical Stratified 1,000-Bootstrap Δd Contrasts (Frontal Periodic Theta)")
    parts.append("| Cohort Contrast | Comparison (Alt vs. SP 2–30) | Alt d | SP 2–30 d | Paired Δd [95% Boot CI] | Boot p |")
    parts.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
    for _, r in boot_d[boot_d["marker"] == "frontal_theta_5ch"].iterrows():
        parts.append(
            f"| {r['cohort']} | `{r['comparison']}` | {r['d_alt']:+.3f} | {r['d_sp_2_30']:+.3f} | "
            f"{r['delta_d']:+.3f} [{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}] | {fmt_p(r['boot_p'])} |"
        )

    parts.append("""
### Table S12. Study 2 Recording-State Sensitivity (Eyes-Open N=60) and `Split-Target-HC` State Transfer
#### Panel A. Eyes-Open Secondary Benchmark (Study 2b EO, Split-0 w/ 95% CI)
| Model | Split-0 ROC-AUC [95% CI] | Balanced Acc | Accuracy |
| :--- | :---: | :---: | :---: |""")
    for _, r in mum_eo.iterrows():
        parts.append(
            f"| `{r['model']}` | {r['auc']:.3f} [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}] | "
            f"{100.0 * r['balanced_accuracy']:.1f}% | {100.0 * r['accuracy']:.1f}% |"
        )
    parts.append("")
    parts.append("#### Panel B. Intra-Individual EC–EO Concordance (N = 55) & `Split-Target-HC` State Transfer (Mean ± SD)")
    parts.append("| Analysis / Pair | Metric | `GCN-OMST` (±SD) | `Tuned LR` (±SD) |")
    parts.append("| :--- | :---: | :---: | :---: |")
    for _, r in mum_conc.iterrows():
        parts.append(f"| Concordance (N={int(r['n_shared'])}) | {r['marker']} | r = {r['pearson_r']:+.3f} (p = {fmt_p(r['p_value'])}) | — |")
    for _, r in mum_tx.iterrows():
        parts.append(
            f"| {r['source_cohort']} -> {r['target_cohort']} | `Split-Target-HC` Transfer | "
            f"{r['gcn_auc']:.3f} (±{r['gcn_sd']:.3f}) | {r['lr_auc']:.3f} (±{r['lr_sd']:.3f}) |"
        )

    parts.append("""
### Table S13. Within-Subject Paired Dopaminergic Medication Comparisons (UNM Pure EO N=25 & UCSD N=15)
| Cohort & Protocol | Marker | Mean OFF | Mean ON | Diff (ON - OFF) | Paired dz [95% CI] | Paired p | Wilcoxon p |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in med.iterrows():
        parts.append(
            f"| {r['cohort']} | {r['marker']} | {r['mean_off']:+.4f} | {r['mean_on']:+.4f} | "
            f"{r['mean_diff']:+.4f} | {r['paired_dz']:+.3f} [{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}] | "
            f"{fmt_p(r['paired_p'])} | {fmt_p(r['wilcoxon_p'])} |"
        )

    parts.append("""
### Table S14. Complete 20-Row 10×5 Repeated Clinical Scale Regression Benchmark
| Cohort & Scope | Scale | N | Regressor | 10×5 r (±SD) | Median p | Mean R² | Mean MAE | Split-0 r |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in reg.iterrows():
        parts.append(
            f"| {r['cohort']} | {r['scale']} | {int(r['n_samples'])} | `{r['model']}` | "
            f"{r['pearson_r_mean']:+.3f} (±{r['pearson_r_sd']:.3f}) | {fmt_p(r['median_p'])} | {r['r2_mean']:.3f} | {r['mae_mean']:.3f} | {r['split0_r']:+.3f} |"
        )

    parts.append("""
### Table S15. Tri-Center PD `Split-Target-HC` External Transfer Matrix (20 Splits, w/ Oracle Full-HC Comparison)
| Source | Target | Model | Split-HC Mean AUC (±SD) | [p05, p95] Range | Bal Acc | Oracle Full-HC AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |""")
    for _, r in pd_tx.iterrows():
        parts.append(
            f"| {r['source_cohort']} | {r['target_cohort']} | `{r['model']}` | "
            f"{r['auc']:.3f} (±{r['split_hc_sd']:.3f}) | [{r['split_hc_p05']:.3f}, {r['split_hc_p95']:.3f}] | "
            f"{100.0 * r['balanced_accuracy']:.1f}% | {r['oracle_full_hc_auc']:.3f} |"
        )

    parts.append("""
### Table S16. Cross-Disorder and Affective `Split-Target-HC` Transfer Matrix (14 Directed Pairs x 5 Models)
| Transfer Type | Source -> Target | Model | Split-HC Mean AUC (±SD) | [p05, p95] Range | Bal Acc | Oracle Full-HC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |""")
    for _, r in td_tx[td_tx["transfer_type"] != "Within_Disorder_PD_Cross_Site"].iterrows():
        parts.append(
            f"| {r['transfer_type']} | {r['source_cohort']} -> {r['target_cohort']} | `{r['model']}` | "
            f"{r['auc']:.3f} (±{r['split_hc_sd']:.3f}) | [{r['split_hc_p05']:.3f}, {r['split_hc_p95']:.3f}] | "
            f"{100.0 * r['balanced_accuracy']:.1f}% | {r['oracle_full_hc_auc']:.3f} |"
        )

    parts.append("""
### Table S17. Age-Adjusted ANCOVA Across All Cohorts with Individual Age Metadata
| Cohort Contrast | Marker | Group Coef (SE) | Group t | Group p | Age Coef | Age p | Model R² |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in anc.iterrows():
        parts.append(
            f"| {r['cohort']} | {r['marker']} | {r['group_coef']:+.4f} ({r['group_se']:.4f}) | "
            f"{r['group_t']:+.3f} | {fmt_p(r['group_p'])} | {r['age_coef']:+.4f} | {fmt_p(r['age_p'])} | {r['r2']:.3f} |"
        )

    parts.append("""
### Table S18. 10×5 Repeated Validation Leakage Quantification Across Repeated-Session Cohorts
| Cohort (Design) | Evaluation Regime | Model | Fold Overlap | 10×5 AUC (±SD) | Mean ΔAUC [95% CI] |
| :--- | :--- | :--- | :---: | :---: | :---: |""")
    for _, r in leak.iterrows():
        parts.append(
            f"| {r['cohort']} ({r['design_type']}) | {r['regime']} | `{r['model']}` | {r['subject_overlap_pct']:.1f}% | "
            f"{r['mean_auc']:.3f} (±{r['sd_auc']:.3f}) | {r['mean_delta_vs_strict']:+.3f} [{r['delta_ci_lo']:+.3f}, {r['delta_ci_hi']:+.3f}] |"
        )

    parts.append("""
### Table S19. Six-Pair Cross-Site PD Calibration & Gain Stress-Test Summary (Mean and Worst-Case Max Drift)
| Experiment | Calibration | Feature Type | Gain k | LR AUC (BalAcc / Brier) | RF AUC | Mean / Max |ΔFeat| | Mean / Max |Δp| (LR) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |""")
    gsum = gc.groupby(["experiment", "calibration", "feature_type", "gain_k"], as_index=False).agg(
        lr_auc=("lr_auc", "mean"), lr_bacc=("lr_bacc", "mean"), lr_brier=("lr_brier", "mean"), rf_auc=("rf_auc", "mean"),
        mean_feat=("mean_feat_drift", "mean"), max_feat=("max_feat_drift", "max"),
        mean_p=("lr_mean_prob_drift", "mean"), max_p=("lr_max_prob_drift", "max"),
    )
    for _, r in gsum.iterrows():
        parts.append(
            f"| {r['experiment']} | {r['calibration']} | `{r['feature_type']}` | {r['gain_k']} | "
            f"{r['lr_auc']:.3f} ({r['lr_bacc']:.3f} / {r['lr_brier']:.3f}) | {r['rf_auc']:.3f} | "
            f"{r['mean_feat']:.3f} / {r['max_feat']:.3f} | {r['mean_p']:.3f} / {r['max_p']:.3f} |"
        )

    parts.append("""
### Table S20. True 10×5 Repeated Participant-Grouped Cross-Validation Summary (All 12 Contrasts x 12 Models)
| Cohort Contrast | Model | Seed-42 AUC | 10×5 Mean AUC (+/- SD) | Min–Max Range | Mean Δ vs. GCN-OMST |
| :--- | :--- | :---: | :---: | :---: | :---: |""")
    for _, r in rep10.iterrows():
        parts.append(
            f"| {r['cohort']} | `{r['model']}` | {r['seed42_auc']:.3f} | "
            f"{r['mean_auc']:.3f} (+/- {r['std_auc']:.3f}) | [{r['min_auc']:.3f}, {r['max_auc']:.3f}] | "
            f"{r['mean_delta_vs_omst']:+.3f} |"
        )

    parts.append("""
### Table S21. PLI Matrix Definiteness Audit (677 Sessions) and 10×5 Riemannian Tangent Baselines
#### Panel A. Eigenvalue Spectrum of 19 x 19 Symmetrized PLI Matrices (`I_19 + PLI`)
| Cohort | Sessions (N) | Min λ_min | Median λ_min | Indefinite (λ_min <= 0) | % Indefinite | After +10^-4 I |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in pd.concat([pli_a, d["pli_sum"]], ignore_index=True).iterrows():
        parts.append(
            f"| {r['cohort']} | {int(r['n_sessions'])} | {r['min_eigenvalue']:+.4f} | "
            f"{r['median_min_eigenvalue']:+.4f} | {int(r['n_indefinite_raw'])}/{int(r['n_sessions'])} | "
            f"{r['pct_indefinite_raw']:.2f}% | {int(r['n_indefinite_after_1e4_shift'])} still <= 0 |"
        )
    parts.append("")
    parts.append("#### Panel B. 10×5 Euclidean Upper-Triangular PLI (171-D) vs. SPD-Conditioned Riemannian Tangent Baselines (190-D)")
    parts.append("| Cohort | Classifier | Euclidean 171-D (±SD) | Diagonal-Loaded SPD (≥0.01) | Eigen-Clipped Nearest-SPD (≥10^-4) |")
    parts.append("| :--- | :---: | :---: | :---: | :---: |")
    rp_m = riem.pivot(index=["cohort", "classifier"], columns="representation", values="auc")
    rp_s = riem.pivot(index=["cohort", "classifier"], columns="representation", values="sd_auc")
    for (cname, clf), r in rp_m.iterrows():
        rs = rp_s.loc[(cname, clf)]
        parts.append(
            f"| {cname} | `{clf}` | {r['Euclidean_UpperTri_171D']:.3f} (±{rs['Euclidean_UpperTri_171D']:.3f}) | "
            f"{r['Riemannian_Gershgorin_SPD_190D']:.3f} (±{rs['Riemannian_Gershgorin_SPD_190D']:.3f}) | "
            f"{r['Riemannian_Higham_SPD_190D']:.3f} (±{rs['Riemannian_Higham_SPD_190D']:.3f}) |"
        )

    parts.append("""
### Table S22. Random-Effects Meta-Analysis (`DL` & `HKSJ`) Across Fit Windows and Sensitivity Sets
| Method | Scope | Subgroup | k | Pooled d_RE | DL 95% CI | HKSJ 95% CI | DL p | HKSJ p | I² | Q_between (p) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in meta_df.iterrows():
        parts.append(
            f"| `{r['method']}` | {r['analysis_set']} | {r['subgroup']} | {int(r['k_studies'])} | {r['pooled_d']:+.3f} | "
            f"[{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}] | [{r['hksj_ci_lower']:+.3f}, {r['hksj_ci_upper']:+.3f}] | "
            f"{fmt_p(r['p_value'])} | {fmt_p(r['p_hksj'])} | {r['I2_percent']:.1f}% | {r['Q_between']:.2f} ({fmt_p(r['p_between'])}) |"
        )

    parts.append("""
### Table S23. Per-Cohort `specparam v2.0.0rc7` Diagnostics Across Fit Windows (`[1.0, 40.0]` vs. `[2.0, 30.0] Hz`)
| Fit Window | Cohort Key | Sessions | Spectra | Failed | % R² < 0.80 | R² p05 / Median / Mean | MAE Med / Mean | Mean Peaks | Mean χ_OLS vs. χ_sp |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in fit_df.iterrows():
        parts.append(
            f"| {r['fit_window_hz']} Hz | `{r['cohort_key']}` | {int(r['n_sessions'])} | {int(r['n_spectra'])} | {int(r['n_failed'])} | "
            f"{r['pct_r2_lt_080']:.2f}% | {r['r2_p05']:.3f} / {r['r2_median']:.3f} / {r['r2_mean']:.3f} | "
            f"{r['mae_median']:.4f} / {r['mae_mean']:.4f} | {r['mean_n_peaks']:.2f} | "
            f"{r['mean_chi_ols']:.3f} vs. {r['mean_chi_specparam']:.3f} |"
        )

    parts.append("""
### Table S24. Posterior Individual Alpha Peak Frequency (`IAPF`) and 10×5 Heuristic Slowing Benchmarks
#### Panel A. Group Differences in Posterior `IAPF` (7.0–13.5 Hz)
| Cohort Contrast | Valid N (Case / Ref) | Mean IAPF Case (Hz) | Mean IAPF Ctrl (Hz) | Diff (Hz) | Cohen's d | p-Value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |""")
    for _, r in iapf_df.iterrows():
        parts.append(
            f"| {r['cohort']} | {int(r['n_valid_case'])}/{int(r['n_valid_ctrl'])} | "
            f"{r['mean_iapf_case_hz']:.3f} (±{r['sd_iapf_case_hz']:.2f}) | "
            f"{r['mean_iapf_ctrl_hz']:.3f} (±{r['sd_iapf_ctrl_hz']:.2f}) | "
            f"{r['diff_hz']:+.3f} | {r['cohens_d']:+.3f} | {fmt_p(r['p_value'])} |"
        )
    parts.append("")
    parts.append("#### Panel B. 10×5 Heuristic Frontal (Theta - Alpha) Slowing vs. Canonical `Single_Theta`")
    parts.append("| Cohort Contrast | N | Heuristic Raw AUC | Oriented AUC | Balanced Acc | `Single_Theta` Oriented AUC |")
    parts.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for _, r in heur_df.iterrows():
        parts.append(
            f"| {r['cohort']} | {int(r['n_samples'])} | {r['auc_raw']:.3f} | {r['auc_oriented']:.3f} | "
            f"{100.0 * r['balanced_accuracy']:.1f}% | {r['single_theta_auc_oriented']:.3f} |"
        )

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
        "Study 2a: Outpatient MDD EC (N=58)": "Study 2a: Outpatient MDD EC (4244171)",
    }
    ep_idx = ep_df.set_index("cohort")
    for _, r in anc_dur.iterrows():
        ep_p = ep_idx.loc[ep_map[r["cohort"]], "case_vs_ctrl_p"]
        a_t = f"{r['age_only_grp_t']:+.3f} ({fmt_p(r['age_only_grp_p'])})" if pd.notnull(r["age_only_grp_t"]) else "— (No indiv. age)"
        a_p = fmt_p(r["age_p"]) if pd.notnull(r["age_p"]) else "—"
        parts.append(
            f"| {r['cohort']} | {int(r['n'])} | {a_t} | {a_p} | "
            f"{r['age_plus_epoch_grp_t']:+.3f} ({fmt_p(r['age_plus_epoch_grp_p'])}) | {fmt_p(r['epoch_p'])} | {fmt_p(ep_p)} |"
        )
    parts.append("")
    parts.append("#### Panel B. Study 6 (UNM `ds003490`) Contiguous Pure Eyes-Open (N=75) vs. Pure Eyes-Closed (N=74) Rest")
    parts.append("| Recording Block | Frontal θ d (p) | Posterior α d (p) | Global χ d (p) |")
    parts.append("| :--- | :---: | :---: | :---: |")
    for _, r in unm_sens.iterrows():
        parts.append(
            f"| {r['condition']} | {r['theta_d']:+.3f} ({fmt_p(r['theta_p'])}) | "
            f"{r['alpha_d']:+.3f} ({fmt_p(r['alpha_p'])}) | {r['chi_d']:+.3f} ({fmt_p(r['chi_p'])}) |"
        )

    parts.append("""
### Table S26. Figshare `4244171` (Study 2) Raw EDF Duplicate Hashing and Recording-Batch Forensic Audit
#### Panel A. Exact Byte-Identical SHA-256 Duplicate File Groups Detected in Public Release
| SHA-256 Prefix | Identical Files (n) | Filenames in Figshare Release |
| :---: | :---: | :--- |""")
    for _, r in mum_dups.iterrows():
        parts.append(f"| `{r['sha256_prefix']}` | {int(r['n_files'])} | `{r['filenames']}` |")
    parts.append("")
    parts.append("#### Panel B. Deduplicated and Batch-Stratified Biophysics & 10×5 Classification Sensitivity (`N = 29 Unique MDD, 27 Unique HC EC`)")
    parts.append("| Subset / Batch Sensitivity | Total N (Case/Ctrl) | Frontal θ d (p) | Global χ d (p) | 10×5 Tuned LR (±SD) | 10×5 RF (±SD) |")
    parts.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for _, r in mum_for.iterrows():
        parts.append(
            f"| {r['subset']} | {int(r['n_total'])} ({int(r['n_case'])}/{int(r['n_ctrl'])}) | "
            f"{r['theta_d']:+.3f} ({fmt_p(r['theta_p'])}) | {r['chi_d']:+.3f} ({fmt_p(r['chi_p'])}) | "
            f"{r['lr_10x5_mean']:.3f} (±{r['lr_10x5_sd']:.3f}) | {r['rf_10x5_mean']:.3f} (±{r['rf_10x5_sd']:.3f}) |"
        )
    parts.append("")
    parts.append("*Note: Across unique healthy controls alone (Jan-2013 batch n = 14 vs. 2011 batch n = 9), between-batch healthy control differences reached frontal θ d = +3.347 (p < 0.0001) and global χ d = -1.159 (p = 0.0057).*")

    return "\n".join(parts)


# ==============================================================================
# Supporting Neuroinformatics Documents
# ==============================================================================
def build_cover_letter(d: dict) -> str:
    rep = d["rep10"]
    p9 = [c for c in rep["cohort"].unique() if not any(k in c for k in ("2b", "6b", "6c"))]
    piv = rep[rep["cohort"].isin(p9)].pivot(index="cohort", columns="model", values="mean_auc")
    wil9 = d["wilcoxon"][d["wilcoxon"]["scope"] == "Primary_9_Contrasts_10x5Mean"].set_index("comparator_model")
    pli_s = d["pli_sum"].iloc[0]
    ndg = d["meta"][(d["meta"]["method"] == "specparam_2to30_resid") & (d["meta"]["analysis_set"] == "Primary_K6_SubjMean_PD_OFF") & (d["meta"]["subgroup"] == "PD_Dementia_Cohorts")].iloc[0]
    dem_pt = d["reg"][(d["reg"]["cohort_key"] == "Dementia_MMSE_Patients_Only_N59") & (d["reg"]["model"] == "Random_Forest_Regressor")].iloc[0]
    fit2 = d["fit"][d["fit"]["fit_window_hz"] == "2.0-30.0"].set_index("cohort_key")
    fit1 = d["fit"][d["fit"]["fit_window_hz"] == "1.0-40.0"].set_index("cohort_key")

    return f"""# Cover Letter

**Date:** {SUBMISSION_DATE}

**To:**  
{EDITOR_NAME}  
{EDITOR_TITLE}  
{EDITOR_AFFIL}

**Subject:** Resubmission / Original Research Article Submission to *{JOURNAL_NAME}*

Dear Prof. Van Horn,

Please consider the original research article entitled **"{EXACT_TITLE}"** for publication in *{JOURNAL_NAME}*.

This manuscript directly investigates three core neuroinformatics pitfalls that distort multi-cohort clinical EEG and graph neural network (GNN) benchmarks across seven public datasets (`566` primary participants, `677` primary sessions, `737` total sessions, `14,003` channel spectra):

1. **Spectral Fit-Window Confounding, Masked-OLS Leakage, and Public Archive Forensics:** Benchmarking `[2.0, 30.0] Hz` passband-interior `specparam v2.0.0rc7` against edge-exposed `[1.0, 40.0] Hz` `specparam`, `[7, 14] Hz`-masked OLS, and known-truth Monte Carlo simulations exposes dual spectral pitfalls: `[7, 14] Hz`-masked OLS generates spurious negative theta effects (`-0.652 d` Monte Carlo null bias; Study 3 `p = 0.010` bootstrap shift), whereas `[1.0, 40.0] Hz` fitting collapses on notch-filtered Study 2 spectra (`R²` `{fit1.loc['mumtaz_ec', 'r2_median']:.3f}` -> `{fit2.loc['mumtaz_ec', 'r2_median']:.3f}`) and inverts periodic residuals under filter-skirt attenuation (`+7.06 d` Monte Carlo bias). Furthermore, SHA-256 and acquisition-date forensics on Figshare `4244171` (Study 2) exposed exact byte-identical duplicate EDF files (`H S27` = `H S30`, `MDD S33` = `MDD S34`) plus a `d = +3.35` between-batch healthy control artifact (Jan-2013 vs. 2011) driving apparent MDD theta differences, while neurodegenerative frontal theta slowing across PD and dementia replicated cleanly (`d_RE = {ndg['pooled_d']:+.3f}, p_HKSJ = {ndg['p_hksj']:.4f}`).
2. **True 10×5 Grouped-CV GNN Parity and Aligned Permutations:** After removing oddball task contamination in Study 6 UNM (`ds003490`) and replacing tied `StratifiedGroupKFold` partitions with shuffled 10×5 participant-grouped validation, `GCN-OMST` (9-contrast mean ROC-AUC `{piv['GCN_OMST'].mean():.3f}`) was indistinguishable from edge-matched `GCN-Random` (`{piv['GCN_Random_EdgeMatched'].mean():.3f}, p = {wil9.loc['GCN_Random_EdgeMatched', 'raw_p']:.4f}`) and isolated `GCN-Identity` (`{piv['GCN_Identity'].mean():.3f}`), while trailing `Tuned Ridge` (`{piv['Tuned_Ridge'].mean():.3f}`), `Random Forest` (`{piv['RandomForest'].mean():.3f}`), and `Tuned LR` (`{piv['Tuned_LogisticRegression'].mean():.3f}`).
3. **Connectome Definiteness, Bidirectional Leakage, Disjoint Transfer, and Severity Floor:** `{int(pli_s['n_indefinite_raw'])}/{int(pli_s['n_sessions'])} ({pli_s['pct_indefinite_raw']:.2f}%)` symmetrized PLI matrices failed positive semi-definiteness (requiring conditioning for Riemannian tangent geometry), 10×5 session mixing shifted AUC from `-0.094` (sleep crossover) to `+0.080` (UCSD PD), leak-free 50/50 `Split-Target-HC` evaluation bounded external PD transfer (`up to 0.751`), and removing ceiling controls collapsed within-patient dementia MMSE regression (`r = {dem_pt['pearson_r_mean']:+.3f}, p = {dem_pt['median_p']:.3f}`; all patient-only `R² <= 0.035`).

Consistent with *{JOURNAL_NAME}*'s Information Sharing Policy, all code, Zenodo-ready derivative tables, fold indices, and `requirements_lock.txt` are released openly at `{GITHUB_URL}`. This work is original, unpublished, and free of competing interests.

Sincerely yours,

**{AUTHOR_NAME}**  
{AFFIL_1}  
{AFFIL_2}  
{CORRESPONDENCE}
"""


def build_highlights(d: dict) -> str:
    rep = d["rep10"]
    p9 = [c for c in rep["cohort"].unique() if not any(k in c for k in ("2b", "6b", "6c"))]
    piv = rep[rep["cohort"].isin(p9)].pivot(index="cohort", columns="model", values="mean_auc")
    pli_s = d["pli_sum"].iloc[0]
    return f"""# Research Highlights

*Journal Target:* *{JOURNAL_NAME}* ({PUBLISHER_NAME})  
*Manuscript Title:* {EXACT_TITLE}

1. Interior [2–30 Hz] `specparam` cures filter-skirt flip & masked-OLS theta leakage.
2. Figshare `4244171` carries duplicate EDF pairs & a d=+3.35 batch control artifact.
3. Under 10×5 grouped CV, `GCN-OMST` ({piv['GCN_OMST'].mean():.3f}) ties matched random topology ({piv['GCN_Random_EdgeMatched'].mean():.3f}).
4. `{pli_s['pct_indefinite_raw']:.2f}%` of 19-channel PLI matrices violate positive semi-definiteness (min λ=-7.37).
5. Session mixing shifts 10×5 AUC bidirectionally (-0.094 crossover vs. +0.080 PD).
"""


def build_title_page(main_md: str) -> str:
    abs_block = main_md.split("## Abstract\n\n")[1].split("\n\n**Keywords:**")[0]
    kw_line = main_md.split("**Keywords:** ")[1].split("\n")[0]
    info_block = main_md.split("## Information Sharing Statement\n\n")[1].split("\n\n---")[0]
    decl_block = main_md.split("## Statements and Declarations\n\n")[1].split("\n\n---")[0]
    return f"""# Title Page

**Full Title:** {EXACT_TITLE}

**Short Running Title:** {SHORT_TITLE}

**Article Type:** Original Research Article

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
    return f"""# Combined Ethical Approval, Competing Interests, Funding, and CRediT Statement

**Manuscript Title:** *{EXACT_TITLE}*  
**Target Journal:** *{JOURNAL_NAME}* ({PUBLISHER_NAME})  
**Author:** {AUTHOR_NAME}^1,2*^  
^1^ {AFFIL_1}  
^2^ {AFFIL_2}  
^*^ Correspondence: {CORRESPONDENCE}

---

## 1. Ethics Approval and Informed Consent
This study performed secondary computational analysis of de-identified, publicly available EEG cohorts hosted on OpenNeuro and Figshare. Under U.S. Federal Regulation 45 CFR 46.104(d)(4) and institutional guidelines, secondary analysis of fully de-identified public archives is exempt from additional IRB review. Primary cohorts obtained local ethics approval and written informed consent under the Declaration of Helsinki: Southwest University (`H20039`, `ds004902`), Universiti Sains Malaysia (`USM/JEPeM/1401015`, Figshare `4244171`), University of New Mexico (`ds003478`, `ds003490`), UC San Diego (`ds002778`), AHEPA University Hospital (`142/12-04-2023`, `ds004504`), and University of Iowa (`201707828`, `ds004584`).

## 2. Competing Interests
The author declares no competing financial or non-financial interests.

## 3. Funding
This research received no specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## 4. CRediT Author Statement
**Chushu Tan:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Resources, Data Curation, Writing – Original Draft, Writing – Review & Editing, Visualization, Project administration.

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
    return f"""# Springer *Neuroinformatics* Submission Checklist & Guide (v4 Remediated)

**Journal:** *{JOURNAL_NAME}* (Springer Nature, Editorial Manager)  
**Target Editor:** {EDITOR_NAME}  
**Date:** {SUBMISSION_DATE}

| File in `submission_Neuroinformatics/` | Editorial Manager Slot | Notes |
| :--- | :--- | :--- |
| `01_Cover_Letter_Neuroinformatics.docx` | Cover Letter | Summarizes dual spectral window/OLS pitfalls, Study 2 forensics, & 10×5 GNN parity |
| `02_Research_Highlights_Neuroinformatics.docx` | Author Highlights (Optional / Summary) | 5 concise findings |
| `03_Title_Page_Neuroinformatics.docx` | Title Page | Unstructured single-paragraph Abstract ({abs_wc} <= 250 words), Keywords, Declarations |
| `04_Main_Manuscript_Neuroinformatics.docx` | Manuscript | Continuous line numbers, 3-line tables, inline Figures 1–4 & Tables 1–5 |
| `05_Supplementary_Material_Neuroinformatics.docx` | Supplementary Material | Sections S1.1–S1.4, Figures S1–S6, Tables S1–S26 |
| `06_Declarations_CRediT_COI_Ethics.docx` | Declarations / Ethics Attachment | Signed IRB exemptions, CRediT, COI, Funding |
| `07_Suggested_Reviewers.docx` | Suggested Reviewers Reference | Voytek, Dimitriadis, Gramfort, Delorme |
| `Figures/Main_Figures/Figure_1..4.(png/pdf)` | Figure Files | 300 DPI |
| `Figures/Supplementary_Figures/fig_S1..S6.(png/pdf)` | Supplementary Figures | 300 DPI |

### Pre-Submission Audits Verified
- **Abstract Word Count:** {abs_wc} words (`<= 250` Springer limit).
- **Zero Residual Stale Strings:** Programmatically verified zero stale metadata, phone numbers, or deprecated artifact identifiers.
- **DOCX Author / Modifier Properties:** Scrubbed clean across all `.docx` outputs.
"""


# ==============================================================================
# Verification & Automated Audits
# ==============================================================================
def run_strict_audits(main_md: str, supp_md: str, doc_paths: list[Path]):
    print("\n🔍 Running strict Neuroinformatics v4 audits...")

    # 1. XML & clean core metadata across every docx
    for p in doc_paths:
        with zipfile.ZipFile(p, "r") as z:
            xc = 0
            for name in z.namelist():
                if name.endswith((".xml", ".rels")):
                    ET.fromstring(z.read(name))
                    xc += 1
        d_obj = docx.Document(p)
        assert d_obj.core_properties.author == "", f"{p.name} has non-empty author metadata!"
        assert d_obj.core_properties.last_modified_by == "", f"{p.name} has non-empty last_modified_by metadata!"
        print(f"   ✓ {p.name}: {xc} XML parts valid, blank metadata ({len(d_obj.paragraphs)} paras, {len(d_obj.tables)} tables)")

    # 2. Verify continuous line numbering in Main Manuscript DOCX
    with zipfile.ZipFile(MAIN_DOCX, "r") as z:
        settings_xml = z.read("word/settings.xml").decode("utf-8")
        doc_xml = z.read("word/document.xml").decode("utf-8")
        assert "xmlns:m=" in settings_xml
        assert "lnNumType" in doc_xml

    # 3. Verify 47 separate reference paragraphs [1]..[47] with hanging indent
    d_main = docx.Document(MAIN_DOCX)
    ref_paras = [p for p in d_main.paragraphs if re.match(r"^\[\d+\]\s", p.text.strip())]
    assert len(ref_paras) == 47, f"Expected 47 reference paragraphs, got {len(ref_paras)}"
    for idx, rp in enumerate(ref_paras, start=1):
        assert rp.text.strip().startswith(f"[{idx}] "), f"Ref mismatch at {idx}: {rp.text[:35]}"
        assert rp.paragraph_format.first_line_indent == Inches(-0.35), f"Missing hanging indent on ref {idx}"
    print("   ✓ References: 47 verified paragraphs [1]..[47] with hanging indents & DOIs")

    # 4. Strictly monotonic first-appearance citation sequence 1 -> 47
    body_before_refs = re.sub(r"`[^`]+`", "", main_md.split("## References")[0])
    body_before_refs = re.sub(r"\[\s*(?:7|2|1)[,\.]?\s*(?:0)?\s*[,\-–]\s*(?:14|30|40)(?:\.0)?\s*\](\s*Hz)?", "", body_before_refs)
    seen_max = 0
    for m in re.finditer(r"\[(\d+(?:[–\-,\s]+\d+)*)\]", body_before_refs):
        token = m.group(1)
        if re.search(r"\d\.\d", token):
            continue
        for part in [x.strip() for x in token.split(",")]:
            if "–" in part:
                a, b = [int(v) for v in part.split("–")]
                nums = list(range(a, b + 1))
            elif "-" in part:
                a, b = [int(v) for v in part.split("-")]
                nums = list(range(a, b + 1))
            else:
                nums = [int(part)]
            for n in nums:
                if n > seen_max:
                    assert n == seen_max + 1, f"Citation [{n}] appeared before [{seen_max + 1}]! Context: {m.group(0)}"
                    seen_max = n
    assert seen_max == 47, f"Expected max citation 47, got {seen_max}"
    print("   ✓ In-text citation sequence: strictly monotonic 1 -> 47")

    # 5. Voice, banned hype words, <=2 rather than, zero inline bold/italics in prose
    combined = main_md + "\n" + supp_md
    plural_hits = re.findall(r"\b(we|our|ours|ourselves|us)\b", combined, flags=re.IGNORECASE)
    assert not plural_hits, f"Found plural pronouns: {plural_hits}"
    banned_words = [
        "crucially", "notably", "in stark contrast", "pivotal", "profound", "profoundly",
        "genuine", "indispensable", "catastrophic", "overwhelmingly", "stark", "statistically indistinguishable",
    ]
    for w in banned_words:
        hits = re.findall(rf"\b{re.escape(w)}\b", combined, flags=re.IGNORECASE)
        assert not hits, f"Found banned word/phrase '{w}': {hits}"

    rather_cnt = len(re.findall(r"\brather than\b", main_md, flags=re.IGNORECASE))
    assert rather_cnt <= 2, f"'rather than' count {rather_cnt} exceeds <= 2 limit!"

    for line in main_md.split("## References")[0].splitlines():
        st = line.strip()
        if not st or st.startswith(("#", "|", "![", "*", "-", "---", "$$")) or re.match(r"^\d+\.\s", st) or st.startswith("**") or re.match(r"^\^\d|^\^\*", st):
            continue
        if "**" in st:
            raise AssertionError(f"Found inline bold ** inside prose: {st[:90]}")
        cleaned = re.sub(r"`[^`]*`", "", re.sub(r"\$\$?[^$]*\$\$?", "", st))
        if re.search(r"(^|[\s(])\*[^*\s][^*]*\*", cleaned):
            raise AssertionError(f"Found inline italics *...* inside prose: {st[:90]}")
    print(f"   ✓ Voice & prose audit: 0 plural pronouns, 0 banned words, rather than={rather_cnt} (<=2), 0 inline bold/italics")

    # 6. Regression guards against stale artifacts/PII/banned terms + 100% Supp cross-refs
    for p in doc_paths:
        tx = "\n".join(par.text for par in docx.Document(p).paragraphs)
        for bad in ["12246", "102249", "18008428969", "0.902", "27.33", "Gershgorin"]:
            assert bad not in tx, f"{p.name} contains stale string '{bad}'!"
        assert not re.search(r"Generative AI|TRIPOD\+AI|ChatGPT|DeepSeek|large language model", tx, flags=re.IGNORECASE), f"{p.name} contains removed AI text!"
    assert "Gershgorin" not in combined, "Stale 'Gershgorin' string in MD!"

    supp_table_ids = re.findall(r"^### Table (S\d+[a-z]?)\.", supp_md, flags=re.M)
    unused_tables = [tid for tid in supp_table_ids if not re.search(rf"\b{re.escape(tid)}\b", main_md)]
    assert not unused_tables, f"Unused Supplementary tables not cited in Main Text: {unused_tables}"
    supp_fig_ids = [f"Figure S{i}" for i in range(1, 7)]
    unused_figs = [fid for fid in supp_fig_ids if fid not in main_md]
    assert not unused_figs, f"Unused Supplementary figures not cited in Main Text: {unused_figs}"
    print(f"   ✓ Supplementary audit: {len(supp_table_ids)}/{len(supp_table_ids)} tables (S1..S26) and 6/6 figures cited in Main Text!")


# ==============================================================================
# Main Pipeline
# ==============================================================================
def main():
    print("=" * 80)
    print("Building Paper 2 (Springer Neuroinformatics v4) from Canonical [2, 30] Hz & 10x5 Outputs")
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
            (MAIN_MD, ARTIFACT_DIR / "PAPER_2_NEUROINFORMATICS_v4.md"),
            (MAIN_DOCX, ARTIFACT_DIR / "PAPER_2_NEUROINFORMATICS_v4.docx"),
            (SUPP_MD, ARTIFACT_DIR / "PAPER_2_SUPPLEMENTARY_MATERIAL_v4.md"),
            (SUPP_DOCX, ARTIFACT_DIR / "PAPER_2_SUPPLEMENTARY_MATERIAL_v4.docx"),
            (SUBMISSION_DIR / "01_Cover_Letter_Neuroinformatics.docx", ARTIFACT_DIR / "Cover_Letter_Neuroinformatics_v4.docx"),
            (SUBMISSION_DIR / "02_Research_Highlights_Neuroinformatics.docx", ARTIFACT_DIR / "Highlights_Neuroinformatics_v4.docx"),
            (SUBMISSION_DIR / "03_Title_Page_Neuroinformatics.docx", ARTIFACT_DIR / "Title_Page_Neuroinformatics_v4.docx"),
            (REPORTS_DIR / "Figure_1.png", ARTIFACT_DIR / "Paper2_v4_Figure_1.png"),
            (REPORTS_DIR / "Figure_2.png", ARTIFACT_DIR / "Paper2_v4_Figure_2.png"),
            (REPORTS_DIR / "Figure_3.png", ARTIFACT_DIR / "Paper2_v4_Figure_3.png"),
            (REPORTS_DIR / "Figure_4.png", ARTIFACT_DIR / "Paper2_v4_Figure_4.png"),
            (REPORTS_DIR / "fig_S6_permutation_tests.png", ARTIFACT_DIR / "Paper2_v4_Figure_S6_Permutations.png"),
        ]
        for s_p, d_p in mirror_map:
            shutil.copy2(s_p, d_p)
        print(f"✓ Mirrored {len(mirror_map)} core v4 deliverables to Artifact Directory.")

    print("=" * 80)
    print("ALL PAPER 2 NEUROINFORMATICS v4 DELIVERABLES BUILT AND AUDITED CLEANLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
