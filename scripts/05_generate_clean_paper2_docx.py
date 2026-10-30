#!/usr/bin/env python3
"""
Compiles PAPER_2_MULTI_COHORT_TRANSDIAGNOSTIC_EEG_BENCHMARK.md into a professional,
publication-ready Word .docx manuscript.
Features:
  - Complete elimination of raw LaTeX code, math markup, and garbled symbols.
  - Full Unicode conversion of Greek letters, mathematical operators, superscripts, and subscripts.
  - Clean table rendering with alternating row fills and proper cell margins.
  - High-resolution (300 DPI) embedded figures with formatted captions.
  - Professional typography adhering to medical and neuroscience journal standards.
"""

import re
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
MD_PATH = BENCHMARK_ROOT / "manuscript" / "PAPER_2_MULTI_COHORT_TRANSDIAGNOSTIC_EEG_BENCHMARK.md"
REPORTS_DIR = BENCHMARK_ROOT / "reports"
OUTPUT_DOCX = BENCHMARK_ROOT / "manuscript" / "PAPER_2_MULTI_COHORT_TRANSDIAGNOSTIC_EEG_BENCHMARK.docx"

GREEK_AND_SYMBOLS = {
    r'\Delta': 'Δ',
    r'\beta': 'β',
    r'\alpha': 'α',
    r'\theta': 'θ',
    r'\delta': 'δ',
    r'\chi': 'χ',
    r'\Omega': 'Ω',
    r'\omega': 'ω',
    r'\rho': 'ρ',
    r'\sigma': 'σ',
    r'\mu': 'μ',
    r'\lambda': 'λ',
    r'\nabla': '∇',
    r'\times': '×',
    r'\le': '≤',
    r'\leq': '≤',
    r'\ge': '≥',
    r'\geq': '≥',
    r'\pm': '±',
    r'\approx': '≈',
    r'\neq': '≠',
    r'\propto': '∝',
    r'\to': '→',
    r'\sim': '~',
    r'\mu\text{V}': 'µV',
    r'\text{--}': '–',
    r'\text{-}': '-',
    r'\quad': ' ',
    r'\dagger': '†',
    r'\Sigma': 'Σ',
}

SUPERSCRIPTS = {
    '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
    '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
    'n': 'ⁿ', 'i': 'ⁱ', 'a': 'ᵃ', 'b': 'ᵇ', 'c': 'ᶜ',
    'k': 'ᵏ', 'm': 'ᵐ', 'r': 'ʳ', 't': 'ᵗ', 'x': 'ˣ',
    '*': '*', '†': '†',
}

SUBSCRIPTS = {
    '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
    '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
    '+': '₊', '-': '₋', '=': '₌', '(': '₍', ')': '₎',
    'a': 'ₐ', 'e': 'ₑ', 'o': 'ₒ', 'x': 'ₓ', 'h': 'ₕ',
    'i': 'ᵢ', 'j': 'ⱼ', 'k': 'ₖ', 'l': 'ₗ', 'm': 'ₘ',
    'n': 'ₙ', 'p': 'ₚ', 's': 'ₛ', 't': 'ₜ', 'u': 'ᵤ',
    'v': 'ᵥ', 'z': '𝓏',
}


def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def set_cell_shading(cell, color_hex):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)


def clean_latex(text: str) -> str:
    """Translates common math/LaTeX tokens into clean, elegant Unicode text."""
    if not text:
        return ""
    s = text

    # Standard symbols
    for macro, sym in GREEK_AND_SYMBOLS.items():
        s = s.replace(macro, sym)

    # Special accents and wrappers
    s = re.sub(r'\\hat\{y\}', 'ŷ', s)
    s = re.sub(r'\\hat\{P\}', 'P̂', s)
    s = re.sub(r'\\tilde\{D\}', 'D̃', s)
    s = re.sub(r'\\tilde\{A\}', 'Ã', s)
    s = re.sub(r'\\mathbf\{([^}]+)\}', r'\1', s)
    s = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', s)
    s = re.sub(r'\\mathit\{([^}]+)\}', r'\1', s)
    s = re.sub(r'\\mathcal\{([^}]+)\}', r'\1', s)
    s = re.sub(r'\\mathbb\{([^}]+)\}', r'\1', s)
    s = re.sub(r'\\boldsymbol\{([^}]+)\}', r'\1', s)
    s = re.sub(r'\\text\{([^}]+)\}', r'\1', s)

    # Superscripts
    def sub_sup(m):
        content = m.group(1)
        return "".join(SUPERSCRIPTS.get(c, c) for c in content)
    s = re.sub(r'\^\{([^}]+)\}', sub_sup, s)
    s = re.sub(r'\^([0-9a-zA-Z\*\+\-†])', lambda m: SUPERSCRIPTS.get(m.group(1), m.group(1)), s)

    # Subscripts
    def sub_sub(m):
        content = m.group(1)
        return "".join(SUBSCRIPTS.get(c, c) for c in content)
    s = re.sub(r'_\{([^}]+)\}', sub_sub, s)
    s = re.sub(r'_([0-9a-zA-Z\*\+\-])', lambda m: SUBSCRIPTS.get(m.group(1), m.group(1)), s)

    # Strip math delimiters
    s = re.sub(r'\$([^$]+)\$', r'\1', s)
    s = s.replace('$', '')
    s = s.replace('{', '').replace('}', '')
    s = s.replace(r'\,', ' ').replace(r'\;', ' ')
    s = re.sub(r'\\[a-zA-Z]+', '', s)
    return s


def build_docx():
    print(f"📄 Building Clean Word Document from {MD_PATH.name}...")
    doc = docx.Document()

    # Page Margins (1.0 inch all around)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Normal Style
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.25
    normal_style.paragraph_format.space_after = Pt(6)

    lines = MD_PATH.read_text(encoding='utf-8').splitlines()

    i = 0
    in_table = False
    table_lines = []

    while i < len(lines):
        line = lines[i].strip()

        # Handle Markdown Tables
        if line.startswith('|') and line.endswith('|'):
            in_table = True
            table_lines.append(line)
            i += 1
            continue
        elif in_table:
            render_markdown_table(doc, table_lines)
            in_table = False
            table_lines = []

        if not line:
            i += 1
            continue

        # Skip horizontal rules
        if line == '---':
            i += 1
            continue

        # Title
        if line.startswith('# '):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(clean_latex(line[2:]))
            run.font.size = Pt(16)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x11, 0x22, 0x44)
            p.paragraph_format.space_after = Pt(10)
            i += 1
            continue

        # Heading 1
        if line.startswith('## '):
            p = doc.add_paragraph()
            run = p.add_run(clean_latex(line[3:]))
            run.font.size = Pt(13)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x1a, 0x36, 0x5d)
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(4)
            i += 1
            continue

        # Heading 2
        if line.startswith('### '):
            p = doc.add_paragraph()
            run = p.add_run(clean_latex(line[4:]))
            run.font.size = Pt(11.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x2d, 0x37, 0x48)
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(3)
            i += 1
            continue

        # Heading 3
        if line.startswith('#### '):
            p = doc.add_paragraph()
            run = p.add_run(clean_latex(line[5:]))
            run.font.size = Pt(11)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2)
            i += 1
            continue

        # Check for figure insertion points
        if "Table 1 and Figure 1" in line:
            p = doc.add_paragraph()
            add_formatted_text(p, clean_latex(line))
            fig1_path = REPORTS_DIR / "Figure_1_Multicohort_Biophysical_Effect_Sizes.png"
            if fig1_path.exists():
                doc.add_paragraph().paragraph_format.space_before = Pt(6)
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.add_run().add_picture(str(fig1_path), width=Inches(6.4))
                p_cap = doc.add_paragraph()
                p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run_cap = p_cap.add_run("Figure 1. Multi-Cohort Biophysical Effect Sizes Across Pathologies (Forest plots of Cohen's d with 95% Confidence Intervals; * indicates Benjamini-Hochberg FDR q < 0.05).")
                run_cap.font.size = Pt(9.5)
                run_cap.font.italic = True
                p_cap.paragraph_format.space_after = Pt(10)
            i += 1
            continue

        if "Table 2 and Figure 2" in line:
            p = doc.add_paragraph()
            add_formatted_text(p, clean_latex(line))
            fig2_path = REPORTS_DIR / "Figure_2_Within_vs_Cross_Cohort_Performance.png"
            if fig2_path.exists():
                doc.add_paragraph().paragraph_format.space_before = Pt(6)
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.add_run().add_picture(str(fig2_path), width=Inches(6.2))
                p_cap = doc.add_paragraph()
                p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run_cap = p_cap.add_run("Figure 2. NormativeGCN Classification Benchmarks: Within-Cohort Cross-Validation vs. Zero-Shot Cross-Cohort Model Transfer (ROC-AUC with 95% Bootstrap Confidence Intervals).")
                run_cap.font.size = Pt(9.5)
                run_cap.font.italic = True
                p_cap.paragraph_format.space_after = Pt(10)
            i += 1
            continue

        # Regular paragraph
        p = doc.add_paragraph()
        add_formatted_text(p, clean_latex(line))
        i += 1

    if in_table and table_lines:
        render_markdown_table(doc, table_lines)

    doc.save(OUTPUT_DOCX)
    print(f"✓ Successfully generated clean publication Word manuscript: {OUTPUT_DOCX}")


def add_formatted_text(p, text):
    parts = re.split(r'(\*\*[^*]+\*\*|\*[^*]+\*)', text)
    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            run = p.add_run(part[2:-2])
            run.bold = True
        elif part.startswith('*') and part.endswith('*'):
            run = p.add_run(part[1:-1])
            run.italic = True
        else:
            p.add_run(part)


def render_markdown_table(doc, table_lines):
    clean_rows = []
    for line in table_lines:
        cells = [c.strip() for c in line.split('|')[1:-1]]
        if all(re.match(r'^:?-+:?$', c) for c in cells):
            continue
        clean_rows.append(cells)

    if not clean_rows:
        return

    n_cols = max(len(r) for r in clean_rows)
    table = doc.add_table(rows=len(clean_rows), cols=n_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for r_idx, row in enumerate(clean_rows):
        is_header = (r_idx == 0)
        for c_idx in range(n_cols):
            cell = table.cell(r_idx, c_idx)
            val = row[c_idx] if c_idx < len(row) else ""
            val = val.replace('<br>', '\n')
            val = clean_latex(val)

            p = cell.paragraphs[0]
            p.text = ""
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(2)

            add_formatted_text(p, val)

            set_cell_margins(cell, top=70, bottom=70, left=90, right=90)
            if is_header:
                for run in p.runs:
                    run.font.bold = True
                    run.font.size = Pt(9.5)
                set_cell_shading(cell, "EBF3FA")
            else:
                for run in p.runs:
                    run.font.size = Pt(9)
                if r_idx % 2 == 1:
                    set_cell_shading(cell, "F9FAFC")

    doc.add_paragraph().paragraph_format.space_after = Pt(6)


if __name__ == "__main__":
    build_docx()
