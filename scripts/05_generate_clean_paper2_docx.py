#!/usr/bin/env python3
"""
Compiles PAPER_2_MULTI_COHORT_TRANSDIAGNOSTIC_EEG_BENCHMARK.md into a publication-ready Word .docx.
Includes professional formatting, clean tables, and embedded 300-DPI publication figures.
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


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
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
    # Basic math cleanups
    text = text.replace(r'\text{--}', '–').replace(r'\text{-}', '-')
    text = text.replace(r'\mu\text{V}', 'µV')
    text = text.replace(r'\chi', 'χ').replace(r'\delta', 'δ').replace(r'\theta', 'θ').replace(r'\alpha', 'α')
    text = text.replace(r'\Delta', 'Δ').replace(r'\pm', '±')
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\$([^$]+)\$', r'\1', text)
    return text


def build_docx():
    print(f"📄 Building Word Document from {MD_PATH.name}...")
    doc = docx.Document()

    # Margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.3
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
            # Render accumulated table
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
            run.font.size = Pt(17)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x11, 0x22, 0x44)
            p.paragraph_format.space_after = Pt(12)
            i += 1
            continue

        # Heading 1
        if line.startswith('## '):
            p = doc.add_paragraph()
            run = p.add_run(clean_latex(line[3:]))
            run.font.size = Pt(14)
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
            run.font.size = Pt(12)
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
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2)
            i += 1
            continue

        # Check for figure insertion points
        if "Table 1 and Figure 1" in line:
            p = doc.add_paragraph()
            add_formatted_text(p, clean_latex(line))
            # Insert Figure 1
            fig1_path = REPORTS_DIR / "Figure_1_Multicohort_Biophysical_Effect_Sizes.png"
            if fig1_path.exists():
                doc.add_paragraph().paragraph_format.space_before = Pt(6)
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.add_run().add_picture(str(fig1_path), width=Inches(6.3))
                p_cap = doc.add_paragraph()
                p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run_cap = p_cap.add_run("Figure 1. Multi-Cohort Biophysical Effect Sizes Across Pathologies (Forest Plots with 95% Confidence Intervals).")
                run_cap.font.size = Pt(9.5)
                run_cap.font.italic = True
            i += 1
            continue

        if "Table 2 and Figure 2" in line:
            p = doc.add_paragraph()
            add_formatted_text(p, clean_latex(line))
            # Insert Figure 2
            fig2_path = REPORTS_DIR / "Figure_2_Within_vs_Cross_Cohort_Performance.png"
            if fig2_path.exists():
                doc.add_paragraph().paragraph_format.space_before = Pt(6)
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.add_run().add_picture(str(fig2_path), width=Inches(6.0))
                p_cap = doc.add_paragraph()
                p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run_cap = p_cap.add_run("Figure 2. NormativeGCN Classification Benchmarks: Within-Cohort 5-Fold Cross-Validation vs. Zero-Shot Cross-Cohort Model Transfer (ROC-AUC with 95% Bootstrap CI).")
                run_cap.font.size = Pt(9.5)
                run_cap.font.italic = True
            i += 1
            continue

        # Regular paragraph
        p = doc.add_paragraph()
        add_formatted_text(p, clean_latex(line))
        i += 1

    if in_table and table_lines:
        render_markdown_table(doc, table_lines)

    doc.save(OUTPUT_DOCX)
    print(f"✓ Successfully generated publication Word manuscript: {OUTPUT_DOCX}")


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

            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
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
