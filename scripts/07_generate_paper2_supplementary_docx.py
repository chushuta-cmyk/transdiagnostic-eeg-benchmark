#!/usr/bin/env python3
"""
Compiles PAPER_2_SUPPLEMENTARY_MATERIAL.md into a professional,
publication-ready Word .docx supplementary package conforming to NeuroImage / IEEE standards.

Features:
  - Document-level Heading hierarchy (Title, Heading 1, Heading 2, Heading 3).
  - Continuous line numbering and page numbering in footer.
  - Complete elimination of raw LaTeX code, math markup, and markdown artifacts.
  - Full Unicode conversion of Greek letters, mathematical operators, superscripts, and subscripts.
  - Journal-standard APA/IEEE Three-Line Tables (top border, header bottom border, table bottom border, zero vertical borders).
  - High-resolution (300 DPI) embedded figures (S1 through S6) with formatted academic captions.
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
MD_PATH = BENCHMARK_ROOT / "manuscript" / "PAPER_2_SUPPLEMENTARY_MATERIAL.md"
REPORTS_DIR = BENCHMARK_ROOT / "reports"
OUTPUT_DOCX = BENCHMARK_ROOT / "manuscript" / "PAPER_2_SUPPLEMENTARY_MATERIAL.docx"
OUTPUT_DOCX_ALT = BENCHMARK_ROOT / "manuscript" / "Supplementary_Material.docx"

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
    r'\mu\mathrm{V}': 'µV',
    r'\text{--}': '–',
    r'\text{-}': '-',
    r'\quad': ' ',
    r'\dagger': '†',
    r'\Sigma': 'Σ',
    r'\phi': 'ϕ',
    r'\eta': 'η',
}

SUPERSCRIPTS = {
    '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
    '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
    'n': 'ⁿ', 'i': 'ⁱ', '*': '*', 'T': 'ᵀ', '–': '⁻'
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


def clean_latex(text: str) -> str:
    """Converts LaTeX math expressions and Markdown symbols into clean Unicode strings."""
    if not text:
        return ""

    for cmd, sym in GREEK_AND_SYMBOLS.items():
        text = text.replace(cmd, sym)

    # Convert subscripts
    text = re.sub(r'_\{([a-zA-Z0-9\+\-]+)\}', lambda m: ''.join(SUBSCRIPTS.get(c, c) for c in m.group(1)), text)
    text = re.sub(r'_([a-zA-Z0-9])', lambda m: SUBSCRIPTS.get(m.group(1), m.group(1)), text)

    # Convert superscripts
    text = re.sub(r'\^\{([a-zA-Z0-9\+\-\*\–]+)\}', lambda m: ''.join(SUPERSCRIPTS.get(c, c) for c in m.group(1)), text)
    text = re.sub(r'\^([a-zA-Z0-9])', lambda m: SUPERSCRIPTS.get(m.group(1), m.group(1)), text)

    # Remove residual math delimiters
    text = text.replace('$', '')
    text = text.replace(r'\(', '').replace(r'\)', '')
    text = text.replace(r'\[', '').replace(r'\]', '')
    text = text.replace(r'\mathrm{', '').replace(r'\mathbf{', '').replace(r'\mathit{', '')
    text = text.replace(r'\text{', '')
    text = text.replace(r'\in', '∈')
    text = text.replace(r'\mathbb{R}', 'R')
    text = text.replace(r'\bar{k}', 'k̄')
    text = text.replace(r'\ ', ' ')
    text = text.replace(r'^\dagger', '†')
    text = text.replace('^†', '†')
    text = text.replace('^*', '*')
    text = text.replace('^**', '**')
    text = text.replace('^***', '***')
    text = text.replace(r'^{**}', '**')
    text = text.replace(r'^{*}', '*')
    text = text.replace(r'^{***}', '***')
    text = text.replace(r'^{\dagger}', '†')
    text = text.replace('`', '')
    text = text.replace('{', '').replace('}', '')
    text = text.replace('\\', '')

    return text.strip()


def set_cell_margins(cell, top=80, bottom=80, left=100, right=100):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def set_cell_shading(cell, color_hex="F8FAFC"):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)


def apply_three_line_table(table, header_bg="F1F5F9"):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tblPr = table._tbl.tblPr
    tblBorders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>\n'
        f'  <w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>\n'
        f'  <w:left w:val="none"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'  <w:insideH w:val="none"/>\n'
        f'  <w:insideV w:val="none"/>\n'
        f'</w:tblBorders>'
    )
    tblPr.append(tblBorders)

    for i, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))

        if i == 0:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
            for cell in row.cells:
                set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
                set_cell_shading(cell, header_bg)
                tcBorders = parse_xml(
                    f'<w:tcBorders {nsdecls("w")}>\n'
                    f'  <w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>\n'
                    f'</w:tcBorders>'
                )
                cell._tc.get_or_add_tcPr().append(tcBorders)
                for p in cell.paragraphs:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for r in p.runs:
                        r.font.name = 'Times New Roman'
                        r.font.size = Pt(8.5)
                        r.font.bold = True
        else:
            for c_idx, cell in enumerate(row.cells):
                set_cell_margins(cell, top=60, bottom=60, left=100, right=100)
                for p in cell.paragraphs:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
                    for r in p.runs:
                        r.font.name = 'Times New Roman'
                        r.font.size = Pt(8.0)


def enable_continuous_line_numbering(section):
    sectPr = section._sectPr
    lnNumType = OxmlElement('w:lnNumType')
    lnNumType.set(qn('w:countBy'), '1')
    lnNumType.set(qn('w:restart'), 'continuous')
    sectPr.append(lnNumType)


def add_page_number_to_footer(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = p.add_run("Page ")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    
    fldSimple = OxmlElement('w:fldSimple')
    fldSimple.set(qn('w:instr'), 'PAGE')
    p._p.append(fldSimple)


def add_formatted_text(paragraph, text: str):
    tokens = re.split(r'(\*\*.*?\*\*|\*.*?\*|\^\{.*?\}|\^[0-9a-zA-Z*])', text)
    for token in tokens:
        if not token:
            continue
        if token.startswith('**') and token.endswith('**'):
            run = paragraph.add_run(clean_latex(token[2:-2]))
            run.font.bold = True
        elif token.startswith('*') and token.endswith('*'):
            run = paragraph.add_run(clean_latex(token[1:-1]))
            run.font.italic = True
        elif token.startswith('^{') and token.endswith('}'):
            run = paragraph.add_run(clean_latex(token[2:-1]))
            run.font.superscript = True
        elif token.startswith('^') and len(token) > 1:
            run = paragraph.add_run(clean_latex(token[1:]))
            run.font.superscript = True
        else:
            run = paragraph.add_run(clean_latex(token))
        run.font.name = 'Times New Roman'


def build_supplementary_docx():
    print(f"📖 Reading markdown supplementary: {MD_PATH}")
    with open(MD_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    doc = docx.Document()

    section = doc.sections[0]
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)

    enable_continuous_line_numbering(section)
    add_page_number_to_footer(section)

    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(10.5)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.20
    normal_style.paragraph_format.space_after = Pt(4)

    h1 = doc.styles['Heading 1']
    h1.font.name = 'Times New Roman'
    h1.font.size = Pt(13)
    h1.font.bold = True
    h1.font.color.rgb = RGBColor(0x0f, 0x17, 0x2a)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(4)

    h2 = doc.styles['Heading 2']
    h2.font.name = 'Times New Roman'
    h2.font.size = Pt(11.5)
    h2.font.bold = True
    h2.font.color.rgb = RGBColor(0x1e, 0x29, 0x3b)
    h2.paragraph_format.space_before = Pt(10)
    h2.paragraph_format.space_after = Pt(3)

    h3 = doc.styles['Heading 3']
    h3.font.name = 'Times New Roman'
    h3.font.size = Pt(10.5)
    h3.font.bold = True
    h3.font.italic = True
    h3.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
    h3.paragraph_format.space_before = Pt(8)
    h3.paragraph_format.space_after = Pt(2)

    lines = content.split('\n')
    i = 0

    fig_map = {
        "1": "fig_S1_specparam_example.png",
        "2": "fig_S2_omst_topology.png",
        "3": "fig_S3_gain_invariance.png",
        "4": "fig_S4_cohort_spectra.png",
        "5": "fig_S5_enrollment_flowchart.png",
        "6": "fig_S6_permutation_tests.png"
    }

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        if line == '---':
            i += 1
            continue

        # Title
        if line.startswith('# '):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(clean_latex(line[2:]))
            run.font.name = 'Times New Roman'
            run.font.size = Pt(16)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x0f, 0x17, 0x2a)
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(10)
            i += 1
            continue

        # Heading 1
        if line.startswith('## '):
            h_text = clean_latex(line[3:])
            p = doc.add_heading(level=1)
            run = p.add_run(h_text)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(13)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x0f, 0x17, 0x2a)
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(4)
            i += 1
            continue

        # Heading 2
        if line.startswith('### '):
            if line.startswith('### Table '):
                p = doc.add_paragraph()
                run = p.add_run(clean_latex(line[4:]))
                run.font.name = 'Times New Roman'
                run.font.size = Pt(10)
                run.font.bold = True
                p.paragraph_format.space_before = Pt(10)
                p.paragraph_format.space_after = Pt(3)
                i += 1
                continue

            h_text = clean_latex(line[4:])
            p = doc.add_heading(level=2)
            run = p.add_run(h_text)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(11.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x1e, 0x29, 0x3b)
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(3)
            i += 1
            continue

        # Figure insertion from markdown image syntax
        if line.startswith('![Figure S'):
            m = re.search(r'!\[Figure S(\d+)\]', line)
            fig_num = m.group(1) if m else "1"
            img_file = fig_map.get(fig_num, f"fig_S{fig_num}_example.png")
            fig_path = REPORTS_DIR / img_file

            if fig_path.exists():
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.paragraph_format.space_before = Pt(8)
                p_img.paragraph_format.space_after = Pt(4)
                # Max width scaled cleanly
                w = 6.4 if fig_num in ["4", "5", "6"] else 5.8
                p_img.add_run().add_picture(str(fig_path), width=Inches(w))

            i += 1
            continue

        # Figure caption
        if line.startswith('**Figure S'):
            p_cap = doc.add_paragraph()
            add_formatted_text(p_cap, line)
            p_cap.paragraph_format.space_before = Pt(2)
            p_cap.paragraph_format.space_after = Pt(10)
            for r in p_cap.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(9.0)
            i += 1
            continue

        # Table Parsing
        if line.startswith('|') and '|' in line[1:]:
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip())
                i += 1

            if len(table_lines) >= 2:
                headers = [clean_latex(c.strip()) for c in table_lines[0].split('|')[1:-1]]
                data_rows = []
                for t_line in table_lines[1:]:
                    if re.match(r'^\|[\s\-:\|]+\|$', t_line) and set(t_line.replace('|', '').strip()).issubset({'-', ':', ' '}):
                        continue
                    cols = [clean_latex(c.strip()) for c in t_line.split('|')[1:-1]]
                    if cols:
                        data_rows.append(cols)

                table = doc.add_table(rows=len(data_rows) + 1, cols=len(headers))
                apply_three_line_table(table)

                for c_idx, h_text in enumerate(headers):
                    cell = table.cell(0, c_idx)
                    cell.paragraphs[0].text = ""
                    add_formatted_text(cell.paragraphs[0], h_text)

                for r_idx, row_data in enumerate(data_rows):
                    for c_idx, val in enumerate(row_data):
                        if c_idx < len(headers):
                            cell = table.cell(r_idx + 1, c_idx)
                            cell.paragraphs[0].text = ""
                            add_formatted_text(cell.paragraphs[0], val)

                apply_three_line_table(table)

                p_sp = doc.add_paragraph()
                p_sp.paragraph_format.space_before = Pt(2)
                p_sp.paragraph_format.space_after = Pt(4)
            continue

        # Regular paragraph
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.20
        p.paragraph_format.space_after = Pt(4)
        add_formatted_text(p, line)
        for r in p.runs:
            r.font.size = Pt(10)

        i += 1

    print(f"💾 Saving compiled supplementary docx to: {OUTPUT_DOCX}")
    doc.save(OUTPUT_DOCX)
    doc.save(OUTPUT_DOCX_ALT)
    print("✓ Successfully generated publication-ready supplementary material docx!")


if __name__ == "__main__":
    build_supplementary_docx()
