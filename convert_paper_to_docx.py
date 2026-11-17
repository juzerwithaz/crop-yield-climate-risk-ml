"""
convert_paper_to_docx.py
Converts RESEARCH_PAPER_DRAFT.md to RESEARCH_PAPER_DRAFT.docx with academic styling.
"""

import os
import re
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def add_styled_paragraph(doc, text, style_name='Normal', space_after=6, line_spacing=1.15, align=WD_ALIGN_PARAGRAPH.LEFT, bold_prefix=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    p.alignment = align
    
    # Process text with bold and formatting
    parts = re.split(r'(\*\*.*?\*\*|\*.*?\*|`.*?`)', text)
    for part in parts:
        if not part:
            continue
        if part.startswith('**') and part.endswith('**'):
            run = p.add_run(part[2:-2])
            run.bold = True
            run.font.name = 'Calibri'
            run.font.size = Pt(11)
            run.font.color.rgb = RGBColor(30, 41, 59)
        elif part.startswith('*') and part.endswith('*'):
            run = p.add_run(part[1:-1])
            run.italic = True
            run.font.name = 'Calibri'
            run.font.size = Pt(11)
            run.font.color.rgb = RGBColor(51, 65, 85)
        elif part.startswith('`') and part.endswith('`'):
            run = p.add_run(part[1:-1])
            run.font.name = 'Consolas'
            run.font.size = Pt(10)
            run.font.color.rgb = RGBColor(194, 65, 12)
        else:
            run = p.add_run(part)
            run.font.name = 'Calibri'
            run.font.size = Pt(11)
            run.font.color.rgb = RGBColor(51, 65, 85)
    return p

def convert_md_to_docx(md_path, docx_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    doc = docx.Document()
    
    # Page setup (Standard 1 inch margins)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    in_code_block = False
    code_lines = []
    in_table = False
    table_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\n')
        
        # Code block handler
        if line.startswith('```'):
            if in_code_block:
                # End code block
                table = doc.add_table(rows=1, cols=1)
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                cell = table.cell(0, 0)
                set_cell_background(cell, "F1F5F9")
                set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
                cp = cell.paragraphs[0]
                cp.paragraph_format.space_after = Pt(2)
                c_run = cp.add_run('\n'.join(code_lines))
                c_run.font.name = 'Consolas'
                c_run.font.size = Pt(9.5)
                c_run.font.color.rgb = RGBColor(15, 23, 42)
                p_spacer = doc.add_paragraph()
                p_spacer.paragraph_format.space_after = Pt(4)
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
                code_lines = []
            i += 1
            continue
            
        if in_code_block:
            code_lines.append(line)
            i += 1
            continue
            
        # Table handler
        if line.startswith('|') and line.endswith('|'):
            table_lines.append(line)
            # Peek next line
            if i + 1 < len(lines) and lines[i+1].startswith('|') and lines[i+1].endswith('|'):
                i += 1
                continue
            else:
                # Process full table
                rows = []
                for tl in table_lines:
                    # Skip delimiter row (|:---|:---:|)
                    if re.match(r'^\|[\s\:\-]+(?:\|[\s\:\-]+)*\|$', tl.strip()):
                        continue
                    cells = [c.strip() for c in tl.strip().strip('|').split('|')]
                    rows.append(cells)
                    
                if rows:
                    num_cols = max(len(r) for r in rows)
                    tbl = doc.add_table(rows=len(rows), cols=num_cols)
                    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                    for r_idx, r_data in enumerate(rows):
                        for c_idx in range(num_cols):
                            cell = tbl.cell(r_idx, c_idx)
                            val = r_data[c_idx] if c_idx < len(r_data) else ""
                            cell_p = cell.paragraphs[0]
                            cell_p.paragraph_format.space_after = Pt(2)
                            cell_p.paragraph_format.space_before = Pt(2)
                            set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
                            
                            if r_idx == 0:
                                set_cell_background(cell, "1E293B")
                                run = cell_p.add_run(re.sub(r'[\*\`]', '', val))
                                run.bold = True
                                run.font.name = 'Calibri'
                                run.font.size = Pt(9.5)
                                run.font.color.rgb = RGBColor(255, 255, 255)
                            else:
                                if r_idx % 2 == 1:
                                    set_cell_background(cell, "FFFFFF")
                                else:
                                    set_cell_background(cell, "F8FAFC")
                                run = cell_p.add_run(re.sub(r'[\*\`]', '', val))
                                run.font.name = 'Calibri'
                                run.font.size = Pt(9.5)
                                run.font.color.rgb = RGBColor(30, 41, 59)
                    p_spacer = doc.add_paragraph()
                    p_spacer.paragraph_format.space_after = Pt(6)
                table_lines = []
                i += 1
                continue
                
        # Title (# )
        if line.startswith('# '):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(12)
            run = p.add_run(line[2:])
            run.bold = True
            run.font.name = 'Calibri'
            run.font.size = Pt(20)
            run.font.color.rgb = RGBColor(15, 23, 42)
            i += 1
            continue
            
        # Heading 2 (## )
        if line.startswith('## '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run(line[3:])
            run.bold = True
            run.font.name = 'Calibri'
            run.font.size = Pt(14)
            run.font.color.rgb = RGBColor(30, 58, 138) # Dark blue
            i += 1
            continue
            
        # Heading 3 (### )
        if line.startswith('### '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(line[4:])
            run.bold = True
            run.font.name = 'Calibri'
            run.font.size = Pt(12)
            run.font.color.rgb = RGBColor(15, 23, 42)
            i += 1
            continue
            
        # Horizontal Rule (---)
        if line.strip() == '---':
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run("―" * 50)
            run.font.color.rgb = RGBColor(203, 213, 225)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
            continue
            
        # Bullet list
        if line.startswith('- ') or line.startswith('* '):
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15
            content = line[2:]
            parts = re.split(r'(\*\*.*?\*\*|\*.*?\*|`.*?`)', content)
            for part in parts:
                if not part:
                    continue
                if part.startswith('**') and part.endswith('**'):
                    run = p.add_run(part[2:-2])
                    run.bold = True
                    run.font.name = 'Calibri'
                    run.font.size = Pt(10.5)
                elif part.startswith('`') and part.endswith('`'):
                    run = p.add_run(part[1:-1])
                    run.font.name = 'Consolas'
                    run.font.size = Pt(9.5)
                    run.font.color.rgb = RGBColor(194, 65, 12)
                else:
                    run = p.add_run(part)
                    run.font.name = 'Calibri'
                    run.font.size = Pt(10.5)
            i += 1
            continue
            
        # Numbered list
        num_match = re.match(r'^(\d+)\.\s+(.*)', line)
        if num_match:
            p = doc.add_paragraph(style='List Number')
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15
            content = num_match.group(2)
            parts = re.split(r'(\*\*.*?\*\*|\*.*?\*|`.*?`)', content)
            for part in parts:
                if not part:
                    continue
                if part.startswith('**') and part.endswith('**'):
                    run = p.add_run(part[2:-2])
                    run.bold = True
                    run.font.name = 'Calibri'
                    run.font.size = Pt(10.5)
                elif part.startswith('`') and part.endswith('`'):
                    run = p.add_run(part[1:-1])
                    run.font.name = 'Consolas'
                    run.font.size = Pt(9.5)
                    run.font.color.rgb = RGBColor(194, 65, 12)
                else:
                    run = p.add_run(part)
                    run.font.name = 'Calibri'
                    run.font.size = Pt(10.5)
            i += 1
            continue
            
        # Normal prose paragraph
        if line.strip():
            # Check for authors/affiliation lines
            if 'Ansh Tiwari' in line or 'Department of Data Science' in line:
                add_styled_paragraph(doc, line, space_after=4, align=WD_ALIGN_PARAGRAPH.CENTER)
            else:
                add_styled_paragraph(doc, line, space_after=6)
                
        i += 1
        
    doc.save(docx_path)
    print(f"[OK] Saved Word manuscript to {docx_path}")

if __name__ == '__main__':
    convert_md_to_docx('RESEARCH_PAPER_DRAFT.md', 'RESEARCH_PAPER_DRAFT.docx')
