from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(r'E:\Projects\CN project')
OUT = ROOT / 'output/docx/PRESTO-R_Computer_Networks_Project_Final_Report.docx'
TMP = ROOT / 'tmp/docx'
LOGO = TMP / 'vit_logo.png'
FIGS = ROOT / 'results/figures'
REPORT_MD = ROOT / 'results/report.md'
BLUE = '17365D'
LIGHT_BLUE = 'EAF0F6'
GRAY = 'E7E6E6'
BLACK = '000000'
TNR = 'Times New Roman'
PAGE_W = Inches(8.5)
PAGE_H = Inches(11)
CONTENT_W = Inches(6.0)


def set_run_font(run, size=12, bold=None, italic=None, color=BLACK):
    run.font.name = TNR
    run._element.get_or_add_rPr().rFonts.set(qn('w:ascii'), TNR)
    run._element.get_or_add_rPr().rFonts.set(qn('w:hAnsi'), TNR)
    run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), TNR)
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn('w:shd'))
    if shd is None:
        shd = OxmlElement('w:shd')
        tc_pr.append(shd)
    shd.set(qn('w:fill'), fill)


def set_cell_margins(cell, top=70, start=85, bottom=70, end=85):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in('w:tcMar')
    if tc_mar is None:
        tc_mar = OxmlElement('w:tcMar')
        tc_pr.append(tc_mar)
    for m, v in [('top', top), ('start', start), ('bottom', bottom), ('end', end)]:
        node = tc_mar.find(qn(f'w:{m}'))
        if node is None:
            node = OxmlElement(f'w:{m}')
            tc_mar.append(node)
        node.set(qn('w:w'), str(v))
        node.set(qn('w:type'), 'dxa')


def set_table_borders(table):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in('w:tblBorders')
    if borders is None:
        borders = OxmlElement('w:tblBorders')
        tbl_pr.append(borders)
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        tag = 'w:' + edge
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn('w:val'), 'single')
        element.set(qn('w:sz'), '5')
        element.set(qn('w:space'), '0')
        element.set(qn('w:color'), '666666')


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement('w:tblHeader')
    tbl_header.set(qn('w:val'), 'true')
    tr_pr.append(tbl_header)


def add_page_field(paragraph):
    run = paragraph.add_run()
    set_run_font(run, 12)
    begin = OxmlElement('w:fldChar')
    begin.set(qn('w:fldCharType'), 'begin')
    instr = OxmlElement('w:instrText')
    instr.set(qn('xml:space'), 'preserve')
    instr.text = ' PAGE '
    separate = OxmlElement('w:fldChar')
    separate.set(qn('w:fldCharType'), 'separate')
    text = OxmlElement('w:t')
    text.text = '1'
    end = OxmlElement('w:fldChar')
    end.set(qn('w:fldCharType'), 'end')
    run._r.extend([begin, instr, separate, text, end])


def set_page_number_format(section, fmt, start=1):
    sect_pr = section._sectPr
    existing = sect_pr.find(qn('w:pgNumType'))
    if existing is not None:
        sect_pr.remove(existing)
    node = OxmlElement('w:pgNumType')
    node.set(qn('w:fmt'), fmt)
    node.set(qn('w:start'), str(start))
    sect_pr.append(node)


def add_bookmark(paragraph, name, bookmark_id):
    start = OxmlElement('w:bookmarkStart')
    start.set(qn('w:id'), str(bookmark_id))
    start.set(qn('w:name'), name)
    end = OxmlElement('w:bookmarkEnd')
    end.set(qn('w:id'), str(bookmark_id))
    paragraph._p.insert(0, start)
    paragraph._p.append(end)


def add_page_ref(paragraph, name):
    run = paragraph.add_run()
    set_run_font(run, 12)
    begin = OxmlElement('w:fldChar')
    begin.set(qn('w:fldCharType'), 'begin')
    instr = OxmlElement('w:instrText')
    instr.set(qn('xml:space'), 'preserve')
    instr.text = f' PAGEREF {name} \\h '
    sep = OxmlElement('w:fldChar')
    sep.set(qn('w:fldCharType'), 'separate')
    text = OxmlElement('w:t')
    text.text = '00'
    end = OxmlElement('w:fldChar')
    end.set(qn('w:fldCharType'), 'end')
    run._r.extend([begin, instr, sep, text, end])


def configure_section(section, roman=False, start=1, footer=True):
    section.page_width = PAGE_W
    section.page_height = PAGE_H
    section.left_margin = Inches(1.5)
    section.right_margin = Inches(1.0)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.header_distance = Inches(0.5)
    section.footer_distance = Inches(0.5)
    section.different_first_page_header_footer = False
    set_page_number_format(section, 'lowerRoman' if roman else 'decimal', start)
    section.footer.is_linked_to_previous = False
    p = section.footer.paragraphs[0]
    p.clear()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if footer:
        add_page_field(p)


def fmt_para(paragraph, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=True,
             before=0, after=0, line=1.5, keep=False):
    paragraph.alignment = align
    fmt = paragraph.paragraph_format
    fmt.line_spacing = line
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.first_line_indent = Inches(0.5) if first_indent else Inches(0)
    fmt.keep_together = keep


def add_body(doc, text, first=False, bold_lead=None):
    p = doc.add_paragraph()
    fmt_para(p, first_indent=not first, after=0, line=1.5)
    if bold_lead and text.startswith(bold_lead):
        r = p.add_run(bold_lead)
        set_run_font(r, bold=True)
        r = p.add_run(text[len(bold_lead):])
        set_run_font(r)
    else:
        r = p.add_run(text)
        set_run_font(r)
    return p


def add_chapter(doc, n, title, intro):
    if n != 1:
        doc.add_page_break()
    p = doc.add_paragraph()
    fmt_para(p, WD_ALIGN_PARAGRAPH.CENTER, first_indent=False, after=0, line=1.0, keep=True)
    r = p.add_run(f'CHAPTER {n}')
    set_run_font(r, 14, True)
    p = doc.add_paragraph()
    fmt_para(p, WD_ALIGN_PARAGRAPH.CENTER, first_indent=False, before=0, after=20, line=1.0, keep=True)
    r = p.add_run(title.upper())
    set_run_font(r, 14, True)
    b = add_body(doc, intro, first=True)
    return b


def add_sub(doc, title, paragraphs, bookmark=None, bookmark_id=1, page_break=False):
    if page_break:
        doc.add_page_break()
    p = doc.add_paragraph()
    p.style = doc.styles['Normal']
    fmt_para(p, WD_ALIGN_PARAGRAPH.LEFT, first_indent=False, before=8, after=2, line=1.0, keep=True)
    r = p.add_run(title)
    set_run_font(r, 12, True)
    if bookmark:
        add_bookmark(p, bookmark, bookmark_id)
    for i, text in enumerate(paragraphs):
        add_body(doc, text, first=False)
    return p


def add_bullet(doc, text, number=False):
    p = doc.add_paragraph(style='List Number' if number else 'List Bullet')
    fmt_para(p, WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=False, after=0, line=1.5)
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.first_line_indent = Inches(-0.25)
    r = p.add_run(text)
    set_run_font(r)
    return p


def add_table(doc, caption, headers, rows, widths=None, chapter=1, table_idx=1, caption_name=None,
              first_cell_left=True, cell_font=12):
    cap_text = f'Table {chapter}.{table_idx} {caption}'
    p = doc.add_paragraph()
    fmt_para(p, WD_ALIGN_PARAGRAPH.CENTER, first_indent=False, before=8, after=4, line=1.0, keep=True)
    r = p.add_run(cap_text)
    set_run_font(r, 10)
    if caption_name:
        add_bookmark(p, caption_name, 1000 + table_idx + chapter*100)
    t = doc.add_table(rows=1, cols=len(headers))
    t.autofit = False
    t.alignment = 1
    set_table_borders(t)
    if widths is None:
        widths = [CONTENT_W / len(headers)] * len(headers)
    for c, (cell, label) in enumerate(zip(t.rows[0].cells, headers)):
        cell.width = widths[c]
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_shading(cell, LIGHT_BLUE)
        set_cell_margins(cell)
        p0 = cell.paragraphs[0]
        fmt_para(p0, WD_ALIGN_PARAGRAPH.CENTER, first_indent=False, after=0, line=1.0)
        rr = p0.add_run(str(label))
        set_run_font(rr, cell_font, True)
    set_repeat_table_header(t.rows[0])
    for row in rows:
        cells = t.add_row().cells
        tr_pr = t.rows[-1]._tr.get_or_add_trPr()
        tr_pr.append(OxmlElement('w:cantSplit'))
        for c, (cell, val) in enumerate(zip(cells, row)):
            cell.width = widths[c]
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            p0 = cell.paragraphs[0]
            fmt_para(p0, WD_ALIGN_PARAGRAPH.LEFT if c == 0 and first_cell_left else WD_ALIGN_PARAGRAPH.CENTER,
                     first_indent=False, after=0, line=1.0)
            rr = p0.add_run(str(val))
            set_run_font(rr, cell_font)
    return t


def add_figure(doc, path, caption, chapter, idx, bookmark=None, width=5.8):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_together = True
    p.add_run().add_picture(str(path), width=Inches(width))
    cp = doc.add_paragraph()
    fmt_para(cp, WD_ALIGN_PARAGRAPH.CENTER, first_indent=False, before=0, after=6, line=1.0, keep=True)
    r = cp.add_run(f'Fig. {chapter}.{idx} {caption}')
    set_run_font(r, 10)
    if bookmark:
        add_bookmark(cp, bookmark, 2000 + chapter*100 + idx)
    return cp


def add_equation(doc, text, number):
    t = doc.add_table(rows=1, cols=2)
    t.autofit = False
    t.alignment = 1
    t.columns[0].width = Inches(5.2)
    t.columns[1].width = Inches(0.8)
    for c, value, align in [(t.cell(0,0), text, WD_ALIGN_PARAGRAPH.CENTER),
                            (t.cell(0,1), f'({number})', WD_ALIGN_PARAGRAPH.RIGHT)]:
        p = c.paragraphs[0]
        fmt_para(p, align, first_indent=False, before=2, after=2, line=1.0)
        set_run_font(p.add_run(value), 12)
    return t


def md_tables_by_heading(path):
    lines = path.read_text(encoding='utf-8').splitlines()
    result = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if (line.startswith('### T') or line.startswith('#### T') or
                line.startswith('#### RGG test') or line.startswith('#### Horizon-sweep') or
                line.startswith('### Mixed-scenario')):
            title = line.strip('# ').strip()
            tab = []
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            while j < len(lines) and lines[j].startswith('|'):
                cells = [x.strip() for x in lines[j].strip().strip('|').split('|')]
                if not all(set(x) <= {'-', ':'} for x in cells):
                    tab.append(cells)
                j += 1
            if tab:
                result[title] = tab
            i = j
        else:
            i += 1
    return result


def md_table(data, prefix):
    for k, v in data.items():
        if k.startswith(prefix):
            return v
    raise KeyError(prefix)


def fmt(v):
    if pd.isna(v) or v == '':
        return 'n/a'
    if isinstance(v, float):
        return f'{v:.6g}'
    return str(v).replace('—', 'n/a').replace('–', '-')


def split_csv_rows(csv_path):
    return pd.read_csv(csv_path).replace({float('nan'): 'n/a'}).fillna('n/a')


def add_metric_table(doc, title, headers, rows, chapter, idx):
    # Dense result matrices use 10 pt cells to keep labels and decimals intact
    # within the sample's portrait page width; narrative and compact tables stay 12 pt.
    n = len(headers)
    weights = [1.0] * n
    if n > 4:
        weights[0] = 1.45
        weights[-1] = 1.2
    total = sum(weights)
    widths = [CONTENT_W * w / total for w in weights]
    return add_table(doc, title, headers, rows, widths, chapter, idx,
                     cell_font=10 if n > 4 else 12)


def make_architecture(path):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new('RGB', (1800, 620), 'white')
    d = ImageDraw.Draw(im)
    regular = ImageFont.truetype(r'C:\Windows\Fonts\times.ttf', 30)
    bold = ImageFont.truetype(r'C:\Windows\Fonts\timesbd.ttf', 32)
    navy, blue, pale, ink = '#17365D', '#2D5D87', '#EAF0F6', '#17212B'
    items = [
        ('SIMULATOR', 'Topologies, links,\ntraffic, failures'),
        ('TELEMETRY', 'Noisy delay, loss,\nqueue, utilization'),
        ('GRU RISK', 'Link degradation\nprobabilities'),
        ('DQN POLICY', 'Choose one of four\ncandidate paths'),
        ('ROLLOUTS', 'Held-out episodes\nand metric analysis'),
    ]
    x0, bw, gap, y, bh = 30, 315, 40, 188, 230
    for i, (title, body) in enumerate(items):
        x = x0 + i*(bw+gap)
        d.rounded_rectangle((x,y,x+bw,y+bh),radius=18,fill=pale,outline=blue,width=5)
        box=d.textbbox((0,0),title,font=bold)
        d.text((x+(bw-(box[2]-box[0]))/2,y+40),title,font=bold,fill=navy)
        for j,line in enumerate(body.split('\n')):
            box=d.textbbox((0,0),line,font=regular)
            d.text((x+(bw-(box[2]-box[0]))/2,y+112+j*50),line,font=regular,fill=ink)
        if i < len(items)-1:
            ax=x+bw+7; ex=ax+gap-12; ay=y+bh/2
            d.line((ax,ay,ex,ay),fill=blue,width=6)
            d.polygon([(ex,ay),(ex-15,ay-10),(ex-15,ay+10)],fill=blue)
    d.text((45, 500), 'Separate train, validation, and held-out test seeds; preserve resolved configuration and episode logs.', font=regular, fill='#4C5C6B')
    im.save(path)


def set_heading_styles(doc):
    normal = doc.styles['Normal']
    normal.font.name = TNR
    normal._element.rPr.rFonts.set(qn('w:ascii'), TNR)
    normal._element.rPr.rFonts.set(qn('w:hAnsi'), TNR)
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(0)
    for name in ('Title', 'Heading 1', 'Heading 2', 'Heading 3', 'List Bullet', 'List Number'):
        style = doc.styles[name]
        style.font.name = TNR
        style._element.rPr.rFonts.set(qn('w:ascii'), TNR)
        style._element.rPr.rFonts.set(qn('w:hAnsi'), TNR)
        style.font.size = Pt(12)
        style.font.color.rgb = RGBColor.from_string(BLACK)


def add_page_heading(doc, text, style='Title'):
    p = doc.add_paragraph()
    p.style = doc.styles[style]
    fmt_para(p, WD_ALIGN_PARAGRAPH.CENTER, first_indent=False, after=12, line=1.0, keep=True)
    set_run_font(p.add_run(text), 14 if style != 'Title' else 16, style != 'Title')
    return p


def add_front_page_number(doc, title, roman_section=False, new_page=True):
    if new_page:
        doc.add_page_break()
    add_page_heading(doc, title, 'Heading 1')


def source_excerpt(path, start, end):
    lines = path.read_text(encoding='utf-8').splitlines()
    return '\n'.join(lines[start-1:end])


def add_code(doc, label, text):
    p=doc.add_paragraph()
    fmt_para(p, WD_ALIGN_PARAGRAPH.LEFT, first_indent=False, before=5, after=2, line=1.0, keep=True)
    set_run_font(p.add_run(label), 12, True)
    for line in text.splitlines():
        pp=doc.add_paragraph()
        fmt_para(pp, WD_ALIGN_PARAGRAPH.LEFT, first_indent=False, before=0, after=0, line=1.0)
        pp.paragraph_format.left_indent=Inches(0.18)
        rr=pp.add_run(line if line else ' ')
        set_run_font(rr, 12)


doc = Document()
set_heading_styles(doc)
sec = doc.sections[0]
configure_section(sec, roman=True, start=1, footer=False)
sec.footer.paragraphs[0].clear()

# Cover page, styled to the supplied VIT course-report sample.
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(20); p.paragraph_format.space_after=Pt(6)
p.add_run().add_picture(str(LOGO), width=Inches(2.65))
for text, size, bold in [
    ('SCHOOL OF INFORMATION TECHNOLOGY AND ENGINEERING', 12, True),
    ('BCSE308P COURSE PROJECT', 14, True),
    ('COMPUTER NETWORKS', 14, True),
    ('PRESTO-R: Predictive, Stability-Aware Reinforcement Learning Routing', 16, True),
]:
    p=doc.add_paragraph(); fmt_para(p, WD_ALIGN_PARAGRAPH.CENTER, False, before=5, after=5, line=1.1)
    set_run_font(p.add_run(text), size, bold)
p=doc.add_paragraph(); fmt_para(p, WD_ALIGN_PARAGRAPH.CENTER, False, before=16, after=8, line=1.1)
set_run_font(p.add_run('Formal Project Final Report'), 12, True)
cover_data = [
    ('Submitted by', 'Yashwanth Gajula'),
    ('Registration Number', '24BDE0079'),
    ('Slot', 'C2+TC2'),
    ('Faculty Guide', 'Dr. Navaneethan C'),
    ('Semester', 'Fall Semester 2026-2027'),
    ('Date', '23 October 2026'),
]
t=doc.add_table(rows=0, cols=2); t.autofit=False; t.alignment=1; set_table_borders(t)
for label,value in cover_data:
    cells=t.add_row().cells; cells[0].width=Inches(2.0); cells[1].width=Inches(3.4)
    for c in cells: set_cell_margins(c, 40, 90, 40, 90)
    p0=cells[0].paragraphs[0]; fmt_para(p0, WD_ALIGN_PARAGRAPH.LEFT, False, line=1.0)
    set_run_font(p0.add_run(label),12,True)
    p1=cells[1].paragraphs[0]; fmt_para(p1, WD_ALIGN_PARAGRAPH.LEFT, False, line=1.0)
    set_run_font(p1.add_run(value),12)
p=doc.add_paragraph(); fmt_para(p, WD_ALIGN_PARAGRAPH.CENTER, False, before=14, line=1.0)
set_run_font(p.add_run('VIT Vellore'),12,True)

# Roman-numbered preliminary matter begins at page ii.
sec2=doc.add_section(WD_SECTION.NEW_PAGE); configure_section(sec2, roman=True, start=2, footer=True)

add_front_page_number(doc,'DECLARATION',new_page=False)
for text in [
    'I, Yashwanth Gajula, Registration Number 24BDE0079, declare that this report presents the PRESTO-R Computer Networks project and identifies its methods, data sources, software, evaluation procedures, results, limitations, and references. The numerical results are transcribed from the saved project outputs and are presented with their stated aggregation and uncertainty conventions.',
    'I have distinguished project-generated evidence from interpretation and have cited the research and software sources used to describe the methods. The reported evaluation is limited to the simulator configurations, scenarios, seeds, and saved model versions documented in the project repository.',
    'Place: Vellore, Tamil Nadu\nDate: 23 October 2026',
]: add_body(doc,text,first=(text==''))
p=doc.add_paragraph(); fmt_para(p,WD_ALIGN_PARAGRAPH.LEFT,False,before=36,line=1.0); set_run_font(p.add_run('Signature of the Candidate: ______________________________'),12)

add_front_page_number(doc,'CERTIFICATE')
add_body(doc,'This report documents the project titled PRESTO-R: Predictive, Stability-Aware Reinforcement Learning Routing, submitted by Yashwanth Gajula (24BDE0079) for BCSE308P Computer Networks at VIT Vellore under the guidance of Dr. Navaneethan C.',first=True)
add_body(doc,'The report records the project design, implementation, experimental evaluation, findings, and limitations. Faculty review and certification are reserved for the authorized guide.',first=False)
for line in ['Faculty Guide: Dr. Navaneethan C','Signature: ____________________________________________','Date: ____________________','Place: Vellore']:
    p=doc.add_paragraph(); fmt_para(p,WD_ALIGN_PARAGRAPH.LEFT,False,before=16,line=1.0); set_run_font(p.add_run(line),12)

add_front_page_number(doc,'ACKNOWLEDGEMENT')
add_body(doc,'I express my sincere gratitude to Dr. Navaneethan C for academic guidance and feedback during the Computer Networks project. I also acknowledge VIT Vellore for the academic setting in which the work was developed.',first=True)
add_body(doc,'I am grateful to the open-source maintainers whose libraries support the simulator, machine-learning workflow, statistical analysis, and visualizations. The report preserves the project’s recorded results, including unsuccessful criteria and methodological limitations, so that the conclusions remain traceable to the evidence.',first=False)
add_body(doc,'Yashwanth Gajula\n24BDE0079',first=False)

add_front_page_number(doc,'EXECUTIVE SUMMARY')
summary=('PRESTO-R combines link-degradation prediction with deep reinforcement learning and a penalty for changing routes. The project implements a discrete-time fluid-network simulator, episode-separated telemetry datasets, logistic-regression and GRU predictors, reactive and predictive DQN baselines, and a stability-aware DQN routing policy. The completed evaluation covers NSFNET and random geometric graph topologies, five traffic and failure scenarios, held-out scenario seeds, horizon and penalty sweeps, and three robustness conditions. In NSFNET mixed episodes, PRESTO-R averaged 5.197 route switches and PDR 0.9625, compared with 20.988 switches and PDR 0.9608 for predictive DQN. In RGG mixed episodes, PRESTO-R averaged 1.913 switches and PDR 0.9545, compared with 11.873 switches and PDR 0.9546. The largest unresolved gap is reliable early warning: GRU onset F1 was 0.5296 against a target of 0.7000. PRESTO-R did not pass every comparison with reactive DQN, and the no-prediction plus selected-penalty factorial cell was not run. Findings therefore support reduced route churn in the tested simulator, not production-network performance.')
add_body(doc,summary,first=True)
p=doc.add_paragraph(); fmt_para(p,WD_ALIGN_PARAGRAPH.LEFT,False,before=12,line=1.0); set_run_font(p.add_run(f'Word count: {len(summary.split())}'),10)

toc_entries = [
('CHAPTER 1 INTRODUCTION','ch1'),('1.1 Objective','h11'),('1.2 Motivation','h12'),('1.3 Background','h13'),('1.4 Problem Statement','h14'),('1.5 Gap Identification','h15'),('1.6 Objectives and Evaluation Alignment','h16'),('1.7 Scope and Contributions','h17'),
('CHAPTER 2 PROJECT DESCRIPTION AND GOALS','ch2'),('2.1 Project Overview','h21'),('2.2 Goals and Acceptance Criteria','h22'),('2.3 System Boundary','h23'),('2.4 Actors and Operating Workflow','h24'),('2.5 Assumptions','h25'),('2.6 Evidence and Deliverables','h26'),
('CHAPTER 3 TECHNICAL SPECIFICATION','ch3'),('3.1 Network and Simulator Specification','h31'),('3.2 Topology Specification','h32'),('3.3 Traffic and Failure Specification','h33'),('3.4 Telemetry and State Specification','h34'),('3.5 Predictor and Agent Specification','h35'),('3.6 Software Stack','h36'),('3.7 Data and Reproducibility Specification','h37'),('3.8 Metric Definitions','h38'),
('CHAPTER 4 DESIGN APPROACH AND DETAILS','ch4'),('4.1 Design Approach Materials and Methods','h41'),('4.2 Codes and Standards','h42'),('4.3 Constraints Alternatives and Tradeoffs','h43'),('4.4 Statistical Design and Validity','h44'),
('CHAPTER 5 SCHEDULE TASKS AND MILESTONES','ch5'),('5.1 Work Breakdown','h51'),('5.2 Phase Schedule and Status','h52'),('5.3 Milestone Evidence','h53'),('5.4 Reproducibility and Release','h54'),
('CHAPTER 6 PROJECT DEMONSTRATION','ch6'),('6.1 Sample Codes','h61'),('6.2 Sample Screenshots','h62'),('6.3 End-to-End Demonstration','h63'),
('CHAPTER 7 COST ANALYSIS RESULT AND DISCUSSION','ch7'),('7.1 Cost and Resource Analysis','h71'),('7.2 Evaluation Protocol','h72'),('7.3 Predictor Results','h73'),('7.4 NSFNET Results','h74'),('7.5 Paired Tests','h75'),('7.6 Ablation and Penalty Sweep','h76'),('7.7 Horizon Sensitivity','h77'),('7.8 RGG Results','h78'),('7.9 Robustness Evaluation','h79'),('7.10 Success Criteria','h710'),('7.11 Findings by Scenario','h711'),('7.12 Threats to Validity','h712'),
('CHAPTER 8 SUMMARY','ch8'),('8.1 Objective Achievement','h81'),('8.2 Conclusions','h82'),('8.3 Recommendations','h83'),('8.4 Future Work','h84'),
('CHAPTER 9 REFERENCES','ch9'),
('APPENDIX A REPRODUCTION AND DATA DICTIONARY','appA')]

add_front_page_number(doc,'CONTENTS')
page_map_path=TMP/'page_map.json'
page_map=json.loads(page_map_path.read_text(encoding='utf-8')) if page_map_path.exists() else {}
for label,key in toc_entries:
    p=doc.add_paragraph()
    p.paragraph_format.line_spacing=1.0; p.paragraph_format.space_after=Pt(2)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(5.85), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
    if label.startswith('CHAPTER') or label.startswith('APPENDIX'):
        p.paragraph_format.left_indent=Inches(0)
        set_run_font(p.add_run(label),12,True)
    else:
        p.paragraph_format.left_indent=Inches(0.25)
        set_run_font(p.add_run(label),12)
    p.add_run('\t')
    set_run_font(p.add_run(str(page_map.get(key,'00'))),12)

figures = json.loads((TMP/'figure_list.json').read_text(encoding='utf-8'))
add_front_page_number(doc,'LIST OF FIGURES')
for item in figures:
    label, page = item['label'], item['page']
    p=doc.add_paragraph(); p.paragraph_format.line_spacing=1.0; p.paragraph_format.space_after=Pt(5)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(5.85), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
    set_run_font(p.add_run(label),12); p.add_run('\t'); set_run_font(p.add_run(str(page)),12)

tables_for_list = json.loads((TMP/'table_list.json').read_text(encoding='utf-8'))
add_front_page_number(doc,'LIST OF TABLES')
for item in tables_for_list:
    label, page = item['label'], item['page']
    p=doc.add_paragraph(); p.paragraph_format.line_spacing=1.0; p.paragraph_format.space_after=Pt(4)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(5.85), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
    set_run_font(p.add_run(label),12); p.add_run('\t'); set_run_font(p.add_run(str(page)),12)

add_front_page_number(doc,'LIST OF ABBREVIATIONS')
abbr=[('AI','Artificial Intelligence'),('AUC','Area Under The Curve'),('CPU','Central Processing Unit'),('DQN','Deep Q-Network'),('F1','Harmonic Mean Of Precision And Recall'),('GRU','Gated Recurrent Unit'),('H','Prediction Horizon'),('NSFNET','National Science Foundation Network Topology'),('PDR','Packet Delivery Ratio'),('PR-AUC','Area Under The Precision-Recall Curve'),('RGG','Random Geometric Graph'),('RL','Reinforcement Learning'),('SB3','Stable-Baselines3'),('SDE','Standard Deviation Across Training-Seed Means'),('TNR','Times New Roman'),('VIT','Vellore Institute Of Technology'),('W','Telemetry Window Length')]
add_table(doc,'Abbreviations Used In This Report',['Abbreviation','Meaning'],abbr,[Inches(1.5),Inches(4.5)],0,1)

add_front_page_number(doc,'SYMBOLS AND NOTATIONS')
symbols=[('α','Delay weight in the reward'),('β','Loss weight in the reward'),('γ','Throughput weight in the reward'),('λ','Route-switch penalty'),('μ','Hop or energy weight'),('H','Prediction and event-analysis horizon'),('W','Telemetry history window'),('PDR','Delivered traffic divided by offered demand'),('p_Holm','Holm-adjusted paired-test p-value'),('r','Signed effect-size statistic reported by paired tests'),('σ','Across-training-seed standard deviation')]
add_table(doc,'Symbols And Notations',['Symbol','Meaning'],symbols,[Inches(1.5),Inches(4.5)],0,2)

# Main report begins with Arabic page 1.
sec3=doc.add_section(WD_SECTION.NEW_PAGE); configure_section(sec3,roman=False,start=1,footer=True)
md=md_tables_by_heading(REPORT_MD)
chapter_bookmark_ids={}

# CHAPTER 1
add_chapter(doc,1,'Introduction','Computer networks carry traffic over paths whose quality changes as load, link health, and failures evolve. A routing policy must preserve delivery and recover from faults without destabilizing the network through excessive route changes. PRESTO-R examines a specific design hypothesis: adding predicted link risk to a learned route selector can help it anticipate degradation, while a switching cost can restrain unnecessary path changes. The project implements that hypothesis in a controlled simulator and evaluates it against reactive learning, prediction without the selected penalty, and shortest-path failover. The central result is a substantial reduction in route switching in the evaluated mixed scenarios, with PDR remaining close to the learned baselines. The evidence also identifies an unresolved weakness: the predictor misses the predeclared onset-F1 target, and PRESTO-R fails some comparisons against reactive DQN.')
add_sub(doc,'1.1 Objective',[
'The primary objective is to design, implement, and evaluate a route-selection system that augments current path measurements with an estimate of near-future link degradation. The action space consists of four candidate paths selected for a managed source-destination flow. The system must expose a stable observation vector, compute an interpretable reward, and record enough event-level truth to evaluate delivery, switching, recovery, proactive response, and prediction quality.',
'A second objective is to test whether a route-switch penalty reduces route churn without materially reducing packet delivery. The policy comparison separates three effects: reactive DQN observes current measurements only; predictive DQN adds learned risk; PRESTO-R adds risk and the selected switching penalty. A shortest-path failover heuristic supplies a non-learning reference. These internal baselines do not claim to reproduce external routing algorithms beyond their implemented behavior.',
'A third objective is to characterize the predictor itself. Logistic regression and a GRU are trained from simulator-derived telemetry, and performance is reported on all samples and on the onset subset. Subgroup results distinguish scenario, topology, and event type. This distinction matters because an all-sample score can look strong when healthy observations dominate, while onset detection remains weak.',
'The evaluation objective is to compare these systems under the same held-out scenario seeds across NSFNET and a random geometric graph. It includes steady, congestion, node-failure, fluctuation, and mixed scenarios, plus horizon sensitivity and evaluation-only robustness conditions. Test data is reserved for final measurement and is not used to retune the policy or risk threshold.'
])
add_sub(doc,'1.2 Motivation',[
'Reactive routing observes a path after its delay, loss, or availability has already changed. Such feedback can be sufficient for many systems, but a policy that selects among candidate paths at each step may react to noisy or transient measurements. Repeated switching is costly even when a new path is not meaningfully better: it can complicate operations, produce unstable forwarding decisions, and create transient performance penalties. The project therefore treats route stability as a measurable outcome alongside packet delivery.',
'Prediction provides a possible earlier signal. A temporal model can use recent link telemetry to estimate whether a link will become degraded within a selected horizon. A routing policy can then include risk as one of its path features. The signal is imperfect: the predictor can produce false alarms, miss sudden failures, or generalize poorly across scenarios. For this reason, PRESTO-R is evaluated on held-out episodes and compared with a policy that has the same predictive features but no selected switch cost.',
'A switching penalty provides a second control. It discourages changing routes for small or temporary differences, but it can also delay necessary movement. The design question is not whether switching is always harmful; switching is useful when a selected route is failing. The objective is to reduce avoidable switches while preserving delivery and recovery. The project reports proactive rate and recovery time to make this balance visible rather than relying on switch count alone.',
'The project uses simulation because the full training grid requires repeatable interactions across many combinations of topology, event type, and seed. A discrete-time fluid model permits many rollouts at manageable cost. The controlled setting supports reproducible comparisons, but it also limits external validity. It cannot reproduce every packet-level, transport, routing-protocol, or operational effect of a production network.'
])
add_sub(doc,'1.3 Background',[
'Reinforcement learning models sequential decisions as interactions between an agent and an environment. At each time step, the agent receives an observation, chooses an action, and receives a reward. A DQN estimates the value of each available action from the observation and selects a high-value action. Mnih and colleagues demonstrated deep Q-networks as a method for learning action values with neural representations [1]. PRESTO-R uses the same broad DQN family through the project’s Stable-Baselines3 training interface; its network states and action meanings are specific to this routing simulator.',
'A GRU is a recurrent neural network unit intended to summarize sequence history. In this project, the predictor reads a window of recent telemetry for each link and emits a probability that the link will be bad within a future horizon. The use of a sequence model follows work introducing gated recurrent units in recurrent encoder-decoder learning [2]. The project also trains logistic regression as a simpler baseline. Comparing the two helps distinguish the contribution of sequence modeling from the existence of any predictive score.',
'Candidate-path routing reduces a large network decision to a small discrete action set. The project uses up to four loop-free paths ordered by base delay. The general problem of finding K shortest loopless paths was studied by Yen [3]. In the implementation, NetworkX provides a shortest-simple-path generator weighted by base delay; endpoints are sampled so the best two alternatives have limited link overlap. Thus actions have fixed meaning within an episode and have enough diversity for switching to be meaningful.',
'Several performance concepts are needed. PDR summarizes delivered traffic relative to offered demand. Mean path delay summarizes latency for traffic delivered. Recovery time measures how long a selected path remains outside the configured quality threshold after a failure. Proactive rate measures failures for which the policy has already left the affected path by the failure time. Route-switch count and unnecessary-switch rate expose stability behavior. The predictor uses precision, recall, F1, PR-AUC, ROC-AUC, Brier score, and median lead time to characterize both ranking and thresholded detection.'
])
add_sub(doc,'1.4 Problem Statement',[
'The project addresses the problem of selecting among a small number of candidate network paths when link quality is only partially observed and may degrade after the current observation. The controller sees noisy delay, loss, queue, utilization, and up/down measurements. Link health and future failure schedules remain hidden from the learned routing agent. At each decision point, the agent chooses one path and is evaluated on delivered traffic, delay, route stability, and recovery behavior.',
'The engineering challenge is to combine prediction and control without allowing the predictor to dominate current evidence or causing excessive route movement. A predictor can produce a risk estimate, but it does not itself select a route. The DQN must learn when the estimated risk justifies paying the cost of a switch. Reward weights, candidate-route quality, telemetry noise, failure timing, and the prediction horizon all affect this decision. The experiment therefore requires both controlled comparisons and explicit limits on what can be inferred from a simulated environment.',
'The project question is: under the tested simulator configurations, can a risk-aware DQN with a selected switch penalty reduce route changes relative to reactive and predictive baselines while maintaining comparable delivery and recovery outcomes? A related question is whether the link-risk predictor reaches the predeclared onset-detection target. The report answers these separately because routing stability can improve even when onset prediction is below target.'
])
add_sub(doc,'1.5 Gap Identification',[
'The project-specific gap is the lack of a complete, controlled comparison showing how predicted risk and switching cost interact in the implemented routing environment. A reactive policy can show how DQN behaves with current telemetry. A predictive policy can show what changes when risk features are supplied. PRESTO-R can show the combined risk-plus-penalty policy. A shortest-path failover policy supplies a simple reference. The gap is especially important because prediction may prompt more switching, while a penalty may suppress both unnecessary and necessary switches.',
'The design addresses this gap by using a shared simulator, candidate-path interface, rollout code, and episode-level metric definitions. The comparison is repeated across two topologies and five scenario classes. It includes held-out seed matching, across-training-seed uncertainty, paired tests, predictor subgroup metrics, a lambda sweep, horizon sensitivity, and robustness checks. This structure allows the report to state where PRESTO-R reduces route churn, where PDR remains comparable, and where predefined criteria are not met.',
'The evidence leaves two material gaps open. First, the completed grid omits the no-prediction plus selected-penalty condition, so it does not form a full two-by-two factorial design. The available ablation cannot isolate a causal main effect for prediction and a causal main effect for switching cost. Second, the onset predictor misses its 0.70 F1 target. These are reported as limitations rather than filled with inferred values. Sudden-failure results also demonstrate that precursor-based prediction cannot warn about events that have no precursor.',
'This project does not claim to close a general scientific gap for all networks or reproduce an external RQSMR implementation. Its narrower contribution is a reproducible simulator-based evaluation of a specific risk-aware routing design, with negative and positive criteria reported together. This scope statement prevents simulated improvements from being presented as production-network evidence.'
])
add_sub(doc,'1.6 Objectives and Evaluation Alignment',[
'The evaluation sheet groups the project into four criteria. Criterion A covers the problem statement, gap identification, and objectives. This report makes the routing-stability problem explicit, names the missing comparison, and ties each objective to a measurable outcome. Criterion B covers design methodology, innovation, and creativity. The design combines sequence-based risk, a shared path-feature state, DQN action values, a switch cost, and a held-out evaluation plan; the report describes this as a project-level design combination rather than a claim of unqualified algorithmic novelty.',
'Criterion C covers implementation, findings, and conclusion. The report details the simulator, telemetry, risk models, reward, training interface, evaluation pipeline, result tables, paired tests, robustness checks, limitations, and follow-up work. Criterion D covers presentation and project report. The document follows the supplied Times New Roman and page-layout rules, includes front matter, numbered chapters, figures and tables with captions, references, appendices, and an evaluation evidence map. No marks are assigned in this report; the rubric values shown in the evaluation sheet are maximum weights, not scores.'
])
add_table(doc,'Evaluation Criteria And Evidence Map',['Criterion','Maximum Weight','Report Evidence'],[
('A. Problem Statement, Gap Identification And Objectives','5','Chapter 1: problem, specific gap, objectives, scope, and evaluation alignment'),
('B. Design Methodology, Innovation And Creativity','5','Chapter 4: architecture, methods, alternatives, trade-offs, and validity'),
('C. Implementation, Findings And Conclusion','10','Chapters 3, 6, 7, and 8: specifications, implementation, complete reported outcomes, conclusions'),
('D. Presentation And Project Report','10','Full document structure, formatting, figures, tables, citations, references, and reproducibility record'),
], [Inches(2.3),Inches(0.9),Inches(2.8)],1,1)
add_sub(doc,'1.7 Scope and Contributions',[
'The core scope includes implementation phases 0 through 6: repository setup, fluid simulator, failure schedules, telemetry and state construction, logistic regression and GRU predictors, heuristic baselines, reactive and predictive DQN policies, PRESTO-R, experiment grid, statistical reporting, figures, and local model packaging. The project includes NSFNET and RGG results, sensitivity sweeps, and three evaluation-only robustness variants. Optional NS-3 validation is outside the completed work.',
'The report contributes a traceable specification of the implemented system; an explicit statement of the research and evaluation gap; complete reporting of aggregate tables saved in the project outputs; comparison of positive and failed success criteria; and a list of limitations that constrain interpretation. It also preserves exact project metrics and distinguishes seed-level standard deviations from episode-level paired tests. These contributions support review and reproduction without implying production deployment.'
])

# CHAPTER 2
add_chapter(doc,2,'Project Description and Goals','PRESTO-R is a predictive, stability-aware routing system implemented as an experimental pipeline rather than a deployable router. A seeded simulator generates a network episode, noisy link observations, and scheduled events. A risk provider estimates link degradation from telemetry. The observation builder aggregates measurements and risk into one row of path features per candidate. A policy selects one of four paths, receives reward and episode feedback, and writes step and event logs. Evaluation code transforms those logs into per-episode metrics, summaries, paired comparisons, figures, and a report. The following description separates project goals from achieved evidence and clarifies the boundary between simulation and live networking.')
add_sub(doc,'2.1 Project Overview',[
'The project begins with topology and traffic initialization. NSFNET is represented by 14 nodes and 21 undirected links; RGG uses 24 nodes generated in a unit-square random geometric graph with connection radius 0.32, subject to connectivity and minimum-degree constraints. Each edge receives a stable link identifier, capacity, and base delay. The simulator creates background traffic, event schedules, and hidden health states. One managed flow is sampled with four sufficiently diverse routes.',
'After a ten-tick warm-up, the environment exposes measurements through a fixed-size observation. The default telemetry history is W = 10 ticks and the default prediction horizon is H = 5 ticks. A link-risk provider can supply zeros, an oracle upper bound for debugging, a GRU probability, or a logistic-regression probability. The deployed-in-simulation experiment uses learned risk for predictive policies. Oracle risk is not treated as a realistic operational method.',
'The routing policy chooses a candidate-path index. The simulator advances one tick, calculates true path quality, adds configured measurement noise to observations, updates telemetry, computes reward components, and records information. The policy is evaluated over a 200-tick measured episode following warm-up. Each run preserves resolved configuration and seed information so results can be tied to the exact model and settings.'
])
add_sub(doc,'2.2 Goals and Acceptance Criteria',[
'The project goals are operationalized as measurable checks. The simulator must expose controlled scenarios with visible quality degradation; the predictor must produce calibrated risk estimates and be evaluated on onset samples; a reactive learned policy should be compared with shortest-path failover; PRESTO-R should reduce switches at comparable PDR; and a complete experiment grid must support statistical analysis and reproducible report generation.',
'The success record shows that PRESTO-R passed the comparisons against predictive DQN in mixed and fluctuation scenarios, passed the steady-state delay check, and passed the reproducibility check. It failed both specified comparisons against reactive DQN in node-failure and fluctuation scenarios, and failed the predictor onset-F1 target. These outcomes are kept visible in the report. Passing some checks does not change the failed ones or establish universal superiority.',
'A further project-level acceptance criterion is the availability of complete saved result artifacts. The test grid contains 38,400 episode metric rows, and the repository includes figures, summary tables, paired tests, robustness results, horizon and lambda sweeps, model weights, and a checksum manifest. The saved outputs support a comprehensive report, though the raw per-step logs are too granular to reproduce as thousands of pages in the main report.'
])
add_table(doc,'Project Goals And Measures',['Goal','Measure Or Evidence','Recorded Outcome'],[
('Implement a controlled network simulator','Seeded episodes, two topologies, five scenarios','Implemented; full evaluation grid saved'),
('Estimate future link risk','Onset F1, PR-AUC, calibration, lead time','GRU onset F1 0.5296; target 0.7000 not met'),
('Reduce route churn with PRESTO-R','Switches per episode and PDR','Large switch reductions versus predictive DQN in mixed scenarios'),
('Preserve delivery and recovery','PDR, delay, recovery time','PDR close to learned baselines; some reactive comparisons fail'),
('Support repeatable review','Seed separation, saved configs, hash verification','Step and event hashes match on a fresh rerun'),
], [Inches(1.6),Inches(2.2),Inches(2.2)],2,1)
add_sub(doc,'2.3 System Boundary',[
'The system boundary includes a Python fluid simulator, Gymnasium routing environment, data and predictor packages, reinforcement-learning policies, an experiment grid, statistical analysis, figure generation, and local artifact packaging. The input is a resolved configuration plus a deterministic seed stream. The outputs are episode actions, observed telemetry, event logs, metric tables, figures, and model artifacts.',
'The system does not include an IP routing protocol implementation, network device integration, live telemetry collectors, control-plane synchronization, multi-agent coordination, traffic engineering across multiple managed flows, or packet-level queue simulation. There is one managed flow per episode; other traffic is background and re-routed by the simulator. These boundaries are central when interpreting results and are repeated in the limitations chapter.',
'The optional NS-3 integration is a planned extension and is not part of the completed core workflow. No production network was accessed, and no live packets were routed by the learned agents. Therefore all evidence in this report is simulation evidence, including robustness conditions.'
])
add_sub(doc,'2.4 Actors and Operating Workflow',[
'The policy is the decision-making actor. It observes only current measured path features, the current route indicator, a recency-since-switch feature, and, where configured, predicted risk. The simulator is the environment actor. It owns hidden link health, failure schedules, and true per-link state, but exposes only noisy observations to the policy. The evaluator reads rollout logs and ground truth after decisions have been made; it does not leak future information to the learned agent.',
'The predictor is trained as a separate supervised component. It uses episode-separated telemetry sequences and labels determined from future degradation within a horizon. Validation data selects a classification threshold. During evaluation, the predictor receives a rolling telemetry window and returns a probability per link. A risk provider interface allows the environment to substitute learned risk, zero risk, or oracle risk for diagnostics.',
'The experiment runner enumerates method, topology, scenario, and seed combinations. It skips cells with existing logs, making the grid resumable. A metrics stage computes one row per episode. A statistical stage aggregates over training seeds, performs matched-seed tests, and applies Holm correction. Plot and report stages present aggregate results and illustrative traces. Each stage has a clear data contract so an evaluator can trace a result back to a run.'
])
add_sub(doc,'2.5 Assumptions',[
'The model assumes a fixed candidate set within an episode and one selected path at a time. It assumes a discrete 100 ms time step, 200 evaluation ticks after warm-up, one managed flow, and a configured set of background flows. It assumes the observation fields can be normalized to a bounded feature vector and that risk scores are finite probabilities in [0,1].',
'The experiment assumes consistent metric definitions across methods, matched held-out scenario seeds where paired comparisons are possible, and no retuning on test data. The saved reports identify across-training-seed standard deviations for summary tables and use matched test-seed episode values for paired tests. Different result tables may therefore represent different aggregation layers.',
'The simulator uses synthetic event schedules, including precursor and sudden failures. A precursor model can only help if observable telemetry changes before a failure. The sudden-failure robustness condition deliberately removes precursor warning and is therefore a test of policy behavior when the predictor has no lead signal, not a fair prediction-accuracy test.'
])
add_sub(doc,'2.6 Evidence and Deliverables',[
'The main code deliverables include modules for simulator dynamics, topology generation, traffic, failure schedules, telemetry, state construction, risk providers, reward, the Gymnasium environment, predictors, policies, training, rollout, metrics, statistics, figures, and report creation. Configuration files define base settings, topologies, scenarios, predictor variants, agent settings, and the evaluation grid.',
'The main evidence deliverables include 38,400 episode metric rows, topology-specific summary tables, paired-test tables, robustness data, horizon and lambda sweeps, predictor metrics for H5 and alternate horizons, nine result figures, a markdown report, and a local model-release archive with checksums. The report below reproduces all aggregate result rows described by these saved outputs. It does not inline every per-tick observation from the underlying 38,400 episodes.'
])

# CHAPTER 3
add_chapter(doc,3,'Technical Specification','This chapter records the configuration and interfaces that determine how a run behaves. The project stores numerical settings in YAML and resolves them into run-specific configuration files. The simulator, predictor, agent, and evaluation grid share a common configuration system and seeded random streams. The specifications below distinguish values explicitly present in the repository from values that are not recorded. In particular, no specific host CPU model, memory size, GPU, or measured project cost is inferred.')
add_sub(doc,'3.1 Network and Simulator Specification',[
'The simulator is a discrete-time fluid model. Each tick is 100 ms, each evaluation episode has 200 ticks after a warm-up of 10, and link capacity is 10.0 rate units. The NSFNET configuration samples base delays from 2 to 12 ms. The maximum queue is 20.0. Link capacity is multiplied by hidden health; a down link has zero effective capacity. Queue overflow, base loss, and degradation loss determine per-link losses, while delay includes a queue-dependent term and a timeout value for down links.',
'Background traffic is configured as 20 flows in the main mixed scenario, with log-normal rate parameters, a diurnal component, and small load noise. The managed flow rate is 2.0. It is routed over one of four paths. When links change availability, background traffic is re-routed. The model collects link load, utilization, delay, queue, loss, availability, and event records, then adds measurement noise to observable fields.',
'The simulator isolates random streams for topology, traffic, failures, and measurement noise. This prevents an extra noise draw from shifting a future failure schedule. Resolved configurations and seed identifiers are saved so a run can be reconstructed and comparisons can use aligned seeds.'
])
add_sub(doc,'3.2 Topology Specification',[
'NSFNET is a fixed undirected graph with 14 nodes and 21 links. Link identifiers are assigned from sorted canonical endpoint pairs so that the feature arrays have stable indexing. Base delays are sampled within the configured interval and capacities are assigned from the topology configuration.',
'The RGG topology contains 24 nodes placed in a unit square. A random geometric graph connects nodes within radius 0.32. A generated topology is accepted only if it is connected and every node has degree at least two. Edge capacity is 10.0, queue maximum is 20.0, and base delay is 30 ms multiplied by Euclidean edge length. This produces a second topology with a different graph structure and path-length distribution.',
'For each episode, source and destination are selected with a seeded flow generator. Candidate paths are ordered by total base delay and must be loop-free. The first two paths are required to have limited overlap, providing meaningful alternatives. Up to four paths are selected and retained for the episode. The fixed action meanings simplify learning but do not model dynamic discovery of entirely new paths.'
])
add_sub(doc,'3.3 Traffic and Failure Specification',[
'The scenario set contains steady, congestion, node_failure, fluctuation, and mixed conditions. The steady scenario schedules no congestion, link-failure, node-failure, or fluctuation events. The congestion scenario schedules an average of two congestion events per episode. The node-failure scenario uses link-failure and node-failure event rates of 0.5 and 1.0. The fluctuation scenario uses a rate of 1.5. The mixed scenario combines congestion, link failure, node failure, and fluctuation, with rates 1.0, 1.0, 0.3, and 0.5.',
'Mixed-scenario event targeting is configured to place an event on a candidate link with probability 0.7. Precursor duration ranges from 3 to 10 ticks, and 20 percent of applicable events may have no precursor. Down durations range from 15 to 60 ticks. Congestion multipliers range from 2.5 to 4.0 and durations from 10 to 40 ticks. Fluctuation periods range from 6 to 20 ticks with depth from 0.3 to 0.7. These values define simulator stress, not observed real-network frequency.',
'The sudden-failure robustness override sets no_precursor_prob to 1.0. The doubled-noise condition raises delay noise relative scale to 0.1, absolute loss noise to 0.01, and queue noise relative scale to 0.1. The 150 percent load condition increases background flow count from 20 to 30. These changes are evaluation-only; they are not used to retrain the agents or retune the selected penalty.'
])
add_table(doc,'Core Simulation Parameters',['Parameter','Configured Value','Interpretation'],[
('Tick duration','100 ms','Discrete simulator time step'),('Warm-up','10 ticks','Fills initial observation history'),('Measured episode','200 ticks','Common evaluation horizon'),('Link capacity','10.0','Configured per-link capacity'),('Base delay, NSFNET','Uniform 2-12 ms','Per-link initial delay range'),('Queue maximum','20.0','Per-link queue cap'),('Candidate paths','4','Discrete action choices per episode'),('Telemetry window W','10 ticks','Historical link measurements'),('Default risk horizon H','5 ticks','Future interval for bad-link label'),('Selected switch penalty lambda','0.5','PRESTO-R reward configuration'),('Agent training steps','300,000','Configured training budget')
], [Inches(1.7),Inches(1.55),Inches(2.85)],3,1)
add_table(doc,'Scenario Configuration',['Scenario','Event Configuration'],[
('Steady','No congestion, link failure, node failure, or fluctuation events.'),('Congestion','2.0 congestion events per episode on average.'),('Node failure','0.5 link-failure and 1.0 node-failure event rates.'),('Fluctuation','1.5 fluctuation events per episode on average.'),('Mixed','Congestion 1.0; link failure 1.0; node failure 0.3; fluctuation 0.5.')
], [Inches(1.4),Inches(4.6)],3,2)
add_sub(doc,'3.4 Telemetry and State Specification',[
'The measured fields are delay, loss, queue, utilization, and an up flag. Delay noise is relative to delay or base delay; loss noise is absolute; queue noise is scaled by maximum queue; utilization has configured noise. Hidden link health, event schedules, and future degradation labels are not placed in the policy observation. The oracle risk provider can access future schedules only as a diagnostic upper bound and is not used as an ordinary learned policy input.',
'For each candidate path, the state builder computes normalized path delay, aggregate path loss, bottleneck queue, bottleneck utilization, normalized hop count, maximum predicted risk, mean predicted risk, and validity. When risk is disabled, the risk features are omitted. The observation appends a one-hot encoding of the current path and a bounded time-since-switch feature. For four paths with risk, the observation dimension is 4 times 8 path features, plus 4 current-route indicators, plus one recency value, for 37 scalar inputs.',
'The state builder checks feature shapes, risk range, candidate-link identifiers, path index, reference delay, and maximum hops. Empty path slots are padded with zeros. Final values are clipped to [0,1] and emitted as float32. These checks protect the policy from invalid state shapes while keeping a fixed-size observation across episodes.'
])
add_sub(doc,'3.5 Predictor and Agent Specification',[
'The main GRU predictor uses five telemetry features per link, a hidden size of 32, one recurrent layer, and dropout 0.1. The default training configuration uses a window of 10, horizon 5, bad-loss threshold 0.05, bad-delay factor 3.0, batch size 512, learning rate 0.001, up to 30 epochs, patience 5, and automatic positive-class weighting. Episode-level splits are 70 percent training, 15 percent validation, and 15 percent test, with negative subsampling configured at 0.3.',
'Logistic regression uses telemetry summaries and supplies a non-sequential predictor baseline. LearnedRisk loads a GRU checkpoint, normalizes input by saved means and standard deviations, and returns a probability per link. LogRegRisk loads a saved classifier and clips scores into [0,1]. Threshold selection maximizes validation F1; held-out test data is not used for threshold selection.',
'The routing agents use Stable-Baselines3 DQN. Reactive DQN disables risk features; predictive DQN enables risk but uses no selected switch penalty; PRESTO-R enables risk with lambda = 0.5. The same general training interface is used across methods. NSFNET uses training seeds 0 through 4, and RGG uses seeds 0 through 2. The best checkpoint is selected using validation reward.'
])
add_sub(doc,'3.6 Software Stack',[
'The project pins Python packages in requirements.txt. The stack includes NumPy 1.26.4, pandas 2.2.2, SciPy 1.13.1, NetworkX 3.3, Gymnasium 1.0.0, Stable-Baselines3 2.4.0, sb3-contrib 2.4.0, PyTorch 2.4.1, scikit-learn 1.5.1, Matplotlib 3.9.2, PyYAML 6.0.2, tqdm 4.66.5, PyArrow 17.0.0, and pytest 8.3.2. The configured Python range is 3.10 through 3.12.',
'The module structure separates simulator code from the Gymnasium wrapper, predictors from routing policies, and rollouts from metrics. Configuration files reside under configs; source code under presto; scripts under scripts; tests under tests; results and figures under results; models under models; and saved training runs under runs. This separation supports small smoke executions as well as resumable large grids.',
'The project repository does not record a specific hardware inventory, total GPU hours, power draw, or labor cost. Training and full-grid evaluation runtime depend on the host, worker count, and storage. The report therefore lists observed computational volumes and software versions, while keeping unrecorded hardware and monetary amounts explicitly unavailable.'
])
add_table(doc,'Software Dependencies',['Package','Version'],[
('Python','3.10-3.12'),('NumPy','1.26.4'),('pandas','2.2.2'),('SciPy','1.13.1'),('NetworkX','3.3'),('Gymnasium','1.0.0'),('Stable-Baselines3','2.4.0'),('PyTorch','2.4.1'),('scikit-learn','1.5.1'),('Matplotlib','3.9.2'),('PyArrow','17.0.0'),('pytest','8.3.2')
], [Inches(3),Inches(3)],3,3)
add_sub(doc,'3.7 Data and Reproducibility Specification',[
'Training seeds are below 5000. Validation seeds are 5000 through 5049. Held-out test seeds are 10000 through 10199. The distinction is central: threshold and penalty selection use validation data; final test metrics are not used for tuning. Agent training seeds and scenario seeds are separate identifiers, and random streams isolate topology, traffic, failures, and noise.',
'The grid spans five scenarios and two topologies, with five learned-agent training seeds on NSFNET and three on RGG. The complete saved episode-metric table contains 38,400 rows. Step logs, event logs, resolved configurations, predictor metrics, summary tables, paired tests, robustness metrics, horizon tables, lambda sweep tables, and figures provide multiple review layers.',
'A fresh rerun produced matching hashes for saved step and event parquet outputs, according to the project success record. The model release bundle contains 88 curated files: four predictor checkpoints, 42 best agent policies, and their resolved configuration files. The local zip archive is 12.61 MB (13,227,437 bytes), with SHA-256 a0c3fcdd914f88ff2d899f4618fe9c4399ef3d3388806a067a5ea5dac4e1c398. Publishing to a remote host is not configured.'
])
add_sub(doc,'3.8 Metric Definitions',[
'PDR is total delivered traffic divided by total offered demand over the episode. The mean delay is weighted by delivered traffic; it is not the unweighted mean of each path delay. Throughput is the mean delivered rate. Switches count actions that change the selected route. The unnecessary-switch rate counts switches for which the previous path remains healthy now and through the next configured horizon H.',
'A failure event is a true up-to-down transition on a link that was present on the selected path H ticks earlier. A proactive event is counted when the policy has already left the affected link by the failure time; such an event has zero-tick recovery. Otherwise recovery time is the first subsequent valid selected path with loss at or below 0.02, capped at the last recorded tick if no such recovery is observed. Some episodes have no qualifying events, so recovery time and proactive rate may be unavailable.',
'For predictor evaluation, precision is the share of predicted positive labels that are positive; recall is the share of actual positives detected; F1 is their harmonic mean. PR-AUC measures precision-recall ranking, ROC-AUC measures ranking across thresholds, and Brier score is mean squared probability error. Median lead time reports how far in advance a positive event is signaled under the project’s definition. Each metric is reported with its evaluation subset and sample count.'
])
add_equation(doc,'PDR = sum(delivered traffic) / sum(offered demand)','3.1')
add_body(doc,'In Eqn. 3.1, delivered traffic and offered demand are summed over measured episode ticks. PDR lies between zero and one for the recorded simulator outputs.',first=False)
add_equation(doc,'F1 = 2 x precision x recall / (precision + recall)','3.2')
add_body(doc,'In Eqn. 3.2, precision and recall are computed at a threshold selected on validation data. The report emphasizes onset F1 because this is the predeclared predictor success criterion.',first=False)
add_equation(doc,'Mean Delay = sum(path delay x delivered traffic) / sum(delivered traffic)','3.3')
add_body(doc,'In Eqn. 3.3, the weighting gives greater influence to delay experienced by delivered traffic and avoids treating fully lost traffic as ordinary successful delivery.',first=False)

# CHAPTER 4
add_chapter(doc,4,'Design Approach and Details','The design separates five concerns: realistic-enough traffic and failure dynamics for controlled experiments; observation construction that hides future state; prediction from historical telemetry; route choice under an interpretable reward; and an evaluation process that separates tuning from final measurement. The system uses modular interfaces so the same simulator and rollout logic can compare heuristic, reactive, predictive, and stability-aware policies. The design does not depend on a production protocol stack and is not presented as one.')
add_sub(doc,'4.1 Design Approach Materials and Methods',[
'The main design decision is to treat link-risk prediction as an additional observation feature and route switching as an explicit reward cost. This keeps the predictor and controller separable. The predictor emits per-link probabilities; the state builder aggregates them to maximum and mean risk for each candidate path. The DQN then learns whether those risk values, together with observed path quality, justify an action. This architecture supports ablations and diagnostics because risk can be disabled, learned, or replaced with oracle estimates without rewriting route-state logic.',
'The experimental method follows a staged workflow. First, simulator and environment behavior are validated. Second, telemetry is generated and predictor checkpoints are trained. Third, baselines and route agents are trained with multiple random seeds. Fourth, model checkpoints are selected on validation reward. Finally, the held-out grid is evaluated and summarized. The split prevents final test outcomes from influencing the selected threshold or switch cost.'
])
make_architecture(TMP/'architecture.png')
add_figure(doc,TMP/'architecture.png','PRESTO-R System Architecture',4,1,'f41')
add_figure(doc,FIGS/'nsfnet_topology.png','NSFNET Topology Used In The Main Evaluation',4,2,'f42')
add_sub(doc,'Network Dynamics and Failure Model',[
'The network simulator stores one state vector per link for capacity, base delay, queue, utilization, loss, and availability. At every tick it applies a precomputed event schedule to hidden health and availability, re-routes background flows when link-up state changes, adds the managed flow load to the chosen route, and updates queues from load minus effective capacity. Queue overflow contributes to loss. Delay increases with queue relative to effective capacity. A down link receives full loss and a timeout-based delay.',
'The fluid abstraction preserves causal relationships relevant to route choice while remaining computationally tractable for reinforcement learning. Capacity, delay, congestion, degradation, and failure are all configurable. The abstraction omits packet-level burst dynamics, TCP behavior, protocol convergence, route installation delay, and interactions among multiple controlled flows. These omissions make the test grid useful for controlled comparison, but not sufficient to certify a live deployment.',
'The event generator places congestion, link failures, node failures, and fluctuations according to scenario configuration. Some events include precursor behavior; some are intentionally sudden. The simulator records true schedules for evaluation and labels, while the environment observation includes only noisy measurements. This separation makes risk prediction nontrivial and prevents a policy from reading future ground truth.'
])
add_sub(doc,'Candidate Path Selection and State Construction',[
'Candidate routes are generated using shortest-simple-path ordering by base delay. A seeded source-destination flow is sampled repeatedly until four candidate routes are available and the best two do not share more than half the links of the shorter route. This diversity condition reduces the chance that all actions are effectively the same route. The candidates remain fixed in one episode so the action index has a stable interpretation.',
'The state builder aggregates link measurements across each path. Normalized delay is bounded relative to a reference path delay. Path loss is computed from component link losses. Queue and utilization features capture bottleneck conditions. Hop count is normalized by maximum candidate hops. Risk is represented by maximum and mean link risk. A binary validity flag indicates whether all links are up. The current path is encoded as one-hot, and elapsed ticks since the last switch are clipped and normalized.',
'The risk features are useful only if they carry information about future path quality. The project therefore compares reactive and predictive models and includes a risk-permutation diagnostic in the evaluation package. If permuting risk values did not affect behavior, that would suggest the policy was not using them. This diagnostic supports interpretation but does not eliminate the predictor’s low onset-F1 limitation.'
])
add_sub(doc,'Telemetry Dataset and Label Design',[
'Predictor training examples are generated from simulated episodes across congestion, node failure, fluctuation, and mixed scenarios and across NSFNET and RGG. Episodes are split before windows are formed, preventing adjacent windows from one episode from being scattered across training and test sets. The configured proportions are 70 percent training, 15 percent validation, and 15 percent test. Negative subsampling reduces the dominant healthy class during dataset construction.',
'A link is labeled bad when future simulator truth within the selected horizon meets the configured degradation conditions, including loss and delay thresholds or down state. The onset subset focuses on samples before a link is already bad, which is closer to the early-warning task than classifying all time points. Threshold tuning maximizes F1 on validation data only. The held-out test split is then scored at that threshold.',
'The main horizon is H=5, with H=3 and H=10 studied separately. A longer horizon gives more lead time but asks the model to forecast farther ahead. That can increase ambiguity and lower onset classification quality. The report presents each horizon’s predictor metrics alongside routing outcomes so model-level and policy-level performance are not conflated.'
])
add_sub(doc,'GRU and Logistic Regression Predictors',[
'The GRU receives a time window for one link, with five telemetry features per time step. The recurrent layer summarizes sequence information in its hidden state. A linear head maps the final hidden state to a logit, and a sigmoid produces a bad-link probability. Input features are standardized using saved training means and standard deviations. Batched inference limits memory use during evaluation.',
'Logistic regression uses telemetry summaries rather than a recurrent hidden state. It offers a simpler comparison with lower temporal modeling capacity. Both models produce one probability per link. Scenario, topology, and event-type subgroups expose variation hidden by an overall score. The precursor and sudden-failure rows are particularly informative: sudden events lack predictive warning by design, while precursor events contain signals that could be learned.',
'The main report shows high all-sample scores for both predictors, but all-sample scores are not the primary criterion. GRU onset F1 is 0.5296, and logistic-regression onset F1 is 0.3949. The F1 target is 0.7000. The gap between overall and onset results shows why subgroup reporting is necessary. Calibration scores and ranking metrics provide additional context but do not substitute for the missed onset target.'
])
add_sub(doc,'DQN Policies and Reward Design',[
'The action space contains one discrete action per candidate path. DQN learns an action-value function from state and reward interactions. The reactive agent sees current path features without risk. The predictive DQN sees risk features but uses no selected penalty. PRESTO-R uses risk features and lambda=0.5. The saved trained policies are evaluated using deterministic rollouts on held-out scenarios.',
'The reward is the sum of interpretable terms. Delay is normalized relative to a reference path and capped. Loss is penalized strongly. Throughput can be weighted, though the main configured gamma is zero. A switch incurs lambda. Hop count contributes a small energy or path-length term. In the core configuration, alpha=1.0, beta=4.0, gamma=0.0, lambda=0.5 for PRESTO-R, and mu=0.05. The reward exposes each component in the environment info record, supporting diagnostics of why a policy action receives its value.',
'A switch cost introduces a direct stability preference. At a high value, it can suppress a beneficial path change. At a low value, the policy may react frequently to small fluctuations. The project sweeps five lambda values on held-out mixed scenarios after selecting the main configuration on validation. The sweep is reported as an evaluation of sensitivity, not as permission to choose a new value using test data.'
])
add_equation(doc,'R = -alpha D - beta L + gamma T - lambda S - mu E','4.1')
add_body(doc,'In Eqn. 4.1, D is normalized delay, L is path loss, T is normalized throughput, S is a binary switch indicator, and E is normalized hop count. The configured PRESTO-R coefficients are alpha=1.0, beta=4.0, gamma=0.0, lambda=0.5, and mu=0.05.',first=False)
add_sub(doc,'Rollout, Metric, and Statistical Method',[
'Each rollout records per-step policy choice, path links, observed telemetry, risk values, true quality values, demand, delivered rate, reward, and scenario seed. Event logs record scheduled event types and timings. The metrics stage groups rows by episode and computes PDR, delay, throughput, switches, unnecessary switches, recovery time, proactive rate, energy, reward, failure events, and scheduled events.',
'Summary tables first average episode metrics within a training seed, then average those seed-level means across training seeds. Standard deviations describe training-seed variation. Paired tests use matched held-out scenario seeds and Wilcoxon signed-rank comparisons. Holm correction adjusts across the global comparison family. Effect size r is signed according to the median difference. Recovery time only has pairs for episodes with relevant events, so its sample count is often below the PDR or switch sample count.',
'The summary table and paired test table answer different questions. Mean +/- SD communicates variability across trained policies. A paired test evaluates matched test-seed episode differences and yields its own median difference and adjusted p-value. The report keeps both views separate and does not infer a paired statistical conclusion from overlapping standard deviations.'
])
add_sub(doc,'Robustness and Sensitivity Design',[
'The robustness tests modify one evaluation condition while keeping the trained checkpoint. Sudden failure removes warning. Doubled measurement noise changes the observation quality. A 50 percent increase in background flows raises offered load and congestion pressure. These variants determine whether the qualitative stability effect survives changes to event warning, observation quality, or background load.',
'The horizon sweep uses H=3 and H=10 PRESTO-R variants on mixed and node-failure scenarios. It reports PDR, switches, proactive rate, recovery time, and predictor onset F1. The lambda sweep uses values 0.0, 0.05, 0.1, 0.25, and 0.5 for mixed scenarios. Together they expose trade-offs: more penalty tends to reduce switches, while longer forecast horizon can improve lead time but weaken onset classification.',
'The no-prediction plus selected-penalty condition was not included in the completed test grid. The report therefore describes the available partial ablation and explicitly avoids calling it a complete factorial experiment. This is essential to the causal interpretation of the combined PRESTO-R design.'
])
add_sub(doc,'4.2 Codes and Standards',[
'The project uses the DQN family for discrete path selection and uses GRU and logistic-regression models for risk estimation. The DQN approach is related to Mnih et al. [1]; the gated recurrent design is related to Cho et al. [2]; and the k-shortest loopless path concept is related to Yen [3]. These citations establish methodological lineage and do not imply that the project reproduces those papers’ full experiments.',
'The software dependency versions are pinned in requirements.txt. Configuration-driven values, separated seeds, resolved configs, hash manifests, and deterministic evaluation outputs support computational reproducibility. The repository includes tests for topology, telemetry, simulator behavior, state construction, reward, predictor, metrics, rollout, reporting, and experiment phases. The saved success record specifically reports a fresh-rerun hash match for step and event parquet outputs.',
'No external network protocol standard such as OSPF, IS-IS, BGP, MPLS, or an IEEE forwarding standard is implemented by the simulation. The project abstracts path selection at the candidate-route level. This boundary is stated to avoid presenting an algorithmic simulator as protocol-compliant software. The report follows the page, typography, caption, and numbering conventions shown in the supplied VIT project-report sample.'
])
add_table(doc,'Design Alternatives And Trade-Offs',['Design Choice','Alternative','Reason For Choice','Consequence'],[
('Fluid network simulation','Packet-level network simulation','Allows large RL training and held-out grids on a CPU','Less packet and protocol fidelity'),
('One managed flow','Many controlled flows','Keeps the discrete action and state tractable','No inter-agent coordination evidence'),
('Four fixed paths per episode','Dynamic route discovery every step','Stable action meanings and compact policy state','Cannot learn dynamic candidate generation'),
('GRU predictor plus logistic baseline','Single predictor only','Tests sequential model against a simpler classifier','Both learn simulator-derived patterns'),
('DQN policies','PPO or rule-only control','Discrete action set and shared training interface','No algorithm-family comparison in final grid'),
('Switch penalty lambda=0.5','No cost or alternate penalty only','Validation-screened configuration with test sweep','Full no-prediction plus penalty factorial is missing'),
], [Inches(1.2),Inches(1.25),Inches(1.65),Inches(1.9)],4,1)
add_sub(doc,'4.3 Constraints Alternatives and Tradeoffs',[
'The fluid model trades packet fidelity for throughput of experiment. It represents congestion and queue effects, but it does not model individual packets, TCP congestion control, real routing protocols, or deployment overhead. Results are therefore evidence about the defined simulator only. A packet-level follow-up should preserve event and metric semantics where possible so model rankings can be compared.',
'One managed flow provides a tractable action space but leaves multi-flow conflicts untested. Background flows create shared load, yet they are not strategic agents. A multi-flow extension would need coordination, per-flow fairness, and an expanded reward. It could change both the value of a route switch and the learned risk response.',
'Fixed candidate routes make DQN action semantics stable. A dynamic path generator could offer alternatives when a route fails, but would add state churn and variable action meaning. The fixed set can also become stale if topology changes substantially. This is a deliberate initial trade-off, not an accidental omission.',
'A GRU and logistic-regression baseline enable a sequential-versus-summary comparison. Both operate on synthetic telemetry generated by the same simulator family. Transfer to real traffic is unknown. The onset sample imbalance is an analytical limitation: high all-sample F1 does not mean reliable precursor detection. Better event-balanced training, calibrated subgroup thresholds, or domain-randomized data could be investigated, but must be evaluated on held-out scenarios.',
'The selected switch cost reduces route changes, but a penalty can delay a necessary switch. The mixed and fluctuation results show strong switching reductions versus predictive DQN. Comparisons with reactive DQN fail some predeclared checks, and the full factorial to isolate component effects is incomplete. A future experiment should include every prediction-by-penalty combination and pre-register the paired criteria before test evaluation.'
])
add_sub(doc,'4.4 Statistical Design and Validity',[
'The experimental grid uses held-out scenario seeds from 10000 to 10199. NSFNET learned methods use five training seeds, while RGG learned methods use three. Seed-level means are used to report variability across training runs. Paired tests match scenario seeds between PRESTO-R and baselines, drop unavailable paired metrics, and adjust p-values with Holm correction. These choices reduce the risk of treating repeated episodes from one trained policy as independent training replications.',
'The sample count in paired PDR and switch rows is generally 200 episodes. Recovery time counts only episodes containing the relevant events, giving smaller n values. A median difference of zero may coexist with a small p-value when a large number of matched episodes contain many tied values plus a few nonzero ranks; interpretation must retain effect size and median difference, not just p-value.',
'External validity is limited by one managed flow, synthetic event schedules, simulator-derived telemetry, a fluid traffic model, and a finite topology set. Construct validity depends on metric definitions such as the horizon used to classify unnecessary switches and the 0.02 loss quality threshold for recovery. Reproducibility evidence is positive for saved step and event hashes, but it does not establish external validity.'
])

# CHAPTER 5
add_chapter(doc,5,'Schedule, Tasks and Milestones','The project plan is organized into core phases 0 through 6 and an optional Phase 7. The phase durations below are planning estimates recorded in the implementation plan, not reconstructed calendar times. The repository indicates that the core simulator, predictors, routing agents, full experiment grid, figures, report artifacts, and model package are present. Actual start and finish dates for each phase are not logged in the material reviewed, so the report does not invent them.')
add_sub(doc,'5.1 Work Breakdown',[
'Phase 0 established the repository, environment, configuration system, and CI-style checks. Phase 1 implemented the network simulator, failure scenarios, telemetry, state, reward, and Gymnasium environment. Phase 2 created the link-degradation dataset, logistic regression and GRU models, risk interfaces, and evaluation. Phase 3 implemented heuristic baselines, deterministic rollouts, and reactive DQN. Phase 4 added predictive and PRESTO-R policies, penalty screening, ablations, and diagnostics.',
'Phase 5 built the complete experiment grid, episode metrics, statistical tests, robustness checks, horizon sweep, lambda sweep, and success criteria. Phase 6 generated figures and report artifacts, documented reproduction, and packaged models with a checksum manifest. Phase 7 proposed optional NS-3 validation after the core work; that work is not included in the completed project evidence.',
'The workflow dependencies are sequential where simulator contracts are required, while predictor and reactive-agent work can proceed in parallel after the environment is stable. Experiment grid generation depends on trained models and resolved configurations. Report generation depends on saved raw results and derived tables. The phase structure supports incremental validation and makes it possible to identify which component would need revision if a metric or policy behavior changed.'
])
add_table(doc,'Phase Schedule And Completion Status',['Phase','Planned Work','Estimated Effort','Status In Repository'],[
('0','Repository, environment, config, checks','0.5-1 day','Core workflow present'),('1','Simulator, failures, telemetry, state, reward, environment','4-5 days','Implemented'),('2','Dataset, logistic regression, GRU, risk providers','2-3 days','Implemented; saved metrics available'),('3','Heuristics, rollouts, reactive DQN','2-3 days','Implemented'),('4','PRESTO-R, lambda tuning, ablations, diagnostics','2-3 days','Implemented with partial factorial gap'),('5','Experiment grid, metrics, statistics, sweeps','2-3 days','Completed; 38,400 episode rows'),('6','Figures, report, reproduction and model release','1-2 days','Artifacts present'),('7','Optional NS-3/ns3-gym validation','3+ days','Optional; not included')
], [Inches(0.55),Inches(2.4),Inches(1.0),Inches(2.05)],5,1)
add_sub(doc,'5.2 Phase Schedule and Status',[
'The phase map lists about 14 to 20 working days for the core path, excluding the optional NS-3 phase. These values are estimated effort rather than actual elapsed time. The repository does not include a dated work log that would support a claim about calendar completion. A formal academic schedule should therefore use the planned effort as a baseline and not present it as recorded duration.',
'The setup phase is evidenced by pinned dependencies, project configuration, and a Makefile. Phase 1 is evidenced by simulator and environment modules and tests. Phase 2 is evidenced by dataset, training, evaluation, trace, model checkpoints, and predictor metric outputs. Phases 3 and 4 are evidenced by agents, policies, training runs, switch-penalty variants, and diagnostics. Phase 5 is evidenced by a full grid and derived tables. Phase 6 is evidenced by generated figures, a project report, and the local model archive.',
'Phase 7 remains optional. No NS-3 validation results are cited in this report. The absence of this phase does not mean the Python simulator was validated against packet-level output. It marks a known next step.'
])
add_sub(doc,'5.3 Milestone Evidence',[
'The simulator milestone is supported by discrete-time dynamics, event schedules, noisy observations, and episode state. The predictor milestone is supported by saved GRU and logistic regression metrics across subgroups and horizons. The routing milestone is supported by reactive, predictive, PRESTO-R, and heuristic policies. The evaluation milestone is supported by topology-specific result tables, paired tests, robustness outcomes, and success criteria.',
'The report milestone is supported by a generated markdown report, figures, and an export script. The release milestone is supported by an 88-file curated bundle and a SHA-256 manifest. The reproducibility milestone is supported by the recorded fresh-rerun hash match. These evidence items verify that artifacts exist; they do not independently verify every result from an external clean-room environment.'
])
add_sub(doc,'5.4 Reproducibility and Release',[
'The README gives commands for environment setup, smoke checks, predictor data creation and training, agent training, grid evaluation, metrics, statistics, success checks, figures, report generation, and model packaging. The experiment grid is resumable and skips completed cells. Resolved configurations accompany runs so metric values can be tied to exact parameters.',
'The model package contains predictor checkpoints, best agent policies, and resolved configuration files. The manifest records SHA-256 digests. The archive size is 12.61 MB and its exact byte size is 13,227,437. There is no configured remote release destination. Model publishing is therefore local-only in the observed workspace.'
])

# CHAPTER 6
add_chapter(doc,6,'Project Demonstration','The demonstration is a reproducible software workflow rather than a live network console. It begins with a configuration and seed, produces a simulated episode, updates telemetry and risk, selects a candidate path, and records observations and outcomes. Evaluation commands then transform saved episode logs into statistics and visualizations. The code excerpts below are selected from implemented modules and are included to make the central mechanics reviewable; they are not placeholders or pseudo-code.')
add_sub(doc,'6.1 Sample Codes',[
'The simulator and environment are separate layers. The simulator advances network dynamics and returns per-link statistics. The environment exposes a Gymnasium observation and action space, calls the risk provider, computes reward, and packages diagnostic information. This separation allows a heuristic policy and a learned policy to share the same simulated network and metrics.',
'The observation builder is intentionally fixed-size. It validates input shapes, encodes path measurements, appends optional risk and current-route indicators, and clips the resulting vector to the agent’s declared bounds. The reward function returns both a scalar training signal and a dictionary of component values, which supports analysis of delay, loss, throughput, switching, and hop contributions.',
'Metric computation occurs after rollout. It uses true schedules and event records to determine failures and recovery, while the policy only receives the noisy observations. This distinction keeps evaluation truth out of the learning interface. The following excerpt shows the reward decomposition as implemented in presto/reward.py.'
])
add_code(doc,'Source Excerpt 6.1 Reward Components - presto/reward.py',source_excerpt(ROOT/'presto/reward.py',1,44))
add_body(doc,'The returned terms are D, L, T, S, and E, corresponding to delay, loss, throughput, switching, and hop cost. The method validates positive normalization constants before calculating reward. This design makes a switch penalty inspectable in logs and allows the same reward structure to be used across agent configurations.',first=False)
add_sub(doc,'Observation Construction',[
'The state builder represents each route with normalized path delay, path loss, bottleneck queue, bottleneck utilization, hop count, optional maximum and average link risk, and validity. It pads missing routes with zeros, adds a one-hot current-route vector, and includes time since switch. Because the same feature layout is used at every step, the DQN can compare route values consistently.',
'The main implementation validates candidate paths and risk arrays before encoding. It rejects invalid link indices, bad feature dimensions, nonpositive reference delay, and out-of-range current path values. Those checks convert silent state corruption into explicit errors, which is important when the grid contains thousands of episodes.'
])
add_code(doc,'Source Excerpt 6.2 Path Feature Definitions - presto/state.py',source_excerpt(ROOT/'presto/state.py',1,25))
add_sub(doc,'Episode Evaluation',[
'Episode metrics are computed from saved step rows and event truth. A switch is unnecessary if the old path remains healthy for the current and next H ticks. Failure events are true up-to-down transitions on a path selected H ticks earlier. If the link is no longer on the selected path at failure time, the event is proactive and recovery is counted as zero ticks. Otherwise the first valid selected path at or below the QoS loss threshold determines recovery.',
'The metric implementation weights mean delay by delivered traffic, divides total delivery by total demand for PDR, and counts switches directly from the recorded action transitions. When event truth is unavailable, recorded down-link information is used as a documented fallback. The complete function is long, so the excerpt below preserves the key metric rules in source form.'
])
add_code(doc,'Source Excerpt 6.3 Metric Contract - presto/eval/metrics.py',source_excerpt(ROOT/'presto/eval/metrics.py',66,111))
add_sub(doc,'6.2 Sample Screenshots',[
'The project stores analysis visualizations as PNG and PDF images. These figures are generated from saved simulator runs and evaluation summaries. They are included here as project-output screenshots, not screenshots of a live production network. Each figure caption identifies what the plot shows and whether it is aggregate or illustrative.'
])
add_figure(doc,FIGS/'F8_learning_curves.png','Representative Training Curves',6,1,'f61')
add_body(doc,'The learning-curve figure displays training reward and switching behavior for the learned methods. It provides training context but does not replace held-out test results. Training curves can indicate whether reward changes over the training budget; final comparisons rely on the held-out evaluation grid and its seed-aware summaries.',first=False)
add_figure(doc,FIGS/'F6_event_trace.png','Held-Out Event Trace Demonstration',6,2,'f62')
add_body(doc,'The event trace is selected from the first suitable held-out mixed episode among the first 20 test seeds. It illustrates model and policy behavior around a particular event. It is not a representative average and should be read with the per-episode statistics and scenario summaries.',first=False)
add_sub(doc,'6.3 End-to-End Demonstration',[
'The documented workflow first installs the pinned requirements and runs the smoke validation. It then creates episode-separated predictor data, trains and evaluates the GRU and logistic baseline, trains the reactive, predictive, and PRESTO-R agents, and executes the experiment grid. The metrics command generates the episode parquet table; the statistics command builds paired and grouped summaries; the success checker records predeclared outcomes; plot and report commands generate the final figures and report.',
'The principal output locations are configs for parameterization, runs for trained agents and logs, data for predictor sequences, results/raw for step and event logs, results/tables for derived metrics, results/figures for plots, models for release artifacts, and results/report.md plus results/report.pdf for report outputs. The experiment grid can resume by skipping cells with saved logs.',
'A reproducibility review should use the resolved configuration associated with each run, the seed identifiers, and the exact model checkpoint. The project’s checksum manifest applies to packaged model files. It does not hash every report image or every raw episode file. The reported fresh-rerun hash match specifically applies to step and event parquet outputs noted in the success criteria.'
])

# CHAPTER 7 begins with broad evaluation and all result tables.
add_chapter(doc,7,'Cost Analysis Result and Discussion','This chapter presents every aggregate result recorded in the project report, predictor metrics, summary tables, paired-test table, horizon sweep, lambda sweep, robustness data, and predeclared success record. Mean values are kept with their reported dispersion or paired-test statistics. “n/a” indicates that the source table did not report a value or that the episode did not contain an event needed to calculate it. The report does not convert unavailable values to zero or suppress failed comparisons.')
add_sub(doc,'7.1 Cost and Resource Analysis',[
'The repository does not record monetary expenditures, hourly labor, host hardware specifications, electricity use, or cloud billing. No dollar or rupee total can be reported without inventing data. The cost analysis therefore documents measurable computational assets and the information gap for monetary costing.',
'The configured model training budget is 300,000 steps per main agent run. NSFNET learned methods use five agent-training seeds and RGG methods use three. The held-out grid contains 38,400 episode metric rows. The predictor uses up to 30 epochs, batch size 512, and early stopping patience 5. These values characterize computational scale, but do not by themselves determine runtime or hardware cost.',
'The local model archive is 12.61 MB (13,227,437 bytes), contains 88 curated files, and is accompanied by a checksum manifest. The project uses pinned open-source Python dependencies listed in Chapter 3. No remote release destination is configured. A future operational cost study would need measured training and evaluation time, worker count, storage use, hardware model, electricity rate, and personnel hours.'
])
add_table(doc,'Observed Computational Assets And Unrecorded Cost Items',['Category','Recorded Evidence','Monetary Cost'],[
('Training','300,000 configured agent steps; multiple seeds','Not recorded'),('Evaluation','38,400 episode metric rows; five scenarios; two topologies','Not recorded'),('Software','Pinned open-source Python package versions','No license payment recorded in project files'),('Model package','88 curated files; 12.61 MB archive','Not recorded'),('Hardware','Host CPU/GPU/RAM inventory absent','Not recorded'),('Labor','Phase effort estimates exist; actual hours absent','Not recorded'),('External publishing','No remote release destination configured','No publishing charge recorded')
], [Inches(1.25),Inches(3.1),Inches(1.5)],7,1)
add_sub(doc,'7.2 Evaluation Protocol',[
'The principal grid compares sp_failover, reactive_dqn, predictive_dqn, and presto_dqn across five scenarios on NSFNET and RGG. Main NSFNET learned agents have five training seeds; RGG agents have three. Each summary cell averages episode-level metrics within each training seed, then reports mean and standard deviation across seed means. The deterministic shortest-path heuristic has no between-training-seed deviation, so its standard deviation is unavailable.',
'The paired comparison table evaluates PRESTO-R against reactive and predictive DQN on matched NSFNET held-out seeds. It reports episode count, median target-minus-baseline difference, Holm-adjusted p-value, and effect-size r. The paired tests include PDR, switches, and recovery time where available. The table’s episode count for recovery may be substantially smaller because only event-bearing episodes contribute.',
'The main risk horizon is five ticks. Horizon-specific variants use three and ten ticks. Lambda screening considers zero, 0.05, 0.1, 0.25, and 0.5. Robustness cases are sudden failure, twice the measurement noise, and 150 percent background load. Robustness uses existing policies and evaluation overrides rather than retraining on altered test conditions.'
])

# Figures for results and predictor subgroup tables.
add_sub(doc,'7.3 Predictor Results',[
'The main held-out predictor results distinguish all-sample classification from the onset subset. The onset subset is central because it excludes links already bad at the prediction time. The all-sample GRU F1 of 0.9769 is high, but onset F1 is 0.5296. The validation-selected onset threshold is 0.816824. The project’s target is 0.7000, so the primary predictor criterion fails.',
'Scenario subgroups show stronger GRU onset performance in fluctuation than in node failure, while topology subgroups show different precision-recall trade-offs. Event-type subgroups show high precursor PR-AUC but lower Brier calibration and smaller sample sizes; the sudden-failure group cannot be interpreted as predictive early warning because by design it contains no precursor. Logistic regression has lower overall onset F1 and lower onset PR-AUC than the GRU in the main H5 test metrics.',
'The full metric tables include threshold, precision, recall, F1, PR-AUC, ROC-AUC, Brier score, lead time where calculated, and sample count. H3 and H10 predictor subgroups are also included. This prevents the report from hiding subgroup behavior behind one average.'
])
add_figure(doc,FIGS/'F2_predictor_quality.png','Held-Out Predictor Quality',7,1,'f71')
add_figure(doc,FIGS/'F3_risk_trace.png','Learned And Oracle Link-Risk Trace',7,2,'f72')
metrics_main=pd.read_csv(ROOT/'results/predictor/metrics.csv').fillna('n/a')
table_idx=2
for model in ['gru','logreg']:
    df=metrics_main[metrics_main.model==model]
    rows=[]
    for _,r in df.iterrows(): rows.append([r['subset'],r['group'],fmt(r['threshold']),fmt(r['precision']),fmt(r['recall']),fmt(r['f1']),fmt(r['samples'])])
    add_metric_table(doc,f'{model.upper()} Test Classification And Sample Counts',['Subset','Group','Threshold','Precision','Recall','F1','Samples'],rows,7,table_idx); table_idx+=1
    rows=[]
    for _,r in df.iterrows(): rows.append([r['subset'],r['group'],fmt(r['pr_auc']),fmt(r['roc_auc']),fmt(r['brier']),fmt(r['median_lead_time']),fmt(r['samples'])])
    add_metric_table(doc,f'{model.upper()} Test Ranking Calibration And Lead Time',['Subset','Group','PR-AUC','ROC-AUC','Brier','Median Lead','Samples'],rows,7,table_idx); table_idx+=1
for horizon in [3,10]:
    df=pd.read_csv(ROOT/f'results/predictor/horizon_H{horizon}/metrics.csv').fillna('n/a')
    rows=[]
    for _,r in df.iterrows(): rows.append([r['subset'],r['group'],fmt(r['threshold']),fmt(r['precision']),fmt(r['recall']),fmt(r['f1']),fmt(r['samples'])])
    add_metric_table(doc,f'GRU H={horizon} Classification And Sample Counts',['Subset','Group','Threshold','Precision','Recall','F1','Samples'],rows,7,table_idx); table_idx+=1
    rows=[]
    for _,r in df.iterrows(): rows.append([r['subset'],r['group'],fmt(r['pr_auc']),fmt(r['roc_auc']),fmt(r['brier']),fmt(r['median_lead_time']),fmt(r['samples'])])
    add_metric_table(doc,f'GRU H={horizon} Ranking Calibration And Lead Time',['Subset','Group','PR-AUC','ROC-AUC','Brier','Median Lead','Samples'],rows,7,table_idx); table_idx+=1

add_sub(doc,'7.4 NSFNET Results',[
'The NSFNET summary includes 20 method-scenario rows. Across all five scenarios, PRESTO-R makes fewer switches per episode than both learned comparators. In mixed episodes, PRESTO-R averages 5.197 switches versus 20.988 for predictive DQN and 18.799 for reactive DQN, while PDR is 0.9625 versus 0.9608 and 0.9619. The 75.2 percent switch reduction relative to predictive DQN is consistent with the intended stability effect.',
'PDR remains high for learned policies across scenarios, but the improvements over reactive DQN are not sufficient to claim every planned criterion passes. In node failure, PRESTO-R PDR is 0.9667 versus 0.9662 for reactive DQN; the success record nevertheless marks the comparison as failed because the paired criterion also considers recovery and corrected statistical results. In fluctuation, PRESTO-R PDR is 0.9749 versus 0.9740, but recovery comparison and multiplicity-adjusted tests again prevent a pass.',
'The shortest-path failover policy offers a clear non-learning reference, with substantially lower PDR and much higher delay. Its switch count is near zero because the heuristic does not behave like the learned route selectors. A low switch count alone is not a good outcome when delivery is poor. Therefore the main interpretation considers PDR, delay, recovery, switching, proactive response, and unnecessary switching together.'
])
add_figure(doc,FIGS/'F4_nsfnet_results.png','NSFNET Main Results',7,3,'f73')
T3=md_table(md,'T3.')
ns_rows=[r for r in T3[1:] if len(r)>=8]
for group,cols in [('Delivery Delay And Recovery',[0,1,2,3,4]),('Switching Proactive Response And Unnecessary Switches',[0,1,5,6,7])]:
    headers=[T3[0][i].replace('_',' ').title() for i in cols]
    rows=[[r[i] for i in cols] for r in ns_rows]
    add_metric_table(doc,f'NSFNET {group} (Mean +/- Standard Deviation Across Training Seeds)',headers,rows,7,table_idx); table_idx+=1

# Per-scenario detailed summaries make the gap and outcomes explicit.
nsf_scenarios={
'Steady':[
'In steady NSFNET episodes the shortest-path baseline has PDR 0.8966 and mean delay 119.9542. Reactive DQN reaches PDR 0.9797 with delay 46.3062 and 20.007 switches. Predictive DQN reaches PDR 0.9786, delay 46.7965, and 23.024 switches. PRESTO-R reaches PDR 0.9804, delay 46.2766, and 5.212 switches. Thus the switch penalty sharply reduces route changes while PDR and delay remain near the learned baselines.',
'No recovery time or proactive rate is available for the learned steady episodes because there are no failure events that qualify for those measures. The heuristic has zero switches in steady state, but its delivery and delay are substantially worse than the learned policies. This demonstrates why route stability cannot be judged without delivery quality.'
],
'Congestion':[
'Under NSFNET congestion, the heuristic PDR is 0.8747 and delay is 128.1184. Reactive DQN records PDR 0.9721, delay 52.6087, and 19.913 switches. Predictive DQN records PDR 0.9708, delay 53.0682, and 23.064 switches. PRESTO-R records PDR 0.9729, delay 52.4105, and 5.478 switches. Relative to predictive DQN, PRESTO-R has slightly higher PDR and lower delay while making fewer switches.',
'The paired tests show a statistically significant but numerically very small PDR median difference against predictive DQN, and a substantial switch-count difference against both predictive and reactive baselines. The key interpretation is the magnitude distinction: a p-value can be small because 200 matched episodes are available even when the PDR median difference is tiny. Route-switch reductions are much larger in absolute units.'
],
'Node Failure':[
'In NSFNET node-failure episodes, PDR is 0.9662 for reactive DQN, 0.9652 for predictive DQN, and 0.9667 for PRESTO-R. Mean delays are 52.6682, 53.4841, and 52.5669 respectively. Mean recovery times in the training-seed summaries are 8.8951, 7.6455, and 9.4236. PRESTO-R switches 5.182 times, versus 19.149 for reactive DQN and 21.457 for predictive DQN.',
'The result shows the central trade-off. PRESTO-R reduces switching and maintains PDR near the reactive policy, but does not deliver faster average recovery in the summary table. The success criteria mark the comparison against reactive DQN as failed. The predictor’s onset subgroup for node failure also has lower F1 than the fluctuation subgroup. Failure management must therefore be judged by recovery and detection, not by switch reduction alone.'
],
'Fluctuation':[
'In NSFNET fluctuation episodes, PDR is 0.9740 for reactive DQN, 0.9726 for predictive DQN, and 0.9749 for PRESTO-R. Mean delay is 49.7406, 50.7261, and 49.8912. PRESTO-R switches 5.557 times per episode, compared with 20.176 for reactive DQN and 23.419 for predictive DQN. PRESTO-R has fewer switches and slightly higher PDR than either learned comparator, while its recovery-time summary is 7.9852 versus 7.4906 for reactive DQN.',
'The project success record marks PRESTO-R versus predictive DQN as passed on the switch/PDR criterion and marks PRESTO-R versus reactive DQN as failed under the predeclared comparison. Paired tests report corrected PDR p-values and a zero median difference against reactive DQN despite the summary PDR mean difference. These statistics are not interchangeable; both are reported.'
],
'Mixed':[
'Mixed NSFNET episodes combine congestion, link failure, node failure, and fluctuation. The summary PDR is 0.9619 for reactive DQN, 0.9608 for predictive DQN, and 0.9625 for PRESTO-R. Mean delays are 56.6613, 57.1976, and 56.3287. Recovery times are 7.7741, 8.0037, and 8.8419. Switch counts are 18.799, 20.988, and 5.197. PRESTO-R therefore reduces switches by 75.2 percent versus predictive DQN, while its PDR is higher by 0.0017.',
'The proactive rate is 0.3901 for PRESTO-R, compared with 0.4572 for reactive DQN and 0.5108 for predictive DQN. Unnecessary-switch rate is 0.2344 for PRESTO-R, lower than 0.3444 and 0.4128 for the two comparators. These measures suggest the penalty curbs unnecessary route movement but also changes the policy’s proactive behavior. The missing factorial cell prevents isolating how much of this behavior is caused by risk features versus the penalty.'
]
}
for title,paras in nsf_scenarios.items():
    add_sub(doc,f'NSFNET Scenario Analysis: {title}',paras)

add_sub(doc,'7.5 Paired Tests',[
'Paired comparisons use common held-out scenario seeds and report the median difference PRESTO-R minus baseline. Negative switch differences favor PRESTO-R because fewer switches are desired. Positive PDR differences favor PRESTO-R. Holm-adjusted p-values control the family-wise error rate across the specified comparisons. Effect size r provides direction and magnitude information under the saved analysis convention.',
'Across scenarios, switch comparisons against predictive and reactive DQN are strongly significant with negative effects, reflecting fewer route changes under PRESTO-R. PDR comparisons generally have small median differences, even where adjusted p-values are small. Recovery-time tests use only failure episodes, so n varies. Several recovery comparisons have zero median difference and nonsignificant p-values; the node-failure comparison with predictive DQN has a corrected p of 0.0493 and an effect size of zero in the saved table, so it should not be described as a large recovery effect.',
'All paired-test rows from the saved NSFNET table are reproduced below, including zero differences and nonsignificant results. This avoids selecting only favorable tests. The summary tables remain separately available for across-training-seed means and standard deviations.'
])
T4=md_table(md,'T4.')
pairrows=T4[1:]
add_metric_table(doc,'Paired Wilcoxon Results With Holm Correction',['Scenario','Metric','Baseline','Episodes','Median Difference','p Holm'],[[r[0],r[1],r[2],r[3],r[4],r[5]] for r in pairrows],7,table_idx); table_idx+=1
add_metric_table(doc,'Paired-Test Signed Effect Sizes',['Scenario','Metric','Baseline','Effect Size r'],[[r[0],r[1],r[2],r[6]] for r in pairrows],7,table_idx); table_idx+=1

add_sub(doc,'7.6 Ablation and Penalty Sweep',[
'The available mixed-scenario ablation cells compare no prediction/no penalty, prediction/no penalty, and prediction/selected penalty. Reactive DQN has PDR 0.9619 and 18.799 switches. The zero-penalty predictive variant has PDR 0.9628 and 17.812 switches. The selected-penalty PRESTO-R cell has PDR 0.9625 and 5.197 switches. The no-prediction/selected-penalty cell was not run in the completed P0 grid.',
'The lambda sweep shows average mixed-scenario PDR staying within a narrow band as switch rate falls from 17.8125 at lambda 0.0 to 5.3625 at lambda 0.5. Unnecessary-switch rate decreases through lambda 0.25 and rises slightly at 0.5. Recovery and proactive metrics vary substantially across episodes and their standard deviations are large. The sweep illustrates a stability-delivery trade-off but does not identify a universal optimum.',
'Because the no-prediction plus penalty condition is absent, the effect of prediction cannot be separated from the effect of the switch penalty in a complete factorial analysis. The report treats the partial cells descriptively and avoids a causal decomposition.'
])
add_figure(doc,FIGS/'F5_lambda_tradeoff.png','Mixed-Scenario Switch-Penalty Trade-Off',7,4,'f74')
T5=md_table(md,'T5.')
add_metric_table(doc,'Available Ablation Cells',['Ablation','Method','PDR Mean','Switches Mean'],[r[:4] for r in T5[1:]],7,table_idx); table_idx+=1
lam=md_table(md,'Mixed-scenario')
lh=lam[0]; lrows=lam[1:]
for title, inds in [
('Lambda Sweep PDR And Switches',[0,1,2,3,4,5]),
('Lambda Sweep Unnecessary Switching And Recovery',[0,1,6,7,8,9]),
('Lambda Sweep Proactive Rate',[0,1,10,11]),
]:
    heads=[lh[i].replace('_',' ').title() for i in inds]
    add_metric_table(doc,title,heads,[[r[i] for i in inds] for r in lrows],7,table_idx); table_idx+=1

add_sub(doc,'7.7 Horizon Sensitivity',[
'The H=3 mixed result has PDR 0.9613, 6.8733 switches, proactive rate 0.3463, recovery time 9.7458, and onset F1 0.5992. H=10 mixed has PDR 0.9616, 5.5583 switches, proactive rate 0.3912, recovery time 8.8698, and onset F1 0.4734. On node failure, H=3 has PDR 0.9661 and H=10 has 0.9658; switch means are 6.6750 and 5.8983, with F1 of 0.5992 and 0.4734.',
'The longer horizon yields fewer switches in both scenarios and similar PDR, but onset F1 is lower. This result illustrates why horizon selection cannot rely only on routing stability. A model that forecasts farther ahead may produce behavior that changes route decisions, yet has a weaker onset classifier. Recovery means and proactive rates also vary by scenario and include their reported across-seed standard deviations.'
])
add_figure(doc,FIGS/'F7_horizon_sweep.png','Prediction-Horizon Sensitivity',7,5,'f76')
HORIZ=md_table(md,'Horizon-sweep')
for title,inds in [('Horizon Sweep PDR And Switching',[0,1,2,3,4,5]),('Horizon Sweep Proactivity Recovery And Predictor F1',[0,1,6,7,8,9,10])]:
    add_metric_table(doc,title,[HORIZ[0][i].replace('_',' ').title() for i in inds],[[r[i] for i in inds] for r in HORIZ[1:]],7,table_idx); table_idx+=1

add_sub(doc,'7.8 RGG Results',[
'The RGG topology is a second structural check with 24 nodes and a random geometric link layout. Main learned policies use three training seeds. In mixed episodes, predictive DQN reaches PDR 0.9546, delay 38.4670, recovery time 17.1224, and 11.873 switches. PRESTO-R reaches PDR 0.9545, delay 38.1368, recovery 16.3996, and 1.913 switches. The switch reduction is large while PDR differs by only 0.0001.',
'In steady episodes, PRESTO-R has PDR 0.9669 and 1.0383 switches, while reactive DQN has 0.9665 and 12.4383 switches. In congestion, PRESTO-R records PDR 0.9627, delay 35.5245, and 1.3033 switches. Node failure and mixed scenarios show longer recoveries on RGG than NSFNET, reflecting different topology and event exposure. The topology-specific values should not be compared as if only the policy changed; network structure also changes.',
'Across all five RGG scenarios, PRESTO-R reduces route switches relative to both learned baselines. PDR is close to the predictive policy. The RGG results strengthen evidence that the stability effect is not limited to one topology, while the small number of training seeds and simulated setting still limit generalization.'
])
add_figure(doc,FIGS/'F9_rgg_results.png','RGG Main Results',7,6,'f78')
T_RG=[r for r in md_table(md,'RGG test')[1:] if len(r)>=8]
for title,cols in [('RGG Delivery Delay And Recovery',[0,1,2,3,4]),('RGG Switching Proactivity And Unnecessary Switching',[0,1,5,6,7])]:
    add_metric_table(doc,title,[md_table(md,'RGG test')[0][i].replace('_',' ').title() for i in cols],[[r[i] for i in cols] for r in T_RG],7,table_idx); table_idx+=1
rgg_scenarios={
'Steady':[
'RGG steady episodes show PDR 0.9665 for reactive DQN, 0.9664 for predictive DQN, and 0.9669 for PRESTO-R. Their delays are 34.6919, 35.2924, and 34.4730. Switches are 12.4383, 10.9417, and 1.0383. The shortest-path policy has PDR 0.9242 and delay 68.3566.',
'The result supports lower route churn with comparable delivery in the steady RGG setting. Recovery and proactive rate are unavailable where no qualifying event is recorded. The deterministic heuristic provides a lower-performing route-quality baseline.'
],
'Congestion':[
'Under RGG congestion, PDR is 0.9625 for reactive DQN, 0.9625 for predictive DQN, and 0.9627 for PRESTO-R. Delays are 35.6699, 36.3546, and 35.5245. PRESTO-R switches 1.3033 times, compared with 13.1983 for reactive and 11.0900 for predictive. The shortest-path baseline PDR is 0.9170 and delay 70.6413.',
'The learned policies preserve similar delivery, while PRESTO-R has the lowest switch count. Unnecessary-switch rates are 0.3733 for PRESTO-R, 0.6268 for reactive, and 0.6616 for predictive. The reported across-seed standard deviations are retained in the full table.'
],
'Node Failure':[
'RGG node-failure PDR is 0.9505 for reactive DQN, 0.9504 for predictive DQN, and 0.9508 for PRESTO-R. Delays are 36.8101, 37.2256, and 36.5316. Recovery times are 17.8846, 19.0515, and 18.5616. Switches are 12.5583, 10.8100, and 1.6250.',
'PRESTO-R retains similar PDR and lowers switching, but recovery is not faster than reactive DQN. Its unnecessary-switch rate is 0.2928, against 0.6007 reactive and 0.6129 predictive. This pattern again supports a stability effect while showing that recovery outcomes depend on scenario and topology.'
],
'Fluctuation':[
'RGG fluctuation PDR is 0.9643 for reactive DQN, 0.9647 for predictive DQN, and 0.9647 for PRESTO-R. Delays are 35.6269, 36.0663, and 35.5024. Switch means are 13.0617, 11.7017, and 1.3267. Recovery-time means are 9.2333, 7.7180, and 9.6881.',
'The route-switch reduction is pronounced, but the recovery summary for PRESTO-R is higher than both comparators. Proactive rate is 0.0859 for PRESTO-R, 0.0667 for reactive DQN, and 0.2968 for predictive DQN. These values argue against describing lower switching as a universal improvement in all dimensions.'
],
'Mixed':[
'RGG mixed PDR is 0.9542 for reactive DQN, 0.9546 for predictive DQN, and 0.9545 for PRESTO-R. Delays are 37.9024, 38.4670, and 38.1368. Recovery times are 15.8897, 17.1224, and 16.3996. Switch counts are 12.7883, 11.8733, and 1.9133.',
'PRESTO-R has a large switching reduction versus predictive DQN and PDR lower by 0.0001. Its unnecessary-switch rate is 0.2932, compared with 0.5656 reactive and 0.6037 predictive. The mixed RGG result aligns with the NSFNET stability pattern, while its recovery time remains longer than reactive DQN.'
]
}
for title,paras in rgg_scenarios.items(): add_sub(doc,f'RGG Scenario Analysis: {title}',paras)

add_sub(doc,'7.9 Robustness Evaluation',[
'The sudden-failure case removes precursor warning. Reactive DQN PDR is 0.9619, delay 56.6613, switches 18.799, recovery 7.7741, proactive rate 0.4572, and unnecessary-switch rate 0.3444. PRESTO-R PDR is 0.9625, delay 56.3287, switches 5.197, recovery 8.8419, proactive rate 0.3901, and unnecessary-switch rate 0.2344. Thus route stability improves, but recovery is slower and proactive rate lower.',
'Under twice the measurement noise, reactive DQN records PDR 0.9618, delay 56.5514, switches 19.916, recovery 8.0025, proactive rate 0.4418, and unnecessary-switch rate 0.3656. PRESTO-R records PDR 0.9626, delay 56.3226, switches 5.582, recovery 9.0004, proactive rate 0.3994, and unnecessary-switch rate 0.2762. Mean delivery and delay remain close, with fewer switches under PRESTO-R.',
'At 150 percent background load, congestion reduces PDR to 0.8728 for reactive DQN and 0.8736 for PRESTO-R. Delay rises to 133.9481 and 130.1015. Switches are 18.846 and 4.985. Recovery times are 16.1781 and 17.0380, while proactive rates are 0.4341 and 0.3972. The stability effect persists under load, but the absolute PDR decrease emphasizes the stress of the modified traffic condition.',
'The complete robustness CSV contains mean, standard deviation, and training seed count for PDR, delay, switches, unnecessary-switch rate, recovery time, and proactive rate for each method and case. The table below reproduces all saved rows rather than only a selected subset.'
])
rob=pd.read_csv(ROOT/'results/tables/robustness.csv').fillna('n/a')
rob_rows=[]
for _,r in rob.iterrows():
    rob_rows.append([r['case'],r['method'],r['metric'],fmt(r['mean_over_training_seeds']),fmt(r['std_over_training_seeds']),fmt(r['training_seeds'])])
add_metric_table(doc,'Evaluation-Only Robustness Results',['Case','Method','Metric','Mean','SD','Seeds'],rob_rows,7,table_idx); table_idx+=1

add_sub(doc,'7.10 Success Criteria',[
'The predeclared criteria distinguish performance against reactive and predictive DQN, steady-state delay, predictor onset F1, and reproducibility. PRESTO-R passes the specified predictive comparisons for mixed and fluctuation scenarios on switch reduction with comparable PDR. It passes the steady delay limit and reproducibility criterion. It fails the comparison against reactive DQN in node failure and fluctuation and fails the onset-F1 threshold.',
'The failed node-failure criterion records PDR 0.9667 versus 0.9662 and recovery 9.440 versus 8.915, with Holm p-values of 1 for PDR and 0.2617 for recovery. The fluctuation criterion records PDR 0.9749 versus 0.9740 and recovery 7.844 versus 7.386, with Holm p-values 0.01042 for PDR and 1 for recovery. The predictor criterion records GRU validation onset F1 0.5296 against 0.7000. These are reported as failures even though some point estimates favor PRESTO-R.',
'The source success sheet and the T3 summary table contain slightly different recovery aggregates. T3 reports PRESTO-R and reactive means of 9.4236 and 8.8951 for node failure, and 7.9852 and 7.4906 for fluctuation. The success sheet reports 9.440 and 8.915, and 7.844 and 7.386. These values are preserved in their source contexts. They reflect different summary outputs and should be reconciled against raw episode records before any submission that requires one single recovery estimate.'
])
success_rows=[
('PRESTO-R vs reactive in node_failure','FAIL','PDR 0.9667 vs 0.9662; recovery 9.440 vs 8.915; Holm p PDR=1, recovery=0.2617'),
('PRESTO-R vs reactive in fluctuation','FAIL','PDR 0.9749 vs 0.9740; recovery 7.844 vs 7.386; Holm p PDR=0.01042, recovery=1'),
('PRESTO-R vs predictive in mixed','PASS','Switches 5.197 vs 20.988; PDR 0.9625 vs 0.9608'),
('PRESTO-R vs predictive in fluctuation','PASS','Switches 5.557 vs 23.419; PDR 0.9749 vs 0.9726'),
('Steady-state delay','PASS','PRESTO-R delay 46.2766; limit 50.9369 (reactive x 1.10)'),
('Predictor onset F1','FAIL','GRU validation onset F1=0.5296; threshold=0.7000'),
('Reproducibility','PASS','Step and event parquet hashes match on a fresh rerun')]
add_table(doc,'Predeclared Success Criteria',['Check','Result','Details'],success_rows,[Inches(1.8),Inches(0.65),Inches(3.35)],7,table_idx); table_idx+=1
add_table(doc,'Success-Criteria And Summary Aggregation Difference',['Scenario','Source','PRESTO-R Recovery','Reactive Recovery','Reporting Note'],[
('Node failure','Success criteria','9.440','8.915','Paired success summary values'),('Node failure','T3 mean over seed summaries','9.4236','8.8951','Across-training-seed summary table'),('Fluctuation','Success criteria','7.844','7.386','Paired success summary values'),('Fluctuation','T3 mean over seed summaries','7.9852','7.4906','Across-training-seed summary table')
], [Inches(1.1),Inches(1.5),Inches(1.1),Inches(1.1),Inches(1.0)],7,table_idx); table_idx+=1

add_sub(doc,'7.11 Findings by Scenario',[
'Across steady conditions, the switch penalty reduces route changes while preserving or slightly improving mean delivery and delay. There are no qualifying event-based recovery measures in steady runs. Under congestion, PRESTO-R retains similar PDR, lowers delay slightly, and makes substantially fewer switches. These findings are consistent on both topologies, though absolute delays and switch baselines differ.',
'Under node failure, route-switch counts remain much lower, but recovery is not consistently improved. The project’s predeclared comparison against reactive DQN fails. In fluctuation, PRESTO-R shows a strong switch reduction and close PDR, but the success criterion against reactive DQN still fails under its paired rule. This is an example of why a summary mean alone cannot override a planned inferential criterion.',
'Mixed scenarios combine the strongest evidence for stability with useful limitations. On NSFNET, switches fall from 20.988 to 5.197 relative to predictive DQN while PDR rises by 0.0017. On RGG, switches fall from 11.873 to 1.913 while PDR changes by -0.0001. Recovery is higher for PRESTO-R than reactive DQN in the mixed summaries, so route stability is not equivalent to faster recovery.',
'Robustness tests show that reduced switching persists under sudden failures, doubled noise, and increased background load. They also show lower proactive rates or slower recovery in some conditions, and a large PDR decline under the 150 percent load condition for both policies. These checks strengthen the stability observation but do not erase the predictor target failure or prove external generalization.'
])
add_sub(doc,'7.12 Threats to Validity',[
'Construct validity is limited by the metric definitions and synthetic event process. Unnecessary switching depends on an H-tick lookahead, proactive behavior depends on failure truth, and recovery is based on a loss threshold of 0.02. Different operational definitions could change metric values. Predictor labels depend on simulator thresholds for bad loss and bad delay.',
'Internal validity depends on matched seeds, resolved configurations, and consistent model checkpoints. The project records separated training, validation, and test seed ranges and reports a fresh-rerun hash match. However, table-level recovery summaries differ across success criteria and T3. This discrepancy is explicitly retained and should be reconciled from the raw metric files if a single value is required.',
'External validity is limited to one fluid simulator, one managed flow, synthetic failures, two topologies, and the selected seed ranges. The results do not establish production performance, protocol convergence, or transfer to networks with different traffic and failures. NS-3 validation is optional and absent.',
'Statistical validity is limited by the number of training seeds, subgroup sizes, available event episodes, multiple comparisons, and incomplete factorial ablation. The Holm adjustment reduces family-wise error across the specified comparison family, but it cannot correct for every design omission. The missing no-prediction plus penalty cell prevents a full causal decomposition.'
])

# CHAPTER 8
add_chapter(doc,8,'Summary','PRESTO-R implements a predictive routing design in which GRU-based link risk augments path observations and DQN selects a route under a switching penalty. The evaluation shows that the stability objective was achieved in the tested simulator: PRESTO-R substantially reduces route switching relative to predictive DQN on mixed episodes for both NSFNET and RGG. Delivery remains close to learned baselines. The evidence is not uniformly positive. The GRU onset-F1 target is missed, comparisons against reactive DQN fail in two predeclared conditions, and the complete prediction-by-penalty factorial is missing.')
add_sub(doc,'8.1 Objective Achievement',[
'The simulator objective is achieved through a seeded discrete-time fluid model, two topologies, five scenarios, hidden link health, noisy measurements, event schedules, and a shared routing environment. The predictor objective is partially achieved: GRU and logistic-regression models are trained, evaluated, and integrated as risk providers, but the primary onset-F1 target is not reached.',
'The routing objective is achieved with reactive, predictive, and stability-aware DQN policies plus a shortest-path failover baseline. The main mixed-scenario stability comparison shows fewer route changes for PRESTO-R and similar PDR. The full result tables, paired tests, sensitivity analyses, and robustness results are saved and reported. The missing factorial ablation limits causal attribution of the combined design.',
'The reproducibility and reporting objective is supported by resolved configurations, separate seed ranges, a resumable evaluation grid, a 38,400-row episode-metric table, figures, checksums, and a recorded hash match for fresh step and event outputs. The report preserves failures and uncertainty instead of presenting only favorable evidence.'
])
add_sub(doc,'8.2 Conclusions',[
'In the tested simulator, adding the selected switching cost to the risk-aware DQN substantially reduces route churn. On NSFNET mixed episodes, PRESTO-R averages 5.197 switches compared with 20.988 for predictive DQN, with PDR 0.9625 compared with 0.9608. On RGG mixed episodes, PRESTO-R averages 1.913 switches compared with 11.873, with PDR 0.9545 compared with 0.9546.',
'This conclusion is deliberately bounded. PRESTO-R is not uniformly superior to reactive DQN on every predeclared criterion. The onset predictor fails its F1 target. Some recovery outcomes are not improved, and the full causal ablation is unavailable. The correct conclusion is a demonstrated route-stability benefit under specific simulated conditions, coupled with unresolved prediction and recovery limitations.'
])
add_sub(doc,'8.3 Recommendations',[
'The next experiment should complete the two-by-two comparison of prediction presence and switch-penalty presence, keeping training budget and other hyperparameters matched. It should preregister the policy-comparison criteria, specify which aggregation supports each criterion, and report all event-bearing recovery sample counts. This would clarify whether the combined benefit comes from predictor features, switching cost, or their interaction.',
'The predictor should be improved and re-evaluated on onset samples, with calibration and subgroup metrics retained. Sudden events should remain a separate case because no precursor can be learned when there is no warning signal. Scenario and topology splits should remain isolated to test transfer rather than only within-distribution performance.',
'A packet-level or real-trace evaluation should follow after the simulator findings are stable. A multi-flow setting should measure fairness and interference. Any operational recommendation should require instrumentation, safety fallback, route-change validation, and an offline shadow evaluation before live control.'
])
add_sub(doc,'8.4 Future Work',[
'Planned extensions include multiple managed flows, multipath splitting, packet-level or NS-3 validation, online predictor updates, domain randomization, broader topology sets, and more robust uncertainty estimation. The optional NS-3 phase is not completed in the current project. It should be reported separately after a simulator-to-packet comparison is available.',
'Future reporting should reconcile the recovery values between the success criteria summary and T3 summaries, preferably from a single scripted aggregation pipeline. It should also archive the exact report-generation version, summary CSV hashes, and predictor metric files with the final document. These steps would make all table values traceable to a unique saved computation.'
])

# CHAPTER 9 references
add_chapter(doc,9,'References','The references below support the algorithmic context and identify the project artifacts from which the system details and measured results were taken. Academic references are not presented as evidence that the project reproduces their experiments. Repository outputs are cited as the source of the numerical results.')
refs=[
'[1] V. Mnih et al., “Human-level control through deep reinforcement learning,” Nature, vol. 518, pp. 529-533, 2015. doi:10.1038/nature14236.',
'[2] K. Cho, B. van Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, “Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation,” Proceedings of EMNLP, 2014. doi:10.48550/arXiv.1406.1078.',
'[3] J. Y. Yen, “Finding the K Shortest Loopless Paths in a Network,” Management Science, vol. 17, no. 11, pp. 712-716, 1971. doi:10.1287/mnsc.17.11.712.',
'[4] PRESTO-R source repository, simulator, predictor, agent, policy, evaluation, and test modules, Computer Networks project workspace, accessed October 2026.',
'[5] PRESTO-R configuration files under configs/, including base, topology, scenario, predictor, agent, and evaluation-grid settings, Computer Networks project workspace, accessed October 2026.',
'[6] PRESTO-R saved evaluation tables under results/tables/ and results/predictor/, including topology summaries, paired tests, robustness, horizon, lambda, and predictor metrics, Computer Networks project workspace, accessed October 2026.',
'[7] PRESTO-R generated report and success criteria, results/report.md and results/success_criteria.md, Computer Networks project workspace, accessed October 2026.',
'[8] NumPy 1.26.4; pandas 2.2.2; SciPy 1.13.1; NetworkX 3.3; Gymnasium 1.0.0; Stable-Baselines3 2.4.0; PyTorch 2.4.1; scikit-learn 1.5.1; Matplotlib 3.9.2; and PyArrow 17.0.0, versions recorded in requirements.txt.',
'[9] Vellore Institute of Technology, School of Information Technology and Engineering, BCSE308P Computer Networks Course Project Evaluation Sheet, Fall Semester 2026-2027, supplied course material.',
'[10] VIT course project report sample, “BCSE308P Course Project,” supplied in review docs, used for report structure, typography, margins, page numbering, and caption rules.'
]
for i,ref in enumerate(refs):
    p=doc.add_paragraph(); fmt_para(p,WD_ALIGN_PARAGRAPH.JUSTIFY,first_indent=False,after=8,line=1.5)
    p.paragraph_format.left_indent=Inches(0.3); p.paragraph_format.first_line_indent=Inches(-0.3)
    set_run_font(p.add_run(ref),12)

# Appendix follows references and intentionally has no page numbers.
sec4=doc.add_section(WD_SECTION.NEW_PAGE); configure_section(sec4,roman=False,start=1,footer=False)
add_page_heading(doc,'APPENDIX A REPRODUCTION AND DATA DICTIONARY','Heading 1')
add_body(doc,'This appendix records the primary project commands and a compact dictionary of the principal result fields. It supplements the report and does not replace the saved repository outputs. Commands are included as reproducibility evidence; executing them requires the pinned dependencies, model artifacts, and raw result files described in the project README.',first=True)
add_sub(doc,'A.1 Reproduction Commands',[
'The repository supports a quick environment check, predictor training, routing-agent training, full grid evaluation, metrics aggregation, statistical analysis, success checking, figure generation, report export, and model packaging. The sequence below follows the saved README. Full-grid commands depend on trained policies and raw logs already being present.'
])
commands=[
'python -m pytest',
'python -m scripts.smoke_env',
'python -m presto.predictor.dataset --config configs/predictor/gru.yaml',
'python -m presto.predictor.train --config configs/predictor/gru.yaml',
'python -m presto.predictor.train --config configs/predictor/logreg.yaml',
'python -m presto.agents.train --config configs/agent/reactive_dqn.yaml --seed 0',
'python -m presto.agents.train --config configs/agent/predictive_dqn.yaml --seed 0',
'python -m presto.agents.train --config configs/agent/presto_dqn.yaml --seed 0',
'python -m presto.eval.run_grid --grid configs/grid/main.yaml --workers 4',
'python -m presto.eval.metrics --raw-dir results/raw --out results/episode_metrics.parquet',
'python -m presto.eval.stats --metrics results/episode_metrics.parquet --out-dir results/tables',
'python -m presto.eval.check_success',
'python -m presto.eval.plots',
'python -m presto.eval.report',
'python -m scripts.export_report_pdf',
'python -m scripts.package_models']
for cmd in commands:
    add_code(doc,'Command',cmd)
add_sub(doc,'A.2 Data Dictionary',[
'The saved episode_metrics.parquet contains one row per evaluated episode and identifies method, topology, scenario, training seed, scenario seed, horizon, and lambda. The following definitions describe the main aggregate metrics used in the report.'
])
add_table(doc,'Metric Data Dictionary',['Field','Definition'],[
('pdr','Total delivered traffic divided by total offered demand.'),('delay_mean','Delivered-traffic-weighted mean path delay.'),('throughput','Mean delivered traffic rate.'),('switches','Number of route changes within an episode.'),('unnecessary_switch_rate','Fraction of switches for which the previous route remains healthy through horizon H.'),('recovery_time','Mean ticks from qualifying failure to first valid selected route at or below the QoS loss threshold.'),('proactive_rate','Share of qualifying failures for which the route had already left the failing link.'),('energy','Sum of selected-path hop counts over the episode.'),('reward','Sum of per-step reward.'),('failure_events','Count of qualifying true link up-to-down transitions.'),('scheduled_events','Number of scheduled simulator events associated with the episode.')
], [Inches(1.6),Inches(4.4)],0,1)
add_sub(doc,'A.3 Project Artifact Map',[
'configs/ contains resolved experiment settings; presto/ contains simulator, predictor, policy, agent, and evaluation modules; tests/ contains automated tests; runs/ stores training checkpoints and logs; data/ stores predictor datasets; results/ contains raw evaluation, episode metrics, tables, figures, and report artifacts; models/ contains the curated release bundle and checksum manifest. The model archive is local-only because no remote release destination is configured.'
])

# Refresh fields when opened in Word; page-number fields are used in footers.
settings=doc.settings._element
update=OxmlElement('w:updateFields'); update.set(qn('w:val'),'true'); settings.append(update)
doc.core_properties.title='PRESTO-R Predictive Stability-Aware Reinforcement Learning Routing'
doc.core_properties.subject='BCSE308P Computer Networks Course Project Final Report'
doc.core_properties.author='Yashwanth Gajula'
doc.core_properties.keywords='PRESTO-R, routing, reinforcement learning, computer networks'
doc.save(OUT)
print(OUT)
