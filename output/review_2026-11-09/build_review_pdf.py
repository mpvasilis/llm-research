from pathlib import Path
import json
import re
from html import escape

import fitz
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether, Table, TableStyle


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'output/pdf/submission_review_2026-11-09.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
data = json.loads((HERE / 'review_content.json').read_text(encoding='utf-8'))
literature = (HERE / 'literature.txt').read_text(encoding='utf-8')

for name, filename in [('Body', 'segoeui.ttf'), ('Bold', 'segoeuib.ttf'), ('Italic', 'segoeuii.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(Path('C:/Windows/Fonts') / filename)))
pdfmetrics.registerFontFamily('Body', normal='Body', bold='Bold', italic='Italic', boldItalic='Bold')

NAVY = colors.HexColor('#173246')
TEAL = colors.HexColor('#147D85')
INK = colors.HexColor('#243642')
MUTED = colors.HexColor('#566772')
PALE = colors.HexColor('#EDF4F5')
LINE = colors.HexColor('#D8E4E8')
WIDTH, HEIGHT = A4
MARGIN = 49
CONTENT_W = WIDTH - MARGIN * 2
CONTENT_H = HEIGHT - 106

styles = {
    'kicker': ParagraphStyle('kicker', fontName='Bold', fontSize=9, leading=12, textColor=TEAL, spaceAfter=10),
    'title': ParagraphStyle('title', fontName='Bold', fontSize=25, leading=30, textColor=NAVY, spaceAfter=12),
    'sub': ParagraphStyle('sub', fontName='Body', fontSize=12, leading=17, textColor=MUTED, spaceAfter=14),
    'h1': ParagraphStyle('h1', fontName='Bold', fontSize=19, leading=24, textColor=NAVY, spaceAfter=15),
    'h2': ParagraphStyle('h2', fontName='Bold', fontSize=11.4, leading=15, textColor=NAVY, spaceBefore=6, spaceAfter=5, keepWithNext=True),
    'body': ParagraphStyle('body', fontName='Body', fontSize=10.5, leading=14.6, textColor=INK, spaceAfter=10, allowWidows=0, allowOrphans=0),
    'small': ParagraphStyle('small', fontName='Body', fontSize=9.2, leading=12.8, textColor=MUTED, spaceAfter=7, allowWidows=0, allowOrphans=0),
    'lit': ParagraphStyle('lit', fontName='Body', fontSize=9.8, leading=13.5, textColor=INK, spaceAfter=7, allowWidows=0, allowOrphans=0),
    'author': ParagraphStyle('author', fontName='Bold', fontSize=9.2, leading=12.5, textColor=TEAL, spaceAfter=4),
    'callout': ParagraphStyle('callout', fontName='Bold', fontSize=11.3, leading=16.5, textColor=NAVY, spaceAfter=0),
}


def clean(text):
    return text.replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', ' - ')


def markup(text):
    text = clean(text)
    chunks = []
    last = 0
    urls = list(re.finditer(r'https?://[^\s;]+', text))
    for i, match in enumerate(urls):
        chunks.append(escape(text[last:match.start()]))
        url = match.group().rstrip(').,')
        trailing = match.group()[len(url):]
        label = 'Code repository' if 'github.com' in url else ('OSF materials' if 'osf.io' in url else ('Primary source' if i == 0 else 'Full text'))
        chunks.append(f'<a href="{escape(url, quote=True)}" color="#147D85"><u>{label}</u></a>{escape(trailing)}')
        last = match.end()
    chunks.append(escape(text[last:]))
    return ''.join(chunks)


def p(text, style='body'):
    return Paragraph(markup(text), styles[style])


def title(text, eyebrow='REVIEW / OLMo-2'):
    return [p(eyebrow, 'kicker'), p(text, 'h1')]


def item(label, body, compact=False):
    return [p(label, 'h2'), p(body, 'lit' if compact else 'body')]


def callout(text):
    table = Table([[p(text, 'callout')]], colWidths=[CONTENT_W])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), PALE),
        ('BOX', (0, 0), (-1, -1), .5, LINE),
        ('LEFTPADDING', (0, 0), (-1, -1), 14),
        ('RIGHTPADDING', (0, 0), (-1, -1), 14),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 13),
    ]))
    return [table, Spacer(1, 16)]


sections = data['sections']
pages = []

page = [p('SUBMISSION REVIEW  /  ' + data['date'].upper(), 'kicker'),
        p(data['title'], 'title'), p(data['subtitle'], 'sub')]
page += callout(data['verdict'])
page += [p('The contribution worth building around', 'h2')]
page += [p(text) for text in sections[0]['paragraphs']]
pages.append(page)

page = title('What the existing results establish', '01 / EVIDENCE AUDIT')
for label, body in sections[1]['items']:
    page += item(label, body, compact=True)
pages.append(page)

page = title('Three questions, three experiments', '02 / EXPERIMENTAL STRUCTURE')
page += [p(text) for text in sections[2]['paragraphs']]
update = sections[2].get('retrospective_update')
if update:
    contrasts = json.loads((HERE / 'component_reanalysis/paired_measurement_contrasts.json').read_text(encoding='utf-8'))['contrasts']
    assert len(update['rows']) == len(contrasts) == 4
    for row, contrast in zip(update['rows'], contrasts):
        assert float(row[1]) == round(100 * contrast['estimate'], 1)
        report_interval = [float(value) for value in re.findall(r'[+-]?\d+\.\d+', row[2])]
        assert report_interval == [round(100 * value, 1) for value in contrast['ci95_simultaneous']]
    page += [p(update['heading'], 'h2'), p(update['summary'], 'small')]
    result_rows = [[p('Category / targeted control', 'small'), p('Change', 'small'), p('95% interval', 'small')]]
    result_rows += [[p(cell, 'small') for cell in row] for row in update['rows']]
    result_table = Table(result_rows, colWidths=[CONTENT_W * .57, CONTENT_W * .15, CONTENT_W * .28])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PALE),
        ('LINEBELOW', (0, 0), (-1, 0), .6, LINE),
        ('LINEBELOW', (0, -1), (-1, -1), .6, LINE),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    page += [result_table, Spacer(1, 7), p(update['method'], 'small')]
for label, body in sections[2]['items']:
    page += item(label, body)
pages.append(page)

page = title('First: validate and generate', '03 / PRIORITY WORK')
for label, body in sections[3]['items'][:3]:
    page += item(label, body)
page += [p('Extensions if the claim requires them', 'h2')]
for label, body in sections[4]['items'][:2]:
    page += item(label, body, compact=True)
pages.append(page)

page = title('Then: estimate and report', '04 / PRIORITY WORK')
for label, body in sections[3]['items'][3:]:
    page += item(label, body)
page += [p('Extensions if the claim requires them', 'h2')]
for label, body in sections[4]['items'][2:]:
    page += item(label, body, compact=True)
pages.append(page)

page = title('Make the paper easy to follow', '05 / READABILITY')
for text in sections[5]['paragraphs']:
    page += [p(text)]
page += callout('Use the same sequence for each result: question, estimate, uncertainty, interpretation. Report effect sizes in percentage points and define the comparison direction.')
pages.append(page)

page = title('What ready for submission means', '06 / COMPLETION AND EVIDENCE')
page += [p(text) for text in sections[6]['paragraphs']]
page += [p('Local evidence inspected', 'h2')]
evidence_labels = [
    'Current manuscript - acl_latex.tex',
    'Recovered-output diagnostics - raw_output_robustness.json',
    'AI-label diagnostic - ai_consensus_validation_v3.json',
    'Checkpoint contrasts - interaction_tests_v3.json',
    'Paired DPO analysis - conditional_results.json',
    'Prospective validation protocol - 15 September 2026',
    'Extended study protocol - 15 September 2026',
    'Frozen extended design - full_study/protocol_v1.json',
    'Completion audit - 15 September 2026',
    'New detector comparison - paired_measurement_contrasts.json (9 November 2026)',
]
for label in evidence_labels:
    page += [p(label, 'small')]
page += [Spacer(1, 8), p('How to read the literature comparison', 'h2'),
         p('The following pages summarize a targeted review of ten relevant works, not a systematic literature review. Verified methods and proposed adaptations are separated. Lee et al. was verified from its repository abstract only; detailed implementation claims are withheld. Ibrahim et al. is cited as the 2026 journal paper, with detailed methods checked in the earlier preprint.', 'small')]
pages.append(page)

matrix = literature.split('1. Lee,', 1)[1].split('RECOMMENDED ADDITIONS AND NON-ADDITIONS', 1)[0]
matrix = '1. Lee,' + matrix
blocks = re.split(r'(?m)(?=^\d+\. )', matrix)
blocks = [block.strip() for block in blocks if block.strip()]
assert len(blocks) == 10, len(blocks)


def paper_card(block):
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    result = [p(lines[0], 'author'), p(lines[1], 'h2')]
    for line in lines[2:]:
        if line.startswith(('Source:', 'Sources:', 'Published source:', 'Published record:', 'Author source:', 'Full-text methods checked:', 'Full-text methods checked in earlier preprint:', 'Code:', 'Code/data:', 'Paper-linked resources:')):
            result.append(p(line, 'small'))
        else:
            result.append(p(line, 'lit'))
    result.append(Spacer(1, 12))
    return result

groups = [(0, 2, 'Metric gains and human judgments'), (2, 4, 'Style and construct validity'),
          (4, 6, 'Training stages and attribution'), (6, 8, 'Response length and evaluation'),
          (8, 10, 'Human evaluation and judge reliability')]
for i, (start, end, heading) in enumerate(groups):
    page = title(heading, f'07 / RELATED WORK  /  {i+1} OF 5')
    for block in blocks[start:end]:
        page += paper_card(block)
    if i == 4:
        page += [p('Where this paper can contribute', 'h2'), p('The established literature already addresses style changes, length-sensitive evaluation and the need to validate metrics. The distinctive contribution here should be a reproducible, human-checked demonstration of how detector components and observed response windows alter the same checkpoint comparison. Keep the title and conclusions at the scope supported by that evidence.', 'lit')]
    pages.append(page)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.line(MARGIN, 40, WIDTH - MARGIN, 40)
    canvas.setFont('Body', 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 26, 'OLMo-2 submission review  |  9 November 2026')
    canvas.drawRightString(WIDTH - MARGIN, 26, str(doc.page))
    canvas.restoreState()


estimated = []
for page in pages:
    height = 0
    for flow in page:
        _, h = flow.wrap(CONTENT_W, CONTENT_H)
        height += h + flow.getSpaceBefore() + flow.getSpaceAfter()
    estimated.append(round(height, 1))
print('Planned page heights:', estimated, 'available:', CONTENT_H)
story = []
for i, page in enumerate(pages):
    if i:
        story.append(PageBreak())
    story.extend(page)

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                        topMargin=48, bottomMargin=58, title=data['title'],
                        author='Research review', subject=data['subtitle'])
doc.build(story, onFirstPage=footer, onLaterPages=footer)

pdf = fitz.open(OUT)
render_dir = HERE / 'review_pdf_pages'
render_dir.mkdir(exist_ok=True)
stale_preview = render_dir / 'page-13.png'
if stale_preview.exists():
    stale_preview.unlink()
counts = []
for n, page in enumerate(pdf):
    page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False).save(render_dir / f'page-{n+1:02d}.png')
    counts.append(len(page.get_text()))
    assert '\ufffd' not in page.get_text(), f'Replacement character on page {n+1}'
links = sum(len(page.get_links()) for page in pdf)
result = {
    'pdf': str(OUT), 'pages': len(pdf), 'characters_per_page': counts,
    'clickable_links': links, 'planned_page_heights': estimated,
    'source_json': str(HERE / 'review_content.json'),
    'source_update': 'Direct retrospective detector-comparison result added on 9 November 2026.',
    'retrospective_comparison_rows_verified': 4,
    'missing_evidence_paths': [p for p in data['local_evidence'] if not (ROOT / p).exists()],
    'render_directory': str(render_dir),
}
(HERE / 'review_pdf_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
