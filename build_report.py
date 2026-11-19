"""Render the research note to PDF from the committed Markdown report.

Run after: pip install -r requirements.txt
    python build_report.py

This is a lightweight renderer; minor pagination differences across platforms are
possible. It does not change the experimental results or research note.
"""
from pathlib import Path
import html
import re

from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent
markdown = (ROOT / "research_note.md").read_text(encoding="utf-8")
output = ROOT / "research_note.pdf"
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ResearchTitle", parent=styles["Title"], fontName="Helvetica-Bold",
                          fontSize=16, leading=19, textColor=HexColor("#0F2F43"), spaceAfter=10))
styles.add(ParagraphStyle(name="ResearchH2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                          fontSize=11, leading=14, spaceBefore=10, spaceAfter=5,
                          textColor=HexColor("#0F2F43")))
styles.add(ParagraphStyle(name="ResearchBody", parent=styles["BodyText"], fontName="Helvetica",
                          fontSize=8.6, leading=12.3, spaceAfter=6, alignment=TA_LEFT))
styles.add(ParagraphStyle(name="ResearchSmall", parent=styles["ResearchBody"],
                          fontSize=7.7, leading=10.6))

def make_paragraph(s, style="ResearchBody"):
    s = html.escape(s.strip())
    s = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\x60([^\x60]+)\x60", r'<font face="Courier">\1</font>', s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", s)
    return Paragraph(s, styles[style])

doc = SimpleDocTemplate(str(output), pagesize=(612,792),
                        leftMargin=45,rightMargin=45,topMargin=44,bottomMargin=49,
                        title="SE.LLMA: Swedish weight-space programming pilot")
story = []
lines = markdown.splitlines()
i = 0
while i < len(lines):
    line=lines[i].strip()
    if not line:
        i += 1
        continue
    if line.startswith("|"):
        data=[]
        while i < len(lines) and lines[i].strip().startswith("|"):
            cells=[x.strip() for x in lines[i].strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?",x) for x in cells):
                data.append([make_paragraph(c,"ResearchSmall") for c in cells])
            i+=1
        if data:
            table=Table(data, colWidths=[240,80,80,122][:len(data[0])], repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),HexColor("#DCE8EF")),
                ("VALIGN",(0,0),(-1,-1),"TOP"),
                ("BOTTOMPADDING",(0,0),(-1,-1),5),
                ("TOPPADDING",(0,0),(-1,-1),5),
                ("LINEBELOW",(0,0),(-1,-1),0.35,HexColor("#CCCCCC")),
            ]))
            story.append(table)
            story.append(Spacer(1,8))
        continue
    if line.startswith("# "):
        story.append(make_paragraph(line[2:],"ResearchTitle"))
    elif line.startswith("## "):
        story.append(make_paragraph(line[3:],"ResearchH2"))
    else:
        paragraph=line
        while i+1<len(lines) and lines[i+1].strip() and not lines[i+1].startswith(("#","|")):
            i+=1
            paragraph+=" "+lines[i].strip()
        story.append(make_paragraph(paragraph))
    i+=1

def footer(canvas,doc):
    canvas.saveState()
    canvas.setFont("Helvetica",8)
    canvas.setFillColor(HexColor("#777777"))
    canvas.drawString(45,29,"SE.LLMA | Weight-Space Programming pilot")
    canvas.drawRightString(565,29,str(doc.page))
    canvas.restoreState()

doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(f"Wrote {output} ({output.stat().st_size} bytes)")
