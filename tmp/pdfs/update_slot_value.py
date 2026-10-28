from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

root = Path(r'E:\Projects\CN project')
source = root / 'output/pdf/PRESTO-R_Computer_Networks_Project_Report_Updated.pdf'
target = root / 'output/pdf/PRESTO-R_Computer_Networks_Project_Report_Revised.pdf'
buf = BytesIO()
c = canvas.Canvas(buf, pagesize=A4)
# Cover only the existing slot value, then set the corrected value in its place.
x = 127.6 + stringWidth('Slot: ', 'Helvetica-Bold', 9)
y = 544
c.setFillColor(colors.HexColor('#10243A'))
c.rect(x - 0.35, y - 2.4, 39, 12, fill=1, stroke=0)
c.setFillColor(colors.white)
c.setFont('Helvetica', 9)
c.drawString(x, y, 'C2+TC2')
c.save()
buf.seek(0)
overlay = PdfReader(buf).pages[0]
reader = PdfReader(str(source))
reader.pages[0].merge_page(overlay)
writer = PdfWriter()
for page in reader.pages:
    writer.add_page(page)
writer.add_metadata(reader.metadata or {})
with target.open('wb') as stream:
    writer.write(stream)
print(target)
