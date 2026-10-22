"""Export results/report.md to results/report.pdf using Microsoft Edge headless."""

import html
from pathlib import Path
import re
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def markdown_to_html(md_text: str, base_dir: Path) -> str:
    lines = md_text.splitlines()
    body_parts: list[str] = []

    in_table = False
    table_headers: list[str] = []
    table_rows: list[list[str]] = []

    def flush_table():
        nonlocal in_table, table_headers, table_rows
        if not in_table:
            return
        html_tbl = ['<div class="table-container"><table>']
        if table_headers:
            html_tbl.append("<thead><tr>")
            for h in table_headers:
                html_tbl.append(f"<th>{inline_format(h)}</th>")
            html_tbl.append("</tr></thead>")
        if table_rows:
            html_tbl.append("<tbody>")
            for r in table_rows:
                html_tbl.append("<tr>")
                for cell in r:
                    html_tbl.append(f"<td>{inline_format(cell)}</td>")
                html_tbl.append("</tr>")
            html_tbl.append("</tbody>")
        html_tbl.append("</table></div>")
        body_parts.append("\n".join(html_tbl))
        in_table = False
        table_headers = []
        table_rows = []

    def inline_format(text: str) -> str:
        # Escape HTML entities first (except if already entity)
        t = html.escape(text.strip())
        # Inline code: `code`
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        # Bold: **bold**
        t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
        # Italic: *italic*
        t = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", t)
        return t

    for line in lines:
        stripped = line.strip()

        # Check table
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped[1:-1].split("|")]
            # Check if separator row
            if all(re.match(r"^:?-+:?$", c) for c in cells):
                continue
            if not in_table:
                in_table = True
                table_headers = cells
            else:
                table_rows.append(cells)
            continue
        elif in_table:
            flush_table()

        if not stripped:
            continue

        # Headers
        if stripped.startswith("#### "):
            body_parts.append(f"<h4>{inline_format(stripped[5:])}</h4>")
        elif stripped.startswith("### "):
            body_parts.append(f"<h3>{inline_format(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            body_parts.append(f"<h2>{inline_format(stripped[3:])}</h2>")
        elif stripped.startswith("# "):
            body_parts.append(f"<h1>{inline_format(stripped[2:])}</h1>")
        # Images: ![alt](src)
        elif m := re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped):
            alt_text, src = m.group(1), m.group(2)
            # Resolve image relative to base_dir
            img_path = (base_dir / src).resolve()
            img_uri = img_path.as_uri()
            body_parts.append(
                f'<figure class="figure-container">'
                f'<img src="{img_uri}" alt="{html.escape(alt_text)}" />'
                f"<figcaption>{inline_format(alt_text)}</figcaption>"
                f"</figure>"
            )
        else:
            # Paragraph
            body_parts.append(f"<p>{inline_format(stripped)}</p>")

    if in_table:
        flush_table()

    css = """
    @page {
        size: A4 portrait;
        margin: 18mm 16mm 18mm 16mm;
        @bottom-right {
            content: counter(page);
            font-size: 8pt;
            color: #64748b;
        }
    }
    *, *::before, *::after {
        box-sizing: border-box;
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 9.5pt;
        line-height: 1.5;
        color: #1e293b;
        margin: 0;
        padding: 0;
    }
    h1 {
        font-size: 18pt;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 14pt 0;
        padding-bottom: 6pt;
        border-bottom: 2px solid #0284c7;
        page-break-after: avoid;
        break-after: avoid;
    }
    h2 {
        font-size: 13pt;
        font-weight: 600;
        color: #0f172a;
        margin: 18pt 0 8pt 0;
        padding-bottom: 3pt;
        border-bottom: 1px solid #e2e8f0;
        page-break-after: avoid;
        break-after: avoid;
    }
    h3 {
        font-size: 11pt;
        font-weight: 600;
        color: #1e293b;
        margin: 12pt 0 6pt 0;
        page-break-after: avoid;
        break-after: avoid;
    }
    h4 {
        font-size: 10pt;
        font-weight: 600;
        color: #334155;
        margin: 10pt 0 4pt 0;
        page-break-after: avoid;
        break-after: avoid;
    }
    p {
        margin: 0 0 7pt 0;
        text-align: justify;
    }
    code {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 8.5pt;
        background-color: #f1f5f9;
        padding: 1px 4px;
        border-radius: 3px;
        color: #0f172a;
    }
    .figure-container {
        margin: 12pt auto;
        text-align: center;
        page-break-inside: avoid;
        break-inside: avoid;
    }
    .figure-container img {
        max-width: 95%;
        max-height: 400px;
        height: auto;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    figcaption {
        font-size: 8.5pt;
        color: #475569;
        margin-top: 5pt;
        font-weight: 500;
    }
    .table-container {
        margin: 10pt 0;
        overflow-x: auto;
        page-break-inside: avoid;
        break-inside: avoid;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 8pt;
        line-height: 1.35;
        page-break-inside: avoid;
        break-inside: avoid;
    }
    th, td {
        border: 1px solid #cbd5e1;
        padding: 4pt 6pt;
        text-align: left;
        vertical-align: middle;
    }
    th {
        background-color: #f1f5f9;
        color: #0f172a;
        font-weight: 600;
    }
    tbody tr:nth-child(even) {
        background-color: #f8fafc;
    }
    """

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PRESTO-R Technical Report</title>
<style>
{css}
</style>
</head>
<body>
{"\n".join(body_parts)}
</body>
</html>
"""
    return doc


def main():
    report_md = PROJECT_ROOT / "results" / "report.md"
    if not report_md.exists():
        print(f"Error: {report_md} not found.", file=sys.stderr)
        sys.exit(1)

    with open(report_md, "r", encoding="utf-8") as f:
        md_text = f.read()

    base_dir = report_md.parent
    html_content = markdown_to_html(md_text, base_dir)

    report_html = PROJECT_ROOT / "results" / "report.html"
    with open(report_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Wrote HTML intermediate: {report_html}")

    report_pdf = PROJECT_ROOT / "results" / "report.pdf"

    # Edge executable detection
    edge_paths = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    edge_exe = None
    for ep in edge_paths:
        if ep.exists():
            edge_exe = ep
            break

    if not edge_exe:
        print("Error: Microsoft Edge executable not found for PDF export.", file=sys.stderr)
        sys.exit(1)

    print(f"Using browser: {edge_exe}")
    cmd = [
        str(edge_exe),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={report_pdf.resolve()}",
        str(report_html.resolve().as_uri()),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if not report_pdf.exists() or report_pdf.stat().st_size == 0:
        print(f"PDF conversion failed! Output:\n{res.stdout}\n{res.stderr}", file=sys.stderr)
        sys.exit(1)

    pdf_size = report_pdf.stat().st_size
    print(f"Successfully exported PDF: {report_pdf} ({pdf_size} bytes, {pdf_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
