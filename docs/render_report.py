#!/usr/bin/env python3
import re, subprocess, sys, pathlib
from markdown_it import MarkdownIt

ROOT = pathlib.Path(__file__).resolve().parent
md_path = ROOT / "oxidize_research_report.md"
html_path = ROOT / "oxidize_research_report.html"
pdf_path = ROOT / "oxidize_research_report.pdf"

md_text = md_path.read_text()

md = MarkdownIt("commonmark", {"html": True, "linkify": False, "typographer": True}).enable("table").enable("strikethrough")
body_html = md.render(md_text)

# Resolve image paths to absolute file URIs
def fix_img(m):
    src = m.group(1)
    p = (ROOT / src).resolve()
    return f'src="file://{p}"'
body_html = re.sub(r'src="([^"]+)"', fix_img, body_html)

# Add page breaks before each H1
body_html = re.sub(r"<h1>", '<h1 class="chapter">', body_html)

css = """
@page { size: Letter; margin: 0.9in 0.85in 0.95in 0.85in; }
html, body { font-family: Georgia, "Times New Roman", serif; color: #1a1a1a; font-size: 11pt; line-height: 1.55; }
body { max-width: 100%; }
h1.chapter { page-break-before: always; font-size: 22pt; margin: 0.2in 0 0.25in 0; color: #111; border-bottom: 2px solid #222; padding-bottom: 6pt; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
h1.chapter:first-of-type { page-break-before: avoid; }
h2 { font-size: 15pt; margin-top: 22pt; color: #111; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
h3 { font-size: 12pt; margin-top: 16pt; color: #222; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
h4 { font-size: 11pt; margin-top: 12pt; color: #333; font-style: italic; }
p { margin: 0.6em 0; text-align: justify; hyphens: auto; }
ul, ol { margin: 0.5em 0 0.8em 1.3em; }
li { margin: 0.18em 0; }
table { border-collapse: collapse; margin: 14pt 0; width: 100%; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 5pt 8pt; text-align: left; vertical-align: top; }
th { background: #f0f0f0; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-weight: 600; }
tr:nth-child(even) td { background: #fafafa; }
blockquote { border-left: 3px solid #888; margin: 0.8em 0; padding: 0.2em 0 0.2em 14pt; color: #444; font-style: italic; background: #f7f7f7; }
img { max-width: 88%; display: block; margin: 16pt auto 6pt auto; page-break-inside: avoid; border: 1px solid #ddd; padding: 4pt; background: white; }
hr { border: none; border-top: 1px solid #aaa; margin: 24pt 0; }
strong { color: #000; }
code { font-family: "SFMono-Regular", Menlo, Consolas, monospace; font-size: 9.5pt; background: #f3f3f3; padding: 1pt 4pt; border-radius: 3px; }
pre { background: #f6f6f6; padding: 10pt 12pt; border-radius: 4px; font-size: 9pt; overflow: hidden; page-break-inside: avoid; }
.cover { text-align: center; padding-top: 1.6in; }
"""

html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Oxidize Technical Research Report</title>
<style>{css}</style></head><body>{body_html}</body></html>"""

html_path.write_text(html)
print(f"wrote {html_path}")

# Render to PDF with chromium
cmd = [
    "chromium",
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_path}",
    f"file://{html_path}",
]
print(" ".join(cmd))
r = subprocess.run(cmd, capture_output=True, text=True)
sys.stdout.write(r.stdout)
sys.stderr.write(r.stderr)
print(f"wrote {pdf_path} ({pdf_path.stat().st_size} bytes)")
