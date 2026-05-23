#!/usr/bin/env python3
import re, subprocess, sys, pathlib
from markdown_it import MarkdownIt

ROOT = pathlib.Path(__file__).resolve().parent
md_path = ROOT / "oxidize_research_report.md"
html_path = ROOT / "oxidize_research_report.html"
pdf_path = ROOT / "oxidize_research_report.pdf"

md_text = md_path.read_text()

md = MarkdownIt("commonmark", {"html": True, "linkify": False, "typographer": False}).enable("table").enable("strikethrough")
body_html = md.render(md_text)

def fix_img(m):
    src = m.group(1)
    p = (ROOT / src).resolve()
    return f'src="file://{p}"'
body_html = re.sub(r'src="([^"]+)"', fix_img, body_html)

# Page breaks before each H1 (chapters/appendices)
body_html = re.sub(r"<h1>", '<h1 class="chapter">', body_html)

# arxiv-style: numbered figures from <img> blocks
fig_counter = {"n": 0}
def figurize(m):
    fig_counter["n"] += 1
    alt = m.group("alt")
    src = m.group("src")
    return (f'<figure class="arxiv-fig">'
            f'<img src="{src}" alt="{alt}"/>'
            f'<figcaption><b>Figure {fig_counter["n"]}.</b> {alt}</figcaption>'
            f'</figure>')
body_html = re.sub(r'<p><img src="(?P<src>[^"]+)" alt="(?P<alt>[^"]+)"[^>]*/?></p>', figurize, body_html)

css = r"""
@page { size: Letter; margin: 1.1in 1.25in 1.2in 1.25in; }
html, body {
  font-family: "Latin Modern Roman", "CMU Serif", "Computer Modern", "Linux Libertine", "Libertinus Serif", "Times New Roman", Georgia, serif;
  color: #111;
  font-size: 11pt;
  line-height: 1.55;
  text-rendering: optimizeLegibility;
}
body { max-width: 100%; }

/* arxiv-style cover */
.arxiv-cover { text-align: center; margin-top: 0.1in; margin-bottom: 0.25in; }
.arxiv-preprint { font-size: 9pt; color: #444; letter-spacing: 0.04em; margin-bottom: 0.4in; font-style: italic; }
.arxiv-title { font-size: 21pt; line-height: 1.18; margin: 0 auto 0.35in auto; font-weight: 700; letter-spacing: -0.005em; max-width: 5.5in; }
.arxiv-authors { font-size: 13pt; margin-top: 0.1in; }
.arxiv-affil { font-size: 10.5pt; color: #333; margin-top: 0.02in; font-style: italic; }
.arxiv-affil tt { font-family: "SFMono-Regular", Menlo, Consolas, monospace; font-style: normal; font-size: 9.5pt; }

.arxiv-abstract {
  margin: 0.35in 0.5in 0.25in 0.5in;
  font-size: 10pt;
  line-height: 1.5;
  text-align: justify;
  hyphens: auto;
}
.arxiv-abstract p { margin: 0.55em 0; }
.arxiv-abstract p:first-child { margin-top: 0; }
.arxiv-abstract strong:first-child { font-variant: small-caps; font-weight: 700; letter-spacing: 0.04em; }

.arxiv-meta { text-align: center; font-size: 8.5pt; color: #444; margin: 0.05in 0.5in 0.1in 0.5in; font-style: italic; }

/* chapter headings */
h1.chapter {
  page-break-before: always;
  font-size: 15pt;
  margin: 0 0 0.18in 0;
  color: #111;
  font-weight: 700;
  border-bottom: 0.5pt solid #333;
  padding-bottom: 5pt;
}
h1.chapter:first-of-type, .arxiv-cover + .arxiv-abstract + .arxiv-meta + hr + h1.chapter { page-break-before: avoid; }

h2 { font-size: 12pt; margin: 16pt 0 6pt 0; color: #111; font-weight: 700; }
h3 { font-size: 10.8pt; margin: 12pt 0 4pt 0; color: #111; font-weight: 700; }
h4 { font-size: 10.5pt; margin: 10pt 0 4pt 0; color: #222; font-style: italic; font-weight: 600; }

p { margin: 0.45em 0; text-align: justify; hyphens: auto; }

ul, ol { margin: 0.35em 0 0.7em 1.2em; padding-left: 0.4em; }
li { margin: 0.15em 0; }

table {
  border-collapse: collapse;
  margin: 12pt auto;
  width: 96%;
  font-size: 9pt;
  page-break-inside: avoid;
}
th, td { border-top: 0.5pt solid #555; border-bottom: 0.5pt solid #555; padding: 4pt 7pt; text-align: left; vertical-align: top; }
th { border-top: 0.8pt solid #111; border-bottom: 0.8pt solid #111; background: transparent; font-weight: 700; font-size: 8.8pt; }
tr td { border-left: none; border-right: none; }
tr:last-child td { border-bottom: 0.8pt solid #111; }

blockquote {
  border-left: 2pt solid #999;
  margin: 0.7em 0.2in;
  padding: 0.2em 0 0.2em 12pt;
  color: #333;
  font-size: 9.5pt;
}

figure.arxiv-fig { margin: 14pt auto; text-align: center; page-break-inside: avoid; }
figure.arxiv-fig img { max-width: 82%; max-height: 4.2in; border: 0.5pt solid #999; padding: 3pt; background: white; }
figure.arxiv-fig figcaption { font-size: 9pt; color: #222; margin-top: 5pt; max-width: 5.2in; margin-left: auto; margin-right: auto; text-align: left; }

hr { border: none; border-top: 0.5pt solid #888; margin: 18pt 0; }

strong { color: #000; }
em { font-style: italic; }
code, tt { font-family: "SFMono-Regular", Menlo, Consolas, monospace; font-size: 9pt; background: #f4f4f4; padding: 0 3pt; border-radius: 2px; }

a, a:link, a:visited { color: #1a3b8a; text-decoration: none; }
"""

html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Oxidize: A Rust-Native Stack for Local LLM Inference</title>
<style>{css}</style></head><body>{body_html}</body></html>"""

html_path.write_text(html)
print(f"wrote {html_path}")

cmd = [
    "chromium", "--headless=new", "--disable-gpu", "--no-sandbox",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_path}",
    f"file://{html_path}",
]
print(" ".join(cmd))
r = subprocess.run(cmd, capture_output=True, text=True)
sys.stdout.write(r.stdout); sys.stderr.write(r.stderr)
print(f"wrote {pdf_path} ({pdf_path.stat().st_size} bytes)")
