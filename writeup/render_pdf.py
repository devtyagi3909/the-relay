"""
render_pdf.py - Compiles writeup/writeup.md to a strict <= 2-page writeup/writeup.pdf.
Uses PyMuPDF (fitz) and markdown to format and layout the two-page technical report.
"""

import sys
from pathlib import Path
import markdown
import pymupdf

ROOT_DIR = Path(__file__).resolve().parent.parent
MD_PATH = ROOT_DIR / "writeup" / "writeup.md"
PDF_PATH = ROOT_DIR / "writeup" / "writeup.pdf"

CSS = """
<style>
  body {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 7.2pt;
    line-height: 1.22;
    color: #1a1a1a;
  }
  h1 {
    font-size: 11.5pt;
    font-weight: bold;
    color: #0f172a;
    margin: 0 0 3px 0;
    text-align: center;
  }
  h2 {
    font-size: 8.5pt;
    font-weight: bold;
    color: #1e3a8a;
    border-bottom: 0.5px solid #cbd5e1;
    margin: 5px 0 2px 0;
    padding-bottom: 1px;
  }
  h3 {
    font-size: 7.5pt;
    font-weight: bold;
    color: #334155;
    margin: 3px 0 1px 0;
  }
  p {
    margin: 0 0 3px 0;
    text-align: justify;
  }
  ul, ol {
    margin: 0 0 3px 0;
    padding-left: 12px;
  }
  li {
    margin-bottom: 1.5px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 3px 0 3px 0;
  }
  th {
    background-color: #f1f5f9;
    color: #0f172a;
    font-size: 6.2pt;
    font-weight: bold;
    border: 0.5px solid #94a3b8;
    padding: 1.5px 2px;
    text-align: left;
  }
  td {
    font-size: 6.0pt;
    border: 0.5px solid #cbd5e1;
    padding: 1px 2px;
  }
  code {
    font-family: Courier, monospace;
    font-size: 6.2pt;
    background-color: #f8fafc;
    padding: 0 1px;
  }
  pre {
    background-color: #f8fafc;
    border: 0.5px solid #cbd5e1;
    padding: 3px 4px;
    font-size: 6.0pt;
    margin: 2px 0;
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  hr {
    border: none;
    border-top: 0.5px solid #e2e8f0;
    margin: 3px 0;
  }
</style>
"""

def render():
    if not MD_PATH.exists():
        raise FileNotFoundError(f"Missing {MD_PATH}")

    text = MD_PATH.read_text(encoding="utf-8")

    # Balanced split: Page 1 covers Title, Sections 1, 2 (Table 1), 3 (Tables 2a/2b), 4 (Trace)
    # Page 2 covers Section 5, Retrieval Ablation (Table 3), Misleading nuances, 6, 7, 8, Feedback, Assistant
    split_marker = "## 5 When a limit is hit"
    if split_marker not in text:
        raise ValueError(f"Could not find '{split_marker}' in writeup.md")

    part1, part2 = text.split(split_marker, 1)
    part2 = split_marker + "\n" + part2

    html1 = CSS + markdown.markdown(part1, extensions=["tables", "fenced_code"])
    html2 = CSS + markdown.markdown(part2, extensions=["tables", "fenced_code"])

    doc = pymupdf.open()
    rect = pymupdf.Rect(20, 20, 595 - 20, 842 - 20)

    p1 = doc.new_page(width=595, height=842)
    p1.insert_htmlbox(rect, html1)

    p2 = doc.new_page(width=595, height=842)
    p2.insert_htmlbox(rect, html2)

    page_count = len(doc)
    assert page_count <= 2, f"Target page budget exceeded! Got {page_count} pages."

    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(PDF_PATH)
    print(f"[SUCCESS] Compiled {MD_PATH.name} -> {PDF_PATH} ({page_count} pages, {PDF_PATH.stat().st_size} bytes).")

if __name__ == "__main__":
    render()
