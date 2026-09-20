#!/usr/bin/env python3
"""Check the rendered WCL length, abstract, and unresolved LaTeX references."""
import json
import re
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
tex = (ROOT / "tex/main.tex").read_text()
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
words = len(abstract.split())
pdf = pymupdf.open(ROOT / "tex/main.pdf")
text = "\n".join(page.get_text() for page in pdf)
errors = []
if len(pdf) != 5:
    errors.append(f"Expected exactly five pages, found {len(pdf)}")
if not 75 <= words <= 100:
    errors.append(f"Abstract has {words} whitespace-delimited words; expected 75--100")
if "??" in text:
    errors.append("Unresolved reference found in PDF")
log = ROOT / "tex/main.log"
if log.exists() and re.search(r"Overfull \\[hv]box", log.read_text()):
    errors.append("Overfull box in main.log")
result = {"pages": len(pdf), "abstract_words": words,
          "page_dimensions_pt": [list(page.rect) for page in pdf], "errors": errors,
          "authors_pending": "Author Names and Affiliations to Be Supplied" in tex}
out = ROOT / "tex/paper_checks.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
if errors:
    raise SystemExit(1)
