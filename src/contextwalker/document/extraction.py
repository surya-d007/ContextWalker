import re
from typing import Dict, List

import pymupdf as fitz


def extract_pdf(pdf_path: str) -> List[Dict]:
    print("\n[1] Reading PDF...")
    doc = fitz.open(pdf_path)
    pages = []

    for page_number, page in enumerate(doc):
        text = page.get_text("text")
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            pages.append({"page": page_number + 1, "text": text})

    print(f"[OK] Extracted {len(pages)} pages.")
    return pages

