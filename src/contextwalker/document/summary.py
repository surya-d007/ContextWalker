import os
from typing import Dict, List

from contextwalker.config import SUMMARY_CACHE_FILE
from contextwalker.ollama import ollama_generate


def create_document_summary(
    pages: List[Dict],
    summary_cache_file: str = SUMMARY_CACHE_FILE,
) -> str:
    if os.path.exists(summary_cache_file):
        print("[CACHE] Loading document summary.")
        with open(summary_cache_file, "r", encoding="utf-8") as file:
            return file.read()

    print("\n[2] Creating document summary...")
    sample_text = ""

    for page in pages[:15]:
        sample_text += page["text"][:2500] + "\n"

    prompt = f"""
You are analyzing a document that will later
be used by a Retrieval Augmented Generation system.

Create a concise description of the document.

Include:

- main subject
- document type
- major systems or products
- important concepts
- terminology
- error codes
- likely searchable topics

Do not answer questions.

DOCUMENT:

{sample_text}


DOCUMENT DESCRIPTION:
"""

    summary = ollama_generate(prompt, temperature=0, max_tokens=700)

    with open(summary_cache_file, "w", encoding="utf-8") as file:
        file.write(summary)

    return summary
