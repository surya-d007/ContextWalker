import json
import os
from dataclasses import asdict
from typing import List

from contextwalker.config import CONTEXT_CACHE_FILE
from contextwalker.document.cache import load_cached_chunks
from contextwalker.ollama import ollama_generate
from contextwalker.schema import Chunk


def generate_context_for_chunk(
    chunk: Chunk,
    chunks: List[Chunk],
    document_summary: str,
) -> str:
    previous_text = ""
    next_text = ""

    if chunk.chunk_id > 0:
        previous_text = chunks[chunk.chunk_id - 1].text[-700:]

    if chunk.chunk_id < len(chunks) - 1:
        next_text = chunks[chunk.chunk_id + 1].text[:700]

    prompt = f"""
You are preparing a chunk for a
Retrieval Augmented Generation system.

Do NOT simply summarize the chunk.

Generate a short contextual description
explaining where this chunk belongs.

Include retrieval useful information such as:

- document topic
- section topic
- product names
- system names
- technical concepts
- error codes
- entities
- relationships
- terminology

Keep it between 1 and 3 sentences.


DOCUMENT DESCRIPTION:

{document_summary}


PREVIOUS TEXT:

{previous_text}


CURRENT CHUNK:

{chunk.text}


NEXT TEXT:

{next_text}


Return only the contextual description.
"""

    return ollama_generate(prompt, temperature=0, max_tokens=180)


def contextualize_chunks(
    chunks: List[Chunk],
    document_summary: str,
) -> List[Chunk]:
    if os.path.exists(CONTEXT_CACHE_FILE):
        print("\n[CACHE] Loading contextual chunks.")
        return load_cached_chunks()

    print("\n[4] Generating context for chunks...")

    for i, chunk in enumerate(chunks):
        print(f"[CONTEXT] {i + 1}/{len(chunks)}")

        try:
            context = generate_context_for_chunk(
                chunk,
                chunks,
                document_summary,
            )
        except Exception as error:
            print(
                "[WARN] Context "
                f"generation failed for chunk {chunk.chunk_id}: {error}"
            )
            context = ""

        chunk.context = context
        chunk.contextual_text = context + "\n\n" + chunk.text

    with open(CONTEXT_CACHE_FILE, "w", encoding="utf-8") as file:
        json.dump(
            [asdict(chunk) for chunk in chunks],
            file,
            indent=2,
            ensure_ascii=False,
        )

    return chunks

