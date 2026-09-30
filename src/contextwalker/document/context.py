import os
from typing import List

from contextwalker.config import CONTEXT_CACHE_FILE
from contextwalker.document.cache import load_cached_chunks, save_cached_chunks
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

    return ollama_generate(prompt, temperature=0, max_tokens=500)


def contextualize_chunks(
    chunks: List[Chunk],
    document_summary: str,
    context_cache_file: str = CONTEXT_CACHE_FILE,
) -> List[Chunk]:
    cached_by_key = {}

    if os.path.exists(context_cache_file):
        print("\n[CACHE] Loading contextual chunks.")
        try:
            cached_chunks = load_cached_chunks(context_cache_file)
            cached_by_key = {
                (chunk.chunk_id, chunk.page, chunk.text): chunk
                for chunk in cached_chunks
                if chunk.context.strip()
            }
        except (OSError, TypeError, ValueError, KeyError) as error:
            print(f"[WARN] Ignoring invalid context cache: {error}")

    reused = 0
    for chunk in chunks:
        cached = cached_by_key.get((chunk.chunk_id, chunk.page, chunk.text))
        if cached is None:
            continue

        chunk.context = cached.context.strip()
        chunk.contextual_text = f"{chunk.context}\n\n{chunk.text}"
        reused += 1

    missing_chunks = [chunk for chunk in chunks if not chunk.context.strip()]

    if reused:
        print(f"[CACHE] Reusing {reused} complete contextual chunks.")

    if not missing_chunks:
        print("[CACHE] All contextual chunks are complete.")
        return chunks

    print(
        "\n[4] Generating context for "
        f"{len(missing_chunks)} missing chunks..."
    )

    for i, chunk in enumerate(missing_chunks):
        print(f"[CONTEXT] {i + 1}/{len(missing_chunks)} (chunk {chunk.chunk_id})")

        try:
            context = generate_context_for_chunk(
                chunk,
                chunks,
                document_summary,
            )
        except Exception as error:
            raise RuntimeError(
                "Context generation failed for chunk "
                f"{chunk.chunk_id}. Completed chunks were saved and the "
                "next run will resume from this chunk."
            ) from error

        context = context.strip()
        if not context:
            raise RuntimeError(
                f"Context generation returned empty text for chunk "
                f"{chunk.chunk_id}; the incomplete result was not accepted."
            )

        chunk.context = context
        chunk.contextual_text = f"{context}\n\n{chunk.text}"
        save_cached_chunks(chunks, context_cache_file)

    return chunks
