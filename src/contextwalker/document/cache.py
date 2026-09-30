import json
import os
from dataclasses import asdict
from typing import List

from contextwalker.config import CONTEXT_CACHE_FILE
from contextwalker.schema import Chunk


def load_cached_chunks(
    context_cache_file: str = CONTEXT_CACHE_FILE,
) -> List[Chunk]:
    with open(context_cache_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("context cache must contain a JSON list")

    chunks = []

    for item in data:
        if not isinstance(item, dict):
            raise ValueError("every context cache entry must be an object")

        text = item["text"]
        context = str(item.get("context") or "").strip()
        contextual_text = f"{context}\n\n{text}" if context else text

        chunks.append(
            Chunk(
                chunk_id=item["chunk_id"],
                page=item["page"],
                text=text,
                context=context,
                contextual_text=contextual_text,
            )
        )

    print(f"[OK] Loaded {len(chunks)} chunks.")
    return chunks


def save_cached_chunks(
    chunks: List[Chunk],
    context_cache_file: str = CONTEXT_CACHE_FILE,
) -> None:
    """Atomically save chunks so an interrupted write cannot corrupt cache."""

    cache_directory = os.path.dirname(os.path.abspath(context_cache_file))
    os.makedirs(cache_directory, exist_ok=True)
    temporary_file = context_cache_file + ".tmp"

    try:
        with open(temporary_file, "w", encoding="utf-8") as file:
            json.dump(
                [asdict(chunk) for chunk in chunks],
                file,
                indent=2,
                ensure_ascii=False,
            )
        os.replace(temporary_file, context_cache_file)
    finally:
        if os.path.exists(temporary_file):
            os.remove(temporary_file)
