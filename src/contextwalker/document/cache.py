import json
from typing import List

from contextwalker.config import CONTEXT_CACHE_FILE
from contextwalker.schema import Chunk


def load_cached_chunks() -> List[Chunk]:
    with open(CONTEXT_CACHE_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    chunks = []

    for item in data:
        context = item.get("context", "")
        contextual_text = item.get("contextual_text")

        if not contextual_text:
            contextual_text = context + "\n\n" + item["text"]

        chunks.append(
            Chunk(
                chunk_id=item["chunk_id"],
                page=item["page"],
                text=item["text"],
                context=context,
                contextual_text=contextual_text,
            )
        )

    print(f"[OK] Loaded {len(chunks)} chunks.")
    return chunks

