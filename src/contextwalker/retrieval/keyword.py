import re
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi

from contextwalker.schema import Chunk


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b[\w\-.]+\b", text.lower())


def create_bm25(chunks: List[Chunk]):
    print("\n[6] Building Contextual BM25...")
    tokenized_corpus = [tokenize(chunk.contextual_text) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    row_to_chunk_id = [chunk.chunk_id for chunk in chunks]
    print("[OK] BM25 created.")
    return bm25, row_to_chunk_id


def bm25_search(query: str, bm25, row_to_chunk_id, top_k: int):
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    max_k = min(top_k, len(scores))
    indices = np.argsort(scores)[::-1][:max_k]
    results = []

    for rank, row_index in enumerate(indices):
        chunk_id = row_to_chunk_id[int(row_index)]
        results.append(
            {
                "chunk_id": chunk_id,
                "score": float(scores[row_index]),
                "rank": rank + 1,
                "source": "bm25",
            }
        )

    return results

