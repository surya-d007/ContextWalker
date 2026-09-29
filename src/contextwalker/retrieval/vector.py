from typing import List

import faiss

from contextwalker.retrieval.embeddings import embed_documents, embed_query
from contextwalker.schema import Chunk


def create_faiss_index(chunks: List[Chunk]):
    print("\n[5] Building FAISS index...")
    texts = [chunk.contextual_text for chunk in chunks]
    embeddings = embed_documents(texts)
    dimensions = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimensions)
    index.add(embeddings)

    # FAISS returns vector row indexes, so keep the original ID mapping.
    row_to_chunk_id = [chunk.chunk_id for chunk in chunks]
    print(f"[OK] FAISS contains {index.ntotal} vectors.")
    return index, row_to_chunk_id


def vector_search(query: str, index, row_to_chunk_id, top_k: int):
    max_k = min(top_k, index.ntotal)
    query_embedding = embed_query(query)
    scores, indices = index.search(query_embedding, max_k)
    results = []

    for rank, row_index in enumerate(indices[0]):
        if row_index < 0:
            continue

        chunk_id = row_to_chunk_id[row_index]
        results.append(
            {
                "chunk_id": chunk_id,
                "score": float(scores[0][rank]),
                "rank": rank + 1,
                "source": "vector",
            }
        )

    return results

