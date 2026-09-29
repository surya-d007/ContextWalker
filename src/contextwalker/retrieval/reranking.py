from typing import Dict, List

from contextwalker.retrieval.model_runtime import reranker_model
from contextwalker.schema import Chunk


def rerank(
    query: str,
    candidates: List[Dict],
    chunk_lookup: Dict[int, Chunk],
    top_k: int,
):
    print(f"\n[7] Reranking {len(candidates)} candidates...")
    pairs = []
    valid_candidates = []

    for candidate in candidates:
        chunk_id = candidate["chunk_id"]

        if chunk_id not in chunk_lookup:
            continue

        chunk = chunk_lookup[chunk_id]
        pairs.append([query, chunk.contextual_text])
        valid_candidates.append(candidate)

    if not pairs:
        return []

    scores = reranker_model.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False,
    )
    ranked = []

    for candidate, score in zip(valid_candidates, scores):
        ranked.append({**candidate, "reranker_score": float(score)})

    ranked.sort(key=lambda item: item["reranker_score"], reverse=True)
    return ranked[:top_k]

