from typing import Dict, List


def reciprocal_rank_fusion(
    result_lists: List[List[Dict]],
    k: int = 60,
):
    fused_scores = {}
    sources = {}

    for results in result_lists:
        for result in results:
            chunk_id = result["chunk_id"]
            rank = result["rank"]
            score = 1 / (k + rank)
            fused_scores.setdefault(chunk_id, 0.0)
            fused_scores[chunk_id] += score
            sources.setdefault(chunk_id, [])
            sources[chunk_id].append(result["source"])

    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    output = []

    for chunk_id, score in ranked:
        output.append(
            {
                "chunk_id": chunk_id,
                "rrf_score": score,
                "sources": sources[chunk_id],
            }
        )

    return output

