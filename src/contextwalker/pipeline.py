import os
from typing import List

from contextwalker.agent.explorer import agent_answer
from contextwalker.agent.neighbors import build_chunk_lookup
from contextwalker.config import (
    BM25_TOP_K,
    CONTEXT_CACHE_FILE,
    FINAL_TOP_K,
    FUSED_TOP_K,
    PDF_PATH,
    VECTOR_TOP_K,
)
from contextwalker.document.cache import load_cached_chunks
from contextwalker.document.chunking import chunk_pages
from contextwalker.document.context import contextualize_chunks
from contextwalker.document.extraction import extract_pdf
from contextwalker.document.summary import create_document_summary
from contextwalker.retrieval.fusion import reciprocal_rank_fusion
from contextwalker.retrieval.keyword import bm25_search, create_bm25
from contextwalker.retrieval.reranking import rerank
from contextwalker.retrieval.vector import create_faiss_index, vector_search
from contextwalker.schema import Chunk


def search(
    query: str,
    index,
    vector_row_map,
    bm25,
    bm25_row_map,
    chunks: List[Chunk],
):
    chunk_lookup = build_chunk_lookup(chunks)

    print("\n" + "=" * 80)
    print("QUERY:", query)
    print("=" * 80)

    vector_results = vector_search(
        query=query,
        index=index,
        row_to_chunk_id=vector_row_map,
        top_k=VECTOR_TOP_K,
    )
    print(f"\n[SEARCH] Vector results: {len(vector_results)}")

    bm25_results = bm25_search(
        query=query,
        bm25=bm25,
        row_to_chunk_id=bm25_row_map,
        top_k=BM25_TOP_K,
    )
    print(f"[SEARCH] BM25 results: {len(bm25_results)}")

    fused = reciprocal_rank_fusion([vector_results, bm25_results])
    fused = fused[:FUSED_TOP_K]
    print(f"[SEARCH] Fused candidates: {len(fused)}")

    final_results = rerank(
        query=query,
        candidates=fused,
        chunk_lookup=chunk_lookup,
        top_k=FINAL_TOP_K,
    )

    print("\n" + "=" * 80)
    print("TOP INITIAL RETRIEVAL RESULTS")
    print("=" * 80)

    for rank, result in enumerate(final_results, start=1):
        chunk = chunk_lookup[result["chunk_id"]]
        print(f"\n#{rank}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Page: {chunk.page}")
        print(f"RRF Score: {result['rrf_score']:.6f}")
        print(f"Reranker Score: {result['reranker_score']:.6f}")
        print(f"Sources: {result['sources']}")
        print("\nTEXT:")
        print(chunk.text[:600])
        print("-" * 80)

    print("\n" + "=" * 80)
    print("STARTING AGENTIC DOCUMENT EXPLORATION")
    print("=" * 80)

    answer = agent_answer(
        query=query,
        final_results=final_results,
        chunk_lookup=chunk_lookup,
    )
    return answer, final_results


def build_system():
    if os.path.exists(CONTEXT_CACHE_FILE):
        print("\n[CACHE] Found existing contextual_chunks.json")
        chunks = load_cached_chunks()
    else:
        pages = extract_pdf(PDF_PATH)
        document_summary = create_document_summary(pages)
        chunks = chunk_pages(pages)
        chunks = contextualize_chunks(chunks, document_summary)

    index, vector_row_map = create_faiss_index(chunks)
    bm25, bm25_row_map = create_bm25(chunks)

    return index, vector_row_map, bm25, bm25_row_map, chunks

