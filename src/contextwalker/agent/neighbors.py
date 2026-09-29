from typing import Dict

from contextwalker.config import MAX_NEIGHBOR_RADIUS
from contextwalker.schema import Chunk


def build_chunk_lookup(chunks):
    return {chunk.chunk_id: chunk for chunk in chunks}


def fetch_neighbor_chunks(
    chunk_id: int,
    radius: int,
    chunk_lookup: Dict[int, Chunk],
) -> Dict:
    try:
        chunk_id = int(chunk_id)
        radius = int(radius)
    except Exception:
        return {"error": "chunk_id and radius must be integers."}

    radius = max(1, min(radius, MAX_NEIGHBOR_RADIUS))

    if chunk_id not in chunk_lookup:
        available_ids = sorted(chunk_lookup.keys())
        return {
            "error": f"Chunk {chunk_id} does not exist.",
            "minimum_chunk_id": available_ids[0],
            "maximum_chunk_id": available_ids[-1],
        }

    available_ids = sorted(chunk_lookup.keys())
    min_id = available_ids[0]
    max_id = available_ids[-1]
    start = max(min_id, chunk_id - radius)
    end = min(max_id, chunk_id + radius)
    results = []

    for current_id in range(start, end + 1):
        if current_id not in chunk_lookup:
            continue

        chunk = chunk_lookup[current_id]
        results.append(
            {
                "chunk_id": chunk.chunk_id,
                "page": chunk.page,
                "context": chunk.context,
                "text": chunk.text,
            }
        )

    return {
        "requested_center_chunk": chunk_id,
        "radius": radius,
        "returned_range": {
            "start_chunk": start,
            "end_chunk": end,
        },
        "chunks": results,
    }


NEIGHBOR_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_neighbor_chunks",
        "description": (
            "Read chunks surrounding ANY valid chunk "
            "in the document. The center chunk does "
            "not need to be one of the original "
            "retrieval results. You may call this "
            "tool repeatedly, including around chunks "
            "returned by previous calls, allowing you "
            "to walk forward or backward through the "
            "document until sufficient relevant "
            "evidence is found. Radius is 1 to 5."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chunk_id": {
                    "type": "integer",
                    "description": (
                        "Any valid chunk ID to use "
                        "as the center of exploration. "
                        "This may be a Top-5 chunk or "
                        "a chunk discovered during an "
                        "earlier tool call."
                    ),
                },
                "radius": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": (
                        "How many chunks before and "
                        "after the center chunk to read. "
                        "For example radius 2 returns "
                        "chunk_id-2 through chunk_id+2."
                    ),
                },
            },
            "required": ["chunk_id", "radius"],
        },
    },
}

