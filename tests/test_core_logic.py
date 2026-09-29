import unittest

from contextwalker.agent.neighbors import fetch_neighbor_chunks
from contextwalker.retrieval.fusion import reciprocal_rank_fusion
from contextwalker.schema import Chunk


class NeighborChunkTests(unittest.TestCase):
    def test_neighbor_radius_is_clamped_to_five(self):
        lookup = {
            index: Chunk(index, index + 1, f"chunk {index}")
            for index in range(12)
        }

        result = fetch_neighbor_chunks(6, 99, lookup)

        self.assertEqual(result["radius"], 5)
        self.assertEqual(
            [chunk["chunk_id"] for chunk in result["chunks"]],
            list(range(1, 12)),
        )


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_chunk_found_by_both_retrievers_ranks_first(self):
        vector = [
            {"chunk_id": 1, "rank": 1, "source": "vector"},
            {"chunk_id": 2, "rank": 2, "source": "vector"},
        ]
        bm25 = [
            {"chunk_id": 2, "rank": 1, "source": "bm25"},
            {"chunk_id": 3, "rank": 2, "source": "bm25"},
        ]

        result = reciprocal_rank_fusion([vector, bm25])

        self.assertEqual(result[0]["chunk_id"], 2)
        self.assertEqual(result[0]["sources"], ["vector", "bm25"])


if __name__ == "__main__":
    unittest.main()

