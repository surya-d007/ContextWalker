import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from contextwalker import ContextWalker, ask_pdf
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


class PublicApiTests(unittest.TestCase):
    def make_fake_pipeline(self):
        module = types.ModuleType("contextwalker.pipeline")
        calls = {"build": 0, "search": 0}

        def build_system(**kwargs):
            calls["build"] += 1
            calls["build_kwargs"] = kwargs
            return "index", [0], "bm25", [0], ["chunk"]

        def search(**kwargs):
            calls["search"] += 1
            calls["search_kwargs"] = kwargs
            return "supported answer", [{"chunk_id": 0}]

        module.build_system = build_system
        module.search = search
        return module, calls

    def test_ask_builds_lazily_and_reuses_indexes(self):
        fake_pipeline, calls = self.make_fake_pipeline()

        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "manual.pdf"
            pdf_path.write_bytes(b"%PDF-1.4")
            cache_dir = Path(directory) / "cache"
            walker = ContextWalker(pdf_path, cache_dir=cache_dir)

            self.assertFalse(walker.is_ready)

            with patch.dict(
                sys.modules,
                {"contextwalker.pipeline": fake_pipeline},
            ):
                first = walker.ask("First question")
                second, results = walker.ask(
                    "Second question",
                    return_results=True,
                )

            self.assertEqual(first, "supported answer")
            self.assertEqual(second, "supported answer")
            self.assertEqual(results, [{"chunk_id": 0}])
            self.assertEqual(calls["build"], 1)
            self.assertEqual(calls["search"], 2)
            self.assertEqual(calls["build_kwargs"]["pdf_path"], str(pdf_path))

    def test_ask_pdf_convenience_function(self):
        fake_pipeline, calls = self.make_fake_pipeline()

        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "manual.pdf"
            pdf_path.write_bytes(b"%PDF-1.4")

            with patch.dict(
                sys.modules,
                {"contextwalker.pipeline": fake_pipeline},
            ):
                answer = ask_pdf(
                    pdf_path,
                    "What is covered?",
                    cache_dir=Path(directory) / "cache",
                )

            self.assertEqual(answer, "supported answer")
            self.assertEqual(calls["build"], 1)
            self.assertEqual(calls["search"], 1)

    def test_empty_question_is_rejected_before_build(self):
        walker = ContextWalker("missing.pdf")

        with self.assertRaisesRegex(ValueError, "non-empty"):
            walker.ask("   ")


if __name__ == "__main__":
    unittest.main()
