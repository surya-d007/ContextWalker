import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from contextwalker import ContextWalker, ask_pdf
from contextwalker.agent.neighbors import fetch_neighbor_chunks
from contextwalker.document.context import contextualize_chunks
from contextwalker.ollama import ollama_generate
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


class OllamaGenerationTests(unittest.TestCase):
    @staticmethod
    def response(payload):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = payload
        return response

    def test_empty_or_truncated_response_retries_with_more_tokens(self):
        responses = [
            self.response(
                {
                    "response": "A truncated contextual description",
                    "thinking": "reasoning used the output budget",
                    "done_reason": "length",
                }
            ),
            self.response(
                {
                    "response": "A complete contextual description.",
                    "done_reason": "stop",
                }
            ),
        ]

        with patch(
            "contextwalker.ollama.requests.post",
            side_effect=responses,
        ) as post:
            result = ollama_generate("prompt", max_tokens=100)

        self.assertEqual(result, "A complete contextual description.")
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args_list[0].kwargs["json"]["think"], "low")
        self.assertEqual(
            post.call_args_list[0].kwargs["json"]["options"]["num_predict"],
            100,
        )
        self.assertEqual(
            post.call_args_list[1].kwargs["json"]["options"]["num_predict"],
            200,
        )

    def test_repeated_empty_responses_raise_clear_error(self):
        response = self.response({"response": "", "done_reason": "stop"})

        with patch(
            "contextwalker.ollama.requests.post",
            return_value=response,
        ):
            with self.assertRaisesRegex(RuntimeError, "empty final response"):
                ollama_generate("prompt", max_attempts=2)


class ContextCacheRepairTests(unittest.TestCase):
    def test_only_missing_contexts_are_regenerated(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_file = Path(directory) / "contextual_chunks.json"
            cached_data = [
                {
                    "chunk_id": 0,
                    "page": 1,
                    "text": "first chunk",
                    "context": "Existing context.",
                    "contextual_text": "stale value",
                },
                {
                    "chunk_id": 1,
                    "page": 1,
                    "text": "second chunk",
                    "context": "",
                    "contextual_text": "\n\nsecond chunk",
                },
            ]
            cache_file.write_text(json.dumps(cached_data), encoding="utf-8")
            chunks = [
                Chunk(0, 1, "first chunk"),
                Chunk(1, 1, "second chunk"),
            ]

            with patch(
                "contextwalker.document.context.generate_context_for_chunk",
                return_value="Repaired context.",
            ) as generate:
                result = contextualize_chunks(
                    chunks,
                    "Document summary",
                    str(cache_file),
                )

            self.assertEqual(generate.call_count, 1)
            self.assertEqual(generate.call_args.args[0].chunk_id, 1)
            self.assertEqual(result[0].context, "Existing context.")
            self.assertEqual(result[1].context, "Repaired context.")

            saved = json.loads(cache_file.read_text(encoding="utf-8"))
            self.assertTrue(all(item["context"].strip() for item in saved))
            self.assertEqual(
                saved[0]["contextual_text"],
                "Existing context.\n\nfirst chunk",
            )

    def test_failed_repair_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_file = Path(directory) / "contextual_chunks.json"
            chunks = [Chunk(0, 1, "only chunk")]

            with patch(
                "contextwalker.document.context.generate_context_for_chunk",
                side_effect=RuntimeError("Ollama unavailable"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "next run will resume",
                ):
                    contextualize_chunks(
                        chunks,
                        "Document summary",
                        str(cache_file),
                    )


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
