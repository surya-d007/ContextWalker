"""Public Python API for building and querying a ContextWalker RAG system."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union


Answer = Union[str, Tuple[str, List[Dict]]]


class ContextWalker:
    """Build and query ContextWalker for one PDF document.

    Model loading and index construction are intentionally lazy. Calling
    :meth:`ask` automatically builds the system on the first question, or
    callers may invoke :meth:`build` explicitly and reuse it for many queries.
    """

    def __init__(
        self,
        pdf_path: Union[str, os.PathLike],
        cache_dir: Union[str, os.PathLike] = "./rag_cache",
    ) -> None:
        self.pdf_path = os.fspath(pdf_path)
        self.cache_dir = os.fspath(cache_dir)
        self.context_cache_file = os.path.join(
            self.cache_dir,
            "contextual_chunks.json",
        )
        self.summary_cache_file = os.path.join(
            self.cache_dir,
            "document_summary.txt",
        )

        self.index: Optional[Any] = None
        self.vector_row_map: Optional[Any] = None
        self.bm25: Optional[Any] = None
        self.bm25_row_map: Optional[Any] = None
        self.chunks: Optional[List[Any]] = None
        self.last_results: List[Dict] = []

    @property
    def is_ready(self) -> bool:
        """Return whether the document indexes have been built."""

        return self.index is not None and self.chunks is not None

    def build(self) -> "ContextWalker":
        """Prepare contextual chunks and build the retrieval indexes."""

        if not os.path.isfile(self.pdf_path):
            raise FileNotFoundError(
                f"ContextWalker PDF does not exist: {self.pdf_path}"
            )

        os.makedirs(self.cache_dir, exist_ok=True)

        # Imported lazily so `import contextwalker` does not load large models.
        from contextwalker.pipeline import build_system

        (
            self.index,
            self.vector_row_map,
            self.bm25,
            self.bm25_row_map,
            self.chunks,
        ) = build_system(
            pdf_path=self.pdf_path,
            context_cache_file=self.context_cache_file,
            summary_cache_file=self.summary_cache_file,
        )
        return self

    def ask(
        self,
        question: str,
        return_results: bool = False,
    ) -> Answer:
        """Answer a question using the configured PDF.

        Set ``return_results=True`` to receive ``(answer, retrieval_results)``.
        The retrieval results are the same top-five dictionaries returned by
        the existing search pipeline.
        """

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        if not self.is_ready:
            self.build()

        # Imported lazily for the same reason as build_system above.
        from contextwalker.pipeline import search

        answer, results = search(
            query=question.strip(),
            index=self.index,
            vector_row_map=self.vector_row_map,
            bm25=self.bm25,
            bm25_row_map=self.bm25_row_map,
            chunks=self.chunks,
        )
        self.last_results = results

        if return_results:
            return answer, results
        return answer


def ask_pdf(
    pdf_path: Union[str, os.PathLike],
    question: str,
    cache_dir: Union[str, os.PathLike] = "./rag_cache",
    return_results: bool = False,
) -> Answer:
    """Build ContextWalker for a PDF and answer one question."""

    walker = ContextWalker(pdf_path=pdf_path, cache_dir=cache_dir)
    return walker.ask(question, return_results=return_results)

