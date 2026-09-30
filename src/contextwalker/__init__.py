"""ContextWalker: contextual, hybrid, agentic RAG for local PDFs."""

from contextwalker.api import ContextWalker, ask_pdf
from contextwalker.schema import Chunk

__all__ = ["Chunk", "ContextWalker", "ask_pdf"]
__version__ = "0.2.1"
