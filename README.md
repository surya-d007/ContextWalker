# ContextWalker

ContextWalker is a local PDF question-answering system that combines contextual
chunking, semantic retrieval, BM25 keyword retrieval, Reciprocal Rank Fusion,
neural reranking, and agent-driven exploration of neighboring chunks.

![ContextWalker agentic RAG architecture](https://raw.githubusercontent.com/surya-d007/ContextWalker/main/docs/assets/contextwalker-system-architecture.png)

This repository is a modular reorganization of `agent_sample.py`. Its retrieval
and answering logic is intentionally unchanged.

## How it works

```text
PDF extraction
  -> document summary
  -> overlapping chunks
  -> LLM-generated context for every chunk
  -> FAISS vector search + BM25 keyword search
  -> Reciprocal Rank Fusion
  -> cross-encoder reranking
  -> top five starting chunks
  -> agent explores neighboring chunks
  -> supported answer
```

See the
[architecture guide](https://github.com/surya-d007/ContextWalker/blob/main/docs/ARCHITECTURE.md)
for a module-by-module explanation.

## Requirements

- Python 3.9 or newer
- Ollama running locally at `http://localhost:11434`
- The Ollama model `gpt-oss:20b`
- A PDF named `data.pdf` in the directory from which ContextWalker is run

## Installation

Install the published package from PyPI:

```bash
python -m pip install contextwalker
ollama pull gpt-oss:20b
```

For local development from a cloned repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
ollama pull gpt-oss:20b
```

Place the document at `data.pdf`, start Ollama, and run:

```bash
contextwalker
```

Alternatively:

```bash
python -m contextwalker
```

Type `exit`, `quit`, or `q` to stop the interactive prompt.

## Runtime cache

The first run creates `rag_cache/document_summary.txt` and
`rag_cache/contextual_chunks.json`. Later runs reuse the contextual chunks but
rebuild the in-memory FAISS and BM25 indexes. Delete `rag_cache/` when changing
the PDF or chunk-generation settings.

## Configuration

The original constants are preserved in `src/contextwalker/config.py`:

- chunks: 900 characters with 150 characters of overlap
- vector candidates: 60
- BM25 candidates: 60
- fused candidates: 100
- final reranked candidates: 5
- neighbor radius: at most 5 chunks
- agent reasoning steps: at most 25
- tool calls per question: at most 40

## Development

Validate imports and syntax without starting the heavyweight models:

```bash
python -m compileall -q src
```

The original `agent_sample.py` remains outside this repository and has not been
modified.

Release maintainers should follow the
[publishing guide](https://github.com/surya-d007/ContextWalker/blob/main/docs/PUBLISHING.md)
for the PyPI release process.
