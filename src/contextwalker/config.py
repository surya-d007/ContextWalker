import os


PDF_PATH = "data.pdf"

CACHE_DIR = "./rag_cache"
CONTEXT_CACHE_FILE = os.path.join(CACHE_DIR, "contextual_chunks.json")
SUMMARY_CACHE_FILE = os.path.join(CACHE_DIR, "document_summary.txt")

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gpt-oss:20b"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

VECTOR_TOP_K = 60
BM25_TOP_K = 60
FUSED_TOP_K = 100
FINAL_TOP_K = 5

MAX_NEIGHBOR_RADIUS = 5
MAX_AGENT_STEPS = 25
MAX_TOOL_CALLS = 40

os.makedirs(CACHE_DIR, exist_ok=True)

