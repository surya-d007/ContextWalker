from sentence_transformers import CrossEncoder, SentenceTransformer

from contextwalker.config import EMBEDDING_MODEL, RERANKER_MODEL


print("\n[MODEL] Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print("[MODEL] Embedding model loaded.")

print("\n[MODEL] Loading reranker...")
reranker_model = CrossEncoder(RERANKER_MODEL)
print("[MODEL] Reranker loaded.")

