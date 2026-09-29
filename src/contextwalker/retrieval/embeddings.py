from typing import List

import numpy as np

from contextwalker.retrieval.model_runtime import embedding_model


def embed_documents(texts: List[str]) -> np.ndarray:
    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    instruction = "Represent this sentence for searching relevant passages: "
    embedding = embedding_model.encode(
        [instruction + query],
        normalize_embeddings=True,
    )
    return np.asarray(embedding, dtype="float32")

