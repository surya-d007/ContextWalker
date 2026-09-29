from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from contextwalker.config import CHUNK_OVERLAP, CHUNK_SIZE
from contextwalker.schema import Chunk


def chunk_pages(pages: List[Dict]) -> List[Chunk]:
    print("\n[3] Chunking document...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    chunk_id = 0

    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        for text in page_chunks:
            chunks.append(
                Chunk(chunk_id=chunk_id, page=page["page"], text=text)
            )
            chunk_id += 1

    print(f"[OK] Generated {len(chunks)} chunks.")
    return chunks

