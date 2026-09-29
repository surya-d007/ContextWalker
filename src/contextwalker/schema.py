from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: int
    page: int
    text: str
    context: str = ""
    contextual_text: str = ""

