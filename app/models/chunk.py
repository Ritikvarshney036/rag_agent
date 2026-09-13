from dataclasses import dataclass

@dataclass
class Chunk:
    chunk_id: int
    document_name: str
    start_page: int
    end_page: int
    text: str
    token_count: int

