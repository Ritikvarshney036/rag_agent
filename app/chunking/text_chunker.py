import tiktoken

from app.config import settings
from app.models.document import Document
from app.models.chunk import Chunk


class TextChunker:
    """
    Splits a parsed document into token-based overlapping chunks.
    Chunks can span across multiple PDF pages.
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        overlap: int = settings.CHUNK_OVERLAP,
        model_name: str = "gpt-4",
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.encoding = tiktoken.encoding_for_model(model_name)

        if self.overlap >= self.chunk_size:
            raise ValueError(
                "Overlap must be smaller than chunk size."
            )

    def chunk(self, document: Document) -> list[Chunk]:

        all_tokens = []
        token_pages = []

        # Convert all pages into one continuous token stream
        for page in document.pages:

            page_tokens = self.encoding.encode(page.text)

            all_tokens.extend(page_tokens)

            token_pages.extend(
                [page.page_number] * len(page_tokens)
            )

            # Add a small separator between PDF pages
            separator_tokens = self.encoding.encode("\n\n")

            all_tokens.extend(separator_tokens)

            token_pages.extend(
                [page.page_number] * len(separator_tokens)
            )

        chunks = []

        step = self.chunk_size - self.overlap

        chunk_id = 1

        for start in range(0, len(all_tokens), step):

            end = start + self.chunk_size

            chunk_tokens = all_tokens[start:end]

            if not chunk_tokens:
                break

            chunk_text = self.encoding.decode(chunk_tokens)

            start_page = token_pages[start]

            end_page = token_pages[
                min(end - 1, len(token_pages) - 1)
            ]

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_name=document.name,
                    start_page=start_page,
                    end_page=end_page,
                    text=chunk_text,
                    token_count=len(chunk_tokens),
                )
            )

            chunk_id += 1

            if end >= len(all_tokens):
                break

        return chunks