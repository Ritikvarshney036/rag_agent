import chromadb

from app.config import settings
from app.models.chunk import Chunk


class ChromaStore:
    """
    Handles storing and retrieving document chunks
    using ChromaDB.
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_DB_PATH
        )

        self.collection = self.client.get_or_create_collection(
            name="pdf_documents"
        )

    def add_chunks(
            self,
            chunks:list[Chunk],
            embeddings: list[list[float]],
            ) -> None:

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embedding must be the same."
            )
        if not chunks:
            return 

        ids = [
            f"{chunk.document_name}_{chunk.chunk_id}"
            for chunk in chunks
        ]

        documents = [
            chunk.text
            for chunk in chunks
        ]

        metadatas = [
            {
                "document": chunk.document_name,
                "start_page": chunk.start_page,
                "end_page": chunk.end_page,
                "token_count": chunk.token_count,
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )