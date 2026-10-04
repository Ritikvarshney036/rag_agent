import chromadb

from langchain_core.documents import Document

from app.config import settings
from app.retrieval.reranker import Reranker


class Retriever:
    """
    Retrieves the most relevant document chunks
    from ChromaDB.
    """

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_DB_PATH
        )

        self.collection = self.client.get_collection(
            name="pdf_documents"
        )

        self.reranker = Reranker()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        retrieved_chunks = []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i in range(len(ids)):

            metadata = metadatas[i]

            retrieved_chunks.append(
                {
                    "chunk_id": ids[i],
                    "text": documents[i],
                    "start_page": metadata["start_page"],
                    "end_page": metadata["end_page"],
                    "distance": distances[i],
                }
            )

        return retrieved_chunks

    def search_with_reranking(
        self,
        query: str,
        query_embedding: list[float],
        retrieval_k: int = 10,
        top_k: int = 5,
    ) -> list[dict]:

        # Stage 1: Vector retrieval

        chunks = self.search(
            query_embedding=query_embedding,
            top_k=retrieval_k,
        )

        # Stage 2: Reranking

        reranked_chunks = self.reranker.rerank(
            query=query,
            chunks=chunks,
            top_k=top_k,
        )

        return reranked_chunks

    def search_documents(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[Document]:

        chunks = self.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        documents = []

        for chunk in chunks:

            documents.append(
                Document(
                    page_content=chunk["text"],
                    metadata={
                        "chunk_id": chunk["chunk_id"],
                        "start_page": chunk["start_page"],
                        "end_page": chunk["end_page"],
                        "distance": chunk["distance"],
                    },
                )
            )

        return documents