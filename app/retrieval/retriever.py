import chromadb

from app.config import settings


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