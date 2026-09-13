from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.retriever import Retriever
from app.generation.llm_service import LLMService


class RAGService:
    """
    Coordinates the complete RAG pipeline.
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.retriever = Retriever()
        self.llm_service = LLMService()

    def answer(
        self,
        question: str,
        top_k: int = 5,
    ):

        # 1. Convert question into embedding
        query_embedding = self.embedding_service.embed_text(
            question
        )

        # 2. Retrieve relevant chunks
        chunks = self.retriever.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        # 3. Build context for the LLM
        context_parts = []

        for chunk in chunks:

            context_parts.append(
                f"""
Source: {chunk['chunk_id']}
Pages: {chunk['start_page']} - {chunk['end_page']}

{chunk['text']}
"""
            )

        context = "\n\n".join(context_parts)

        # 4. Generate answer
        answer = self.llm_service.generate_answer(
            question=question,
            context=context,
        )

        return {
            "question": question,
            "answer": answer,
            "sources": chunks,
        }