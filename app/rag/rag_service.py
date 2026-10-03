from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.retriever import Retriever
from app.generation.llm_service import LLMService
from app.memory.memory_manager import conversation_memory


class RAGService:
    """
    Coordinates the complete conversational RAG pipeline.
    """

    def __init__(self):

        self.embedding_service = EmbeddingService()
        self.retriever = Retriever()
        self.llm_service = LLMService()

    def answer(
        self,
        question: str,
        session_id: str,
        top_k: int = 5,
    ):

        # --------------------------------
        # 1. Get conversation history
        # --------------------------------

        history = conversation_memory.get_history(
            session_id
        )

        # --------------------------------
        # 2. Create query embedding
        # --------------------------------

        query_embedding = self.embedding_service.embed_text(
            question
        )

        # --------------------------------
        # 3. Retrieve relevant chunks
        # --------------------------------

        chunks = self.retriever.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        # --------------------------------
        # 4. Build context
        # --------------------------------

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

        # --------------------------------
        # 5. Generate answer
        # --------------------------------

        answer = self.llm_service.generate_answer(
            question=question,
            context=context,
        )

        # --------------------------------
        # 6. Save user message
        # --------------------------------

        conversation_memory.add_message(
            session_id=session_id,
            role="user",
            content=question,
        )

        # --------------------------------
        # 7. Save assistant response
        # --------------------------------

        conversation_memory.add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
        )

        return {
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "sources": chunks,
            "history": history,
        }