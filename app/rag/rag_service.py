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
        # 2. Rewrite question using history
        # --------------------------------

        standalone_question = self.llm_service.rewrite_question(
            question=question,
            history=history,
        )

        # --------------------------------
        # 3. Create embedding from rewritten question
        # --------------------------------

        query_embedding = self.embedding_service.embed_text(
            standalone_question
        )

        # --------------------------------
        # 4. Retrieve relevant chunks
        # --------------------------------

        chunks = self.retriever.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        # --------------------------------
        # 5. Build document context
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
        # 6. Generate answer
        # --------------------------------

        answer = self.llm_service.generate_answer(
            question=standalone_question,
            context=context,
        )

        # --------------------------------
        # 7. Save user message
        # --------------------------------

        conversation_memory.add_message(
            session_id=session_id,
            role="user",
            content=question,
        )

        # --------------------------------
        # 8. Save assistant response
        # --------------------------------

        conversation_memory.add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
        )

        return {
            "session_id": session_id,
            "question": question,
            "standalone_question": standalone_question,
            "answer": answer,
            "sources": chunks,
            "history": history,
        }