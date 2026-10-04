from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.retriever import Retriever
from app.generation.llm_service import LLMService
from app.memory.memory_manager import conversation_memory
from app.memory.long_term_memory import long_term_memory


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
        # 1. Get short-term conversation history
        # --------------------------------

        history = conversation_memory.get_history(
            session_id
        )

        # --------------------------------
        # 2. Get long-term user memory
        # --------------------------------

        user_memory = long_term_memory.get_memory(
            session_id
        )

        # --------------------------------
        # 3. Rewrite question using conversation history
        # --------------------------------

        standalone_question = self.llm_service.rewrite_question(
            question=question,
            history=history,
        )

        # --------------------------------
        # 4. Create embedding from standalone question
        # --------------------------------

        query_embedding = self.embedding_service.embed_text(
            standalone_question
        )

        # --------------------------------
        # 5. Retrieve relevant document chunks
        # --------------------------------

        query_embedding = self.embedding_service.embed_text(
            standalone_question
        )

        chunks = self.retriever.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        # --------------------------------
        # 6. Build document context
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
        # 7. Generate answer
        # --------------------------------

        answer = self.llm_service.generate_answer(
            question=standalone_question,
            context=context,
            history=history,
            user_memory=user_memory,
        )

        # --------------------------------
        # 8. Extract long-term memory
        # --------------------------------

        memory_update = self.llm_service.extract_memory(
            question=question,
            answer=answer,
            history=history,
        )

        # --------------------------------
        # 9. Save user message
        # --------------------------------

        conversation_memory.add_message(
            session_id=session_id,
            role="user",
            content=question,
        )

        # --------------------------------
        # 10. Save assistant response
        # --------------------------------

        conversation_memory.add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
        )

        # --------------------------------
        # 11. Update long-term preferences
        # --------------------------------

        if memory_update.get("preferences"):

            long_term_memory.update_memory(
                user_id=session_id,
                preferences=memory_update["preferences"],
            )

        # --------------------------------
        # 12. Update long-term facts
        # --------------------------------

        if memory_update.get("facts"):

            long_term_memory.update_memory(
                user_id=session_id,
                facts=memory_update["facts"],
            )

        # --------------------------------
        # 13. Return response
        # --------------------------------

        return {
            "session_id": session_id,
            "question": question,
            "standalone_question": standalone_question,
            "answer": answer,
            "sources": chunks,
            "history": history,
            "user_memory": user_memory,
            "memory_update": memory_update,
        }