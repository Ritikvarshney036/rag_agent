from openai import OpenAI

from app.config import settings


class LLMService:
    """
    Handles LLM operations:
    - Answer generation
    - Question rewriting
    """

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url="https://api.euron.one/api/v1/euri"
        )

        self.model = settings.CHAT_MODEL

    def rewrite_question(
        self,
        question: str,
        history: list,
    ) -> str:

        history_text = ""

        for message in history:
            history_text += (
                f"{message['role']}: "
                f"{message['content']}\n"
            )

        system_prompt = """
You are a question rewriting assistant.

Your job is to rewrite the user's latest question
into a standalone question that can be understood
without conversation history.

Rules:

- Use the conversation history to resolve references.
- Resolve words like:
  "it", "its", "they", "this", "that", "those"
- Do not answer the question.
- Do not add information that is not present in the conversation.
- If the question is already standalone, return it unchanged.
- Return ONLY the rewritten question.
"""

        user_prompt = f"""
Conversation history:
----------------
{history_text}
----------------

Latest user question:
{question}

Standalone question:
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    def generate_answer(
        self,
        question: str,
        context: str,
    ) -> str:

        system_prompt = """
You are a helpful question-answering assistant.

Answer the user's question using ONLY the provided context.

Rules:

- Do not use outside knowledge.
- If the answer cannot be found in the context, say:
  "I couldn't find the answer in the provided document."
- Do not make up information.
- Give a clear and concise answer.
"""

        user_prompt = f"""
Context:
----------------
{context}
----------------

Question:
{question}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()