from openai import OpenAI

from app.config import settings


class LLMService:
    """
    Generates answers using the retrieved document context.
    """

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url="https://api.euron.one/api/v1/euri"
        )

        self.model = settings.CHAT_MODEL

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
        )

        return response.choices[0].message.content