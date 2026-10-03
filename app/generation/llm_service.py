from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import settings


class LLMService:
    """
    Generates answers using LangChain.
    """

    def __init__(self):

        self.llm = ChatOpenAI(
            model=settings.CHAT_MODEL,
            api_key=settings.OPENAI_API_KEY,
            base_url="https://api.euron.one/api/v1/euri",
            temperature=0,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are a helpful question-answering assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do not use outside knowledge.
- If the answer cannot be found in the context, say:
  "I couldn't find the answer in the provided document."
- Do not make up information.
- Give a clear and concise answer.
"""
            ),
            (
                "human",
                """
Context:
----------------
{context}
----------------

Question:
{question}
"""
            )
        ])

        self.chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

    def generate_answer(
        self,
        question: str,
        context: str,
    ) -> str:

        answer = self.chain.invoke({
            "question": question,
            "context": context,
        })

        return answer