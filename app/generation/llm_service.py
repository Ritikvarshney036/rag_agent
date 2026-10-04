from openai import OpenAI
import json
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
        history: list,
        context: str,
        user_memory: dict,
    ) -> str:

        history_text = ""

        for message in history:
            history_text += (
                f"{message['role']}: "
                f"{message['content']}\n"
            )

        system_prompt = """
    You are a helpful question-answering assistant.

    Answer the user's question using the provided document context.

    Rules:

    - Use the document context as the source of factual answers.
    - Do not invent information.
    - Do not use outside knowledge for document-related questions.
    - Use conversation history only to understand the current conversation.
    - Use user memory only to personalize the response style or behavior.
    - Never treat user memory as factual information from the PDF.
    - If the answer cannot be found in the document context, say:
    "I couldn't find the answer in the provided document."
    - Give a clear and concise answer.
    """

        user_prompt = f"""
    Conversation history:
    ----------------
    {history_text}
    ----------------

    User memory:
    ----------------
    {user_memory}
    ----------------

    Document context:
    ----------------
    {context}
    ----------------

    Current question:
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

    def extract_memory(
        self,
        question: str,
        answer: str,
        history: list,
    ) -> dict:

        history_text = ""

        for message in history:
            history_text += (
                f"{message['role']}: "
                f"{message['content']}\n"
            )

        system_prompt = """
    You are a memory extraction assistant.

    Identify information from the conversation that would be
    useful for future conversations with this user.

    Extract only stable user preferences or useful facts.

    Examples of useful information:
    - preferred answer style
    - preferred language
    - technical interests
    - learning goals
    - persistent preferences

    Do NOT store:
    - temporary questions
    - information from the PDF
    - the assistant's answer
    - sensitive personal information
    - information that is only relevant to the current question

    Return ONLY valid JSON in this format:

    {
        "preferences": {},
        "facts": []
    }

    If there is nothing worth remembering, return:

    {
        "preferences": {},
        "facts": []
    }
    """

        user_prompt = f"""
        Conversation history:
        ----------------
        {history_text}
        ----------------

        Latest user question:
        {question}

        Assistant answer:
        {answer}
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

        return json.loads(
            response.choices[0].message.content
        )