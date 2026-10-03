from langchain_openai import ChatOpenAI

from app.config import settings


class ChatModel:

    def __init__(self):

        self.llm = ChatOpenAI(
            model=settings.CHAT_MODEL,
            api_key=settings.OPENAI_API_KEY,
            base_url="https://api.euron.one/api/v1/euri",
            temperature=0,
        )

    def get_llm(self):
        return self.llm