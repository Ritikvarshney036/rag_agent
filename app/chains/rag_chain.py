from app.llm.chat_model import ChatModel
from app.prompts.rag_prompt import rag_prompt


class RAGChain:

    def __init__(self):

        chat_model = ChatModel()

        self.chain = (
            rag_prompt
            | chat_model.get_llm()
        )

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:

        response = self.chain.invoke(
            {
                "question": question,
                "context": context,
            }
        )

        return response.content