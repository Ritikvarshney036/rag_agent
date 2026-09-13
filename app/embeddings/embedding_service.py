from openai import OpenAI

from app.config import settings


class EmbeddingService:
    """
    Responsible for converting text into vector embeddings
    using OpenAI's embedding model.
    """

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url="https://api.euron.one/api/v1/euri"
        )

        self.model = settings.EMBEDDING_MODEL

    def embed_text(self, text: str) -> list[float]:
        """
        Generate an embedding for a single piece of text.
        """

        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )

        return response.data[0].embedding

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in a single API request.
        """

        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )

        return [item.embedding for item in response.data]