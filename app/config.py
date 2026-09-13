from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str

    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHAT_MODEL: str = "gpt-5-mini"

    CHROMA_DB_PATH: str = "./chroma_db"
    UPLOAD_FOLDER: str = "./uploads"

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 75

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()