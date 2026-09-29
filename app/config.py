import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    openai_api_key: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: str = "agentic-ai-ebook"
    
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    
    top_k_results: int = 4
    similarity_threshold: float = 0.35
    
    pdf_path: str = "ebook.pdf"
    chunk_size: int = 1000
    chunk_overlap: int = 150

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
