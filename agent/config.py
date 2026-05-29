"""환경변수 및 설정 관리."""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """애플리케이션 설정."""

    # LLM
    google_api_key: str = ""
    llm_model: str = "gemini-2.0-flash"

    # SQLite RAG DB
    db_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "krx_regulations.db")

    # Agent
    max_steps: int = 5
    rag_top_k: int = 5

    # Data
    regulations_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "krx_regulations")
    
    # Embedding
    embedding_model_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "local_model")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
