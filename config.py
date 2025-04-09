from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    MODEL_PATH: str = r"models"
    huggingface_token: Optional[str] = None
    MAX_BATCH_SIZE: int = 4
    MAX_QUEUE_SIZE: int = 16
    INFERENCE_TIMEOUT: int = 300
    PORT: int = 8002
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"  # Allow extra fields

@lru_cache()
def get_settings():
    return Settings()