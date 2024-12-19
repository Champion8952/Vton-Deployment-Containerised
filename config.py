from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    MODEL_PATH: str = r"C:\Users\AdminAilusion\Desktop\AilusionModelRepo\Ailusion-VTON-DEMO-v1.1"
    MAX_BATCH_SIZE: int = 4
    MAX_QUEUE_SIZE: int = 16
    INFERENCE_TIMEOUT: int = 300
    PORT: int = 8002
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()