from pydantic_settings import BaseSettings, SettingsConfigDict
import os
project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
env_file_path = os.path.join(project_root, ".env")
class Settings(BaseSettings):
    OLLAMA_BASE_URL: str
    OLLAMA_MODEL: str
    OLLAMA_EMBED_MODEL: str
    model_config = SettingsConfigDict(env_file=env_file_path, extra="ignore")
settings = Settings()