
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LangGraph / FastAPI
    DB_URI: str
    OPENAI_API_KEY: str
    SUPABASE_URL: str
    SUPABASE_KEY: str
    TABLE_NAME: str
    TABLE_NAME_VEHICLES: str
    TABLE_NAME_DEALERSHIP: str
    LANGCHAIN_PROJECT: str
    LANGCHAIN_API_KEY: str
    LANGCHAIN_TRACING_V2: str
    GOOGLE_API_KEY:str

    # Graphrag
    GRAPHRAG_API_KEY: str
    GRAPHRAG_LLM_MODEL: str
    GRAPHRAG_EMBEDDING_MODEL: str

    class Config:
        env_file = ".env"
