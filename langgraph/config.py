
from pydantic_settings import BaseSettings
import logging, sys
from pythonjsonlogger import jsonlogger
class Settings(BaseSettings):
    LOG_LEVEL: str = "INFO"
    
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
    GOOGLE_CREDENTIALS_PATH:str
    GOOGLE_TOKEN_PATH : str

    # Graphrag
    GRAPHRAG_API_KEY: str
    GRAPHRAG_LLM_MODEL: str
    GRAPHRAG_EMBEDDING_MODEL: str

    def configure_logging(self):
        # remove existing handlers
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)
        # attach JSON handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            jsonlogger.JsonFormatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s"
            )
        )
        root.addHandler(handler)
        root.setLevel(self.LOG_LEVEL)

    class Config:
        env_file = ".env"
