import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- Embeddings / vector DB ---
    JINA_API_KEY = os.getenv("JINA_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "agentic_rag"
    QDRANT_CLEAN_ONLY = os.getenv("QDRANT_CLEAN_ONLY", "false").lower() == "true"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # --- Groq (four keys, each from a separate account) ---
    GROQ_API_KEY_1 = os.getenv("GROQ_API_KEY_1")
    GROQ_API_KEY_2 = os.getenv("GROQ_API_KEY_2")
    GROQ_API_KEY_3 = os.getenv("GROQ_API_KEY_3")
    GROQ_API_KEY_4 = os.getenv("GROQ_API_KEY_4")
    GROQ_API_KEY = GROQ_API_KEY_1  # kept so existing code using settings.GROQ_API_KEY still works
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    GROQ_MODEL_PLANNER = os.getenv("GROQ_MODEL_PLANNER", "qwen/qwen3.8-27b")

    # --- Portkey gateway ---
    USE_PORTKEY = os.getenv("USE_PORTKEY", "false").lower() == "true"
    PORTKEY_API_KEY = (os.getenv("PORTKEY_API_KEY") or "").strip().strip("\"'") or None
    PORTKEY_CONFIG_GENERATOR = os.getenv("PORTKEY_CONFIG_GENERATOR", "pc-genera-3c27e4").strip().strip("\"'")
    PORTKEY_CONFIG_PLANNER = os.getenv("PORTKEY_CONFIG_PLANNER", "pc-planne-0b87f5").strip().strip("\"'")
    GROQ_SLUG = os.getenv("GROQ_SLUG")
    GROQ_SLUG_2 = os.getenv("GROQ_SLUG_2")
    GROQ_SLUG_3 = os.getenv("GROQ_SLUG_3")
    GROQ_SLUG_4 = os.getenv("GROQ_SLUG_4")

    # --- Tracing (LangSmith) ---
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "agentic-rag-evals")

    APP_ENV = os.getenv("APP_ENV", "dev")


settings = Settings()