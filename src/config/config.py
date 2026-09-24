import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    JINA_API_KEY = os.getenv("JINA_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "agentic_rag"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    GROQ_MODEL_PLANNER = os.getenv("GROQ_MODEL_PLANNER", "qwen/qwen3.8-27b")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")

    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG = os.getenv("GROQ_SLUG")
    GROQ_SLUG_2 = os.getenv("GROQ_SLUG_2")

    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "agentic-rag-evals")

    QDRANT_CLEAN_ONLY = os.getenv("QDRANT_CLEAN_ONLY", "false").lower() == "true"

settings = Settings()