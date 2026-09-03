import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "agentic_rag"

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    GROQ_MODEL_PLANNER=os.getenv("GROQ_MODEL_PLANNER","qwen/qwen3.6-27b")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")

    PORTKEY_API_KEY = os.getenv("PORTKEY_API")   # note: your .env key is "PORTKEY_API", not "PORTKEY_API_KEY" — either rename it in .env to PORTKEY_API_KEY for consistency, or keep this os.getenv("PORTKEY_API") mapping
    GROQ_SLUG = os.getenv("GROQ_SLUG")
    GROQ_SLUG_2 = os.getenv("GROQ_SLUG_2")


settings = Settings()

if not settings.GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file before starting the API.")
    
