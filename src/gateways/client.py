import logfire
from langchain_groq import ChatGroq
from src.config.config import settings

# ============================================================
# PORTKEY TEMPORARILY BYPASSED — 500 error on routing config
# Restore after debugging Portkey dashboard config
# Original Portkey implementation preserved below (commented)
# ============================================================

def get_langchain_llm(feature: str = "rag", model: str = None) -> ChatGroq:
    m = model or (
        settings.GROQ_MODEL_PLANNER if feature == "planner"
        else settings.GROQ_MODEL
    )
    # Split traffic across two API keys to double the effective rate limit
    api_key = (
        settings.GROQ_FALLBACK_API_KEY 
        if feature == "planner" 
        else settings.GROQ_API_KEY
    )
    max_tok = 4096 if feature == "planner" else 950
    return ChatGroq(
        api_key=api_key,
        model=m,
        temperature=0,
        max_tokens=max_tok
    )

def extract_cache_status(response) -> str:
    return "MISS"  # no-op until Portkey is restored


# ============================================================
# PORTKEY IMPLEMENTATION — restore when config is fixed
# ============================================================
# from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL
# from langchain_openai import ChatOpenAI
#
# PORTKEY_CONFIGS = {
#     "planner": "pc-planne-38c870",
#     "default": "pc-portke-2052fc"
# }
#
# def get_langchain_llm(feature: str = "rag", model: str = None) -> ChatOpenAI:
#     config_id = PORTKEY_CONFIGS.get(feature, PORTKEY_CONFIGS["default"])
#     resolved_model = model or f"@{settings.GROQ_SLUG}/openai/gpt-oss-120b"
#     return ChatOpenAI(
#         api_key=settings.PORTKEY_API_KEY,
#         base_url=PORTKEY_GATEWAY_URL,
#         model=resolved_model,
#         temperature=0,
#         default_headers=createHeaders(
#             api_key=settings.PORTKEY_API_KEY,
#             config=config_id,
#             metadata={
#                 "feature": feature,
#                 "_user": "rag-system",
#                 "environment": "dev"
#             }
#         )
#     )
#
# def extract_cache_status(response) -> str:
#     for attr in ("_raw_response", "_response", "_http_response"):
#         raw = getattr(response, attr, None)
#         if raw is not None:
#             status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
#             if status:
#                 return status.upper()
#     return "MISS"