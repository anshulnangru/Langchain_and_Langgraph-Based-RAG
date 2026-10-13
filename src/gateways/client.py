import itertools
import threading

import groq
import logfire
from langchain_groq import ChatGroq

from src.config.config import settings

# Errors that mean "this key/account/gateway is unavailable".
# Auth and validation errors are not included: they should fail loudly.
_retryable = [
    groq.RateLimitError,
    groq.APIConnectionError,
    groq.InternalServerError,
    groq.APITimeoutError,
]
try:  # openai is only present when langchain-openai is installed (needed for Portkey)
    import openai

    _retryable += [
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.InternalServerError,
        openai.APITimeoutError,
    ]
except ImportError:
    pass
_RETRYABLE = tuple(_retryable)

# The four Groq keys, skipping any that are not set
_GROQ_KEYS = [
    k
    for k in (
        settings.GROQ_API_KEY_1,
        settings.GROQ_API_KEY_2,
        settings.GROQ_API_KEY_3,
        settings.GROQ_API_KEY_4,
    )
    if k
]
_key_cycle = itertools.cycle(_GROQ_KEYS) if _GROQ_KEYS else None
_key_lock = threading.Lock()


def _resolve(feature: str, model):
    m = model or (
        settings.GROQ_MODEL_PLANNER if feature == "planner" else settings.GROQ_MODEL
    )
    max_tok = 150 if feature == "planner" else 950
    return m, max_tok


def _direct_groq(model: str, max_tok: int):
    """Direct Groq call. Starts on the next key, falls to the others on 429/5xx."""
    if not _GROQ_KEYS:
        raise RuntimeError("No Groq API key configured. Set GROQ_API_KEY_1..4 in .env.")

    with _key_lock:
        first = next(_key_cycle)

    def build(key: str) -> ChatGroq:
        return ChatGroq(
            api_key=key,
            model=model,
            temperature=0,
            max_tokens=max_tok,
            max_retries=2,
            timeout=60,
        )

    llm = build(first)
    others = [build(k) for k in _GROQ_KEYS if k != first]
    if others:
        llm = llm.with_fallbacks(others, exceptions_to_handle=_RETRYABLE)
    return llm


def _portkey(feature: str, model: str, max_tok: int):
    """Portkey call. The config ID decides the fallback chain and the model."""
    from langchain_openai import ChatOpenAI
    from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL

    config_id = (
        settings.PORTKEY_CONFIG_PLANNER
        if feature == "planner"
        else settings.PORTKEY_CONFIG_GENERATOR
    )
    logfire.info(f"LLM via Portkey: feature={feature} config={config_id}")
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=model,  # placeholder: the config's override_params sets the real model
        temperature=0,
        max_tokens=max_tok,
        max_retries=1,
        timeout=60,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=config_id,
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": settings.APP_ENV,
            },
        ),
    )


def get_langchain_llm(feature: str = "rag", model: str = None):
    m, max_tok = _resolve(feature, model)

    if not (settings.USE_PORTKEY and settings.PORTKEY_API_KEY):
        logfire.info(f"LLM direct Groq: feature={feature} model={m}")
        return _direct_groq(m, max_tok)

    # Portkey first; if Portkey itself is unavailable, fall back to direct Groq.
    return _portkey(feature, m, max_tok).with_fallbacks(
        [_direct_groq(m, max_tok)], exceptions_to_handle=_RETRYABLE
    )


def extract_cache_status(response) -> str:
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"