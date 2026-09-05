import time
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config.config import settings   
import requests

BATCH_SIZE = 50
_JINA_DIM = 1024
_FALLBACK_DIM = 768  # all-mpnet-base-v2

_active_model = None
_model_type: str | None = None  # "jina" or "fallback"

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v3" 

# ── Model initialisation ───────────────────────────────────────────────────────

def _probe_jina():
    """Try one embed call to verify Jina is reachable."""
    if not settings.JINA_API_KEY:
        return False
    try:
        resp = requests.post(
            JINA_API_URL,
            headers={
                "Authorization": f"Bearer {settings.JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": JINA_MODEL, "input": ["probe"]},
            timeout=15,
        )
        resp.raise_for_status()
        logfire.info(f"Jina embeddings ready ({JINA_MODEL}, {_JINA_DIM}-dim).")
        return True
    except Exception as e:
        logfire.warning(f"Jina probe failed: {e}. Trying Sentence Transformer next.")
        return False    


def _load_fallback():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading sentence-transformers fallback (all-mpnet-base-v2, 768-dim).")
    return SentenceTransformer("all-mpnet-base-v2")


def _init():
    """Initialise embedding model once per process. Called lazily on first use."""
    global _active_model, _model_type
    if _active_model is not None:
        return

    jina = _probe_jina()
    if jina:
        _active_model = jina
        _model_type = "jina"
        
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """Return the vector dimension for the active model. Call after _init()."""
    _init()
    return _JINA_DIM if _model_type == "jina" else _FALLBACK_DIM

# ── Jina batch call ─────────────────────────────────────────────────────

def _embed_jina(batch: list[str]) -> list[list[float]]:
    for attempt in range(4):
        try:
            resp = requests.post(
                JINA_API_URL,
                headers={
                    "Authorization": f"Bearer {settings.JINA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"model": JINA_MODEL, "input": batch},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = any(x in err for x in ("429", "rate", "quota"))
            if is_rate_limit and attempt < 3:
                wait = 2 ** attempt
                logfire.warning(f"Jina rate limit — retrying in {wait}s (attempt {attempt+1}/4).")
                time.sleep(wait)
            else:
                logfire.error(f"Jina embedding failed: {e}")
                raise
    raise RuntimeError("Jina rate limit persisted after 4 attempts.")

# ── Batch embedding with retry ─────────────────────────────────────────────────

def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _model_type == "jina":
        return _embed_jina(batch)
    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()


# ── Public API (same signatures as before) ─────────────────────────────────────

def embed_query(query: str) -> list[float]:
    _init()
    if _model_type == "jina":
        return _embed_jina([query])[0]
    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
    return all_embeddings