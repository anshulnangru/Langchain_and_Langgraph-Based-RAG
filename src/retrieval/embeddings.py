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

# Jina v3 is an asymmetric retrieval model: queries and passages must be
# embedded with different "task" values so they land in compatible regions
# of the vector space. Without this, Jina falls back to a generic/symmetric
# embedding mode, where long, broad, vocabulary-dense chunks score
# artificially high against almost any query (a "vocabulary magnet" effect)
# instead of the most specific, on-topic chunk winning. Confirmed via
# diagnose_recall.py: dedicated, narrow-topic files (e.g. add-memory_md.md)
# were being starved out of retrieval by broad reference pages
# (graph-api_md.md, overview_md.md, checkpointers_md.md) despite comparable
# chunk counts and chunk sizes across all of them — ruling out chunking
# granularity as the cause and pointing at the embedding task mismatch.
JINA_TASK_QUERY = "retrieval.query"
JINA_TASK_PASSAGE = "retrieval.passage"

# Pause between successive embed batches within the same file, Jina only.
# Without this, a file needing 2+ batches (anything over BATCH_SIZE chunks)
# fires them back-to-back with zero gap, which was reliably tripping Jina's
# rate limit on dense files even when SLEEP_BETWEEN_FILES (in processor.py)
# gave plenty of space *between* files. This is the intra-file gap that was
# missing.
INTER_BATCH_SLEEP_SECONDS = 3

# ── Model initialisation ───────────────────────────────────────────────────────

def _probe_jina():
    """Try one embed call to verify Jina is reachable, with retry on transient
    DNS blips and rate limits. Without retrying on 429s specifically, a
    single rate-limited probe call — very likely right after a heavy
    ingestion run that just hammered the Jina API — silently and
    permanently downgrades the *entire* process to the 768-dim fallback
    model, with no visible error, since _active_model/_model_type are
    decided once and cached globally. Every subsequent query then gets
    embedded at the wrong dimension and Qdrant rejects every search with a
    'expected dim: 1024, got 768' error -- which is exactly what happened."""
    if not settings.JINA_API_KEY:
        return False
    last_err = None
    for attempt in range(1, 5):
        try:
            resp = requests.post(
                JINA_API_URL,
                headers={
                    "Authorization": f"Bearer {settings.JINA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"model": JINA_MODEL, "input": ["probe"], "task": JINA_TASK_QUERY},
                timeout=15,
            )
            resp.raise_for_status()
            logfire.info(f"Jina embeddings ready ({JINA_MODEL}, {_JINA_DIM}-dim).")
            return True
        except Exception as e:
            last_err = e
            msg = str(e)
            msg_lower = msg.lower()
            is_rate_limit = any(x in msg_lower for x in ("429", "rate", "quota"))
            transient = (
                "No address associated" in msg
                or "Name or service not known" in msg
                or "Temporary failure in name resolution" in msg
                or "Connection" in msg
                or is_rate_limit
            )
            if not transient or attempt == 4:
                logfire.warning(
                    f"Jina probe failed after {attempt} attempt(s): {e}. "
                    f"Trying Sentence Transformer next."
                )
                return False
            # Rate limits need longer waits than DNS blips.
            wait = 10 * attempt if is_rate_limit else 2 ** (attempt - 1)
            logfire.warning(
                f"Jina probe hit {'rate limit' if is_rate_limit else 'transient error'} "
                f"(attempt {attempt}/4): {e} — retrying in {wait}s."
            )
            time.sleep(wait)
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
        raise RuntimeError(
            "Jina embeddings are unavailable (probe failed) and the Qdrant "
            "collection is 1024-dim. Refusing to fall back to the 768-dim "
            "model, which would return wrong-dimension vectors for every "
            "query. Check JINA_API_KEY / balance / rate limits."
        )


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """Return the vector dimension for the active model. Call after _init()."""
    _init()
    return _JINA_DIM if _model_type == "jina" else _FALLBACK_DIM

# ── Jina batch call ─────────────────────────────────────────────────────

def _embed_jina(batch: list[str], task: str) -> list[list[float]]:
    """task must be JINA_TASK_QUERY for a search query or JINA_TASK_PASSAGE
    for corpus chunks being embedded at ingestion time — see note above on
    why this asymmetry matters. Never call this without an explicit task."""
    for attempt in range(4):
        try:
            resp = requests.post(
                JINA_API_URL,
                headers={
                    "Authorization": f"Bearer {settings.JINA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"model": JINA_MODEL, "input": batch, "task": task},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = any(x in err for x in ("429", "rate", "quota"))
            if is_rate_limit and attempt < 3:
                # 5s, 10s, 20s -- longer than the original 1/2/4s, since that
                # was proving too short to actually clear the rate-limit
                # window once a batch had already tripped it.
                wait = min(5 * (2 ** attempt), 60)
                logfire.warning(f"Jina rate limit — retrying in {wait}s (attempt {attempt+1}/4).")
                time.sleep(wait)
            else:
                logfire.error(f"Jina embedding failed: {e}")
                raise
    raise RuntimeError("Jina rate limit persisted after 4 attempts.")

# ── Batch embedding with retry ─────────────────────────────────────────────────

def _embed_batch(batch: list[str]) -> list[list[float]]:
    """Always used for corpus/passage text (see embed_texts) — never for a
    live search query, which goes through embed_query instead."""
    if _model_type == "jina":
        return _embed_jina(batch, task=JINA_TASK_PASSAGE)
    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()


# ── Public API (same signatures as before) ─────────────────────────────────────

def embed_query(query: str) -> list[float]:
    _init()
    if _model_type == "jina":
        return _embed_jina([query], task=JINA_TASK_QUERY)[0]
    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    batch_starts = list(range(0, len(texts), BATCH_SIZE))
    for batch_num, i in enumerate(batch_starts):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
        # Pause between batches (not after the last one) to stay under
        # Jina's rate limit proactively, instead of relying entirely on
        # reactive backoff inside _embed_jina after a 429 already happened.
        is_last_batch = batch_num == len(batch_starts) - 1
        if _model_type == "jina" and not is_last_batch:
            time.sleep(INTER_BATCH_SLEEP_SECONDS)
    return all_embeddings