import time
import logfire
from flashrank import Ranker, RerankRequest

# Lazy initialization - Ranker is loaded on first use to ensure logfire.configure() has run
_ranker = None

# FIX: the model actually being loaded here was ms-marco-TinyBERT-L-2-v2 —
# NOT ms-marco-MiniLM-L-6-v2 as the old docstring claimed. Ranker(cache_dir=...)
# with no model_name argument silently defaults to FlashRank's smallest
# "Nano" model (TinyBERT, ~4MB). The logfire message a few lines down
# ("Initializing FlashRank Model (TinyBERT)...") was actually accurate; the
# docstring comment above it was not, which is how this went unnoticed.
#
# Traced directly via diagnose_recall.py + a raw top-30 Qdrant query: for a
# conditional-routing question, the chunk containing add_conditional_edges
# ranked #1 and #2 by raw cosine similarity (well within fetch_k=15), but
# TinyBERT's rerank pushed BOTH out of the final top-6 entirely, replacing
# them with lower-raw-score, more generic content. A 2-layer cross-encoder
# is too weak to reliably outperform raw similarity ordering on this corpus.
#
# ms-marco-MiniLM-L-12-v2 is FlashRank's own documented "best cross-encoder
# reranker" (~34MB, still fully local/CPU/free — no API key, no cost, same
# one-time-download-then-cached behavior as the old model).
RERANK_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"


def _get_ranker() -> Ranker:
    """
    Initializes the FlashRank engine lazily.
    FlashRank uses a local ONNX cross-encoder model for reranking — no
    external API calls, no cost, fully CPU-based.
    """
    global _ranker
    if _ranker is None:
        logfire.info(f"🧠 Initializing FlashRank Model ({RERANK_MODEL_NAME}) locally...")
        try:
            # We use a specific cache directory to avoid permission issues in production
            _ranker = Ranker(model_name=RERANK_MODEL_NAME, cache_dir="/tmp/flashrank")
        except Exception:
            _ranker = Ranker(model_name=RERANK_MODEL_NAME)
    return _ranker


def rerank_documents(query: str, documents: list[str], top_n: int = 5, return_scores: bool = False):
    """
    Refines retrieval results by re-scoring documents against the query semantically.

    Why FlashRank?
    Standard vector search (Cosine Similarity) is fast but mathematically "fuzzy."
    FlashRank uses a Cross-Encoder approach which is much more precise but usually slow.
    FlashRank solves this by using highly optimized, quantized ONNX models locally.

    return_scores: when False (default, backward-compatible with existing
    callers), returns list[str] of reranked text only. When True, returns
    list[tuple[str, float]] of (text, rerank_score) pairs — useful for
    callers that want to sanity-check the reranker's confidence rather than
    trust reordering blindly (e.g. flagging when the top result's score is
    barely above the rest, or blending against raw cosine score).
    """
    if not documents:
        return []

    start_time = time.time()
    logfire.info(f"📡 [Reranker] Sending {len(documents)} docs to FlashRank Cross-Encoder ({RERANK_MODEL_NAME})...")

    try:
        ranker = _get_ranker()

        # FlashRank expects a list of dictionaries with 'id' and 'text'
        passages = [
            {"id": i, "text": doc}
            for i, doc in enumerate(documents)
        ]

        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)

        # Results are returned sorted by highest semantic score first
        top_results = results[:top_n]

        duration = time.time() - start_time
        top_score = results[0]['score'] if results else 'N/A'
        logfire.info(f"✅ [Reranker] Done in {duration:.2f}s. Top semantic score: {top_score}")

        if return_scores:
            return [(res['text'], res['score']) for res in top_results]
        return [res['text'] for res in top_results]

    except Exception as e:
        # FIX: logfire.error() is a silent no-op unless logfire.configure()
        # was called — print() guarantees visibility regardless. This is
        # the exception handler most likely to explain "clean empty
        # context with zero error trace" if FlashRank's ONNX runtime hits
        # an intermittent issue (e.g. a transient file-lock on the cached
        # model, or a malformed passage).
        import traceback
        print(f"\n❌ [Reranker] Failed for query={query!r} on {len(documents)} docs: {e}")
        traceback.print_exc()
        logfire.error(f"❌ [Reranker] Semantic Reranking Failed: {e}")
        # Fallback to the original Qdrant order to ensure the user still gets an answer
        fallback = documents[:top_n]
        if return_scores:
            return [(doc, None) for doc in fallback]
        return fallback