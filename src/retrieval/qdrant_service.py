import os
import time
import logfire
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.config.config import settings
from src.retrieval.embeddings import embed_query
from src.retrieval.ranking_service import rerank_documents

client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY
)

# ── Toggle: set QDRANT_CLEAN_ONLY=true in .env to filter noise during eval ────
# Default is False — production behavior is unchanged.
_CLEAN_ONLY = os.getenv("QDRANT_CLEAN_ONLY", "false").lower() == "true"

# ── FIX: per-source diversity cap ──────────────────────────────────────────
# diagnose_recall.py's MAGNET CHECK showed a small handful of broad/generic
# chunks (docs_..._langgraph_overview_md.md, ..._graph-api_md.md,
# ..._checkpointers_md.md) appearing in the production top-6 for 11-17 out
# of 15 golden questions — across completely unrelated topics (streaming,
# subgraphs, conditional routing, the functional API, error handling).
# These chunks pack in broad domain vocabulary that makes them embed near
# the center of the whole topic space, so they score decently against
# almost ANY question in this corpus on raw similarity/rerank score alone,
# even when they don't actually contain what's being asked. Left unchecked,
# a handful of "hub" chunks structurally crowd out more specific, genuinely
# relevant chunks for other sources — this was traced directly to a
# contextual_recall=0 / faithfulness=0 failure on a conditional-routing
# question where the "Core benefits" overview chunk reached the generator
# instead of the actual routing documentation.
#
# Fix: after reranking, cap how many of the final top_n slots any single
# source file can occupy. Excess picks from an over-represented source are
# skipped in favor of the next-best-ranked chunk from a source that hasn't
# hit the cap yet. This is applied AFTER rerank (which still determines
# ordering/quality) — it only prevents one source from saturating the
# limited number of final slots.
MAX_CHUNKS_PER_SOURCE_IN_RESULT = int(os.getenv("MAX_CHUNKS_PER_SOURCE_IN_RESULT", "2"))


def _query_with_retry(query_vector, query_filter, fetch_k, attempts: int = 5, delay: float = 2.0):
    """
    FIX: root cause of intermittent empty-context bugs — Qdrant queries were
    failing with transient DNS errors ('No address associated with
    hostname' / 'Name or service not known') with zero retry, same failure
    class already seen and retried-for in the ingestion pipeline
    (processor.py) but never carried over to this query-time path. A single
    blip here silently zeroed out retrieval for whatever question happened
    to be running at that moment.

    Bumped to 5 attempts with exponential backoff (2s, 4s, 8s, 16s) after
    observing the DNS issue is bursty — sometimes 2-3 consecutive failures
    before the resolver recovers, not just isolated single blips.
    """
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            return client.query_points(
                collection_name=settings.QDRANT_COLLECTION,
                query=query_vector,
                limit=fetch_k,
                with_payload=True,
                query_filter=query_filter if query_filter else None,
            )
        except Exception as e:
            last_err = e
            msg = str(e)
            transient = (
                "No address associated" in msg
                or "Name or service not known" in msg
                or "Temporary failure in name resolution" in msg
                or "Connection" in msg
            )
            if not transient or attempt == attempts:
                raise
            wait = delay * (2 ** (attempt - 1))
            print(f"\n⏳ Qdrant query hit a transient network error "
                  f"(attempt {attempt}/{attempts}): {msg} — retrying in {wait}s")
            time.sleep(wait)
    raise last_err


def _apply_source_diversity_cap(reranked_texts: list[str], source_map: dict[str, str], limit: int) -> list[str]:
    """
    Walk the reranked list in order (best-first) and keep a chunk only if
    its source hasn't already hit MAX_CHUNKS_PER_SOURCE_IN_RESULT. Skipped
    chunks are dropped, not reordered — a lower-ranked chunk from an
    under-represented source takes the freed slot instead. If the capped
    result ends up shorter than `limit` (e.g. very few distinct sources in
    the candidate pool), it's fine to return fewer than `limit` — that's
    still better than one source saturating every slot.
    """
    per_source_count: dict[str, int] = {}
    kept: list[str] = []

    for text in reranked_texts:
        if len(kept) >= limit:
            break
        source = source_map.get(text, "Unknown")
        count = per_source_count.get(source, 0)
        if count >= MAX_CHUNKS_PER_SOURCE_IN_RESULT:
            continue
        kept.append(text)
        per_source_count[source] = count + 1

    return kept


def search_enterprise_knowledge(query: str, limit: int = 3, fetch_k: int = 15):
    """
    Retrieves fetch_k candidates via vector search, then reranks down to `limit`
    using FlashRank's cross-encoder for higher precision, then applies a
    per-source diversity cap so a small number of broad/generic "hub" chunks
    can't structurally crowd out more specific, relevant chunks from other
    sources (see MAX_CHUNKS_PER_SOURCE_IN_RESULT above for why this exists).

    If QDRANT_CLEAN_ONLY=true in .env, restricts search to vectors with
    label='true' (LangGraph/LangChain docs only — no noise).
    Default (QDRANT_CLEAN_ONLY=false) searches the full corpus as before.
    """
    try:
        query_vector = embed_query(query)

        # ── Build optional noise filter ───────────────────────────────────────
        query_filter = None
        if _CLEAN_ONLY:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="label",
                        match=MatchValue(value="true")
                    )
                ]
            )
            logfire.info("🔍 Clean-only mode: filtering to true-label vectors only.")

        response = _query_with_retry(query_vector, query_filter, fetch_k)

        candidates = []
        source_map = {}
        for res in response.points:
            text = res.payload.get("text", "")
            source = res.payload.get("source", "Unknown")
            candidates.append(text)
            source_map[text] = source

        if not candidates:
            # FIX: this used to return [] with zero visibility. Print
            # directly since logfire is a no-op unless logfire.configure()
            # was called (it wasn't, during eval runs) — this print is
            # guaranteed to show up regardless of logfire's config state.
            print(f"\n⚠️  Qdrant returned 0 candidates for query={query!r} "
                  f"(clean_only={_CLEAN_ONLY}, fetch_k={fetch_k})")
            return []

        # Rerank against a larger pool than `limit` so the diversity cap
        # below has real alternatives to pick from instead of just
        # whatever the top `limit` happened to already be.
        rerank_pool_size = max(limit * 3, limit + 5)
        reranked_texts = rerank_documents(query, candidates, top_n=min(rerank_pool_size, len(candidates)))

        if not reranked_texts:
            print(f"\n⚠️  Reranker returned 0 results for query={query!r} "
                  f"out of {len(candidates)} candidates from Qdrant")
            return []

        diversified_texts = _apply_source_diversity_cap(reranked_texts, source_map, limit)

        results = []
        for text in diversified_texts:
            results.append({
                "content": text,
                "source": source_map.get(text, "Unknown"),
            })

        return results
    except Exception as e:
        # FIX: logfire.error() alone is a silent no-op whenever
        # logfire.configure() hasn't been called (e.g. during eval runs) —
        # this is why retrieval failures here have been invisible. print()
        # is a guaranteed fallback regardless of logfire's config state.
        import traceback
        print(f"\n❌ Qdrant Search Failed for query={query!r}: {e}")
        traceback.print_exc()
        logfire.error(f"❌ Qdrant Search Failed: {e}")
        return []