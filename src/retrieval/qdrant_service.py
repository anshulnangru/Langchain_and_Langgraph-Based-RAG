import logfire
from qdrant_client import QdrantClient
from src.config.config import settings
from src.retrieval.embeddings import embed_query
from src.retrieval.ranking_service import rerank_documents

client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY
)

def search_enterprise_knowledge(query: str, limit: int = 5, fetch_k: int = 15):
    """
    Retrieves fetch_k candidates via vector search, then reranks down to `limit`
    using FlashRank's cross-encoder for higher precision.
    """
    try:
        query_vector = embed_query(query)

        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            limit=fetch_k,
            with_payload=True
        )

        candidates = []
        source_map = {}
        for res in response.points:
            text = res.payload.get("text", "")
            source = res.payload.get("source", "Unknown")
            candidates.append(text)
            source_map[text] = source

        if not candidates:
            return []

        reranked_texts = rerank_documents(query, candidates, top_n=limit)

        results = []
        for text in reranked_texts:
            results.append({
                "content": text,
                "source": source_map.get(text, "Unknown"),
            })

        return results
    except Exception as e:
        logfire.error(f"❌ Qdrant Search Failed: {e}")
        return []