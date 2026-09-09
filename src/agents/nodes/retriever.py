import logfire

from src.agents.state import AgentState
from src.retrieval.qdrant_service import search_enterprise_knowledge
from src.retrieval.ranking_service import rerank_documents


def retrieve_node(state: AgentState):
    query = state["current_query"]

    with logfire.span("🔍 Knowledge Retrieval"):
        logfire.info(f"Searching Qdrant for: {query}")
        results = search_enterprise_knowledge(query, limit=3)  # already reranked
        logfire.info(f"Retrieved {len(results)} reranked chunks")

        formatted_docs = [f"CONTENT: {doc['content']}" for doc in results]

    return {
        "documents": formatted_docs,
        "status": "Found technical context.",
        "plan": state["plan"] + ["Context Retrieved"]
    }