import logfire
from src.agents.state import AgentState
from src.gateways import get_langchain_llm, extract_cache_status
import re

llm = get_langchain_llm(feature="rag")


def _invoke_with_retry(llm, prompt, retries=3, delay=1.5):
    import time
    last_err = None
    for attempt in range(retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            last_err = e
            time.sleep(delay)
    raise last_err


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    """
    query = state["current_query"]

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are RAG Assistant, a Senior Technical AI Assistant built on 
        an Agentic RAG pipeline using LangGraph. You are NOT ChatGPT and you answer vaguely when asked about you.

        Answer using ONLY the facts, definitions, and code shown in the 
        TECHNICAL CONTEXT below. Do not add explanations, definitions, 
        descriptions, or code usage patterns from your own general 
        knowledge of LangGraph, LangChain, or similar frameworks — even 
        if you are confident they are accurate. If the context does not 
        fully answer part of the question, say so explicitly (e.g. "the 
        provided context doesn't specify X") rather than filling the gap 
        yourself. It is better to give an incomplete but fully-grounded 
        answer than a complete answer that includes anything not 
        explicitly stated in the context below.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """
    else:
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        prompt = f"""
        You are RAG Assistant, a Senior Technical AI Assistant built on 
        an Agentic RAG pipeline using LangGraph. You are NOT ChatGPT and you answer vaguely when asked about you.
        Answer using ONLY the TECHNICAL CONTEXT provided below.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    with logfire.span("✍️ LLM Synthesis"):
        try:
            response = _invoke_with_retry(llm, prompt)
            content = response.content
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:
                logfire.info("⚡ Gateway Cache Hit — response served from Portkey cache.")
                plan_update = state["plan"] + ["Cache: Hit ⚡"]
                status = "Cache hit — instant response."
            else:
                logfire.info("✅ Response synthesised via LLM.")
                plan_update = state["plan"]
                status = "Response generated."

            return {
                "final_answer": content,
                "status": status,
                "plan": plan_update,
                "messages": [{"role": "assistant", "content": content}]
            }

        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise e