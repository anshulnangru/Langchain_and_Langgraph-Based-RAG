from src.agents.state import AgentState
from src.gateways import get_langchain_llm
from src.config.config import settings
import logfire
import time

# Portkey-backed LLM: fallback + cache + retry — same .invoke() interface as ChatGroq
llm = get_langchain_llm(feature="planner")

def _invoke_with_retry(prompt: str, retries: int = 3, delay: float = 1.5):
    """Retries the LLM call on Groq's intermittent Harmony parse failures."""
    last_err = None
    for attempt in range(retries):
        try:
            return llm.invoke(
                prompt
            )
        except Exception as e:
            last_err = e
            logfire.warning(f"Planner LLM attempt {attempt + 1}/{retries} failed: {e}")
            time.sleep(delay)
    raise last_err


def planner_node(state: AgentState):
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
    You are an intelligent Assistant Planner. 
    Analyze the conversation history and the latest user message.
    
    CONVERSATION HISTORY:
    {history}
    
    LATEST MESSAGE:
    "{user_message}"
    
    Task:
    1. If the latest message is a greeting (hi, hello) or a question that can be answered using ONLY the conversation history above (e.g., "what is my name"), respond with 'CONVERSATIONAL'.
    2. If it is a technical question that requires fresh documentation, output a refined search query.
    
    Output ONLY 'CONVERSATIONAL' or the search query. Nothing else.
    """

    with logfire.span("🧠 Planner Decision"):
        decision = _invoke_with_retry(prompt).content.strip()
        logfire.info(f"Intent identified: {decision}")

    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }

    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"]
    }