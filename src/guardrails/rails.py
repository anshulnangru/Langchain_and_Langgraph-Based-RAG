"""
Lightweight guardrails: same interface as the previous NeMo version
(initialize_rails() and guard(message) -> (fired, response)), but with no
local embedding model.

Order of checks, cheapest first:
  1. Regex rules: jailbreak attempts, "what model are you" questions
  2. Regex rules: greeting, farewell, capabilities
  3. Keyword fast-path: anything clearly about AI engineering passes
  4. Small Groq classifier for the remaining ambiguous messages
     (fails open: if the call fails, the message is allowed through)
"""
import os
import re

import logfire
from langchain_groq import ChatGroq

from src.config.config import settings

# ── Canned responses (same wording as the previous Colang rails) ─────────────
OFF_TOPIC_MSG = (
    "I'm an AI Engineering Assistant focused on LangChain, LangGraph, and agentic "
    "AI systems. I can't help with that — but ask me anything about building AI pipelines!"
)
JAILBREAK_MSG = (
    "I maintain consistent guidelines regardless of how I am prompted. I am here to "
    "help with LangChain, LangGraph, and agentic AI systems. What can I help you with?"
)
GREETING_MSG = (
    "Hello! I'm your AI Engineering Assistant. I specialise in LangChain, LangGraph, "
    "RAG pipelines, and agentic AI systems. What can I help you with today?"
)
CAPABILITIES_MSG = (
    "I'm an AI Engineering Assistant with expertise in: LangGraph (state machines, "
    "nodes, edges, memory, checkpointing), LangChain (chains, agents, tools, memory, "
    "integrations), RAG pipelines (retrieval, reranking, embeddings), and agentic AI "
    "systems. Ask me anything in these areas!"
)
FAREWELL_MSG = (
    "Goodbye! Feel free to return whenever you have more AI engineering questions. "
    "Have a great day!"
)

# ── Rules ────────────────────────────────────────────────────────────────────
_JAILBREAK = re.compile(
    r"ignore (all |any )?(your |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)"
    r"|disregard (all |any )?(your |the )?(training|instructions|rules|guidelines)"
    r"|forget (all |about )?(your |the )?(system )?(prompt|instructions|rules)"
    r"|you are now (dan|in developer mode|unrestricted|free)"
    r"|developer mode"
    r"|pretend (that )?you (have|are under) no (restrictions|rules|limits)"
    r"|your new instructions are"
    r"|act as (if you were trained differently|an? unrestricted)"
    r"|(override|bypass|disable|turn off) (all |any )?(your |the )?(safety|guidelines|filters|restrictions)"
    r"|(reveal|show|print|repeat) (me )?(your |the )?(system|hidden) prompt"
    r"|\bjailbreak\b|\bdan mode\b",
    re.IGNORECASE,
)

# "what model are you", "are you chatgpt", "what llm powers you"
_IDENTITY = re.compile(
    r"\b(what|which) (model|llm|ai|language model)\b[^?]*\b(you|your|powers?|behind)\b"
    r"|\bare you (chat ?gpt|gpt|openai|claude|gemini|llama|deepseek)\b"
    r"|\bwhat llm\b",
    re.IGNORECASE,
)

_GREETING = re.compile(
    r"^(hi+|hello+|hey+|yo|sup|howdy|hola|good (morning|afternoon|evening)|what'?s up)"
    r"( there| team| assistant| bot| everyone)?[\s!.,?]*$",
    re.IGNORECASE,
)

_FAREWELL = re.compile(
    r"^((thanks|thank you),? )?(bye+|goodbye|see you( later| soon)?|cya|that'?s (all|it)|that is all"
    r"|i'?m done|i am done|talk to you later)[\s!.]*$",
    re.IGNORECASE,
)

_CAPABILITIES = re.compile(
    r"^(help( me)?|what can you do|what do you know|what are you|what topics do you cover"
    r"|what can i ask( you)?|what are your capabilities|what are you capable of)[\s!.?]*$",
    re.IGNORECASE,
)

# Clearly on-topic: skip the classifier call entirely (saves latency and tokens).
_ON_TOPIC = re.compile(
    r"graph|lang\s?chain|langsmith|\brag\b|retriev|rerank|embedding|vector|qdrant"
    r"|agent|\bllms?\b|prompt|checkpoint|\bnodes?\b|\bedges?\b|pipeline|\btools?\b"
    r"|memory|groq|portkey|fastapi|python|\bapi\b|chunk|\btokens?\b|orchestrat"
    r"|workflow|guardrail|\bevals?\b|semantic|index|similarity|cross-encoder",
    re.IGNORECASE,
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)

_CLASSIFIER_PROMPT = """You are a topic filter for an AI engineering assistant.
The assistant answers questions about LangChain, LangGraph, RAG pipelines,
vector databases, embeddings, LLM applications and agentic AI systems.

Reply with exactly one word:
ALLOW     - the message is about those topics, about software or AI engineering
            in general, or is a short follow-up such as "explain more", "why?"
            or "give an example"
OFF_TOPIC - anything else (jokes, trivia, math homework, food, sport, weather,
            movies, general chit-chat unrelated to the above)

Message: "{message}"
Answer:"""

_llm: ChatGroq | None = None


def initialize_rails() -> None:
    """
    Build the small classifier client at app startup. No models are loaded
    locally. If this fails, guard() still works using the regex rules alone.
    """
    global _llm
    model = os.getenv("GUARD_MODEL", "openai/gpt-oss-20b")
    try:
        _llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=model,
            temperature=0,
            max_tokens=200,
            timeout=10,
            max_retries=1,
        )
        logfire.info(f"🛡️ Lightweight guardrails initialised (classifier: {model}).")
    except Exception as e:
        _llm = None
        logfire.warning(
            f"⚠️ Guardrail classifier unavailable, using rules only: {type(e).__name__}"
        )


def _classify(message: str) -> str:
    """Return 'OFF_TOPIC' or 'ALLOW'. Fails open on any error."""
    if _llm is None:
        return "ALLOW"
    try:
        resp = _llm.invoke(_CLASSIFIER_PROMPT.format(message=message[:500]))
        out = str(getattr(resp, "content", resp))
        out = _THINK_BLOCK.sub("", out)
        out = _THINK_OPEN.sub("", out).strip().upper()
        match = re.match(r"^\W*(ALLOW|OFF_TOPIC)\b", out)
        return match.group(1) if match else "ALLOW"
    except Exception as e:
        logfire.warning(f"⚠️ Guardrail classifier failed, allowing: {type(e).__name__}")
        return "ALLOW"


def guard(message: str) -> tuple[bool, str | None]:
    """Return (fired, response). fired=True means: answer with `response`, skip the pipeline."""
    text = re.sub(r"\s+", " ", (message or "").strip())

    with logfire.span("🛡️ Guardrails Check"):
        if not text:
            return True, CAPABILITIES_MSG

        if _JAILBREAK.search(text) or _IDENTITY.search(text):
            logfire.info(f"🛡️ Guardrails fired (jailbreak/identity) | query='{text[:80]}'")
            return True, JAILBREAK_MSG

        if _GREETING.match(text):
            return True, GREETING_MSG
        if _FAREWELL.match(text):
            return True, FAREWELL_MSG
        if _CAPABILITIES.match(text):
            return True, CAPABILITIES_MSG

        if _ON_TOPIC.search(text):
            logfire.info("✅ Guardrails passed (on-topic keywords).")
            return False, None

        if _classify(text) == "OFF_TOPIC":
            logfire.info(f"🛡️ Guardrails fired (off-topic) | query='{text[:80]}'")
            return True, OFF_TOPIC_MSG

        logfire.info("✅ Guardrails passed.")
        return False, None