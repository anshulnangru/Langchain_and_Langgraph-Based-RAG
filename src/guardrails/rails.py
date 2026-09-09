import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from src.config.config import settings
from src.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


_rails: LLMRails | None = None


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses llama-3.1-8b-instant for fast intent classification at the gate —
    the heavier llama-3.3-70b-versatile is reserved for the RAG pipeline.
    """
    global _rails

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=150
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info("🛡️ NeMo Guardrails initialised (openai/gpt-oss-20b).")
    
    


def guard(message: str) -> tuple[bool, str | None]:
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        result = _rails.generate(
            messages=[
                {
                    "role": "user",
                    "content": message
                }
            ]
        )

        import re

        # Extract the model response exactly once
        content = (
            result.get("content", "")
            if isinstance(result, dict)
            else str(result)
        )

        # Remove complete <think>...</think> blocks
        content = re.sub(
            r"<think>.*?</think>",
            "",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Handle a truncated/unclosed <think> block
        content = re.sub(
            r"<think>.*$",
            "",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )

        content = content.strip()

        fired = any(
            indicator in content
            for indicator in RAIL_INDICATORS
        )

        if fired:
            logfire.info(
                f"🛡️ Guardrails fired | query='{message[:80]}'"
            )
            return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None