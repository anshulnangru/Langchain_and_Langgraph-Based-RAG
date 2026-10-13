from src.config.config import settings as s
from src.gateways.client import _portkey

print("Portkey key set:", bool(s.PORTKEY_API_KEY))
print("Configs:", s.PORTKEY_CONFIG_GENERATOR, s.PORTKEY_CONFIG_PLANNER)

for feature in ("rag", "planner"):
    try:
        llm = _portkey(feature, s.GROQ_MODEL if feature == "rag" else s.GROQ_MODEL_PLANNER, 50)
        r = llm.invoke("Reply with the single word: pong")
        print(f"[{feature}] OK ->", r.content[:60], "| model:", r.response_metadata.get("model_name"))
    except Exception as e:
        print(f"[{feature}] FAILED ->", type(e).__name__, str(e)[:300])
