"""
RAG Agent Evaluation — LangSmith + Qwen judge
Binary LLM-as-judge evaluators (YES/NO per call) — stays well under Groq OTPM limits.
Results logged to LangSmith dashboard with shareable experiment link.
"""

import json, uuid, sys, os, re, time

# ── path setup MUST come before any src.* imports ─────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

# ── LangSmith env vars must be set before importing langsmith ──────────────────
os.environ["LANGCHAIN_API_KEY"]      = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"]   = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_PROJECT"]      = os.getenv("LANGCHAIN_PROJECT", "agentic-rag-evals")

_groq_key = os.getenv("GROQ_FALLBACK_API_KEY")
if not _groq_key:
    raise RuntimeError("GROQ_FALLBACK_API_KEY missing from .env")
if not os.environ["LANGCHAIN_API_KEY"]:
    raise RuntimeError("LANGCHAIN_API_KEY missing from .env")

from langsmith import Client
from langsmith.evaluation import evaluate
from groq import Groq
from src.agents.graph import rag_agent

# ── Qwen judge client ──────────────────────────────────────────────────────────
groq_client = Groq(api_key=_groq_key)
JUDGE_MODEL  = "qwen/qwen3.8-27b"
METRIC_PAUSE = 20   # seconds between evaluator calls — stays under OTPM


def call_judge(prompt: str) -> str:
    for attempt in range(4):
        try:
            resp = groq_client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "/no_think\nYou are a binary judge. Reply with only YES or NO. No thinking, no explanation."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content or ""
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
            raw = re.sub(r"<think>.*",          "", raw, flags=re.DOTALL)
            raw = raw.strip().upper()
            # fallback — if still empty, return NO
            return raw if raw else "NO"
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < 3:
                wait = 30 * (attempt + 1)
                print(f"\n  ⏳ Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


# ── RAG runner ────────────────────────────────────────────────────────────────

def run_rag(question: str):
    result = rag_agent.invoke(
        {
            "messages":     [{"role": "user", "content": question}],
            "current_query": "",
            "documents":    [],
            "plan":         [],
            "status":       "",
            "final_answer": "",
        },
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )
    answer  = result.get("final_answer", "")
    context = [d.replace("CONTENT: ", "", 1) for d in result.get("documents", [])]
    return answer, context


def rag_pipeline(inputs: dict) -> dict:
    """LangSmith target function — receives dataset example, returns outputs."""
    question = inputs["question"]
    answer, context = run_rag(question)
    return {
        "answer":  answer,
        "context": context,
    }


# ── Binary LLM-as-judge evaluators ───────────────────────────────────────────
# Each makes ONE Qwen call with max_tokens=10 (YES/NO only).
# This keeps output tokens well under Groq's 1000 OTPM free-tier limit.

def faithfulness(run, example) -> dict:
    """
    Is the answer grounded in the retrieved context?
    Score 1.0 = faithful, 0.0 = not faithful.
    """
    answer  = run.outputs.get("answer", "")
    context = "\n".join(run.outputs.get("context", []))

    if not answer or not context:
        return {"key": "faithfulness", "score": 0.0}

    prompt = f"""You are an evaluation judge. 

Context:
{context[:1500]}

Answer:
{answer[:800]}

Is every claim in the Answer supported by the Context above?
Reply with only YES or NO."""

    time.sleep(METRIC_PAUSE)
    verdict = call_judge(prompt)
    score   = 1.0 if "YES" in verdict else 0.0
    return {"key": "faithfulness", "score": score}


def answer_relevancy(run, example) -> dict:
    """
    Does the answer actually address the question asked?
    Score 1.0 = relevant, 0.0 = not relevant.
    """
    question = example.inputs.get("question", "")
    answer   = run.outputs.get("answer", "")

    if not answer:
        return {"key": "answer_relevancy", "score": 0.0}

    prompt = f"""You are an evaluation judge.

Question: {question}

Answer: {answer[:800]}

Does the Answer directly and completely address the Question?
Reply with only YES or NO."""

    time.sleep(METRIC_PAUSE)
    verdict = call_judge(prompt)
    score   = 1.0 if "YES" in verdict else 0.0
    return {"key": "answer_relevancy", "score": score}


def contextual_recall(run, example) -> dict:
    """
    Does the retrieved context contain the information needed to answer?
    Compares context against ground truth.
    Score 1.0 = context covers ground truth, 0.0 = does not.
    """
    ground_truth = example.outputs.get("ground_truth", "")
    context      = "\n".join(run.outputs.get("context", []))

    if not context or not ground_truth:
        return {"key": "contextual_recall", "score": 0.0}

    prompt = f"""You are an evaluation judge.

Retrieved Context:
{context[:1500]}

Expected Answer (ground truth):
{ground_truth[:600]}

Does the Retrieved Context contain enough information to derive the Expected Answer?
Reply with only YES or NO."""

    time.sleep(METRIC_PAUSE)
    verdict = call_judge(prompt)
    score   = 1.0 if "YES" in verdict else 0.0
    return {"key": "contextual_recall", "score": score}


def freshness(run, example) -> dict:
    """
    Freshness check: does the answer avoid hallucinating version numbers,
    dates, or recency claims not present in the retrieved context?
    Score 1.0 = no false recency claims, 0.0 = hallucinates recency.
    """
    answer  = run.outputs.get("answer", "")
    context = "\n".join(run.outputs.get("context", []))

    if not answer:
        return {"key": "freshness", "score": 1.0}  # empty answer doesn't hallucinate

    prompt = f"""You are an evaluation judge checking for hallucinated recency claims.

Context:
{context[:1500]}

Answer:
{answer[:800]}

Does the Answer make any specific version number, release date, or recency claims 
(e.g. "as of 2024", "the latest version is X") that are NOT supported by the Context?
Reply with only YES or NO."""

    time.sleep(METRIC_PAUSE)
    # YES means it hallucinated → score 0, NO means it didn't → score 1
    verdict = call_judge(prompt)
    score   = 0.0 if "YES" in verdict else 1.0
    return {"key": "freshness", "score": score}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    GOLDENS_PATH = os.path.join(os.path.dirname(__file__), "goldens.json")
    DATASET_NAME = "agentic-rag-goldens"

    with open(GOLDENS_PATH) as f:
        goldens = json.load(f)

    # ── Upload dataset to LangSmith (idempotent — skips if already exists) ────
    ls_client = Client()
    existing  = {d.name for d in ls_client.list_datasets()}

    if DATASET_NAME not in existing:
        print(f"📤 Uploading dataset '{DATASET_NAME}' to LangSmith...")
        dataset = ls_client.create_dataset(DATASET_NAME)
        ls_client.create_examples(
            inputs  = [{"question": g["question"]} for g in goldens],
            outputs = [{"ground_truth": g["ground_truth"]} for g in goldens],
            dataset_id = dataset.id,
        )
        print(f"  ✅ {len(goldens)} examples uploaded")
    else:
        print(f"✅ Dataset '{DATASET_NAME}' already exists in LangSmith — skipping upload")

    # ── Run evaluation ────────────────────────────────────────────────────────
    print(f"\n🚀 Starting evaluation run...")
    print(f"   Judge: {JUDGE_MODEL} via Groq")
    print(f"   Metrics: faithfulness, answer_relevancy, contextual_recall, freshness")
    print(f"   Pause between calls: {METRIC_PAUSE}s\n")

    results = evaluate(
        rag_pipeline,
        data        = DATASET_NAME,
        evaluators  = [faithfulness, answer_relevancy, contextual_recall, freshness],
        experiment_prefix = "qwen-judge",
        metadata    = {
            "judge_model":    JUDGE_MODEL,
            "rag_model":      "openai/gpt-oss-120b",
            "planner_model":  "qwen/qwen3.8-27b",
            "embedding_model": "jina-embeddings-v3",
            "reranker":       "flashrank",
            "vector_db":      "qdrant",
            "top_k":          15,
            "top_n":          3,
        },
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  AGGREGATE RESULTS")
    print(f"{'=' * 60}")

    scores: dict[str, list[float]] = {}
    for r in results._results:
        for fb in r.get("evaluation_results", {}).get("results", []):
            key   = fb.key
            score = fb.score
            if score is not None:
                scores.setdefault(key, []).append(score)

    for metric, vals in scores.items():
        avg    = round(sum(vals) / len(vals), 3)
        passed = sum(1 for v in vals if v >= 0.5)
        print(f"  {metric}: avg={avg}  pass={passed}/{len(vals)}")

    print(f"{'=' * 60}")
    print(f"\n🔗 View full results at: https://smith.langchain.com")
    print(f"   Project: {os.environ['LANGCHAIN_PROJECT']}\n")


if __name__ == "__main__":
    main()