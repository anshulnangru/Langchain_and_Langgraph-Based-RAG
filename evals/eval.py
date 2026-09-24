"""
RAG Agent Evaluation — LangSmith + Qwen judge
Binary LLM-as-judge evaluators (YES/NO per call) — stays well under Groq OTPM limits.
Results logged to LangSmith dashboard with shareable experiment link.

[... existing docstring/FIX comments unchanged ...]
"""

import json, uuid, sys, os, re, time, argparse

# ── path setup MUST come before any src.* imports ─────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

# ── Parse CLI args BEFORE any src.* import, since config.py reads env vars
# at import time — GROQ_API_KEY must be set before
# `from src.agents.graph import rag_agent` runs below. ─────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--goldens", default="goldens.json",
                     help="Goldens file under evals/, e.g. goldens_original.json")
parser.add_argument("--dataset-suffix", default="",
                     help="Appended to dataset name + experiment prefix, e.g. -extended")
parser.add_argument("--key-pair", type=int, choices=[1, 2], default=1,
                     help="1 = GROQ_API_KEY_1 (agent) + GROQ_API_KEY_2 (judge). "
                          "2 = GROQ_API_KEY_3 (agent) + GROQ_API_KEY_4 (judge).")
args = parser.parse_args()

if args.key_pair == 1:
    _agent_key_name, _judge_key_name = "GROQ_API_KEY_1", "GROQ_API_KEY_2"
else:
    _agent_key_name, _judge_key_name = "GROQ_API_KEY_3", "GROQ_API_KEY_4"

_agent_key = os.getenv(_agent_key_name)
_judge_key = os.getenv(_judge_key_name)

if not _agent_key:
    raise RuntimeError(f"{_agent_key_name} missing from .env")
if not _judge_key:
    raise RuntimeError(f"{_judge_key_name} missing from .env")

# Assign into the exact env var name config.py already expects for the
# agent (GROQ_API_KEY) — everything downstream (planner.py, responder.py,
# via get_langchain_llm -> settings.GROQ_API_KEY) picks this up with zero
# changes to config.py or the node files. GROQ_FALLBACK_API_KEY here is
# purely an internal name used later in this file for the judge client —
# it does not need to exist in .env.
os.environ["GROQ_API_KEY"] = _agent_key
os.environ["GROQ_FALLBACK_API_KEY"] = _judge_key

print(f"🔑 Using key pair {args.key_pair}: agent={_agent_key_name}, judge={_judge_key_name}")

# ── LangSmith env vars must be set before importing langsmith ──────────────────
os.environ["LANGCHAIN_API_KEY"]      = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"]   = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_PROJECT"]      = os.getenv("LANGCHAIN_PROJECT", "agentic-rag-evals")

_groq_key = os.environ["GROQ_FALLBACK_API_KEY"]
if not os.environ["LANGCHAIN_API_KEY"]:
    raise RuntimeError("LANGCHAIN_API_KEY missing from .env")

from langsmith import Client
from langsmith.evaluation import evaluate
from groq import Groq
from src.agents.graph import rag_agent   # imported AFTER env vars are set above

# ── Qwen judge client ──────────────────────────────────────────────────────────
groq_client = Groq(api_key=_groq_key)
JUDGE_MODEL  = "qwen/qwen3.8-27b"
METRIC_PAUSE = 20   # seconds between evaluator calls — stays under OTPM


def call_judge(prompt: str, system_extra: str = "", max_tokens: int = 250) -> str:
    """[unchanged from your current file]"""
    system_content = system_extra or (
        "/no_think\n"
        "You are a careful evaluation judge. Judge only the specific "
        "question asked in the user message — do not apply outside "
        "assumptions about what YES or NO should mean beyond what the "
        "question states, since different questions use YES/NO for "
        "different things.\n\n"
        "First, in one short sentence, explain your reasoning. "
        "Then, on a new final line by itself, write exactly one word: "
        "YES or NO."
    )

    for attempt in range(4):
        try:
            resp = groq_client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
            raw = resp.choices[0].message.content or ""
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
            raw = re.sub(r"<think>.*",          "", raw, flags=re.DOTALL)
            return raw.strip()
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < 3:
                wait = 30 * (attempt + 1)
                print(f"\n  ⏳ Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    return ""


def _parse_verdict(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    last_line = lines[-1].upper() if lines else ""

    if "YES" in last_line and "NO" not in last_line:
        return "YES"
    if "NO" in last_line:
        return "NO"

    whole = text.upper()
    if re.search(r"\bYES\b", whole) and not re.search(r"\bNO\b", whole):
        return "YES"
    return "NO"


def call_judge_single(prompt: str) -> str:
    raw = call_judge(prompt, max_tokens=250)
    return _parse_verdict(raw)


def call_judge_multi(prompt: str, labels: list[str]) -> dict[str, str]:
    format_lines = "\n".join(f"{label.upper()}: YES or NO" for label in labels)

    system_content = (
        "/no_think\n"
        "You are a careful evaluation judge answering multiple related "
        "questions about the SAME context and answer. Judge each question "
        "independently — do not let your verdict on one question influence "
        "another, since they ask different things.\n\n"
        "FIRST, on the very first lines of your reply, output exactly one "
        "verdict line per question in this exact format (one per line, "
        "nothing before them):\n"
        f"{format_lines}\n\n"
        "THEN, after all verdict lines, add a brief justification — at "
        "most one short sentence per question. Keep it tight; the verdicts "
        "above are what get scored, the justification is just for a human "
        "reviewer skimming later."
    )

    raw = call_judge(prompt, system_extra=system_content, max_tokens=700)

    results: dict[str, str] = {}
    for label in labels:
        pattern = re.compile(rf"{re.escape(label.upper())}\s*:\s*(YES|NO)", re.IGNORECASE)
        matches = pattern.findall(raw)
        if matches:
            results[label] = matches[0].upper()
        else:
            print(f"  ⚠️  call_judge_multi: could not find '{label}' verdict in judge output — defaulting to NO.\n     Raw tail: {raw[-200:]!r}")
            results[label] = "NO"
    return results


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
    question = inputs["question"]
    answer, context = run_rag(question)

    if not context:
        print(f"  ⚠️  WARNING: empty context for question: {question!r}")
    if not answer:
        print(f"  ⚠️  WARNING: empty answer for question: {question!r}")

    return {
        "answer":  answer,
        "context": context,
    }


def context_metrics(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    question     = inputs.get("question", "")
    answer       = outputs.get("answer", "")
    context      = "\n".join(outputs.get("context", []))
    ground_truth = reference_outputs.get("ground_truth", "")

    results = []

    if not answer or not context:
        print(f"  ⚠️  context_metrics: skipping judge call — answer_empty={not answer} context_empty={not context}")
        results.append({"key": "faithfulness", "score": 0.0})
        results.append({"key": "contextual_recall", "score": 0.0})
        results.append({"key": "freshness", "score": 1.0 if not answer else 0.0})
        return {"results": results}

    prompt = f"""You are evaluating a RAG system's answer against its retrieved context and
the expected (ground-truth) answer. Answer THREE separate questions below about
the SAME Context/Answer pair. Judge each independently.

Context:
{context[:4000]}

Question asked: {question}

Answer given: {answer[:1200]}

Expected Answer (ground truth): {ground_truth[:600] if ground_truth else "(not provided)"}

1. FAITHFULNESS — Is the Answer grounded in the Context? A claim counts as
   supported if it is stated in the Context, is a reasonable paraphrase or
   summary of it, is a direct inference from it, OR is a reorganization,
   categorization, or synthesis of multiple facts already present in the
   Context (e.g. grouping several context sentences into a table, a
   numbered list, or under new descriptive headings does not make those
   claims unsupported — the underlying facts are still traceable to the
   Context). General domain terminology used to label or introduce a
   concept the Context describes (even if that exact label doesn't appear
   verbatim in the Context) is NOT grounds for NO, provided the concept
   itself is accurately described using Context content.
   Answer NO only if the Answer states a concrete fact, number, name, or
   claim that contradicts the Context, or that has no reasonable basis in
   it at all — not merely because the Answer is more organized,
   better-formatted, or uses different phrasing than the Context.

   Answer NO only if the Answer states a concrete fact, number, name, or
   claim that contradicts the Context, or that has no reasonable basis in
   it at all — not merely because the Answer is more organized,
   better-formatted, or uses different phrasing than the Context.
   (This leniency applies ONLY to FAITHFULNESS — apply RECALL and
   FRESHNESS using their own criteria below, independently and without
   extending this same leniency to them.)
   
2. RECALL — Does the Context contain enough information to derive the core of
   the Expected Answer? It does not need every minor detail or verbatim wording
   — reasonable inference from what's given counts as sufficient. If no
   Expected Answer was provided, answer NO.

3. FRESHNESS — Does the Answer make any specific version number, release date,
   or recency claim (e.g. "as of 2024", "the latest version is X") that is NOT
   supported by the Context? Answer YES only if such an unsupported claim IS
   present — answer NO if the Answer makes no such claims, or if any such
   claims it makes ARE supported by the Context."""

    time.sleep(METRIC_PAUSE)
    verdicts = call_judge_multi(prompt, ["FAITHFULNESS", "RECALL", "FRESHNESS"])

    results.append({"key": "faithfulness", "score": 1.0 if verdicts["FAITHFULNESS"] == "YES" else 0.0})
    results.append({"key": "contextual_recall", "score": 1.0 if verdicts["RECALL"] == "YES" else 0.0})
    results.append({"key": "freshness", "score": 0.0 if verdicts["FRESHNESS"] == "YES" else 1.0})

    return {"results": results}


def answer_relevancy(inputs: dict, outputs: dict) -> dict:
    question = inputs.get("question", "")
    answer   = outputs.get("answer", "")

    if not answer:
        print("  ⚠️  answer_relevancy: skipping judge call — answer_empty=True")
        return {"key": "answer_relevancy", "score": 0.0}

    prompt = f"""You are judging answer relevancy: whether the Answer directly and
substantially addresses the Question asked, regardless of whether it is
factually correct or complete in every detail.

Question: {question}

Answer: {answer[:800]}

Does the Answer directly and substantially address the Question?"""

    time.sleep(METRIC_PAUSE)
    verdict = call_judge_single(prompt)
    score   = 1.0 if verdict == "YES" else 0.0
    return {"key": "answer_relevancy", "score": score}


def main():
    GOLDENS_PATH = os.path.join(os.path.dirname(__file__), args.goldens)
    DATASET_NAME = "agentic-rag-goldens" + args.dataset_suffix

    with open(GOLDENS_PATH) as f:
        goldens = json.load(f)

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

    print(f"\n🚀 Starting evaluation run...")
    print(f"   Goldens: {args.goldens}")
    print(f"   Judge: {JUDGE_MODEL} via Groq")
    print(f"   Metrics: faithfulness, contextual_recall, freshness (merged call) + answer_relevancy (separate)")
    print(f"   Pause between calls: {METRIC_PAUSE}s\n")

    results = evaluate(
        rag_pipeline,
        data        = DATASET_NAME,
        evaluators  = [context_metrics, answer_relevancy],
        experiment_prefix = "qwen-judge-clean" + args.dataset_suffix,
        metadata    = {
            "judge_model":    JUDGE_MODEL,
            "rag_model":      "openai/gpt-oss-120b",
            "planner_model":  "qwen/qwen3.8-27b",
            "embedding_model": "jina-embeddings-v3",
            "reranker":       "flashrank",
            "vector_db":      "qdrant",
            "top_k":          15,
            "top_n":          6,
            "key_pair":       args.key_pair,
        },
    )

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