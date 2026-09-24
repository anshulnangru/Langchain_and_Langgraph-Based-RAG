import json
import os
import sys
import uuid
import random
import argparse
from datetime import datetime, timezone

# ------------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------------

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

# ------------------------------------------------------------------
# ENVIRONMENT
# ------------------------------------------------------------------

from dotenv import load_dotenv

load_dotenv()

# Your src/config expects GROQ_API_KEY to exist before src imports.
# The freeze runner only needs the RAG agent, not the judge.
agent_key = os.getenv("GROQ_API_KEY_1")

if not agent_key:
    raise RuntimeError("GROQ_API_KEY_1 missing from .env")

os.environ["GROQ_API_KEY"] = agent_key

# ------------------------------------------------------------------
# IMPORT RAG AGENT
# ------------------------------------------------------------------

from src.agents.graph import rag_agent


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Run the RAG pipeline once for every golden question and freeze outputs."
)

parser.add_argument(
    "--goldens",
    default="goldens.json",
    help="Golden dataset inside evals/"
)

parser.add_argument(
    "--output",
    default="frozen_rag.json",
    help="Output file inside evals/"
)

parser.add_argument(
    "--seed",
    type=int,
    default=20260923,
    help="Random seed used to shuffle the golden dataset."
)

parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume from an existing output file, skipping questions already completed successfully."
)

args = parser.parse_args()


# ------------------------------------------------------------------
# RUN ONE RAG QUESTION
# ------------------------------------------------------------------

def run_rag(question: str) -> dict:
    thread_id = str(uuid.uuid4())

    initial_state = {
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ],
        "current_query": "",
        "documents": [],
        "plan": [],
        "status": "",
        "final_answer": "",
    }

    result = rag_agent.invoke(
        initial_state,
        config={
            "configurable": {
                "thread_id": thread_id
            }
        },
    )

    answer = result.get("final_answer", "")

    documents = result.get("documents", [])

    # Remove the display-only "CONTENT: " prefix.
    context = [
        doc.replace("CONTENT: ", "", 1)
        for doc in documents
    ]

    planner_query = result.get("current_query", "")

    plan = result.get("plan", [])

    status = result.get("status", "")

    return {
        "thread_id": thread_id,
        "planner_query": planner_query,
        "context": context,
        "answer": answer,
        "plan": plan,
        "status": status,
    }


# ------------------------------------------------------------------
# SAVE HELPERS
# ------------------------------------------------------------------

def build_metadata(seed: int, question_count: int) -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "question_count": question_count,

        "generator_model": "openai/gpt-oss-20b",
        "planner_model": "qwen/qwen3.8-27b",
        "embedding_model": "jina-embeddings-v3",
        "reranker": "flashrank",
        "vector_db": "qdrant",
    }


def save_progress(output_path: str, seed: int, question_count: int, frozen_results: list) -> None:
    """Write current progress to disk. Called after every question so a
    Ctrl+C, crash, or rate limit never loses more than one question's work."""

    output = {
        "metadata": build_metadata(seed, question_count),
        "results": frozen_results,
    }

    # Write to a temp file then rename, so a crash mid-write never
    # corrupts the previous good save.
    tmp_path = output_path + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    os.replace(tmp_path, output_path)


def load_existing(output_path: str) -> dict:
    """Load a previous (possibly partial) frozen_rag.json, if present."""

    if not os.path.exists(output_path):
        return {}

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Index by question index for quick lookup.
    by_index = {
        item["index"]: item
        for item in data.get("results", [])
    }

    return by_index


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():

    goldens_path = os.path.join(
        os.path.dirname(__file__),
        args.goldens
    )

    output_path = os.path.join(
        os.path.dirname(__file__),
        args.output
    )

    # --------------------------------------------------------------
    # LOAD GOLDENS
    # --------------------------------------------------------------

    with open(goldens_path, "r", encoding="utf-8") as f:
        goldens = json.load(f)

    if not isinstance(goldens, list):
        raise ValueError("Golden dataset must contain a JSON list.")

    if len(goldens) != 30:
        raise ValueError(
            f"Expected exactly 30 golden questions, found {len(goldens)}."
        )

    # --------------------------------------------------------------
    # SHUFFLE (deterministic seed -> same order every time)
    # --------------------------------------------------------------

    rng = random.Random(args.seed)
    rng.shuffle(goldens)

    # --------------------------------------------------------------
    # RESUME SUPPORT
    # --------------------------------------------------------------

    existing_by_index = {}

    if args.resume:
        existing_by_index = load_existing(output_path)
        if existing_by_index:
            done = sum(
                1 for item in existing_by_index.values()
                if item.get("status") != "RAG_ERROR" and item.get("answer")
            )
            print(f"Resuming: found {len(existing_by_index)} prior results "
                  f"({done} successful) in {output_path}")
        else:
            print("Resume requested but no prior output file found — starting fresh.")

    print("=" * 70)
    print("RAG FREEZE RUN")
    print("=" * 70)
    print(f"Questions : {len(goldens)}")
    print(f"Seed      : {args.seed}")
    print(f"Output    : {output_path}")
    print(f"Resume    : {args.resume}")
    print("=" * 70)

    # Pre-fill frozen_results with existing successful entries (by index),
    # placeholders elsewhere, so the list is always length 30 and ordered.
    frozen_results = [None] * len(goldens)

    for i in range(len(goldens)):
        idx = i + 1
        prior = existing_by_index.get(idx)
        if prior is not None and prior.get("status") != "RAG_ERROR" and prior.get("answer"):
            frozen_results[i] = prior

    # --------------------------------------------------------------
    # RUN RAG
    # --------------------------------------------------------------

    for i, golden in enumerate(goldens):
        index = i + 1

        # Skip questions already completed successfully on a prior run.
        if frozen_results[i] is not None:
            print(f"[{index}/30] (skipped, already completed) {golden['question']}")
            continue

        question = golden["question"]
        ground_truth = golden["ground_truth"]

        print()
        print(f"[{index}/30] {question}")

        try:
            rag_result = run_rag(question)

            frozen_item = {
                "index": index,
                "question": question,
                "ground_truth": ground_truth,

                "thread_id": rag_result["thread_id"],

                "planner_query": rag_result["planner_query"],

                "context": rag_result["context"],

                "answer": rag_result["answer"],

                "plan": rag_result["plan"],

                "status": rag_result["status"],
            }

            frozen_results[i] = frozen_item

            print(
                f"  Planner query : {rag_result['planner_query']!r}"
            )
            print(
                f"  Context chunks: {len(rag_result['context'])}"
            )
            print(
                f"  Answer chars  : {len(rag_result['answer'])}"
            )

        except Exception as e:

            print(f"  ❌ RAG failed: {e}")

            frozen_results[i] = {
                "index": index,
                "question": question,
                "ground_truth": ground_truth,

                "thread_id": None,

                "planner_query": None,

                "context": [],

                "answer": "",

                "plan": [],

                "status": "RAG_ERROR",

                "error": str(e),
            }

        # Save after every question — success or failure — so an
        # interruption (Ctrl+C, crash, rate limit) never costs more
        # than the question currently in flight.
        save_progress(output_path, args.seed, len(goldens), frozen_results)

    # --------------------------------------------------------------
    # FINAL SUMMARY (file is already fully saved by this point)
    # --------------------------------------------------------------

    successful = sum(
        1
        for item in frozen_results
        if item and item.get("answer")
    )

    print()
    print("=" * 70)
    print("FREEZE COMPLETE")
    print("=" * 70)
    print(f"Successful RAG runs: {successful}/{len(frozen_results)}")
    print(f"Saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()