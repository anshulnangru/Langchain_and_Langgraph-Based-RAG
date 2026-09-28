"""
CONTEXTUAL RECALL eval — completely isolated.

Receives ONLY: question, ground truth, retrieved context.
Does NOT receive: the generated answer, or any other metric's verdict.

This deliberately excludes the generated answer so a well-written answer
can never mask poor retrieval, and a badly-written answer can never drag
down a recall score that should be about the CONTEXT, not the generation.

Run:
    python -m evals.eval_contextual_recall
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evals.eval_common import (
    load_frozen, call_judge, parse_verdict, base_arg_parser,
    save_progress, load_existing_verdicts, print_summary, RateLimitExhausted,
    log_judge_run,
)

SYSTEM_PROMPT = """You are a strict but fair evaluator judging CONTEXTUAL RECALL only.

CONTEXTUAL RECALL asks: does the retrieved context contain enough
information to derive the expected (ground truth) answer? This is a
retrieval-quality check, not a generation-quality check.

You are given ONLY the question, the ground truth (expected) answer, and
the retrieved context. You are NOT given the actual generated answer, and
you must NOT judge anything about how a response was written — only
whether the necessary facts are present somewhere in the retrieved context.

PASS if:
- the retrieved context contains enough information that a well-informed
  reader could construct the ground truth answer from it, even if the
  information is spread across multiple context chunks

FAIL if:
- the retrieved context is missing key facts needed for the ground truth
  answer, is off-topic, or only tangentially related

Respond with ONLY a JSON object, no other text, no markdown fences:
{"score": 0 or 1, "reasoning": "one or two sentences explaining the verdict"}
"""


def build_user_prompt(item: dict) -> str:
    context_block = "\n\n---\n\n".join(item.get("context", []))
    return (
        f"QUESTION:\n{item['question']}\n\n"
        f"GROUND TRUTH (expected answer):\n{item['ground_truth']}\n\n"
        f"RETRIEVED CONTEXT:\n{context_block}"
    )


def main():
    parser = base_arg_parser(default_output="evals/results_contextual_recall.json")
    args = parser.parse_args()

    api_key = os.getenv(args.key_env)
    if not api_key:
        raise RuntimeError(f"{args.key_env} missing from .env")

    items = load_frozen(args.input)

    existing = {}
    if args.resume:
        existing = load_existing_verdicts(args.output)
        if existing:
            print(f"Resuming: {len(existing)} question(s) already scored in {args.output}")

    print("=" * 70)
    print("CONTEXTUAL RECALL EVAL")
    print("=" * 70)
    print(f"Input    : {args.input}")
    print(f"Output   : {args.output}")
    print(f"Judge    : {args.model} (key: {args.key_env})")
    print(f"Questions: {len(items)}")
    print(f"Resume   : {args.resume}")
    print("=" * 70)

    per_question = [None] * len(items)
    for i, item in enumerate(items):
        prior = existing.get(item["index"])
        if prior is not None:
            per_question[i] = prior

    for i, item in enumerate(items):
        if per_question[i] is not None:
            print(f"[{i + 1}/{len(items)}] (skipped, already scored) {item['question']}")
            continue

        print(f"[{i + 1}/{len(items)}] {item['question']}")

        if not item.get("context"):
            print("  skipped (no context retrieved)")
            per_question[i] = {
                "index": item["index"],
                "question": item["question"],
                "score": 0,
                "reasoning": "",
                "parse_error": "no context present in frozen data",
            }
            save_progress(args.output, "contextual_recall", args.model, per_question)
            continue

        try:
            user_prompt = build_user_prompt(item)
            raw = call_judge(SYSTEM_PROMPT, user_prompt, api_key, model=args.model)
            verdict = parse_verdict(raw)
            verdict["index"] = item["index"]
            verdict["question"] = item["question"]
            per_question[i] = verdict

            print(f"  score={verdict['score']}  {verdict.get('reasoning', '')[:80]}")

            if args.langsmith:
                log_judge_run(
                    "contextual_recall", args.langsmith_project, item, verdict,
                    inputs={"question": item["question"], "ground_truth": item["ground_truth"],
                            "context": item.get("context", [])},
                    model=args.model,
                )

        except RateLimitExhausted as e:
            print(f"\n⚠️  {e}")
            per_question[i] = {
                "index": item["index"],
                "question": item["question"],
                "score": 0,
                "reasoning": "",
                "parse_error": "rate limit exhausted, not yet scored",
            }
            save_progress(args.output, "contextual_recall", args.model,
                          [q for q in per_question if q is not None])
            print(f"Progress saved to {args.output}. Re-run with --resume "
                  f"(optionally a different --key-env) to continue.")
            sys.exit(1)

        save_progress(args.output, "contextual_recall", args.model,
                      [q for q in per_question if q is not None])

        if i < len(items) - 1:
            time.sleep(args.pause)

    print_summary("contextual_recall", args.model, per_question)
    print(f"Saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()