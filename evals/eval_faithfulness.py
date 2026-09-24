"""
FAITHFULNESS eval — completely isolated.

Receives ONLY: question, answer, retrieved context.
Does NOT receive: ground truth, or any other metric's verdict.

Run:
    python -m evals.eval_faithfulness
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evals.eval_common import (
    load_frozen, call_judge, parse_verdict, base_arg_parser,
    save_progress, load_existing_verdicts, print_summary, RateLimitExhausted,
)

SYSTEM_PROMPT = """You are a strict but fair evaluator judging FAITHFULNESS only.

FAITHFULNESS asks: is the answer fully supported by the retrieved context?
An answer is faithful if every factual claim it makes can be traced back to
the retrieved context. It is fine — and must NOT be penalized — if the
answer reorganizes, synthesizes, or reformats the context (e.g. turning
prose into a table or list), as long as it does not introduce claims the
context does not support.

FAIL faithfulness if the answer:
- states something the context does not support (fabrication/hallucination)
- contradicts the context
- adds specific facts, numbers, or claims not present in the context

PASS faithfulness if:
- every claim is grounded in the context, even if reworded, reorganized,
  or synthesized across multiple context chunks

You are given ONLY the question, the generated answer, and the retrieved
context. You do NOT have a ground truth answer, and you must NOT judge
whether the answer is complete, well-written, or relevant to the question
— only whether it is faithful to the given context.

Respond with ONLY a JSON object, no other text, no markdown fences:
{"score": 0 or 1, "reasoning": "one or two sentences explaining the verdict"}
"""


def build_user_prompt(item: dict) -> str:
    context_block = "\n\n---\n\n".join(item.get("context", []))
    return (
        f"QUESTION:\n{item['question']}\n\n"
        f"ANSWER:\n{item['answer']}\n\n"
        f"RETRIEVED CONTEXT:\n{context_block}"
    )


def main():
    parser = base_arg_parser(default_output="evals/results_faithfulness.json")
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
    print("FAITHFULNESS EVAL")
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

        if not item.get("answer"):
            print("  skipped (no answer in frozen data)")
            per_question[i] = {
                "index": item["index"],
                "question": item["question"],
                "score": 0,
                "reasoning": "",
                "parse_error": "no answer present in frozen data",
            }
            save_progress(args.output, "faithfulness", args.model, per_question)
            continue

        try:
            user_prompt = build_user_prompt(item)
            raw = call_judge(SYSTEM_PROMPT, user_prompt, api_key, model=args.model)
            verdict = parse_verdict(raw)
            verdict["index"] = item["index"]
            verdict["question"] = item["question"]
            per_question[i] = verdict

            print(f"  score={verdict['score']}  {verdict.get('reasoning', '')[:80]}")

        except RateLimitExhausted as e:
            print(f"\n⚠️  {e}")
            per_question[i] = {
                "index": item["index"],
                "question": item["question"],
                "score": 0,
                "reasoning": "",
                "parse_error": "rate limit exhausted, not yet scored",
            }
            save_progress(args.output, "faithfulness", args.model,
                          [q for q in per_question if q is not None])
            print(f"Progress saved to {args.output}. Re-run with --resume "
                  f"(optionally a different --key-env) to continue.")
            sys.exit(1)

        # Save after every question so Ctrl+C never loses more than
        # the one call currently in flight.
        save_progress(args.output, "faithfulness", args.model,
                      [q for q in per_question if q is not None])

        if i < len(items) - 1:
            time.sleep(args.pause)

    print_summary("faithfulness", args.model, per_question)
    print(f"Saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()