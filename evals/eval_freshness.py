"""
FRESHNESS eval — completely isolated.

Receives ONLY: question, answer, retrieved context.
Does NOT receive: ground truth, or any other metric's verdict.

NOTE ON DEFINITION: freshness here specifically means "does the answer
make any claim about versions, dates, or recency that is NOT backed by
the retrieved context?" It does NOT mean "is the underlying documentation
itself current" — that's outside what a judge can assess from a single
Q&A pair. An answer that makes no version/date/recency claims at all has
nothing to be stale about, so it PASSES automatically. This fixes the
old failure mode where answers with zero recency content were still
failing freshness for reasons unrelated to actual staleness.

Run:
    python -m evals.eval_freshness
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

SYSTEM_PROMPT = """You are a strict but fair evaluator judging FRESHNESS only.

FRESHNESS here means: does the answer make any claim about versions,
dates, "latest"/"current"/"deprecated" status, or recency that is NOT
backed by the retrieved context?

IMPORTANT: if the answer makes NO claims about versions, dates, or
recency at all, it AUTOMATICALLY PASSES freshness. There is nothing to be
stale about if the answer never makes a time-sensitive claim in the first
place. Do not fail an answer for freshness just because it is long,
detailed, well-synthesized, or covers a lot of ground — none of that is
what freshness measures.

FAIL freshness ONLY if the answer:
- states a specific version number, release date, or "as of" claim that
  the retrieved context does not support
- claims something is the "latest," "newest," "current," or "deprecated"
  without that status being stated in the retrieved context
- presents information as up-to-date when the context marks it as
  outdated, superseded, or under a note/warning about change

PASS freshness if:
- the answer makes no version/date/recency claims at all, OR
- every version/date/recency claim it does make is directly supported by
  the retrieved context

You are given ONLY the question, the generated answer, and the retrieved
context. You must NOT judge faithfulness to other claims, completeness,
or relevancy — only whether any recency-type claim is unsupported.

Respond with ONLY a JSON object, no other text, no markdown fences:
{"score": 0 or 1, "reasoning": "one or two sentences explaining the verdict, noting explicitly if the answer contained zero recency claims"}
"""


def build_user_prompt(item: dict) -> str:
    context_block = "\n\n---\n\n".join(item.get("context", []))
    return (
        f"QUESTION:\n{item['question']}\n\n"
        f"ANSWER:\n{item['answer']}\n\n"
        f"RETRIEVED CONTEXT:\n{context_block}"
    )


def main():
    parser = base_arg_parser(default_output="evals/results_freshness.json")
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
    print("FRESHNESS EVAL")
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
            save_progress(args.output, "freshness", args.model, per_question)
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
                    "freshness", args.langsmith_project, item, verdict,
                    inputs={"question": item["question"], "answer": item["answer"],
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
            save_progress(args.output, "freshness", args.model,
                          [q for q in per_question if q is not None])
            print(f"Progress saved to {args.output}. Re-run with --resume "
                  f"(optionally a different --key-env) to continue.")
            sys.exit(1)

        save_progress(args.output, "freshness", args.model,
                      [q for q in per_question if q is not None])

        if i < len(items) - 1:
            time.sleep(args.pause)

    print_summary("freshness", args.model, per_question)
    print(f"Saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()