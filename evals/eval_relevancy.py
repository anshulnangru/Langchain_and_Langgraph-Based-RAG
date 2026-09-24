"""
ANSWER RELEVANCY eval — completely isolated.

Receives ONLY: question, answer.
Does NOT receive: context, ground truth, or any other metric's verdict.

This is the regression test for the bug that motivated this whole
rebuild: a merged judge call once gave a near-perfect score to an answer
about LangSmith deployment/thread search when the question asked about
checkpoint timing. With relevancy judged in total isolation — nothing but
the raw question/answer pair — that kind of off-topic answer has nowhere
to hide behind a good faithfulness or recall score.

Run:
    python -m evals.eval_relevancy
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evals.eval_common import (
    load_frozen, call_judge, parse_verdict, base_arg_parser,
    save_progress, load_existing_verdicts, print_summary, RateLimitExhausted,
)

SYSTEM_PROMPT = """You are a strict but fair evaluator judging ANSWER RELEVANCY only.

ANSWER RELEVANCY asks one narrow question: does this answer actually
address what was asked? You have NO access to any source material, so you
cannot and must not judge factual correctness, completeness, or grounding
— only whether the answer's topic and content match the question's topic
and intent.

FAIL relevancy if the answer:
- addresses a different topic than the one asked about, even if that
  topic is related or adjacent (e.g. the question asks about checkpoint
  TIMING, and the answer discusses checkpoint STORAGE BACKENDS or an
  unrelated deployment feature instead)
- answers a narrower or broader question than the one asked, such that a
  reader would come away without an answer to their actual question
- is generic or evasive where a direct, on-topic answer was expected

PASS relevancy if:
- the answer's content is clearly about what the question asked, even if
  you personally don't know whether every detail in it is correct

Respond with ONLY a JSON object, no other text, no markdown fences:
{"score": 0 or 1, "reasoning": "one or two sentences explaining the verdict"}
"""


def build_user_prompt(item: dict) -> str:
    return (
        f"QUESTION:\n{item['question']}\n\n"
        f"ANSWER:\n{item['answer']}"
    )


def main():
    parser = base_arg_parser(default_output="evals/results_relevancy.json")
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
    print("ANSWER RELEVANCY EVAL")
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
            save_progress(args.output, "relevancy", args.model, per_question)
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
            save_progress(args.output, "relevancy", args.model,
                          [q for q in per_question if q is not None])
            print(f"Progress saved to {args.output}. Re-run with --resume "
                  f"(optionally a different --key-env) to continue.")
            sys.exit(1)

        save_progress(args.output, "relevancy", args.model,
                      [q for q in per_question if q is not None])

        if i < len(items) - 1:
            time.sleep(args.pause)

    print_summary("relevancy", args.model, per_question)
    print(f"Saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()