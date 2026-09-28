"""
Shared infrastructure for the isolated eval scripts.

This module ONLY handles mechanics: calling Groq, parsing the judge's JSON
verdict, and saving/printing results. It has no knowledge of what a metric
is or what inputs it should receive — that lives entirely in each
eval_<metric>.py script, so metrics stay isolated from each other.
"""

import json
import os
import time
import uuid
import argparse
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3.8-27b"

try:
    from langsmith import Client as _LangSmithClient
except ImportError:
    _LangSmithClient = None

_langsmith_client = None
_langsmith_warned = False


def get_langsmith_client():
    """Lazily create a LangSmith client. Returns None (and prints a
    one-time warning) if the langsmith package isn't installed or
    LANGSMITH_API_KEY isn't set, so callers can always call this safely
    and just skip logging when it's unavailable."""

    global _langsmith_client, _langsmith_warned

    if _langsmith_client is not None:
        return _langsmith_client

    if _LangSmithClient is None:
        if not _langsmith_warned:
            print("  (langsmith package not installed -- skipping LangSmith logging. "
                  "pip install langsmith to enable it.)")
            _langsmith_warned = True
        return None

    if not os.getenv("LANGSMITH_API_KEY"):
        if not _langsmith_warned:
            print("  (LANGSMITH_API_KEY not set -- skipping LangSmith logging.)")
            _langsmith_warned = True
        return None

    _langsmith_client = _LangSmithClient()
    return _langsmith_client


def log_judge_run(metric_name: str, project_name: str, item: dict,
                   verdict: dict, inputs: dict, model: str) -> None:
    """Log one already-completed judge call to LangSmith as a standalone
    run, purely for visualization/tracing. This never feeds back into the
    judging logic and never touches another metric's data -- it just
    records what this one isolated judge call saw and decided."""

    client = get_langsmith_client()
    if client is None:
        return

    now = datetime.now(timezone.utc)
    try:
        client.create_run(
            id=uuid.uuid4(),
            name=f"{metric_name}_q{item['index']}",
            run_type="llm",
            project_name=project_name,
            inputs=inputs,
            outputs={"score": verdict.get("score"), "reasoning": verdict.get("reasoning")},
            start_time=now,
            end_time=now,
            extra={
                "metadata": {
                    "metric": metric_name,
                    "question_index": item["index"],
                    "judge_model": model,
                    "parse_error": verdict.get("parse_error"),
                }
            },
        )
    except Exception as e:
        # Never let a logging failure break the actual eval run.
        print(f"    (LangSmith logging failed for q{item['index']}: {e})")


class RateLimitExhausted(Exception):
    """Raised when repeated 429s suggest the key's quota (RPM or TPD) is
    genuinely used up, not just a transient blip. Callers should save
    whatever progress they have and stop, rather than crash or hang."""
    pass


def load_frozen(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["results"]


def call_judge(system_prompt: str, user_prompt: str, api_key: str,
               model: str = DEFAULT_MODEL, temperature: float = 0.0,
               max_retries: int = 6, initial_delay: float = 10.0,
               max_delay: float = 120.0, request_timeout: float = 90.0) -> str:
    """Call the Groq-hosted judge model. Returns raw text content.
    Retries with exponential backoff on 429s. After max_retries, raises
    RateLimitExhausted instead of hanging or crashing with a raw traceback
    -- at that point the key is very likely out of quota for the day and
    further retries just waste time."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    delay = initial_delay
    for attempt in range(max_retries):
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=payload,
                                  timeout=request_timeout)
        except requests.exceptions.Timeout:
            print(f"    request timed out after {request_timeout}s, retrying in {delay}s...")
            time.sleep(delay)
            delay = min(delay * 2, max_delay)
            continue

        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        if resp.status_code == 429:
            print(f"    429 rate limited (attempt {attempt + 1}/{max_retries}), "
                  f"retrying in {delay:.0f}s...")
            time.sleep(delay)
            delay = min(delay * 2, max_delay)
            continue

        if resp.status_code >= 500:
            print(f"    {resp.status_code} server error (attempt {attempt + 1}/{max_retries}), "
                  f"retrying in {delay:.0f}s...")
            time.sleep(delay)
            delay = min(delay * 2, max_delay)
            continue

        resp.raise_for_status()

    raise RateLimitExhausted(
        f"Gave up after {max_retries} retries due to repeated 429s/server errors. "
        f"This is either the key's quota exhausted for the day, or Groq having "
        f"a transient outage -- switch --key-env, wait a bit, or check "
        f"https://groqstatus.com if this keeps happening."
    )


def parse_verdict(raw_text: str) -> dict:
    """Parse the judge's JSON verdict: {"score": 0 or 1, "reasoning": "..."}.
    A malformed response is treated as a FAIL (score=0), never silently
    dropped or counted as a pass, with the raw text preserved for debugging."""

    cleaned = raw_text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        score = parsed.get("score")
        reasoning = parsed.get("reasoning", "")

        if score not in (0, 1):
            return {
                "score": 0,
                "reasoning": reasoning,
                "parse_error": f"score field was {score!r}, expected 0 or 1",
                "raw_response": raw_text,
            }

        return {"score": score, "reasoning": reasoning}

    except json.JSONDecodeError as e:
        return {
            "score": 0,
            "reasoning": "",
            "parse_error": f"JSON parse failed: {e}",
            "raw_response": raw_text,
        }


def _build_output(metric_name: str, model: str, per_question: list) -> dict:
    total = len(per_question)
    passed = sum(r["score"] for r in per_question if isinstance(r.get("score"), int))
    parse_errors = sum(1 for r in per_question if "parse_error" in r)
    avg = passed / total if total else 0.0

    return {
        "metadata": {
            "metric": metric_name,
            "judge_model": model,
            "question_count": total,
            "passed": passed,
            "avg": avg,
            "parse_errors": parse_errors,
        },
        "results": per_question,
    }


def save_progress(output_path: str, metric_name: str, model: str,
                   per_question: list) -> None:
    """Write current progress to disk after every question. Uses a
    temp-file-then-rename so a crash or Ctrl+C mid-write never corrupts
    the previous good save."""

    output = _build_output(metric_name, model, per_question)
    tmp_path = output_path + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    os.replace(tmp_path, output_path)


def load_existing_verdicts(output_path: str) -> dict:
    """Load a previous (possibly partial) results file, if present.
    Returns {index: verdict_dict} for entries that completed with a real
    score (no parse_error), so --resume can skip only genuinely-done work."""

    if not os.path.exists(output_path):
        return {}

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        item["index"]: item
        for item in data.get("results", [])
        if "parse_error" not in item and isinstance(item.get("score"), int)
    }


def print_summary(metric_name: str, model: str, per_question: list) -> None:
    output = _build_output(metric_name, model, per_question)
    meta = output["metadata"]

    print()
    print("=" * 70)
    print(f"{metric_name.upper()} COMPLETE")
    print("=" * 70)
    print(f"avg={meta['avg']:.3f}  pass={meta['passed']}/{meta['question_count']}")
    if meta["parse_errors"]:
        print(f"⚠️  {meta['parse_errors']} response(s) failed to parse — treated as FAIL, see 'parse_error' fields")
    print("=" * 70)


# Kept for backward compatibility with any script still calling the old
# single-shot save function directly.
def save_results(output_path: str, metric_name: str, model: str,
                  per_question: list) -> None:
    save_progress(output_path, metric_name, model, per_question)
    print_summary(metric_name, model, per_question)
    print(f"Saved to: {output_path}")
    print("=" * 70)


def base_arg_parser(default_output: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="evals/frozen_rag.json")
    parser.add_argument("--output", default=default_output)
    parser.add_argument(
        "--key-env", default="GROQ_API_KEY_2",
        help="Name of the .env var holding the Groq API key for this judge call."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--pause", type=float, default=15.0,
        help="Seconds to sleep between judge calls (rate-limit courtesy)."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from an existing output file, skipping questions already scored."
    )
    parser.add_argument(
        "--langsmith", action="store_true",
        help="Also log each judge call to LangSmith for tracing/visualization "
             "(requires LANGSMITH_API_KEY in .env). Purely additive -- never "
             "affects judging."
    )
    parser.add_argument(
        "--langsmith-project", default="agentic-rag-isolated-evals",
        help="LangSmith project to log runs into. Use the same project across "
             "all four scripts to see everything together."
    )
    return parser