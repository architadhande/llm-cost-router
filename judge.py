"""judge.py — Phase 2: grade model answers with Gemini, 10 prompts (20 answers) per call.

Usage: python judge.py data/pilot_ids.txt [--dry-run]
"""

import os
import random
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

import llm_cache
from api_utils import with_retries
from run_models import CHEAP, CONFIG, FRONTIER

load_dotenv(".env")
JUDGE = "gemini-3.8-flash"  # Lite tested: passed 32/32 incl. clear errors; too lenient
THINKING = "LOW"  # judge thinking level; less thinking = faster replies, fewer 504s
RUBRIC_VERSION = "v2"  # v2 = reference answers + list evidence before the verdict
JUDGE_KEY = f"{JUDGE}|think-{THINKING.lower()}|rubric-{RUBRIC_VERSION}"  # cache key
PROMPTS_PER_CALL = 10
MAX_CALLS_PER_RUN = 18  # free tier is 20 requests/day; keep 2 spare
PAUSE = 13  # free tier is 5 requests/minute

RUBRIC = """You are a strict grader of answers from AI assistants. Grade EACH item
independently; do not compare items with each other.

For each item, FIRST fill in the evidence lists, THEN decide:
- unsupported_claims: if the TASK includes a text/context, list every claim in the
  ANSWER that is not stated in that text. Otherwise leave empty.
- factual_errors: list every statement in the ANSWER that is false, or that contradicts
  the REFERENCE ANSWER when one is given. Check names, numbers, places and dates.
- passed: true ONLY if both lists are empty AND the answer does what the task asks,
  completely. Otherwise false.

A REFERENCE ANSWER, when given, was written by a human. For open-ended tasks
(brainstorming, creative writing) the ANSWER may differ from it in content, but it must
not contain factual errors. Return one verdict per item_id."""


class Verdict(BaseModel):
    item_id: str
    unsupported_claims: list[str]  # evidence fields come BEFORE the verdict on purpose
    factual_errors: list[str]
    passed: bool
    reason: str


def build_batch(rows, rng):
    """Turn prompts + both answers into blinded, shuffled items for one judge call."""
    items = []
    for row in rows:
        for model in (CHEAP, FRONTIER):
            ans = llm_cache.get(row.prompt_id, model, CONFIG)["answer"]
            ref = row.reference_answer if isinstance(row.reference_answer, str) else ""
            items.append({"prompt_id": row.prompt_id, "model": model,
                          "task": row.prompt, "answer": ans, "reference": ref.strip()})
    rng.shuffle(items)  # mix the two models' answers so neither is always first
    for i, it in enumerate(items):
        it["item_id"] = f"a{i:02d}"  # blind: the judge never sees the model name

    def fmt(it):
        ref = f"\n\nREFERENCE ANSWER:\n{it['reference']}" if it["reference"] else ""
        return f"### item_id: {it['item_id']}\nTASK:\n{it['task']}{ref}\n\nANSWER:\n{it['answer']}"

    text = "\n\n".join(fmt(it) for it in items)
    return items, f"{RUBRIC}\n\n{text}"


def grade(contents):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                          http_options=types.HttpOptions(timeout=120_000))
    r = client.models.generate_content(
        model=JUDGE, contents=contents,
        config=types.GenerateContentConfig(response_mime_type="application/json",
                                           response_schema=list[Verdict],
                                           thinking_config=types.ThinkingConfig(thinking_level=THINKING)))
    return r.parsed, r.usage_metadata


if __name__ == "__main__":
    ids = list(dict.fromkeys(open(sys.argv[1]).read().split()))  # keep file order, drop repeats
    dry_run = "--dry-run" in sys.argv
    prompts = (pd.read_parquet("data/prompts_all.parquet").set_index("prompt_id")
               .loc[ids].reset_index())  # judge in the order the ids file lists them
    prompts = prompts[prompts.source != "gsm8k"]  # graded by exact match instead
    answered = [r for r in prompts.itertuples()  # both models must have answered already
                if all(llm_cache.get(r.prompt_id, m, CONFIG) for m in (CHEAP, FRONTIER))]
    if len(answered) < len(prompts):
        print(f"skipping {len(prompts) - len(answered)} prompts not yet answered by both models")
    todo = [r for r in answered
            if not all(llm_cache.is_judged(r.prompt_id, m, CONFIG, JUDGE_KEY) for m in (CHEAP, FRONTIER))]
    batches = [todo[i:i + PROMPTS_PER_CALL] for i in range(0, len(todo), PROMPTS_PER_CALL)]
    print(f"{len(todo)} prompts to judge -> {len(batches)} judge calls")
    rng = random.Random(42)
    for n, batch in enumerate(batches[:MAX_CALLS_PER_RUN], 1):
        items, contents = build_batch(batch, rng)
        if dry_run:
            print(f"\n--- DRY RUN: batch {n}, {len(items)} items, ~{len(contents)//4} tokens ---")
            print(contents[:700], "...")
            break
        verdicts, usage = with_retries(lambda: grade(contents), attempts=3, first_wait=60)
        by_id = {it["item_id"]: it for it in items}
        for v in verdicts:
            it = by_id.get(v.item_id)
            if it:  # ignore any item_id the judge invented
                issues = [f"UNSUPPORTED: {c}" for c in v.unsupported_claims] + \
                         [f"ERROR: {e}" for e in v.factual_errors]
                reason = v.reason + (" | " + " | ".join(issues) if issues else "")
                llm_cache.put_judgment(it["prompt_id"], it["model"], CONFIG, JUDGE_KEY, v.passed, reason)
        print(f"[call {n}] {len(verdicts)}/{len(items)} verdicts | tokens in/out "
              f"{usage.prompt_token_count}/{usage.candidates_token_count} "
              f"(thinking {usage.thoughts_token_count or 0})")
        time.sleep(PAUSE)
