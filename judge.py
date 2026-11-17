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
JUDGE = "gemini-3.8-flash"
PROMPTS_PER_CALL = 10
MAX_CALLS_PER_RUN = 18  # free tier is 20 requests/day; keep 2 spare
PAUSE = 13  # free tier is 5 requests/minute

RUBRIC = """You are grading answers from AI assistants. For EACH item, decide PASS or FAIL
independently. Do not compare items with each other.
PASS only if the answer: (1) does what the task asks, (2) is correct, (3) is complete
(not cut off mid-sentence), and (4) for tasks that give a text, uses only facts from
that text. Give a one-sentence reason. Return one verdict per item_id."""


class Verdict(BaseModel):
    item_id: str
    passed: bool
    reason: str


def build_batch(rows, rng):
    """Turn prompts + both answers into blinded, shuffled items for one judge call."""
    items = []
    for row in rows:
        for model in (CHEAP, FRONTIER):
            ans = llm_cache.get(row.prompt_id, model, CONFIG)["answer"]
            items.append({"prompt_id": row.prompt_id, "model": model,
                          "task": row.prompt, "answer": ans})
    rng.shuffle(items)  # mix the two models' answers so neither is always first
    for i, it in enumerate(items):
        it["item_id"] = f"a{i:02d}"  # blind: the judge never sees the model name
    text = "\n\n".join(f"### item_id: {it['item_id']}\nTASK:\n{it['task']}\n\nANSWER:\n{it['answer']}"
                       for it in items)
    return items, f"{RUBRIC}\n\n{text}"


def grade(contents):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                          http_options=types.HttpOptions(timeout=120_000))
    r = client.models.generate_content(
        model=JUDGE, contents=contents,
        config=types.GenerateContentConfig(response_mime_type="application/json",
                                           response_schema=list[Verdict]))
    return r.parsed, r.usage_metadata


if __name__ == "__main__":
    ids = set(open(sys.argv[1]).read().split())
    dry_run = "--dry-run" in sys.argv
    prompts = pd.read_parquet("data/prompts_all.parquet").query("prompt_id in @ids")
    prompts = prompts[prompts.source != "gsm8k"]  # graded by exact match instead
    todo = [r for r in prompts.itertuples()
            if not all(llm_cache.is_judged(r.prompt_id, m, CONFIG, JUDGE) for m in (CHEAP, FRONTIER))]
    batches = [todo[i:i + PROMPTS_PER_CALL] for i in range(0, len(todo), PROMPTS_PER_CALL)]
    print(f"{len(todo)} prompts to judge -> {len(batches)} judge calls")
    rng = random.Random(42)
    for n, batch in enumerate(batches[:MAX_CALLS_PER_RUN], 1):
        items, contents = build_batch(batch, rng)
        if dry_run:
            print(f"\n--- DRY RUN: batch {n}, {len(items)} items, ~{len(contents)//4} tokens ---")
            print(contents[:700], "...")
            break
        verdicts, usage = with_retries(lambda: grade(contents), attempts=4, first_wait=15)
        by_id = {it["item_id"]: it for it in items}
        for v in verdicts:
            it = by_id.get(v.item_id)
            if it:  # ignore any item_id the judge invented
                llm_cache.put_judgment(it["prompt_id"], it["model"], CONFIG, JUDGE, v.passed, v.reason)
        print(f"[call {n}] {len(verdicts)}/{len(items)} verdicts | tokens in/out "
              f"{usage.prompt_token_count}/{usage.candidates_token_count} "
              f"(thinking {usage.thoughts_token_count or 0})")
        time.sleep(PAUSE)
