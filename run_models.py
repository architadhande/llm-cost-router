"""run_models.py — Phase 2: answer prompts with the cheap and frontier models (cached)."""

import sys
import time

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

import llm_cache
from api_utils import with_retries

load_dotenv(".env")
CHEAP, FRONTIER = "openai/gpt-oss-20b", "openai/gpt-oss-120b"
REASONING_EFFORT, MAX_TOKENS = "low", 1000
SYSTEM = "Answer concisely: under 300 words unless the task needs more."
SYSTEM_VERSION = "concise-v1"  # bump this if SYSTEM changes, so old answers are not reused
CONFIG = f"{REASONING_EFFORT}|{MAX_TOKENS}|{SYSTEM_VERSION}"  # part of the cache key
PAUSE = 2.0  # seconds between real calls (stays under per-minute limits)

groq = Groq(timeout=60, max_retries=0)


def answer(prompt_id, prompt, model):
    """Return the model's answer, from cache if we have it, otherwise call Groq once."""
    cached = llm_cache.get(prompt_id, model, CONFIG)
    if cached:
        return cached, False
    start = time.perf_counter()
    r = with_retries(lambda: groq.chat.completions.create(
        model=model, messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}],
        reasoning_effort=REASONING_EFFORT, max_completion_tokens=MAX_TOKENS,
    ), attempts=5, first_wait=10)
    details = r.usage.completion_tokens_details
    result = {
        "prompt_id": prompt_id, "model": model, "config": CONFIG,
        "answer": r.choices[0].message.content or "",
        "finish_reason": r.choices[0].finish_reason,
        "input_tokens": r.usage.prompt_tokens,
        "output_tokens": r.usage.completion_tokens,  # includes reasoning
        "reasoning_tokens": (details.reasoning_tokens if details else 0) or 0,
        "seconds": time.perf_counter() - start,
    }
    llm_cache.put(result)
    return result, True


if __name__ == "__main__":
    ids_file = sys.argv[1]  # a .txt of prompt_ids, one per line
    ids = set(open(ids_file).read().split())
    prompts = pd.read_parquet("data/prompts_all.parquet").query("prompt_id in @ids")
    calls = 0
    for n, row in enumerate(prompts.itertuples(), 1):
        for model in (CHEAP, FRONTIER):
            _, was_called = answer(row.prompt_id, row.prompt, model)
            if was_called:
                calls += 1
                time.sleep(PAUSE)
        print(f"[{n}/{len(prompts)}] {row.source}/{row.category} done")
    print(f"Real API calls this run: {calls}")
