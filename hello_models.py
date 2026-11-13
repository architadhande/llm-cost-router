"""hello_models.py — Phase 0: send one question to two LLMs and compare them."""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq

load_dotenv()  # copies the keys in .env into environment variables (never print them)

QUESTION = "In two sentences, why is the sky blue?"
GROQ_MODEL = "openai/gpt-oss-20b"  # Groq retired llama-3.1-8b-instant
GEMINI_MODEL = "gemini-3.8-flash"

# timeout = give up on a single attempt after 30 s; with_retries() handles retrying
groq_client = Groq(timeout=30, max_retries=0)
gemini_client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=30_000),  # this SDK uses milliseconds
)


def with_retries(call, attempts=3, first_wait=2.0):
    """Run call(). On a temporary failure, wait and retry, doubling the wait each time."""
    wait = first_wait
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as e:
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            # 4xx = our request is wrong, retrying won't help (except 429 = "slow down")
            permanent = isinstance(status, int) and 400 <= status < 500 and status != 429
            if permanent or attempt == attempts:
                raise
            print(f"  attempt {attempt} failed ({status or type(e).__name__}), retrying in {wait:.0f}s")
            time.sleep(wait)
            wait *= 2


def ask_groq(question):
    start = time.perf_counter()
    r = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": question}],
    )
    seconds = time.perf_counter() - start
    details = r.usage.completion_tokens_details
    return {
        "model": GROQ_MODEL,
        "answer": r.choices[0].message.content,
        "seconds": seconds,
        "input_tokens": r.usage.prompt_tokens,
        "output_tokens": r.usage.completion_tokens,  # already includes reasoning
        "reasoning_tokens": (details.reasoning_tokens if details else 0) or 0,
    }


def ask_gemini(question):
    start = time.perf_counter()
    r = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=question)
    seconds = time.perf_counter() - start
    u = r.usage_metadata
    thinking = u.thoughts_token_count or 0
    return {
        "model": GEMINI_MODEL,
        "answer": r.text,
        "seconds": seconds,
        "input_tokens": u.prompt_token_count,
        "output_tokens": (u.candidates_token_count or 0) + thinking,  # billed together
        "reasoning_tokens": thinking,
    }


def print_result(result):
    """Print one model's answer plus its speed and token usage."""
    print(f"=== {result['model']} ===")
    print(result['answer'])
    print(f"{result['seconds']:.2f}s | {result['input_tokens']} in / {result['output_tokens']} out tokens ({result['reasoning_tokens']} reasoning)")
    print()


if __name__ == "__main__":
    print(f"Question: {QUESTION}\n")
    for ask in (ask_groq, ask_gemini):
        try:
            result = with_retries(lambda: ask(QUESTION))
        except Exception as e:
            print(f"{ask.__name__} gave up: {type(e).__name__}: {str(e)[:150]}\n")
            continue
        print_result(result)
