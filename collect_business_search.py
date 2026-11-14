"""collect_business_search.py — Phase 1: fetch public business snippets via Tavily + Exa.

Results are cached in data/search_cache.json, so re-running costs zero API credits.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from exa_py import Exa
from tavily import TavilyClient

from api_utils import with_retries

load_dotenv()
CACHE = Path("data/search_cache.json")
RESULTS_PER_QUERY = 5
PAUSE = 1.0  # seconds between calls, to stay polite with free tiers

TOPICS = [
    "quarterly earnings results press release", "new product launch announcement",
    "company layoffs restructuring announcement", "acquisition merger announcement",
    "subscription price increase announcement", "product recall customer complaints",
    "supply chain disruption company impact", "regulator fines company",
    "company hiring expansion new office", "analyst downgrade stock rating",
]
ANGLES = ["2026", "technology company", "retail company"]  # 3 phrasings per topic
QUERIES = [f"{topic} {angle}" for topic in TOPICS for angle in ANGLES]  # 30 queries

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
exa = Exa(api_key=os.environ["EXA_API_KEY"])


def search_tavily(query):
    r = tavily.search(query, search_depth="basic", max_results=RESULTS_PER_QUERY)
    return [{"title": x["title"], "url": x["url"], "text": x["content"]} for x in r["results"]]


def search_exa(query):
    r = exa.search(query, num_results=RESULTS_PER_QUERY, contents={"text": {"max_characters": 2000}})
    return [{"title": x.title, "url": x.url, "text": x.text} for x in r.results]


cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
for i, query in enumerate(QUERIES):
    if query in cache:
        continue  # already paid for this one
    provider, search = ("tavily", search_tavily) if i % 2 == 0 else ("exa", search_exa)
    print(f"[{i + 1}/{len(QUERIES)}] {provider}: {query}")
    cache[query] = {"provider": provider, "results": with_retries(lambda: search(query))}
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=1))  # save after every call
    time.sleep(PAUSE)

print(f"\nCached queries: {len(cache)} | snippets: {sum(len(v['results']) for v in cache.values())}")
