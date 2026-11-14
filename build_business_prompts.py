"""build_business_prompts.py — Phase 1: turn cached business snippets into prompts.

No API calls: reads data/search_cache.json (made by collect_business_search.py).
"""

import json
from pathlib import Path

import pandas as pd

CACHE = Path("data/search_cache.json")
OUT = Path("data/prompts_business.parquet")

# Same 4 tasks for every snippet, roughly easy -> hard
TEMPLATES = {
    "business_summary": "Summarize the text below in 2 sentences for a busy executive.",
    "business_extract": "Extract every number and metric from the text below into a table "
                        "with columns: metric, value, period.",
    "business_email": "Write a short, professional email to customers explaining the news "
                      "in the text below.",
    "business_risks": "Based only on the text below, what are the top 3 business risks, and "
                      "what evidence supports each? If the text does not support a risk, "
                      "say 'not enough information' instead of guessing.",
}

cache = json.loads(CACHE.read_text())
snippets = pd.DataFrame(
    [{"provider": v["provider"], **r} for v in cache.values() for r in v["results"]]
)
snippets = snippets.drop_duplicates(subset="url").reset_index(drop=True)

rows = []
for _, s in snippets.iterrows():
    for category, task in TEMPLATES.items():
        rows.append({
            "source": "business",
            "category": category,
            "prompt": f"{task}\n\nText:\n{s['text'].strip()}",
            "reference_answer": None,  # open-ended: the judge grades these in Phase 2
            "provider": s["provider"],
            "url": s["url"],
        })

prompts = pd.DataFrame(rows)
prompts.to_parquet(OUT, index=False)
print(f"{len(snippets)} unique snippets x {len(TEMPLATES)} templates = {len(prompts)} prompts")
print(f"Saved to {OUT}")
