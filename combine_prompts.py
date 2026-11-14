"""combine_prompts.py — Phase 1: merge all prompt sources into one table with stable IDs."""

import hashlib
from pathlib import Path

import pandas as pd

SOURCES = ["dolly", "gsm8k", "business"]
OUT = Path("data/prompts_all.parquet")


def make_id(prompt):
    """Same prompt text -> same ID, every run. Later this is our cache key."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


df = pd.concat(
    [pd.read_parquet(f"data/prompts_{name}.parquet") for name in SOURCES],
    ignore_index=True,  # renumber rows 0..N-1 instead of repeating 0.. per file
)
df.insert(0, "prompt_id", df["prompt"].map(make_id))

before = len(df)
df = df.drop_duplicates(subset="prompt_id").reset_index(drop=True)

df.to_parquet(OUT, index=False)
print(f"Combined {before} prompts, removed {before - len(df)} duplicates -> {len(df)} saved to {OUT}")
print(df.groupby(["source", "category"]).size().to_string())
