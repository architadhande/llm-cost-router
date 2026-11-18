"""select_sample.py — Phase 2: pick the v1 labelling sample, balanced across categories.

Keeps the 20 pilot prompts (already answered and partly judged), then tops up each
category at random to PER_CATEGORY.
"""

from pathlib import Path

import pandas as pd

PER_CATEGORY = 70  # 13 categories x 70 = 910 prompts
SEED = 42
PILOT = Path("data/pilot_ids.txt")
OUT = Path("data/sample_v1_ids.txt")

df = pd.read_parquet("data/prompts_all.parquet")
pilot_ids = set(PILOT.read_text().split())

parts = []
for category, group in df.groupby("category"):
    in_pilot = group[group.prompt_id.isin(pilot_ids)]
    rest = group[~group.prompt_id.isin(pilot_ids)]
    top_up = rest.sample(n=PER_CATEGORY - len(in_pilot), random_state=SEED)
    parts.append(pd.concat([in_pilot, top_up]))

sample = pd.concat(parts)
OUT.write_text("\n".join(sample.prompt_id))
print(f"{len(sample)} prompts ({sample.prompt_id.isin(pilot_ids).sum()} from the pilot) -> {OUT}")
print(sample.groupby("source").size().to_string())
