"""embed_prompts.py — Phase 3: compute router features for every prompt ($0, runs locally).

Features come from features.py (shared with the app). We do NOT use `source`/`category`
as features: real traffic doesn't arrive labelled.
"""

from pathlib import Path

import pandas as pd

from features import compute_features, load_embedder

OUT = Path("data/features.parquet")

df = pd.read_parquet("data/prompts_all.parquet")
feats = compute_features(df["prompt"], load_embedder())
features = pd.concat([df[["prompt_id"]].reset_index(drop=True), feats], axis=1)
features.to_parquet(OUT, index=False)
print(f"{len(features)} prompts x {features.shape[1] - 1} features -> {OUT}")
