"""embed_prompts.py — Phase 3: turn every prompt into features the router can learn from.

Two kinds of features, both computed locally ($0, no API):
- embedding: 384 numbers capturing what the prompt is about (all-MiniLM-L6-v2)
- simple stats: length, digits, whether it includes a context text, etc.
We do NOT use `source`/`category` as features: real traffic doesn't come labelled.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, good general embeddings
OUT = Path("data/features.parquet")

df = pd.read_parquet("data/prompts_all.parquet")
model = SentenceTransformer(MODEL)
emb = model.encode(df["prompt"].tolist(), batch_size=64, show_progress_bar=True,
                   normalize_embeddings=True)

text = df["prompt"]
stats = pd.DataFrame({
    "n_chars": text.str.len(),
    "n_words": text.str.split().str.len(),
    "n_digits": text.str.count(r"\d"),
    "n_questions": text.str.count(r"\?"),
    "has_context": text.str.contains(r"\n\n(?:Context|Text):\n").astype(int),
    "log_chars": np.log1p(text.str.len()),
})
emb_cols = pd.DataFrame(emb, columns=[f"e{i}" for i in range(emb.shape[1])])

features = pd.concat([df[["prompt_id"]], stats, emb_cols], axis=1)
features.to_parquet(OUT, index=False)
print(f"{len(features)} prompts x {features.shape[1] - 1} features -> {OUT}")
