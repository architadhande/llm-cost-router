"""features.py — the ONE place prompt features are computed (training and the app share it).

Sharing this code avoids train/serve skew: the app must compute features exactly the
way training did, or the router sees numbers it never learned from.
"""

import numpy as np
import pandas as pd

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, runs locally


def load_embedder():
    from sentence_transformers import SentenceTransformer  # heavy import: only when needed
    return SentenceTransformer(EMBED_MODEL)


def compute_features(prompts, embedder):
    """List of prompt strings -> DataFrame of 6 simple stats + 384 embedding columns."""
    text = pd.Series(list(prompts))
    emb = embedder.encode(text.tolist(), batch_size=64, normalize_embeddings=True)
    stats = pd.DataFrame({
        "n_chars": text.str.len(),
        "n_words": text.str.split().str.len(),
        "n_digits": text.str.count(r"\d"),
        "n_questions": text.str.count(r"\?"),
        "has_context": text.str.contains(r"\n\n(?:Context|Text):\n").astype(int),
        "log_chars": np.log1p(text.str.len()),
    })
    emb_cols = pd.DataFrame(emb, columns=[f"e{i}" for i in range(emb.shape[1])])
    return pd.concat([stats, emb_cols], axis=1)
