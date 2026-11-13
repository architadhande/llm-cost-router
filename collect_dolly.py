"""collect_dolly.py — Phase 1: take a balanced sample of prompts from Dolly-15k."""

from pathlib import Path

from datasets import load_dataset

PER_CATEGORY = 150  # 8 categories x 150 = 1,200 prompts
SEED = 42  # fixed seed -> same sample every run (reproducible)
OUT = Path("data/prompts_dolly.parquet")


def build_prompt(instruction, context):
    """Some Dolly tasks refer to a passage in `context`; the model needs to see it."""
    if context and context.strip():
        return f"{instruction.strip()}\n\nContext:\n{context.strip()}"
    return instruction.strip()


df = load_dataset("databricks/databricks-dolly-15k", split="train").to_pandas()
df["prompt"] = [build_prompt(i, c) for i, c in zip(df["instruction"], df["context"])]

# Stratified sample: the same number of rows from every category
sample = df.groupby("category").sample(n=PER_CATEGORY, random_state=SEED)

sample = sample.rename(columns={"response": "reference_answer"})
sample["source"] = "dolly"
sample = sample[["source", "category", "prompt", "reference_answer"]].reset_index(drop=True)

OUT.parent.mkdir(exist_ok=True)
sample.to_parquet(OUT, index=False)
print(f"Saved {len(sample)} prompts to {OUT}")
print(sample["category"].value_counts().to_string())
