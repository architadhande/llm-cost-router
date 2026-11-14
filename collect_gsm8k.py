"""collect_gsm8k.py — Phase 1: take a random sample of math word problems from GSM8K."""

from pathlib import Path

from datasets import load_dataset

N_PROMPTS = 400
SEED = 42  # fixed seed -> same sample every run (reproducible)
OUT = Path("data/prompts_gsm8k.parquet")

df = load_dataset("openai/gsm8k", "main", split="train").to_pandas()

# One category only, so a plain random sample (no groupby needed)
sample = df.sample(n=N_PROMPTS, random_state=SEED)

sample = sample.rename(columns={"question": "prompt", "answer": "reference_answer"})
# GSM8K answers end with "#### <number>"; keep that number on its own for exact checks
sample["final_answer"] = sample["reference_answer"].str.split("####").str[-1].str.strip()
sample["source"] = "gsm8k"
sample["category"] = "math_reasoning"
sample = sample[["source", "category", "prompt", "reference_answer", "final_answer"]]
sample = sample.reset_index(drop=True)

OUT.parent.mkdir(exist_ok=True)
sample.to_parquet(OUT, index=False)
print(f"Saved {len(sample)} prompts to {OUT}")
