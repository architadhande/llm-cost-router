"""build_labels.py — Phase 2 -> 3: one row per prompt with pass/fail and modelled cost.

GSM8K is graded by exact match; everything else by the Gemini judge.
Only prompts graded for BOTH models are kept.
"""

from pathlib import Path

import pandas as pd

import llm_cache
from grading import grade_gsm8k
from judge import JUDGE_KEY
from run_models import CHEAP, CONFIG, FRONTIER

# Groq published prices, USD per 1M tokens (console.groq.com/docs/models, Nov 2026)
PRICES = {CHEAP: (0.075, 0.30), FRONTIER: (0.15, 0.60)}  # (input, output)
OUT = Path("data/labels.parquet")


def cost(row):
    price_in, price_out = PRICES[row.model]
    return (row.input_tokens * price_in + row.output_tokens * price_out) / 1e6


prompts = pd.read_parquet("data/prompts_all.parquet")
sample_ids = set(Path("data/sample_v1_ids.txt").read_text().split())
prompts = prompts[prompts.prompt_id.isin(sample_ids)]

resp = llm_cache.con.execute("SELECT * FROM responses WHERE config = ?", [CONFIG]).df()
resp = resp[resp.prompt_id.isin(sample_ids)].copy()
resp["cost"] = resp.apply(cost, axis=1)

# Pass/fail: exact match for GSM8K, judge verdict for the rest
resp = resp.merge(prompts[["prompt_id", "source", "final_answer"]], on="prompt_id")
gsm = resp.source == "gsm8k"
resp.loc[gsm, "passed"] = [grade_gsm8k(a, f) for a, f in zip(resp[gsm].answer, resp[gsm].final_answer)]
judged = llm_cache.con.execute(
    "SELECT prompt_id, model, passed AS judge_passed FROM judgments WHERE config = ? AND judge = ?",
    [CONFIG, JUDGE_KEY]).df()
resp = resp.merge(judged, on=["prompt_id", "model"], how="left")
resp.loc[~gsm, "passed"] = resp.loc[~gsm, "judge_passed"]

# One row per prompt: cheap vs frontier side by side
wide = resp.pivot_table(index="prompt_id", columns="model", values=["passed", "cost"], aggfunc="first")
labels = pd.DataFrame({
    "cheap_pass": wide[("passed", CHEAP)], "frontier_pass": wide[("passed", FRONTIER)],
    "cheap_cost": wide[("cost", CHEAP)], "frontier_cost": wide[("cost", FRONTIER)],
}).dropna().reset_index()
labels[["cheap_pass", "frontier_pass"]] = labels[["cheap_pass", "frontier_pass"]].astype(bool)
labels = labels.merge(prompts[["prompt_id", "source", "category"]], on="prompt_id")
labels["needs_frontier"] = ~labels.cheap_pass  # router target: will the cheap model fail?

labels.to_parquet(OUT, index=False)
print(f"{len(labels)} labelled prompts (of {len(prompts)}) -> {OUT}")
print(labels.groupby("source")[["cheap_pass", "frontier_pass"]].mean().round(2))
print(f"cheap fails on {labels.needs_frontier.mean():.1%} of prompts")
