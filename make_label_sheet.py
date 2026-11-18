"""make_label_sheet.py — Phase 2: build a blinded sheet of 96 answers for hand labelling.

4 prompts per non-GSM8K category (pilot prompts first) x 2 models = 96 answers.
GSM8K is skipped: it is graded exactly by grade_gsm8k, no human or judge needed.
"""

import sys
from pathlib import Path

import pandas as pd

import llm_cache
from run_models import CHEAP, CONFIG, FRONTIER

PER_CATEGORY = 4  # 96 answers: ~10 expected FAILs, enough to check the judge catches them
SEED = 7
SHEET = Path("labels/hand_labels.csv")  # you fill in this one
KEY = Path("labels/hand_labels_key.csv")  # which model wrote each answer (don't peek)

if SHEET.exists() and "--force" not in sys.argv:
    sys.exit(f"{SHEET} already exists - not overwriting your labels. Use --force to rebuild.")

df = pd.read_parquet("data/prompts_all.parquet")
sample_ids = set(Path("data/sample_v1_ids.txt").read_text().split())
pilot_ids = set(Path("data/pilot_ids.txt").read_text().split())
df = df[df.prompt_id.isin(sample_ids) & (df.source != "gsm8k")].copy()
df["in_pilot"] = df.prompt_id.isin(pilot_ids)

# Pilot prompts first (they already have judge verdicts to compare), then random
chosen = (df.sample(frac=1, random_state=SEED)
            .sort_values("in_pilot", ascending=False, kind="stable")
            .groupby("category").head(PER_CATEGORY))

rows = []
for p in chosen.itertuples():
    for model in (CHEAP, FRONTIER):
        rows.append({"prompt_id": p.prompt_id, "model": model, "category": p.category,
                     "task": p.prompt, "reference_answer": p.reference_answer or "",
                     "answer": llm_cache.get(p.prompt_id, model, CONFIG)["answer"]})

sheet = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
sheet.insert(0, "label_id", [f"L{i:03d}" for i in range(1, len(sheet) + 1)])
sheet["your_label"] = ""  # PASS or FAIL
sheet["notes"] = ""       # optional: why

SHEET.parent.mkdir(exist_ok=True)
sheet[["label_id", "prompt_id", "model"]].to_csv(KEY, index=False)
cols = ["label_id", "category", "task", "reference_answer", "answer", "your_label", "notes"]
sheet[cols].to_csv(SHEET, index=False, encoding="utf-8-sig")  # utf-8-sig: opens cleanly in Excel
print(f"{len(sheet)} answers ({chosen.in_pilot.sum()} pilot prompts) -> {SHEET}")
print(f"Blinding key -> {KEY}")
