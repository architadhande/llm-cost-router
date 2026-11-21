"""judge_agreement.py — how often does the Gemini judge agree with a reference grader?

Usage: python judge_agreement.py labels/hand_labels.csv your_label
       python judge_agreement.py labels/claude_labels.csv claude_label
Reads the cache read-only, so it's safe to run while nothing else is writing.
"""

import sys

import duckdb
import pandas as pd

JUDGE_KEY = "gemini-3.8-flash|think-low|rubric-v2"  # same as judge.py (not imported: it opens the DB read-write)

path, column = sys.argv[1], sys.argv[2]
labels = pd.read_csv(path)
labels = labels[labels[column].astype(str).str.lower().isin(["pass", "fail"])]
key = pd.read_csv("labels/hand_labels_key.csv")
con = duckdb.connect("data/cache.duckdb", read_only=True)
judge = con.execute("SELECT prompt_id, model, passed AS judge_pass FROM judgments "
                    "WHERE judge = ? AND config LIKE '%concise-v1'", [JUDGE_KEY]).df()

m = labels.merge(key, on="label_id").merge(judge, on=["prompt_id", "model"])
m["ref_pass"] = m[column].str.lower().eq("pass")
print(f"{len(labels)} labelled answers, {len(m)} also judged")
if len(m):
    print(f"agreement: {(m.ref_pass == m.judge_pass).mean():.1%}")
    print(pd.crosstab(m.ref_pass, m.judge_pass, rownames=["reference"], colnames=["judge"]))
    print("\nby model (reference pass rate vs judge pass rate):")
    print(m.groupby("model")[["ref_pass", "judge_pass"]].mean().round(3))
