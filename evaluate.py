"""evaluate.py — Phase 4: cost vs quality, and cost per successful outcome.

Usage: python evaluate.py [frontier_price_multiplier]
  multiplier 1 = real Groq prices (frontier ~2x cheap); try 10 for a premium frontier model.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # draw to a file, no window
import matplotlib.pyplot as plt  # noqa: E402

MULT = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
d = pd.read_parquet("data/labels.parquet").merge(pd.read_parquet("data/router_scores.parquet"))
d["frontier_cost"] *= MULT


def outcome(to_frontier):
    """Success rate, total cost and cost per success if `to_frontier` prompts use the big model."""
    passed = np.where(to_frontier, d.frontier_pass, d.cheap_pass)
    cost = np.where(to_frontier, d.frontier_cost, d.cheap_cost).sum()
    return {"success_rate": passed.mean(), "total_cost": cost,
            "cost_per_success": cost / max(passed.sum(), 1), "pct_frontier": np.mean(to_frontier)}


rows = {
    "always cheap": outcome(np.zeros(len(d), bool)),
    "always frontier": outcome(np.ones(len(d), bool)),
    # oracle: frontier only when cheap fails AND frontier passes (perfect knowledge)
    "oracle": outcome((~d.cheap_pass & d.frontier_pass).to_numpy()),
    "rule": outcome(d["rule"].to_numpy() > 0.5),
}
curves = {}
for router in ["logreg", "boosted_trees"]:
    pts = [outcome(d[router].to_numpy() > t) | {"threshold": t} for t in np.linspace(0, 1, 101)]
    curves[router] = pd.DataFrame(pts)
    c = curves[router]
    # Operating point A: cheapest threshold whose quality is within 2 pts of always-frontier
    ok = c[c.success_rate >= rows["always frontier"]["success_rate"] - 0.02]
    a = ok.loc[ok.total_cost.idxmin()]
    rows[f"{router} @quality (t={a.threshold:.2f})"] = a.drop("threshold").to_dict()
    # Operating point B: lowest cost per successful outcome
    b = c.loc[c.cost_per_success.idxmin()]
    rows[f"{router} @best $/success (t={b.threshold:.2f})"] = b.drop("threshold").to_dict()

table = pd.DataFrame(rows).T
base = rows["always frontier"]["total_cost"]
table["savings_vs_frontier"] = 1 - table.total_cost / base
pd.set_option("display.float_format", lambda v: f"{v:.4f}")
pd.set_option("display.width", 200)
print(f"Frontier price multiplier: {MULT:g}x   ({len(d)} prompts)\n")
print(table[["success_rate", "pct_frontier", "total_cost", "cost_per_success", "savings_vs_frontier"]])

fig, ax = plt.subplots(figsize=(7, 4.5))
for router, c in curves.items():
    ax.plot(c.total_cost, c.success_rate, label=router)
for name in ["always cheap", "always frontier", "oracle", "rule"]:
    ax.scatter(rows[name]["total_cost"], rows[name]["success_rate"], zorder=3)
    ax.annotate(name, (rows[name]["total_cost"], rows[name]["success_rate"]),
                textcoords="offset points", xytext=(5, -10), fontsize=8)
ax.set_xlabel(f"Total modelled cost, USD (frontier price x{MULT:g})")
ax.set_ylabel("Success rate")
ax.set_title("Router: cost vs quality")
ax.legend()
Path("results").mkdir(exist_ok=True)
out = Path(f"results/cost_vs_quality_x{MULT:g}.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
table.to_csv(f"results/summary_x{MULT:g}.csv")
print(f"\nSaved {out} and results/summary_x{MULT:g}.csv")
