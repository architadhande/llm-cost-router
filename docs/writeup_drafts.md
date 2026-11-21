# Write-up drafts (update the numbers when labelling finishes)

Numbers below are **preliminary** (350 labelled prompts, Nov 20 2026). Re-check them against `results/summary_x1.csv` and `results/summary_x10.csv` before posting.

## LinkedIn post

I built an ML "triage nurse" for LLM requests, and the most useful thing it taught me was when *not* to use it.

The idea: send easy prompts to a cheap model and hard ones to a frontier model, then measure **cost per successful outcome**, not just token spend.

What I built ($0, free tiers only):
• 910 public prompts (Dolly, GSM8K math, and business tasks from live search snippets), answered by gpt-oss-20b and gpt-oss-120b
• Gemini as a blinded LLM judge, forced to list evidence before giving a verdict. I rejected a cheaper judge after it passed answers with errors I had checked by hand.
• A router on local sentence embeddings (logistic regression and gradient-boosted trees), evaluated out-of-fold

What I found:
• At real prices the frontier model is only 2× the cost, so "always cheap" had the best cost per success. Routing pays off when the gap is large: at a 10× gap the router saved 40–54% vs. always-frontier.
• The cheap model failed 31% of prompts, but the frontier model failed 19% of those too. Only 12% were "rescues". Predicting *which prompts the bigger model actually fixes* is the real problem.
• Free tiers shape the design. Caching every call by prompt hash and running resumable jobs sized to daily quotas is what made it possible at $0.

Code + results: <GitHub link> · Demo: <Streamlit link>

#MachineLearning #LLM #DataScience

## Resume bullets

- Built an LLM cost-per-outcome router (Python, scikit-learn, sentence-transformers, DuckDB, Streamlit) that predicts whether a cheap model can handle a prompt. Evaluated it on 910 prompts against always-cheap, always-frontier and oracle baselines.
- Designed an LLM-as-judge labelling pipeline (blinded, reference-guided, evidence-before-verdict). Validated it against hand labels and rejected a cheaper judge that missed verified errors.
- Showed routing savings depend on the price gap: 40–54% lower cost than always-frontier at a 10× price gap, but no gain at a 2× gap. Identified that only 12% of cheap-model failures are recoverable by the larger model.
- Ran the whole project at $0 on free tiers, using content-hash caching, retries with backoff and quota-aware resumable batch jobs.

## Interview answer (60 seconds)

"I wanted to know whether a cheap LLM could handle most requests, and to measure that by cost per successful answer, not token count. I built a labelled dataset by running 910 prompts through two models and grading them with a blinded LLM judge that I validated against hand labels. Then I trained a router on prompt embeddings. The honest result: with only a 2× price gap, always using the cheap model wins. Routing pays off at bigger gaps, around 40–54% savings at 10×. The bigger lesson was the target: most cheap-model failures are failures for the big model too, so the router should predict 'the big model would fix this', not 'the small model fails'."
