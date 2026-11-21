# llm-cost-router
ML router that sends each LLM request to the cheapest model that can handle it, measured by **cost per successful outcome**, not just token spend.

Like a triage nurse: easy requests go to a cheap model (`gpt-oss-20b`), hard ones to a frontier model (`gpt-oss-120b`), and an LLM judge checks the outcomes. Built entirely on free tiers ($0).

> **Status: v1 results on 350 labelled prompts** (all 70 GSM8K + 280 judged, balanced across all 13 categories). All 910 sampled prompts have answers from both models. Judging the remaining 560 is limited by the Gemini free tier (20 requests/day, often overloaded), so the numbers may shift as more labels arrive.

## Results (v1, 350 labelled prompts)

![cost vs quality](results/cost_vs_quality_x1.png)

| Strategy (real Groq prices) | Success rate | Cost per success | Savings vs. always-frontier |
|---|---|---|---|
| Always cheap (20b) | 68.9% | $0.000141 | 59% |
| Always frontier (120b) | 76.3% | $0.000308 | — |
| Rule baseline (long or number-heavy → frontier) | 72.0% | $0.000217 | 34% |
| Logistic-regression router (quality setting) | 74.3% | $0.000238 | 25% |
| Oracle (perfect router, upper bound) | 81.1% | $0.000141 | 51% |

**What I learned from the numbers**
- **The router learns a weak signal so far:** ROC-AUC is 0.62 (logreg) vs. 0.54 (rule). It keeps about 2/3 of the frontier model's quality gain for 25% less cost.
- **The price gap matters more than the router.** On Groq the frontier model is only 2× the price of the cheap one, so always-cheap has the best cost per success. With a premium frontier model at 10× the price ([what-if chart](results/cost_vs_quality_x10.png)), every routing strategy saves 40–54%.
- **Routing can't fix prompts that both models fail.** The cheap model fails 31% of prompts, but the frontier model also fails 19% of them. It rescues only 12%. The next iteration trains on "frontier rescues this prompt" instead of "cheap fails".
- **Retargeting needs more data first.** Training on "frontier rescues it" (cheap fails *and* frontier passes) scored worse (ROC-AUC 0.56 vs. 0.62), because there are only 43 such prompts among the 350. It's the right target once more prompts are labelled.
- **The oracle (81%) beats always-frontier (76%),** because on 5% of prompts the cheap model passes and the frontier model fails. Bigger isn't always better.

## How it works
1. **Data (Phase 1):** 2,178 public prompts: Databricks Dolly (8 task types), GSM8K math, and business tasks built from Tavily/Exa search snippets with 4 fixed templates. The v1 sample is 910 prompts, 70 per category.
2. **Labels (Phase 2):** each prompt goes to both models via Groq (`reasoning_effort=low`, 1,000 tokens max, concise system prompt, so 0% of answers get truncated).
   - **GSM8K:** graded by exact match on the final number.
   - **Everything else:** graded by Gemini 3.8 Flash. The judge is blinded (no model names), shuffled, and grades each answer pass/fail with the human reference answer. It must list unsupported claims and factual errors *before* its verdict.
   - **Judge validation:** a cheaper judge (Flash-Lite) passed answers that I verified had errors, so it was rejected. 96 blinded hand labels measure how often the judge agrees with a human (`python judge_agreement.py labels/hand_labels.csv your_label`).
   - **Second-opinion check:** Claude also graded the same 96 answers (`labels/claude_labels.csv`; this is an LLM, not a human). On the 56 answers both have graded, it agrees with the Gemini judge 82% of the time. Most disagreements are long brainstorming answers with invented details (fake apps, wrong prices) that the judge passed.
3. **Router (Phase 3):** features are local MiniLM sentence embeddings (384 dims) plus 6 simple text stats. Prompt category is deliberately *not* used, since real traffic doesn't arrive labelled. Models: a rule baseline, logistic regression, and gradient-boosted trees, all scored with 5-fold out-of-fold predictions.
4. **Evaluation (Phase 4):** success rate, total cost and cost per success for each strategy, at two thresholds: "match frontier quality within 2 pts" and "best $/success". Costs are modelled from Groq's published per-token prices on real token counts.
5. **Demo (Phase 5):** `streamlit run app.py` takes a prompt and shows P(cheap fails), the chosen model and the estimated savings. It can also run the prompt live.

Every API response is cached in DuckDB by `(prompt hash, model, settings)`, so no call is ever paid for twice and every run can resume where it stopped.

## Limitations and next steps
- **Small labelled set:** with 350 prompts the router's AUC has wide error bars. Judging the remaining 560 prompts (already answered and cached) is the cheapest improvement.
- **Judge reliability:** on 56 answers graded by both, an independent LLM second opinion agreed with the Gemini judge 82% of the time. The judge misses invented details in long answers. Human hand labels (`labels/hand_labels.csv`) are the final check.
- **Modelled costs:** costs use Groq's published per-token prices on real token counts. Free tiers aren't billed.
- **Next:** more labels → "frontier rescues" target → per-category thresholds → a third, cheaper tier.

## Roadmap
- [x] Phase 0: Setup — folder structure, `.env`, first "hello world" call to Groq and Gemini
- [x] Phase 1: Data — collect ~2–3k prompts (Hugging Face + business-style prompts via Tavily/Exa)
- [x] Phase 2: Labels — run prompts through both models, judge with Gemini, validate judge on ~100 hand labels *(910/910 answered; 350 labelled; human hand labels pending)*
- [x] Phase 3: Router — baseline rules → logistic regression on embeddings → gradient-boosted trees
- [x] Phase 4: Evaluation — cost-vs-quality curve, cost per successful outcome vs. baselines
- [ ] Phase 5: Demo — Streamlit app, deployed free *(app built; deploy pending)*
- [ ] Phase 6: Write-up — README results, LinkedIn post, update resume with real numbers *(README done; post and resume drafts in `docs/writeup_drafts.md`)*

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# create .env with GROQ_API_KEY and GEMINI_API_KEY (never commit it)
python hello_models.py                      # check both APIs work

# full pipeline (resumable; respects free-tier limits)
python collect_dolly.py && python collect_gsm8k.py
python collect_business_search.py && python build_business_prompts.py && python combine_prompts.py
python select_sample.py
python run_models.py data/sample_v1_ids.txt  # ~2–3 days on Groq's free tier
python judge.py data/sample_v1_ids.txt       # 18 calls/day on Gemini's free tier
python embed_prompts.py && python build_labels.py
python train_router.py && python evaluate.py && python evaluate.py 10
streamlit run app.py
```
