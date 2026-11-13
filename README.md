# llm-cost-router
ML router that sends each LLM request to the cheapest model that can handle it, measured by cost per successful outcome

## Roadmap
- [x] Phase 0: Setup — folder structure, `.env`, first "hello world" call to Groq and Gemini
- [ ] Phase 1: Data — collect ~2–3k prompts (Hugging Face + business-style prompts via Tavily/Exa)
- [ ] Phase 2: Labels — run prompts through both models, judge with Gemini, validate judge on ~100 hand labels
- [ ] Phase 3: Router — baseline rules → logistic regression on embeddings → XGBoost
- [ ] Phase 4: Evaluation — cost-vs-quality curve, cost per successful outcome vs. baselines
- [ ] Phase 5: Demo — Streamlit app, deployed free
- [ ] Phase 6: Write-up — README results, LinkedIn post, update resume with real numbers

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# create .env with GROQ_API_KEY and GEMINI_API_KEY
python hello_models.py
```
