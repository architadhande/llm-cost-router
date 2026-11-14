# CLAUDE.md — LLM Cost-per-Outcome Router

## Who I am
I'm Archita, a data analyst (MS Analytics, USC) rebuilding hands-on ML and LLM skills.
I know Python, SQL, pandas and basic ML (I've built classification models before),
but I'm new to LLM APIs, embeddings, and LLM evaluation. This project is how I learn them.

## What we're building
An ML "router" that looks at each incoming LLM request and predicts whether a cheap
small model can handle it or whether it needs an expensive frontier model.
We prove it works by measuring **cost per successful outcome**, not just token spend.

Analogy: a hospital triage nurse. Easy cases go to the general doctor (cheap model),
hard cases go to the specialist (frontier model), and a judge checks outcomes.

## Stack (everything must stay $0)
- Cheap model: Groq `openai/gpt-oss-20b` (Groq retired `llama-3.1-8b-instant`)
- Frontier model: Groq `openai/gpt-oss-120b` (`llama-3.3-70b-versatile` no longer offered)
- Judge: Google Gemini Flash (`gemini-3.8-flash`) via AI Studio (different model family than the models being judged)
- Data collection: Tavily + Exa APIs (free monthly credits — use basic search depth only)
- Datasets: Hugging Face
- Embeddings: `sentence-transformers` running locally
- ML: scikit-learn, XGBoost, pandas
- Storage/cache: DuckDB (cache EVERY model response so we never re-call an API for the same prompt)
- Demo: Streamlit
- Costs are MODELED: we price each call using providers' published per-token rates,
  since free tiers aren't billed.

## How to work with me (most important section)
1. **Teach before you code.** Before each new piece, explain the concept in plain
   language (2–5 sentences, with an analogy if it helps) and why we need it.
2. **Small steps.** One small, runnable step at a time. Never write more than ~40 lines
   of new code in one go without stopping to explain it.
3. **Let me write some of it.** At least once per session, describe what a function should
   do and let me write it myself. Then review my code kindly but honestly.
4. **Quiz me lightly.** After a concept, ask me one quick question to check I understood.
5. **Keep a learning log.** After each session, append to `LEARNING.md`: the concepts we
   covered, in my own words where possible, plus one "interview answer" I could give about it.
6. **Explain errors.** When something breaks, explain what the error means before fixing it.
7. **Respect free-tier limits.** Add rate limiting and retries to all API calls. Warn me
   before running anything that will make more than ~100 API calls.

## Rules
- API keys live ONLY in `.env` (loaded with python-dotenv). Never print, log, or commit keys.
  Confirm `.env` is in `.gitignore` before the first commit.
- Commit to git after each working step with a clear message, so my GitHub shows steady progress.
- Keep the README updated with the roadmap checkboxes as phases complete.
- Use only public or synthetic prompts — never real company data (Gemini free tier may use data for training).

## Roadmap
- [ ] Phase 0: Setup — folder structure, `.env`, first "hello world" call to Groq and Gemini
- [ ] Phase 1: Data — collect ~2–3k prompts (Hugging Face + business-style prompts via Tavily/Exa)
- [ ] Phase 2: Labels — run prompts through both models, judge with Gemini, validate judge on ~100 hand labels
- [ ] Phase 3: Router — baseline rules → logistic regression on embeddings → XGBoost
- [ ] Phase 4: Evaluation — cost-vs-quality curve, cost per successful outcome vs. baselines
- [ ] Phase 5: Demo — Streamlit app, deployed free
- [ ] Phase 6: Write-up — README results, LinkedIn post, update resume with real numbers
