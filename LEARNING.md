# Learning Log

## Session 1 — Phase 0: Hello, models (2026-11-12)

**Concepts covered**
- **API keys in `.env`:** keys live in a file that is gitignored and loaded with python-dotenv, so they never end up in code or on GitHub.
- **Retries with exponential backoff:** temporary server errors (429 rate limit, 503 overloaded, 504 timeout) are worth retrying after waiting 2s, 4s, 8s… Client errors like 401 (bad key) or 400 (bad request) are permanent, so retrying just wastes time.
- **Token usage:** each response reports input tokens, output tokens and reasoning ("thinking") tokens. Reasoning tokens are billed as output, so a "short" answer can still cost a lot. Gemini used 523 of its 585 output tokens on thinking.
- **Latency varies a lot:** Groq answered in ~0.5s; Gemini took ~8s after two 504 retries.
- **Python debugging:** `NameError` means using a name that was never defined. A file edit only counts once it's saved to the path the program actually reads.

**Interview answer**
> "Every API call in my project goes through a retry wrapper with exponential backoff. It retries transient failures like 429s and 5xx errors but fails fast on other 4xx errors, because those mean the request itself is wrong. On day one Gemini returned 503 'high demand' several times in a row, and the wrapper is what kept that from crashing the pipeline."

**To fill in (in your own words):** quiz answer — why retry a 503 but not a 401?
