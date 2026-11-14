"""llm_cache.py — DuckDB cache: never pay for the same (prompt, model, settings) twice."""

from pathlib import Path

import duckdb

DB = Path("data/cache.duckdb")
COLUMNS = ["prompt_id", "model", "config", "answer", "finish_reason",
           "input_tokens", "output_tokens", "reasoning_tokens", "seconds"]

DB.parent.mkdir(exist_ok=True)
con = duckdb.connect(str(DB))
con.execute("""
    CREATE TABLE IF NOT EXISTS responses (
        prompt_id VARCHAR, model VARCHAR, config VARCHAR,
        answer VARCHAR, finish_reason VARCHAR,
        input_tokens INTEGER, output_tokens INTEGER, reasoning_tokens INTEGER,
        seconds DOUBLE,
        created_at TIMESTAMP DEFAULT current_timestamp,
        PRIMARY KEY (prompt_id, model, config)
    )
""")


def get(prompt_id, model, config):
    """Return the cached row as a dict, or None if we have never made this call."""
    row = con.execute(
        "SELECT * FROM responses WHERE prompt_id = ? AND model = ? AND config = ?",
        [prompt_id, model, config],
    ).fetchone()
    return dict(zip([d[0] for d in con.description], row)) if row else None


def put(result):
    """Save one response immediately, so a crash never loses a paid-for call."""
    con.execute(
        f"INSERT OR REPLACE INTO responses ({', '.join(COLUMNS)}) "
        f"VALUES ({', '.join('?' * len(COLUMNS))})",
        [result[c] for c in COLUMNS],
    )
