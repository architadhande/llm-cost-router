"""api_utils.py — shared helpers for every script that calls an API."""

import time


def with_retries(call, attempts=3, first_wait=2.0):
    """Run call(). On a temporary failure, wait and retry, doubling the wait each time."""
    wait = first_wait
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as e:
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            # 4xx = our request is wrong, retrying won't help (except 429 = "slow down")
            permanent = isinstance(status, int) and 400 <= status < 500 and status != 429
            if permanent or attempt == attempts:
                raise
            print(f"  attempt {attempt} failed ({status or type(e).__name__}), retrying in {wait:.0f}s")
            time.sleep(wait)
            wait *= 2
