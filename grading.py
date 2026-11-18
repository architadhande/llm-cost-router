"""grading.py — exact-match grading for GSM8K (no judge needed)."""

import re

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def to_number(text):
    """'1,200' -> 1200.0, '$50' -> 50.0; None if there is no number."""
    found = NUMBER.findall(text.replace(",", ""))
    return float(found[-1]) if found else None


def grade_gsm8k(model_answer: str, final_answer: str) -> bool:
    """True if the LAST number in model_answer equals final_answer."""
    got, expected = to_number(model_answer or ""), to_number(final_answer)
    return got is not None and expected is not None and abs(got - expected) < 1e-6


if __name__ == "__main__":
    assert grade_gsm8k("15 + 35 = 50, so they bought 50 apples altogether.", "50") is True
    assert grade_gsm8k("The total cost is $1,200.", "1200") is True
    assert grade_gsm8k("Diane bought 35 apples, so together they have 50.", "35") is False
    # extra edge cases: markdown/LaTeX formatting, decimals, no number at all
    assert grade_gsm8k("**Answer: \\boxed{50}**", "50") is True
    assert grade_gsm8k("That is 12.0 dollars.", "12") is True
    assert grade_gsm8k("I am not sure.", "50") is False
    print("All 6 tests passed")
