# Hand labels: how to fill in `hand_labels.csv`

These labels are the **gold standard** we use to check the Gemini judge.
Use the same rules the judge uses, so the comparison is fair.

For each row, read `task` (and `reference_answer` if there is one), then the `answer`.
Write **PASS** or **FAIL** in `your_label`.

**PASS** only if all of these are true:
1. The answer does what the task asks, completely (it isn't cut off or missing parts).
2. Nothing in it is factually wrong: check names, numbers, places and dates.
3. If the task includes a text or context, the answer uses **only** facts from that text.
   Opinions or outside facts added to a summary mean FAIL.
4. If there is a `reference_answer`, the answer doesn't contradict it.
   For brainstorming or creative tasks it can differ in content, as long as it has no errors.

Otherwise, **FAIL**. Add a few words in `notes` saying why (e.g. "wrong location for museum").

Tips:
- Don't open `hand_labels_key.csv`. It shows which model wrote each answer, and knowing
  that biases your labels.
- When you're unsure, it's fine to look things up. Write "unsure" in notes, then decide.
- Rows are in random order on purpose. 96 rows takes about 1.5–2 hours, and you can do it in chunks.
