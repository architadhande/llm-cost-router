"""train_router.py — Phase 3: predict "will the cheap model fail?" for each prompt.

Every prompt gets an out-of-fold score (5-fold CV), so all labelled prompts are used for
evaluation without any model grading prompts it was trained on.
Output: data/router_scores.parquet (one score column per router) for evaluate.py.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT = Path("data/router_scores.parquet")
MODELS = Path("models")  # final routers for the app (small; committed to git)

labels = pd.read_parquet("data/labels.parquet")
features = pd.read_parquet("data/features.parquet")
d = labels.merge(features, on="prompt_id")
TARGET = sys.argv[1] if len(sys.argv) > 1 else "needs_frontier"  # or "rescue"
X, y = d.drop(columns=labels.columns), d[TARGET].astype(int)
print(f"target={TARGET}: {len(d)} prompts, {y.sum()} positives ({y.mean():.1%})")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
routers = {
    "logreg": make_pipeline(StandardScaler(),
                            LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)),
    # gradient-boosted trees (same idea as XGBoost; sklearn's version needs no system OpenMP)
    "boosted_trees": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                    class_weight="balanced", random_state=SEED),
}

scores = d[["prompt_id"]].copy()
# Rule baseline (no learning): long prompts or number-heavy prompts go to the big model
scores["rule"] = ((d.n_chars > d.n_chars.median()) | (d.n_digits >= 5)).astype(float)
for name, model in routers.items():
    scores[name] = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

print(f"\n{'router':10} {'ROC-AUC':>8} {'PR-AUC':>8}   (PR-AUC of random guessing = {y.mean():.3f})")
for name in ["rule", *routers]:
    print(f"{name:10} {roc_auc_score(y, scores[name]):8.3f} {average_precision_score(y, scores[name]):8.3f}")

scores.to_parquet(OUT, index=False)
print(f"\nSaved out-of-fold scores -> {OUT}")

# Final routers: train on ALL labelled prompts and save for the Streamlit app
MODELS.mkdir(exist_ok=True)
for name, model in routers.items():
    model.fit(X, y)
    joblib.dump({"model": model, "columns": list(X.columns), "n_train": len(X)},
                MODELS / f"router_{name}.joblib")
print(f"Saved final routers -> {MODELS}/")
