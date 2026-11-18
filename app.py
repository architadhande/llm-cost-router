"""app.py — Phase 5: Streamlit demo of the LLM cost router.  Run: streamlit run app.py"""

import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from features import compute_features, load_embedder

CHEAP, FRONTIER = "openai/gpt-oss-20b", "openai/gpt-oss-120b"
PRICES = {CHEAP: (0.075, 0.30), FRONTIER: (0.15, 0.60)}  # USD per 1M tokens (Groq, Nov 2026)
AVG_OUTPUT_TOKENS = {CHEAP: 250, FRONTIER: 300}  # from our pilot runs (concise-v1 settings)
EXAMPLES = {
    "Simple fact": "What is the capital of Australia?",
    "Math word problem": "A store sells pens at 3 for $4. Maya buys 18 pens and pays with a "
                         "$50 bill. How much change does she get?",
    "Business risk analysis": "Based only on the text below, what are the top 3 business "
                              "risks, and what evidence supports each?\n\nText:\nAcme Corp "
                              "reported Q3 revenue of $1.2B, down 8% year over year, citing "
                              "weaker demand in Europe and a delayed product launch.",
    "Brainstorming": "What are some fun ways to entertain family visiting Boston?",
}

st.set_page_config(page_title="LLM Cost Router", page_icon="🔀", layout="wide")


@st.cache_resource
def load_embedder_cached():
    return load_embedder()


@st.cache_resource
def load_router(name):
    path = Path(f"models/router_{name}.joblib")
    return joblib.load(path) if path.exists() else None


def call_cost(model, prompt):
    price_in, price_out = PRICES[model]
    return (len(prompt) / 4 * price_in + AVG_OUTPUT_TOKENS[model] * price_out) / 1e6


def groq_key():
    try:
        return st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    except Exception:  # no secrets file locally
        from dotenv import load_dotenv
        load_dotenv(".env")
        return os.environ.get("GROQ_API_KEY")


st.title("🔀 LLM Cost Router")
st.caption("Predicts whether a cheap model (gpt-oss-20b) can handle a prompt, or whether it "
           "needs the frontier model (gpt-oss-120b). Like a triage nurse for LLM requests.")
tab_try, tab_results = st.tabs(["Try the router", "Results"])

with tab_try:
    left, right = st.columns([3, 2])
    with right:
        router_name = st.selectbox("Router", ["logreg", "boosted_trees"])
        threshold = st.slider("Send to frontier if P(cheap fails) is above", 0.0, 1.0, 0.5, 0.05,
                              help="Lower = safer but more expensive. Higher = cheaper, more risk.")
        example = st.selectbox("Example prompt", ["(write your own)", *EXAMPLES])
    with left:
        prompt = st.text_area("Prompt", EXAMPLES.get(example, ""), height=180)

    router = load_router(router_name)
    if router is None:
        st.warning("No trained router yet. Run `python train_router.py` first.")
    elif prompt.strip():
        feats = compute_features([prompt], load_embedder_cached())[router["columns"]]
        p_fail = float(router["model"].predict_proba(feats)[0, 1])
        chosen = FRONTIER if p_fail > threshold else CHEAP
        c1, c2, c3 = st.columns(3)
        c1.metric("P(cheap model fails)", f"{p_fail:.0%}")
        c2.metric("Route to", "Frontier (120b)" if chosen == FRONTIER else "Cheap (20b)")
        saved = 1 - call_cost(chosen, prompt) / call_cost(FRONTIER, prompt)
        c3.metric("Est. cost vs always-frontier", f"-{saved:.0%}" if saved > 0 else "same")
        st.caption(f"Router trained on {router['n_train']} labelled prompts. Costs are modelled "
                   "from Groq's published per-token prices.")

        if groq_key() and st.button(f"Run it on {chosen.split('/')[-1]}"):
            from groq import Groq
            with st.spinner("Calling Groq..."):
                r = Groq(api_key=groq_key(), timeout=60).chat.completions.create(
                    model=chosen, reasoning_effort="low", max_completion_tokens=1000,
                    messages=[{"role": "system", "content": "Answer concisely: under 300 "
                               "words unless the task needs more."},
                              {"role": "user", "content": prompt}])
            st.markdown(r.choices[0].message.content)

with tab_results:
    for mult in ["1", "10"]:
        img, csv = Path(f"results/cost_vs_quality_x{mult}.png"), Path(f"results/summary_x{mult}.csv")
        if img.exists():
            st.subheader("Real Groq prices (frontier ≈ 2× cheap)" if mult == "1"
                         else "What-if: premium frontier model (10× price)")
            st.image(str(img))
            st.dataframe(pd.read_csv(csv, index_col=0).round(4))
    if not Path("results/cost_vs_quality_x1.png").exists():
        st.info("No results yet. Run `python evaluate.py` and `python evaluate.py 10`.")
