"""
Agentic Harness - Streamlit frontend (improved version)

HOW STREAMLIT WORKS (read this once):
- Every st.something(...) call draws one thing on the page, top to bottom.
- Whenever the user clicks a button or changes a widget, Streamlit re-runs
  this whole file from top to bottom.
- st.session_state is a dictionary that remembers values between re-runs.

We only change the frontend. harness.py and cache.py are NOT touched.
"""
import re
import json
import sqlite3

import pandas as pd
import streamlit as st

from cache import init_db, DB_NAME
from harness import run_harness

# ---------------------------------------------------------------- setup
init_db()
st.set_page_config(page_title="Agentic Harness", page_icon="🛡️", layout="wide")

PASS_SCORE = 0.80  # same threshold the backend uses

EXAMPLES = {
    "JSON format": "Return ONLY valid JSON with keys 'language' and 'year' for Python.",
    "Word limit": "Explain binary search in fewer than 35 words.",
    "Bullet points": "List exactly 3 bullet points explaining SQL indexes. No intro or outro text.",
    "SQL code": "Write a SQL query to count employees per department in 'emp'.",
}


# ---------------------------------------------------------------- helpers
def score_icon(score: float) -> str:
    """Traffic-light emoji for a score."""
    if score >= PASS_SCORE:
        return "🟢"
    if score >= 0.5:
        return "🟡"
    return "🔴"


def set_prompt(text: str):
    """Called when an example button is clicked: fills the text box."""
    st.session_state["prompt_text"] = text


def show_answer(answer: str):
    """Show JSON nicely if the answer contains JSON, otherwise show as text."""
    start, end = answer.find("{"), answer.rfind("}") + 1
    if start != -1 and end > 0:
        try:
            st.json(json.loads(answer[start:end]))
            return
        except json.JSONDecodeError:
            pass
    st.markdown(answer)


def show_trace(trace: list):
    """Show each attempt as its own bordered box instead of a plain list."""
    pattern = re.compile(r"Attempt (\d+): Score = ([\d.]+) \| Feedback: (.*)", re.S)
    for step in trace:
        match = pattern.match(step)
        with st.container(border=True):
            if match:
                number, score, feedback = match.groups()
                score = float(score)
                st.markdown(f"**Attempt {number}** &nbsp; {score_icon(score)} score **{score:.2f}**")
                st.caption(feedback)
            else:
                st.markdown(f"⚡ {step}")  # e.g. the "Cache HIT" message


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    use_cache = st.toggle("Semantic cache", value=True,
                          help="Reuse answers for prompts with a similar meaning.")
    max_retries = st.slider("Max retries", 0, 3, 2,
                            help="How many times the harness may fix a bad answer.")
    st.divider()
    st.subheader("Try an example")
    for label, text in EXAMPLES.items():
        st.button(label, on_click=set_prompt, args=(text,), width="stretch")
    st.divider()
    st.caption(f"A score of {PASS_SCORE:.2f} or higher counts as reliable.")

# ---------------------------------------------------------------- header
st.title("Agentic Harness")
st.caption("An LLM answer is generated, checked, fixed if needed, and cached once it is reliable.")

tab_play, tab_stats, tab_how = st.tabs(["Playground", "Analytics", "How it works"])

# ================================================================ TAB 1
with tab_play:
    prompt = st.text_area(
        "Your prompt",
        key="prompt_text",
        height=110,
        placeholder="Try a strict instruction, e.g. 'List exactly 3 bullet points about caching.'",
    )
    run_clicked = st.button("Run", type="primary")

    if run_clicked and not prompt.strip():
        st.warning("Type a prompt first, or pick an example from the sidebar.")

    if run_clicked and prompt.strip():
        with st.spinner("Generating, checking and scoring the answer..."):
            try:
                result = run_harness(prompt, max_retries=max_retries, use_cache=use_cache)
            except Exception as e:
                st.error(f"The harness failed: {e}")
                result = None

        if result:
            # --- verdict banner
            if result["cached"]:
                st.info(f"Served from cache (similarity {result['similarity']}). No API call was needed.")
            elif result["final_score"] >= PASS_SCORE:
                if result["retries"] > 0:
                    st.success(f"Fixed after {result['retries']} retr{'y' if result['retries'] == 1 else 'ies'}. The answer is now reliable.")
                else:
                    st.success("Reliable on the first attempt.")
            else:
                st.warning("The answer did not reach the reliability threshold. It was not cached.")

            # --- metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("First attempt", f"{result['initial_score']:.2f}")
            c2.metric("Final score", f"{result['final_score']:.2f}",
                      delta=round(result["final_score"] - result["initial_score"], 2))
            c3.metric("Retries", result["retries"])
            c4.metric("Latency", f"{result['latency_ms']:.0f} ms",
                      delta="cache hit" if result["cached"] else "live API",
                      delta_color="off")

            # --- answer and trace side by side
            left, right = st.columns([3, 2])
            with left:
                st.subheader("Verified answer")
                with st.container(border=True):
                    show_answer(result["answer"])
            with right:
                st.subheader("How it was judged")
                show_trace(result["trace"])

# ================================================================ TAB 2
with tab_stats:
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM logs", conn)
    conn.close()

    if df.empty:
        st.info("No runs yet. Run a prompt in the Playground (with the cache on) and the history appears here.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total runs", len(df))
        m2.metric("Average score", f"{df['initial_score'].mean():.2f} → {df['final_score'].mean():.2f}")
        m3.metric("Cache hit rate", f"{df['cached'].mean() * 100:.0f}%")
        m4.metric("Average retries", f"{df['retries'].mean():.2f}")

        st.subheader("Score before and after correction")
        chart_df = df[["initial_score", "final_score"]].rename(
            columns={"initial_score": "First attempt", "final_score": "Final"})
        st.line_chart(chart_df)

        st.subheader("Average latency: live API vs cache")
        by_source = (df.assign(Source=df["cached"].map({0: "Live API", 1: "Cache hit"}))
                       .groupby("Source")["latency_ms"].mean())
        st.bar_chart(by_source)

        st.subheader("Recent runs")
        st.dataframe(df.tail(15).iloc[::-1], width="stretch", hide_index=True)

# ================================================================ TAB 3
with tab_how:
    st.subheader("What happens when you click Run")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        with st.container(border=True):
            st.markdown("**1. Cache check**")
            st.caption("The prompt is turned into numbers (an embedding). If a past prompt means almost the same thing, its saved answer is returned instantly.")
    with s2:
        with st.container(border=True):
            st.markdown("**2. Generate**")
            st.caption("If nothing is cached, the Llama-family model on Groq writes an answer.")
    with s3:
        with st.container(border=True):
            st.markdown("**3. Check and score**")
            st.caption("Simple rules run first (for example, does the JSON parse?). Then a second model call grades the answer from 0 to 1.")
    with s4:
        with st.container(border=True):
            st.markdown("**4. Fix and save**")
            st.caption("Below 0.80, the feedback is sent back and the model tries again. Only answers at 0.80 or above are cached.")