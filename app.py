# app.py

import streamlit as st
import json
from pathlib import Path

from routing_engine import RoutingEngine
from router.router_llm import RouterLLM


# ----------------------------
# Constants
# ----------------------------

HISTORY_FILE = Path("chat_history.json")


# ----------------------------
# App setup
# ----------------------------

st.set_page_config(
    page_title="LLM Router Demo",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Dynamic Parameter Allocation")
st.caption("Task-aware routing with optimized parameter planning")


# ----------------------------
# Persistence helpers
# ----------------------------

def load_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


# ----------------------------
# Initialize system
# ----------------------------

@st.cache_resource
def init_engine():
    return RoutingEngine(), RouterLLM()


engine, router = init_engine()


# ----------------------------
# Session state
# ----------------------------

if "history" not in st.session_state:
    st.session_state.history = load_history()

if "current_query" not in st.session_state:
    st.session_state.current_query = ""


# ----------------------------
# Submit handler
# ----------------------------

def handle_submit():
    query = st.session_state.current_query.strip()
    if not query:
        return

    # Step 1: Router decision (with context from history)
    decision = router.route(query, history=st.session_state.history)

    # Step 2: Final response (with context from history)
    response = engine.run(query, history=st.session_state.history)

    entry = {
        "query": query,
        "decision": decision.model_dump(),
        "response": response.content,
    }

    # Update in-memory history
    st.session_state.history.append(entry)

    # Persist to disk
    save_history(st.session_state.history)

    # Clear input for next prompt
    st.session_state.current_query = ""


# ----------------------------
# Input UI
# ----------------------------

st.text_input(
    "Enter your query",
    placeholder="e.g. Write a Python function to reverse a string",
    key="current_query",
    on_change=handle_submit,  # Enter key submits
)

st.button("Run", on_click=handle_submit)


# ----------------------------
# Display history
# ----------------------------

st.divider()
st.subheader("🗂️ Chat History")

if not st.session_state.history:
    st.info("No queries yet. Enter a prompt above to get started.")
else:
    for idx, item in enumerate(reversed(st.session_state.history), 1):
        interaction_no = len(st.session_state.history) - idx + 1

        with st.expander(
            f"Interaction #{interaction_no}: {item['query']}",
            expanded=True,
        ):
            st.markdown("### 🔍 Optimized Router Decision")
            st.code(
                json.dumps(item["decision"], indent=2),
                language="json",
            )

            st.markdown("### 🤖 Model Response")
            st.write(item["response"])


# ----------------------------
# Optional: clear history
# ----------------------------

st.divider()
if st.button("🗑️ Clear History"):
    st.session_state.history = []
    save_history([])
    st.success("History cleared.")
