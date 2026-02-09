"""
LLM Router - Cost & Effectiveness Dashboard

Streamlit dashboard for visualizing:
- Routing accuracy metrics
- Cost analysis (router vs single model)
- Latency distribution
- Fallback rate

Run with: streamlit run dashboard.py
"""

import streamlit as st
import json
from pathlib import Path

# Try to import plotly, but work without it
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ----------------------------
# Configuration
# ----------------------------

RESULTS_FILE = Path("evaluation/results.json")

# Model pricing for comparison (per 1M tokens)
MODEL_COSTS = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gemini-2.5-flash-preview-04-17": {"input": 0.15, "output": 0.60},
}


# ----------------------------
# Page Setup
# ----------------------------

st.set_page_config(
    page_title="LLM Router Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 LLM Router - Cost & Effectiveness Dashboard")
st.caption("Demonstrating Dynamic Parameter Allocation for Multitask LLM Specialization")


# ----------------------------
# Load Data
# ----------------------------

def load_results():
    """Load evaluation results from JSON file."""
    if not RESULTS_FILE.exists():
        return None
    
    with open(RESULTS_FILE, "r") as f:
        return json.load(f)


data = load_results()

if data is None:
    st.warning("⚠️ No evaluation results found. Run the evaluation first:")
    st.code("python evaluation/run_evaluation.py", language="bash")
    st.stop()


summary = data["summary"]
results = data["results"]


# ----------------------------
# Key Metrics Cards
# ----------------------------

st.subheader("📈 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Routing Accuracy",
        f"{summary['accuracy']:.1%}",
        help="Percentage of queries routed to the correct expert"
    )

with col2:
    st.metric(
        "Avg Latency",
        f"{summary['avg_latency']:.2f}s",
        help="Average total latency (router + expert)"
    )

with col3:
    st.metric(
        "Avg Cost/Query",
        f"${summary['avg_cost']:.6f}",
        help="Average cost per query (router + expert)"
    )

with col4:
    st.metric(
        "Fallback Rate",
        f"{summary['fallback_rate']:.1%}",
        help="Percentage of queries using fallback expert"
    )


# ----------------------------
# Cost Comparison
# ----------------------------

st.divider()
st.subheader("💰 Cost Comparison: Router vs Single Model")

# Calculate what it would cost with single models
total_samples = summary["total_samples"]

# Estimate tokens per query (average)
avg_input_tokens = 500
avg_output_tokens = 400

# Cost with different strategies
router_total = summary["total_cost"]

gpt4o_total = total_samples * (
    (avg_input_tokens / 1_000_000) * MODEL_COSTS["gpt-4o"]["input"] +
    (avg_output_tokens / 1_000_000) * MODEL_COSTS["gpt-4o"]["output"]
)

gpt4o_mini_total = total_samples * (
    (avg_input_tokens / 1_000_000) * MODEL_COSTS["gpt-4o-mini"]["input"] +
    (avg_output_tokens / 1_000_000) * MODEL_COSTS["gpt-4o-mini"]["output"]
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🎯 LLM Router (Our System)",
        f"${router_total:.4f}",
        help="Actual cost using our intelligent routing"
    )

with col2:
    savings_vs_gpt4o = ((gpt4o_total - router_total) / gpt4o_total * 100) if gpt4o_total > 0 else 0
    st.metric(
        "GPT-4o Only (Baseline)",
        f"${gpt4o_total:.4f}",
        delta=f"-{savings_vs_gpt4o:.1f}% savings with router",
        delta_color="inverse"
    )

with col3:
    st.metric(
        "GPT-4o-mini Only",
        f"${gpt4o_mini_total:.4f}",
        help="Cheapest option but lower quality"
    )

# Cost comparison chart
if PLOTLY_AVAILABLE:
    cost_data = {
        "Strategy": ["LLM Router", "GPT-4o Only", "GPT-4o-mini Only"],
        "Cost": [router_total, gpt4o_total, gpt4o_mini_total],
    }
    
    fig = px.bar(
        cost_data,
        x="Strategy",
        y="Cost",
        color="Strategy",
        color_discrete_sequence=["#00cc88", "#ff6b6b", "#4ecdc4"],
        title="Total Cost by Strategy"
    )
    fig.update_layout(showlegend=False, yaxis_title="Cost (USD)")
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# Accuracy by Category
# ----------------------------

st.divider()
st.subheader("🎯 Routing Accuracy by Task Type")

category_accuracy = summary["category_accuracy"]
category_stats = summary["category_stats"]

col1, col2, col3, col4 = st.columns(4)

categories = ["code_generation", "reasoning", "summarization", "general"]
columns = [col1, col2, col3, col4]
icons = ["💻", "🧠", "📝", "💬"]

for col, cat, icon in zip(columns, categories, icons):
    acc = category_accuracy.get(cat, 0)
    stats = category_stats.get(cat, {"correct": 0, "total": 0})
    with col:
        st.metric(
            f"{icon} {cat.replace('_', ' ').title()}",
            f"{acc:.0%}",
            f"{stats['correct']}/{stats['total']} correct"
        )

# Accuracy chart
if PLOTLY_AVAILABLE:
    acc_data = {
        "Category": [cat.replace("_", " ").title() for cat in categories],
        "Accuracy": [category_accuracy.get(cat, 0) * 100 for cat in categories],
    }
    
    fig = px.bar(
        acc_data,
        x="Category",
        y="Accuracy",
        color="Category",
        color_discrete_sequence=["#667eea", "#764ba2", "#f093fb", "#f5576c"],
        title="Routing Accuracy by Task Type (%)"
    )
    fig.update_layout(showlegend=False, yaxis_title="Accuracy (%)")
    fig.update_yaxis(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# Latency Distribution
# ----------------------------

st.divider()
st.subheader("⚡ Latency Distribution")

# Extract latencies from results
latencies = []
experts_used = []
for r in results:
    if "metrics" in r:
        latencies.append(r["metrics"]["latency"]["total"])
        experts_used.append(r["metrics"]["routing"]["selected_expert"])

if latencies and PLOTLY_AVAILABLE:
    fig = px.histogram(
        x=latencies,
        nbins=30,
        title="Response Time Distribution",
        labels={"x": "Latency (seconds)", "y": "Count"},
        color_discrete_sequence=["#667eea"]
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# Expert Usage
# ----------------------------

st.divider()
st.subheader("🤖 Expert Usage Distribution")

# Count expert usage
from collections import Counter
expert_counts = Counter(experts_used)

col1, col2 = st.columns([2, 1])

with col1:
    if PLOTLY_AVAILABLE and expert_counts:
        fig = px.pie(
            names=list(expert_counts.keys()),
            values=list(expert_counts.values()),
            title="Queries by Expert",
            color_discrete_sequence=["#667eea", "#764ba2", "#f093fb", "#f5576c", "#4ecdc4"]
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.write("**Expert Breakdown:**")
    for expert, count in sorted(expert_counts.items(), key=lambda x: -x[1]):
        pct = count / len(experts_used) * 100 if experts_used else 0
        st.write(f"- **{expert}**: {count} ({pct:.1f}%)")


# ----------------------------
# Paper Claims Validation
# ----------------------------

st.divider()
st.subheader("📄 Paper Claims Validation")

claims = [
    {
        "claim": "Task-Expert Alignment",
        "metric": f"{summary['accuracy']:.1%} routing accuracy",
        "status": "✅" if summary['accuracy'] > 0.8 else "⚠️",
    },
    {
        "claim": "Efficient Resource Utilization",
        "metric": f"${summary['avg_cost']:.6f} avg cost/query",
        "status": "✅" if router_total < gpt4o_total else "⚠️",
    },
    {
        "claim": "Real-Time Task Switching",
        "metric": f"{summary['avg_latency']:.2f}s avg latency",
        "status": "✅" if summary['avg_latency'] < 5.0 else "⚠️",
    },
    {
        "claim": "Robustness (Fallback)",
        "metric": f"{summary['fallback_rate']:.1%} fallback rate",
        "status": "✅" if summary['fallback_rate'] < 0.2 else "⚠️",
    },
]

for c in claims:
    col1, col2, col3 = st.columns([3, 3, 1])
    with col1:
        st.write(f"**{c['claim']}**")
    with col2:
        st.write(c["metric"])
    with col3:
        st.write(c["status"])


# ----------------------------
# Footer
# ----------------------------

st.divider()
st.caption(f"Data from: {summary.get('timestamp', 'Unknown')} | Samples: {summary['total_samples']}")
