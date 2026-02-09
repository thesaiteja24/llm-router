#!/usr/bin/env python3
"""
Paper Charts Generator for LLM Router Evaluation Results.

Generates publication-quality charts from evaluation/results.json
for inclusion in IEEE papers.

Usage:
    python generate_paper_charts.py

Output:
    paper_charts/ directory with 8 high-resolution PNG files
"""

import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Configuration
# ============================================================

RESULTS_FILE = Path(__file__).parent / "evaluation" / "results.json"
OUTPUT_DIR = Path(__file__).parent / "paper_charts"

# IEEE paper-ready styling
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette for consistency
COLORS = {
    'code_generation': '#2ecc71',    # Green
    'reasoning': '#3498db',           # Blue
    'summarization': '#9b59b6',       # Purple
    'general': '#e74c3c',             # Red
    'router': '#f39c12',              # Orange
    'expert': '#1abc9c',              # Teal
    'gpt_code': '#2ecc71',
    'gpt_reasoning': '#3498db',
    'gpt_summary': '#9b59b6',
    'fallback': '#95a5a6',            # Gray
    'o3-mini': '#f39c12',
    'o4-mini': '#3498db',
    'gpt-4.1-mini': '#95a5a6',
    'gpt-4o': '#2ecc71',
    'gpt-4o-mini': '#9b59b6',
}

CATEGORY_LABELS = {
    'code_generation': 'Code Generation',
    'reasoning': 'Reasoning',
    'summarization': 'Summarization',
    'general': 'General Knowledge',
}


def load_results():
    """Load evaluation results from JSON file."""
    with open(RESULTS_FILE, 'r') as f:
        return json.load(f)


def save_chart(fig, filename):
    """Save chart to output directory."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    filepath = OUTPUT_DIR / filename
    fig.savefig(filepath, format='png', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ Saved: {filepath}")


# ============================================================
# Chart 1: Routing Accuracy by Task Category
# ============================================================

def chart_accuracy_by_category(data):
    """Bar chart showing routing accuracy for all 4 task categories."""
    summary = data['summary']
    
    categories = ['code_generation', 'reasoning', 'summarization', 'general']
    labels = [CATEGORY_LABELS[c] for c in categories]
    accuracies = [summary['category_accuracy'][c] * 100 for c in categories]
    colors = [COLORS[c] for c in categories]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(labels, accuracies, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Task Category')
    ax.set_ylabel('Routing Accuracy (%)')
    ax.set_title('LLM Router Accuracy by Task Category')
    ax.set_ylim(0, 110)
    ax.axhline(y=summary['accuracy'] * 100, color='red', linestyle='--', 
               linewidth=2, label=f"Overall: {summary['accuracy']*100:.1f}%")
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    save_chart(fig, '01_accuracy_by_category.png')


# ============================================================
# Chart 2: Cost Comparison - Router vs Single Model Baselines
# ============================================================

def chart_cost_comparison(data):
    """Grouped bar chart comparing routing costs vs baselines."""
    results = data['results']
    summary = data['summary']
    
    # Calculate costs
    total_queries = summary['total_samples']
    router_cost = summary['total_cost']
    
    # Estimate baseline costs (if all queries used single model)
    # GPT-4o: $2.50/1M input, $10.00/1M output
    # GPT-4o-mini: $0.15/1M input, $0.60/1M output
    gpt4o_cost_per_query = 0.0035
    gpt4o_mini_cost_per_query = 0.00035
    
    gpt4o_baseline = total_queries * gpt4o_cost_per_query
    gpt4o_mini_baseline = total_queries * gpt4o_mini_cost_per_query
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    approaches = ['LLM Router\n(Our Approach)', 'GPT-4o Only\n(High Quality)', 'GPT-4o-mini Only\n(Low Cost)']
    costs = [router_cost, gpt4o_baseline, gpt4o_mini_baseline]
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    bars = ax.bar(approaches, costs, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax.annotate(f'${cost:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Approach')
    ax.set_ylabel('Total Cost (USD)')
    ax.set_title(f'Cost Comparison for {total_queries} Queries')
    ax.grid(axis='y', alpha=0.3)
    
    # Add savings annotation
    savings = ((gpt4o_baseline - router_cost) / gpt4o_baseline) * 100
    ax.annotate(f'{savings:.1f}% savings\nvs GPT-4o only',
                xy=(0.15, 0.85), xycoords='axes fraction',
                fontsize=11, ha='left',
                bbox=dict(boxstyle='round', facecolor='#d5f5e3', alpha=0.8))
    
    save_chart(fig, '02_cost_comparison.png')


# ============================================================
# Chart 3: Latency Distribution by Expert
# ============================================================

def chart_latency_distribution(data):
    """Box plot showing latency distribution for each expert."""
    results = data['results']
    
    # Collect latencies by expert
    expert_latencies = defaultdict(list)
    for item in results:
        if 'metrics' in item and item['metrics']:
            metrics = item['metrics']
            expert = metrics['routing']['selected_expert']
            total_latency = metrics['latency']['total']
            expert_latencies[expert].append(total_latency)
    
    # Prepare data for plotting
    experts = ['gpt_code', 'gpt_reasoning', 'gpt_summary', 'fallback']
    expert_labels = ['GPT-4o\n(Code)', 'o4-mini\n(Reasoning)', 'GPT-4o-mini\n(Summary)', 'gpt-4.1-mini\n(Fallback)']
    latency_data = [expert_latencies.get(e, [0]) for e in experts]
    colors = [COLORS.get(e, '#95a5a6') for e in experts]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bp = ax.boxplot(latency_data, labels=expert_labels, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Expert Model')
    ax.set_ylabel('Total Latency (seconds)')
    ax.set_title('Response Time Distribution by Expert')
    ax.grid(axis='y', alpha=0.3)
    
    # Add median annotations
    medians = [np.median(d) if d else 0 for d in latency_data]
    for i, median in enumerate(medians):
        ax.annotate(f'{median:.2f}s',
                    xy=(i + 1, median),
                    xytext=(25, 0),
                    textcoords="offset points",
                    ha='left', va='center',
                    fontsize=9, fontweight='bold')
    
    save_chart(fig, '03_latency_distribution.png')


# ============================================================
# Chart 4: Cost per Query by Task Type
# ============================================================

def chart_cost_per_query(data):
    """Bar chart showing average cost per query for each task type."""
    results = data['results']
    
    # Collect costs by task type
    task_costs = defaultdict(list)
    for item in results:
        if 'metrics' in item and item['metrics']:
            task = item['expected_task']
            cost = item['metrics']['cost']['total']
            task_costs[task].append(cost)
    
    categories = ['code_generation', 'reasoning', 'summarization', 'general']
    labels = [CATEGORY_LABELS[c] for c in categories]
    avg_costs = [np.mean(task_costs.get(c, [0])) * 1000 for c in categories]  # Convert to millicents
    colors = [COLORS[c] for c in categories]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(labels, avg_costs, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bar, cost in zip(bars, avg_costs):
        height = bar.get_height()
        ax.annotate(f'${cost/1000:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Task Category')
    ax.set_ylabel('Average Cost per Query ($ × 10⁻³)')
    ax.set_title('Cost Efficiency by Task Type')
    ax.grid(axis='y', alpha=0.3)
    
    # Highlight that summarization uses cheaper model
    ax.annotate('Uses GPT-4o-mini\n(cost-efficient)',
                xy=(2, avg_costs[2]),
                xytext=(2.5, avg_costs[2] + 1),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9, ha='left')
    
    save_chart(fig, '04_cost_per_query.png')


# ============================================================
# Chart 5: Router Overhead Analysis
# ============================================================

def chart_router_overhead(data):
    """Stacked bar showing router vs expert latency breakdown."""
    results = data['results']
    
    # Collect latencies by task type
    task_router = defaultdict(list)
    task_expert = defaultdict(list)
    
    for item in results:
        if 'metrics' in item and item['metrics']:
            task = item['expected_task']
            task_router[task].append(item['metrics']['latency']['router'])
            task_expert[task].append(item['metrics']['latency']['expert'])
    
    categories = ['code_generation', 'reasoning', 'summarization', 'general']
    labels = [CATEGORY_LABELS[c] for c in categories]
    router_times = [np.mean(task_router.get(c, [0])) for c in categories]
    expert_times = [np.mean(task_expert.get(c, [0])) for c in categories]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(labels))
    width = 0.6
    
    bars1 = ax.bar(x, router_times, width, label='Router (Classification)', 
                   color=COLORS['router'], edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x, expert_times, width, bottom=router_times, label='Expert (Generation)', 
                   color=COLORS['expert'], edgecolor='black', linewidth=1.2)
    
    # Add total time labels
    total_times = [r + e for r, e in zip(router_times, expert_times)]
    for i, (total, r) in enumerate(zip(total_times, router_times)):
        ax.annotate(f'{total:.2f}s',
                    xy=(i, total),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        # Router overhead percentage
        overhead = (r / total) * 100 if total > 0 else 0
        ax.annotate(f'({overhead:.0f}% overhead)',
                    xy=(i, r / 2),
                    ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')
    
    ax.set_xlabel('Task Category')
    ax.set_ylabel('Average Latency (seconds)')
    ax.set_title('Latency Breakdown: Router Classification vs Expert Generation')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    save_chart(fig, '05_router_overhead.png')


# ============================================================
# Chart 6: Confusion Matrix Heatmap
# ============================================================

def chart_confusion_matrix(data):
    """Heatmap showing predicted vs expected task types."""
    results = data['results']
    
    categories = ['code_generation', 'reasoning', 'summarization', 'general']
    labels = ['Code Gen', 'Reasoning', 'Summary', 'General']
    
    # Build confusion matrix
    matrix = np.zeros((4, 4), dtype=int)
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    
    for item in results:
        if 'metrics' in item and item['metrics']:
            expected = item['expected_task']
            predicted = item['metrics']['routing']['predicted_task']
            if expected in cat_to_idx and predicted in cat_to_idx:
                matrix[cat_to_idx[expected]][cat_to_idx[predicted]] += 1
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.imshow(matrix, cmap='Blues')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Number of Queries', rotation=-90, va="bottom")
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix[i, j]
            color = "white" if value > matrix.max() / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center", 
                    color=color, fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Predicted Task Type')
    ax.set_ylabel('Expected Task Type')
    ax.set_title('Routing Confusion Matrix')
    
    save_chart(fig, '06_confusion_matrix.png')


# ============================================================
# Chart 7: Expert Usage Distribution
# ============================================================

def chart_expert_usage(data):
    """Pie chart showing distribution of queries across experts."""
    results = data['results']
    
    # Count queries per expert
    expert_counts = defaultdict(int)
    for item in results:
        if 'metrics' in item and item['metrics']:
            expert = item['metrics']['routing']['selected_expert']
            expert_counts[expert] += 1
    
    experts = ['gpt_code', 'gpt_reasoning', 'gpt_summary', 'fallback']
    labels = ['GPT-4o (Code)', 'o4-mini (Reasoning)', 'GPT-4o-mini (Summary)', 'gpt-4.1-mini (Fallback)']
    sizes = [expert_counts.get(e, 0) for e in experts]
    colors = [COLORS.get(e, '#95a5a6') for e in experts]
    
    # Remove empty slices
    non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
    if non_zero:
        labels, sizes, colors = zip(*non_zero)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(sizes))})',
        startangle=90, explode=[0.02] * len(sizes),
        textprops={'fontsize': 10},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    # Bold percentage text
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    ax.set_title('Expert Usage Distribution')
    
    save_chart(fig, '07_expert_usage.png')


# ============================================================
# Chart 8: Confidence Distribution
# ============================================================

def chart_confidence_distribution(data):
    """Histogram showing router confidence scores."""
    results = data['results']
    
    # Collect confidence scores
    confidences = []
    for item in results:
        if 'metrics' in item and item['metrics']:
            conf = item['metrics']['routing']['confidence']
            confidences.append(conf)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n, bins, patches = ax.hist(confidences, bins=10, range=(0, 1), 
                                color='#3498db', edgecolor='black', 
                                linewidth=1.2, alpha=0.7)
    
    # Color bars by confidence level
    for i, patch in enumerate(patches):
        if bins[i] >= 0.8:
            patch.set_facecolor('#2ecc71')  # High confidence - green
        elif bins[i] >= 0.5:
            patch.set_facecolor('#f39c12')  # Medium confidence - orange
        else:
            patch.set_facecolor('#e74c3c')  # Low confidence - red
    
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Number of Queries')
    ax.set_title('Router Confidence Score Distribution')
    ax.grid(axis='y', alpha=0.3)
    
    # Add mean line
    mean_conf = np.mean(confidences)
    ax.axvline(x=mean_conf, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_conf:.2f}')
    ax.legend(loc='upper left')
    
    # Add statistics box
    stats_text = f'Mean: {mean_conf:.3f}\nStd: {np.std(confidences):.3f}\nMin: {min(confidences):.2f}\nMax: {max(confidences):.2f}'
    ax.annotate(stats_text,
                xy=(0.02, 0.72), xycoords='axes fraction',
                fontsize=10, ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    save_chart(fig, '08_confidence_distribution.png')


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Generating Paper Charts from Evaluation Results")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading results from: {RESULTS_FILE}")
    data = load_results()
    print(f"  Loaded {data['summary']['total_samples']} samples")
    
    # Check if assigned_parameters is available in this data
    has_params = any(
        item.get('metrics', {}).get('assigned_parameters') 
        for item in data.get('results', [])
    )
    
    # Generate all charts
    print(f"\nGenerating charts...")
    
    chart_accuracy_by_category(data)
    chart_cost_comparison(data)
    chart_latency_distribution(data)
    chart_cost_per_query(data)
    chart_router_overhead(data)
    chart_confusion_matrix(data)
    chart_expert_usage(data)
    chart_confidence_distribution(data)
    
    # Parameter charts (only if data has assigned_parameters)
    if has_params:
        chart_temperature_by_task(data)
        chart_token_usage_comparison(data)
    else:
        print("  ⚠ Skipping parameter charts (no assigned_parameters in data)")
        print("    Re-run evaluation to capture parameters: python evaluation/run_evaluation.py")
    
    print(f"\n" + "=" * 60)
    print(f"All charts saved to: {OUTPUT_DIR.absolute()}")
    print("=" * 60)


# ============================================================
# Chart 9: Temperature Distribution by Task Type
# ============================================================

def chart_temperature_by_task(data):
    """Box plot showing temperature settings by task type."""
    results = data['results']
    
    # Collect temperatures by task type
    task_temps = defaultdict(list)
    for item in results:
        if 'metrics' in item and item['metrics']:
            params = item['metrics'].get('assigned_parameters')
            if params and 'temperature' in params:
                task = item['expected_task']
                task_temps[task].append(params['temperature'])
    
    categories = ['code_generation', 'reasoning', 'summarization', 'general']
    labels = [CATEGORY_LABELS[c] for c in categories]
    temp_data = [task_temps.get(c, [0]) for c in categories]
    colors = [COLORS[c] for c in categories]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bp = ax.boxplot(temp_data, tick_labels=labels, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Task Category')
    ax.set_ylabel('Temperature')
    ax.set_title('Dynamic Temperature Assignment by Task Type')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    # Add mean annotations
    for i, temps in enumerate(temp_data):
        if temps:
            mean_temp = np.mean(temps)
            ax.annotate(f'μ={mean_temp:.2f}',
                        xy=(i + 1, mean_temp),
                        xytext=(25, 0),
                        textcoords="offset points",
                        ha='left', va='center',
                        fontsize=9, fontweight='bold')
    
    # Add annotation explaining dynamic assignment
    ax.annotate('Lower temp for precision tasks\nHigher temp for creative tasks',
                xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=9, ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    save_chart(fig, '09_temperature_by_task.png')


# ============================================================
# Chart 10: Token Usage Comparison (Our Approach vs Normal Usage)
# ============================================================

def chart_token_usage_comparison(data):
    """
    Grouped bar chart comparing token usage:
    - Our Approach: Router tokens + Expert tokens
    - Normal Usage: Expert tokens only (simulating direct call)
    """
    results = data['results']
    
    # Collect tokens by task type
    task_tokens_our = defaultdict(list)
    task_tokens_normal = defaultdict(list)
    
    for item in results:
        if 'metrics' in item and item['metrics']:
            task = item['expected_task']
            tokens = item['metrics']['tokens']
            
            # Our approach: Router + Expert
            total_our = (tokens['router_input'] + tokens['router_output'] + 
                         tokens['expert_input'] + tokens['expert_output'])
            
            # Normal usage: Expert only (assuming direct prompt)
            total_normal = tokens['expert_input'] + tokens['expert_output']
            
            task_tokens_our[task].append(total_our)
            task_tokens_normal[task].append(total_normal)
    
    categories = ['code_generation', 'reasoning', 'summarization', 'general']
    labels = [CATEGORY_LABELS[c] for c in categories]
    
    avg_our = [np.mean(task_tokens_our.get(c, [0])) for c in categories]
    avg_normal = [np.mean(task_tokens_normal.get(c, [0])) for c in categories]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, avg_our, width, label='Our Approach\n(Router + Expert)', 
                   color='#2ecc71', edgecolor='black', linewidth=1.0)
    bars2 = ax.bar(x + width/2, avg_normal, width, label='Normal Usage\n(Expert Only)', 
                   color='#95a5a6', edgecolor='black', linewidth=1.0, hatch='//')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Task Category')
    ax.set_ylabel('Average Total Tokens')
    ax.set_title('Token Usage Comparison: Router Overhead')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add annotation about minimal overhead
    # Calculate average overhead percentage
    total_our_sum = sum(avg_our)
    total_normal_sum = sum(avg_normal)
    if total_normal_sum > 0:
        overhead_pct = ((total_our_sum - total_normal_sum) / total_normal_sum) * 100
        ax.annotate(f'Only ~{overhead_pct:.1f}% overhead\nfor routing intelligence',
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=10, ha='left', va='top',
                    bbox=dict(boxstyle='round', facecolor='#e8f6f3', alpha=1.0))
    
    save_chart(fig, '10_token_usage_comparison.png')


if __name__ == "__main__":
    main()

