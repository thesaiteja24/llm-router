"""
Evaluation Runner for LLM Router.

Runs the full evaluation benchmark and saves detailed results.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from routing_engine import RoutingEngine
from dataset import EVALUATION_DATASET


RESULTS_FILE = Path(__file__).parent / "results.json"


def run_evaluation(max_samples: int = None):
    """
    Run evaluation on the dataset and return detailed metrics.
    
    Args:
        max_samples: Optional limit on number of samples to evaluate
    """
    engine = RoutingEngine()
    
    dataset = EVALUATION_DATASET
    if max_samples:
        dataset = dataset[:max_samples]
    
    total = len(dataset)
    correct = 0
    
    results = []
    category_stats = {
        "code_generation": {"correct": 0, "total": 0},
        "reasoning": {"correct": 0, "total": 0},
        "summarization": {"correct": 0, "total": 0},
        "general": {"correct": 0, "total": 0},
    }
    
    total_cost = 0.0
    total_latency = 0.0
    fallback_count = 0

    print("=" * 60)
    print("LLM Router Evaluation")
    print("=" * 60)
    print(f"Dataset size: {total} samples")
    print()

    for i, item in enumerate(dataset, 1):
        query = item["query"]
        expected = item["task_type"]
        
        # Truncate query for display
        display_query = query[:80] + "..." if len(query) > 80 else query
        print(f"[{i}/{total}] {display_query}")
        
        try:
            response, metrics = engine.run_with_metrics(query, expected_task=expected)
            
            # Update statistics
            is_correct = metrics.is_correct
            if is_correct:
                correct += 1
                category_stats[expected]["correct"] += 1
            
            category_stats[expected]["total"] += 1
            total_cost += metrics.total_cost
            total_latency += metrics.total_latency
            
            if metrics.used_fallback:
                fallback_count += 1
            
            # Store result
            results.append({
                "query": query,
                "expected_task": expected,
                "metrics": metrics.to_dict(),
                "response_preview": response.content[:200] if response.content else "",
            })
            
            status = "✓" if is_correct else "✗"
            print(f"    {status} Predicted: {metrics.predicted_task} | "
                  f"Latency: {metrics.total_latency:.2f}s | "
                  f"Cost: ${metrics.total_cost:.6f}")
            
        except Exception as e:
            print(f"    ✗ Error: {str(e)[:50]}")
            category_stats[expected]["total"] += 1
            results.append({
                "query": query,
                "expected_task": expected,
                "error": str(e),
            })

    # Calculate summary statistics
    results_with_metrics = [r for r in results if "metrics" in r]
    accuracy = correct / total if total > 0 else 0
    avg_latency = total_latency / len(results_with_metrics) if results_with_metrics else 0
    avg_cost = total_cost / len(results_with_metrics) if results_with_metrics else 0
    fallback_rate = fallback_count / total if total > 0 else 0
    
    # Category accuracies
    category_accuracy = {}
    for cat, stats in category_stats.items():
        if stats["total"] > 0:
            category_accuracy[cat] = stats["correct"] / stats["total"]
        else:
            category_accuracy[cat] = 0.0

    # Summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": total,
        "accuracy": accuracy,
        "avg_latency": avg_latency,
        "avg_cost": avg_cost,
        "total_cost": total_cost,
        "fallback_rate": fallback_rate,
        "category_accuracy": category_accuracy,
        "category_stats": category_stats,
    }

    # Save results
    output = {
        "summary": summary,
        "results": results,
    }
    
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Overall Accuracy:  {accuracy:.1%}")
    print(f"Average Latency:   {avg_latency:.2f}s")
    print(f"Average Cost:      ${avg_cost:.6f}")
    print(f"Total Cost:        ${total_cost:.4f}")
    print(f"Fallback Rate:     {fallback_rate:.1%}")
    print()
    print("Accuracy by Category:")
    for cat, acc in category_accuracy.items():
        count = category_stats[cat]["total"]
        print(f"  {cat:20s}: {acc:.1%} ({category_stats[cat]['correct']}/{count})")
    print()
    print(f"Results saved to: {RESULTS_FILE}")

    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM Router evaluation")
    parser.add_argument(
        "--max-samples", 
        type=int, 
        default=None,
        help="Maximum number of samples to evaluate (default: all)"
    )
    args = parser.parse_args()
    
    run_evaluation(max_samples=args.max_samples)
