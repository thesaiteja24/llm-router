import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from routing_engine import RoutingEngine
from dataset import EVALUATION_DATASET


def run():
    engine = RoutingEngine()
    
    correct = 0
    total = len(EVALUATION_DATASET)

    latencies = []
    tokens = []

    for item in EVALUATION_DATASET:
        query = item["query"]
        expected = item["task_type"]

        response, metrics = engine.run_with_metrics(query)

        predicted = engine.router.route(query).task_type

        if predicted == expected:
            correct += 1

        latencies.append(metrics.total_latency)

        if metrics.input_tokens and metrics.output_tokens:
            tokens.append(metrics.input_tokens + metrics.output_tokens)

        print("\nQuery:", query)
        print("Expected:", expected)
        print("Predicted:", predicted)
        print("Latency:", metrics.total_latency)
        print("Model:", metrics.model)

    print("\n=== SUMMARY ===")
    print("Accuracy:", correct / total)
    print("Avg Latency:", sum(latencies) / len(latencies))
    if tokens:
        print("Avg Tokens:", sum(tokens) / len(tokens))


if __name__ == "__main__":
    run()
