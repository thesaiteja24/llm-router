"""
Evaluation Dataset Loader.

Loads the 500-sample dataset from JSON file, or falls back to 
hardcoded samples if the file doesn't exist.
"""

import json
from pathlib import Path
from typing import List, Dict


DATASET_FILE = Path(__file__).parent / "dataset_500.json"


# Fallback dataset (original 5 samples)
FALLBACK_DATASET = [
    {
        "query": "Write a Python function to reverse a string",
        "task_type": "code_generation",
    },
    {
        "query": "Explain how binary search works",
        "task_type": "reasoning",
    },
    {
        "query": "Summarize this text: Artificial intelligence enables machines to learn.",
        "task_type": "summarization",
    },
    {
        "query": "What is the capital of France?",
        "task_type": "general",
    },
    {
        "query": "Debug this Python code that raises an IndexError",
        "task_type": "code_generation",
    },
]


def load_evaluation_dataset() -> List[Dict]:
    """
    Load the evaluation dataset from JSON file.
    Falls back to hardcoded samples if file doesn't exist.
    """
    if DATASET_FILE.exists():
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return FALLBACK_DATASET


# For backwards compatibility
EVALUATION_DATASET = load_evaluation_dataset()
