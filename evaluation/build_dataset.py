"""
Dataset Builder for LLM Router Evaluation.

Fetches 500+ samples from established HuggingFace benchmarks:
- Code Generation: HumanEval, MBPP
- Reasoning: GSM8K (math reasoning)
- Summarization: XSum
- General: TriviaQA

Output: evaluation/dataset_500.json

If 'datasets' package is unavailable, generates a synthetic dataset.
"""

import json
import random
from pathlib import Path
from typing import List, Dict

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


SAMPLES_PER_CATEGORY = 125
OUTPUT_FILE = Path(__file__).parent / "dataset_500.json"


# ----------------------------
# Synthetic Dataset (Fallback)
# ----------------------------

SYNTHETIC_CODE_QUERIES = [
    "Write a Python function to reverse a string",
    "Implement a binary search algorithm in Python",
    "Create a function to check if a number is prime",
    "Write a function to find the factorial of a number",
    "Implement a stack data structure using a list",
    "Create a function to merge two sorted arrays",
    "Write a function to find the longest common subsequence",
    "Implement a queue using two stacks",
    "Create a function to validate a binary search tree",
    "Write a function to find all permutations of a string",
    "Implement depth-first search for a graph",
    "Create a function to find the maximum subarray sum",
    "Write a function to detect a cycle in a linked list",
    "Implement breadth-first search for a graph",
    "Create a function to find the kth largest element",
]

SYNTHETIC_REASONING_QUERIES = [
    "A train travels 120 miles in 2 hours. What is its average speed?",
    "If a shirt costs $25 and is on sale for 20% off, what is the sale price?",
    "John has 3 apples and Mary gives him 5 more. How many apples does John have?",
    "A rectangle has a length of 8 cm and width of 5 cm. What is its area?",
    "If 5 workers can build a wall in 10 days, how long will 10 workers take?",
    "What is 15% of 200?",
    "A car travels 60 mph for 3 hours. How far does it travel?",
    "If a book costs $12 and you have $50, how many books can you buy?",
    "The sum of two numbers is 15 and their difference is 3. What are the numbers?",
    "A pizza is cut into 8 slices. If 3 people share it equally, how many slices each?",
    "Explain the concept of recursion and give an example",
    "Why does water expand when it freezes?",
    "How does photosynthesis work in plants?",
    "Explain the difference between RAM and ROM",
    "What causes the seasons on Earth?",
]

SYNTHETIC_SUMMARIZATION_QUERIES = [
    "Summarize this article: AI has transformed industries worldwide. From healthcare to finance, machine learning algorithms now power everything from diagnosis to fraud detection. Companies are investing billions in AI research.",
    "Summarize: Climate change is affecting global weather patterns. Scientists report rising sea levels and more frequent extreme weather events. International cooperation is needed to address these challenges.",
    "Summarize the following: The stock market saw significant volatility this week as investors reacted to new economic data. Tech stocks led the decline, while energy shares showed resilience.",
    "Summarize: Electric vehicles are becoming increasingly popular as battery technology improves and charging infrastructure expands. Many countries have set targets to phase out gasoline cars.",
    "Summarize this text: Remote work has become the new normal for many companies. Employees appreciate the flexibility, while employers are reconsidering office space needs.",
    "Summarize: Space exploration has entered a new era with private companies launching rockets. Missions to Mars are being planned by multiple organizations.",
    "Summarize: Renewable energy sources like solar and wind are growing rapidly. Costs have dropped significantly, making them competitive with fossil fuels.",
    "Summarize the article: Cybersecurity threats are increasing as more businesses move online. Companies are investing in advanced security measures to protect their data.",
    "Summarize: The global supply chain is facing disruptions due to various factors. Manufacturers are seeking ways to make their supply chains more resilient.",
    "Summarize: Quantum computing promises to revolutionize computing power. Researchers are making progress but practical applications are still years away.",
]

SYNTHETIC_GENERAL_QUERIES = [
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the largest planet in our solar system?",
    "When did World War II end?",
    "What is the chemical symbol for gold?",
    "Who painted the Mona Lisa?",
    "What is the tallest mountain in the world?",
    "What year did the Berlin Wall fall?",
    "What is the speed of light?",
    "Who discovered penicillin?",
    "What is the largest ocean on Earth?",
    "Who was the first person to walk on the moon?",
    "What is the capital of Japan?",
    "What is the hardest natural substance on Earth?",
    "Who wrote 'To Kill a Mockingbird'?",
]

# ----------------------------
# Ambiguous Edge-Case Queries
# (Blur category boundaries for realistic evaluation)
# ----------------------------

# ----------------------------
# Ambiguous Query Templates
# (Template Expansion for ~60 unique queries per category)
# ----------------------------

AMBIGUOUS_TEMPLATES = {
    "code_generation": {
        "templates": [
            "How do I fix a {ERROR} in {LANG}?",
            "Show me how to {CSS_TASK} in CSS",
            "Write a regex to match {ENTITY}",
            "Construct a {LANG} class for {OBJECT}",
            "Script to {SYS_TASK} in {SHELL}?",
            "Write a function to {FUNC_TASK}",
            "Implement a {DS} in {LANG}",
        ],
        "variables": {
            "ERROR": ["Segmentation Fault", "NullPointerException", "IndexError", "TypeMismatch", "RecursionError"],
            "LANG": ["Python", "Java", "C++", "JavaScript", "Rust", "Go"],
            "CSS_TASK": ["center a div", "make a grid", "hide an element", "animate a button", "create a sticky header"],
            "ENTITY": ["emails", "IP addresses", "dates", "credit card numbers", "hex colors"],
            "OBJECT": ["User", "Product", "Order", "Vehicle", "Employee"],
            "SYS_TASK": ["backup files", "check disk space", "kill a process", "parse logs", "monitor memory"],
            "SHELL": ["Bash", "PowerShell", "Zsh", "Python"],
            "FUNC_TASK": ["validate emails", "parse JSON", "encrypt passwords", "resize images"],
            "DS": ["Linked List", "Binary Tree", "Hash Map", "Graph"],
        }
    },
    "reasoning": {
        "templates": [
            "What is the time complexity of {ALGO}?",
            "Analyze the {ASPECT} of this text",
            "Plan a {N}-day trip to {CITY}",
            "Compare {CONCEPT_A} and {CONCEPT_B} in computing",
            "Why is {CONST} important in math?",
            "Explain the logic behind {PHENOMENON}",
            "Solve for x: {EQUATION}",
            "Derive the formula for {FORMULA}",
        ],
        "variables": {
            "ALGO": ["QuickSort", "MergeSort", "Binary Search", "Dijkstra's Algorithm", "A* Search", "K-Means", "PageRank", "RSA Encryption"],
            "ASPECT": ["sentiment", "logic", "bias", "structure", "rhetoric", "tone", "style", "grammar", "arguments"],
            "N": ["3", "5", "7", "10", "14", "21", "30"],
            "CITY": ["Paris", "Tokyo", "Rome", "New York", "London", "Berlin", "Sydney", "Dubai", "Singapore", "Mumbai", "Toronto"],
            "CONCEPT_A": ["TCP", "Stack", "SQL", "Process", "Compiler", "Git", "HTTP", "Monolith"],
            "CONCEPT_B": ["UDP", "Queue", "NoSQL", "Thread", "Interpreter", "SVN", "HTTPS", "Microservices"],
            "CONST": ["Pi", "Euler's number", "the Golden Ratio", "Avogadro's Constant", "Planck's Constant", "Speed of Light"],
            "PHENOMENON": ["gravity", "inflation", "supply and demand", "refraction", "quantum entanglement", "photosynthesis", "osmosis"],
            "EQUATION": ["2x + 5 = 15", "x^2 - 4 = 0", "log(x) = 2", "e^x = 5", "sin(x) = 0.5", "3y - 2 = 10"],
            "FORMULA": ["kinetic energy", "compound interest", "entropy", "relativity", "force (F=ma)", "momentum"],
        }
    },
    "general": {
        "templates": [
            "Who created the {LANG} programming language?",
            "What is the value of {CONST} to 2 decimal places?",
            "List the ingredients for {RECIPE}",
            "What year was {EVENT}?",
            "Capital city of {COUNTRY}",
            "Who wrote {BOOK}?",
            "What is the chemical symbol for {ELEMENT}?",
            "Distance from Earth to {CELESTIAL}?",
        ],
        "variables": {
            "LANG": ["Python", "Java", "C++", "Rust", "Go", "Perl", "Ruby", "Swift", "PHP", "Kotlin", "Scala", "Haskell"],
            "CONST": ["Pi", "Tao", "e", "Avogadro's constant", "Planck's constant"],
            "RECIPE": ["pancakes", "risotto", "brownies", "pizza dough", "hummus", "guacamole", "chocolate chip cookies", "lasagna"],
            "EVENT": ["the iPhone launch", "World War 1", "the moon landing", "the fall of Berlin Wall", "the invention of the internet", "the first flight"],
            "COUNTRY": ["France", "Japan", "Brazil", "Canada", "Egypt", "Australia", "Germany", "India", "China", "Russia", "Argentina"],
            "BOOK": ["1984", "The Great Gatsby", "Hamlet", "Pride and Prejudice", "Dune", "The Hobbit", "To Kill a Mockingbird", "Moby Dick"],
            "ELEMENT": ["Gold", "Silver", "Iron", "Oxygen", "Carbon", "Helium", "Uranium", "Silicon", "Neon"],
            "CELESTIAL": ["the Moon", "Mars", "the Sun", "Jupiter", "Saturn", "Venus", "Alpha Centauri"],
        }
    },
    "summarization": {
        "templates": [
            "Summarize this code function: {CODE_FUNC}",
            "Give me the gist of this article about {TOPIC}",
            "TL;DR of the history of {SUBJECT}",
            "Condense this email about {TOPIC}",
            "What is the main takeaway from this {DOC_TYPE}?",
            "Compress this paragraph about {TOPIC} into 1 sentence",
        ],
        "variables": {
            "CODE_FUNC": ["def calculate_pi():...", "class NeuralNetwork:...", "void sort(int[] arr)...", "function parseJSON(str)...", "SELECT * FROM users...", "const router = express()..."],
            "TOPIC": ["AI advancement", "climate policy", "market trends", "remote work", "cybersecurity", "renewable energy", "space exploration", "meditation benefits", "healthy eating", "blockchain"],
            "SUBJECT": ["computing", "the internet", "cryptography", "aviation", "genetics", "robotics", "quantum physics", "neuroscience", "philosophy"],
            "DOC_TYPE": ["memo", "proposal", "contract", "report", "whitepaper", "newsletter", "press release", "technical guide"],
        }
    }
}


def expand_templates(category: str, count: int) -> List[Dict]:
    """Generate unique queries from templates."""
    data = AMBIGUOUS_TEMPLATES[category]
    templates = data["templates"]
    variables = data["variables"]
    
    samples = []
    
    # Generate as many unique combinations as needed
    attempts = 0
    max_attempts = count * 10
    
    while len(samples) < count and attempts < max_attempts:
        tmpl = random.choice(templates)
        query = tmpl
        for var_name, values in variables.items():
            if f"{{{var_name}}}" in query:
                query = query.replace(f"{{{var_name}}}", random.choice(values))
        
        # Check uniqueness to avoid duplicates
        if not any(s["query"] == query for s in samples):
            samples.append({
                "query": query,
                "task_type": category,  # Use the category as the ground truth
                "source": f"ambiguous_{category}",
                "expected_answer": None
            })
        attempts += 1
        
    return samples


def generate_synthetic_samples(
    base_queries: List[str], 
    task_type: str, 
    count: int = 125
) -> List[Dict]:
    """Generate synthetic samples by expanding base queries."""
    samples = []
    
    # Repeat and slightly vary to reach count
    while len(samples) < count:
        for query in base_queries:
            if len(samples) >= count:
                break
            samples.append({
                "query": query,
                "task_type": task_type,
                "source": "synthetic",
                "expected_answer": None,
            })
    
    random.shuffle(samples)
    return samples[:count]


def build_synthetic_dataset() -> List[Dict]:
    """Build dataset from synthetic samples (no network required)."""
    print("=" * 50)
    print("Building Synthetic Evaluation Dataset")
    print("(750 samples: 500 clean + 250 ambiguous)")
    print("=" * 50)
    
    random.seed(42)
    
    # 1. Generate Clean Samples (500 total, 125 per category)
    clean_samples_per_cat = 125
    
    code_samples = generate_synthetic_samples(
        SYNTHETIC_CODE_QUERIES, "code_generation", clean_samples_per_cat
    )
    reasoning_samples = generate_synthetic_samples(
        SYNTHETIC_REASONING_QUERIES, "reasoning", clean_samples_per_cat
    )
    summarization_samples = generate_synthetic_samples(
        SYNTHETIC_SUMMARIZATION_QUERIES, "summarization", clean_samples_per_cat
    )
    general_samples = generate_synthetic_samples(
        SYNTHETIC_GENERAL_QUERIES, "general", clean_samples_per_cat
    )
    
    # 2. Generate Ambiguous Samples (250 total, ~62-63 per category)
    ambiguous_count = 62 
    # To hit exactly 250, we'll do 63, 63, 62, 62
    
    amb_code = expand_templates("code_generation", 63)
    amb_reasoning = expand_templates("reasoning", 63)
    amb_summary = expand_templates("summarization", 62)
    amb_general = expand_templates("general", 62)
    
    all_samples = (
        code_samples + 
        reasoning_samples + 
        summarization_samples + 
        general_samples +
        amb_code +
        amb_reasoning +
        amb_summary +
        amb_general
    )
    
    random.shuffle(all_samples)
    
    # Count by category
    category_counts = {}
    ambiguous_total = len(amb_code) + len(amb_reasoning) + len(amb_summary) + len(amb_general)
    
    for s in all_samples:
        cat = s["task_type"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"\nGenerated {len(all_samples)} total samples")
    print(f"  Code Generation: {category_counts.get('code_generation', 0)} (includes ambiguous)")
    print(f"  Reasoning:       {category_counts.get('reasoning', 0)} (includes ambiguous)")
    print(f"  Summarization:   {category_counts.get('summarization', 0)} (includes ambiguous)")
    print(f"  General:         {category_counts.get('general', 0)} (includes ambiguous)")
    print(f"  (Ambiguous Subset: {ambiguous_total} samples)")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    
    print(f"\nDataset saved to: {OUTPUT_FILE}")
    
    return all_samples


# ----------------------------
# HuggingFace Dataset Fetchers
# ----------------------------

def fetch_code_generation_samples(count: int = 125) -> List[Dict]:
    """Fetch code generation samples from HumanEval and MBPP."""
    samples = []
    
    print("Loading HumanEval dataset...")
    humaneval = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
    
    for item in humaneval:
        prompt = item["prompt"]
        samples.append({
            "query": f"Write a Python function to solve:\n{prompt}",
            "task_type": "code_generation",
            "source": "humaneval",
            "expected_answer": item.get("canonical_solution", None),
        })
    
    print("Loading MBPP dataset...")
    mbpp = load_dataset("mbpp", split="test", trust_remote_code=True)
    
    for item in mbpp:
        samples.append({
            "query": item["text"],
            "task_type": "code_generation",
            "source": "mbpp",
            "expected_answer": item.get("code", None),
        })
    
    random.shuffle(samples)
    return samples[:count]


def fetch_reasoning_samples(count: int = 125) -> List[Dict]:
    """Fetch reasoning samples from GSM8K."""
    samples = []
    
    print("Loading GSM8K dataset...")
    gsm8k = load_dataset("gsm8k", "main", split="test", trust_remote_code=True)
    
    for item in gsm8k:
        samples.append({
            "query": item["question"],
            "task_type": "reasoning",
            "source": "gsm8k",
            "expected_answer": item.get("answer", None),
        })
    
    random.shuffle(samples)
    return samples[:count]


def fetch_summarization_samples(count: int = 125) -> List[Dict]:
    """Fetch summarization samples from XSum."""
    samples = []
    
    print("Loading XSum dataset...")
    xsum = load_dataset("EdinburghNLP/xsum", split="test", trust_remote_code=True)
    
    for item in xsum:
        document = item["document"]
        if len(document) > 2000:
            document = document[:2000] + "..."
        
        samples.append({
            "query": f"Summarize the following article:\n\n{document}",
            "task_type": "summarization",
            "source": "xsum",
            "expected_answer": item.get("summary", None),
        })
    
    random.shuffle(samples)
    return samples[:count]


def fetch_general_samples(count: int = 125) -> List[Dict]:
    """Fetch general knowledge samples from TriviaQA."""
    samples = []
    
    print("Loading TriviaQA dataset...")
    trivia = load_dataset(
        "trivia_qa", "unfiltered", 
        split="validation", 
        trust_remote_code=True
    )
    
    for item in trivia:
        samples.append({
            "query": item["question"],
            "task_type": "general",
            "source": "triviaqa",
            "expected_answer": item.get("answer", {}).get("value", None),
        })
    
    random.shuffle(samples)
    return samples[:count]


def build_dataset_from_huggingface() -> List[Dict]:
    """Build dataset from HuggingFace (requires network)."""
    print("=" * 50)
    print("Building LLM Router Evaluation Dataset")
    print("(750 samples: 500 clean + 250 ambiguous)")
    print("=" * 50)
    
    random.seed(42)
    
    # 1. Generate Clean Samples (500 total, 125 per category)
    clean_samples_per_cat = 125
    
    code_samples = fetch_code_generation_samples(clean_samples_per_cat)
    reasoning_samples = fetch_reasoning_samples(clean_samples_per_cat)
    summarization_samples = fetch_summarization_samples(clean_samples_per_cat)
    general_samples = fetch_general_samples(clean_samples_per_cat)
    
    # 2. Generate Ambiguous Samples (250 total, ~62-63 per category)
    ambiguous_count = 62 
    # To hit exactly 250, we'll do 63, 63, 62, 62
    
    amb_code = expand_templates("code_generation", 63)
    amb_reasoning = expand_templates("reasoning", 63)
    amb_summary = expand_templates("summarization", 62)
    amb_general = expand_templates("general", 62)
    
    all_samples = (
        code_samples + 
        reasoning_samples + 
        summarization_samples + 
        general_samples +
        amb_code +
        amb_reasoning +
        amb_summary +
        amb_general
    )
    
    random.shuffle(all_samples)
    
    # Count by category
    category_counts = {}
    ambiguous_total = len(amb_code) + len(amb_reasoning) + len(amb_summary) + len(amb_general)
    
    for s in all_samples:
        cat = s["task_type"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)
    print(f"Code Generation: {category_counts.get('code_generation', 0)} (includes ambiguous)")
    print(f"Reasoning:       {category_counts.get('reasoning', 0)} (includes ambiguous)")
    print(f"Summarization:   {category_counts.get('summarization', 0)} (includes ambiguous)")
    print(f"General:         {category_counts.get('general', 0)} (includes ambiguous)")
    print(f"Total:           {len(all_samples)} samples")
    print(f"(Ambiguous Subset: {ambiguous_total} samples)")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    
    print(f"\nDataset saved to: {OUTPUT_FILE}")
    
    return all_samples


def build_dataset() -> List[Dict]:
    """
    Build evaluation dataset.
    Uses HuggingFace if available, otherwise generates synthetic data.
    """
    if DATASETS_AVAILABLE:
        return build_dataset_from_huggingface()
    else:
        print("\n⚠️  'datasets' package not installed.")
        print("    Generating synthetic dataset instead.")
        print("    For full dataset, install with: pip install datasets\n")
        return build_synthetic_dataset()


if __name__ == "__main__":
    build_dataset()
