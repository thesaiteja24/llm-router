from router.router_llm import RouterLLM

router = RouterLLM()

queries = [
    "Write a Python function to reverse a string",
    "Explain how quicksort works",
    "Summarize this paragraph: Artificial intelligence is...",
    "What is the capital of France?"
]

for q in queries:
    decision = router.route(q)
    print("\nQuery:", q)
    print(decision.model_dump())
