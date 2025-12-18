from experts.gpt_code import GPTCodeExpert
from experts.gemini_summary import GeminiSummaryExpert

def test_code_expert():
    expert = GPTCodeExpert()
    res = expert.generate("Write a Python function to reverse a string")
    assert isinstance(res.content, str)

def test_gemini_summary():
    expert = GeminiSummaryExpert()
    res = expert.generate("Summarize: Python is a programming language.")
    assert isinstance(res.content, str)
