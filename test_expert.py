from experts.gpt_code import GPTCodeExpert
from experts.gemini_summary import GeminiSummaryExpert

expert = GPTCodeExpert()
res = expert.generate("Write a Python function to reverse a string")
print(res["content"])

gemini_expert = GeminiSummaryExpert()
gemini_res = gemini_expert.generate("Write a Python function to reverse a string")
print(gemini_res["content"])
