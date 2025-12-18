from openai import OpenAI
from experts.base import LLMExpert, LLMResponse
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


class GPTReasoningExpert(LLMExpert):
    """
    GPT-based expert optimized for reasoning and explanations.
    """

    def __init__(self, model: str = "gpt-4o"):
        super().__init__(model)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> LLMResponse:

        system_prompt = (
            "You are a precise reasoning assistant. "
            "Explain step by step when appropriate. "
            "Be concise and logically structured."
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        message = response.choices[0].message.content

        return {
            "content": message,
            "provider": "openai",
            "model": self.model,
            "usage": response.usage.model_dump() if response.usage else None,
        }
