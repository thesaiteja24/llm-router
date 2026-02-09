from openai import OpenAI
from experts.base import LLMExpert, LLMResponse
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


class GPTCodeExpert(LLMExpert):
    """
    GPT-based expert optimized for code generation and debugging.
    """

    def __init__(self, model: str = "gpt-4o"):
        super().__init__(model)

    def generate(
        self,
        prompt: str,
        **kwargs,
    ) -> LLMResponse:

        system_prompt = (
            "You are an expert software engineer. "
            "Produce correct, idiomatic, and runnable code. "
            "Follow constraints exactly. "
            "Explain assumptions briefly after the code."
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=kwargs.get("temperature", 0.3),
            max_tokens=kwargs.get("max_tokens", 1200),
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            presence_penalty=kwargs.get("presence_penalty", 0.0),
        )

        message = response.choices[0].message.content

        return LLMResponse(
            content=message,
            provider="openai",
            model=self.model,
            usage=response.usage.model_dump() if response.usage else None,
        )
