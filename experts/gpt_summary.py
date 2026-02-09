from openai import OpenAI
from experts.base import LLMExpert, LLMResponse
from config import OPENAI_API_KEY


client = OpenAI(api_key=OPENAI_API_KEY)


class GPTSummaryExpert(LLMExpert):
    """
    GPT-based expert optimized for summarization and extraction.
    Uses GPT-4o-mini for cost efficiency on summarization tasks.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__(model)

    def generate(
        self,
        prompt: str,
        **kwargs,
    ) -> LLMResponse:
        system_prompt = (
            "You are a concise summarization expert. "
            "Extract the key points and present them clearly. "
            "Be brief but comprehensive. Avoid unnecessary details."
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 400),
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            presence_penalty=kwargs.get("presence_penalty", 0.0),
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            provider="openai",
            model=self.model,
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
        )
