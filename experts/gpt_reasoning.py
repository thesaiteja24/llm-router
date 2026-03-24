from openai import OpenAI
from experts.base import LLMExpert, LLMResponse
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


class GPTReasoningExpert(LLMExpert):
    """
    GPT-based expert optimized for reasoning and explanations.
    """

    def __init__(self, model: str = "o3-mini"):
        super().__init__(model)

    def generate(
        self,
        prompt: str,
        **kwargs,
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
            # temperature is not supported by o3-mini
            # temperature=kwargs.get("temperature", 0.3),
            max_completion_tokens=kwargs.get("max_tokens", 1000),
            # frequency_penalty and presence_penalty are not supported by o3-mini
            # frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            # presence_penalty=kwargs.get("presence_penalty", 0.0),
        )

        message = response.choices[0].message.content

        return LLMResponse(
            content=message,
            provider="openai",
            model=self.model,
            usage=response.usage.model_dump() if response.usage else None,
        )
