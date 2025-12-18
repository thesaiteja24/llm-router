from google import genai
from experts.base import LLMExpert, LLMResponse
from config import GOOGLE_API_KEY


class GeminiSummaryExpert(LLMExpert):
    """
    Gemini-based expert optimized for summarization and extraction.
    Uses the new google.genai SDK.
    """

    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        super().__init__(model)
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 400,
    ) -> LLMResponse:

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )

        return LLMResponse(
            content=response.text,
            provider="google",
            model=self.model,
            usage=None,  # usage not consistently exposed yet
        )
