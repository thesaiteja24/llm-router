import time
from google import genai
from experts.base import LLMExpert, LLMResponse
from config import GOOGLE_API_KEY


class GeminiSummaryExpert(LLMExpert):
    """
    Gemini-based expert optimized for summarization and extraction.
    Uses the new google.genai SDK.
    Includes retry logic for rate limiting.
    """

    MAX_RETRIES = 3
    BASE_DELAY = 2  # seconds

    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        super().__init__(model)
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

    def generate(
        self,
        prompt: str,
        **kwargs,
    ) -> LLMResponse:
        
        last_error = None
        
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "temperature": kwargs.get("temperature", 0.2),
                        "max_output_tokens": kwargs.get("max_tokens", 400),
                        "presence_penalty": kwargs.get("presence_penalty", 0.0),
                        "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                    },
                )

                return LLMResponse(
                    content=response.text,
                    provider="google",
                    model=self.model,
                    usage=None,  # usage not consistently exposed yet
                )
                
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Check if it's a rate limit error (429)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    delay = self.BASE_DELAY * (2 ** attempt)  # Exponential backoff
                    print(f"    ⏳ Rate limited, waiting {delay}s before retry {attempt + 1}/{self.MAX_RETRIES}...")
                    time.sleep(delay)
                else:
                    # Non-retryable error, raise immediately
                    raise
        
        # All retries exhausted
        raise RuntimeError(f"Gemini API failed after {self.MAX_RETRIES} retries: {last_error}")

