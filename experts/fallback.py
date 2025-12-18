from experts.base import LLMExpert, LLMResponse
from experts.gpt_code import GPTCodeExpert


class FallbackExpert(LLMExpert):
    """
    General-purpose fallback expert.
    """

    def __init__(self):
        super().__init__(model="gpt-4o-mini")
        self.expert = GPTCodeExpert(model=self.model)

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        return self.expert.generate(
            prompt,
            temperature=0.4,
            max_tokens=800,
        )
