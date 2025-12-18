from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pydantic import BaseModel


class LLMResponse(BaseModel):
    content: str
    provider: str
    model: str
    usage: Optional[Dict[str, Any]] = None



class LLMExpert(ABC):
    """
    Abstract base class for all LLM experts.
    """

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt (str): User or system prompt
            kwargs: model-specific parameters (temperature, max_tokens, etc.)

        Returns:
            LLMResponse
        """
        raise NotImplementedError
