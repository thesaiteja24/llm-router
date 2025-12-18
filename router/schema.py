from pydantic import BaseModel, Field
from typing import Literal, Optional


TaskType = Literal[
    "code_generation",
    "reasoning",
    "summarization",
    "general",
]

ExpertType = Literal[
    "gpt_code",
    "gpt_reasoning",
    "gemini_summary",
    "fallback",
]


class GenerationParameters(BaseModel):
    """
    Parameters controlling LLM generation.
    These are advisory and will be clamped by the routing engine.
    """
    temperature: float = Field(
        ge=0.0,
        le=1.0,
        description="Sampling temperature"
    )
    max_tokens: int = Field(
        ge=50,
        le=4000,
        description="Maximum tokens for generation"
    )


class RouterDecision(BaseModel):
    """
    Output produced by the router LLM.
    """
    task_type: TaskType
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Router confidence in classification"
    )
    expert: ExpertType
    parameters: GenerationParameters
