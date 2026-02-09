"""
Metrics classes for LLM Router evaluation.

Provides detailed metrics tracking for:
- Latency (router + expert)
- Token usage (input + output)
- Cost calculation
- Routing accuracy
"""

from dataclasses import dataclass, field
from typing import Optional, Dict


# Model pricing per 1M tokens (as of 2024)
MODEL_COSTS: Dict[str, Dict[str, float]] = {
    # OpenAI models
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # Router & New Models (User Specified)
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    # Google models
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash-preview-04-17": {"input": 0.15, "output": 0.60},
}


@dataclass
class Metrics:
    """Basic metrics for backwards compatibility."""
    router_latency: float
    expert_latency: float
    total_latency: float

    provider: str
    model: str

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


@dataclass
class DetailedMetrics:
    """
    Comprehensive metrics for evaluation and dashboard.
    
    Captures all metrics needed to validate paper claims:
    - Task-Expert Alignment (routing accuracy)
    - Dynamic Parameter Assignment (temperature, max_tokens, etc.)
    - Efficient Resource Utilization (cost per query)
    - Real-Time Task Switching (latency)
    - Robustness (fallback rate)
    """
    
    # Latency metrics
    router_latency: float
    expert_latency: float
    total_latency: float
    
    # Router token usage
    router_input_tokens: int
    router_output_tokens: int
    
    # Expert token usage
    expert_input_tokens: int
    expert_output_tokens: int
    
    # Cost breakdown
    router_cost: float
    expert_cost: float
    total_cost: float
    
    # Routing decision
    predicted_task: str
    expected_task: Optional[str]
    is_correct: Optional[bool]
    confidence: float
    
    # Expert selection
    selected_expert: str
    used_fallback: bool
    
    # Model information
    router_model: str
    expert_model: str
    expert_provider: str
    
    # Assigned generation parameters (dynamically set by router)
    assigned_parameters: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "latency": {
                "router": self.router_latency,
                "expert": self.expert_latency,
                "total": self.total_latency,
            },
            "tokens": {
                "router_input": self.router_input_tokens,
                "router_output": self.router_output_tokens,
                "expert_input": self.expert_input_tokens,
                "expert_output": self.expert_output_tokens,
            },
            "cost": {
                "router": self.router_cost,
                "expert": self.expert_cost,
                "total": self.total_cost,
            },
            "routing": {
                "predicted_task": self.predicted_task,
                "expected_task": self.expected_task,
                "is_correct": self.is_correct,
                "confidence": self.confidence,
                "selected_expert": self.selected_expert,
                "used_fallback": self.used_fallback,
            },
            "models": {
                "router": self.router_model,
                "expert": self.expert_model,
                "expert_provider": self.expert_provider,
            },
            "assigned_parameters": self.assigned_parameters,
        }


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate cost in USD based on model and token counts.
    
    Args:
        model: Model name (e.g., 'gpt-4o-mini')
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
    
    Returns:
        Cost in USD
    """
    if model not in MODEL_COSTS:
        # Default to cheapest pricing if model unknown
        return 0.0
    
    pricing = MODEL_COSTS[model]
    
    # Convert from per-million to actual cost
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    
    return input_cost + output_cost
