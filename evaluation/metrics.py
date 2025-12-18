from dataclasses import dataclass
from typing import Optional


@dataclass
class Metrics:
    router_latency: float
    expert_latency: float
    total_latency: float

    provider: str
    model: str

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
