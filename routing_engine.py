from typing import Dict

from router.router_llm import RouterLLM
from router.schema import RouterDecision

from experts.gpt_code import GPTCodeExpert
from experts.gpt_reasoning import GPTReasoningExpert
from experts.gemini_summary import GeminiSummaryExpert
from experts.fallback import FallbackExpert
from experts.base import LLMResponse

import time
from evaluation.metrics import Metrics


class RoutingEngine:
    """
    Central execution engine.
    Uses router decisions to select and run the appropriate expert.
    """

    CONFIDENCE_THRESHOLD = 0.6

    def __init__(self):
        self.router = RouterLLM()

        # Register experts by symbolic name
        self.experts: Dict[str, object] = {
            "gpt_code": GPTCodeExpert(),
            "gpt_reasoning": GPTReasoningExpert(),
            "gemini_summary": GeminiSummaryExpert(),
            "fallback": FallbackExpert(),
        }

    def _clamp_parameters(self, decision: RouterDecision) -> Dict:
        """
        Enforce safe parameter bounds regardless of router suggestion.
        """
        params = decision.parameters

        temperature = min(max(params.temperature, 0.0), 1.0)
        max_tokens = min(max(params.max_tokens, 50), 2000)

        return {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def run(self, user_query: str) -> LLMResponse:
        """
        Full request lifecycle:
        - Route
        - Validate
        - Execute
        """

        try:
            decision = self.router.route(user_query)
        except Exception:
            # Router failure → fallback
            return self.experts["fallback"].generate(user_query)

        # Low confidence → fallback
        if decision.confidence < self.CONFIDENCE_THRESHOLD:
            return self.experts["fallback"].generate(user_query)

        expert = self.experts.get(decision.expert)

        # Unknown expert → fallback
        if expert is None:
            return self.experts["fallback"].generate(user_query)

        params = self._clamp_parameters(decision)

        return expert.generate(user_query, **params)
    
    def run_with_metrics(self, user_query: str):
        start_total = time.perf_counter()

        # --- Router ---
        start_router = time.perf_counter()
        try:
            decision = self.router.route(user_query)
        except Exception:
            decision = None
        end_router = time.perf_counter()

        # --- Expert selection ---
        if decision is None or decision.confidence < self.CONFIDENCE_THRESHOLD:
            expert = self.experts["fallback"]
            params = {}
        else:
            expert = self.experts.get(decision.expert, self.experts["fallback"])
            params = self._clamp_parameters(decision)

        # --- Expert execution ---
        start_expert = time.perf_counter()
        response = expert.generate(user_query, **params)
        end_expert = time.perf_counter()

        end_total = time.perf_counter()

        usage = response.usage or {}

        metrics = Metrics(
            router_latency=end_router - start_router,
            expert_latency=end_expert - start_expert,
            total_latency=end_total - start_total,
            provider=response.provider,
            model=response.model,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

        return response, metrics

