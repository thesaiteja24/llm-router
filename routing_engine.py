# routing_engine.py

import time
from typing import Dict, List, Optional, Tuple

from router.router_llm import RouterLLM
from router.schema import RouterDecision

from experts.gpt_code import GPTCodeExpert
from experts.gpt_reasoning import GPTReasoningExpert
from experts.gpt_summary import GPTSummaryExpert
from experts.fallback import FallbackExpert
from experts.base import LLMResponse

from evaluation.metrics import DetailedMetrics, calculate_cost


# Expert to model mapping for cost calculation
EXPERT_MODELS = {
    "gpt_code": ("openai", "gpt-4o"),
    "gpt_reasoning": ("openai", "o4-mini"),
    "gpt_summary": ("openai", "gpt-4o-mini"),
    "fallback": ("openai", "gpt-4.1-mini"),
}


class RoutingEngine:
    """
    Central execution engine.

    Responsibilities:
    - Invoke router (with optional context history)
    - Enforce confidence thresholds
    - Clamp unsafe parameters
    - Execute the selected expert
    - Track detailed metrics for evaluation
    """

    CONFIDENCE_THRESHOLD = 0.6

    def __init__(self):
        self.router = RouterLLM()

        self.experts: Dict[str, object] = {
            "gpt_code": GPTCodeExpert(),
            "gpt_reasoning": GPTReasoningExpert(),
            "gpt_summary": GPTSummaryExpert(),
            "fallback": FallbackExpert(),
        }

    def _clamp_parameters(self, decision: RouterDecision) -> Dict:
        """
        Clamp router-suggested parameters to safe, system-defined bounds.
        """
        params = decision.parameters

        clamped = {
            "temperature": min(max(params.temperature, 0.0), 1.0),
            "max_tokens": min(max(params.max_tokens, 50), 2000),
            "reasoning_depth": params.reasoning_depth,
            "frequency_penalty": min(max(params.frequency_penalty, 0.0), 2.0),
            "presence_penalty": min(max(params.presence_penalty, 0.0), 2.0),
        }

        return clamped

    def run(
        self, 
        user_query: str, 
        history: Optional[List[Dict]] = None
    ) -> LLMResponse:
        """
        Full request lifecycle:
        - Route (with optional history context)
        - Validate
        - Execute
        """

        try:
            decision = self.router.route(user_query, history=history)
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

        parameters = self._clamp_parameters(decision)

        return expert.generate(user_query, **parameters)

    def run_with_metrics(
        self,
        user_query: str,
        expected_task: Optional[str] = None,
        history: Optional[List[Dict]] = None,
    ) -> Tuple[LLMResponse, DetailedMetrics]:
        """
        Execute query and return response with detailed metrics.
        
        Used for evaluation and dashboard visualization.
        
        Args:
            user_query: The user's input query
            expected_task: Expected task type for accuracy calculation
            history: Optional conversation history for context-aware routing
        
        Returns:
            Tuple of (LLMResponse, DetailedMetrics)
        """
        used_fallback = False
        selected_expert = "fallback"
        
        # Time the router
        router_start = time.time()
        try:
            decision = self.router.route(user_query, history=history)
            router_latency = time.time() - router_start
            
            predicted_task = decision.task_type
            confidence = decision.confidence
            selected_expert = decision.expert
            
            # Check confidence threshold
            if decision.confidence < self.CONFIDENCE_THRESHOLD:
                selected_expert = "fallback"
                used_fallback = True
            
            # Check if expert exists
            if self.experts.get(decision.expert) is None:
                selected_expert = "fallback"
                used_fallback = True
                
        except Exception:
            # Router failure
            router_latency = time.time() - router_start
            predicted_task = "unknown"
            confidence = 0.0
            selected_expert = "fallback"
            used_fallback = True
            decision = None
        
        # Get expert and model info
        expert = self.experts[selected_expert]
        provider, model = EXPERT_MODELS.get(selected_expert, ("openai", "gpt-4o-mini"))
        
        # Prepare parameters
        assigned_params = None
        
        # Time the expert
        expert_start = time.time()
        if decision and not used_fallback:
            parameters = self._clamp_parameters(decision)
            assigned_params = parameters  # Track what parameters were assigned
            response = expert.generate(user_query, **parameters)
        else:
            response = expert.generate(user_query)
        expert_latency = time.time() - expert_start
        
        # Extract token counts (with defaults)
        router_tokens = getattr(self.router, 'last_token_usage', {})
        router_input_tokens = router_tokens.get('input', 100)  # Estimate
        router_output_tokens = router_tokens.get('output', 50)  # Estimate
        
        expert_input_tokens = getattr(response, 'input_tokens', None) or 200
        expert_output_tokens = getattr(response, 'output_tokens', None) or 300
        
        # Calculate costs
        router_cost = calculate_cost("gpt-4o-mini", router_input_tokens, router_output_tokens)
        expert_cost = calculate_cost(model, expert_input_tokens, expert_output_tokens)
        
        # Build metrics
        metrics = DetailedMetrics(
            # Latency
            router_latency=router_latency,
            expert_latency=expert_latency,
            total_latency=router_latency + expert_latency,
            
            # Token usage
            router_input_tokens=router_input_tokens,
            router_output_tokens=router_output_tokens,
            expert_input_tokens=expert_input_tokens,
            expert_output_tokens=expert_output_tokens,
            
            # Cost
            router_cost=router_cost,
            expert_cost=expert_cost,
            total_cost=router_cost + expert_cost,
            
            # Routing
            predicted_task=predicted_task,
            expected_task=expected_task,
            is_correct=(predicted_task == expected_task) if expected_task else None,
            confidence=confidence,
            
            # Expert selection
            selected_expert=selected_expert,
            used_fallback=used_fallback,
            
            # Models
            router_model=self.router.model,
            expert_model=model,
            expert_provider=provider,
            
            # Assigned parameters (dynamics)
            assigned_parameters=assigned_params,
        )
        
        return response, metrics

