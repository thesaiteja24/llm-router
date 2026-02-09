import json
from typing import List, Dict, Optional
from openai import OpenAI
from pydantic import ValidationError

from config import OPENAI_API_KEY
from router.prompt import ROUTER_SYSTEM_PROMPT
from router.schema import RouterDecision


client = OpenAI(api_key=OPENAI_API_KEY)


class RouterLLM:
    """
    LLM-based router that classifies user queries into tasks
    and selects the appropriate expert.
    
    Supports context-aware routing via conversation history.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.last_token_usage = {}  # Track token usage for metrics

    def _format_history(self, history: Optional[List[Dict]]) -> str:
        """
        Format conversation history for the prompt.
        Uses last 3 interactions for context.
        """
        if not history:
            return "No previous conversation."
        
        # Take last 3 interactions
        recent = history[-3:] if len(history) > 3 else history
        
        formatted = []
        for i, item in enumerate(recent, 1):
            query = item.get("query", "")
            decision = item.get("decision", {})
            task_type = decision.get("task_type", "unknown")
            expert = decision.get("expert", "unknown")
            
            formatted.append(
                f"[{i}] User: {query}\n"
                f"    Routed to: {expert} (task: {task_type})"
            )
        
        return "\n".join(formatted)

    def route(
        self, 
        user_query: str, 
        history: Optional[List[Dict]] = None
    ) -> RouterDecision:
        """
        Route a user query to an expert.
        
        Args:
            user_query: The user's input query
            history: Optional conversation history for context
        
        Raises:
            RuntimeError if routing fails after retries.
        """
        # Format history into prompt
        history_context = self._format_history(history)
        system_prompt = ROUTER_SYSTEM_PROMPT.format(history=history_context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        # We allow exactly ONE retry on malformed output
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=300,
                )

                # Track token usage for metrics
                if hasattr(response, 'usage') and response.usage:
                    self.last_token_usage = {
                        'input': response.usage.prompt_tokens,
                        'output': response.usage.completion_tokens,
                    }

                raw_content = response.choices[0].message.content.strip()

                # Enforce JSON parsing
                parsed = json.loads(raw_content)

                # Validate against schema
                decision = RouterDecision.model_validate(parsed)

                return decision

            except (json.JSONDecodeError, ValidationError) as e:
                if attempt == 1:
                    raise RuntimeError(
                        f"Router failed to produce valid output: {e}"
                    )

                # Retry with a stronger instruction
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "REMINDER: Output ONLY valid JSON that matches the schema. "
                            "Do NOT include explanations or formatting."
                        ),
                    }
                )

        # This line should never be reached
        raise RuntimeError("Router failed unexpectedly")

