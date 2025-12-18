import json
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
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def route(self, user_query: str) -> RouterDecision:
        """
        Route a user query to an expert.

        Raises:
            RuntimeError if routing fails after retries.
        """

        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]

        # We allow exactly ONE retry on malformed output
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=300,
                )

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
