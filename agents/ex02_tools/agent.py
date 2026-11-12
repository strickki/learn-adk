"""02 · Tools: let the model call your Python functions.

Concepts:
- Any plain function can be a tool. ADK builds the tool schema from the
  function name, type hints, and docstring, so write those for the model.
- Return a dict. Include a "status" key so the model can tell success from
  failure without guessing.
- The model decides when to call tools; ADK runs them and feeds results back
  until the model produces a final answer (the "agent loop").

Run it:
    uv run python agents/ex02_tools/agent.py
    uv run adk web agents          # the Events tab shows each tool call
"""

import asyncio
import os

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner

load_dotenv()
# LiteLLM sends "openrouter/..." models to OpenRouter using OPENROUTER_API_KEY.
MODEL = LiteLlm(model=f"openrouter/{os.getenv('MODEL', 'anthropic/claude-haiku-5.5')}")

# Fake data standing in for a real orders API. The same data is used in
# learn-langgraph so you can compare the two frameworks on identical tasks.
ORDERS = {
    "A100": {"item": "Mechanical keyboard", "price": 129.00, "status": "delivered", "days_since_delivery": 12},
    "A101": {"item": "USB-C dock", "price": 89.50, "status": "shipped", "days_since_delivery": None},
    "A102": {"item": "Monitor arm", "price": 45.00, "status": "delivered", "days_since_delivery": 40},
}


def get_order_status(order_id: str) -> dict:
    """Look up an order's item, price, and shipping status.

    Args:
        order_id: The order ID, for example "A100".
    """
    order = ORDERS.get(order_id.upper())
    if order is None:
        return {"status": "error", "message": f"No order with ID {order_id}."}
    return {"status": "success", "order": order}


def calculate_refund(order_id: str) -> dict:
    """Work out how much would be refunded if the customer returned an order.

    Policy: full refund within 30 days of delivery, nothing after that.
    Orders that haven't been delivered yet get a full refund.

    Args:
        order_id: The order ID, for example "A100".
    """
    order = ORDERS.get(order_id.upper())
    if order is None:
        return {"status": "error", "message": f"No order with ID {order_id}."}
    days = order["days_since_delivery"]
    eligible = days is None or days <= 30
    return {
        "status": "success",
        "eligible": eligible,
        "refund_amount": order["price"] if eligible else 0.0,
    }


root_agent = LlmAgent(
    name="support_agent",
    model=MODEL,
    description="Answers questions about customer orders.",
    instruction=(
        "You are a customer support agent. Use your tools to look up orders "
        "and refunds; never guess order details. Be concise."
    ),
    tools=[get_order_status, calculate_refund],
)


async def main() -> None:
    runner = InMemoryRunner(agent=root_agent)
    # verbose=True prints each tool call and its result.
    await runner.run_debug(
        "What's the status of order A100, and how much would I get back if I returned it?",
        verbose=True,
    )


if __name__ == "__main__":
    asyncio.run(main())

# Try next:
# - Ask about A102 (outside the refund window) or a made-up order ID.
# - Delete the docstring from calculate_refund and see whether the model still
#   uses it correctly.
# - Add a `cancel_order` tool that only works when status is "shipped".
