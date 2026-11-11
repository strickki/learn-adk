"""05 · Multi-agent: a coordinator that delegates to specialists.

Contrast with 04: there, *your code* chose the branch. Here, *the LLM* chooses
which specialist to hand the conversation to.

Concepts:
- sub_agents: agents the coordinator can transfer control to. ADK gives the
  coordinator a `transfer_to_agent` tool automatically.
- `description` is what the coordinator reads to pick a specialist, so make it
  specific.
- After a transfer, the specialist talks to the user directly and can transfer
  back to the coordinator.
- Alternative pattern: AgentTool(agent=...) wraps a specialist as a *tool*. The
  coordinator calls it, gets an answer back, and stays in charge.

Run it:
    uv run python agents/05_multi_agent/agent.py
    uv run adk web agents          # watch the author of each event change
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


orders_agent = LlmAgent(
    name="orders_agent",
    model=MODEL,
    description="Handles questions about a specific order: status, shipping, price.",
    instruction="Answer order questions using get_order_status. Be concise.",
    tools=[get_order_status],
)

tech_agent = LlmAgent(
    name="tech_agent",
    model=MODEL,
    description="Handles technical troubleshooting for products that aren't working.",
    instruction="Give up to three clear troubleshooting steps.",
)

root_agent = LlmAgent(
    name="coordinator",
    model=MODEL,
    description="Front desk for customer support.",
    instruction=(
        "You are the front desk. Greet the customer, then hand off to the "
        "specialist that best fits their question. Don't answer specialist "
        "questions yourself."
    ),
    sub_agents=[orders_agent, tech_agent],
)


async def main() -> None:
    runner = InMemoryRunner(agent=root_agent)
    await runner.run_debug("Where is my order A101?", session_id="s1", verbose=True)
    await runner.run_debug("My keyboard's backlight stopped working.", session_id="s2", verbose=True)


if __name__ == "__main__":
    asyncio.run(main())

# Try next:
# - Replace sub_agents with tools=[AgentTool(agent=orders_agent), AgentTool(agent=tech_agent)]
#   (from google.adk.tools import AgentTool). How does the conversation change?
# - Ask a question that needs *both* specialists in one message.
