"""06 · Human in the loop: pause for approval before a risky action.

Concepts:
- FunctionTool(func, require_confirmation=True): before running the tool, ADK
  pauses and emits a special `adk_request_confirmation` function call instead.
- Your app shows that request to a human, then sends back a FunctionResponse
  with {"confirmed": true/false}. ADK resumes and runs (or skips) the tool.
- `adk web` renders the approval prompt for you; the script below does it by
  hand so you can see the protocol.

Run it:
    uv run python agents/06_human_in_loop/agent.py
    uv run adk web agents
"""

import asyncio
import os

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner, Runner
from google.adk.tools import FunctionTool
from google.genai import types

load_dotenv()
# LiteLLM sends "openrouter/..." models to OpenRouter using OPENROUTER_API_KEY.
MODEL = LiteLlm(model=f"openrouter/{os.getenv('MODEL', 'anthropic/claude-haiku-5.5')}")


def issue_refund(order_id: str, amount: float) -> dict:
    """Refund money to the customer. This moves real money, so it needs approval.

    Args:
        order_id: The order ID, for example "A100".
        amount: The refund amount in dollars.
    """
    print(f"   >>> REFUND SENT: ${amount:.2f} for {order_id}")
    return {"status": "success", "refunded": amount}


root_agent = LlmAgent(
    name="refund_agent",
    model=MODEL,
    instruction=(
        "You process refund requests. Call issue_refund with the order ID and "
        "amount the customer asks for. Tell the customer whether it went through."
    ),
    tools=[FunctionTool(issue_refund, require_confirmation=True)],
)

USER, SESSION = "kim", "refund-1"


async def send(runner: Runner, message: types.Content) -> list[types.FunctionCall]:
    """Run one turn; print replies; return any confirmation requests."""
    pending = []
    async for event in runner.run_async(user_id=USER, session_id=SESSION, new_message=message):
        for call in event.get_function_calls():
            if call.name == "adk_request_confirmation":
                pending.append(call)
        if event.is_final_response() and event.content and event.content.parts:
            text = "".join(p.text or "" for p in event.content.parts)
            if text:
                print(f"agent: {text}")
    return pending


async def main() -> None:
    runner = InMemoryRunner(agent=root_agent)
    await runner.session_service.create_session(app_name=runner.app_name, user_id=USER, session_id=SESSION)

    print("you: Please refund $129 for order A100.")
    pending = await send(
        runner, types.Content(role="user", parts=[types.Part(text="Please refund $129 for order A100.")])
    )

    for call in pending:
        tool_call = call.args["originalFunctionCall"]
        answer = input(f"\nApprove {tool_call['name']}({tool_call['args']})? [y/N] ")
        reply = types.Part(
            function_response=types.FunctionResponse(
                id=call.id,
                name="adk_request_confirmation",
                response={"confirmed": answer.strip().lower() == "y"},
            )
        )
        await send(runner, types.Content(role="user", parts=[reply]))


if __name__ == "__main__":
    asyncio.run(main())

# Try next:
# - Make approval conditional: require_confirmation can be a function
#   (order_id, amount) -> bool, e.g. only amounts over $50 need a human.
# - Answer "n" and read how the agent explains the refusal.
