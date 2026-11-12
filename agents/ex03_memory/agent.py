"""03 · Memory and state: conversations that remember.

Concepts:
- Session: one conversation. It holds the event history (what was said) and a
  `state` dict (what you choose to remember).
- SessionService: stores sessions. InMemorySessionService forgets everything
  when the process exits; Database/Vertex versions persist.
- State key prefixes control scope:
    "key"       -> this session only
    "user:key"  -> every session for this user
    "app:key"   -> every user of this app
    "temp:key"  -> this invocation only, never saved
- Tools receive `tool_context` (ADK fills it in) to read and write state.
- `{user:name?}` in an instruction is filled from state; `?` means "blank if
  missing" instead of an error.

Run it:
    uv run python agents/ex03_memory/agent.py
    uv run adk web agents          # the State tab shows state changing
"""

import asyncio
import os

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types

load_dotenv()
# LiteLLM sends "openrouter/..." models to OpenRouter using OPENROUTER_API_KEY.
MODEL = LiteLlm(model=f"openrouter/{os.getenv('MODEL', 'anthropic/claude-haiku-5.5')}")


def remember_customer(name: str, favourite_order_id: str, tool_context: ToolContext) -> dict:
    """Save the customer's name and the order they care about for future conversations.

    Args:
        name: The customer's first name.
        favourite_order_id: The order ID they mentioned, for example "A100".
    """
    # "user:" prefix: survives into this user's *other* sessions too.
    tool_context.state["user:name"] = name
    tool_context.state["user:order_id"] = favourite_order_id
    return {"status": "success"}


root_agent = LlmAgent(
    name="memory_agent",
    model=MODEL,
    description="A support agent that remembers returning customers.",
    instruction=(
        "You are a support agent. Known customer name: '{user:name?}'. "
        "Known order: '{user:order_id?}'. If these are blank and the customer "
        "tells you their name and order, call remember_customer. "
        "Greet known customers by name."
    ),
    tools=[remember_customer],
)

APP, USER = "memory_demo", "kim"


async def say(runner: Runner, session_id: str, text: str) -> None:
    """Send one message and print the final reply. This is what run_debug wraps."""
    print(f"\n[{session_id}] you: {text}")
    message = types.Content(role="user", parts=[types.Part(text=text)])
    async for event in runner.run_async(user_id=USER, session_id=session_id, new_message=message):
        if event.is_final_response() and event.content and event.content.parts:
            print(f"[{session_id}] agent: {event.content.parts[0].text}")


async def main() -> None:
    sessions = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name=APP, session_service=sessions)

    await sessions.create_session(app_name=APP, user_id=USER, session_id="chat-1")
    await say(runner, "chat-1", "Hi, I'm Kim. I'm asking about order A100.")
    await say(runner, "chat-1", "What did I say my name was?")  # same session: remembers

    # A brand-new conversation. History is gone, but "user:" state carries over.
    await sessions.create_session(app_name=APP, user_id=USER, session_id="chat-2")
    await say(runner, "chat-2", "Hello again! Which order was I asking about?")

    session = await sessions.get_session(app_name=APP, user_id=USER, session_id="chat-2")
    print("\nState in chat-2:", session.state)


if __name__ == "__main__":
    asyncio.run(main())

# Try next:
# - Change "user:name" to plain "name" and rerun. What does chat-2 forget?
# - Run `uv add 'google-adk[db]'`, swap InMemorySessionService for
#   DatabaseSessionService(db_url="sqlite+aiosqlite:///sessions.db"), and run the
#   script twice: the second run still knows Kim.
