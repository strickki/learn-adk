"""01 · Hello agent: the smallest useful ADK program.

Concepts:
- LlmAgent: a model, a name, and an instruction (the system prompt).
- Runner: executes an agent and streams back Events. InMemoryRunner bundles an
  in-memory session store so you don't need to set one up.
- `root_agent`: the variable name `adk web` / `adk run` look for.

Run it:
    uv run python agents/ex01_hello/agent.py
    uv run adk web agents          # then pick "ex01_hello" in the browser
"""

import asyncio
import os

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner

load_dotenv()  # reads OPENROUTER_API_KEY and MODEL from the repo's .env
# LiteLLM sends "openrouter/..." models to OpenRouter using OPENROUTER_API_KEY.
MODEL = LiteLlm(model=f"openrouter/{os.getenv('MODEL', 'anthropic/claude-haiku-5.5')}")

root_agent = LlmAgent(
    name="hello_agent",
    model=MODEL,
    description="A friendly assistant that answers briefly.",
    instruction="You are a friendly assistant. Answer in at most two sentences.",
)


async def main() -> None:
    runner = InMemoryRunner(agent=root_agent)
    # run_debug hides session plumbing; ex03_memory shows what it does for you.
    await runner.run_debug("In one sentence, what makes something an 'AI agent'?")


if __name__ == "__main__":
    asyncio.run(main())

# Try next:
# - Change the instruction to give the agent a persona and see how answers shift.
# - Pass verbose=True to run_debug and look at the raw events.
