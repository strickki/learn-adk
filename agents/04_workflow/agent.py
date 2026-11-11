"""04 · Workflow graph: you decide the control flow, LLMs fill in the steps.

This is the example to compare most closely with learn-langgraph's 04.

Concepts:
- Workflow: a graph of nodes. Edges are tuples read left to right:
      (START, a, b)            START -> a -> b
      (a, {"x": b, "y": c})    a -> b if a emits route "x", a -> c if "y"
- Nodes can be plain functions or agents. A function's return value becomes
  its output, which is passed to the next node as `node_input`.
- Function parameters named after state keys are filled from session state;
  a `ctx` parameter gives you the Context (state, routing, etc.).
- `output_schema` forces an LlmAgent to return structured JSON; `output_key`
  also saves that JSON to state.
- Routing: a node returns Event(output=..., route="x") to choose the branch.

Graph:
    START -> intake -> triage (LLM) -> route --billing--> billing_agent (LLM)
                                            \\-technical-> tech_agent (LLM)

Run it:
    uv run python agents/04_workflow/agent.py
    uv run adk web agents          # inspect events to see which path ran
"""

import asyncio
import os
from typing import Literal

from dotenv import load_dotenv
from google.adk.agents import Context, LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.events.event import Event
from google.adk.runners import InMemoryRunner
from google.adk.workflow import START, Workflow
from pydantic import BaseModel, Field

load_dotenv()
# LiteLLM sends "openrouter/..." models to OpenRouter using OPENROUTER_API_KEY.
MODEL = LiteLlm(model=f"openrouter/{os.getenv('MODEL', 'anthropic/claude-haiku-5.5')}")


class Triage(BaseModel):
    category: Literal["billing", "technical"]
    urgency: Literal["low", "high"]
    summary: str = Field(description="The customer's problem in under 15 words.")


def intake(node_input: str, ctx: Context) -> str:
    """Plain Python node: store the raw ticket in state for later nodes."""
    ctx.state["ticket"] = node_input
    return node_input


triage = LlmAgent(
    name="triage",
    model=MODEL,
    instruction="Classify the customer support ticket.",
    output_schema=Triage,
    output_key="triage",  # also saved to ctx.state["triage"]
)


def route(node_input: dict, ticket: str) -> Event:
    """Deterministic branch: no LLM decides where we go, the code does.

    `node_input` is triage's output; `ticket` is filled from state.
    """
    return Event(output=ticket, route=node_input["category"])


billing_agent = LlmAgent(
    name="billing_agent",
    model=MODEL,
    instruction=(
        "You handle billing problems. Triage notes: {triage}. "
        "Reply to the customer in 2-3 sentences."
    ),
)

tech_agent = LlmAgent(
    name="tech_agent",
    model=MODEL,
    instruction=(
        "You handle technical problems. Triage notes: {triage}. "
        "Give the customer up to three troubleshooting steps."
    ),
)

root_agent = Workflow(
    name="support_workflow",
    edges=[
        (START, intake, triage, route, {"billing": billing_agent, "technical": tech_agent}),
    ],
)


async def main() -> None:
    runner = InMemoryRunner(node=root_agent)
    await runner.run_debug("I was charged twice for order A100 this morning!", session_id="t1")
    await runner.run_debug("My new USB-C dock won't detect my second monitor.", session_id="t2")


if __name__ == "__main__":
    asyncio.run(main())

# Try next:
# - Add a third category ("returns") with its own agent.
# - Add an edge so high-urgency tickets also go to an `escalate` function node.
#   Hint: route can return a list of routes, e.g. route=["billing", "urgent"].
# - Compare with learn-langgraph/examples/04_workflow.py. Which is easier to read?
