"""Offline smoke tests: run every example with a scripted fake model.

No API key needed. They check the wiring (tools get called, routes are taken,
confirmation pauses) rather than the model's answers. Run: uv run pytest
"""

import importlib
import sys
from pathlib import Path
from typing import AsyncGenerator

import pytest
from google.adk.cli.utils.agent_loader import AgentLoader
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

AGENTS_DIR = Path(__file__).parent.parent / "agents"
sys.path.insert(0, str(AGENTS_DIR))
EXAMPLES = sorted(p.name for p in AGENTS_DIR.iterdir() if p.is_dir() and p.name[0].isdigit())


class ScriptedLlm(BaseLlm):
    """Returns pre-written responses in order. A dict means 'call this tool'."""

    model: str = "scripted"
    script: list = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        step = self.script.pop(0)
        if isinstance(step, dict):
            part = types.Part(function_call=types.FunctionCall(name=step["name"], args=step["args"]))
        else:
            part = types.Part(text=step)
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def load(example: str):
    return importlib.import_module(f"{example}.agent")


def text_of(events) -> str:
    return " ".join(p.text for e in events if e.content for p in e.content.parts or [] if p.text)


@pytest.mark.parametrize("example", EXAMPLES)
def test_adk_web_can_load(example):
    AgentLoader(str(AGENTS_DIR)).load_agent(example)


async def test_tools_example_calls_tools():
    mod = load("02_tools")
    mod.root_agent.model = ScriptedLlm(script=[
        {"name": "calculate_refund", "args": {"order_id": "A102"}},
        "Not eligible.",
    ])
    events = await InMemoryRunner(agent=mod.root_agent).run_debug("refund A102?", quiet=True)
    results = [r.response for e in events for r in e.get_function_responses()]
    assert results == [{"status": "success", "eligible": False, "refund_amount": 0.0}]


async def test_memory_example_user_state_crosses_sessions():
    mod = load("03_memory")
    mod.root_agent.model = ScriptedLlm(script=[
        {"name": "remember_customer", "args": {"name": "Kim", "favourite_order_id": "A100"}},
        "Nice to meet you, Kim.",
    ])
    runner = InMemoryRunner(agent=mod.root_agent, app_name="t")
    await runner.run_debug("I'm Kim", session_id="one", quiet=True)
    await runner.session_service.create_session(app_name="t", user_id="debug_user_id", session_id="two")
    two = await runner.session_service.get_session(app_name="t", user_id="debug_user_id", session_id="two")
    assert two.state["user:name"] == "Kim"


@pytest.mark.parametrize("category,expected", [("billing", "BILLING"), ("technical", "TECH")])
async def test_workflow_routes_by_category(category, expected):
    mod = load("04_workflow")
    scripts = {
        "triage": [f'{{"category": "{category}", "urgency": "low", "summary": "x"}}'],
        "billing_agent": ["BILLING reply"],
        "tech_agent": ["TECH reply"],
    }
    # Workflow copies agents into its graph, so swap models on the graph's nodes.
    for node in mod.root_agent.graph.nodes:
        if node.name in scripts:
            node.model = ScriptedLlm(script=scripts[node.name])
    events = await InMemoryRunner(node=mod.root_agent).run_debug("help", quiet=True)
    assert f"{expected} reply" in text_of(events)


async def test_multi_agent_transfers_to_specialist():
    mod = load("05_multi_agent")
    mod.root_agent.model = ScriptedLlm(script=[
        {"name": "transfer_to_agent", "args": {"agent_name": "orders_agent"}},
    ])
    mod.orders_agent.model = ScriptedLlm(script=["Your order shipped."])
    events = await InMemoryRunner(agent=mod.root_agent).run_debug("where is A101", quiet=True)
    assert events[-1].author == "orders_agent"


@pytest.mark.parametrize("approve,refunded", [(True, True), (False, False)])
async def test_refund_pauses_for_confirmation(approve, refunded, monkeypatch, capsys):
    mod = load("06_human_in_loop")
    mod.root_agent.model = ScriptedLlm(script=[
        {"name": "issue_refund", "args": {"order_id": "A100", "amount": 129.0}},
        "Done.",
    ])
    monkeypatch.setattr("builtins.input", lambda _: "y" if approve else "n")
    await mod.main()
    out = capsys.readouterr().out
    assert ("REFUND SENT" in out) is refunded
