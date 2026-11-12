# learn-adk

A sandbox for learning [Google's Agent Development Kit (ADK)](https://google.github.io/adk-docs/) (`google-adk` 2.x).
Its twin, `../learn-langgraph`, builds the **same six examples on the same fake orders data**, so you can compare the frameworks directly.

## Setup

```bash
cp .env.example .env      # add your OpenRouter key from https://openrouter.ai/keys
uv sync
```

Models are called through [OpenRouter](https://openrouter.ai) using ADK's `LiteLlm` wrapper (`model="openrouter/<id>"`). Change `MODEL` in `.env` to any OpenRouter model ID that supports tool calling. The default is `anthropic/claude-haiku-5.5`.

## Running

```bash
uv run python agents/ex02_tools/agent.py   # run one example as a script
uv run adk web agents                    # dev UI: chat, events, state, traces for every example
uv run adk run agents/ex02_tools           # chat in the terminal
uv run pytest                            # offline tests with a fake model (no API key needed)
```

`adk web` is the best way to learn: open an example, chat with it, and watch the Events and State tabs.

## Examples

Each `agents/exNN_name/agent.py` has a header that explains its concepts, and ends with **Try next** exercises.

| # | Example | You learn |
|---|---------|-----------|
| 01 | `hello` | `LlmAgent`, `Runner`, `root_agent` |
| 02 | `tools` | Plain functions as tools; how docstrings become tool schemas |
| 03 | `memory` | Sessions, `state`, `user:`/`app:`/`temp:` prefixes, `ToolContext`, `{var?}` templating |
| 04 | `workflow` | ADK 2.x `Workflow` graphs: function and agent nodes, structured output, routing |
| 05 | `multi_agent` | `sub_agents` and LLM-driven transfer; `AgentTool` as the alternative |
| 06 | `human_in_loop` | `require_confirmation` and the `adk_request_confirmation` pause and resume protocol |

`tests/test_examples.py` shows how to test agents without calling a real LLM, using a scripted `BaseLlm`.

## ADK ↔ LangGraph map

| Idea | ADK | LangGraph |
|------|-----|-----------|
| Agent with tools | `LlmAgent(tools=[fn])` | `create_agent(model, tools)`, or a hand-built `agent ⇄ tools` graph |
| Run it | `Runner` / `InMemoryRunner` → stream of `Event`s | `graph.invoke` / `graph.stream` → state |
| One conversation | Session (`session_id`) | Thread (`thread_id`) and a checkpointer |
| Remember across conversations | `user:`-prefixed state | `Store` (namespaced key-value) |
| Explicit control flow | `Workflow(edges=[...])`, `SequentialAgent`, `ParallelAgent`, `LoopAgent` | `StateGraph` with `add_edge` / `add_conditional_edges` |
| LLM picks the specialist | `sub_agents` (transfer) / `AgentTool` | Handoffs (`Command(goto=...)`) / subagent-as-tool |
| Human approval | `FunctionTool(require_confirmation=True)` | `interrupt()` + `Command(resume=...)` |
| Dev UI | `adk web` | `langgraph dev` (Studio) |
| Use OpenRouter | `LiteLlm(model="openrouter/...")` (needs `google-adk[extensions]`) | `init_chat_model("openrouter:...")` (needs `langchain-openrouter`) |

## Things to watch for while comparing

- How much code each example needs, and how easy it is to read a month later.
- Where state lives, and how explicit it is.
- How easy the dev UI makes debugging a bad tool call.
- Testing: how simple is it to fake the model?
- Lock-in: ADK is built Gemini-first (some features, such as built-in Google Search, only work with Gemini) and leans towards Google Cloud (Vertex AI Agent Engine); LangGraph leans towards LangSmith and LangGraph Platform.
