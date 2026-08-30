"""
The Gift Design Agent — a tool-calling ReAct agent built with LangGraph.

Instead of a fixed pipeline, the model is given a small tool library and decides
which tools to call, and in what order, to fulfil the user's request:

    search_product_catalog · draft_prompt · rewrite_prompt_with_style
    · generate_design · generate_preview_image

The agent loops (call model -> run tool(s) -> feed results back) until it has a
final recommendation. Out-of-scope requests (e.g. "book a hotel") are refused
without calling any tools. Every step is streamed to the browser as a live
Agent Trace via the `emit` callback in the graph config.
"""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Awaitable, Callable, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

import llm
import tools
from llm import chat


SYSTEM_PROMPT = """You are a gift-design assistant. You ONLY help users design and \
preview CUSTOM PRINTED GIFTS using your tools.

For an in-scope request, you have these capabilities:
  - generate a preview of a design on a chosen product,
  - rewrite an image prompt with a visual style (optional),
  - search the catalog for a product that fits the recipient and budget,
  - generate the design image from a prompt,
  - draft an image prompt for a design matching the recipient's interests.
You decide the order and which steps are needed, but you must generate a preview
before finishing.

BEFORE calling any tool, first reply with a short numbered PLAN (one line per
step) that decomposes the task, then start executing it. Before EACH tool call,
state in one short sentence which plan step you are executing and why (this is
shown to the user). If the plan changes mid-way, briefly say what changed.

When choosing the product, compare ALL catalog candidates against the
recipient's interests and budget, and explicitly explain in 1-2 sentences why
the chosen product beats the alternatives — do this BEFORE generating the
preview. Do not default to the first search result. End with a short
recommendation: the product, the design idea, and one or two sentences on
why it fits.

If the request is NOT about designing or previewing a printed gift (e.g. booking
hotels or flights, writing code, general questions), politely REFUSE in one or
two sentences and do NOT call any tools."""


# --- Tool schemas the model can call (executed manually in tools_node) --------

@tool
def search_product_catalog(query: str, budget: float = 50) -> str:
    """Search the printable gift catalog for products priced within `budget` (USD).
    Returns products with id, name, price."""


@tool
def draft_prompt(concept: str) -> str:
    """Draft a concise text-to-image prompt from a short design concept/idea."""


@tool
def rewrite_prompt_with_style(prompt: str, style: str) -> str:
    """Rewrite an image prompt to apply a visual style (e.g. 'flat vector')."""


@tool
def generate_design(prompt: str) -> str:
    """Generate the design artwork from a prompt (text-to-image). Shown to the user."""


@tool
def generate_preview_image(product_id: str) -> str:
    """Render the latest generated design printed on the given catalog product."""


TOOLS = [search_product_catalog, draft_prompt, rewrite_prompt_with_style,
         generate_design, generate_preview_image]


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    products: list
    selected_product: dict
    design_prompt: str
    design_uri: str
    preview_uri: str


Emit = Callable[[dict], Awaitable[None]]


def _emitter(config) -> Emit:
    return config["configurable"]["emit"]


async def trace(emit: Emit, kind: str, title: str, lines: list[str] | None = None,
                text: str | None = None, image: str | None = None, delay: float = 0.5) -> None:
    """Send one trace card to the UI, then pause so it feels 'live' in class."""
    await emit({"type": "trace", "kind": kind, "title": title, "lines": lines or [],
                "text": text, "image": image})
    await asyncio.sleep(delay)


async def data(emit: Emit, field: str, value) -> None:
    """Push a piece of the final result to the left-hand UI panel."""
    await emit({"type": "data", "field": field, "value": value})


def _text(msg) -> str:
    """Message content as plain text (some providers return a list of blocks)."""
    c = msg.content
    if isinstance(c, list):
        return " ".join(b.get("text", "") for b in c if isinstance(b, dict)
                        and b.get("type") != "reasoning").strip()
    return c or ""


def _thinking(msg) -> str:
    """Model's internal reasoning summary, if the provider returned one."""
    # OpenAI Responses API: content blocks of type 'reasoning' carry summaries.
    parts = []
    if isinstance(msg.content, list):
        for b in msg.content:
            if isinstance(b, dict) and b.get("type") == "reasoning":
                for s in b.get("summary", []):
                    parts.append(s.get("text", "") if isinstance(s, dict) else str(s))
    return "\n".join(p for p in parts if p).strip()


def _fmt_args(args: dict) -> list[str]:
    """One 'key: value' line per argument, untruncated so students see the full input."""
    return [f"{k}: {v}" for k, v in args.items()]


# ---------------------------------------------------------------------------
# Nodes: agent (decide) <-> tools (act), looping until a final answer.
# ---------------------------------------------------------------------------

async def agent_node(state: State, config) -> dict:
    emit = _emitter(config)
    msgs = state["messages"]
    if len(msgs) == 2:  # system + first user message
        await trace(emit, "thought", "Understanding user request", text=_text(msgs[-1]))

    model = llm.get_model().bind_tools(TOOLS)
    resp = await model.ainvoke(msgs)

    thinking = _thinking(resp)
    if thinking:  # the model's actual chain-of-thought summary (reasoning models only)
        await trace(emit, "thought", "Thinking 🧠", text=thinking)

    if resp.tool_calls:
        reasoning = _text(resp)
        if reasoning:  # the model's stated rationale for its next action
            await trace(emit, "thought", "Reasoning", text=reasoning)
        for tc in resp.tool_calls:
            await trace(emit, "action", f"Calling tool — {tc['name']}",
                        lines=[f"{tc['name']}()", "Input:"] + _fmt_args(tc["args"]))
    else:
        answer = _text(resp)
        await data(emit, "reasoning", answer)
        await trace(emit, "result", "Finished ✅", text=answer, delay=0.1)
    return {"messages": [resp]}


async def tools_node(state: State, config) -> dict:
    emit = _emitter(config)
    calls = state["messages"][-1].tool_calls
    patch: dict = {}
    out_msgs = []
    for tc in calls:
        result, lines, image = await _run_tool(emit, {**state, **patch}, patch,
                                               tc["name"], tc["args"])
        out_msgs.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        await trace(emit, "observation", f"Observation — {tc['name']}",
                    lines=["Output:"] + lines, image=image)
    patch["messages"] = out_msgs
    return patch


async def _run_tool(emit, cur, patch, name, args):
    """Execute one tool call. Returns (text_for_model, output_lines, image) and mutates `patch`."""
    if name == "search_product_catalog":
        products = tools.search_products(args.get("query", ""), args.get("budget", 50))
        patch["products"] = products
        await data(emit, "products", products)
        result = json.dumps([{k: p[k] for k in ("id", "name", "price", "note")} for p in products])
        return result, [f"{p['name']} (${p['price']}) — id={p['id']}" for p in products], None

    if name == "draft_prompt":
        p = chat("Write ONE concise text-to-image prompt (under 80 words) based on this "
                 f"idea: {args.get('concept', '')}")
        patch["design_prompt"] = p
        return p, [p], None

    if name == "rewrite_prompt_with_style":
        p = chat(f"Rewrite this text-to-image prompt to apply the visual style "
                 f"{args.get('style', '')!r}; keep it under 80 words:\n{args.get('prompt', '')}")
        patch["design_prompt"] = p
        return p, [p], None

    if name == "generate_design":
        prompt = args.get("prompt") or cur.get("design_prompt", "")
        uri = tools.generate_design(prompt)
        patch["design_prompt"] = prompt
        patch["design_uri"] = uri
        await data(emit, "design_svg", uri)
        return ("Design image generated and shown to the user.",
                ["PNG image · Stability SD3.5 Large (Bedrock)"], uri)

    if name == "generate_preview_image":
        pid = args.get("product_id", "")
        product = next((p for p in cur.get("products", []) if p["id"] == pid), None)
        if not product:
            return (f"No product with id {pid!r}. Call search_product_catalog first.",
                    [f"unknown id {pid!r}"], None)
        design_uri = cur.get("design_uri")
        if not design_uri:
            return "No design image yet. Call generate_design first.", ["no design image"], None
        uri = tools.create_product_preview(product, design_uri)
        patch["selected_product"] = product
        patch["preview_uri"] = uri
        await data(emit, "selected_product", product)
        await data(emit, "preview_svg", uri)
        return (f"Preview of the design on the {product['name']} generated and shown.",
                [f"PNG image · design composited onto {product['ref_image']}"], uri)

    return f"Unknown tool {name}.", [f"unknown tool {name}"], None


def should_continue(state: State):
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(State)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


GRAPH = build_graph()

def initial_state(user_request: str) -> dict:
    return {"messages": [SystemMessage(SYSTEM_PROMPT), HumanMessage(user_request)]}
