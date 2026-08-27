# 🎁 Gift Design Agent — LangGraph demo

An end-to-end **agent workflow** demo built on **LangGraph**. The point of the
demo is *not* the final picture — it's to make the agent's decision loop
**visible**: search a tool, observe the result, decide, act again (ReAct style).

The page is split in two:

- **Left** — the final recommendation (product, AI design image, product
  preview, and a short reason).
- **Right** — a **live Agent Trace** streamed step-by-step: each tool the model
  decides to call and the observation it gets back.

## How it works — a tool-calling agent

The model is given a **tool library** and decides which tools to call, and in
what order, to fulfil the request:

```
search_product_catalog · draft_prompt · rewrite_prompt_with_style
· generate_design · generate_preview_image
```

The graph is a ReAct loop:

```
User Request
   → agent (LLM decides: call a tool, or answer)
       ↔ tools (run the chosen tool, feed the result back)   ← loops
   → Final Recommendation (product + design + preview + why)
```

The LLM chooses the path — a Python fan may get a mug, someone else a different
product/design. **Out-of-scope requests** (e.g. "book a hotel in New York") are
politely **refused without calling any tools**.

Each step reports itself through an async `emit` callback; the FastAPI server
drains those events and streams them to the browser over SSE.

## Run it

```bash
./run.sh
```

Then open http://127.0.0.1:8000 and click **Run Agent**.

Manual alternative:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
python -m uvicorn server:app --app-dir backend --port 8000
```

## Configure (required)

This demo calls real models — there is no offline mock mode. Copy `.env.example`
to `.env` and set:

- **an LLM** (tool orchestration, prompt drafting, final reply) via `LLM_PROVIDER`
  = `openai` | `bedrock` | `vllm`, plus its credentials.
- **Bedrock image access** for the design artwork:
  - design → Stability **SD3.5 Large** (`stability.sd3-5-large-v1:0`)
  - preview → the generated design is **composited locally** onto the product
    photo in `backend/img/` (no extra model needed)

```bash
cp .env.example .env    # then edit .env
./run.sh
```

AWS credentials are read from your environment / profile (`AWS_PROFILE`,
`AWS_REGION`/`AWS_BEDROCK_REGION`). The Stability models must be enabled in your
Bedrock account.

## Files

| File | Purpose |
|------|---------|
| `backend/agent.py` | LangGraph tool-calling agent (the ReAct loop) |
| `backend/tools.py` | product search + design/preview tools |
| `backend/imagegen.py` | Bedrock SD3.5 design + local preview compositing |
| `backend/llm.py` | switchable LLM helper (OpenAI / Bedrock / vLLM) |
| `backend/server.py` | FastAPI + SSE streaming of the trace |
| `frontend/index.html` | split-screen UI (result + live trace) |

## Talking points for class

> An agent isn't "one prompt → one response". It's a loop: **decide an action,
> execute it, observe the result, decide again** — while updating shared state.
> The right-hand trace shows exactly that. We only show the *observable* decision
> summary (ReAct), never the model's hidden chain-of-thought.
