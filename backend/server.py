"""
FastAPI server: serves the demo UI and streams the agent trace over SSE.

Endpoints:
  GET /                -> the single-page demo UI
  GET /run?request=... -> Server-Sent Events stream of agent trace + result
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse

from agent import GRAPH, initial_state

app = FastAPI(title="Gift Design Agent Demo")

FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"

DEFAULT_REQUEST = ("My programmer friend's birthday is coming up. "
                   "I want to get him a gift. He likes Python.")


@app.get("/")
async def index():
    return FileResponse(FRONTEND)


@app.get("/run")
async def run(request: Request):
    user_request = request.query_params.get("request") or DEFAULT_REQUEST

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()

        async def emit(ev: dict) -> None:
            await queue.put(ev)

        async def runner():
            try:
                await GRAPH.ainvoke(
                    initial_state(user_request),
                    {"configurable": {"emit": emit}, "recursion_limit": 24},
                )
            except Exception as exc:  # surface errors to the UI instead of hanging
                await queue.put({"type": "trace", "kind": "decision",
                                 "title": "Error", "text": str(exc)})
            finally:
                await queue.put(None)

        task = asyncio.create_task(runner())
        step = 0
        try:
            while True:
                ev = await queue.get()
                if ev is None:
                    break
                if ev.get("type") == "trace":
                    step += 1
                    ev["step"] = step
                yield f"data: {json.dumps(ev)}\n\n"
                if await request.is_disconnected():
                    break
        finally:
            task.cancel()
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
