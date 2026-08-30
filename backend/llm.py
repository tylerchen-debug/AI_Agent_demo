"""
Thin, switchable LLM helper.

Pick a backend with LLM_PROVIDER = openai | bedrock | vllm (required). A real
backend is mandatory — if LLM_PROVIDER is unset or unknown, chat() raises.

Env vars per backend:
  openai  -> OPENAI_API_KEY               (model via DEMO_MODEL, e.g. gpt-4o-mini)
  bedrock -> AWS_REGION / AWS_PROFILE / AWS_ACCESS_KEY_ID
             (model via DEMO_MODEL, e.g. us.anthropic.claude-sonnet-4-6)
  vllm    -> VLLM_BASE_URL                (model via DEMO_MODEL = the --model you served)
             optional VLLM_API_KEY (default "EMPTY")
"""

from __future__ import annotations

import os
from pathlib import Path

# Load a project-root .env if python-dotenv is installed (optional).
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


def _detect_provider() -> str:
    p = os.getenv("LLM_PROVIDER", "").lower()
    if p in ("openai", "vllm", "bedrock"):
        return p
    raise RuntimeError(
        "Set LLM_PROVIDER=openai|vllm|bedrock and the matching credentials "
        "(OPENAI_API_KEY, VLLM_BASE_URL, or AWS creds) — see .env.example.")


_model = None


def _build_openai():
    from langchain_openai import ChatOpenAI
    model = os.getenv("DEMO_MODEL", "gpt-4o-mini")
    # Reasoning models (o-series / gpt-5) can return a thinking summary,
    # but only via the Responses API; they also reject custom temperature.
    if model.startswith(("o", "gpt-5")):
        return ChatOpenAI(
            model=model,
            use_responses_api=True,
            reasoning={"effort": "medium", "summary": "auto"},
        )
    return ChatOpenAI(model=model, temperature=0.7)


def _build_vllm():
    # vLLM exposes an OpenAI-compatible API, so reuse the OpenAI client.
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=os.getenv("DEMO_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1"),
        api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
        temperature=0.7,
    )


def _build_bedrock():
    from langchain_aws import ChatBedrockConverse
    return ChatBedrockConverse(
        model=os.getenv("DEMO_MODEL", "us.anthropic.claude-sonnet-4-6"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        temperature=0.7,
    )


def get_model():
    """The cached chat model (supports .bind_tools() for tool-calling)."""
    global _model
    if _model is None:
        builders = {"openai": _build_openai, "vllm": _build_vllm, "bedrock": _build_bedrock}
        _model = builders[_detect_provider()]()
    return _model


def chat(prompt: str) -> str:
    """Return the LLM's text response for `prompt`."""
    c = get_model().invoke(prompt).content
    if isinstance(c, list):  # Responses API returns content blocks
        c = " ".join(b.get("text", "") for b in c
                     if isinstance(b, dict) and b.get("type") == "text")
    return c.strip()
