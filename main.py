import os
import time
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from router import (
    ANTHROPIC,
    OPENAI,
    PERPLEXITY,
    choose_available,
    provider_name,
)


app = FastAPI(title="OMENZ Backend", version="0.3.2")


class RouteRequest(BaseModel):
    task_type: str = Field(
        ...,
        description="Task category such as chat, code, reasoning, analysis, research, or search.",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="The user request that the selected provider should process.",
    )


def event_id():
    return str(uuid.uuid4())


def now_ms(start):
    return round((time.time() - start) * 1000)


def openai_usage(data):
    usage = data.get("usage", {}) if isinstance(data, dict) else {}

    tokens_in = usage.get("prompt_tokens", 0) or 0
    tokens_out = usage.get("completion_tokens", 0) or 0
    total_tokens = usage.get("total_tokens", tokens_in + tokens_out) or 0

    return tokens_in, tokens_out, total_tokens


def anthropic_usage(data):
    usage = data.get("usage", {}) if isinstance(data, dict) else {}

    tokens_in = usage.get("input_tokens", 0) or 0
    tokens_out = usage.get("output_tokens", 0) or 0
    total_tokens = tokens_in + tokens_out

    return tokens_in, tokens_out, total_tokens


def perplexity_usage(data):
    usage = data.get("usage", {}) if isinstance(data, dict) else {}

    tokens_in = usage.get("prompt_tokens", 0) or 0
    tokens_out = usage.get("completion_tokens", 0) or 0
    total_tokens = usage.get("total_tokens", tokens_in + tokens_out) or 0

    return tokens_in, tokens_out, total_tokens


async def call_openai(message, run_id, start):
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        return {
            "provider": OPENAI,
            "provider_name": provider_name(OPENAI),
            "status": "missing_key",
            "model": model,
            "run_id": run_id,
            "latency_ms": now_ms(start),
            "tokens_in": 0,
            "tokens_out": 0,
            "total_tokens": 0,
        }, 500

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are PENI inside the OMENZ stack.",
                        },
                        {
                            "role": "user",
                            "content": message,
                        },
                    ],
                },
            )

        data = response.json()
        tokens_in, tokens_out, total_tokens = openai_usage(data)

        return {
            "provider": OPENAI,
            "provider_name": provider_name(OPENAI),
            "status": "ok" if response.status_code == 200 else "error",
            "status_code": response.status_code,
            "model": model,
            "run_id": run_id,
            "latency_ms": now_ms(start),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "total_tokens": total_tokens,
            "reply": data.get("choices", [{}])[0]
            .get("message", {})
            .get("content"),
            "raw": data if response.status_code != 200 else None,
        }, response.status_code

    except Exception as error:
        return {
            "provider": OPENAI,
            "provider_name": provider_name(OPENAI),
            "status": "exception",
            "error": str(error),
            "model": model,
            "run_id": run_id,
            "latency_ms": now_ms(start),
            "tokens_in": 0,
            "tokens_out": 0,
            "total_tokens": 0,
        }, 500


async def call_anthropic(message, run_id, start):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")

    if not api_key:
        return {
            "provider": ANTHROPIC,
            "provider_name": provider_name(ANTHROPIC),
            "status": "missing_key",
            "model": model,
            "run_id": run_id,
            "latency_ms": now_ms(start),
            "tokens_in": 0,
            "tokens_out": 0,
            "total_tokens": 0,
        }, 500

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 300,
                    "messages": [
                        {
                            "role": "user",
                            "content": message,
                        }
                    ],
                },
            )

        data = response.json()

        text = None
        if isinstance(data.get("content"), list) and data["content"]:
            text = data["content"][0].get("text")

        tokens_in, tokens_out, total_tokens = anthropic_usage(data)

        return {
            "provider": ANTHROPIC,
            "provider_name": provider_name(ANTHROPIC),
            "status": "ok" if response.status_code == 200 else "error",
            "status_code": response.status_code,
            "model": model,
            "run_id": run_id,
            "latency_ms": now_ms(start),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "total_tokens": total_tokens,
            "reply": text,
            "raw": data if response.status_code != 200 else None,
        }, response.status_code

    except Exception as error:
        return {
            "provider": ANTHROPIC,
            "provider_name": provider_name(ANTHROPIC),
            "status": "exception",
            "error": str(error),
            "model": model,
            "run_id": run_id,
            "latency_ms": now_ms(start),
            "tokens_in": 0,
            "tokens_out": 0,
            "total_tokens": 0,
        }, 500


async def call_perplexity(message, run_id, start):
    api_key = os.getenv("PERPLEXITY_API_KEY")
    model = os.getenv("PERPLEXITY_MODEL", "sonar")

    if not api_key:
        return {
            "provider": PERPLEXITY,
            "provider_name": provider_name(PERPLEXITY),
            "status": "missing_key",
            "model": model,
            "run_id": run_id,
            "latency_ms": now_ms(start),
            "tokens_in": 0,
            "tokens_out": 0,
            "total_tokens": 0,
        }, 500

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 300,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are XITY inside the OMENZ stack.",
                        },
                        {
                            "role": "user",
                            "content": message,
                        },
                    ],
                },
            )

        data = response.json()
        tokens_in, tokens_out, total_tokens = perplexity_usage(data)

        return {
            "provider": PERPLEXITY,
            "provider_name": provider_name(PERPLEXITY),
            "status": "ok" if response.status_code == 200 else "error",
            "status_code": response.status_code,
            "model": model,
            "run_id": run_id,
            "latency_ms": now_ms(start),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "total_tokens": total_tokens,
            "reply": data.get("choices", [{}])[0]
            .get("message", {})
            .get("content"),
            "raw": data if response.status_code != 200 else None,
        }, response.status_code

    except Exception as error:
        return {
            "provider": PERPLEXITY,
            "provider_name": provider_name(PERPLEXITY),
            "status": "exception",
            "error": str(error),
            "model": model,
            "run_id": run_id,
            "latency_ms": now_ms(start),
            "tokens_in": 0,
            "tokens_out": 0,
            "total_tokens": 0,
        }, 500


async def execute_provider(provider, message, run_id, start):
    if provider == OPENAI:
        return await call_openai(message, run_id, start)

    if provider == ANTHROPIC:
        return await call_anthropic(message, run_id, start)

    if provider == PERPLEXITY:
        return await call_perplexity(message, run_id, start)

    return {
        "provider": provider,
        "provider_name": provider_name(provider),
        "status": "routing_error",
        "error": "No available provider could be selected.",
        "run_id": run_id,
        "latency_ms": now_ms(start),
        "tokens_in": 0,
        "tokens_out": 0,
        "total_tokens": 0,
    }, 503


@app.get("/")
def root():
    return {
        "service": "OMENZ Backend",
        "status": "online",
        "message": "OMENZ Stack backend is running.",
        "version": "0.3.2",
        "router": "online",
        "route_endpoint": "/route",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "router": "online",
    }


@app.get("/auth/patreon/callback")
async def patreon_callback(request: Request):
    return JSONResponse(
        {
            "status": "received",
            "query_params": dict(request.query_params),
        }
    )


@app.get("/test/env")
def test_env():
    return {
        "openai_key_loaded": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic_key_loaded": bool(os.getenv("ANTHROPIC_API_KEY")),
        "perplexity_key_loaded": bool(os.getenv("PERPLEXITY_API_KEY")),
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "anthropic_model": os.getenv(
            "ANTHROPIC_MODEL",
            "claude-opus-4-6",
        ),
        "perplexity_model": os.getenv("PERPLEXITY_MODEL", "sonar"),
    }


@app.post("/route")
async def route_request(route_request: RouteRequest):
    start = time.time()
    run_id = event_id()

    selected_provider = choose_available(route_request.task_type)

    if selected_provider is None:
        return JSONResponse(
            {
                "task_type": route_request.task_type,
                "status": "no_provider_available",
                "run_id": run_id,
                "latency_ms": now_ms(start),
            },
            status_code=503,
        )

    result, status_code = await execute_provider(
        provider=selected_provider,
        message=route_request.message,
        run_id=run_id,
        start=start,
    )

    result["task_type"] = route_request.task_type
    result["router_selected"] = selected_provider
    result["router_selected_name"] = provider_name(selected_provider)

    return JSONResponse(result, status_code=status_code)


@app.get("/test/openai")
async def test_openai():
    start = time.time()
    run_id = event_id()

    result, status_code = await call_openai(
        message="Reply with: PENI online.",
        run_id=run_id,
        start=start,
    )

    return JSONResponse(result, status_code=status_code)


@app.get("/test/anthropic")
async def test_anthropic():
    start = time.time()
    run_id = event_id()

    result, status_code = await call_anthropic(
        message="Reply with: AUDE online.",
        run_id=run_id,
        start=start,
    )

    return JSONResponse(result, status_code=status_code)


@app.get("/test/perplexity")
async def test_perplexity():
    start = time.time()
    run_id = event_id()

    result, status_code = await call_perplexity(
        message="Reply with: XITY online.",
        run_id=run_id,
        start=start,
    )

    return JSONResponse(result, status_code=status_code)


@app.get("/test/providers")
async def test_all_providers():
    return {
        "message": "Run these one at a time first:",
        "openai": "/test/openai",
        "anthropic": "/test/anthropic",
        "perplexity": "/test/perplexity",
        "router": {
            "endpoint": "/route",
            "method": "POST",
            "example": {
                "task_type": "research",
                "message": "What is the latest development in AI?",
            },
        },
    }
