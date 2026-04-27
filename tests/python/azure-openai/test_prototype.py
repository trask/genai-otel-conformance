"""Conformance test: prototype instrumentation for Azure OpenAI."""

import json
import os
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry._logs import get_logger_provider

from otel_setup import flush_and_shutdown, setup_otel

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]

_prototype_tracer = trace.get_tracer("gen_ai.prototype")


def run_chat_prototype(client):
    """Scenario: basic chat completion with prototype instrumentation."""
    print("  [chat] basic chat completion (prototype)")
    request_model = "gpt-4o-mini"
    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        endpoint = urlparse(MOCK_BASE_URL)
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        if endpoint.hostname:
            span.set_attribute("server.address", endpoint.hostname)
        if endpoint.port is not None:
            span.set_attribute("server.port", endpoint.port)
        messages = [{"role": "user", "content": "Say hello."}]
        resp = client.chat.completions.create(
            model=request_model,
            messages=messages,
        )
        span.set_attribute("gen_ai.response.model", resp.model)
        span.set_attribute("gen_ai.response.id", resp.id)
        span.set_attribute("gen_ai.response.finish_reasons", [c.finish_reason for c in resp.choices])
        if resp.usage:
            span.set_attribute("gen_ai.usage.input_tokens", resp.usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", resp.usage.completion_tokens)
            if hasattr(resp.usage, "completion_tokens_details") and resp.usage.completion_tokens_details:
                if getattr(resp.usage.completion_tokens_details, "reasoning_tokens", None):
                    span.set_attribute("gen_ai.usage.reasoning.output_tokens", resp.usage.completion_tokens_details.reasoning_tokens)

        # Emit inference operation details event
        event_attrs = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": request_model,
            "gen_ai.response.id": resp.id,
            "gen_ai.response.model": resp.model,
            "gen_ai.response.finish_reasons": [c.finish_reason for c in resp.choices],
            "gen_ai.input.messages": json.dumps([
                {"role": m["role"], "parts": [{"type": "text", "content": m["content"]}]}
                for m in messages
            ]),
            "gen_ai.output.messages": json.dumps([
                {
                    "role": c.message.role,
                    "parts": [{"type": "text", "content": c.message.content}],
                    "finish_reason": c.finish_reason,
                }
                for c in resp.choices
            ]),
        }
        if resp.usage:
            event_attrs["gen_ai.usage.input_tokens"] = resp.usage.prompt_tokens
            event_attrs["gen_ai.usage.output_tokens"] = resp.usage.completion_tokens
            if hasattr(resp.usage, "completion_tokens_details") and resp.usage.completion_tokens_details:
                if getattr(resp.usage.completion_tokens_details, "reasoning_tokens", None):
                    event_attrs["gen_ai.usage.reasoning.output_tokens"] = resp.usage.completion_tokens_details.reasoning_tokens
        if endpoint.hostname:
            event_attrs["server.address"] = endpoint.hostname
        if endpoint.port is not None:
            event_attrs["server.port"] = endpoint.port
        get_logger_provider().get_logger("gen_ai.prototype").emit(
            event_name="gen_ai.client.inference.operation.details",
            body="Inference operation details",
            attributes=event_attrs,
        )

        print(f"    -> {resp.choices[0].message.content[:60]}")


def run_chat_streaming_prototype(client):
    """Scenario: streaming chat completion with prototype instrumentation."""
    print("  [chat_streaming] streaming chat completion (prototype)")
    request_model = "gpt-4o-mini"
    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        endpoint = urlparse(MOCK_BASE_URL)
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        if endpoint.hostname:
            span.set_attribute("server.address", endpoint.hostname)
        if endpoint.port is not None:
            span.set_attribute("server.port", endpoint.port)
        stream = client.chat.completions.create(
            model=request_model,
            messages=[{"role": "user", "content": "Tell me a joke."}],
            stream=True,
            stream_options={"include_usage": True},
        )
        text = ""
        model = None
        response_id = None
        finish_reasons = []
        input_tokens = None
        output_tokens = None
        reasoning_tokens = None
        for chunk in stream:
            model = model or getattr(chunk, "model", None)
            response_id = response_id or getattr(chunk, "id", None)
            if chunk.choices and chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reasons.append(chunk.choices[0].finish_reason)
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens
                if hasattr(chunk.usage, "completion_tokens_details") and chunk.usage.completion_tokens_details:
                    if getattr(chunk.usage.completion_tokens_details, "reasoning_tokens", None):
                        reasoning_tokens = chunk.usage.completion_tokens_details.reasoning_tokens
        if model:
            span.set_attribute("gen_ai.response.model", model)
        if response_id:
            span.set_attribute("gen_ai.response.id", response_id)
        if finish_reasons:
            span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)
        if input_tokens is not None:
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        if output_tokens is not None:
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        if reasoning_tokens:
            span.set_attribute("gen_ai.usage.reasoning.output_tokens", reasoning_tokens)
        print(f"    -> {text[:60]}")


def run_chat_tool_call_prototype(client):
    """Scenario: chat with tool calling with prototype instrumentation."""
    print("  [chat_tool_call] chat with tool calling (prototype)")
    request_model = "gpt-4o-mini"
    request_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        },
    }
    tools = [request_tool]
    with _prototype_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        endpoint = urlparse(MOCK_BASE_URL)
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        span.set_attribute("gen_ai.tool.definitions", json.dumps(tools))
        if endpoint.hostname:
            span.set_attribute("server.address", endpoint.hostname)
        if endpoint.port is not None:
            span.set_attribute("server.port", endpoint.port)
        resp = client.chat.completions.create(
            model=request_model,
            messages=[{"role": "user", "content": "What's the weather in Seattle?"}],
            tools=tools,
        )
        span.set_attribute("gen_ai.response.model", resp.model)
        span.set_attribute("gen_ai.response.id", resp.id)
        span.set_attribute("gen_ai.response.finish_reasons", [c.finish_reason for c in resp.choices])
        if resp.usage:
            span.set_attribute("gen_ai.usage.input_tokens", resp.usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", resp.usage.completion_tokens)
            if hasattr(resp.usage, "completion_tokens_details") and resp.usage.completion_tokens_details:
                if getattr(resp.usage.completion_tokens_details, "reasoning_tokens", None):
                    span.set_attribute("gen_ai.usage.reasoning.output_tokens", resp.usage.completion_tokens_details.reasoning_tokens)
        choice = resp.choices[0]
        if choice.message.tool_calls:
            print(f"    -> tool_call: {choice.message.tool_calls[0].function.name}")
        else:
            print(f"    -> {choice.message.content[:60]}")


def run_embeddings_prototype(client):
    """Scenario: embedding generation with prototype instrumentation."""
    print("  [embeddings] embedding generation (prototype)")
    request_model = "text-embedding-3-small"
    with _prototype_tracer.start_as_current_span("embeddings text-embedding-3-small") as span:
        endpoint = urlparse(MOCK_BASE_URL)
        span.set_attribute("gen_ai.operation.name", "embeddings")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        if endpoint.hostname:
            span.set_attribute("server.address", endpoint.hostname)
        if endpoint.port is not None:
            span.set_attribute("server.port", endpoint.port)
        resp = client.embeddings.create(
            model=request_model,
            input="Hello, world!",
        )
        span.set_attribute("gen_ai.response.model", resp.model)
        if resp.data and resp.data[0].embedding is not None:
            span.set_attribute("gen_ai.embeddings.dimension.count", len(resp.data[0].embedding))
        if resp.usage:
            span.set_attribute("gen_ai.usage.input_tokens", resp.usage.prompt_tokens)
        print(f"    -> embedding dim: {len(resp.data[0].embedding)}")


def main():
    print("=== Prototype: Azure OpenAI Conformance Test ===")

    tp, lp, mp = setup_otel()

    import openai

    client = openai.AzureOpenAI(
        azure_endpoint=MOCK_BASE_URL,
        api_key="mock-key",
        api_version="2024-06-01",
    )

    run_chat_prototype(client)
    run_chat_streaming_prototype(client)
    run_chat_tool_call_prototype(client)
    run_embeddings_prototype(client)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
