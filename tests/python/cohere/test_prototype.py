"""Conformance test: prototype instrumentation for Cohere.

Exercises: chat, embeddings
against a mock Cohere server, with manual OTel spans.
"""

import json
import os

from otel_setup import setup_otel, flush_and_shutdown
from opentelemetry import trace
from opentelemetry._logs import get_logger_provider

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]

_prototype_tracer = trace.get_tracer("gen_ai.prototype")


def run_chat(client):
    """Scenario: basic chat completion with prototype instrumentation."""
    print("  [chat] basic chat completion (prototype)")
    request_model = "command-r-plus"
    with _prototype_tracer.start_as_current_span("chat command-r-plus") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "cohere")
        span.set_attribute("gen_ai.request.model", request_model)
        messages = [{"role": "user", "content": "Say hello."}]
        resp = client.chat(
            model=request_model,
            messages=messages,
        )
        if hasattr(resp, "id") and resp.id:
            span.set_attribute("gen_ai.response.id", resp.id)
        if hasattr(resp, "finish_reason") and resp.finish_reason:
            span.set_attribute("gen_ai.response.finish_reasons", [resp.finish_reason])
        if hasattr(resp, "usage") and resp.usage:
            if hasattr(resp.usage, "tokens") and resp.usage.tokens:
                if hasattr(resp.usage.tokens, "input_tokens"):
                    span.set_attribute("gen_ai.usage.input_tokens", int(resp.usage.tokens.input_tokens))
                if hasattr(resp.usage.tokens, "output_tokens"):
                    span.set_attribute("gen_ai.usage.output_tokens", int(resp.usage.tokens.output_tokens))

        # Emit inference operation details event
        content = resp.message.content[0].text
        event_attrs = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": request_model,
            "gen_ai.input.messages": json.dumps([
                {"role": m["role"], "parts": [{"type": "text", "content": m["content"]}]}
                for m in messages
            ]),
            "gen_ai.output.messages": json.dumps([
                {
                    "role": "assistant",
                    "parts": [{"type": "text", "content": content}],
                    "finish_reason": resp.finish_reason if hasattr(resp, "finish_reason") else None,
                }
            ]),
        }
        if hasattr(resp, "id") and resp.id:
            event_attrs["gen_ai.response.id"] = resp.id
        if hasattr(resp, "finish_reason") and resp.finish_reason:
            event_attrs["gen_ai.response.finish_reasons"] = [resp.finish_reason]
        if hasattr(resp, "usage") and resp.usage and hasattr(resp.usage, "tokens") and resp.usage.tokens:
            if hasattr(resp.usage.tokens, "input_tokens"):
                event_attrs["gen_ai.usage.input_tokens"] = int(resp.usage.tokens.input_tokens)
            if hasattr(resp.usage.tokens, "output_tokens"):
                event_attrs["gen_ai.usage.output_tokens"] = int(resp.usage.tokens.output_tokens)
        get_logger_provider().get_logger("gen_ai.prototype").emit(
            event_name="gen_ai.client.inference.operation.details",
            body="Inference operation details",
            attributes=event_attrs,
        )

        print(f"    -> {content[:60]}")


def run_chat_tool_call(client):
    """Scenario: chat with tool calling with prototype instrumentation."""
    print("  [chat_tool_call] chat with tool calling (prototype)")
    request_model = "command-r-plus"
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
    with _prototype_tracer.start_as_current_span("chat command-r-plus") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "cohere")
        span.set_attribute("gen_ai.request.model", request_model)
        span.set_attribute("gen_ai.tool.definitions", json.dumps([
            {
                "type": t["type"],
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "parameters": t["function"]["parameters"],
            }
            for t in tools
        ]))
        resp = client.chat(
            model=request_model,
            messages=[{"role": "user", "content": "What's the weather in Seattle?"}],
            tools=tools,
        )
        if hasattr(resp, "id") and resp.id:
            span.set_attribute("gen_ai.response.id", resp.id)
        if hasattr(resp, "finish_reason") and resp.finish_reason:
            span.set_attribute("gen_ai.response.finish_reasons", [resp.finish_reason])
        if hasattr(resp, "usage") and resp.usage:
            if hasattr(resp.usage, "tokens") and resp.usage.tokens:
                if hasattr(resp.usage.tokens, "input_tokens"):
                    span.set_attribute("gen_ai.usage.input_tokens", int(resp.usage.tokens.input_tokens))
                if hasattr(resp.usage.tokens, "output_tokens"):
                    span.set_attribute("gen_ai.usage.output_tokens", int(resp.usage.tokens.output_tokens))
        content = resp.message.content[0].text
        if hasattr(resp.message, "tool_calls") and resp.message.tool_calls:
            print(f"    -> tool_call: {resp.message.tool_calls[0].function.name}")
        else:
            print(f"    -> {content[:60]}")


def run_embeddings(client):
    """Scenario: embedding generation with prototype instrumentation."""
    print("  [embeddings] embedding generation (prototype)")
    request_model = "embed-v4.0"
    with _prototype_tracer.start_as_current_span("embeddings embed-v4.0") as span:
        span.set_attribute("gen_ai.operation.name", "embeddings")
        span.set_attribute("gen_ai.provider.name", "cohere")
        span.set_attribute("gen_ai.request.model", request_model)
        resp = client.embed(
            model=request_model,
            texts=["Hello, world!"],
            input_type="search_document",
            embedding_types=["float"],
        )
        if hasattr(resp, "meta") and resp.meta and hasattr(resp.meta, "billed_units") and resp.meta.billed_units:
            input_tokens = getattr(resp.meta.billed_units, "input_tokens", None)
            if input_tokens is not None:
                span.set_attribute("gen_ai.usage.input_tokens", int(input_tokens))
        print(f"    -> embedding dim: {len(resp.embeddings.float_[0])}")


def main():
    print("=== Prototype: Cohere Conformance Test ===")

    tp, lp, mp = setup_otel()
    # NO instrument() call – prototype instrumentation only

    import cohere
    client = cohere.ClientV2(
        api_key="mock-key",
        base_url=MOCK_BASE_URL,
    )

    run_chat(client)
    run_chat_tool_call(client)
    run_embeddings(client)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
