"""Conformance test: prototype instrumentation for Anthropic."""

import json
import os

from opentelemetry import trace
from opentelemetry._logs import get_logger_provider

from otel_setup import flush_and_shutdown, setup_otel

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]

_prototype_tracer = trace.get_tracer("gen_ai.prototype")


def run_chat_prototype(client):
    """Scenario: basic message with prototype instrumentation."""
    print("  [chat] basic message (prototype)")
    request_model = "claude-sonnet-4-20250514"
    with _prototype_tracer.start_as_current_span("chat claude-sonnet-4-20250514") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "anthropic")
        span.set_attribute("gen_ai.request.model", request_model)
        messages = [{"role": "user", "content": "Say hello."}]
        resp = client.messages.create(
            model=request_model,
            max_tokens=100,
            messages=messages,
        )
        span.set_attribute("gen_ai.response.model", resp.model)
        span.set_attribute("gen_ai.response.id", resp.id)
        span.set_attribute("gen_ai.response.finish_reasons", [resp.stop_reason])
        if resp.usage:
            span.set_attribute("gen_ai.usage.input_tokens", resp.usage.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", resp.usage.output_tokens)

        # Emit inference operation details event
        event_attrs = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": request_model,
            "gen_ai.response.id": resp.id,
            "gen_ai.response.model": resp.model,
            "gen_ai.response.finish_reasons": [resp.stop_reason],
            "gen_ai.input.messages": json.dumps([
                {"role": m["role"], "parts": [{"type": "text", "content": m["content"]}]}
                for m in messages
            ]),
            "gen_ai.output.messages": json.dumps([
                {
                    "role": "assistant",
                    "parts": [{"type": "text", "content": block.text}],
                    "finish_reason": resp.stop_reason,
                }
                for block in resp.content if hasattr(block, "text")
            ]),
        }
        if resp.usage:
            event_attrs["gen_ai.usage.input_tokens"] = resp.usage.input_tokens
            event_attrs["gen_ai.usage.output_tokens"] = resp.usage.output_tokens
        get_logger_provider().get_logger("gen_ai.prototype").emit(
            event_name="gen_ai.client.inference.operation.details",
            body="Inference operation details",
            attributes=event_attrs,
        )

        print(f"    -> {resp.content[0].text[:60]}")


def run_chat_streaming_prototype(client):
    """Scenario: streaming message with prototype instrumentation."""
    print("  [chat_streaming] streaming message (prototype)")
    request_model = "claude-sonnet-4-20250514"
    with _prototype_tracer.start_as_current_span("chat claude-sonnet-4-20250514") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "anthropic")
        span.set_attribute("gen_ai.request.model", request_model)
        text = ""
        with client.messages.stream(
            model=request_model,
            max_tokens=100,
            messages=[{"role": "user", "content": "Tell me a joke."}],
        ) as stream:
            for chunk in stream.text_stream:
                text += chunk
        final = stream.get_final_message()
        span.set_attribute("gen_ai.response.model", final.model)
        span.set_attribute("gen_ai.response.id", final.id)
        span.set_attribute("gen_ai.response.finish_reasons", [final.stop_reason])
        if final.usage:
            span.set_attribute("gen_ai.usage.input_tokens", final.usage.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", final.usage.output_tokens)
        print(f"    -> {text[:60]}")


def run_chat_tool_call_prototype(client):
    """Scenario: message with tool calling and prototype instrumentation."""
    print("  [chat_tool_call] tool calling message (prototype)")
    request_model = "claude-sonnet-4-20250514"
    request_tool = {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get weather for",
                }
            },
            "required": ["location"],
        },
    }
    with _prototype_tracer.start_as_current_span("chat claude-sonnet-4-20250514") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "anthropic")
        span.set_attribute("gen_ai.request.model", request_model)
        span.set_attribute("gen_ai.tool.definitions", json.dumps([request_tool]))
        resp = client.messages.create(
            model=request_model,
            max_tokens=100,
            messages=[{"role": "user", "content": "What's the weather in Seattle?"}],
            tools=[request_tool],
        )
        span.set_attribute("gen_ai.response.model", resp.model)
        span.set_attribute("gen_ai.response.id", resp.id)
        span.set_attribute("gen_ai.response.finish_reasons", [resp.stop_reason])
        if resp.usage:
            span.set_attribute("gen_ai.usage.input_tokens", resp.usage.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", resp.usage.output_tokens)
        first_block = resp.content[0]
        if getattr(first_block, "type", None) == "tool_use":
            print(f"    -> tool_call: {first_block.name}")
        else:
            print(f"    -> {first_block.text[:60]}")


def main():
    print("=== Prototype: Anthropic Conformance Test ===")

    tp, lp, mp = setup_otel()

    import anthropic

    client = anthropic.Anthropic(base_url=MOCK_BASE_URL, api_key="mock-key")

    run_chat_prototype(client)
    run_chat_streaming_prototype(client)
    run_chat_tool_call_prototype(client)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
