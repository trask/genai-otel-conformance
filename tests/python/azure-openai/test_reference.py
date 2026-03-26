"""Conformance test: reference instrumentation for Azure OpenAI."""

import os
from urllib.parse import urlparse

from opentelemetry import trace

from otel_setup import flush_and_shutdown, setup_otel

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]

_reference_tracer = trace.get_tracer("gen_ai.reference")


def run_chat_reference(client):
    """Scenario: basic chat completion with reference instrumentation."""
    print("  [chat] basic chat completion (reference)")
    request_model = "gpt-4o-mini"
    with _reference_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        endpoint = urlparse(MOCK_BASE_URL)
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
        if endpoint.hostname:
            span.set_attribute("server.address", endpoint.hostname)
        if endpoint.port is not None:
            span.set_attribute("server.port", endpoint.port)
        resp = client.chat.completions.create(
            model=request_model,
            messages=[{"role": "user", "content": "Say hello."}],
        )
        span.set_attribute("gen_ai.response.model", resp.model)
        span.set_attribute("gen_ai.response.id", resp.id)
        span.set_attribute("gen_ai.response.finish_reasons", [c.finish_reason for c in resp.choices])
        if resp.usage:
            span.set_attribute("gen_ai.usage.input_tokens", resp.usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", resp.usage.completion_tokens)
        print(f"    -> {resp.choices[0].message.content[:60]}")


def run_chat_streaming_reference(client):
    """Scenario: streaming chat completion with reference instrumentation."""
    print("  [chat_streaming] streaming chat completion (reference)")
    request_model = "gpt-4o-mini"
    with _reference_tracer.start_as_current_span("chat gpt-4o-mini") as span:
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
        print(f"    -> {text[:60]}")


def run_chat_tool_call_reference(client):
    """Scenario: chat with tool calling with reference instrumentation."""
    print("  [chat_tool_call] chat with tool calling (reference)")
    request_model = "gpt-4o-mini"
    tools = [
        {
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
    ]
    with _reference_tracer.start_as_current_span("chat gpt-4o-mini") as span:
        endpoint = urlparse(MOCK_BASE_URL)
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", request_model)
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
        choice = resp.choices[0]
        if choice.message.tool_calls:
            print(f"    -> tool_call: {choice.message.tool_calls[0].function.name}")
        else:
            print(f"    -> {choice.message.content[:60]}")


def run_embeddings_reference(client):
    """Scenario: embedding generation with reference instrumentation."""
    print("  [embeddings] embedding generation (reference)")
    request_model = "text-embedding-3-small"
    with _reference_tracer.start_as_current_span("embeddings text-embedding-3-small") as span:
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
    print("=== Reference: Azure OpenAI Conformance Test ===")

    tp, lp, mp = setup_otel()

    import openai

    client = openai.AzureOpenAI(
        azure_endpoint=MOCK_BASE_URL,
        api_key="mock-key",
        api_version="2024-06-01",
    )

    run_chat_reference(client)
    run_chat_streaming_reference(client)
    run_chat_tool_call_reference(client)
    run_embeddings_reference(client)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
