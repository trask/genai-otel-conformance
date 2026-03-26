"""Conformance test: reference instrumentation for Mistral AI.

Exercises: chat, chat_streaming, embeddings
against a mock server, with manual OTel spans.
"""

import os

from otel_setup import setup_otel, flush_and_shutdown
from opentelemetry import trace

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]

_reference_tracer = trace.get_tracer("gen_ai.reference")


def run_chat(client):
    """Scenario: basic chat completion with reference instrumentation."""
    print("  [chat] basic chat completion (reference)")
    request_model = "mistral-large-latest"
    with _reference_tracer.start_as_current_span("chat mistral-large-latest") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "mistral")
        span.set_attribute("gen_ai.request.model", request_model)
        resp = client.chat.complete(
            model=request_model,
            messages=[{"role": "user", "content": "Say hello."}],
        )
        if resp.model:
            span.set_attribute("gen_ai.response.model", resp.model)
        if resp.id:
            span.set_attribute("gen_ai.response.id", resp.id)
        if resp.choices:
            span.set_attribute("gen_ai.response.finish_reasons",
                               [c.finish_reason for c in resp.choices if c.finish_reason])
        if resp.usage:
            span.set_attribute("gen_ai.usage.input_tokens", resp.usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", resp.usage.completion_tokens)
        print(f"    -> {resp.choices[0].message.content[:60]}")


def run_chat_streaming(client):
    """Scenario: streaming chat completion with reference instrumentation."""
    print("  [chat_streaming] streaming chat completion (reference)")
    request_model = "mistral-large-latest"
    with _reference_tracer.start_as_current_span("chat mistral-large-latest") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "mistral")
        span.set_attribute("gen_ai.request.model", request_model)
        text = ""
        stream = client.chat.stream(
            model=request_model,
            messages=[{"role": "user", "content": "Tell me a joke."}],
        )
        for event in stream:
            if event.data.choices and event.data.choices[0].delta.content:
                text += event.data.choices[0].delta.content
        print(f"    -> {text[:60]}")


def run_embeddings(client):
    """Scenario: embedding generation with reference instrumentation."""
    print("  [embeddings] embedding generation (reference)")
    request_model = "mistral-embed"
    with _reference_tracer.start_as_current_span("embeddings mistral-embed") as span:
        span.set_attribute("gen_ai.operation.name", "embeddings")
        span.set_attribute("gen_ai.provider.name", "mistral")
        span.set_attribute("gen_ai.request.model", request_model)
        resp = client.embeddings.create(
            model=request_model,
            inputs=["Hello, world!"],
        )
        if resp.usage:
            span.set_attribute("gen_ai.usage.input_tokens", resp.usage.prompt_tokens)
        print(f"    -> embedding dim: {len(resp.data[0].embedding)}")


def main():
    print("=== Reference: Mistral AI Conformance Test ===")

    tp, lp, mp = setup_otel()
    # NO instrument() call – reference instrumentation only

    from mistralai.client import Mistral
    client = Mistral(
        api_key="mock-key",
        server_url=MOCK_BASE_URL,
    )

    run_chat(client)
    try:
        run_chat_streaming(client)
    except Exception as e:
        print(f"    WARNING: streaming failed: {e}")

    run_embeddings(client)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
