"""Conformance test: prototype instrumentation for Cohere.

Exercises: chat, embeddings
against a mock Cohere server, with manual OTel spans.
"""

import os

from otel_setup import setup_otel, flush_and_shutdown
from opentelemetry import trace

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
        resp = client.chat(
            model=request_model,
            messages=[{"role": "user", "content": "Say hello."}],
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
        print(f"    -> {resp.message.content[0].text[:60]}")


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
    run_embeddings(client)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
