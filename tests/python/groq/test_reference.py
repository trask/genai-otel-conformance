"""Conformance test: reference instrumentation for Groq."""

import os

from opentelemetry import trace

from otel_setup import flush_and_shutdown, setup_otel

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"]

_reference_tracer = trace.get_tracer("gen_ai.reference")


def run_chat_reference(client):
    """Scenario: basic chat completion with reference instrumentation."""
    print("  [chat] basic chat completion (reference)")
    request_model = "llama-3.1-8b-instant"
    with _reference_tracer.start_as_current_span("chat llama-3.1-8b-instant") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "groq")
        span.set_attribute("gen_ai.request.model", request_model)
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
    request_model = "llama-3.1-8b-instant"
    with _reference_tracer.start_as_current_span("chat llama-3.1-8b-instant") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "groq")
        span.set_attribute("gen_ai.request.model", request_model)
        stream = client.chat.completions.create(
            model=request_model,
            messages=[{"role": "user", "content": "Tell me a joke."}],
            stream=True,
        )
        text = ""
        model = None
        response_id = None
        finish_reasons = []
        for chunk in stream:
            model = model or getattr(chunk, "model", None)
            response_id = response_id or getattr(chunk, "id", None)
            if chunk.choices and chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reasons.append(chunk.choices[0].finish_reason)
        if model:
            span.set_attribute("gen_ai.response.model", model)
        if response_id:
            span.set_attribute("gen_ai.response.id", response_id)
        if finish_reasons:
            span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)
        print(f"    -> {text[:60]}")


def main():
    print("=== Reference: Groq Conformance Test ===")

    tp, lp, mp = setup_otel()

    import groq

    client = groq.Groq(base_url=MOCK_BASE_URL, api_key="mock-key")

    run_chat_reference(client)
    run_chat_streaming_reference(client)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
